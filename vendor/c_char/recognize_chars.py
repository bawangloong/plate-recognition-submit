"""C 部分接口。输入有序字符切片，返回逐字符匹配和完整号码。

Python 用法：
    recognizer = CharacterRecognizer()
    result = recognizer.recognize_chars([char1, char2, ...])

识别对象可复用。初始化后只读，适合由 GUI 工作线程调用。
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np

try:  # 作为包被 D 的 GUI 导入（python -m app / PyInstaller）
    from .build_library import IMAGE_SUFFIXES
    from .char_features import FEATURE_DIM, FEATURE_VERSION, character_feature
    from .labels import ALL_LABELS, DIGITS, LETTERS, PROVINCES, SPECIAL
except ImportError:  # 直接在本目录下运行脚本时保持原用法
    from build_library import IMAGE_SUFFIXES
    from char_features import FEATURE_DIM, FEATURE_VERSION, character_feature
    from labels import ALL_LABELS, DIGITS, LETTERS, PROVINCES, SPECIAL

ROOT = Path(__file__).resolve().parent


def _enclosed_hole(image) -> bool:
    """白字里有没有被笔画包住的洞。R 有一个，W 没有。"""
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    if arr.dtype == np.bool_:
        ink = arr
    else:
        ink = arr > (127 if arr.max() > 1 else 0.5)
    if ink.size == 0 or int(ink.sum()) < 8:
        return False
    background = ~ink
    height, width = background.shape
    seen = np.zeros((height, width), dtype=bool)
    stack = []
    for x in range(width):
        if background[0, x]:
            stack.append((0, x))
        if background[height - 1, x]:
            stack.append((height - 1, x))
    for y in range(height):
        if background[y, 0]:
            stack.append((y, 0))
        if background[y, width - 1]:
            stack.append((y, width - 1))
    while stack:
        y, x = stack.pop()
        if y < 0 or x < 0 or y >= height or x >= width or seen[y, x] or not background[y, x]:
            continue
        seen[y, x] = True
        stack.append((y - 1, x))
        stack.append((y + 1, x))
        stack.append((y, x - 1))
        stack.append((y, x + 1))
    holes = background & ~seen
    return int(holes.sum()) >= 4


def _prefer_closed_r(image, result):
    """R 和 W 的余弦分差很小时，用有没有封闭孔来选。

    只在前两名正好是 R 和 W、而且分差很小的时候改结果。其它字不动。
    """
    candidates = result.get("candidates") or []
    if len(candidates) < 2:
        return result
    if {candidates[0]["label"], candidates[1]["label"]} != {"R", "W"}:
        return result
    if candidates[0]["similarity"] - candidates[1]["similarity"] > 0.08:
        return result
    want = "R" if _enclosed_hole(image) else "W"
    if result.get("label") == want:
        return result
    chosen = next(item for item in candidates if item["label"] == want)
    other = next(item for item in candidates if item["label"] != want)
    updated = dict(result)
    updated["label"] = chosen["label"]
    updated["similarity"] = chosen["similarity"]
    margin = chosen["similarity"] - other["similarity"]
    updated["margin"] = margin
    needs_review = chosen["similarity"] < 0.75 or margin < 0.02
    updated["status"] = "needs_review" if needs_review else "ok"
    updated["message"] = "匹配较弱或候选相近，建议复核" if needs_review else ""
    return updated


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", Path(path).name)]


class CharacterRecognizer:
    def __init__(self, library_path=None, review_similarity=0.75, review_margin=0.02):
        self.library_path = Path(library_path) if library_path else ROOT / "character_library/library.npz"
        if not 0 <= review_similarity <= 1 or not 0 <= review_margin <= 1:
            raise ValueError("复核阈值须在 [0,1] 内")
        self.review_similarity = review_similarity
        self.review_margin = review_margin
        with np.load(self.library_path, allow_pickle=False) as library:
            self.features = library["features"].copy()
            self.labels = library["labels"].copy()
            self.sample_paths = library["sample_paths"].copy()
            self.normalized_images = library["normalized_images"].copy()
            self.metadata = json.loads(str(library["metadata"]))
        if self.metadata.get("feature_version") != FEATURE_VERSION:
            raise ValueError("字符库与当前特征实现版本不一致，请重新构库")
        count = len(self.labels)
        if count == 0 or self.features.shape != (count, FEATURE_DIM):
            raise ValueError("字符库特征尺寸错误")
        if self.sample_paths.shape != (count,) or self.normalized_images.shape != (count, 32, 32):
            raise ValueError("字符库样本信息尺寸错误")
        if not np.isfinite(self.features).all():
            raise ValueError("字符库含非有限特征")
        if not np.allclose(np.linalg.norm(self.features, axis=1), 1, atol=1e-5):
            raise ValueError("字符库特征未进行单位长度归一化")
        if not set(self.labels).issubset(ALL_LABELS):
            raise ValueError("字符库含未支持的字符标签")
        self.class_indices = {label: np.flatnonzero(self.labels == label)
                              for label in ALL_LABELS if np.any(self.labels == label)}
        for array in (self.features, self.labels, self.sample_paths, self.normalized_images):
            array.flags.writeable = False

    def rank_features(self, features, allowed_labels=None, top_k=3):
        """批量匹配，供独立评估复用。每一名对应不同字符类别。"""
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k 必须为正整数")
        allowed = set(self.class_indices) if allowed_labels is None else set(allowed_labels)
        if not allowed or not allowed.issubset(ALL_LABELS):
            raise ValueError("allowed_labels 必须包含有效的字符标签")
        active = [label for label in ALL_LABELS if label in allowed and label in self.class_indices]
        if not active:
            raise ValueError("字符库中没有指定字符集合的样本")
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM or not np.isfinite(features).all():
            raise ValueError("查询特征尺寸错误或包含非有限值")
        if not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5):
            raise ValueError("查询特征须先归一化为单位长度")
        similarities = np.clip(features @ self.features.T, 0, 1)
        results = []
        for row in similarities:
            candidates = []
            for label in active:
                indices = self.class_indices[label]
                sample_index = int(indices[int(np.argmax(row[indices]))])
                candidates.append({"label": label, "similarity": float(row[sample_index]),
                                   "sample_index": sample_index,
                                   "sample_path": str(self.sample_paths[sample_index])})
            candidates.sort(key=lambda candidate: (-candidate["similarity"], ALL_LABELS.index(candidate["label"])))
            # 即使只展示 Top1，也用前两种不同类别计算分差。
            best = candidates[0]
            margin = best["similarity"] - candidates[1]["similarity"] if len(candidates) > 1 else None
            needs_review = best["similarity"] < self.review_similarity
            if margin is not None and margin < self.review_margin:
                needs_review = True
            results.append({"label": best["label"], "similarity": best["similarity"],
                            "margin": margin, "status": "needs_review" if needs_review else "ok",
                            "message": "匹配较弱或候选相近，建议复核" if needs_review else "",
                            "candidates": candidates[:top_k]})
        return results

    def recognize_char(self, image, allowed_labels=None, top_k=3):
        """纯色、坏图等返回 invalid_input，保留其在整牌中的位置。"""
        try:
            _, feature = character_feature(image)
        except (ValueError, OSError) as exc:
            return {"label": "?", "similarity": None, "margin": None,
                    "status": "invalid_input", "message": str(exc), "candidates": []}
        if allowed_labels is None:
            allowed_labels = [label for label in ALL_LABELS if label not in SPECIAL]
        result = self.rank_features(feature[None], allowed_labels, top_k)[0]
        return _prefer_closed_r(image, result)

    def recognize_chars(self, characters, layout="standard", top_k=3, place_indices=None):
        """standard 为常见 7/8 位位置约束，unrestricted 为任意顺序字符。"""
        if layout not in {"standard", "unrestricted"}:
            raise ValueError("layout 须为 standard 或 unrestricted")
        if not isinstance(characters, (list, tuple)):
            raise ValueError("characters 须为按阅读顺序排列的 list 或 tuple")
        count = len(characters)
        base = {"text": "", "character_count": count, "layout": layout, "characters": [],
                "warnings": [], "score_definition": "cosine of combined unit features = 0.55*HOG_cosine + 0.45*pixel_cosine; not a correctness probability"}
        if count == 0:
            return {**base, "status": "no_characters", "warnings": ["未收到字符切片"]}
        if layout == "standard" and count not in (7, 8):
            return {**base, "status": "invalid_length", "warnings": ["常见车牌需 7 或 8 个字符，请检查 B 的分割结果"]}
        results = []
        place = set(place_indices or [])
        # 上下叠写的地名只在京、东、西、城里挑，不跟黑、冀比余弦。
        place_labels = list("京东西城")
        for position, image in enumerate(characters):
            allowed = place_labels if position in place else None
            if layout == "standard" and position not in place:
                # 民用牌首位是省份简称。军牌、港澳入出境牌首位是拉丁字母，不能只在省份里挑。
                if position == 0:
                    allowed = PROVINCES + LETTERS
                elif position == 1:
                    allowed = LETTERS
                else:
                    allowed = DIGITS + LETTERS
            result = self.recognize_char(image, allowed, top_k)
            results.append({"position": position, **result})
        has_invalid = any(result["status"] == "invalid_input" for result in results)
        needs_review = any(result["status"] == "needs_review" for result in results)
        status = "partial_failure" if has_invalid else "needs_review" if needs_review else "ok"
        base.update(text="".join(result["label"] for result in results), characters=results, status=status)
        if layout == "standard":
            base["warnings"].append("位置约束仅覆盖常见民用 7/8 位结构，不验证全部号牌规则")
        return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--chars-dir", type=Path, help="按 01.png、02.png 等命名的字符目录")
    inputs.add_argument("--image", type=Path, help="识别单个字符图片")
    parser.add_argument("--library", type=Path, default=ROOT / "character_library/library.npz")
    parser.add_argument("--layout", choices=["standard", "unrestricted"], default="standard")
    parser.add_argument("--charset", choices=["all", "province", "letter", "alnum"], default="all")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visualize", type=Path, help="可选的匹配示例 PNG，需要 Pillow")
    args = parser.parse_args()
    try:
        recognizer = CharacterRecognizer(args.library)
        if args.chars_dir:
            if not args.chars_dir.is_dir():
                raise ValueError(f"字符目录不存在 {args.chars_dir}")
            images = sorted((p for p in args.chars_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
            result = recognizer.recognize_chars(images, args.layout, args.top_k)
        else:
            images = [args.image]
            allowed = {"all": None, "province": PROVINCES, "letter": LETTERS, "alnum": DIGITS + LETTERS}[args.charset]
            result = recognizer.recognize_char(args.image, allowed, args.top_k)
        result["input_files"] = [path.name for path in images]
        content = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
        if args.visualize:
            from visualize_matches import save_match_sheet
            char_results = result.get("characters", [result])
            save_match_sheet(images, char_results, recognizer, args.visualize)
        print(content)
        return 0 if result["status"] in {"ok", "needs_review"} else 2
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"识别失败：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

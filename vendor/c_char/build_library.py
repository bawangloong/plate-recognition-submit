"""将 label/图片 结构的训练样本转换成可离线加载的 HOG 字符库。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

try:  # 作为包导入时使用相对导入
    from .char_features import FEATURE_VERSION, character_feature
    from .labels import ALL_LABELS, decode_folder
except ImportError:  # 直接以脚本方式运行 build_library.py
    from char_features import FEATURE_VERSION, character_feature
    from labels import ALL_LABELS, decode_folder

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def build_library(samples, output):
    samples = Path(samples).resolve()
    output = Path(output).resolve()
    if not samples.is_dir():
        raise ValueError(f"字符样本目录不存在 {samples}")
    features, normalized_images, labels, paths, hashes = [], [], [], [], []
    seen = {}
    excluded = []
    for folder in sorted(samples.iterdir()):
        if not folder.is_dir():
            continue
        label = decode_folder(folder.name)
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            # 坏图和标签错误立即报错，避免静默构建不完整的库。
            normalized, feature = character_feature(path)
            digest = hashlib.sha256(normalized.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError(f"同一归一化图像被标为不同字符 {path}")
                excluded.append(path.relative_to(samples).as_posix())
                continue
            seen[digest] = label
            features.append(feature)
            normalized_images.append(normalized)
            labels.append(label)
            # 路径相对于训练集根目录，搬动整个交付文件夹后仍然有效。
            paths.append(path.relative_to(samples).as_posix())
            hashes.append(digest)
    if not features:
        raise ValueError("没有找到可构库的字符图片")
    metadata = {
        "feature_version": FEATURE_VERSION, "feature_dim": int(features[0].size),
        "samples": len(features), "classes": len(set(labels)),
        "class_counts": dict(sorted(Counter(labels).items())),
        "supported_labels": [label for label in ALL_LABELS if label in labels],
        "excluded_duplicate_samples": excluded,
        "image_convention": "BGR/gray input; 32x32 white foreground on black normalized images",
        "matching": "maximum cosine similarity to a stored sample per candidate class",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # 每个数组均使用数值或 Unicode dtype，加载时无需 pickle。
    with output.open("wb") as file:
        np.savez_compressed(
            file, features=np.stack(features), normalized_images=np.stack(normalized_images),
            labels=np.asarray(labels), sample_paths=np.asarray(paths),
            normalized_sha256=np.asarray(hashes), metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"library": str(output), **metadata}, ensure_ascii=False, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "character_library/train")
    parser.add_argument("--output", type=Path, default=ROOT / "character_library/library.npz")
    args = parser.parse_args()
    try:
        build_library(args.samples, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"构库失败：{exc}\n")

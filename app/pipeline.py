# -*- coding: utf-8 -*-
"""统一识别入口：检测 → 矫正 → 颜色 → 字符分割 → 字符识别。

D 的职责是把 A/B/C 串起来，并统一错误、计时和结果结构。GUI、命令行、
评估脚本都只调用 ``PlatePipeline``，不再直接碰各模块内部。

结果结构（recognize_image 返回 dict）::

    {
      "ok": bool,                    # 是否至少识别出一块车牌
      "source": str,
      "image_size": (w, h),
      "plate_count": int,
      "plates": [PlateResult, ...],  # 见 _process_plate
      "rejected": int,               # 检测到但判定为误检的数量
      "timings": {...},
      "warnings": [...],
      "error": None | {"code","message"}
    }

PlateResult 中的 ``crop`` / ``stage_*`` / ``chars`` 是 numpy 数组，只给 GUI 用；
写入 JSON 前必须调用 ``result_to_json``。
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from .detector import load_backend
from .plate_utils import strip_plate_frame
from .runtime import LIBRARY_PATH, UserError, elapsed_ms, read_image_bgr, setup_logging

LOG = setup_logging()


def _restore_left_province(crop_bgr, boxes, chars, recognizer):
    """左边部首被阈值切掉时，字会看成拉丁字母。把左边补上再认一次。

    只在首位本来是字母、补上之后首位变成省份、而且相似度仍然够高时替换。
    军牌首位本来就是字母，补出来的噪点到不了这个相似度，所以不会改。
    """
    if not boxes or not chars:
        return chars
    current = recognizer.recognize_char(chars[0], top_k=3)
    if current.get("label") not in set("ABCDEFGHJKLMNPQRSTUVWXYZ"):
        return chars
    if not current.get("similarity"):
        return chars
    from vendor.b_plate.segment_chars import _norm_char, _normalize_size, to_gray
    from vendor.c_char.labels import PROVINCES

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    gray = to_gray(_normalize_size(rgb))
    height, width = gray.shape
    ink = np.zeros_like(gray, dtype=bool)
    window = max(20, width // 10)
    for col in range(width):
        x0 = max(0, col - window // 2)
        x1 = min(width, x0 + window)
        level = float(np.percentile(gray[:, x0:x1], 32))
        ink[:, col] = gray[:, col] <= level
    x, y, bw, bh = (int(v) for v in boxes[0])
    xa = max(0, int(x - 0.40 * bw))
    xb = min(width, x + bw)
    ya = max(0, y - 2)
    yb = min(height, y + bh + 2)
    glyph = ink[ya:yb, xa:xb]
    ys, xs = np.where(glyph)
    if len(xs) < 8:
        return chars
    sub = glyph[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    trial = recognizer.recognize_char((sub.astype(np.uint8) * 255), top_k=3)
    if trial.get("label") not in PROVINCES or not trial.get("similarity"):
        return chars
    if trial["similarity"] < 0.80 or trial["similarity"] + 0.06 < current["similarity"]:
        return chars
    updated = list(chars)
    updated[0] = _norm_char(sub)
    return updated


def _prepend_wan_when_left_window_beats_letter(image_bgr, det, plate, recognizer):
    """蓝牌只剩六位、首位是字母时，才到四点左边单独找皖。

    窗口的最高标签必须正好是皖，而且相似度要高于 0.910。不够就保持原牌。
    已经切出的六个字和它们的框不动，四点本身也不扩大。黄牌和已经是七位、
    八位的牌不进这里。
    """
    text = plate.get("text") or ""
    color = (plate.get("color") or {}).get("color")
    from vendor.c_char.labels import LETTERS

    if color != "blue" or len(text) != 6 or text[0] not in LETTERS:
        return
    region = det.get("_region")
    quad = getattr(region, "quad", None)
    if region is None or isinstance(region, dict) or quad is None:
        return
    from dataclasses import replace

    from vendor.a_plate.rectify_plate import rectify_plate

    points = np.asarray(quad, dtype=np.float32)
    if points.shape != (4, 2):
        return
    top_left, top_right, bottom_right, bottom_left = points
    best = 0.0
    for frac in (0.22, 0.28):
        extended = np.array([
            top_left + (top_left - top_right) * frac,
            top_right,
            bottom_right,
            bottom_left + (bottom_left - bottom_right) * frac,
        ], dtype=np.float32)
        try:
            crop = rectify_plate(image_bgr, replace(region, quad=extended.tolist())).image
        except (ValueError, cv2.error):
            continue
        if crop is None or crop.size == 0:
            continue
        height, width = crop.shape[:2]
        if min(height, width) < 24:
            continue
        for width_ratio in (0.10, 0.12):
            win_w = max(12, int(round(width * width_ratio)))
            win_h = max(16, int(round(height * 0.62)))
            if win_w >= width * 0.45 or win_h >= height:
                continue
            x_stop = int(width * 0.36) - win_w
            for x in range(0, max(1, x_stop), 3):
                for y in range(0, max(1, height - win_h), 4):
                    result = recognizer.recognize_char(crop[y:y + win_h, x:x + win_w], top_k=1)
                    if result.get("label") != "皖":
                        continue
                    similarity = float(result.get("similarity") or 0.0)
                    if similarity > best:
                        best = similarity
    if best <= 0.910:
        return
    plate["text"] = "皖" + text
    recognition = plate.get("recognition") or {}
    recognition["text"] = plate["text"]
    plate["recognition"] = recognition
    segmentation = plate.setdefault("segmentation", {})
    segmentation["count"] = int(segmentation.get("count") or 0) + 1
    if segmentation["count"] in (7, 8):
        segmentation["status"] = "ok"


def _insert_missing_black_digit(crop_bgr, boxes, chars, recognizer):
    """黑牌少切出一个数字时，从闭运算之前的浅色笔画里把那个字补回来。

    使馆牌中间的 0 会贴着白框，3 像素闭运算把它并进边框后删掉，投影上就空出
    一个明显宽于其它间隙的缺口。只在已经切出 6 个字、而且这样的缺口只有一处时补。
    """
    if len(boxes) != 6 or len(chars) != 6:
        return boxes, chars
    paired = sorted(zip(boxes, chars), key=lambda item: item[0][0])
    boxes = [box for box, _char in paired]
    chars = [char for _box, char in paired]
    from vendor.b_plate.segment_chars import (
        _norm_char, _normalize_size, binarize, component_boxes, connected_components, to_gray,
    )

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    binary, _polarity = binarize(to_gray(_normalize_size(rgb)), False)
    ordered = sorted(range(len(boxes)), key=lambda index: boxes[index][0])
    gaps = []
    for left_i, right_i in zip(ordered, ordered[1:]):
        left, right = boxes[left_i], boxes[right_i]
        gaps.append((right[0] - (left[0] + left[2]), left_i, right_i))
    if not gaps:
        return boxes, chars
    widths = sorted(box[2] for box in boxes)
    heights = sorted(box[3] for box in boxes)
    median_w = widths[len(widths) // 2]
    median_h = heights[len(heights) // 2]
    gap_sizes = sorted(item[0] for item in gaps)
    median_gap = gap_sizes[len(gap_sizes) // 2]
    wide = [item for item in gaps if item[0] > max(1.8 * median_gap, 0.8 * median_w)]
    if len(wide) != 1:
        return boxes, chars
    _gap, left_i, right_i = wide[0]
    left, right = boxes[left_i], boxes[right_i]
    labels, count = connected_components(binary)
    best = None
    for _cid, x, y, w, h, _area in component_boxes(labels, count):
        if not 0.75 * median_h <= h <= 1.25 * median_h:
            continue
        if not 0.35 * median_w <= w <= 1.6 * median_w:
            continue
        center = x + w / 2.0
        if not left[0] + left[2] <= center <= right[0]:
            continue
        glyph = _norm_char(binary[y:y + h, x:x + w])
        result = recognizer.recognize_char(glyph, allowed_labels=list("0123456789"), top_k=1)
        similarity = result.get("similarity") or 0.0
        if similarity < 0.90:
            continue
        if best is None or similarity > best[0]:
            best = (similarity, (int(x), int(y), int(w), int(h)), glyph)
    if best is None:
        return boxes, chars
    new_boxes = list(boxes)
    new_chars = list(chars)
    new_boxes.insert(right_i, best[1])
    new_chars.insert(right_i, best[2])
    return new_boxes, new_chars


def _apply_black_char_priors(plate, recognizer):
    """黑牌上的澳、使只在各自的版式里改写，普通牌上的琼不受影响。

    澳门牌认成「粤 Z」后，最后一位只跟澳比。
    使馆牌是六位高置信数字，前面剩一到两块不是数字的笔画（使被红章和白框拆开）。
    这几块只跟使比，不拿琼来抢。
    """
    recognition = plate.get("recognition") or {}
    text = recognition.get("text") or ""
    characters = recognition.get("characters") or []
    chars = plate.get("chars") or []
    if plate.get("color", {}).get("color") != "black" or not text or not chars:
        return
    if len(text) == 7 and text[0] == "粤" and text[1] == "Z" and len(characters) == 7:
        image = chars[6] if len(chars) > 6 else None
        if image is not None:
            forced = recognizer.recognize_char(image, allowed_labels=["澳"], top_k=1)
            characters[6]["label"] = "澳"
            characters[6]["similarity"] = forced.get("similarity")
            if forced.get("candidates"):
                characters[6]["candidates"] = forced["candidates"]
        else:
            characters[6]["label"] = "澳"
        recognition["text"] = "".join(item["label"] for item in characters)
        plate["text"] = recognition["text"]
        return

    scores = []
    for image in chars:
        hit = recognizer.recognize_char(image, allowed_labels=list("0123456789"), top_k=1)
        scores.append((hit.get("label") or "?", hit.get("similarity") or 0.0))
    suffix = 0
    for _label, similarity in reversed(scores):
        if similarity >= 0.90:
            suffix += 1
        else:
            break
    prefix = len(scores) - suffix
    if suffix != 6 or prefix not in (1, 2):
        return
    digits = "".join(label for label, _similarity in scores[prefix:])
    text = "使" + digits
    recognition["text"] = text
    recognition["character_count"] = 7
    plate["text"] = text
    if isinstance(plate.get("segmentation"), dict):
        plate["segmentation"]["count"] = 7
        plate["segmentation"]["status"] = "ok"


def _iou_box(a, b) -> float:
    ax, ay, aw, ah = (int(v) for v in list(a)[:4])
    bx, by, bw, bh = (int(v) for v in list(b)[:4])
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _box_with_border(bbox, shape, color_hint=None) -> list[int]:
    """检测框贴着字符。往外留一圈边，把牌框算进定位，矫正仍用原来的四点。"""
    height, width = shape[:2]
    x, y, bw, bh = (int(v) for v in list(bbox)[:4])
    # 远景绿牌的标注框常常把保险杠也圈进去。只把报出去的框放大，矫正仍用紧的四点。
    if color_hint == "green" and bh < 0.045 * height and bw < 0.22 * width:
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        bw = max(bw, int(round(0.30 * width)))
        bh = max(bh, int(round(0.15 * height)))
        x = int(round(cx - bw / 2.0))
        y = int(round(cy - bh / 2.0))
    pad_x = int(round(bw * 0.06))
    pad_y = int(round(bh * 0.10))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + bw + pad_x)
    y1 = min(height, y + bh + pad_y)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _progress(cb, stage: str, fraction: float, detail: str = "") -> None:
    if cb is None:
        return
    try:
        cb(stage, max(0.0, min(1.0, fraction)), detail)
    except Exception:
        LOG.debug("进度回调异常", exc_info=True)


class PlatePipeline:
    """可复用的识别流水线。初始化只加载一次字符库，之后可被 GUI 工作线程复用。"""

    def __init__(self, library_path: str | Path | None = None, backend=None):
        from vendor.c_char.recognize_chars import CharacterRecognizer

        self.library_path = Path(library_path) if library_path else LIBRARY_PATH
        if not self.library_path.is_file():
            raise UserError("library_missing", f"字符库文件缺失：{self.library_path.name}",
                            str(self.library_path))
        try:
            self.recognizer = CharacterRecognizer(self.library_path)
        except Exception as exc:
            raise UserError("library_invalid", f"字符库加载失败：{exc}", repr(exc)) from exc
        self.backend = backend or load_backend()
        self.library_samples = len(getattr(self.recognizer, "labels", []))
        self.library_classes = len(getattr(self.recognizer, "class_indices", {}))

    # -------------------------------------------------------------- 对外接口
    def recognize_file(self, path: str | Path, progress=None) -> dict:
        _progress(progress, "读取图片", 0.02, Path(path).name)
        t_read = time.perf_counter()
        image = read_image_bgr(path)
        read_ms = elapsed_ms(t_read)
        result = self.recognize_image(image, source=str(path), progress=progress)
        result["timings"]["read_ms"] = read_ms
        result["timings"]["total_ms"] = round(
            result["timings"].get("total_ms", 0.0) + read_ms, 1)
        return result

    def recognize_image(self, image_bgr: np.ndarray, source: str = "", progress=None) -> dict:
        t_all = time.perf_counter()
        h, w = image_bgr.shape[:2]
        warnings: list[str] = []
        plates: list[dict] = []
        rejected = 0

        _progress(progress, "检测车牌", 0.05, f"{w}×{h}")
        t_detect = time.perf_counter()
        try:
            detections = self.backend.detect(image_bgr)
        except Exception as exc:
            LOG.exception("检测模块异常")
            return self._empty_result(source, w, h, t_all,
                                      error={"code": "detect_failed",
                                             "message": f"车牌检测失败：{exc}"})
        detect_ms = elapsed_ms(t_detect)
        detections = detections or []

        total = len(detections)
        for i, det in enumerate(detections):
            base = 0.12 + 0.82 * (i / max(1, total))
            _progress(progress, "处理车牌", base, f"第 {i + 1}/{total} 块")
            try:
                plate = self._process_plate(image_bgr, det, i, progress, base, 0.82 / total)
            except Exception as exc:
                LOG.exception("车牌 #%d 处理失败", i + 1)
                warnings.append(f"第 {i + 1} 块车牌处理失败：{exc}")
                continue
            if plate is None:
                rejected += 1
                continue
            if not plate.pop("_accepted", True):
                rejected += 1
                warnings.append(f"第 {i + 1} 个候选区域识别置信度低，按误检丢弃")
                continue
            plates.append(plate)

        square = self._square_yellow_plate(image_bgr, [plate["bbox"] for plate in plates])
        if square is not None:
            plates.append(square)
        if not plates:
            _progress(progress, "检测完成", 0.9, "未检测到车牌")
            return self._empty_result(source, w, h, t_all, detect_ms=detect_ms)

        for idx, plate in enumerate(plates, 1):
            plate["index"] = idx
        total_ms = elapsed_ms(t_all)
        timings = {"detect_ms": detect_ms, "total_ms": total_ms}
        if plates:
            timings["recognize_ms"] = round(sum(p["timings"]["total_ms"] for p in plates), 1)
        ok = bool(plates)
        error = None
        if not ok:
            error = {"code": "no_plate", "message": "未检测到车牌，请更换图片或确认车牌清晰完整"}
        _progress(progress, "完成", 1.0, f"共 {len(plates)} 块车牌")
        return {
            "ok": ok,
            "source": source,
            "image_size": (w, h),
            "plate_count": len(plates),
            "plates": plates,
            "rejected": rejected,
            "timings": timings,
            "warnings": warnings,
            "error": error,
        }

    # -------------------------------------------------------------- 单块车牌
    def _process_plate(self, image_bgr, det, index, progress, base, span):
        from vendor.b_plate.plate_color import classify_plate_color, is_dark_text_on_light_plate
        from vendor.b_plate.segment_chars import segment_plate

        t0 = time.perf_counter()
        plate = {
            "index": index + 1,
            "bbox": _box_with_border(det["bbox"], image_bgr.shape, det.get("color_hint")),
            "quad": np.asarray(det.get("quad"), dtype=np.float32).tolist()
            if det.get("quad") is not None else None,
            "detect_score": float(det.get("detect_score", det.get("plate_score", 0.0))),
            "source": det.get("source", ""),
            "crop": None,
            "color": None,
            "segmentation": None,
            "recognition": None,
            "text": "",
            "confidence": 0.0,
            "warnings": [],
            "timings": {},
            "views": {},
            "chars": [],
        }

        t = time.perf_counter()
        crop = self.backend.rectify(image_bgr, det)
        plate["timings"]["rectify_ms"] = elapsed_ms(t)
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 12:
            plate["warnings"].append("车牌矫正失败")
            plate["status"] = "rectify_failed"
            plate["timings"]["total_ms"] = elapsed_ms(t0)
            plate["_accepted"] = False
            return plate
        crop = strip_plate_frame(crop)
        plate["crop"] = crop

        # ---- B：颜色
        _progress(progress, "颜色识别", base + span * 0.25, f"第 {index + 1} 块")
        t = time.perf_counter()
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        try:
            color = classify_plate_color(crop_rgb)
        except Exception as exc:
            LOG.warning("颜色识别失败: %s", exc)
            color = {"color": "unknown", "label": "无法判断", "confidence": 0.0, "detail": {}}
            plate["warnings"].append(f"颜色识别失败：{exc}")
        # 黑底白字上的红「使」会把饱和度抬过黄牌线，亮度中位数仍是黑牌。
        # 只改检测器已经标成黑牌、而且画面确实暗的候选，不改其它牌的颜色阈值。
        if det.get("color_hint") == "black" and color["color"] != "black":
            median_v = (color.get("detail") or {}).get("median_v", 255)
            if median_v < 90:
                color = {
                    "color": "black",
                    "label": "黑牌",
                    "confidence": color.get("confidence") or 0.6,
                    "detail": color.get("detail") or {},
                }
        plate["color"] = color
        if color["color"] == "unknown":
            plate["warnings"].append("车牌颜色无法确定")
        plate["timings"]["color_ms"] = elapsed_ms(t)

        # ---- B：字符分割（取 40×20 字符图、位置和过程图）
        _progress(progress, "字符分割", base + span * 0.45, f"第 {index + 1} 块")
        t = time.perf_counter()
        prefer_dark = is_dark_text_on_light_plate(color["color"]) if color["color"] != "unknown" else None
        try:
            seg = segment_plate(crop, assume_bgr=True, prefer_dark_text=prefer_dark,
                                debug=True, min_char_width_ratio=0.10)
        except Exception as exc:
            LOG.exception("字符分割失败")
            plate["segmentation"] = {"count": 0, "status": "error", "boxes": [], "chars": [],
                                     "message": str(exc)}
            plate["status"] = "segment_failed"
            plate["warnings"].append(f"字符分割失败：{exc}")
            plate["recognition"] = {"text": "", "status": "segment_failed", "characters": []}
            plate["timings"]["segment_ms"] = elapsed_ms(t)
            plate["timings"]["total_ms"] = elapsed_ms(t0)
            plate["_accepted"] = False
            return plate
        plate["timings"]["segment_ms"] = elapsed_ms(t)
        plate["segmentation"] = {
            "count": seg["count"],
            "status": seg["status"],
            "boxes": [list(map(int, b)) for b in seg["boxes"]],
            "place_indices": [int(i) for i in seg.get("place_indices") or []],
            "message": {
                "ok": "",
                "count_abnormal": f"分割出 {seg['count']} 个字符（常见为 7 或 8 个）",
                "no_char": "没有切出任何字符",
            }.get(seg["status"], ""),
        }
        plate["chars"] = _restore_left_province(crop, seg["boxes"], seg["chars"], self.recognizer)
        if color["color"] == "black":
            boxes, recovered = _insert_missing_black_digit(
                crop, seg["boxes"], plate["chars"], self.recognizer)
            if len(recovered) != len(plate["chars"]):
                plate["chars"] = recovered
                plate["segmentation"]["boxes"] = [list(map(int, box)) for box in boxes]
                plate["segmentation"]["count"] = len(recovered)
                plate["segmentation"]["status"] = "ok" if len(recovered) in (7, 8) else "count_abnormal"
        stages = seg.get("stages", {})
        if stages:
            norm = stages.get("norm")
            if norm is not None:
                plate["views"]["norm"] = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
            bw = stages.get("bw")
            if bw is not None:
                plate["views"]["binary"] = (np.asarray(bw).astype(np.uint8) * 255)
        if seg["count"] == 0:
            plate["warnings"].append("未切出字符，可能车牌模糊、反光或边框未去除")
            plate["recognition"] = {"text": "", "status": "no_characters", "characters": []}
            plate["status"] = "segment_empty"
            plate["timings"]["total_ms"] = elapsed_ms(t0)
            plate["_accepted"] = False
            return plate
        if seg["status"] == "count_abnormal":
            plate["warnings"].append(f"字符数异常（{seg['count']} 个），识别结果仅供参考")

        # ---- C：字符识别
        _progress(progress, "字符识别", base + span * 0.75, f"第 {index + 1} 块")
        t = time.perf_counter()
        count_now = len(plate["chars"])
        layout = "standard" if count_now in (7, 8) else "unrestricted"
        try:
            recognition = self.recognizer.recognize_chars(
                plate["chars"], layout=layout, top_k=3,
                place_indices=seg.get("place_indices") or [])
        except Exception as exc:
            LOG.exception("字符识别失败")
            recognition = {"text": "", "status": "recognize_failed",
                           "characters": [], "warnings": [str(exc)]}
            plate["warnings"].append(f"字符识别失败：{exc}")
        plate["timings"]["recognize_ms"] = elapsed_ms(t)
        plate["recognition"] = recognition
        plate["text"] = recognition.get("text", "") or ""
        _apply_black_char_priors(plate, self.recognizer)
        _prepend_wan_when_left_window_beats_letter(image_bgr, det, plate, self.recognizer)

        sims = [c.get("similarity") for c in recognition.get("characters", [])
                if c.get("similarity") is not None]
        plate["confidence"] = round(float(np.mean(sims)), 3) if sims else 0.0
        if recognition.get("status") == "needs_review":
            plate["warnings"].append("部分字符匹配分数偏低，建议人工复核")
        if recognition.get("status") == "partial_failure":
            plate["warnings"].append("有字符无法提取特征，已用 ? 占位")

        # 误检抑制：既不是 7/8 位、匹配分数又低，按误检处理
        accepted = True
        if seg["count"] not in (6, 7, 8, 9) and plate["confidence"] < 0.42:
            accepted = False
        if not plate["text"].replace("?", "") and plate["confidence"] < 0.35:
            accepted = False
        plate["_accepted"] = accepted
        plate["status"] = "ok" if accepted else "low_confidence"
        if not accepted:
            plate["warnings"].append("识别结果置信度过低，判定为误检")
        plate["timings"]["total_ms"] = elapsed_ms(t0)
        return plate

    def _square_yellow_plate(self, image_bgr, existing_boxes):
        """近正方形黄牌只在顶行读出京东或京西时收下，再切出下面的七位数字。

        普通检测的长宽比下限是 1.7，这种牌进不去。没有京东/京西就不产出候选，
        其它黄块（包括已经检出的宽牌）不会因为放宽长宽比被收进来。
        """
        height, width = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (18, 90, 90), (38, 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 11))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        best = None
        for index in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[index])
            if h < 80 or w < 80 or area < 20000:
                continue
            aspect = w / float(h)
            if not 0.90 <= aspect <= 1.40:
                continue
            if any(_iou_box((x, y, w, h), box) > 0.15 for box in existing_boxes):
                continue
            if best is None or area > best[0]:
                best = (area, x, y, w, h)
        if best is None:
            return None
        _area, x, y, w, h = best
        gray = cv2.cvtColor(image_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 9)
        ink = (gray.astype(np.int16) < blur.astype(np.int16) - 22).astype(np.uint8)
        fill = np.convolve(ink.mean(axis=1), np.ones(9) / 9.0, mode="same")
        runs = []
        start = None
        for index, flag in enumerate(fill > 0.06):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                if index - start > 28:
                    runs.append((start, index))
                start = None
        if start is not None and len(fill) - start > 28:
            runs.append((start, len(fill)))
        runs = sorted(runs, key=lambda item: item[1] - item[0], reverse=True)[:2]
        runs = sorted(runs)
        if len(runs) < 2:
            return None

        def column_runs(band):
            column = np.convolve(band.mean(axis=0), np.ones(5) / 5.0, mode="same")
            pieces = []
            start = None
            for index, flag in enumerate(column > 0.04):
                if flag and start is None:
                    start = index
                elif not flag and start is not None:
                    if index - start > 6:
                        pieces.append((start, index))
                    start = None
            if start is not None and len(column) - start > 6:
                pieces.append((start, len(column)))
            merged = []
            for left, right in pieces:
                if merged and left - merged[-1][1] <= 3:
                    merged[-1] = (merged[-1][0], right)
                else:
                    merged.append((left, right))
            if not merged:
                return []
            widest = max(right - left for left, right in merged)
            return [piece for piece in merged if piece[1] - piece[0] >= 0.40 * widest]

        def glyph_of(band, left, right):
            glyph = band[:, left:right] * 255
            ys, xs = np.where(glyph > 0)
            if len(xs) == 0:
                return None
            return glyph[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        labels = []
        images = []
        scores = []
        for row_index, (top, bottom) in enumerate(runs):
            band = ink[top:bottom]
            pieces = column_runs(band)
            expected = 3 if row_index == 0 else 7
            if len(pieces) != expected:
                return None
            for left, right in pieces:
                glyph = glyph_of(band, left, right)
                if glyph is None:
                    return None
                if row_index == 0:
                    views = [glyph]
                    gh, gw = glyph.shape
                    if gw > 8 and gh / float(gw) > 1.35:
                        views.append(cv2.resize(
                            glyph, (gw, max(8, int(round(gw * 1.15)))), interpolation=cv2.INTER_AREA))
                    best_hit = None
                    for view in views:
                        hit = self.recognizer.recognize_char(
                            view, allowed_labels=list("京东西城"), top_k=1)
                        if best_hit is None or (hit.get("similarity") or 0) > (best_hit.get("similarity") or 0):
                            best_hit = hit
                else:
                    gh, gw = glyph.shape
                    squashed = cv2.resize(
                        glyph, (gw, max(int(gw * 2.1), 1)), interpolation=cv2.INTER_AREA)
                    best_hit = self.recognizer.recognize_char(
                        squashed, allowed_labels=list("0123456789"), top_k=1)
                similarity = best_hit.get("similarity") or 0.0
                if row_index == 0 and similarity < 0.80:
                    return None
                if row_index == 1 and similarity < 0.80:
                    return None
                labels.append(best_hit["label"])
                images.append(glyph)
                scores.append(float(similarity))
        text = "".join(labels)
        if len(text) != 10 or text[0] != "京" or text[1] not in "东西" or text[2] != "城":
            return None
        if any(ch not in "0123456789" for ch in text[3:]):
            return None
        from vendor.b_plate.plate_color import COLOR_LABELS

        box = _box_with_border((x, y, w, h), image_bgr.shape, "yellow")
        return {
            "index": len(existing_boxes) + 1,
            "bbox": box,
            "quad": [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
            "detect_score": 0.7,
            "source": "a_module",
            "crop": None,
            "color": {"color": "yellow", "label": COLOR_LABELS["yellow"], "confidence": 0.9, "detail": {}},
            "segmentation": {
                "count": 10,
                "status": "count_abnormal",
                "boxes": [],
                "place_indices": [0, 1, 2],
                "message": "上下两行的京东城牌",
            },
            "recognition": {
                "text": text,
                "character_count": 10,
                "layout": "unrestricted",
                "characters": [
                    {"position": i, "label": label, "similarity": score, "status": "ok"}
                    for i, (label, score) in enumerate(zip(labels, scores))
                ],
                "status": "ok",
                "warnings": [],
            },
            "text": text,
            "confidence": round(float(np.mean(scores)), 3),
            "warnings": [],
            "timings": {"total_ms": 0.0},
            "views": {},
            "chars": images,
            "status": "ok",
        }

    # -------------------------------------------------------------- 内部
    @staticmethod
    def _empty_result(source, w, h, t0, detect_ms=0.0, error=None) -> dict:
        return {
            "ok": False,
            "source": source,
            "image_size": (w, h),
            "plate_count": 0,
            "plates": [],
            "rejected": 0,
            "timings": {"detect_ms": detect_ms, "total_ms": elapsed_ms(t0)},
            "warnings": [],
            "error": error or {"code": "no_plate", "message": "未检测到车牌"},
        }


# ------------------------------------------------------------------ 序列化
def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def result_to_json(result: dict) -> dict:
    """去掉 numpy 图像，只保留可写入 JSON 的字段（给 GUI 保存、评估复用）。"""
    out = dict(result)
    plates = []
    for p in result.get("plates", []):
        q = {k: v for k, v in p.items() if k not in ("crop", "chars", "views", "_accepted")}
        plates.append(_jsonable(q))
    out["plates"] = plates
    return _jsonable(out)

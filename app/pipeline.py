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

        if not detections:
            _progress(progress, "检测完成", 0.9, "未检测到车牌")
            return self._empty_result(source, w, h, t_all, detect_ms=detect_ms)

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
            "bbox": [int(v) for v in det["bbox"]],
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
            "message": {
                "ok": "",
                "count_abnormal": f"分割出 {seg['count']} 个字符（常见为 7 或 8 个）",
                "no_char": "没有切出任何字符",
            }.get(seg["status"], ""),
        }
        plate["chars"] = seg["chars"]
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
        layout = "standard" if seg["count"] in (7, 8) else "unrestricted"
        try:
            recognition = self.recognizer.recognize_chars(seg["chars"], layout=layout, top_k=3)
        except Exception as exc:
            LOG.exception("字符识别失败")
            recognition = {"text": "", "status": "recognize_failed",
                           "characters": [], "warnings": [str(exc)]}
            plate["warnings"].append(f"字符识别失败：{exc}")
        plate["timings"]["recognize_ms"] = elapsed_ms(t)
        plate["recognition"] = recognition
        plate["text"] = recognition.get("text", "") or ""

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

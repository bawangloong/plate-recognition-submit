# -*- coding: utf-8 -*-
"""仓库根目录的评测入口。

评测脚本导入的是根目录 ``pipeline.run(image_path)``，不是
``PlatePipeline.recognize_file``。这里把检测、矫正、颜色、分割和识别
串成一次调用，具体步骤仍由 ``app.pipeline.PlatePipeline`` 完成。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PIPELINE = None


def _pipeline():
    """字符库只加载一次，同一进程里多次 run 复用。"""
    global _PIPELINE
    if _PIPELINE is None:
        from app.pipeline import PlatePipeline

        _PIPELINE = PlatePipeline()
    return _PIPELINE


def _align_plate(plate: dict) -> dict:
    """把内部结果收成评测读取的字段。

    框从 ``[x, y, w, h]`` 改成 ``{x, y, w, h}``，颜色从字典改成
    ``blue`` 这类字符串，字数放到 ``char_count``。
    """
    aligned = dict(plate)
    box = plate.get("bbox") or [0, 0, 0, 0]
    if isinstance(box, dict):
        x = int(box.get("x", 0))
        y = int(box.get("y", 0))
        w = int(box.get("w", box.get("width", 0)))
        h = int(box.get("h", box.get("height", 0)))
    else:
        x, y, w, h = (int(v) for v in list(box)[:4])
    aligned["bbox"] = {"x": x, "y": y, "w": w, "h": h}

    color = plate.get("color")
    if isinstance(color, dict):
        aligned["color"] = str(color.get("color") or "unknown")
        aligned["color_detail"] = color
    elif color is None:
        aligned["color"] = "unknown"
    else:
        aligned["color"] = str(color)

    segmentation = plate.get("segmentation") or {}
    count = segmentation.get("count", 0) if isinstance(segmentation, dict) else 0
    aligned["char_count"] = int(count or 0)
    return aligned


def run(image_path):
    """对一张图片依次做检测、矫正、颜色、分割和识别。

    参数
        image_path: 图片路径，支持中文路径。
    返回
        可直接 ``json.dumps`` 的结果字典。``plates`` 里每块牌的
        ``bbox`` 是 ``{x, y, w, h}``，``color`` 是颜色字符串，
        ``char_count`` 是字符数。
    """
    from app.pipeline import result_to_json

    pipe = _pipeline()
    # recognize_file 内部顺序：读图 → 检测 → 矫正 → 颜色 → 分割 → 识别
    detected = pipe.recognize_file(image_path)
    result = result_to_json(detected)
    result["plates"] = [_align_plate(plate) for plate in result.get("plates", [])]
    return result

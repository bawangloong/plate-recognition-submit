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


def run(image_path):
    """对一张图片依次做检测、矫正、颜色、分割和识别。

    参数
        image_path: 图片路径，支持中文路径。
    返回
        可直接 ``json.dumps`` 的结果字典。至少包含 ``ok``、``plate_count``、
        ``plates``。每块车牌含矫正后的颜色、分割数量和识别号码。
    """
    from app.pipeline import result_to_json

    pipe = _pipeline()
    # recognize_file 内部顺序：读图 → 检测 → 矫正 → 颜色 → 分割 → 识别
    detected = pipe.recognize_file(image_path)
    return result_to_json(detected)

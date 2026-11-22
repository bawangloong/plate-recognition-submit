# -*- coding: utf-8 -*-
"""结果导出：标框图、车牌裁剪、过程四联图、字符切片、JSON 与文本摘要。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .pipeline import result_to_json
from .plate_utils import compose_process_panel, draw_detections, upscale_char
from .runtime import write_image_unicode


def export_result(result: dict, image_bgr: np.ndarray, out_root: str | Path,
                  source_name: str = "") -> Path:
    """把一次识别结果写到 out_root/<图片名>_结果/ 下，返回目录。"""
    root = Path(out_root)
    stem = Path(source_name or result.get("source") or "image").stem
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = root / f"{stem}_{stamp}_结果"
    folder.mkdir(parents=True, exist_ok=True)

    annotated = draw_detections(image_bgr, result.get("plates", []))
    write_image_unicode(folder / "01_原图标框.png", annotated)

    lines = [f"图片：{source_name or result.get('source', '')}",
             f"尺寸：{result['image_size'][0]}×{result['image_size'][1]}",
             f"车牌数：{result['plate_count']}",
             f"总耗时：{result['timings'].get('total_ms', 0):.0f} ms", ""]
    for plate in result.get("plates", []):
        idx = plate["index"]
        write_image_unicode(folder / f"车牌{idx}_裁剪.png", plate["crop"])
        write_image_unicode(folder / f"车牌{idx}_过程.png", compose_process_panel(plate))
        char_dir = folder / f"车牌{idx}_字符"
        char_dir.mkdir(exist_ok=True)
        chars = plate.get("chars") or []
        for i, ch in enumerate(chars, 1):
            write_image_unicode(char_dir / f"{i:02d}.png", upscale_char(ch, 3))
        lines.append(
            f"车牌{idx}：号码={plate['text'] or '(未识别)'}  "
            f"颜色={plate['color']['label']}  字符数={plate['segmentation']['count']}  "
            f"置信度={plate['confidence']:.2f}")
        for warning in plate.get("warnings", []):
            lines.append(f"    警告：{warning}")
    if result.get("error"):
        lines.append(f"错误：{result['error']['message']}")
    (folder / "02_结果摘要.txt").write_text("\n".join(lines), encoding="utf-8")
    (folder / "03_结果.json").write_text(
        json.dumps(result_to_json(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


def summary_text(result: dict) -> str:
    """给界面/日志用的一行摘要。"""
    if not result.get("ok"):
        err = result.get("error") or {}
        return err.get("message", "未识别到车牌")
    parts = []
    for plate in result["plates"]:
        parts.append(f"#{plate['index']} {plate['text'] or '?'}（{plate['color']['label']}，"
                     f"{plate['segmentation']['count']} 字符）")
    return "；".join(parts)

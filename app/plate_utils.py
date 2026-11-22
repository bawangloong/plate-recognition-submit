# -*- coding: utf-8 -*-
"""图像小工具：去除车牌外框、绘制结果、生成过程四联图。

这些是 D 的展示/集成工具，不改变 A/B/C 的算法输出。
"""

from __future__ import annotations

import cv2
import numpy as np

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"


def _font(size: int):
    try:
        from PIL import ImageFont
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return None


def strip_plate_frame(crop: np.ndarray) -> np.ndarray:
    """去掉与字符同色的车牌外框（蓝牌白框、黑牌白框），避免被当成字符。

    B 的处理链会删除“贴边”的连通域，但当 A 的裁剪把外框带进来、外框与图像
    边缘还有几个像素时，内框会残留并连成一整条，导致投影切分失败。这里在
    交给 B 之前，把“又宽又高/细长条”且靠近边缘的连通域抹成车牌底色。
    """
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 取少数像素作为“墨迹”（字符/边框），黄牌深字、蓝牌浅字都适用
    ink = bw if float(bw.mean()) < 127 else cv2.bitwise_not(bw)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    if n <= 1:
        return crop
    bg = np.median(crop.reshape(-1, 3), axis=0).astype(np.uint8)
    out = crop.copy()
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 0.0005 * h * w:
            continue
        wide, tall = ww > 0.78 * w, hh > 0.78 * h
        thin_h = hh < 0.16 * h
        thin_w = ww < 0.16 * w
        frame = (wide and tall) or (wide and thin_h) or (tall and thin_w)
        if not frame:
            continue
        # 只处理靠近边缘的框线，避免误伤细长的字符“1”
        near_edge = (x < 0.10 * w or y < 0.10 * h
                     or x + ww > 0.90 * w or y + hh > 0.90 * h)
        if near_edge:
            out[labels == i] = bg
    return out


def upscale_char(char: np.ndarray, scale: int = 4) -> np.ndarray:
    """把 20×40 的字符图放大用于界面显示（最近邻保持二值边缘）。"""
    if char is None or char.size == 0:
        return np.zeros((40, 20), np.uint8)
    arr = np.asarray(char).astype(np.uint8)
    return cv2.resize(arr, (arr.shape[1] * scale, arr.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)


def draw_detections(image: np.ndarray, plates: list[dict], thickness: int | None = None) -> np.ndarray:
    """在原图上画车牌框和序号，返回新图（BGR）。"""
    out = image.copy()
    h, w = out.shape[:2]
    base = thickness or max(2, int(min(h, w) / 400))
    for plate in plates:
        quad = plate.get("quad")
        color = (0, 200, 0)
        if quad:
            pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, color, base + 1, cv2.LINE_AA)
        x, y, bw, bh = plate["bbox"]
        label = f"#{plate.get('index', 0)} {plate.get('text', '')}".strip()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.5, min(1.2, min(h, w) / 900))
        (tw, th), _ = cv2.getTextSize(label, font, fs, base)
        ty = max(th + 6, y - 6)
        cv2.rectangle(out, (x, ty - th - 6), (x + tw + 8, ty + 4), (0, 160, 0), -1)
        cv2.putText(out, label, (x + 4, ty - 2), font, fs, (255, 255, 255), base,
                    cv2.LINE_AA)
    return out


def compose_process_panel(plate: dict, char_display: int = 56) -> np.ndarray:
    """生成“原车牌—二值化—字符框—逐个字符”的四联过程图（BGR）。

    供 GUI「处理过程」页、报告和演示视频直接使用。
    """
    crop = plate.get("crop")
    views = plate.get("views", {})
    seg = plate.get("segmentation") or {}
    boxes = seg.get("boxes") or []
    chars = plate.get("chars") or []

    if crop is None:
        return np.full((260, 900, 3), 245, np.uint8)

    target_w = 420
    def fit(img):
        if img is None or img.size == 0:
            return np.full((140, target_w, 3), 240, np.uint8)
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        scale = target_w / arr.shape[1]
        return cv2.resize(arr, (target_w, max(1, int(arr.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)

    panel_h = 220
    canvas = np.full((panel_h + 90, target_w * 4 + 60, 3), 250, np.uint8)
    titles = ["① 车牌裁剪", "② 二值化", "③ 字符框", "④ 字符切片"]
    imgs = [fit(crop), fit(views.get("binary")), None, None]

    # 字符框
    norm = views.get("norm")
    box_img = fit(norm if norm is not None else crop)
    scale = box_img.shape[1] / max(1, (norm.shape[1] if norm is not None else crop.shape[1]))
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(box_img, (int(x * scale), int(y * scale)),
                      (int((x + bw) * scale), int((y + bh) * scale)), (0, 220, 0), 2)
    imgs[2] = box_img

    # 字符切片拼成一行
    if chars:
        cw = upscale_char(chars[0])
        strip = np.full((cw.shape[0], len(chars) * (cw.shape[1] + 6), 3), 255, np.uint8)
        for i, c in enumerate(chars):
            cc = upscale_char(c)
            x = i * (cw.shape[1] + 6)
            strip[:cc.shape[0], x:x + cc.shape[1]] = cv2.cvtColor(cc, cv2.COLOR_GRAY2BGR)
        imgs[3] = fit(strip)

    font = _font(20)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        x = 10 + i * (target_w + 13)
        canvas[36:36 + img.shape[0], x:x + target_w] = img
        if font is not None:
            from PIL import Image, ImageDraw
            pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(pil).text((x + 4, 8), title, font=font, fill=(20, 20, 20))
            canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return canvas


def fit_to_box(image: np.ndarray, max_w: int, max_h: int) -> tuple[np.ndarray, float]:
    """等比缩放到显示区域，返回 (图, 缩放比例)。"""
    h, w = image.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale >= 0.999:
        return image, 1.0
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale

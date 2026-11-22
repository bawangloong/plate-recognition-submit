# -*- coding: utf-8 -*-
"""生成用于 GUI 演示与回归测试的合成场景图。

A 的真实数据未到位时，用这些图验证“整图 → 检测 → 矫正 → 分割 → 识别”的
全流程。合成图只验证流程与工程实现，不能替代真实路测精度。

用法：python tools/make_demo_images.py --out samples
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]

PLATE_W, PLATE_H = 440, 140


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ------------------------------------------------------------------ 车牌
def render_plate(text: str, bg: str, fg: str, gradient: bool = False) -> np.ndarray:
    """渲染一块 440×140 的车牌（RGB），含边框和铆钉。"""
    plate = Image.new("RGB", (PLATE_W, PLATE_H), bg)
    if gradient:
        top = np.array([245, 250, 245], dtype=np.float32)
        bottom = np.array([0, 190, 90], dtype=np.float32)
        for y in range(PLATE_H):
            t = y / (PLATE_H - 1)
            row = (top * (1 - t) + bottom * t).astype(np.uint8)
            ImageDraw.Draw(plate).line([(0, y), (PLATE_W, y)],
                                       fill=tuple(int(v) for v in row))
    draw = ImageDraw.Draw(plate)
    draw.rectangle([0, 0, PLATE_W - 1, PLATE_H - 1], outline=(255, 255, 255), width=4)
    draw.rectangle([7, 7, PLATE_W - 8, PLATE_H - 8], outline=(255, 255, 255), width=2)

    count = len(text)
    font_size = 92 if count <= 7 else 78
    font = load_font(font_size)
    gap = 10 if count <= 7 else 4
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + gap * (count - 1)
    x = (PLATE_W - total) / 2
    for ch, w in zip(text, widths):
        bbox = draw.textbbox((0, 0), ch, font=font)
        y = (PLATE_H - (bbox[3] - bbox[1])) / 2 - bbox[1]
        draw.text((x, y), ch, font=font, fill=fg)
        x += w + gap

    for cx, cy in ((18, 18), (PLATE_W - 19, 18)):
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(210, 210, 210))
    return np.array(plate)


def paste_plate(scene: np.ndarray, plate_rgb: np.ndarray, center, scale=1.0,
                angle=0.0, perspective=0.0, jitter_seed=0) -> np.ndarray:
    """把车牌按旋转/透视贴到场景中，返回 scene。"""
    ph, pw = plate_rgb.shape[:2]
    h, w = scene.shape[:2]
    rng = random.Random(jitter_seed)
    sw, sh = pw * scale, ph * scale

    src = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]])
    dst = src.copy()
    cx, cy = center
    dst[:, 0] += cx - sw / 2
    dst[:, 1] += cy - sh / 2

    if perspective:
        for i in range(4):
            dst[i, 0] += rng.uniform(-1, 1) * sw * perspective
            dst[i, 1] += rng.uniform(-1, 1) * sh * perspective

    rad = math.radians(angle)
    rot = np.float32([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
    c = np.float32([cx, cy])
    dst = (dst - c) @ rot.T + c

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate_rgb, matrix, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), matrix, (w, h),
                               flags=cv2.INTER_NEAREST)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    out = scene.astype(np.float32) * (1 - mask3) + warped.astype(np.float32) * mask3
    return out.astype(np.uint8)


def make_scene(w=1280, h=800, seed=1, cars=0, plate=None, **plate_kw):
    """画一个路口/停车场背景，可选放一辆车。"""
    rng = np.random.default_rng(seed)
    scene = np.zeros((h, w, 3), np.uint8)

    horizon = int(h * 0.42)
    for y in range(horizon):
        t = y / max(1, horizon)
        scene[y, :] = (int(120 + 60 * t), int(150 + 50 * t), int(190 + 40 * t))
    for y in range(horizon, h):
        t = (y - horizon) / max(1, h - horizon)
        v = int(120 - 50 * t)
        scene[y, :] = (v + 4, v + 6, v + 8)

    for _ in range(6):
        bx = int(rng.integers(0, w - 80))
        bw = int(rng.integers(60, 180))
        bh = int(rng.integers(40, 140))
        color = tuple(int(v) for v in rng.integers(80, 150, 3))
        cv2.rectangle(scene, (bx, horizon - bh), (bx + bw, horizon), color, -1)

    for k in range(6):
        y0 = horizon + int((h - horizon) * k / 6) + 10
        y1 = y0 + int((h - horizon) / 12)
        cv2.line(scene, (w // 2 + 20, y0), (w // 2 + 80, y1), (240, 240, 240), 6)

    scene = cv2.GaussianBlur(scene, (3, 3), 0)

    if cars:
        for i in range(cars):
            cx = int(w * (0.3 + 0.42 * i))
            cy = int(h * (0.68 - 0.06 * i))
            body_w = int(w * (0.34 - 0.06 * i))
            body_h = int(h * (0.30 - 0.05 * i))
            color = [(60, 70, 140), (150, 150, 150), (50, 50, 50)][i % 3]
            x0, y0 = cx - body_w // 2, cy - body_h // 2
            cv2.rectangle(scene, (x0, y0), (x0 + body_w, y0 + body_h), color, -1)
            cv2.rectangle(scene, (x0 + body_w // 8, y0 + body_h // 10),
                          (x0 + body_w * 7 // 8, y0 + body_h * 4 // 10), (40, 45, 55), -1)
            cv2.rectangle(scene, (x0 + 12, y0 + body_h * 2 // 3),
                          (x0 + body_w // 4, y0 + body_h * 2 // 3 + 18), (30, 30, 200), -1)
            cv2.rectangle(scene, (x0 + body_w * 3 // 4, y0 + body_h * 2 // 3),
                          (x0 + body_w - 12, y0 + body_h * 2 // 3 + 18), (30, 30, 200), -1)
            if plate is not None and i == 0:
                rgb = cv2.cvtColor(render_plate(*plate), cv2.COLOR_RGB2BGR)
                kw = {k: v for k, v in plate_kw.items() if k != "center"}
                scene = paste_plate(scene, rgb, (cx, cy + body_h // 2 - 30), **kw)
    return scene


def add_glare(img, cx_ratio=0.7, cy_ratio=0.4, strength=0.65):
    h, w = img.shape[:2]
    overlay = np.zeros_like(img, np.float32)
    cv2.circle(overlay, (int(w * cx_ratio), int(h * cy_ratio)), int(min(h, w) * 0.3),
               (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=min(h, w) * 0.08)
    out = img.astype(np.float32) + overlay * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def save(path: Path, img: np.ndarray, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    params = []
    if path.suffix.lower() in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buf = cv2.imencode(path.suffix, img, params)
    if not ok:
        raise SystemExit(f"保存失败 {path}")
    buf.tofile(str(path))
    print("saved", path, img.shape)


def build(out: Path):
    blue = ("京A12345", (30, 90, 168), (255, 255, 255))
    yellow = ("粤B88888", (240, 200, 30), (20, 20, 20))
    green = ("沪AD12345", (0, 190, 90), (20, 20, 20))
    black = ("京A88888", (25, 25, 25), (245, 245, 245))
    white = ("京A12345", (240, 240, 240), (25, 25, 25))

    img = make_scene(1280, 800, seed=1, cars=1, plate=blue,
                     center=(640, 560), scale=1.15, angle=0, perspective=0.0,
                     jitter_seed=11)
    save(out / "normal_blue.png", img)

    img = make_scene(1280, 800, seed=2, cars=1, plate=yellow,
                     center=(620, 545), scale=1.25, angle=13, perspective=0.035,
                     jitter_seed=12)
    save(out / "tilt_yellow.png", img)

    img = make_scene(1600, 1000, seed=3, cars=2, plate=blue,
                     center=(480, 700), scale=1.05, angle=-3, perspective=0.01,
                     jitter_seed=13)
    rgb = cv2.cvtColor(render_plate(*green), cv2.COLOR_RGB2BGR)
    img = paste_plate(img, rgb, (1050, 650), scale=0.95, angle=5, perspective=0.02,
                      jitter_seed=14)
    save(out / "multi_two.png", img)

    img = make_scene(1280, 800, seed=4, cars=0)
    save(out / "no_plate.png", img)

    img = make_scene(1280, 800, seed=5, cars=1, plate=green,
                     center=(640, 555), scale=1.2, angle=0, perspective=0.0,
                     jitter_seed=15)
    save(out / "green_8.png", img)

    img = make_scene(1280, 800, seed=6, cars=1, plate=white,
                     center=(620, 560), scale=1.2, angle=4, perspective=0.01,
                     jitter_seed=16)
    img = add_glare(img, 0.55, 0.55, 0.7)
    img = cv2.GaussianBlur(img, (7, 7), 1.6)
    save(out / "blur_glare.png", img)

    img = make_scene(1280, 800, seed=7, cars=1, plate=black,
                     center=(640, 560), scale=1.2, angle=-6, perspective=0.02,
                     jitter_seed=17)
    save(out / "black_plate.png", img)

    img = make_scene(3000, 2000, seed=8, cars=1, plate=blue,
                     center=(1500, 1380), scale=2.4, angle=2, perspective=0.008,
                     jitter_seed=18)
    save(out / "large_3000.jpg", img, quality=92)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "samples")
    args = parser.parse_args()
    build(args.out)

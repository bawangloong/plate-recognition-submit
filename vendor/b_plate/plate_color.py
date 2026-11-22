# -*- coding: utf-8 -*-
"""
plate_color.py —— 车牌颜色识别

输入：彩色车牌图片（BGR 或 RGB 数组 / 图片路径）
输出：颜色类别 + 置信度 + 颜色统计信息

支持颜色：蓝牌、黄牌、绿牌（新能源）、白牌、黑牌
方法：HSV 色彩空间统计 + 饱和度阈值判断（传统方法，不依赖深度学习）
"""

import numpy as np

# 颜色类别与中文名称
COLOR_LABELS = {
    "blue": "蓝牌",
    "yellow": "黄牌",
    "green": "绿牌（新能源）",
    "white": "白牌",
    "black": "黑牌",
    "unknown": "无法判断",
}

# 判定阈值（可在调参时修改）
SAT_TH = 60          # 饱和度阈值：低于此值视为"无彩色"（白/黑牌）
VAL_TH = 60          # 亮度下限：低于此值视为黑牌
SAT_RATIO_TH = 0.30  # 有彩色像素占比下限：低于此值走白/黑牌分支

HUE_RANGE = {
    # (下限, 上限)，单位为度；上限为 None 表示到 180 度
    "blue": (190, 260),
    "yellow": (20, 65),
    "green": (65, 190),
}


def rgb_to_hsv_array(rgb):
    """RGB(0-255) -> HSV(0-360, 0-1, 0-1)，向量化实现，与 cv2 结果同量纲。"""
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff > 1e-6

    idx = mask & (mx == r)
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6
    idx = mask & (mx == g)
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2
    idx = mask & (mx == b)
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4
    h = h * 60.0

    s = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    v = mx
    return h, s, v


def _to_rgb(img, assume_bgr=False):
    """统一转成 RGB；assume_bgr=True 时把 BGR 翻成 RGB。"""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if assume_bgr:
        arr = arr[..., ::-1]
    return arr


def analyze_color(img, assume_bgr=False, crop_ratio=0.10):
    """
    统计车牌区域的颜色构成。

    crop_ratio: 去掉四周边缘的比例，避免边框、铆钉干扰统计。
    返回：dict(hist 各颜色像素占比, sat_ratio, median_v)
    """
    rgb = _to_rgb(img, assume_bgr)
    h, w = rgb.shape[:2]
    m = int(min(h, w) * crop_ratio)
    core = rgb[m:h - m, m:w - m] if (h - 2 * m > 4 and w - 2 * m > 4) else rgb

    hue, sat, val = rgb_to_hsv_array(core)

    saturated = (sat * 255 > SAT_TH) & (val * 255 > VAL_TH)
    sat_ratio = float(saturated.mean())

    counts = {}
    total = saturated.sum()
    if total > 0:
        for name, (lo, hi) in HUE_RANGE.items():
            sel = saturated & (hue >= lo) & (hue < hi)
            counts[name] = float(sel.sum()) / total
    for name in COLOR_LABELS:
        counts.setdefault(name, 0.0)

    return {
        "hist": counts,
        "sat_ratio": round(sat_ratio, 4),
        "median_v": float(np.median(val) * 255),
    }


def classify_plate_color(img, assume_bgr=False, crop_ratio=0.10):
    """
    判断车牌颜色。

    返回：dict(
        color='blue'|'yellow'|'green'|'white'|'black'|'unknown',
        label='蓝牌' 等中文名,
        confidence=0~1,
        detail=统计信息
    )
    """
    stat = analyze_color(img, assume_bgr=assume_bgr, crop_ratio=crop_ratio)
    hist = stat["hist"]
    med_v = stat["median_v"]

    if stat["sat_ratio"] < SAT_RATIO_TH:
        # 无彩色：按亮度分白牌 / 黑牌
        if med_v > 150:
            color, conf = "white", 1.0 - stat["sat_ratio"]
        elif med_v < 80:
            color, conf = "black", 1.0 - stat["sat_ratio"]
        else:
            color, conf = "unknown", 0.0
    else:
        color = max(("blue", "yellow", "green"), key=lambda c: hist[c])
        conf = hist[color]
        if conf < 0.45:
            color, conf = "unknown", conf

    return {
        "color": color,
        "label": COLOR_LABELS[color],
        "confidence": round(float(conf), 3),
        "detail": stat,
    }


def is_dark_text_on_light_plate(color):
    """该颜色的车牌，字符是深色（True）还是浅色（False）。用于分割时决定二值化极性。"""
    return color in ("yellow", "green", "white")


if __name__ == "__main__":
    import sys
    from PIL import Image

    if len(sys.argv) < 2:
        print("用法: python plate_color.py <车牌图片> [more...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        arr = np.array(Image.open(path).convert("RGB"))
        r = classify_plate_color(arr)
        print("%-28s -> %-12s 置信度 %.2f  sat=%.2f  V=%.0f"
              % (path.split("/")[-1], r["label"], r["confidence"],
                 r["detail"]["sat_ratio"], r["detail"]["median_v"]))

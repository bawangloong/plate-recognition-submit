# -*- coding: utf-8 -*-
"""内置的车牌检测 / 矫正兜底实现（D 的集成后备，不是 A 的最终算法）。

A 同学的正式模块到位后，把 ``detect_plate.py`` 与 ``rectify_plate.py`` 放到
``vendor/a_plate/`` 即可自动接管（接口见 docs/A模块接口约定.md）。
本文件只依赖 numpy + OpenCV，负责在 A 未交付时让整套系统可运行、可测试。

思路
----
1. 缩放到长边 ≤1600 提速；
2. 蓝 / 黄 / 绿三种底色用 HSV 阈值得到候选，白色和黑色车牌用梯度纹理候选；
3. 形态学横向闭运算把字符连成块，取外轮廓 + 最小外接矩形；
4. 用宽高比、面积、矩形度筛掉大部分背景；
5. 透视矫正后做“投影字符数 + 字符高度一致性”打分，保留像车牌的候选；
6. 重叠候选做 NMS，最多保留 4 块。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .runtime import BUNDLE

# 单行车牌宽高比约 3.14；倾斜、透视、边缘裁切会带来偏差，放宽容差
MIN_ASPECT, MAX_ASPECT = 2.0, 6.4
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.00035, 0.45
MAX_PLATES = 4
DETECT_MAX_SIDE = 1600
RECTIFY_WIDTH = 440


# ------------------------------------------------------------------ 几何工具
def _order_points(pts: np.ndarray) -> np.ndarray:
    """把 4 个点排成 左上、右上、右下、左下。车牌倾角 <45°，用和/差排序足够稳。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _quad_side_lengths(quad: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = _order_points(quad)
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return float(w), float(h)


def _rect_quad(rect) -> np.ndarray:
    box = cv2.boxPoints(rect)
    return _order_points(box)


def _bbox_of_quad(quad, shape) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x, y, bw, bh = cv2.boundingRect(np.asarray(quad, dtype=np.float32).astype(np.int32))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    return int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))


def _expand_quad(quad: np.ndarray, scale: float) -> np.ndarray:
    c = quad.mean(axis=0)
    return (c + (quad - c) * scale).astype(np.float32)


def _expand_short_axis(quad: np.ndarray, factor: float) -> np.ndarray:
    """只沿车牌短边方向加高，用于补全绿牌上半部分（渐变白底未被掩膜覆盖）。"""
    tl, tr, br, bl = _order_points(quad)
    uy = (bl - tl)
    length = float(np.linalg.norm(uy))
    if length < 1e-3:
        return quad
    uy = uy / length
    delta = uy * (length * (factor - 1.0) / 2.0)
    return np.array([tl - delta, tr - delta, br + delta, bl + delta], dtype=np.float32)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------ 候选生成
def _color_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    """中国车牌常见底色：蓝、黄、绿（新能源）。返回 bool 掩膜。"""
    h, s, v = cv2.split(hsv)
    return {
        "blue": cv2.inRange(hsv, (100, 65, 40), (142, 255, 255)),
        "yellow": cv2.inRange(hsv, (10, 70, 80), (38, 255, 255)),
        "green": cv2.inRange(hsv, (35, 40, 40), (98, 255, 255)),
    }


def _mask_contours(mask: np.ndarray, kx: int, ky: int):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, kx), max(3, ky)))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _geometry_ok(quad, shape, min_area_ratio=MIN_AREA_RATIO):
    h, w = shape[:2]
    rw, rh = _quad_side_lengths(quad)
    if rh < 12 or rw < 36:
        return None
    aspect = rw / rh
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        return None
    rect = cv2.minAreaRect(np.asarray(quad, dtype=np.float32))
    rect_area = rw * rh
    if rect_area / float(h * w) < min_area_ratio:
        return None
    if rect_area / float(h * w) > MAX_AREA_RATIO:
        return None
    # 最小外接矩形与凸包面积比，排除细长噪点
    hull_area = cv2.contourArea(cv2.convexHull(np.asarray(quad, dtype=np.float32)))
    if rect_area <= 0 or hull_area / rect_area < 0.35:
        return None
    return rw, rh, aspect


def _gradient_candidates(gray: np.ndarray):
    """白牌 / 黑牌没有稳定底色，用强垂直边缘 + 横向闭运算找文本块。"""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    gx = cv2.normalize(gx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bw = cv2.threshold(gx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    h, w = gray.shape
    kx = max(15, int(w * 0.018))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw = cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 60:
            continue
        quad = _rect_quad(cv2.minAreaRect(c))
        if _geometry_ok(quad, gray.shape, MIN_AREA_RATIO * 1.5):
            ww, hh = _quad_side_lengths(quad)
            box = cv2.boundingRect(c)
            fill = cv2.contourArea(c) / max(1.0, float(box[2] * box[3]))
            yield quad, float(fill)


def _candidates(small: np.ndarray):
    h, w = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    masks = _color_masks(hsv)
    # 两种尺度：小车牌用大核，大车牌用小核
    kernels = [(max(9, int(w * 0.012)), max(3, int(h * 0.004))),
               (max(13, int(w * 0.024)), max(3, int(h * 0.006)))]

    for color, mask in masks.items():
        for kx, ky in kernels:
            for c in _mask_contours(mask, kx, ky):
                area = cv2.contourArea(c)
                if area < 80:
                    continue
                quad = _rect_quad(cv2.minAreaRect(c))
                geo = _geometry_ok(quad, small.shape)
                if geo is None:
                    continue
                rw, rh, aspect = geo
                # 绿牌是白→绿渐变，掩膜常只覆盖下半部分：按标准宽高比补高
                if color == "green":
                    target_h = rw / 3.14
                    factor = min(1.45, max(1.0, target_h / max(1.0, rh)))
                    quad = _expand_short_axis(quad, factor)
                elif color == "yellow":
                    quad = _expand_quad(quad, 1.04)
                else:
                    quad = _expand_quad(quad, 1.04)
                box = cv2.boundingRect(quad.astype(np.int32))
                fill = area / max(1.0, float(box[2] * box[3]))
                # 宽高比越接近 3.14 越可信
                aspect_prior = max(0.0, 1.0 - abs(aspect - 3.14) / 3.2)
                yield {
                    "quad": quad,
                    "bbox": _bbox_of_quad(quad, small.shape),
                    "score": 0.55 + 0.25 * aspect_prior + 0.2 * min(1.0, fill),
                    "source": f"color:{color}",
                }

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    for quad, fill in _gradient_candidates(gray):
        yield {
            "quad": quad,
            "bbox": _bbox_of_quad(quad, small.shape),
            "score": 0.35 + 0.25 * min(1.0, fill),
            "source": "gradient",
        }


# ------------------------------------------------------------------ 打分
def _plate_likelihood(plate: np.ndarray) -> tuple[float, int]:
    """在矫正后的车牌上快速判断“像不像一行字”，返回 (0~1 分数, 投影字符数)。"""
    if plate is None or plate.size == 0:
        return 0.0, 0
    h, w = plate.shape[:2]
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 统一成“白字黑底”：前景应为少数像素
    if float(bw.mean()) > 127.0:
        bw = cv2.bitwise_not(bw)
    fg_ratio = float(np.count_nonzero(bw)) / float(h * w)
    if not (0.02 <= fg_ratio <= 0.45):
        return 0.0, 0

    col = (bw > 0).sum(axis=0).astype(np.float32)
    on = col > max(2.0, 0.06 * h)
    segs = []
    start = None
    gap = 0
    min_gap = max(2, int(0.012 * w))
    for x, v in enumerate(on):
        if v:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                segs.append((start, x - gap))
                start = None
    if start is not None:
        segs.append((start, len(on) - 1))

    chars = []
    for x0, x1 in segs:
        sub = bw[:, x0:x1 + 1]
        rows = np.where((sub > 0).any(axis=1))[0]
        if len(rows) == 0:
            continue
        ch = int(rows.max() - rows.min() + 1)
        cw = int(x1 - x0 + 1)
        if ch < 0.35 * h or cw < 0.02 * w:
            continue
        chars.append((cw, ch))
    count = len(chars)
    if count == 0:
        return 0.0, 0

    if count in (7, 8):
        count_score = 1.0
    elif count in (6, 9):
        count_score = 0.78
    elif count in (5, 10):
        count_score = 0.45
    else:
        count_score = 0.0

    heights = np.array([c[1] for c in chars], dtype=np.float32)
    std_ratio = float(heights.std() / max(1.0, heights.mean()))
    consistency = max(0.0, 1.0 - min(1.0, std_ratio / 0.40))
    # 字符高度应占车牌高度的 40%~75%
    median_h = float(np.median(heights)) / h
    height_prior = 1.0 if 0.40 <= median_h <= 0.72 else 0.35
    score = 0.55 * count_score + 0.25 * consistency + 0.20 * height_prior
    return round(float(score), 3), count


def rectify_from_quad(bgr: np.ndarray, quad: np.ndarray, out_width: int = RECTIFY_WIDTH):
    """按四点透视变换把车牌摆正，返回 (车牌图, 四点)。"""
    tl, tr, br, bl = _order_points(quad)
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if w < 1 or h < 1:
        return None
    out_w = int(out_width)
    out_h = int(round(out_w * h / w))
    out_h = max(40, min(out_h, out_w))
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, matrix, (out_w, out_h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # 透视线条可能带进 1~2 像素背景，收一点边
    pad_x, pad_y = int(out_w * 0.01), int(out_h * 0.02)
    if out_w - 2 * pad_x > 20 and out_h - 2 * pad_y > 10:
        warped = warped[pad_y:out_h - pad_y, pad_x:out_w - pad_x]
    return warped


class FallbackBackend:
    """A 模块未提供时的内置实现。"""

    name = "内置兜底检测（A 模块到位后自动替换）"

    def detect(self, bgr: np.ndarray) -> list[dict]:
        h, w = bgr.shape[:2]
        if max(h, w) > DETECT_MAX_SIDE:
            scale = DETECT_MAX_SIDE / float(max(h, w))
            small = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = bgr

        raw = list(_candidates(small))
        # 先按分数排序，再做 NMS
        raw.sort(key=lambda c: c["score"], reverse=True)
        kept = []
        for cand in raw:
            if all(_iou(cand["bbox"], k["bbox"]) < 0.35 for k in kept):
                kept.append(cand)
            if len(kept) >= 12:
                break

        results = []
        inv = 1.0 / scale
        for cand in kept:
            quad = cand["quad"] * inv
            plate = rectify_from_quad(bgr, quad)
            score, count = _plate_likelihood(plate)
            if score < 0.32:
                continue
            results.append({
                "quad": quad,
                "bbox": _bbox_of_quad(quad, bgr.shape),
                "detect_score": round(cand["score"], 3),
                "plate_score": score,
                "projection_chars": count,
                "source": cand["source"],
            })

        # 二次 NMS（不同颜色掩膜会给出几乎相同的框）
        results.sort(key=lambda r: r["plate_score"], reverse=True)
        final = []
        for r in results:
            if all(_iou(r["bbox"], f["bbox"]) < 0.30 for f in final):
                final.append(r)
            if len(final) >= MAX_PLATES:
                break
        final.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        return final

    def rectify(self, bgr: np.ndarray, detection: dict) -> np.ndarray | None:
        return rectify_from_quad(bgr, detection["quad"])


# ------------------------------------------------------------------ A 模块适配
class ABackend:
    """加载 A 同学交付的 detect_plate.py / rectify_plate.py。

    约定：detect_plates(image) 返回 [{'bbox':(x,y,w,h), 'quad':4x2, 'score':float}]，
    坐标为原图；rectify_plate(image, plate) 返回矫正后的 BGR 图。
    """

    def __init__(self, module_dir: Path):
        import importlib.util
        import sys

        self.dir = Path(module_dir)
        self.name = f"A 模块（{self.dir.name}）"
        for mod_name in ("detect_plate", "rectify_plate"):
            path = self.dir / f"{mod_name}.py"
            if not path.is_file():
                raise FileNotFoundError(f"缺少 {path}")
            spec = importlib.util.spec_from_file_location(f"a_plate_{mod_name}", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载 {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            setattr(self, mod_name, module)
        if not hasattr(self.detect_plate, "detect_plates"):
            raise AttributeError("detect_plate.py 必须提供 detect_plates(image)")

    def detect(self, bgr: np.ndarray) -> list[dict]:
        out = self.detect_plate.detect_plates(bgr)
        results = []
        for item in out or []:
            bbox = item.get("bbox")
            quad = item.get("quad")
            if bbox is None and quad is not None:
                bbox = _bbox_of_quad(np.asarray(quad), bgr.shape)
            if bbox is None:
                continue
            results.append({
                "bbox": tuple(int(v) for v in bbox[:4]),
                "quad": np.asarray(quad, dtype=np.float32) if quad is not None else None,
                "detect_score": float(item.get("score", 0.5)),
                "plate_score": float(item.get("score", 0.5)),
                "source": "a_module",
            })
        return results

    def rectify(self, bgr: np.ndarray, detection: dict) -> np.ndarray | None:
        func = getattr(self.rectify_plate, "rectify_plate", None)
        if func is not None:
            return func(bgr, detection)
        return rectify_from_quad(bgr, detection["quad"])


def load_backend() -> object:
    """优先使用 A 的模块，找不到时退回内置实现。"""
    candidates = []
    env = __import__("os").environ.get("PLATE_A_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(BUNDLE / "vendor" / "a_plate")
    for folder in candidates:
        try:
            if (folder / "detect_plate.py").is_file() and (folder / "rectify_plate.py").is_file():
                return ABackend(folder)
        except Exception:
            continue
    return FallbackBackend()

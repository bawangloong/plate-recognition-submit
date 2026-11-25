# -*- coding: utf-8 -*-
"""
detect_plate.py —— A 模块：车牌检测与定位
==========================================
方法：颜色(HSV) + 边缘(Sobel) + 形态学 双通道候选 → 几何筛选 → 字符纹理验证 → 综合评分 → NMS 去重。
不依赖神经网络，可解释、离线可运行；蓝/黄/绿/白底车牌均覆盖。

对外接口（D 集成时只调用这一个函数）：
    detect_plates(image_bgr: np.ndarray) -> list[PlateRegion]
    * 坐标全部对应原图；无车牌时返回 []。
    * 结果按 score 降序排列。

独立运行：
    python detect_plate.py 图片路径 [--out 输出目录]
"""

import argparse
import math

import cv2
import numpy as np

from plate_common import (PlateRegion, load_image, save_image,
                          regions_to_json, order_quad)

# ---------------- 可调参数（E 调参时改这里，勿散落各处） ----------------
DETECT_LONG_SIDE = 1024      # 检测前把长边缩到该尺寸（坐标返回前会换算回原图）
MIN_PLATE_W = 60             # 原图中车牌最小宽度（像素），过小视为噪声
MIN_PLATE_H = 16             # 原图中车牌最小高度（像素）
RATIO_RANGE = (1.7, 4.8)     # 最终验收长宽比（标准牌 3.14，新能源 3.43）
RATIO_INIT = (1.6, 4.6)      # 初筛范围（与验收一致）
IDEAL_RATIO = 3.2            # 评分用的理想长宽比
RATIO_SIGMA = 0.7            # 长宽比评分的高斯宽度（越小越偏好标准牌）
SCORE_THRESH = 0.45          # 综合得分阈值，低于则丢弃
NMS_IOU = 0.3                # 去重 IoU 阈值
MAX_PLATES = 8               # 单图最多返回车牌数（防误检爆炸）
FULL_WIDTH_FRAC = 0.95       # 候选框宽超过图宽该比例视为背景色带，拒绝

# HSV 底色范围（OpenCV H:0-179, S/V:0-255）。B 的颜色识别可参考但独立实现。
# 蓝色放宽 S/V 下限以覆盖阴影/褪色蓝牌（实测真实蓝牌可低至 S≈40,V≈85）。
COLOR_RANGES = {
    "blue":   [((95, 35, 40), (130, 255, 255))],
    "yellow": [((15, 90, 80), (35, 255, 255))],
    "green":  [((38, 50, 50), (90, 255, 255))],   # 新能源渐变绿的宽松范围
    "white":  [((0, 0, 160), (179, 55, 255))],
}
MIN_COLOR_RATIO = 0.30       # 底色覆盖率门槛：低于此值视为无明确牌色的背景纹理


def _color_masks(hsv: np.ndarray) -> dict:
    masks = {}
    for name, ranges in COLOR_RANGES.items():
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        masks[name] = m
    return masks


def _edge_mask(gray: np.ndarray) -> np.ndarray:
    """竖直边缘密集区（车牌字符区特征）→ 形态学闭运算连成块。"""
    blur = cv2.bilateralFilter(gray, 9, 60, 60)
    sobel = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    _, bw = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    w = gray.shape[1]
    kw, kh = max(9, int(w * 0.017)), max(3, int(w * 0.004))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (kw // 2, kh)))
    return opened


def _quad_norm(quad: np.ndarray):
    """规范化 quad：保证 0→1 为长边，返回 (quad, w, h, angle)。"""
    quad = np.asarray(quad, dtype=np.float32)
    e_top = np.linalg.norm(quad[1] - quad[0])
    e_left = np.linalg.norm(quad[3] - quad[0])
    if e_left > e_top:
        quad = np.roll(quad, -1, axis=0)
        e_top, e_left = e_left, e_top
    angle = math.degrees(math.atan2(quad[1][1] - quad[0][1],
                                    quad[1][0] - quad[0][0]))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return quad, float(e_top), float(e_left), float(angle)


def _rect_geometry(rect):
    """minAreaRect → (quad 排序角点, w_long, h_short, angle_deg)。"""
    return _quad_norm(order_quad(cv2.boxPoints(rect)))


def _rot_crop(img: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, m, (out_w, out_h))


def _mask_ratio_in_quad(mask: np.ndarray, quad: np.ndarray) -> float:
    """quad 区域内 mask 命中像素占比（颜色覆盖率）。"""
    crop = _rot_crop(mask, quad, 120, 40)
    return float(np.count_nonzero(crop)) / crop.size


def _refine_quad(mask: np.ndarray, quad: np.ndarray):
    """用底色掩码把粗候选框收紧到真实车牌边界。返回 (quad, 填充率)，失败返回 None。
    填充率 = 掩码轮廓面积 / 外接矩形面积；车牌≈0.95，与背景色块粘连时明显偏低。"""
    cw, ch = 160, 50
    dst = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    crop = cv2.warpPerspective(mask, m, (cw, ch))
    crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)))
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    if area < 0.25 * cw * ch:
        return None
    rect = cv2.minAreaRect(biggest)
    fill = area / max(1.0, rect[1][0] * rect[1][1])
    box = cv2.boxPoints(rect)
    inv = np.linalg.inv(m)
    pts = np.hstack([box, np.ones((4, 1))]) @ inv.T
    return order_quad(pts[:, :2] / pts[:, 2:3]), float(fill)


def _edge_extent(ecrop: np.ndarray):
    """找边缘图中“字符区域”的行列范围。
    只保留连续 >=4 行/列的活跃带，排除背景色块边界这类孤立细线。"""
    h, w = ecrop.shape
    col = ecrop.mean(axis=0) / 255.0
    row = ecrop.mean(axis=1) / 255.0

    def extent(profile, min_run):
        active = profile > 0.10
        keep = np.zeros_like(active)
        run_start = None
        for i, a in enumerate(active):
            if a and run_start is None:
                run_start = i
            elif not a and run_start is not None:
                if i - run_start >= min_run:
                    keep[run_start:i] = True
                run_start = None
        if run_start is not None and len(active) - run_start >= min_run:
            keep[run_start:] = True
        idx = np.nonzero(keep)[0]
        return (int(idx[0]), int(idx[-1])) if len(idx) else None

    ce = extent(col, 4)
    re = extent(row, 4)
    if ce is None or re is None:
        return None
    return ce[0], ce[1], re[0], re[1]


def _tighten_by_texture(gray: np.ndarray, edge: np.ndarray, quad: np.ndarray):
    """候选框与背景色块粘连时，用字符边缘的活动范围反推整牌位置。
    字符带约占牌宽 95%，牌高按标准长宽比反推。返回收紧 quad 或 None。"""
    e_top = np.linalg.norm(quad[1] - quad[0])
    e_left = np.linalg.norm(quad[3] - quad[0])
    ratio = max(e_top, e_left) / max(1.0, min(e_top, e_left))
    cw = 240
    ch = max(30, int(round(cw / ratio)))
    dst = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    ecrop = cv2.warpPerspective(edge, m, (cw, ch))
    # 闭运算连通字符笔画，避免行/列剖面稀疏导致字符带范围被低估
    ecrop = cv2.morphologyEx(ecrop, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    ext = _edge_extent(ecrop)
    if ext is None:
        return None
    x0, x1, y0, y1 = ext
    char_w, char_h = x1 - x0 + 1, y1 - y0 + 1
    if char_w > 0.92 * cw and char_h > 0.92 * ch:
        return None  # 边缘已铺满候选框，无需收紧
    # 字符带约占领牌宽 95%；牌高按标准长宽比反推（Sobel 对横笔画不敏感，
    # 直接用字符带高度会低估），并不小于字符带高度/0.55 以防切字
    pw = char_w / 0.95
    ph = max(pw / IDEAL_RATIO, char_h / 0.55)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0 = max(0, cx - pw / 2)
    by0 = max(0, cy - ph / 2)
    bx1 = min(cw - 1, cx + pw / 2)
    by1 = min(ch - 1, cy + ph / 2)
    if (bx1 - bx0) * (by1 - by0) > 0.9 * cw * ch:
        return None
    box = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]],
                   dtype=np.float64)
    inv = np.linalg.inv(m)
    pts = np.hstack([box, np.ones((4, 1))]) @ inv.T
    return order_quad(pts[:, :2] / pts[:, 2:3])


def _tighten_wide_by_chars(gray: np.ndarray, quad: np.ndarray):
    """宽条颜色候选（车牌与同色草丛/背景粘成宽条）专用：
    滑动窗口用连通域字符数定位车牌子区域，再外扩边距覆盖整牌两端字符。
    仅对 ratio>4.6 的宽条调用。返回收紧 quad 或 None。"""
    e_top = np.linalg.norm(quad[1] - quad[0])
    e_left = np.linalg.norm(quad[3] - quad[0])
    ratio = max(e_top, e_left) / max(1.0, min(e_top, e_left))
    cw = 240
    ch = max(30, int(round(cw / ratio)))
    dst = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    gcrop = cv2.warpPerspective(gray, m, (cw, ch))
    best, best_score = None, 0.0
    for s in (0.95, 0.8, 0.65):
        wh = int(ch * s)
        ww = int(wh * 4.0)  # 新能源宽牌长宽比可达 4+
        if ww > cw or wh < 18:
            continue
        area_frac = (ww * wh) / (cw * ch)
        for y in range(0, ch - wh + 1, max(2, ch // 8)):
            for x in range(0, cw - ww + 1, max(2, cw // 20)):
                win = gcrop[y:y + wh, x:x + ww]
                cc = _cc_char_count(win)
                if cc < 5:
                    continue
                sc = math.exp(-((cc - 9.0) / 3.0) ** 2) * (0.7 + 0.3 * area_frac)
                if sc > best_score:
                    best_score, best = sc, (x, y, ww, wh)
    if best is None:
        return None
    x, y, ww, wh = best
    bx0 = max(0, x - ww * 0.10)
    by0 = max(0, y - wh * 0.15)
    bx1 = min(cw - 1, x + ww * 1.12)
    by1 = min(ch - 1, y + wh * 1.15)
    box = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], dtype=np.float64)
    inv = np.linalg.inv(m)
    pts = np.hstack([box, np.ones((4, 1))]) @ inv.T
    return order_quad(pts[:, :2] / pts[:, 2:3])


def _ring_color(masks: dict, quad: np.ndarray) -> str:
    """用候选框外环（牌面底色所在）判定底色线索，避免白色字符干扰。"""
    best, best_r = "unknown", 0.2
    for name, m in masks.items():
        crop = _rot_crop(m, quad, 120, 40)
        ring = np.concatenate([crop[:5].ravel(), crop[-5:].ravel(),
                               crop[:, :5].ravel(), crop[:, -5:].ravel()])
        r = float(np.count_nonzero(ring)) / ring.size
        if name != "white" and r > best_r:
            best, best_r = name, r
        elif name == "white" and r > max(best_r, 0.5):
            best, best_r = name, r
    return best


def _texture_score(gray_crop: np.ndarray) -> float:
    """字符纹理得分 0~1：车牌字符在水平方向产生多次明暗跳变。"""
    g = cv2.resize(gray_crop, (160, 50))
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    col = np.abs(sx).mean(axis=0)
    if col.max() < 1e-3:
        return 0.0
    col = col / col.max()
    active = col > 0.25
    transitions = int(np.count_nonzero(active[1:] != active[:-1]))
    coverage = float(active.mean())
    # 7~8 个字符大约产生 12~20 次跳变；过少=空白/纯色块，过多=栅格噪声
    t_score = 1.0 if 10 <= transitions <= 34 else max(0.0, 1.0 - abs(transitions - 20) / 20.0)
    c_score = min(1.0, coverage / 0.45)
    return 0.7 * t_score + 0.3 * c_score


def _char_band_count(gray_crop: np.ndarray) -> int:
    """估计字符峰数：车牌 7~8 个字符（汉字多笔画会拆出多峰，实测 8~14）。
    广告牌大字、字符碎片(≤4峰)、纯色块(0~1峰) 会被过滤。"""
    g = cv2.resize(gray_crop, (200, 60))
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(bw) > bw.size / 2:
        bw = cv2.bitwise_not(bw)  # 统一为“字符=白”
    col = bw[9:51, 6:194].mean(axis=0) / 255.0  # 去上下边框带、左右边缘
    col = np.convolve(col, np.ones(3) / 3, mode="same")
    peaks = []
    for i in range(1, len(col) - 1):
        if col[i] >= col[i - 1] and col[i] > col[i + 1] and col[i] > 0.30:
            peaks.append(i)
    merged = []
    for p in peaks:
        if merged and p - merged[-1] < 8:
            if col[p] > col[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    return len(merged)


def _cc_char_count(gray_crop: np.ndarray) -> int:
    """连通域字符计数：统计字符形状的连通域个数。
    真实车牌 7~8 个字符（汉字断笔会略多，实测 6~10）；
    背景色块/广告牌大字/字符碎片通常 ≤2，据此区分。"""
    g = cv2.resize(gray_crop, (200, 60))
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(bw) > bw.size / 2:
        bw = cv2.bitwise_not(bw)  # 统一为“字符=白”
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    cnt = 0
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if 0.30 * 60 <= h <= 0.95 * 60 and 3 <= w <= 45 and w < 1.4 * h and a > 40:
            cnt += 1
    return cnt


def _char_layout_ok(gray_crop: np.ndarray) -> bool:
    """字符带高度验证：排除“字符只占中间一小段”的整块广告牌/车身文字。
    真实车牌紧贴裁剪时字符带可铺满牌高（跨度≈1.0），故只设下限不设上限。
    用字符行（前景占比>0.06）的跨度衡量；纯背景或字符占比过低的区域被排除。"""
    g = cv2.resize(gray_crop, (200, 60))
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(bw) > bw.size / 2:
        bw = cv2.bitwise_not(bw)
    row = bw.mean(axis=1) / 255.0
    idx = np.nonzero(row > 0.06)[0]
    if not len(idx):
        return False
    span_frac = (idx[-1] - idx[0] + 1) / len(row)
    return span_frac >= 0.40


def _candidates_from_mask(mask: np.ndarray, close_kw: int, close_kh: int):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kw, close_kh))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.minAreaRect(c) for c in contours
            if cv2.contourArea(c) > 150]


def _iou_quad_bbox(b1, b2) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def detect_plates(image_bgr: np.ndarray, verbose: bool = False) -> list:
    """检测图中所有车牌。返回 list[PlateRegion]，坐标对应原图，无车牌返回 []。
    verbose=True 时打印每个候选的筛选过程（调试用）。"""
    def log(msg):
        if verbose:
            print("   ", msg)
    if image_bgr is None or image_bgr.size == 0:
        return []
    h0, w0 = image_bgr.shape[:2]
    scale = min(1.0, DETECT_LONG_SIDE / max(h0, w0))
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale) if scale < 1.0 else image_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    masks = _color_masks(hsv)
    edge = _edge_mask(gray)

    kw = max(9, int(small.shape[1] * 0.015))
    kh = max(3, int(small.shape[1] * 0.004))
    # 颜色通道用更小的闭运算核：车牌底色本身连通，核太大会把邻近同色背景并入
    ckw, ckh = max(5, int(small.shape[1] * 0.006)), max(3, int(small.shape[1] * 0.003))
    raw = []  # (rect, channel, color_name)
    for name, m in masks.items():
        for r in _candidates_from_mask(m, ckw, ckh):
            raw.append((r, "color", name))
    for r in _candidates_from_mask(edge, kw, kh):
        raw.append((r, "edge", None))

    results = []
    for rect, channel, cname in raw:
        quad, w, h, angle = _rect_geometry(rect)
        tag = f"{channel}/{cname} bbox~({int(quad[:,0].min()/scale)},{int(quad[:,1].min()/scale)},{int(w/scale)},{int(h/scale)})"
        if h < 5:
            continue
        ratio = w / h
        # 颜色通道允许宽条（车牌与同色背景粘连）进入，后续用字符窗口收紧
        upper = 6.5 if channel == "color" else RATIO_INIT[1]
        if not (RATIO_INIT[0] <= ratio <= upper):
            log(f"{tag} 拒绝:长宽比{ratio:.2f}")
            continue
        if abs(angle) > 45:  # 过大倾角暂不支持（报告中说明局限）
            log(f"{tag} 拒绝:倾角{angle:.1f}")
            continue
        ow, oh = w / scale, h / scale
        if ow < MIN_PLATE_W or oh < MIN_PLATE_H:
            log(f"{tag} 拒绝:尺寸{ow:.0f}x{oh:.0f}")
            continue

        # 宽条颜色候选：用字符窗口搜索收紧到车牌子区域
        if channel == "color" and ratio > RATIO_RANGE[1]:
            wq = _tighten_wide_by_chars(gray, quad)
            if wq is None:
                log(f"{tag} 拒绝:宽条无字符窗口")
                continue
            quad, w, h, angle = _quad_norm(wq)
            ratio = w / h
            if not (RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1]):
                log(f"{tag} 拒绝:宽条收紧后长宽比{ratio:.2f}")
                continue

        # 颜色覆盖率（各底色取最大）
        color_ratios = {n: _mask_ratio_in_quad(m, quad) for n, m in masks.items()}
        best_color = max(color_ratios, key=color_ratios.get)
        color_ratio = color_ratios[best_color]

        # 用底色掩码收紧候选框（边缘通道常把背景色块并入，导致框偏大）。
        # 新能源牌左黄右绿，只按面积最大的一种颜色收紧会切掉省份简称。
        fill = 1.0
        plate_mask = None
        for name, mask in masks.items():
            if name == "white" or color_ratios[name] < 0.08:
                continue
            plate_mask = mask if plate_mask is None else cv2.bitwise_or(plate_mask, mask)
        if color_ratio > 0.30 and plate_mask is not None:
            refined = _refine_quad(plate_mask, quad)
            if refined is not None:
                rq, fill = refined
                rq, e_top, e_left, a2 = _quad_norm(rq)
                if min(e_top, e_left) >= 5:
                    r2 = max(e_top, e_left) / min(e_top, e_left)
                    if (RATIO_RANGE[0] <= r2 <= RATIO_RANGE[1]
                            and e_top / scale >= MIN_PLATE_W
                            and e_left / scale >= MIN_PLATE_H
                            and cv2.contourArea(rq.astype(np.float32))
                            >= 0.35 * cv2.contourArea(quad.astype(np.float32))):
                        quad, w, h, ratio, angle = rq, e_top, e_left, r2, a2

        # 掩码填充率低（L形粘连）或候选过宽（实心同色背景并入）→ 用字符边缘范围反推整牌
        if fill < 0.82 or ratio > 4.2:
            tq = _tighten_by_texture(gray, edge, quad)
            if tq is not None:
                tq, e_top, e_left, a2 = _quad_norm(tq)
                if min(e_top, e_left) >= 5:
                    r2 = max(e_top, e_left) / min(e_top, e_left)
                    if (RATIO_RANGE[0] <= r2 <= RATIO_RANGE[1]
                            and e_top / scale >= MIN_PLATE_W
                            and e_left / scale >= MIN_PLATE_H):
                        quad, w, h, ratio, angle = tq, e_top, e_left, r2, a2

        # 收紧后按严格长宽比/尺寸/倾角验收（初筛宽松，此处把关）
        if not (RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1]):
            log(f"{tag} 拒绝:收紧后长宽比{ratio:.2f}")
            continue
        if w / scale < MIN_PLATE_W or h / scale < MIN_PLATE_H or abs(angle) > 45:
            log(f"{tag} 拒绝:收紧后尺寸/倾角")
            continue

        # 收紧后重新统计颜色；底色线索用外环判定（白色字符不算底色）
        color_ratios = {n: _mask_ratio_in_quad(m, quad) for n, m in masks.items()}
        best_color = max(color_ratios, key=color_ratios.get)
        color_ratio = color_ratios[best_color]
        hint = _ring_color(masks, quad)

        # 底色覆盖率门槛：真实车牌必有明确底色，无明确牌色的多为背景纹理
        if color_ratio < MIN_COLOR_RATIO or hint == "unknown":
            log(f"{tag} 拒绝:底色覆盖{color_ratio:.2f}/hint={hint}")
            continue
        # 白牌候选须有黑色字符结构，排除白色引擎盖/天空/保险杠等纯色背景
        if hint == "white":
            gcrop = _rot_crop(gray, quad, 120, 40)
            dark_frac = float(np.count_nonzero(gcrop < 100)) / gcrop.size
            if not (0.06 <= dark_frac <= 0.55):
                log(f"{tag} 拒绝:白底暗字占比{dark_frac:.2f}不符")
                continue

        crop = _rot_crop(gray, quad, 160, 50)
        cc = _cc_char_count(crop)
        if cc < 3:
            log(f"{tag} 拒绝:连通域字符{cc}<3 (hint={hint})")
            continue  # 字符结构不足：背景色块/广告牌大字/字符碎片
        if hint == "white" and cc < 5:
            log(f"{tag} 拒绝:白牌字符{cc}<5 (hint={hint})")
            continue  # 白牌须有清晰多字符，排除白色背景弱纹理
        bands = _char_band_count(crop)
        if not (4 <= bands <= 18):
            log(f"{tag} 拒绝:字符峰{bands} (cc={cc},hint={hint})")
            continue
        if not _char_layout_ok(crop):
            log(f"{tag} 拒绝:字符布局 (cc={cc},hint={hint})")
            continue
        texture = _texture_score(crop)
        edge_density = float(np.count_nonzero(
            _rot_crop(edge, quad, 120, 40))) / (120 * 40)
        # 单行牌长宽比约 3.1，双行大车牌约 2.0。只拿 3.2 当理想值时，双行牌会被打成接近 0。
        ideal_ratio = 2.0 if ratio < 2.4 else IDEAL_RATIO
        ratio_score = math.exp(-((ratio - ideal_ratio) / RATIO_SIGMA) ** 2)
        # 字符数得分：7~8 字符最理想，连通域计数接近该值加分
        cc_score = math.exp(-((cc - 7.5) / 3.0) ** 2)

        if channel == "color":
            # 颜色通道：底色覆盖率、长宽比、字符结构是车牌的最强证据
            score = (0.35 * min(1.0, color_ratio / 0.55)
                     + 0.15 * texture
                     + 0.25 * ratio_score
                     + 0.15 * cc_score
                     + 0.10 * min(1.0, edge_density / 0.35))
        else:
            # 边缘通道：必须有颜色或纹理支持，否则容易把广告牌/散热格栅当车牌
            support = max(min(1.0, color_ratio / 0.45), texture)
            score = (0.25 * support
                     + 0.15 * texture
                     + 0.30 * ratio_score
                     + 0.15 * cc_score
                     + 0.15 * min(1.0, edge_density / 0.35))
            if support < 0.40:
                log(f"{tag} 拒绝:边缘支持{support:.2f}")
                continue

        if score < SCORE_THRESH:
            log(f"{tag} 拒绝:得分{score:.2f} (bands={bands},hint={hint})")
            continue
        log(f"{tag} 接受:得分{score:.2f} bands={bands} hint={hint} ratio={ratio:.2f}")

        # 换算回原图坐标
        quad0 = quad / scale
        x, y = quad0[:, 0].min(), quad0[:, 1].min()
        bw, bh = quad0[:, 0].max() - x, quad0[:, 1].max() - y
        x = max(0, int(math.floor(x)))
        y = max(0, int(math.floor(y)))
        bw = min(w0 - x, int(math.ceil(bw)))
        bh = min(h0 - y, int(math.ceil(bh)))
        # 横跨整图且很矮的色带才是草地或保险杠。近景整牌本身就会占满画面宽度。
        if bw > FULL_WIDTH_FRAC * w0 and bh < 0.25 * h0:
            log(f"{tag} 拒绝:全图宽色带 bw={bw}/{w0} bh={bh}/{h0}")
            continue
        results.append((PlateRegion(bbox=(x, y, bw, bh),
                                    quad=quad0.tolist(),
                                    angle=angle, score=score, color_hint=hint),
                        channel, cc))

    # NMS 去重 + 双向包含关系去重（碎片框可能得分更高先入列）
    # 重叠时优先保留颜色通道的几何（底色掩码给出整牌边界，边缘通道常只覆盖字符区）
    results.sort(key=lambda t: t[0].score, reverse=True)
    kept = []
    for r, rch, rcc in results:
        dup = False
        for j, (k, kch, kcc) in enumerate(kept):
            if _iou_quad_bbox(r.bbox, k.bbox) > NMS_IOU:
                rw, rh = r.bbox[2], r.bbox[3]
                kw, kh = k.bbox[2], k.bbox[3]
                same_row = abs(rh - kh) < 0.35 * max(rh, kh, 1)
                # 新能源牌的绿色框会切掉左侧黄底省份字。同一行里更宽、分数接近的框是整牌。
                if same_row and rw > kw * 1.2 and r.score >= k.score * 0.75:
                    r.score = max(r.score, k.score)
                    kept[j] = (r, rch, rcc)
                elif same_row and kw > rw * 1.2 and k.score >= r.score * 0.75:
                    pass
                elif rch == "color" and kch == "edge":
                    r.score = max(r.score, k.score)
                    kept[j] = (r, rch, rcc)  # 用颜色通道的完整框替换边缘框
                dup = True
                break
            cx, cy = r.bbox[0] + r.bbox[2] / 2, r.bbox[1] + r.bbox[3] / 2
            kx, ky, kw2, kh2 = k.bbox
            ra, ka = r.bbox[2] * r.bbox[3], kw2 * kh2
            if kx <= cx <= kx + kw2 and ky <= cy <= ky + kh2 and ra < 0.6 * ka:
                dup = True  # r 是已保留大框内的碎片
                break
            kcx, kcy = kx + kw2 / 2, ky + kh2 / 2
            rx, ry, rw2, rh2 = r.bbox
            if rx <= kcx <= rx + rw2 and ry <= kcy <= ry + rh2 and ka < 0.6 * ra:
                kept[j] = (r, rch, rcc)  # 已保留的是碎片，用更大的完整框替换
                dup = True
                break
        if not dup:
            kept.append((r, rch, rcc))
        if len(kept) >= MAX_PLATES:
            break

    # 抑制相对主牌极小且字符少的碎片（如车身小贴纸），保留正常多车牌
    if len(kept) > 1:
        max_area = max(r.bbox[2] * r.bbox[3] for r, _, _ in kept)
        kept = [(r, ch, cc) for r, ch, cc in kept
                if not (cc < 6 and r.bbox[2] * r.bbox[3] < 0.2 * max_area)]
    return [r for r, _, _ in kept]


def draw_regions(image_bgr: np.ndarray, regions: list) -> np.ndarray:
    """在原图副本上画出检测框、序号、得分（供 GUI/报告/视频使用）。"""
    vis = image_bgr.copy()
    for i, r in enumerate(regions, 1):
        pts = np.array(r.quad, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        x, y, w, h = r.bbox
        label = f"#{i} {r.color_hint} {r.score:.2f}"
        ty = max(20, y - 6)
        cv2.putText(vis, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description="A 模块：车牌检测定位独立示例")
    ap.add_argument("image", help="输入图片路径（支持中文路径）")
    ap.add_argument("--out", default="results", help="输出目录，默认 results/")
    args = ap.parse_args()

    img = load_image(args.image)
    regions = detect_plates(img)
    print(regions_to_json(regions, {"image": args.image}))
    save_image(f"{args.out}/detected.png", draw_regions(img, regions))
    print(f"标框图已保存: {args.out}/detected.png")


if __name__ == "__main__":
    main()
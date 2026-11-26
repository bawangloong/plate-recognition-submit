# -*- coding: utf-8 -*-
"""
rectify_plate.py —— A 模块：车牌倾斜矫正与裁剪
================================================
输入：原图 + detect_plates 的定位结果。
输出：方向摆正、字符完整、去掉外框的车牌图（BGR uint8，直接交给 B），
      以及 3x3 单应矩阵 transform（原图坐标 → 矫正图坐标）。

流程：
  1) 按检测 quad 外扩 3% 做透视变换，摆正到标准朝向（防切字）；
  2) 用字符连通域拟合残余倾角，做小角度旋转微调；
  3) 按底色一致性裁掉外框/铆钉边条；
  4) 三步变换复合为 transform，保留与原图的对应关系。

对外接口（D 集成时只调用这一个函数）：
    rectify_plates(image_bgr, regions) -> list[RectifiedPlate]

独立运行：
    python rectify_plate.py 图片路径 [--out 输出目录]
"""

import argparse
import math

import cv2
import numpy as np

from plate_common import (PlateRegion, RectifiedPlate, load_image, save_image)
from detect_plate import detect_plates, draw_regions

# ---------------- 可调参数 ----------------
EXPAND = 1.12        # quad 外扩系数，防止边缘字符被切
TARGET_W_MIN = 160   # 矫正图最小宽度（小车牌放大，便于 B 分割）
TARGET_W_MAX = 480   # 矫正图最大宽度（控制计算量）
FINE_ANGLE_MIN = 0.6 # 残余倾角小于该值不再微调（度）
FINE_ANGLE_MAX = 15  # 残余倾角大于该值认为拟合不可靠，放弃微调
TRIM_FRAC = 0.88     # 行/列底色占比超过该值视为边框，可裁掉
TRIM_LIMIT = 0.15    # 每侧最多裁掉的比例（防误裁字符）


def _expand_quad(quad: np.ndarray, factor: float) -> np.ndarray:
    c = quad.mean(axis=0)
    return (quad - c) * factor + c


def _fine_angle(crop_bgr: np.ndarray) -> float:
    """用字符连通域的 minAreaRect 中值估计残余倾角。失败返回 0。"""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bw.shape
    # 字符可能是亮底暗字或暗底亮字：取占比较少的那种极性
    if np.count_nonzero(bw) > bw.size / 2:
        bw = cv2.bitwise_not(bw)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    angles = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < 30 or not (0.25 < ch / h < 0.95) or cw > ch * 1.5:
            continue  # 只要“字符形状”的连通域
        ys, xs = np.nonzero(labels == i)
        rect = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
        (rw, rh), a = rect[1], rect[2]
        if rw < rh:      # 统一为竖直字符的角度
            a += 90
        a = ((a + 45) % 90) - 45
        if abs(a) <= FINE_ANGLE_MAX:
            angles.append(a)
    if len(angles) < 3:
        return 0.0
    return float(np.median(angles))


def _rotate(img: np.ndarray, angle_deg: float):
    """绕中心旋转（保持尺寸），返回 (旋转图, 3x3 矩阵)。"""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    m3 = np.vstack([m, [0, 0, 1]])
    return out, m3


def _trim_border(crop_bgr: np.ndarray):
    """按底色一致性裁掉四周外框。返回 (裁剪图, 3x3 平移矩阵)。"""
    h, w = crop_bgr.shape[:2]
    ring = np.concatenate([
        crop_bgr[:max(2, h // 20)].reshape(-1, 3),
        crop_bgr[-max(2, h // 20):].reshape(-1, 3),
        crop_bgr[:, :max(2, w // 30)].reshape(-1, 3),
        crop_bgr[:, -max(2, w // 30):].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    dist = np.linalg.norm(crop_bgr.astype(np.float32) - bg, axis=2)
    is_bg = dist < 60  # 与底色接近的像素
    row_frac = is_bg.mean(axis=1)
    col_frac = is_bg.mean(axis=0)

    def span(frac, limit):
        n = len(frac)
        lo, hi = 0, n
        while lo < limit and frac[lo] > TRIM_FRAC:
            lo += 1
        while hi > n - limit and frac[hi - 1] > TRIM_FRAC:
            hi -= 1
        return lo, hi

    y0, y1 = span(row_frac, int(h * TRIM_LIMIT))
    x0, x1 = span(col_frac, int(w * TRIM_LIMIT))
    if y1 - y0 < h * 0.5 or x1 - x0 < w * 0.5:
        return crop_bgr, np.eye(3)  # 裁太多说明判断失误，放弃
    out = crop_bgr[y0:y1, x0:x1]
    m = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    return out, m


def rectify_plate(image_bgr: np.ndarray, region: PlateRegion) -> RectifiedPlate:
    """矫正单块车牌。region 来自 detect_plates。"""
    quad = _expand_quad(np.array(region.quad, dtype=np.float32), EXPAND)
    e_top = np.linalg.norm(quad[1] - quad[0])
    e_left = np.linalg.norm(quad[3] - quad[0])
    ratio = max(e_top, e_left) / max(1.0, min(e_top, e_left))
    out_w = int(np.clip(round(max(e_top, e_left)), TARGET_W_MIN, TARGET_W_MAX))
    out_h = max(24, int(round(out_w / ratio)))

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    m_warp = cv2.getPerspectiveTransform(quad, dst)
    crop = cv2.warpPerspective(image_bgr, m_warp, (out_w, out_h),
                               borderMode=cv2.BORDER_REPLICATE)
    m_total = m_warp.astype(np.float64)

    # 残余倾角微调
    a = _fine_angle(crop)
    if FINE_ANGLE_MIN < abs(a) <= FINE_ANGLE_MAX:
        crop, m_rot = _rotate(crop, a)
        m_total = m_rot @ m_total

    # 去外框
    crop, m_trim = _trim_border(crop)
    m_total = m_trim @ m_total

    return RectifiedPlate(region=region, image=crop, transform=m_total)


def rectify_plates(image_bgr: np.ndarray, regions: list) -> list:
    """批量矫正。返回 list[RectifiedPlate]，与 regions 一一对应。"""
    return [rectify_plate(image_bgr, r) for r in regions]


def main():
    ap = argparse.ArgumentParser(description="A 模块：车牌矫正裁剪独立示例")
    ap.add_argument("image", help="输入图片路径")
    ap.add_argument("--out", default="results", help="输出目录，默认 results/")
    args = ap.parse_args()

    img = load_image(args.image)
    regions = detect_plates(img)
    if not regions:
        print("未检测到车牌。")
        return
    save_image(f"{args.out}/detected.png", draw_regions(img, regions))
    plates = rectify_plates(img, regions)
    for i, p in enumerate(plates, 1):
        path = f"{args.out}/plate_{i}.png"
        save_image(path, p.image)
        print(f"车牌 #{i}: {p.size[0]}x{p.size[1]}  已保存 {path}")


if __name__ == "__main__":
    main()
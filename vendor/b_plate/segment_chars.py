# -*- coding: utf-8 -*-
"""
segment_chars.py —— 车牌字符分割

处理流程：
    车牌图 -> 尺寸归一化 -> 灰度 -> Otsu 二值化（自动判断极性）
          -> 去边框/铆钉/噪点 -> 形态学闭运算补断笔
          -> 垂直投影切分 -> 合并碎块/拆分粘连 -> 过滤小圆点
          -> 输出按从左到右排序的字符图 + 位置

输入：单块车牌彩色图（BGR 或 RGB）
输出：dict(chars=[图], boxes=[(x,y,w,h)], count=n, status='ok'/'count_abnormal'/'no_char')

说明：全部用 numpy + Pillow 实现，不依赖 OpenCV；函数名与 cv2 方案一一对应，
     若要换回 cv2，只需替换 otsu / morph / connected_components 三个函数。
"""

import numpy as np
from PIL import Image, ImageFilter

CHAR_W, CHAR_H = 20, 40          # 输出字符图的标准尺寸（给 C 同学使用）
STD_H = 140                      # 车牌归一化高度
MIN_FG_RATIO, MAX_FG_RATIO = 0.03, 0.45


# ---------------------------------------------------------------- 基础工具
def to_gray(img):
    """RGB -> 灰度（0-255 float -> uint8）"""
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return arr.astype(np.uint8)


def otsu_threshold(gray):
    """Otsu 自适应阈值，返回阈值 t（前景=gray>t 或 gray<=t，由调用方决定极性）"""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    w_b = np.cumsum(hist)
    sum_b = np.cumsum(hist * np.arange(256))
    w_f = total - w_b
    valid = (w_b > 0) & (w_f > 0)
    mean_b = np.divide(sum_b, w_b, out=np.zeros_like(sum_b), where=w_b > 0)
    mean_f = np.divide(sum_all - sum_b, w_f, out=np.zeros_like(sum_b), where=w_f > 0)
    var = np.zeros(256)
    var[valid] = w_b[valid] * w_f[valid] * (mean_b[valid] - mean_f[valid]) ** 2
    return int(np.argmax(var))


def binarize(gray, prefer_dark_text=None):
    """
    二值化并自动确定极性。
    prefer_dark_text: True=字符是深色；False=字符是浅色；None=按前景像素占比自动判断。
    返回 (bw, polarity)，bw 中 True 表示"字符像素"。
    """
    t = otsu_threshold(gray)
    dark = gray <= t                 # 深色像素
    ratio_dark = dark.mean()

    if prefer_dark_text is None:
        prefer_dark_text = ratio_dark <= MAX_FG_RATIO

    if prefer_dark_text:
        bw = dark
        if not (MIN_FG_RATIO <= bw.mean() <= MAX_FG_RATIO):
            bw = ~dark
    else:
        bw = ~dark
        if not (MIN_FG_RATIO <= bw.mean() <= MAX_FG_RATIO):
            bw = dark

    return bw, bool(prefer_dark_text)


def morph(bw, size=3, op="close"):
    """形态学；用 PIL 的 Min/Max 滤波器实现（等价于 cv2.erode / cv2.dilate 的方核）"""
    im = Image.fromarray((bw * 255).astype(np.uint8), mode="L")
    if op == "close":
        im = im.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    elif op == "open":
        im = im.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))
    elif op == "dilate":
        im = im.filter(ImageFilter.MaxFilter(size))
    elif op == "erode":
        im = im.filter(ImageFilter.MinFilter(size))
    return np.array(im) > 127


def connected_components(bw):
    """
    两遍扫描 + 并查集的连通域标记。
    返回 (labels, n)，labels 中 0 表示背景，1..n 为连通域编号。
    """
    h, w = bw.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for y in range(h):
        row = bw[y]
        for x in range(w):
            if not row[x]:
                continue
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0
            if up and left:
                labels[y, x] = min(up, left)
                union(up, left)
            elif up or left:
                labels[y, x] = up or left
            else:
                labels[y, x] = nxt
                parent.append(nxt)
                nxt += 1

    for y in range(h):
        for x in range(w):
            if labels[y, x]:
                labels[y, x] = find(labels[y, x])
    uniq = np.unique(labels)
    uniq = uniq[uniq > 0]
    remap = {int(v): i + 1 for i, v in enumerate(uniq)}
    for y in range(h):
        for x in range(w):
            if labels[y, x]:
                labels[y, x] = remap[int(labels[y, x])]
    return labels, len(uniq)


def component_boxes(labels, n):
    """连通域列表 [(id, x, y, w, h, area)]"""
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        out.append((i, int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1), int(len(xs))))
    return out


# ---------------------------------------------------------------- 主流程
def _normalize_size(img):
    """按高度归一化，保持长宽比（过高或过扁的车牌会被拉回合理范围）"""
    h, w = img.shape[:2]
    scale = STD_H / float(h)
    nw = max(1, int(round(w * scale)))
    pil = Image.fromarray(img.astype(np.uint8))
    return np.array(pil.resize((nw, STD_H), Image.LANCZOS))


def _clean(bw):
    """去除接触边框的连通域、铆钉与小噪点"""
    h, w = bw.shape
    labels, n = connected_components(bw)
    keep = np.zeros_like(bw)
    for (cid, x, y, ww, hh, area) in component_boxes(labels, n):
        # 贴边：整块被裁掉（边框）
        if x <= 1 or y <= 1 or x + ww >= w - 1 or y + hh >= h - 1:
            continue
        # 噪点
        if area < 0.0008 * h * w:
            continue
        # 铆钉：矮且位于上下边缘
        if hh < 0.25 * h and (y < 0.10 * h or y + hh > 0.90 * h):
            continue
        keep |= (labels == cid)
    return keep


def _project_segments(bw, min_gap=2):
    """垂直投影切分：返回 [(x0, x1)] 列区间"""
    col = bw.sum(axis=0)
    on = col > max(1, 0.02 * bw.shape[0])
    segs, start = [], None
    gap = 0
    for x, v in enumerate(on):
        if v:
            if start is None:
                start = x
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap >= min_gap:
                    segs.append((start, x - gap))
                    start = None
    if start is not None:
        segs.append((start, len(on) - 1))
    return segs


def _merge_and_split(segs, bw, target_w):
    """
    合并"笔画断裂"造成的相邻碎块（间距 ≤2px 且合并后不过宽），
    并拆分"字符粘连"造成的过宽块（宽度 > 1.45 标准宽）。
    """
    # 合并：仅当间隙极小，避免把小圆点、细窄字符并进前一个字符
    merged = []
    for (a, b) in segs:
        if merged:
            gap = a - merged[-1][1] - 1
            w_comb = b - merged[-1][0] + 1
            if gap <= 2 and w_comb <= 1.30 * target_w:
                merged[-1] = (merged[-1][0], b)
                continue
        merged.append((a, b))

    # 拆分：过宽块按标准宽等分，切点取投影局部最小处
    out = []
    for (x0, x1) in merged:
        w = x1 - x0 + 1
        if w > 1.45 * target_w:
            k = int(round(w / target_w))
            col = bw[:, x0:x1 + 1].sum(axis=0)
            cuts = [x0]
            for j in range(1, k):
                lo = x0 + int(w * j / k * 0.75)
                hi = x0 + int(w * j / k * 1.25)
                lo, hi = max(lo, x0 + 1), min(hi, x1)
                if hi <= lo:
                    cuts.append(x0 + int(w * j / k))
                else:
                    cuts.append(lo + int(np.argmin(col[lo - x0:hi - x0])))
            cuts.append(x1)
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b - a > 2:
                    out.append((a, b))
        else:
            out.append((x0, x1))
    return out


def segment_plate(img, assume_bgr=False, prefer_dark_text=None, debug=False,
                  min_char_width_ratio=0.25):
    """
    分割单块车牌。

    参数
    ----
    img : 彩色车牌图（numpy 数组或 PIL.Image）
    assume_bgr : 输入是否为 BGR 顺序
    prefer_dark_text : 是否深色字符；None 表示自动判断
    debug : 为 True 时返回中间过程图
    min_char_width_ratio : 字符最小宽度 / 标准字符宽。默认 0.25 与 B 原实现一致；
        D 集成时传 0.10，避免数字“1”被当成噪点过滤掉（见 D 部分报告）。

    返回 dict：
        chars   : [numpy 数组]，按从左到右排序，已归一化到 CHAR_H x CHAR_W
        boxes   : [(x, y, w, h)]，原车牌坐标系
        count   : 字符个数
        status  : 'ok' / 'count_abnormal' / 'no_char'
        stages  : 中间过程图（debug=True 时）
    """
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if assume_bgr:
        img = img[..., ::-1]

    norm = _normalize_size(img)
    gray = to_gray(norm)

    bw, polarity = binarize(gray, prefer_dark_text)
    bw = morph(bw, 3, "close")
    bw = _clean(bw)

    h, w = bw.shape
    target_w = 0.55 * h                      # 车牌字符宽约为车高的 0.5~0.6
    segs = _project_segments(bw)
    segs = _merge_and_split(segs, bw, target_w)

    # 过滤：高度太小（小圆点、噪点）或过窄
    boxes = []
    for (x0, x1) in segs:
        sub = bw[:, x0:x1 + 1]
        ys, xs = np.where(sub)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        hh = y1 - y0 + 1
        ww = x1 - x0 + 1
        if hh < 0.32 * h:                    # 字符高一般在车高的 40%~70%
            continue
        if ww < max(3.0, min_char_width_ratio * target_w):  # 过窄（小圆点、噪点、断笔）
            continue
        boxes.append((int(x0), int(y0), int(ww), int(hh)))

    boxes.sort(key=lambda b: b[0])

    chars = []
    for (x, y, ww, hh) in boxes:
        crop = bw[y:y + hh, x:x + ww]
        chars.append(_norm_char(crop))

    count = len(chars)
    if count == 0:
        status = "no_char"
    elif count in (7, 8):
        status = "ok"
    else:
        status = "count_abnormal"

    result = {"chars": chars, "boxes": boxes, "count": count, "status": status}
    if debug:
        result["stages"] = {"norm": norm, "gray": gray, "bw": bw,
                            "segs": segs, "polarity": polarity}
    return result


def _norm_char(crop):
    """字符图归一化到固定尺寸（保持长宽比，居中补边）"""
    h, w = crop.shape
    if h == 0 or w == 0:
        return np.zeros((CHAR_H, CHAR_W), dtype=np.uint8)
    scale = min((CHAR_H - 4) / h, (CHAR_W - 4) / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    im = Image.fromarray((crop * 255).astype(np.uint8), mode="L")
    im = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (CHAR_W, CHAR_H), 0)
    canvas.paste(im, ((CHAR_W - nw) // 2, (CHAR_H - nh) // 2))
    return np.array(canvas)


if __name__ == "__main__":
    import sys
    from PIL import Image as _I

    if len(sys.argv) < 2:
        print("用法: python segment_chars.py <车牌图片>")
        sys.exit(1)
    for p in sys.argv[1:]:
        arr = np.array(_I.open(p).convert("RGB"))
        r = segment_plate(arr, debug=True)
        print("%-28s 字符数=%d  状态=%s  极性=%s"
              % (p.split("/")[-1], r["count"], r["status"],
                 "深色字" if r["stages"]["polarity"] else "浅色字"))
        print("    boxes:", r["boxes"])

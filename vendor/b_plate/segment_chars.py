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


def _peel_edge_bridge(bw):
    """去掉贴着上下边、只有两三行高的横线。

    黄牌顶上的细线会把左右两个字母粘成一块，整块又因为贴边被删掉。
    字符本身从顶上一直铺到牌高中部，前几行里没有这种凹下去的空隙，这里不动它。
    """
    h, w = bw.shape
    if h < 24 or w < 16:
        return bw
    out = bw.copy()
    cap = max(3, int(round(h * 0.04)))

    for reverse in (False, True):
        fills = out.mean(axis=1)
        if reverse:
            fills = fills[::-1]
        saw = False
        valley = None
        for i in range(cap):
            if fills[i] > 0.15:
                saw = True
            if saw and fills[i] < 0.12:
                valley = i
                break
        if valley is None or valley == 0:
            continue
        if reverse:
            out[h - valley:] = False
        else:
            out[:valley] = False
    return out


def _clean(bw):
    """去除接触边框的连通域、铆钉与小噪点"""
    h, w = bw.shape
    labels, n = connected_components(bw)
    keep = np.zeros_like(bw)
    for (cid, x, y, ww, hh, area) in component_boxes(labels, n):
        # 贴上下边的是边框。只贴左右边、又够高的，多半是被裁到边缘的字符，保留。
        # 裁得很紧时，字符头顶会碰到上边。这种块又窄又高，不是横框，留下。
        touches_tb = y <= 1 or y + hh >= h - 1
        touches_lr = x <= 1 or x + ww >= w - 1
        if touches_tb:
            character_like = (
                not touches_lr
                and hh > 0.42 * h
                and max(3.0, 0.03 * w) <= ww <= 0.22 * w
                and area > 0.004 * h * w
            )
            if not character_like:
                continue
        if touches_lr and not (hh > 0.40 * h and ww >= max(3.0, 0.04 * w)):
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
            # 间隙为 2 像素时多半是两个字挨着，不是断笔。断笔留给后面的碎片合并。
            if gap <= 1 and w_comb <= 1.30 * target_w:
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


def _local_dark(gray):
    """按水平位置分别取阈值。绿牌从左到右变暗时，整张图一个阈值会吃掉中间的字。"""
    height, width = gray.shape
    window = max(15, int(round(width * 0.16)))
    if window % 2 == 0:
        window += 1
    acc = np.zeros(width, dtype=np.float64)
    weight = np.zeros(width, dtype=np.float64)
    step = max(1, window // 4)
    for x in range(0, width, step):
        x0 = max(0, x - window // 2)
        x1 = min(width, x0 + window)
        if x1 - x0 < 8:
            continue
        acc[x0:x1] += otsu_threshold(gray[:, x0:x1])
        weight[x0:x1] += 1
    if weight.max() <= 0:
        return gray <= otsu_threshold(gray)
    threshold = acc / np.maximum(weight, 1.0)
    return gray <= threshold[None, :]


def _split_wide_boxes(bw, boxes):
    """一个投影框里若并排躺着好几个字，按连通域拆开。

    倾斜牌的相邻字在垂直投影上会叠在一起。汉字的左右部首也是多个连通域，
    但它们叠得很深，并且合起来并不比旁边的字宽，这里不拆。
    """
    if not boxes:
        return boxes
    height, width = bw.shape
    labels, n = connected_components(bw)
    pieces = []
    for _cid, x, y, ww, hh, area in component_boxes(labels, n):
        if hh < 0.35 * height or ww > 0.40 * width or ww < 3:
            continue
        if area < 0.001 * height * width:
            continue
        pieces.append((x, y, ww, hh))
    if len(pieces) < 2:
        return boxes
    widths = sorted(b[2] for b in boxes)
    median = widths[len(widths) // 2]
    if len(boxes) == 1:
        median = max(median * 0.5, 0.42 * boxes[0][3])
    if median < 4:
        return boxes
    out = []
    for x, y, bw_, bh in boxes:
        if bw_ <= 1.65 * median:
            out.append((x, y, bw_, bh))
            continue
        inside = []
        for cx, cy, cw, ch in pieces:
            overlap_x = min(x + bw_, cx + cw) - max(x, cx)
            overlap_y = min(y + bh, cy + ch) - max(y, cy)
            if overlap_x > 0.55 * cw and overlap_y > 0.55 * ch:
                inside.append((cx, cy, cw, ch))
        inside.sort(key=lambda item: item[0])
        merged = []
        for cc in inside:
            if not merged:
                merged.append(list(cc))
                continue
            px, py, pw, ph = merged[-1]
            overlap = min(px + pw, cc[0] + cc[2]) - max(px, cc[0])
            if overlap > 0.45 * min(pw, cc[2]):
                nx = min(px, cc[0])
                ny = min(py, cc[1])
                right = max(px + pw, cc[0] + cc[2])
                bottom = max(py + ph, cc[1] + cc[3])
                merged[-1] = [nx, ny, right - nx, bottom - ny]
            else:
                merged.append(list(cc))
        if len(merged) >= 2:
            merged.sort(key=lambda item: item[0])
            out.extend(tuple(int(v) for v in item) for item in merged)
        else:
            out.append((x, y, bw_, bh))
    return out


def _boxes_from_projection(bw, min_char_width_ratio):
    """对一块二值图做垂直投影，返回该图坐标系里的字符框。"""
    h, w = bw.shape
    if h < 8 or w < 8:
        return []
    target_w = 0.55 * h                      # 车牌字符宽约为这一行高度的 0.5~0.6
    segs = _project_segments(bw)
    segs = _merge_and_split(segs, bw, target_w)
    boxes = []
    for (x0, x1) in segs:
        sub = bw[:, x0:x1 + 1]
        ys, xs = np.where(sub)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        hh = y1 - y0 + 1
        ww = x1 - x0 + 1
        if hh < 0.32 * h:
            continue
        if ww < max(3.0, min_char_width_ratio * target_w):
            continue
        boxes.append((int(x0), int(y0), int(ww), int(hh)))
    boxes.sort(key=lambda b: b[0])
    return boxes


def _merge_stroke_fragments(boxes):
    """同一行里贴在一起的窄笔画收成一个字。

    汉字左右部首、字母的断开笔画在垂直投影里会变成两块。
    相邻完整字符之间的空隙更大，这里不合进去。
    """
    if len(boxes) < 2:
        return boxes
    ordered = sorted(boxes, key=lambda b: (b[1] + b[3] / 2.0, b[0]))
    rows = []
    for box in ordered:
        cy = box[1] + box[3] / 2.0
        placed = False
        for row in rows:
            centers = [item[1] + item[3] / 2.0 for item in row]
            height = max(item[3] for item in row)
            if abs(cy - sum(centers) / len(centers)) <= 0.45 * height:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    merged = []
    for row in rows:
        row.sort(key=lambda b: b[0])
        cur = list(row[0])
        for x, y, w, h in row[1:]:
            px, py, pw, ph = cur
            gap = x - (px + pw)
            overlap = min(py + ph, y + h) - max(py, y)
            if overlap < 0.45 * min(ph, h):
                merged.append(tuple(int(v) for v in cur))
                cur = [x, y, w, h]
                continue
            narrow = pw < 0.75 * ph or w < 0.75 * h
            union_w = x + w - px
            union_h = max(py + ph, y + h) - min(py, y)
            # 只收已经叠在一起的断笔。隔开两个像素的是两个字。
            touch = gap <= 0
            if narrow and touch and union_w <= 1.45 * max(union_h, 1):
                cur = [px, min(py, y), union_w, union_h]
            else:
                merged.append(tuple(int(v) for v in cur))
                cur = [x, y, w, h]
        merged.append(tuple(int(v) for v in cur))
    return merged


def _tighten_single_row(boxes):
    """单行牌上，个别框被铆钉或边框拉高时，收回同一条字带。"""
    if len(boxes) < 4:
        return boxes
    top = max(b[1] for b in boxes)
    bot = min(b[1] + b[3] for b in boxes)
    height = max(b[3] for b in boxes)
    if bot - top <= 0.5 * height:
        return boxes
    tops = sorted(b[1] for b in boxes)
    bots = sorted(b[1] + b[3] for b in boxes)
    med_top = tops[len(tops) // 2]
    med_bot = bots[len(bots) // 2]
    band = max(8, med_bot - med_top)
    tightened = []
    for x, y, w, h in boxes:
        y1 = y + h
        if y < med_top - 0.12 * band:
            y = med_top
        if y1 > med_bot + 0.12 * band:
            y1 = med_bot
        if y1 - y > 4:
            tightened.append((x, y, w, y1 - y))
    return tightened or boxes


def _two_row_cut(bw):
    """双行牌在中部有一条更空的横带。单行牌长宽比更大，不在这里切开。"""
    h, w = bw.shape
    if h < 24 or w / float(h) >= 2.45:
        return None
    row = bw.mean(axis=1)
    k = max(3, h // 20)
    if k % 2 == 0:
        k += 1
    smooth = np.convolve(row, np.ones(k) / k, mode="same")
    lo, hi = int(h * 0.34), int(h * 0.66)
    if hi <= lo + 2:
        return None
    cut = lo + int(np.argmin(smooth[lo:hi]))
    top_peak = float(smooth[int(h * 0.12):int(h * 0.34)].max())
    bot_peak = float(smooth[int(h * 0.66):int(h * 0.88)].max())
    if min(top_peak, bot_peak) < 0.04:
        return None
    if smooth[cut] > 0.55 * min(top_peak, bot_peak):
        return None
    return int(cut)


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

    def boxes_of(binary):
        found = _boxes_from_projection(binary, min_char_width_ratio)
        cut = _two_row_cut(binary)
        if cut is not None:
            top = _boxes_from_projection(binary[:cut], min_char_width_ratio)
            bottom = [
                (x, y + cut, w, h)
                for (x, y, w, h) in _boxes_from_projection(binary[cut:], min_char_width_ratio)
            ]
            two_row = top + bottom
            # 双行牌按整张图做垂直投影会把上下两字叠成一块。切开后字数更多时采用切开的结果。
            if len(two_row) > len(found):
                found = two_row
        found = _tighten_single_row(found)
        found = _merge_stroke_fragments(found)
        return _split_wide_boxes(binary, found)

    def ink_from(local):
        if local:
            dark = _local_dark(gray)
            ratio = float(dark.mean())
            light = ~dark
            if prefer_dark_text is True:
                binary = dark if MIN_FG_RATIO <= ratio <= MAX_FG_RATIO else light
            elif prefer_dark_text is False:
                binary = light if MIN_FG_RATIO <= float(light.mean()) <= MAX_FG_RATIO else dark
            else:
                binary = dark if MIN_FG_RATIO <= ratio <= MAX_FG_RATIO else light
        else:
            binary, _pol = binarize(gray, prefer_dark_text)
        binary = morph(binary, 3, "close")
        binary = _peel_edge_bridge(binary)
        binary = _clean(binary)
        return binary

    bw = ink_from(False)
    polarity = bool(prefer_dark_text) if prefer_dark_text is not None else True
    boxes = boxes_of(bw)
    # 全局阈值已经切出 7 或 8 个字时不再换局部阈值，避免改动已经对上的牌。
    if len(boxes) not in (7, 8):
        local_bw = ink_from(True)
        local_boxes = boxes_of(local_bw)
        if len(local_boxes) in (6, 7, 8) and len(local_boxes) != len(boxes):
            bw, boxes = local_bw, local_boxes

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
                            "polarity": polarity}
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

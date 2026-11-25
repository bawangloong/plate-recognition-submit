# -*- coding: utf-8 -*-
"""
plate_common.py —— A 模块公共契约（B/C/D/E 必读）
====================================================
本文件定义了整个小组统一的数据结构、坐标约定和图像 IO 工具。
所有跨模块传递的数据都以本文件为准，修改需通知 D（集成）和 E（评估）。

坐标约定（全组统一，勿改）：
  * 所有坐标一律对应【原始输入图像】的像素坐标系，原点左上角，x 向右，y 向下。
  * bbox = (x, y, w, h)，整数，x/y 为左上角。
  * quad = 四个角点 [[x,y]*4]，浮点，顺序固定为 左上→右上→右下→左下（按车牌自身方向）。
  * 检测阶段内部若缩放图像，返回前必须换算回原图坐标。

图像格式约定：
  * 内存中一律为 OpenCV BGR uint8 ndarray（H, W, 3）。
  * 裁剪出的车牌图交给 B 时同样是 BGR uint8，方向已摆正、字符完整、尽量去掉外框。

错误约定：
  * 文件不存在 / 类型不支持 / 图片损坏 → 抛出 PlateIOError（D 的 GUI 捕获后弹提示）。
  * 图中无车牌 → detect_plates 返回空列表 []，不抛异常。
"""

from dataclasses import dataclass, field
import json
import os

import cv2
import numpy as np

# 支持的输入图片扩展名（D 的文件对话框过滤器也用它）
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class PlateIOError(Exception):
    """输入图片无法读取：不存在、类型不支持、损坏。message 可直接展示给用户。"""


def load_image(path: str) -> np.ndarray:
    """读取图片为 BGR ndarray，支持中文路径。失败抛 PlateIOError。"""
    if not os.path.isfile(path):
        raise PlateIOError(f"文件不存在：{path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise PlateIOError(f"不支持的文件类型：{ext}（支持 {'/'.join(sorted(SUPPORTED_EXTS))}）")
    try:
        data = np.fromfile(path, dtype=np.uint8)  # 兼容中文路径
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        raise PlateIOError(f"图片读取失败：{path}（{e}）")
    if img is None or img.size == 0:
        raise PlateIOError(f"图片已损坏或内容为空：{path}")
    return img


def save_image(path: str, img: np.ndarray) -> None:
    """保存图片，支持中文路径，自动建目录。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise PlateIOError(f"图片编码失败：{path}")
    buf.tofile(path)


@dataclass
class PlateRegion:
    """A 检测模块的输出：原图中一块车牌候选区域。"""
    bbox: tuple                 # (x, y, w, h) 原图坐标，整数，轴对齐外接框
    quad: list                  # [[x,y]*4] 原图坐标，浮点，左上→右上→右下→左下
    angle: float                # 车牌相对水平的倾斜角（度），逆时针为正，范围约 [-45, 45]
    score: float                # 置信度 0~1，越高越可信（供 D 排序展示、E 统计）
    color_hint: str = "unknown" # 检测阶段的底色线索: blue/yellow/green/white/unknown
                                # 注意：这只是线索！最终颜色类别由 B 的 plate_color.py 判定。

    def to_dict(self) -> dict:
        return {
            "bbox": [int(v) for v in self.bbox],
            "quad": [[float(x), float(y)] for x, y in self.quad],
            "angle": round(float(self.angle), 2),
            "score": round(float(self.score), 4),
            "color_hint": self.color_hint,
        }


@dataclass
class RectifiedPlate:
    """A 矫正模块的输出：摆正后的车牌图 + 与原图的对应关系。"""
    region: PlateRegion         # 对应的检测区域（原图坐标）
    image: np.ndarray           # 摆正后的车牌图，BGR uint8，直接交给 B
    transform: np.ndarray       # 3x3 单应矩阵：原图齐次坐标 → 矫正图齐次坐标
                                # B/D 若要把字符位置映射回原图：np.linalg.inv(transform)

    @property
    def size(self) -> tuple:
        """矫正图 (w, h)。"""
        h, w = self.image.shape[:2]
        return w, h


def regions_to_json(regions, extra: dict | None = None) -> str:
    """检测结果序列化为 JSON 字符串（run_demo / D / E 共用格式）。"""
    payload = {"num_plates": len(regions), "plates": [r.to_dict() for r in regions]}
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def order_quad(pts: np.ndarray) -> np.ndarray:
    """把任意顺序的 4 个角点排成 左上→右上→右下→左下（以几何位置为准）。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)
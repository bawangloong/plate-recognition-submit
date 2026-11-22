# -*- coding: utf-8 -*-
"""运行时公共设施：路径解析、图片读取、异常类型、日志。

路径约定
--------
* 源码运行：项目根目录 = app/ 的上一级。
* PyInstaller 打包：只读资源在 sys._MEIPASS（onedir 模式为 _internal/），
  可写目录（results/、日志）在 exe 所在目录，绝不能写到临时解包目录。
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

APP_NAME = "车牌识别系统"
APP_VERSION = "1.0.0"


# ------------------------------------------------------------------ 路径
def is_frozen() -> bool:
    """是否运行在 PyInstaller 生成的 exe 中。"""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """只读资源根目录（字符库、B/C 模块、图标等）。"""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def writable_root() -> Path:
    """可写根目录；打包后是 exe 所在目录，便于用户看到输出。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BUNDLE = bundle_root()
DATA_ROOT = writable_root()
LIBRARY_PATH = BUNDLE / "vendor" / "c_char" / "character_library" / "library.npz"
LIBRARY_JSON = BUNDLE / "vendor" / "c_char" / "character_library" / "library.json"
DEFAULT_RESULT_DIR = DATA_ROOT / "results"
LOG_DIR = DATA_ROOT / "logs"


# ------------------------------------------------------------------ 异常
class UserError(Exception):
    """可以直接展示给用户的错误（带稳定错误码，便于测试断言）。"""

    def __init__(self, code: str, message: str, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


# ------------------------------------------------------------------ 图片
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIN_IMAGE_SIDE = 24
MAX_IMAGE_PIXELS = 120_000_000  # 约 1.2 亿像素，超过则给出明确提示


def read_image_bgr(path: str | os.PathLike) -> "object":
    """读取图片为 BGR ndarray（支持中文路径、EXIF 方向）。

    使用 Pillow 解码而不是 cv2.imread：后者在 Windows 中文路径下会静默失败。
    返回前把图片按 EXIF 方向摆正，手机竖拍照片才能得到正确方向。
    """
    import numpy as np
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    p = Path(path)
    if not p.exists():
        raise UserError("file_not_found", f"文件不存在：{p.name}", str(p))
    if not p.is_file():
        raise UserError("not_a_file", f"选择的不是文件：{p.name}", str(p))
    if p.stat().st_size == 0:
        raise UserError("empty_file", f"文件为空：{p.name}", str(p))
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UserError(
            "unsupported_type",
            f"不支持的文件类型：{p.suffix or '无扩展名'}",
            "支持 JPG / PNG / BMP / TIF / WEBP",
        )
    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            arr = np.asarray(im)
    except Exception as exc:  # 损坏文件、伪扩展名等
        raise UserError("decode_failed", f"图片无法解码或已损坏：{p.name}", repr(exc)) from exc

    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        raise UserError("decode_failed", f"图片格式异常：{p.name}")
    h, w = arr.shape[:2]
    if min(h, w) < MIN_IMAGE_SIDE:
        raise UserError("image_too_small", f"图片尺寸过小（{w}×{h}），无法识别车牌")
    import cv2

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def write_image_unicode(path: str | os.PathLike, image) -> None:
    """支持中文路径的图片保存（cv2.imwrite 不支持中文路径）。"""
    import cv2

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise UserError("encode_failed", f"结果图片编码失败：{p.name}")
    buf.tofile(str(p))


# ------------------------------------------------------------------ 日志
def setup_logging(level: int = 20):
    """文件 + 控制台日志；打包成 windowed exe 后控制台不存在，只写文件。"""
    import logging
    from logging.handlers import RotatingFileHandler

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("plate")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    try:
        fh = RotatingFileHandler(LOG_DIR / "plate_app.log", maxBytes=2_000_000,
                                 backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass
    if sys.stderr is not None and not is_frozen():
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def enable_dpi_awareness() -> None:
    """Windows 高 DPI 下避免 Tk 界面发虚。"""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)

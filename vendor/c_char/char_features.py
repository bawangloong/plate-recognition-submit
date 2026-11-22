"""字符读取、前景归一化和 HOG。构库与推理必须共用这份实现。"""

from pathlib import Path

import cv2
import numpy as np

FEATURE_VERSION = "numpy-hog32-root-pixel045-v2"
HOG_DIM = 324
PIXEL_WEIGHT = 0.45
FEATURE_DIM = HOG_DIM + 32 * 32
IMAGE_SIZE = 32


def read_image(path):
    """通过 imdecode 支持 Windows 中文路径。返回灰度、BGR 或 BGRA。"""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"图片不存在 {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法解码图片 {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"无法编码图片 {path}")
    encoded.tofile(path)


def to_gray(image):
    if isinstance(image, (str, Path)):
        image = read_image(image)
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("字符必须是非空的 NumPy 图像或图片路径")
    if image.ndim not in (2, 3) or min(image.shape[:2]) < 2:
        raise ValueError("字符图像至少为 2×2 像素")
    if image.shape[0] * image.shape[1] > 16_000_000:
        raise ValueError("单字符图像过大，请先由 B 部分切分")
    if image.dtype == np.bool_:
        image = image.astype(np.uint8) * 255
    elif np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("浮点图像须为有限的 [0,1] 数据，或先转为 uint8")
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype == np.uint8:
        if image.max() <= 1:
            image = image * np.uint8(255)
    else:
        raise ValueError("支持 uint8、bool 和 [0,1] 浮点图像")
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    if image.shape[2] == 1:
        return np.ascontiguousarray(image[:, :, 0])
    if image.shape[2] == 4:
        # 将透明背景合成到白色上，避免透明 PNG 的隐藏 RGB 干扰阈值。
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        image = np.rint(image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    elif image.shape[2] != 3:
        raise ValueError("通道数须为 1、3（BGR）或 4（BGRA）")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_character(image):
    """Otsu + 边界背景判断 + 等比例居中，得到 32×32 白字黑底图。"""
    gray = to_gray(image)
    if int(gray.max()) - int(gray.min()) < 2:
        raise ValueError("字符图像为纯色或对比度过低")
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate((binary[0], binary[-1], binary[1:-1, 0], binary[1:-1, -1]))
    if np.count_nonzero(border) > border.size / 2:
        binary = 255 - binary
    ys, xs = np.nonzero(binary)
    if len(xs) < 3:
        raise ValueError("字符有效前景不足")
    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # 保留汉字的分离笔画，不把最大连通域之外的部分误删。
    height, width = crop.shape
    scale = 28 / max(height, width)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    resized = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
    result = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    left = (IMAGE_SIZE - target_width) // 2
    top = (IMAGE_SIZE - target_height) // 2
    result[top:top + target_height, left:left + target_width] = resized
    return result


def extract_hog(normalized):
    """4×4 cell、2×2 cell/block、9方向，3×3×4×9=324维。

    用中心差分求梯度，方向在相邻 bins 线性分票；不做 cell 间空间分票。
    每个 block 做 L2-Hys（截断 0.2 后再 L2），最后做 Root 归一化。
    这是明确参数的 NumPy 实现，不要求 OpenCV 的可选 objdetect 模块。
    """
    if normalized.shape != (32, 32):
        raise ValueError("HOG 输入须为 32×32 归一化灰度图")
    gray = normalized.astype(np.float32) / 255
    padded = np.pad(gray, 1, mode="edge")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    magnitude = np.hypot(gx, gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi) * (9 / np.pi)
    lower = np.floor(orientation).astype(int) % 9
    fraction = orientation - np.floor(orientation)
    cell_y, cell_x = np.indices((32, 32)) // 8
    histograms = np.zeros((4, 4, 9), dtype=np.float32)
    np.add.at(histograms, (cell_y, cell_x, lower), magnitude * (1 - fraction))
    np.add.at(histograms, (cell_y, cell_x, (lower + 1) % 9), magnitude * fraction)
    blocks = []
    for y in range(3):
        for x in range(3):
            block = histograms[y:y + 2, x:x + 2].reshape(-1).copy()
            block /= np.sqrt(np.dot(block, block) + 1e-10)
            block = np.minimum(block, 0.2)
            block /= np.sqrt(np.dot(block, block) + 1e-10)
            blocks.append(block)
    feature = np.concatenate(blocks)
    # 平方根归一化降低少量强梯度的支配作用，再归一到单位长度。
    feature = np.sqrt(feature / (feature.sum() + 1e-12))
    norm = np.linalg.norm(feature)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("字符 HOG 特征为空")
    return feature / norm


def character_feature(image):
    normalized = normalize_character(image)
    hog = extract_hog(normalized)
    pixels = normalized.reshape(-1).astype(np.float32) / 255
    pixels /= np.linalg.norm(pixels)
    # 两个单位向量按 sqrt 权重拼接：其点积等于两种余弦相似度的加权和。
    feature = np.concatenate((np.sqrt(1 - PIXEL_WEIGHT) * hog,
                              np.sqrt(PIXEL_WEIGHT) * pixels)).astype(np.float32)
    return normalized, feature

"""字符标签表。目录名兼容随包提供的公开样本，输出采用规范汉字。"""

PROVINCE_FOLDERS = {
    "zh_cuan": "川", "zh_e": "鄂", "zh_gan": "赣", "zh_gan1": "甘",
    "zh_gui": "贵", "zh_gui1": "桂", "zh_hei": "黑", "zh_hu": "沪",
    "zh_ji": "冀", "zh_jin": "津", "zh_jing": "京", "zh_jl": "吉",
    "zh_liao": "辽", "zh_lu": "鲁", "zh_meng": "蒙", "zh_min": "闽",
    "zh_ning": "宁", "zh_qing": "青", "zh_qiong": "琼", "zh_shan": "陕",
    "zh_su": "苏", "zh_sx": "晋", "zh_wan": "皖", "zh_xiang": "湘",
    "zh_xin": "新", "zh_yu": "豫", "zh_yu1": "渝", "zh_yue": "粤",
    "zh_yun": "云", "zh_zang": "藏", "zh_zhe": "浙",
}
DIGITS = "0123456789"
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
PROVINCES = "".join(PROVINCE_FOLDERS.values())
ALL_LABELS = list(DIGITS + LETTERS + PROVINCES)
FOLDER_TO_LABEL = {char: char for char in DIGITS + LETTERS} | PROVINCE_FOLDERS
LABEL_TO_FOLDER = {label: folder for folder, label in FOLDER_TO_LABEL.items()}


def decode_folder(folder):
    """自建样本既可用 zh_hu 目录，也可直接用 沪 目录。"""
    if folder in FOLDER_TO_LABEL:
        return FOLDER_TO_LABEL[folder]
    if folder in ALL_LABELS:
        return folder
    raise ValueError(f"不支持的字符目录 {folder!r}，请查看 labels.py")

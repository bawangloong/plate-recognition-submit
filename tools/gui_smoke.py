# -*- coding: utf-8 -*-
"""GUI 冒烟测试：不开文件对话框，直接注入图片与结果，截图检查界面。"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tkinter as tk
from PIL import ImageGrab

from app.gui import PlateApp
from app.pipeline import PlatePipeline
from app.runtime import enable_dpi_awareness, read_image_bgr

enable_dpi_awareness()
root = tk.Tk()
app = PlateApp(root, PlatePipeline())
root.geometry("1280x820+40+40")
root.attributes("-topmost", True)
root.lift()
root.update()
time.sleep(0.8)

path = ROOT / "samples" / "multi_two.png"
image = read_image_bgr(path)
app.image = image
app.image_path = str(path)
app.status_var.set("识别完成：共 2 块车牌")
result = app.pipeline.recognize_file(path)
app.result = result
app._build_plate_buttons()
app.select_plate(0)
app._show_image(image, annotate=result["plates"])
root.lift()
root.update()
time.sleep(1.0)
root.update()

x, y = root.winfo_rootx(), root.winfo_rooty()
w, h = root.winfo_width(), root.winfo_height()
img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
out = ROOT / "docs" / "gui_screenshot.png"
out.parent.mkdir(exist_ok=True)
img.save(out)
print("saved", out, img.size)

app.notebook.select(1)
app._render_process(force=True)
root.update()
time.sleep(0.6)
root.update()
img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
out2 = ROOT / "docs" / "gui_process.png"
img.save(out2)
print("saved", out2, img.size)
root.destroy()

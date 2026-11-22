# -*- coding: utf-8 -*-
"""Tkinter 图形界面：选图、识别、多车牌切换、过程图、字符切片和异常提示。

界面只在主线程操作 Tk 控件；识别放在后台线程，通过 queue 回传进度和结果，
因此识别期间窗口仍可拖动、响应取消。
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np
from PIL import Image, ImageTk

from .export import export_result
from .pipeline import PlatePipeline
from .plate_utils import compose_process_panel, draw_detections, fit_to_box, upscale_char
from .runtime import (ALLOWED_EXTENSIONS, APP_NAME, APP_VERSION, DEFAULT_RESULT_DIR,
                      LOG_DIR, UserError, enable_dpi_awareness, read_image_bgr)

FONT = ("Microsoft YaHei UI", 10)
FONT_SMALL = ("Microsoft YaHei UI", 9)
FONT_TITLE = ("Microsoft YaHei UI", 17, "bold")
FONT_NUMBER = ("Microsoft YaHei UI", 26, "bold")
FONT_MONO = ("Consolas", 10)

COLOR_BG = {
    "blue": "#1e5aa8", "yellow": "#f2c200", "green": "#12a05c",
    "white": "#e8e8e8", "black": "#222222", "unknown": "#8a8a8a",
}
COLOR_FG = {
    "blue": "#ffffff", "yellow": "#1a1a1a", "green": "#ffffff",
    "white": "#1a1a1a", "black": "#ffffff", "unknown": "#ffffff",
}

ERROR_HINTS = {
    "unsupported_type": "请选择 JPG / PNG / BMP / TIF / WEBP 图片。",
    "decode_failed": "文件可能已损坏，或扩展名与真实格式不符。",
    "empty_file": "文件大小为 0，请重新选择。",
    "image_too_small": "图片太小，无法定位车牌。",
    "no_plate": "可以换一张更清晰、车牌更完整的图片再试。",
    "library_missing": "发行包缺少字符库，请重新解压完整目录。",
    "library_invalid": "字符库损坏或与程序版本不一致，请重新解压发行包。",
}


class Cancelled(Exception):
    pass


class PlateApp:
    def __init__(self, root: tk.Tk, pipeline: PlatePipeline):
        self.root = root
        self.pipeline = pipeline
        self.image: np.ndarray | None = None
        self.image_path: str = ""
        self.result: dict | None = None
        self.selected = 0
        self.queue: queue.Queue = queue.Queue()
        self.running = False
        self.cancel_flag = False
        self.worker: threading.Thread | None = None
        self._canvas_photo = None
        self._process_photo = None
        self._char_photos: list = []
        self.last_export: Path | None = None

        self._build_ui()
        self.root.after(80, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Control-o>", lambda e: self.choose_image())
        self.root.bind("<F5>", lambda e: self.start_recognition())
        self.root.bind("<Control-s>", lambda e: self.save_result())

    # ------------------------------------------------------------ 界面搭建
    def _build_ui(self):
        self.root.title(f"{APP_NAME}  v{APP_VERSION}")
        self.root.geometry("1280x820")
        self.root.minsize(1024, 680)
        self.root.configure(bg="#f2f5f8")

        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure(".", font=FONT)
        style.configure("TButton", padding=(10, 5))
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Treeview", rowheight=26)
        style.configure("TNotebook.Tab", padding=(14, 6))

        self._build_menu()

        header = tk.Frame(self.root, bg="#16324f", height=56)
        header.pack(side="top", fill="x")
        tk.Label(header, text="车牌识别系统", bg="#16324f", fg="white",
                 font=("Microsoft YaHei UI", 18, "bold")).pack(side="left", padx=18)
        tk.Label(header, text="检测 · 矫正 · 颜色 · 字符分割 · 字符库匹配",
                 bg="#16324f", fg="#a9c4dc", font=FONT_SMALL).pack(side="left", padx=4)
        tk.Label(header, text="D 模块 · 系统集成", bg="#16324f", fg="#7fa6c9",
                 font=FONT_SMALL).pack(side="right", padx=18)

        toolbar = ttk.Frame(self.root, padding=(12, 8))
        toolbar.pack(side="top", fill="x")
        self.btn_open = ttk.Button(toolbar, text="选择图片", command=self.choose_image)
        self.btn_run = ttk.Button(toolbar, text="开始识别", style="Accent.TButton",
                                  command=self.start_recognition)
        self.btn_cancel = ttk.Button(toolbar, text="取消", command=self.cancel,
                                     state="disabled")
        self.btn_save = ttk.Button(toolbar, text="保存结果", command=self.save_result,
                                   state="disabled")
        self.btn_folder = ttk.Button(toolbar, text="打开结果目录",
                                     command=self.open_result_dir)
        for btn in (self.btn_open, self.btn_run, self.btn_cancel, self.btn_save,
                    self.btn_folder):
            btn.pack(side="left", padx=(0, 8))
        self.progress = ttk.Progressbar(toolbar, mode="determinate", maximum=100,
                                        length=220)
        self.progress.pack(side="right", padx=(8, 0))
        self.status_var = tk.StringVar(value="请选择一张本地图片，或按 Ctrl+O")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="right", padx=10)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 8))

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.canvas = tk.Canvas(left, bg="#1d2126", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._render_canvas())

        plate_bar = ttk.LabelFrame(left, text="车牌列表（点击切换）", padding=6)
        plate_bar.pack(fill="x", pady=(8, 0))
        self.plate_frame = ttk.Frame(plate_bar)
        self.plate_frame.pack(fill="x")
        self.plate_hint = ttk.Label(self.plate_frame, text="尚未识别",
                                    foreground="#777777")
        self.plate_hint.pack(side="left", padx=4, pady=6)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)
        self._build_result_tab()
        self._build_process_tab()
        self._build_log_tab()

        status = ttk.Frame(self.root, padding=(12, 4))
        status.pack(side="bottom", fill="x")
        self.info_var = tk.StringVar(
            value=f"字符库 {self.pipeline.library_samples} 样本 / "
                  f"{self.pipeline.library_classes} 类 · 检测后端：{self.pipeline.backend.name}")
        ttk.Label(status, textvariable=self.info_var, foreground="#555555",
                  font=FONT_SMALL).pack(side="left")
        self.time_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.time_var, foreground="#555555",
                  font=FONT_SMALL).pack(side="right")

    def _build_menu(self):
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="打开图片…", accelerator="Ctrl+O",
                              command=self.choose_image)
        file_menu.add_command(label="开始识别", accelerator="F5",
                              command=self.start_recognition)
        file_menu.add_command(label="保存结果…", accelerator="Ctrl+S",
                              command=self.save_result)
        file_menu.add_separator()
        file_menu.add_command(label="打开结果目录", command=self.open_result_dir)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_close)
        menu.add_cascade(label="文件", menu=file_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="操作说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
        menu.add_cascade(label="帮助", menu=help_menu)
        self.root.config(menu=menu)

    def _build_result_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="识别结果")

        top = ttk.Frame(tab)
        top.pack(fill="x")
        self.number_var = tk.StringVar(value="—")
        tk.Label(top, textvariable=self.number_var, font=FONT_NUMBER,
                 fg="#16324f", anchor="w").pack(side="left")
        self.color_label = tk.Label(top, text="", font=("Microsoft YaHei UI", 11, "bold"),
                                    padx=10, pady=3, bg="#cccccc")
        self.color_label.pack(side="right", padx=(6, 0))

        meta = ttk.Frame(tab)
        meta.pack(fill="x", pady=(6, 8))
        self.meta_vars = {}
        for i, (key, title) in enumerate((("chars", "字符数"), ("conf", "平均匹配分"),
                                          ("cost", "本块耗时"), ("status", "状态"))):
            cell = ttk.Frame(meta)
            cell.grid(row=0, column=i, sticky="w", padx=(0, 22))
            ttk.Label(cell, text=title, foreground="#777777", font=FONT_SMALL).pack(anchor="w")
            var = tk.StringVar(value="—")
            ttk.Label(cell, textvariable=var, font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
            self.meta_vars[key] = var

        ttk.Label(tab, text="字符切片与匹配结果", foreground="#555555",
                  font=FONT_SMALL).pack(anchor="w", pady=(6, 2))
        self.char_holder = ttk.Frame(tab)
        self.char_holder.pack(fill="x")
        self.char_empty = ttk.Label(self.char_holder, text="识别后在此显示每个字符的匹配结果",
                                    foreground="#999999")
        self.char_empty.pack(anchor="w", pady=6)

        ttk.Label(tab, text="提示与警告", foreground="#555555",
                  font=FONT_SMALL).pack(anchor="w", pady=(8, 2))
        self.warn_text = tk.Text(tab, height=6, wrap="word", font=FONT_SMALL,
                                 bg="#fbfbfb", relief="solid", borderwidth=1)
        self.warn_text.pack(fill="both", expand=True)
        self.warn_text.configure(state="disabled")

    def _build_process_tab(self):
        tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(tab, text="处理过程")
        self.process_canvas = tk.Canvas(tab, bg="#ffffff", highlightthickness=0)
        self.process_canvas.pack(fill="both", expand=True)
        self.process_canvas.bind("<Configure>", lambda e: self._render_process())

    def _build_log_tab(self):
        tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(tab, text="运行日志")
        self.log_text = ScrolledText(tab, height=10, font=FONT_MONO, wrap="word",
                                     state="disabled", bg="#1d2126", fg="#d8dee9",
                                     insertbackground="#d8dee9")
        self.log_text.pack(fill="both", expand=True)

    # ------------------------------------------------------------ 日志
    def log(self, message: str):
        import time
        line = f"[{time.strftime('%H:%M:%S')}] {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------ 选图
    def choose_image(self):
        if self.running:
            return
        path = filedialog.askopenfilename(
            title="选择要识别的图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                       ("所有文件", "*.*")])
        if not path:  # 取消选图：静默返回
            self.log("已取消选择图片")
            return
        ext = Path(path).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            messagebox.showerror("文件类型不支持",
                                 f"不支持的文件类型：{ext or '无扩展名'}\n\n"
                                 "请选择 JPG / PNG / BMP / TIF / WEBP 图片。")
            self.log(f"拒绝不支持的文件类型：{path}")
            return
        try:
            image = read_image_bgr(path)
        except UserError as exc:
            self._show_user_error(exc)
            return
        self.image_path = path
        self.image = image
        self.result = None
        self.selected = 0
        self._clear_results()
        self.status_var.set(f"已选择：{Path(path).name}（{image.shape[1]}×{image.shape[0]}），点击开始识别")
        self._show_image(image, annotate=None)
        self.btn_run.state(["!disabled"])
        self.btn_save.state(["disabled"])
        self.log(f"打开图片：{path}")

    # ------------------------------------------------------------ 识别
    def start_recognition(self):
        if self.running:
            return
        if self.image is None or not self.image_path:
            messagebox.showinfo("请先选择图片", "请点击“选择图片”或按 Ctrl+O 选择一张本地图片。")
            return
        self.running = True
        self.cancel_flag = False
        self.result = None
        self._clear_results()
        self._set_busy(True)
        self.progress["value"] = 0
        self.status_var.set("正在识别…")
        self.log("开始识别：" + Path(self.image_path).name)
        self.worker = threading.Thread(target=self._worker, args=(self.image_path,), daemon=True)
        self.worker.start()

    def _worker(self, path: str):
        try:
            result = self.pipeline.recognize_file(path, progress=self._progress_cb)
            self.queue.put(("done", result))
        except Cancelled:
            self.queue.put(("cancelled",))
        except UserError as exc:
            self.queue.put(("user_error", exc.code, exc.message, exc.detail))
        except Exception as exc:  # 未预期异常：写日志并给用户可读提示
            self.queue.put(("crash", repr(exc), traceback.format_exc()))

    def _progress_cb(self, stage: str, fraction: float, detail: str):
        if self.cancel_flag:
            raise Cancelled()
        self.queue.put(("progress", stage, fraction, detail))

    def cancel(self):
        if self.running:
            self.cancel_flag = True
            self.status_var.set("正在取消…")
            self.log("用户点击取消")

    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, stage, fraction, detail = msg
                    self.progress["value"] = fraction * 100
                    text = f"{stage}… {detail}" if detail else f"{stage}…"
                    self.status_var.set(text)
                elif kind == "done":
                    self._on_done(msg[1])
                elif kind == "cancelled":
                    self._finish_running()
                    self.status_var.set("已取消识别")
                    self.progress["value"] = 0
                    self.log("识别已取消")
                elif kind == "user_error":
                    self._finish_running()
                    self.progress["value"] = 0
                    _, code, message, detail = msg
                    self._show_user_error(UserError(code, message, detail))
                elif kind == "crash":
                    self._finish_running()
                    self.progress["value"] = 0
                    _, message, tb = msg
                    self.log("未预期异常：\n" + tb)
                    messagebox.showerror("程序异常",
                                         f"识别过程中出现未预期错误：\n{message}\n\n"
                                         "详情已写入 logs/plate_app.log。")
                    self.status_var.set("识别失败")
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _on_done(self, result: dict):
        self._finish_running()
        self.result = result
        self.progress["value"] = 100
        total = result["timings"].get("total_ms", 0)
        self.time_var.set(f"总耗时 {total:.0f} ms")
        self.log(f"识别完成：{result['plate_count']} 块车牌，耗时 {total:.0f} ms")

        if not result.get("ok"):
            self.status_var.set("未检测到车牌")
            self._show_image(self.image, annotate=None)
            messagebox.showinfo("未检测到车牌",
                                "没有在该图片中定位到车牌。\n\n"
                                "建议：确认车牌完整、清晰、不过曝，或换一张图片。")
            return

        self.btn_save.state(["!disabled"])
        self.status_var.set(f"识别完成：共 {result['plate_count']} 块车牌")
        self._build_plate_buttons()
        self.select_plate(0)
        self._show_image(self.image, annotate=result["plates"])

    def _finish_running(self):
        self.running = False
        self._set_busy(False)

    def _set_busy(self, busy: bool):
        for btn in (self.btn_open, self.btn_run):
            if busy:
                btn.state(["disabled"])
            else:
                btn.state(["!disabled"])
        if busy:
            self.btn_cancel.state(["!disabled"])
            self.btn_save.state(["disabled"])
        else:
            self.btn_cancel.state(["disabled"])

    # ------------------------------------------------------------ 多车牌
    def _build_plate_buttons(self):
        for child in self.plate_frame.winfo_children():
            child.destroy()
        self.plate_buttons = []
        plates = self.result.get("plates", []) if self.result else []
        if not plates:
            ttk.Label(self.plate_frame, text="未检测到车牌",
                      foreground="#777777").pack(side="left", padx=4, pady=6)
            return
        for i, plate in enumerate(plates):
            text = (f"#{plate['index']}  {plate['text'] or '(未识别)'}  "
                    f"{plate['color']['label']} · {plate['segmentation']['count']} 字符")
            btn = tk.Button(self.plate_frame, text=text, font=FONT,
                            anchor="w", relief="groove", padx=10, pady=4,
                            command=lambda idx=i: self.select_plate(idx))
            btn.pack(side="left", padx=(0, 6))
            self.plate_buttons.append(btn)
        self._highlight_plate_button()

    def _highlight_plate_button(self):
        if not getattr(self, "plate_buttons", None):
            return
        for i, btn in enumerate(self.plate_buttons):
            active = i == self.selected
            btn.configure(bg="#dce9f7" if active else "#f4f4f4",
                          relief="sunken" if active else "groove")

    def select_plate(self, index: int):
        if not self.result or not self.result.get("plates"):
            return
        plates = self.result["plates"]
        index = max(0, min(index, len(plates) - 1))
        self.selected = index
        plate = plates[index]
        self._highlight_plate_button()
        self._update_result(plate)
        self._render_process(force=True)

    # ------------------------------------------------------------ 结果面板
    def _clear_results(self):
        self.number_var.set("—")
        self.color_label.configure(text="", bg="#cccccc")
        for var in self.meta_vars.values():
            var.set("—")
        for child in self.char_holder.winfo_children():
            child.destroy()
        self._char_photos = []
        ttk.Label(self.char_holder, text="识别后在此显示每个字符的匹配结果",
                  foreground="#999999").pack(anchor="w", pady=6)
        self.warn_text.configure(state="normal")
        self.warn_text.delete("1.0", "end")
        self.warn_text.configure(state="disabled")
        for child in self.plate_frame.winfo_children():
            child.destroy()
        self.plate_hint = ttk.Label(self.plate_frame, text="尚未识别", foreground="#777777")
        self.plate_hint.pack(side="left", padx=4, pady=6)
        self.process_canvas.delete("all")

    def _update_result(self, plate: dict):
        self.number_var.set(plate["text"] or "未识别")
        color = plate.get("color") or {}
        key = color.get("color", "unknown")
        self.color_label.configure(text=color.get("label", "无法判断"),
                                   bg=COLOR_BG.get(key, "#cccccc"),
                                   fg=COLOR_FG.get(key, "#ffffff"))
        self.meta_vars["chars"].set(str(plate["segmentation"]["count"]))
        self.meta_vars["conf"].set(f"{plate.get('confidence', 0):.2f}")
        self.meta_vars["cost"].set(f"{plate['timings'].get('total_ms', 0):.0f} ms")
        status_text = {"ok": "正常", "low_confidence": "低置信", "segment_empty": "无字符"}
        self.meta_vars["status"].set(status_text.get(plate.get("status"), plate.get("status", "—")))

        for child in self.char_holder.winfo_children():
            child.destroy()
        self._char_photos = []
        chars = plate.get("chars") or []
        records = (plate.get("recognition") or {}).get("characters") or []
        if not chars:
            ttk.Label(self.char_holder, text="未分割出字符", foreground="#999999").pack(anchor="w")
        else:
            per_row = 7
            for i, ch in enumerate(chars):
                cell = ttk.Frame(self.char_holder, padding=(2, 2))
                cell.grid(row=i // per_row, column=i % per_row, padx=3, pady=3)
                img = upscale_char(ch, 3)
                pil = Image.fromarray(img).convert("RGB")
                photo = ImageTk.PhotoImage(pil)
                self._char_photos.append(photo)
                tk.Label(cell, image=photo, bg="white", relief="solid",
                         borderwidth=1).pack()
                rec = records[i] if i < len(records) else {}
                label = rec.get("label", "?")
                sim = rec.get("similarity")
                sim_text = f"{sim:.2f}" if isinstance(sim, (int, float)) else "--"
                fg = "#b00020" if (isinstance(sim, (int, float)) and sim < 0.75) else "#1a1a1a"
                tk.Label(cell, text=f"{label}  {sim_text}", font=FONT_SMALL,
                         fg=fg).pack()

        warnings = list(plate.get("warnings") or [])
        if plate.get("segmentation", {}).get("message"):
            warnings.insert(0, plate["segmentation"]["message"])
        self.warn_text.configure(state="normal")
        self.warn_text.delete("1.0", "end")
        self.warn_text.insert("1.0", "\n".join(f"· {w}" for w in warnings) if warnings else "无")
        self.warn_text.configure(state="disabled")

    # ------------------------------------------------------------ 图像显示
    def _show_image(self, image: np.ndarray | None, annotate=None):
        self._display_source = image
        if image is None:
            return
        shown = draw_detections(image, annotate) if annotate else image
        self._display_image = shown
        self._render_canvas()

    def _render_canvas(self):
        img = getattr(self, "_display_image", None)
        canvas = self.canvas
        canvas.delete("all")
        if img is None:
            canvas.create_text(canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                               text="请选择图片\n\n支持 JPG / PNG / BMP / TIF / WEBP",
                               fill="#8fa3b8", font=("Microsoft YaHei UI", 14), justify="center")
            return
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 20 or ch < 20:
            return
        fitted, _ = fit_to_box(img, cw - 16, ch - 16)
        pil = Image.fromarray(cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB))
        self._canvas_photo = ImageTk.PhotoImage(pil)
        canvas.create_image(cw // 2, ch // 2, image=self._canvas_photo, anchor="center")

    def _render_process(self, force=False):
        if not self.result or not self.result.get("plates"):
            return
        plate = self.result["plates"][self.selected]
        panel = compose_process_panel(plate)
        cw = max(200, self.process_canvas.winfo_width())
        ch = max(150, self.process_canvas.winfo_height())
        fitted, _ = fit_to_box(panel, cw - 12, ch - 12)
        pil = Image.fromarray(cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB))
        self._process_photo = ImageTk.PhotoImage(pil)
        self.process_canvas.delete("all")
        self.process_canvas.create_image(cw // 2, ch // 2, image=self._process_photo,
                                         anchor="center")

    # ------------------------------------------------------------ 保存/目录
    def save_result(self):
        if not self.result or not self.result.get("plates"):
            messagebox.showinfo("没有可保存的结果", "请先完成一次识别。")
            return
        DEFAULT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        folder = filedialog.askdirectory(title="选择保存结果的文件夹",
                                         initialdir=str(DEFAULT_RESULT_DIR))
        if not folder:
            self.log("已取消保存")
            return
        try:
            out = export_result(self.result, self.image, folder,
                                Path(self.image_path).name)
        except Exception as exc:
            self.log(f"保存失败：{exc}")
            messagebox.showerror("保存失败", f"写入结果时出错：\n{exc}")
            return
        self.last_export = out
        self.status_var.set(f"结果已保存到 {out}")
        self.log(f"结果已保存：{out}")
        if messagebox.askyesno("保存成功", f"结果已保存到：\n{out}\n\n是否打开该文件夹？"):
            self._open_path(out)

    def open_result_dir(self):
        target = self.last_export or DEFAULT_RESULT_DIR
        target.mkdir(parents=True, exist_ok=True)
        self._open_path(target)

    @staticmethod
    def _open_path(path: Path):
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            pass

    # ------------------------------------------------------------ 异常/帮助
    def _show_user_error(self, exc: UserError):
        self.log(f"错误[{exc.code}]：{exc.message}")
        if exc.detail:
            self.log(f"    详情：{exc.detail}")
        hint = ERROR_HINTS.get(exc.code, "")
        message = exc.message + (f"\n\n{hint}" if hint else "")
        messagebox.showerror("无法完成识别", message)
        self.status_var.set(exc.message)

    def show_help(self):
        messagebox.showinfo(
            "操作说明",
            "1. 点击“选择图片”（Ctrl+O）选择本地图片；\n"
            "2. 点击“开始识别”（F5），识别在后台进行，可随时取消；\n"
            "3. 在左下方车牌列表切换多块车牌；\n"
            "4. “识别结果”页查看号码、颜色、字符数与每个字符的匹配分；\n"
            "5. “处理过程”页查看车牌裁剪、二值化、字符框和字符切片；\n"
            "6. 点击“保存结果”（Ctrl+S）导出标框图、过程图、字符图和 JSON。\n\n"
            "支持 JPG / PNG / BMP / TIF / WEBP；取消选图或选择损坏文件不会影响程序。")

    def show_about(self):
        messagebox.showinfo(
            "关于",
            f"{APP_NAME}  v{APP_VERSION}\n\n"
            "摄像头/图片车牌识别课程设计 · D 模块（GUI 与系统集成）\n"
            f"字符库：{self.pipeline.library_samples} 个样本，"
            f"{self.pipeline.library_classes} 个类别\n"
            f"检测后端：{self.pipeline.backend.name}\n"
            "A：检测矫正  B：颜色与分割  C：字符库识别  D：集成与界面")

    def _on_close(self):
        if self.running:
            if not messagebox.askokcancel("正在识别", "识别尚未结束，确定退出吗？"):
                return
            self.cancel_flag = True
        self.root.destroy()


def run() -> int:
    enable_dpi_awareness()
    root = tk.Tk()
    try:
        splash = tk.Toplevel(root)
        splash.title("正在启动")
        splash.geometry("360x120+%d+%d" % (
            (root.winfo_screenwidth() - 360) // 2,
            (root.winfo_screenheight() - 120) // 2))
        splash.attributes("-topmost", True)
        tk.Label(splash, text="正在加载字符库…", font=FONT).pack(expand=True, pady=20)
        root.update()
    except Exception:
        splash = None

    try:
        pipeline = PlatePipeline()
    except UserError as exc:
        if splash is not None:
            splash.destroy()
        messagebox.showerror("启动失败", f"{exc.message}\n\n请确认发行包目录完整。")
        root.destroy()
        return 1
    if splash is not None:
        splash.destroy()

    PlateApp(root, pipeline)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

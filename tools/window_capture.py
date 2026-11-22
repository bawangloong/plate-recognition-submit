# -*- coding: utf-8 -*-
"""列出窗口标题，并把“车牌识别系统”窗口置前截图（用于验证 exe 启动）。"""

import ctypes
import time
from ctypes import wintypes
from pathlib import Path

from PIL import ImageGrab

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

found = []


def callback(hwnd, lparam):
    if user32.IsWindowVisible(hwnd):
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if "车牌" in buf.value:
                found.append((hwnd, buf.value))
    return True


proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)(callback)
user32.EnumWindows(proc, 0)
print("windows:", found)

if found:
    hwnd, title = found[0]
    user32.ShowWindow(hwnd, 9)          # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
    time.sleep(1.2)
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    print("rect", rect.left, rect.top, rect.right, rect.bottom)
    img = ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom))
    out = Path(__file__).resolve().parent.parent / "docs" / "exe_running.png"
    img.save(out)
    print("saved", out, img.size)
    user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
    print("closed")

# -*- coding: utf-8 -*-
"""D 模块冒烟测试：对 samples/ 下的演示图跑完整流水线并打印结果。

用法：python tools/smoke_test.py [图片...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import PlatePipeline  # noqa: E402

CASES = [
    "normal_blue.png", "tilt_yellow.png", "multi_two.png", "no_plate.png",
    "green_8.png", "blur_glare.png", "black_plate.png", "large_3000.jpg",
]


def main(argv: list[str]) -> int:
    pipeline = PlatePipeline()
    print(f"backend: {pipeline.backend.name}")
    print(f"library: {pipeline.library_samples} samples / {pipeline.library_classes} classes")
    files = argv or [str(ROOT / "samples" / name) for name in CASES]
    for path in files:
        t = time.perf_counter()
        try:
            result = pipeline.recognize_file(path)
        except Exception as exc:
            print(f"{Path(path).name:20s} ERROR {exc}")
            continue
        ms = (time.perf_counter() - t) * 1000
        print(f"{Path(path).name:20s} ok={result['ok']} plates={result['plate_count']} "
              f"rejected={result['rejected']} {ms:6.0f}ms")
        for plate in result["plates"]:
            print(f"    #{plate['index']} bbox={plate['bbox']} "
                  f"{plate['color']['label']} chars={plate['segmentation']['count']} "
                  f"text={plate['text']} conf={plate['confidence']:.2f}")
        if result.get("error"):
            print("    error:", result["error"]["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

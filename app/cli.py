# -*- coding: utf-8 -*-
"""命令行批量识别入口（不弹界面），用于测试、演示和写报告数据。

示例：
    python -m app.cli samples/normal_blue.png
    python -m app.cli samples/*.png --out results --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .export import export_result
from .pipeline import PlatePipeline, result_to_json
from .runtime import UserError, read_image_bgr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="车牌识别系统 命令行入口")
    parser.add_argument("images", nargs="+", help="待识别图片")
    parser.add_argument("--out", type=Path, help="导出结果的根目录（含图与 JSON）")
    parser.add_argument("--json", action="store_true", help="只输出 JSON 到标准输出")
    parser.add_argument("--quiet", action="store_true", help="不打印过程信息")
    args = parser.parse_args(argv)

    try:
        pipeline = PlatePipeline()
    except UserError as exc:
        print(f"启动失败：{exc.message}", file=sys.stderr)
        return 1

    all_results = []
    failures = 0
    for path in args.images:
        t = time.perf_counter()
        try:
            result = pipeline.recognize_file(path)
        except UserError as exc:
            failures += 1
            if args.json:
                all_results.append({"source": path, "ok": False,
                                    "error": {"code": exc.code, "message": exc.message}})
            else:
                print(f"{Path(path).name:24s} 错误[{exc.code}] {exc.message}")
            continue
        all_results.append(result_to_json(result))
        if not args.quiet and not args.json:
            ms = (time.perf_counter() - t) * 1000
            if result["ok"]:
                summary = "；".join(
                    f"#{p['index']} {p['text']}（{p['color']['label']}，"
                    f"{p['segmentation']['count']} 字符，{p['confidence']:.2f}）"
                    for p in result["plates"])
            else:
                summary = result["error"]["message"] if result.get("error") else "无结果"
            print(f"{Path(path).name:24s} {ms:6.0f}ms  {summary}")
        if args.out:
            try:
                image = read_image_bgr(path)
                folder = export_result(result, image, args.out, Path(path).name)
                if not args.quiet and not args.json:
                    print(f"    导出：{folder}")
            except Exception as exc:
                print(f"    导出失败：{exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(all_results, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

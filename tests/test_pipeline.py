# -*- coding: utf-8 -*-
"""流水线与异常处理回归测试（不启动 GUI）。

运行：python -m unittest discover -s tests -v
测试文件写在 tests/_tmp 下，避免系统临时目录的权限限制。
"""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import PlatePipeline  # noqa: E402
from app.runtime import UserError, read_image_bgr  # noqa: E402

SAMPLES = ROOT / "samples"
TMP_ROOT = ROOT / "tests" / "_tmp"


@contextmanager
def workdir():
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = PlatePipeline()

    def test_blue_plate(self):
        result = self.pipeline.recognize_file(SAMPLES / "normal_blue.png")
        self.assertTrue(result["ok"])
        self.assertEqual(result["plate_count"], 1)
        plate = result["plates"][0]
        self.assertEqual(plate["color"]["color"], "blue")
        self.assertEqual(plate["segmentation"]["count"], 7)
        self.assertEqual(plate["text"], "京A12345")

    def test_new_energy_8_chars(self):
        result = self.pipeline.recognize_file(SAMPLES / "multi_two.png")
        self.assertTrue(result["ok"])
        self.assertEqual(result["plate_count"], 2)
        texts = sorted(p["text"] for p in result["plates"])
        self.assertIn("沪AD12345", texts)

    def test_no_plate(self):
        result = self.pipeline.recognize_file(SAMPLES / "no_plate.png")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "no_plate")

    def test_unsupported_type(self):
        with workdir() as tmp:
            path = tmp / "note.txt"
            path.write_text("hello", encoding="utf-8")
            with self.assertRaises(UserError) as ctx:
                read_image_bgr(path)
            self.assertEqual(ctx.exception.code, "unsupported_type")

    def test_corrupt_image(self):
        with workdir() as tmp:
            path = tmp / "broken.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-image")
            with self.assertRaises(UserError) as ctx:
                read_image_bgr(path)
            self.assertEqual(ctx.exception.code, "decode_failed")

    def test_chinese_path(self):
        with workdir() as tmp:
            target = tmp / "中文目录" / "测试图片 京A12345.png"
            target.parent.mkdir()
            shutil.copy(SAMPLES / "normal_blue.png", target)
            result = self.pipeline.recognize_file(target)
            self.assertTrue(result["ok"])
            self.assertEqual(result["plates"][0]["text"], "京A12345")


if __name__ == "__main__":
    unittest.main(verbosity=2)

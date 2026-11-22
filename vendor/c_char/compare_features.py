"""仅使用 validation 比较特征权重；不读取测试标签选择参数。"""

import csv
import json
from pathlib import Path

import numpy as np

from char_features import extract_hog, normalize_character
from labels import ALL_LABELS, PROVINCES, DIGITS, LETTERS
from recognize_chars import CharacterRecognizer

ROOT = Path(__file__).resolve().parent


def compare():
    recognizer = CharacterRecognizer(ROOT / "character_library/library_real_only.npz")
    with (ROOT / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as file:
        records = [row for row in csv.DictReader(file) if row["split"] == "validation"]
    queries = np.stack([normalize_character(ROOT / row["file"]) for row in records])
    qh = np.stack([extract_hog(image) for image in queries])
    lh = np.stack([extract_hog(image) for image in recognizer.normalized_images])
    qp = queries.reshape(len(queries), -1).astype(np.float32) / 255
    lp = recognizer.normalized_images.reshape(len(recognizer.labels), -1).astype(np.float32) / 255
    qp /= np.linalg.norm(qp, axis=1, keepdims=True)
    lp /= np.linalg.norm(lp, axis=1, keepdims=True)
    truth = [row["label"] for row in records]
    results = []
    for weight in [0, 0.15, 0.3, 0.45, 0.6]:
        predictions = []
        for first in range(0, len(records), 128):
            scores = (1 - weight) * (qh[first:first + 128] @ lh.T) + weight * (qp[first:first + 128] @ lp.T)
            predictions.extend(recognizer.labels[np.argmax(scores, axis=1)].tolist())
        row = {"pixel_weight": weight}
        for name, labels in [("all", ALL_LABELS), ("provinces", PROVINCES), ("alnum", DIGITS + LETTERS)]:
            indices = [i for i, label in enumerate(truth) if label in labels]
            row[name] = sum(predictions[i] == truth[i] for i in indices) / len(indices)
        results.append(row)
    output = ROOT / "results/validation_feature_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    compare()

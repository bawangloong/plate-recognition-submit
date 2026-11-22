"""对固定 validation/test 划分进行独立单字符评估，输出逐样本结果。"""

import argparse
from collections import Counter
import csv
import json
import platform
from pathlib import Path
import time

import cv2
import numpy as np
import PIL

from char_features import character_feature
from labels import ALL_LABELS, DIGITS, LETTERS, PROVINCES
from recognize_chars import CharacterRecognizer

ROOT = Path(__file__).resolve().parent


def group_metrics(truth, predictions):
    result = {}
    groups = {"all": ALL_LABELS, "digits": DIGITS, "letters": LETTERS, "provinces": PROVINCES}
    for name, labels in groups.items():
        indices = [i for i, label in enumerate(truth) if label in labels]
        correct = sum(predictions[i]["label"] == truth[i] for i in indices)
        top3 = sum(truth[i] in [c["label"] for c in predictions[i]["candidates"]] for i in indices)
        accepted = [i for i in indices if predictions[i]["status"] == "ok"]
        accepted_correct = sum(predictions[i]["label"] == truth[i] for i in accepted)
        recalls = []
        for label in labels:
            subset = [i for i in indices if truth[i] == label]
            if subset:
                recalls.append(sum(predictions[i]["label"] == label for i in subset) / len(subset))
        result[name] = {
            "count": len(indices), "top1_correct": correct, "top1_accuracy": correct / len(indices) if indices else None,
            "top3_correct": top3, "top3_accuracy": top3 / len(indices) if indices else None,
            "macro_recall": float(np.mean(recalls)) if recalls else None,
            "auto_accepted_count": len(accepted),
            "auto_accepted_accuracy": accepted_correct / len(accepted) if accepted else None,
            "review_count": len(indices) - len(accepted),
        }
    return result


def evaluate(split="test", output=None, library=None):
    output = Path(output) if output else ROOT / "results" / split
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data_manifest.csv").open(encoding="utf-8-sig", newline="") as file:
        records = sorted((r for r in csv.DictReader(file) if r["split"] == split), key=lambda r: r["file"])
    if not records:
        raise ValueError(f"没有 {split} 样本")
    start = time.perf_counter()
    recognizer = CharacterRecognizer(library)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    features = np.stack([character_feature(ROOT / record["file"])[1] for record in records])
    extraction_seconds = time.perf_counter() - start
    truth = [record["label"] for record in records]
    all_predictions = []
    constrained_predictions = []
    start = time.perf_counter()
    for first in range(0, len(records), 128):
        all_predictions.extend(recognizer.rank_features(features[first:first + 128]))
    match_seconds = time.perf_counter() - start
    constrained_predictions = [None] * len(records)
    for labels in (PROVINCES, DIGITS + LETTERS):
        indices = [i for i, label in enumerate(truth) if label in labels]
        for first in range(0, len(indices), 128):
            selected = indices[first:first + 128]
            predictions = recognizer.rank_features(features[selected], labels)
            for i, prediction in zip(selected, predictions):
                constrained_predictions[i] = prediction
    metrics = {
        "split": split, "count": len(records), "classes": len(set(truth)),
        "feature_version": recognizer.metadata["feature_version"],
        "unrestricted": group_metrics(truth, all_predictions),
        "type_constrained": group_metrics(truth, constrained_predictions),
        "constraint_definition": "province characters restricted to provinces; other characters restricted to digits+letters; assumes plate-position type is known",
        "review_thresholds": {"similarity": recognizer.review_similarity, "margin": recognizer.review_margin,
                              "meaning": "heuristics, not calibrated correctness probabilities"},
        "timing": {"library_load_seconds": load_seconds, "feature_extraction_seconds": extraction_seconds,
                   "batched_unrestricted_matching_seconds": match_seconds,
                   "mean_extract_and_batch_match_ms_per_char": (extraction_seconds + match_seconds) * 1000 / len(records),
                   "note": "includes image I/O and preprocessing; batch 128; not GUI latency"},
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__,
                        "numpy": np.__version__, "pillow": PIL.__version__, "platform": platform.platform()},
        "limitations": "single-character within-source holdout; not full-plate end-to-end accuracy or unseen-camera generalization",
    }
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "truth", "prediction", "correct", "top3", "similarity", "margin", "status",
                         "type_prediction", "type_correct"])
        for record, prediction, constrained in zip(records, all_predictions, constrained_predictions):
            writer.writerow([record["file"], record["label"], prediction["label"], int(prediction["label"] == record["label"]),
                             "|".join(c["label"] for c in prediction["candidates"]), prediction["similarity"],
                             prediction["margin"], prediction["status"], constrained["label"],
                             int(constrained["label"] == record["label"])])
    with (output / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "count", "correct", "recall", "type_correct", "type_recall"])
        for label in ALL_LABELS:
            indices = [i for i, item in enumerate(truth) if item == label]
            correct = sum(all_predictions[i]["label"] == label for i in indices)
            type_correct = sum(constrained_predictions[i]["label"] == label for i in indices)
            writer.writerow([label, len(indices), correct, correct / len(indices), type_correct, type_correct / len(indices)])
    matrix = np.zeros((len(ALL_LABELS), len(ALL_LABELS)), dtype=int)
    for label, prediction in zip(truth, all_predictions):
        matrix[ALL_LABELS.index(label), ALL_LABELS.index(prediction["label"])] += 1
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["truth/prediction", *ALL_LABELS])
        for label, row in zip(ALL_LABELS, matrix):
            writer.writerow([label, *row])
    errors = [i for i, p in enumerate(all_predictions) if p["label"] != truth[i]]
    confusions = Counter((truth[i], all_predictions[i]["label"]) for i in errors)
    metrics["common_confusions"] = [{"truth": a, "prediction": b, "count": n}
                                     for (a, b), n in confusions.most_common(15)]
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        from visualize_matches import save_failure_sheet, save_match_sheet
        save_failure_sheet([ROOT / records[i]["file"] for i in errors], [truth[i] for i in errors],
                           [all_predictions[i]["label"] for i in errors], output / "failure_cases.png")
        selected = errors[:6]
        save_match_sheet([ROOT / records[i]["file"] for i in selected], [all_predictions[i] for i in selected],
                         recognizer, output / "failure_top3.png", "错误样本的 Top3（真值：" + "、".join(truth[i] for i in selected) + "）")
    print(json.dumps({"split": split, "unrestricted": metrics["unrestricted"],
                      "type_constrained": metrics["type_constrained"], "timing": metrics["timing"]}, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    evaluate(args.split, args.output, args.library)

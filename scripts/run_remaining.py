"""Дозапуск незавершённых экспериментов и сбор таблицы."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import load_config, load_json, save_json  # noqa: E402


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def collect_results() -> list[dict]:
    rows = []
    for results_file in (ROOT / "runs").rglob("results.json"):
        data = load_json(results_file)
        metrics = data["metrics"]
        row = {
            "Задача": data["task"],
            "Модель": data["model"],
            "Input size": data["input_size"],
            "Эпохи": data["epochs"],
            "Время инференса, мс": round(data["inference_ms_per_image"], 2),
            "Размер модели, МБ": round(data["model_size_mb"], 2),
        }
        if data["task"] == "classification":
            row.update(
                {
                    "Accuracy": round(metrics["accuracy"], 4),
                    "Precision": round(metrics["precision"], 4),
                    "Recall": round(metrics["recall"], 4),
                    "F1": round(metrics["f1"], 4),
                }
            )
        else:
            row.update(
                {
                    "IoU": round(metrics["iou"], 4),
                    "Dice/F1": round(metrics["dice"], 4),
                    "Pixel Accuracy": round(metrics["pixel_accuracy"], 4),
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    cfg = load_config(ROOT / "configs" / "config.yaml")
    py = sys.executable

    remaining_cls = [m for m in cfg["classification_models"] if not (ROOT / "runs/classification" / m / "results.json").exists()]
    remaining_seg = [m for m in cfg["segmentation_models"] if not (ROOT / "runs/segmentation" / m / "results.json").exists()]

    for model in remaining_cls:
        run([py, "src/train_classifier.py", "--model", model])
    for model in remaining_seg:
        run([py, "src/train_segmenter.py", "--model", model])

    rows = collect_results()
    save_json(rows, ROOT / "runs/comparison_table.json")
    df = pd.DataFrame(rows)
    (ROOT / "report").mkdir(exist_ok=True)
    df.to_excel(ROOT / "report/comparison_table.xlsx", index=False)
    df.to_csv(ROOT / "report/comparison_table.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

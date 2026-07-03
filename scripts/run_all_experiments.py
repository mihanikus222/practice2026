"""Запуск полного цикла экспериментов."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, load_json, save_json


def run(cmd: list[str], cwd: Path) -> None:
    print(">", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def collect_results(runs_dir: Path) -> list[dict]:
    rows = []
    for results_file in runs_dir.rglob("results.json"):
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
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "configs" / "config.yaml")
    python = sys.executable

    run(
        [
            python,
            "data/generate_dataset.py",
            "--output",
            "data/raw",
            "--num-samples",
            str(cfg["num_samples"]),
            "--image-size",
            str(cfg["image_size"]),
            "--seed",
            str(cfg["seed"]),
        ],
        cwd=root,
    )

    for model in cfg["classification_models"]:
        run([python, "src/train_classifier.py", "--model", model], cwd=root)

    for model in cfg["segmentation_models"]:
        run([python, "src/train_segmenter.py", "--model", model], cwd=root)

    rows = collect_results(root / "runs")
    save_json(rows, root / "runs/comparison_table.json")
    df = pd.DataFrame(rows)
    df.to_excel(root / "report/comparison_table.xlsx", index=False)
    df.to_csv(root / "report/comparison_table.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

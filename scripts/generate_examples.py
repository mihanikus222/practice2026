"""Генерация примеров успешной и ошибочной работы модели."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import CrackInspectionSystem  # noqa: E402
from utils import load_json  # noqa: E402


def main() -> None:
    out_dir = ROOT / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    system = CrackInspectionSystem(ROOT / "configs" / "config.yaml")
    metadata = load_json(ROOT / "data" / "raw" / "metadata.json")

    results = []
    for item in metadata[:80]:
        image_path = ROOT / "data" / "raw" / "images" / item["file"]
        pred = system.predict(image_path)
        pred["gt_label"] = item["label"]
        pred["correct"] = (pred["has_crack"] and item["label"] == 1) or (
            not pred["has_crack"] and item["label"] == 0
        )
        results.append(pred)

    success = [r for r in results if r["correct"]][:3]
    failures = [r for r in results if not r["correct"]][:3]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, item in zip(axes[0], success):
        img = cv2.cvtColor(cv2.imread(item["overlay_path"]), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"Успех: {item['class_name']}\nуверенность {item['confidence']:.2f}")
        ax.axis("off")
    for ax, item in zip(axes[1], failures):
        img = cv2.cvtColor(cv2.imread(item["overlay_path"]), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"Ошибка: GT={item['gt_label']}, pred={int(item['has_crack'])}")
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "examples_success_failure.png", dpi=150)
    with (out_dir / "examples.json").open("w", encoding="utf-8") as fp:
        json.dump({"success": success, "failures": failures}, fp, ensure_ascii=False, indent=2)
    print(f"Сохранено: {out_dir}")


if __name__ == "__main__":
    main()

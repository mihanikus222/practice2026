"""Утилиты проекта."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_metadata(metadata: list[dict], seed: int, train_ratio: float, val_ratio: float):
    labels = [m["label"] for m in metadata]
    train_meta, temp_meta = train_test_split(
        metadata, test_size=1 - train_ratio, random_state=seed, stratify=labels
    )
    val_size = val_ratio / (1 - train_ratio)
    temp_labels = [m["label"] for m in temp_meta]
    val_meta, test_meta = train_test_split(
        temp_meta, test_size=1 - val_size, random_state=seed, stratify=temp_labels
    )
    return train_meta, val_meta, test_meta


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def segmentation_metrics(
    masks_true: np.ndarray, masks_pred: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    pred = (masks_pred >= threshold).astype(np.uint8)
    true = (masks_true >= 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    dice_num = 2 * intersection
    dice_den = pred.sum() + true.sum()
    iou = intersection / union if union > 0 else 1.0
    dice = dice_num / dice_den if dice_den > 0 else 1.0
    pixel_acc = (pred == true).mean()
    return {
        "iou": float(iou),
        "dice": float(dice),
        "pixel_accuracy": float(pixel_acc),
        "f1": float(dice),
    }


def measure_inference(model: torch.nn.Module, sample: torch.Tensor, device: torch.device, repeats: int = 20) -> float:
    model.eval()
    sample = sample.to(device)

    def _forward() -> None:
        out = model(sample)
        if isinstance(out, dict):
            _ = out["out"]

    with torch.no_grad():
        for _ in range(5):
            _forward()
        start = time.perf_counter()
        for _ in range(repeats):
            _forward()
        elapsed = time.perf_counter() - start
    return elapsed / repeats


def model_size_mb(model: torch.nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters())
    return total * 4 / (1024 ** 2)

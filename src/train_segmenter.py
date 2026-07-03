"""Обучение и оценка моделей сегментации."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CrackSegmentationDataset
from models.segmenters import create_segmenter
from utils import (
    get_device,
    load_config,
    load_json,
    measure_inference,
    model_size_mb,
    save_json,
    segmentation_metrics,
    set_seed,
    split_metadata,
)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs["out"] if isinstance(outputs, dict) else outputs
        loss = dice_loss(logits, masks) + nn.functional.binary_cross_entropy_with_logits(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for images, masks in loader:
        images = images.to(device)
        outputs = model(images)
        logits = outputs["out"] if isinstance(outputs, dict) else outputs
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
        targets.append(masks.numpy())
    preds_arr = np.concatenate(preds, axis=0)
    targets_arr = np.concatenate(targets, axis=0)
    return segmentation_metrics(targets_arr, preds_arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    metadata = load_json(args.data / "metadata.json")
    train_meta, val_meta, test_meta = split_metadata(
        metadata, cfg["seed"], cfg["train_ratio"], cfg["val_ratio"]
    )

    train_loader = DataLoader(
        CrackSegmentationDataset(train_meta, args.data, augment=True),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        CrackSegmentationDataset(val_meta, args.data, augment=False),
        batch_size=cfg["batch_size"],
    )
    test_loader = DataLoader(
        CrackSegmentationDataset(test_meta, args.data, augment=False),
        batch_size=cfg["batch_size"],
    )

    model = create_segmenter(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    history = []
    best_dice = -1.0
    run_dir = Path("runs/segmentation") / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["epochs_segmentation"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), run_dir / "best.pt")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    sample, _ = next(iter(test_loader))
    inference_ms = measure_inference(model, sample[:1], device) * 1000

    result = {
        "task": "segmentation",
        "model": args.model,
        "input_size": cfg["image_size"],
        "epochs": cfg["epochs_segmentation"],
        "learning_rate": cfg["learning_rate"],
        "optimizer": "Adam",
        "batch_size": cfg["batch_size"],
        "metrics": test_metrics,
        "inference_ms_per_image": inference_ms,
        "model_size_mb": model_size_mb(model),
        "history": history,
    }
    save_json(result, run_dir / "results.json")
    print(f"{args.model}: Dice={test_metrics['dice']:.4f}, IoU={test_metrics['iou']:.4f}")


if __name__ == "__main__":
    main()

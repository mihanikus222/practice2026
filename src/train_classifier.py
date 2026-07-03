"""Обучение и оценка классификационных моделей."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CrackClassificationDataset
from models.classifiers import create_classifier
from utils import (
    classification_metrics,
    get_device,
    load_config,
    load_json,
    measure_inference,
    model_size_mb,
    save_json,
    set_seed,
    split_metadata,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        y_true.extend(labels.tolist())
        y_pred.extend(preds)
    return classification_metrics(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    input_size = 224 if args.model.lower() == "vit_b_16" else cfg["image_size"]

    metadata = load_json(args.data / "metadata.json")
    train_meta, val_meta, test_meta = split_metadata(
        metadata, cfg["seed"], cfg["train_ratio"], cfg["val_ratio"]
    )

    train_loader = DataLoader(
        CrackClassificationDataset(train_meta, args.data, augment=True, image_size=input_size),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        CrackClassificationDataset(val_meta, args.data, augment=False, image_size=input_size),
        batch_size=cfg["batch_size"],
    )
    test_loader = DataLoader(
        CrackClassificationDataset(test_meta, args.data, augment=False, image_size=input_size),
        batch_size=cfg["batch_size"],
    )

    model = create_classifier(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    history = []
    best_f1 = -1.0
    run_dir = Path("runs/classification") / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["epochs_classification"] + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"}})
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), run_dir / "best.pt")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    sample, _ = next(iter(test_loader))
    inference_ms = measure_inference(model, sample[:1], device) * 1000

    result = {
        "task": "classification",
        "model": args.model,
        "input_size": input_size,
        "epochs": cfg["epochs_classification"],
        "learning_rate": cfg["learning_rate"],
        "optimizer": "Adam",
        "batch_size": cfg["batch_size"],
        "metrics": test_metrics,
        "inference_ms_per_image": inference_ms,
        "model_size_mb": model_size_mb(model),
        "history": history,
    }
    save_json(result, run_dir / "results.json")
    print(f"{args.model}: F1={test_metrics['f1']:.4f}, acc={test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

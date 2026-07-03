"""Демонстрационный модуль инференса и визуализации."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from models.classifiers import create_classifier
from models.segmenters import create_segmenter
from utils import get_device, load_config, save_json

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_classification(image_bgr: np.ndarray, size: int = 256) -> torch.Tensor:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(image).unsqueeze(0)


def preprocess_segmentation(image_bgr: np.ndarray, size: int = 256) -> torch.Tensor:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(tensor).unsqueeze(0)


def crack_area_percent(mask: np.ndarray) -> float:
    return float((mask > 0.5).mean() * 100)


class CrackInspectionSystem:
    def __init__(self, config_path: Path = Path("configs/config.yaml")):
        self.cfg = load_config(config_path)
        self.device = get_device(self.cfg["device"])
        self.history_path = Path("runs/history.json")
        self.history = []
        if self.history_path.exists():
            with self.history_path.open("r", encoding="utf-8") as fp:
                self.history = json.load(fp)

        self.classifier = create_classifier("efficientnet_b0", num_classes=2).to(self.device)
        cls_weights = Path("runs/classification/efficientnet_b0/best.pt")
        if cls_weights.exists():
            self.classifier.load_state_dict(torch.load(cls_weights, map_location=self.device))

        self.segmenter = create_segmenter("deeplabv3_resnet50").to(self.device)
        seg_weights = Path("runs/segmentation/deeplabv3_resnet50/best.pt")
        if seg_weights.exists():
            self.segmenter.load_state_dict(torch.load(seg_weights, map_location=self.device))

        self.classifier.eval()
        self.segmenter.eval()

    def predict(self, image_path: Path) -> dict:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)

        with torch.no_grad():
            cls_input = preprocess_classification(image).to(self.device)
            cls_logits = self.classifier(cls_input)
            probs = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
            label = int(probs.argmax())
            confidence = float(probs[label])

            seg_input = preprocess_segmentation(image).to(self.device)
            seg_out = self.segmenter(seg_input)
            logits = seg_out["out"] if isinstance(seg_out, dict) else seg_out
            mask = torch.sigmoid(logits)[0, 0].cpu().numpy()

        area = crack_area_percent(mask)
        result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image": str(image_path),
            "has_crack": label == 1,
            "class_name": "трещина обнаружена" if label == 1 else "трещина не обнаружена",
            "confidence": confidence,
            "crack_area_percent": area,
            "mask_path": None,
        }

        out_dir = Path("runs/demo_outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / f"{image_path.stem}_mask.png"
        overlay_path = out_dir / f"{image_path.stem}_overlay.png"

        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        cv2.imwrite(str(mask_path), mask_u8)

        resized = cv2.resize(image, (256, 256))
        overlay = resized.copy()
        overlay[mask_u8 > 0] = (0, 0, 255)
        blended = cv2.addWeighted(resized, 0.7, overlay, 0.3, 0)
        cv2.imwrite(str(overlay_path), blended)

        result["mask_path"] = str(mask_path)
        result["overlay_path"] = str(overlay_path)
        self.history.append(result)
        save_json(self.history, self.history_path)
        return result

    def statistics(self) -> dict:
        if not self.history:
            return {"total": 0}
        cracks = sum(1 for r in self.history if r["has_crack"])
        return {
            "total": len(self.history),
            "with_crack": cracks,
            "without_crack": len(self.history) - cracks,
            "avg_confidence": float(np.mean([r["confidence"] for r in self.history])),
            "avg_crack_area_percent": float(np.mean([r["crack_area_percent"] for r in self.history if r["has_crack"]] or [0])),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    system = CrackInspectionSystem()
    result = system.predict(args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(system.statistics(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

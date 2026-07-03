"""Датасеты для классификации и сегментации."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CrackClassificationDataset(Dataset):
    def __init__(self, metadata: list[dict], root: Path, augment: bool = False, image_size: int = 256):
        self.metadata = metadata
        self.root = root
        self.image_size = image_size
        base = [
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        else:
            self.transform = transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        item = self.metadata[idx]
        image = cv2.imread(str(self.root / "images" / item["file"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image)
        label = torch.tensor(item["label"], dtype=torch.long)
        return tensor, label


class CrackSegmentationDataset(Dataset):
    def __init__(self, metadata: list[dict], root: Path, augment: bool = False):
        self.metadata = [m for m in metadata if m["label"] == 1]
        self.root = root
        self.augment = augment

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        item = self.metadata[idx]
        image = cv2.imread(str(self.root / "images" / item["file"]))
        mask = cv2.imread(str(self.root / "masks" / item["file"]), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        if self.augment and np.random.rand() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(image)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask

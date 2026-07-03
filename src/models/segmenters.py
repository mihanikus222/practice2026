"""Фабрика моделей сегментации."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(128 + 64, 64)
        self.dec1 = DoubleConv(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out(d1)


def create_segmenter(name: str) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNetSmall()
    if name == "unet_small":
        return UNetSmall()
    if name == "deeplabv3_resnet50":
        model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        )
        model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
        return model
    if name == "deeplabv3_mobilenet":
        model = models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
        model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
        return model
    if name == "fcn_resnet50":
        model = models.segmentation.fcn_resnet50(
            weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT
        )
        model.classifier[-1] = nn.Conv2d(512, 1, kernel_size=1)
        return model
    raise ValueError(f"Неизвестная модель сегментации: {name}")

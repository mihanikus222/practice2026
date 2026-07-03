"""Генерация синтетического датасета трещин на бетонных поверхностях."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def _concrete_texture(size: int, noise_level: float) -> np.ndarray:
    base = np.random.normal(145, 25, (size, size)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=3)
    noise = np.random.normal(0, noise_level, (size, size)).astype(np.float32)
    texture = np.clip(base + noise, 60, 220)
    return texture.astype(np.uint8)


def _draw_crack(mask: np.ndarray, image: np.ndarray, thickness: int) -> None:
    h, w = mask.shape
    points = [(random.randint(0, w - 1), random.randint(0, h - 1))]
    steps = random.randint(12, 40)
    for _ in range(steps):
        x, y = points[-1]
        x = int(np.clip(x + random.randint(-18, 18), 0, w - 1))
        y = int(np.clip(y + random.randint(-18, 18), 0, h - 1))
        points.append((x, y))
    for i in range(len(points) - 1):
        cv2.line(mask, points[i], points[i + 1], 255, thickness=thickness)
        cv2.line(image, points[i], points[i + 1], 35, thickness=max(1, thickness - 1))


def generate_sample(size: int, with_crack: bool) -> tuple[np.ndarray, np.ndarray]:
    noise = random.uniform(8, 22)
    gray = _concrete_texture(size, noise)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = np.zeros((size, size), dtype=np.uint8)

    if with_crack:
        thickness = random.choice([1, 1, 2, 2, 3])
        _draw_crack(mask, image, thickness)
        if random.random() < 0.25:
            _draw_crack(mask, image, max(1, thickness - 1))

    if random.random() < 0.35:
        speckle = np.random.randint(0, 40, (size, size), dtype=np.uint8)
        image = cv2.addWeighted(image, 0.9, cv2.cvtColor(speckle, cv2.COLOR_GRAY2BGR), 0.1, 0)

    return image, mask


def generate_dataset(output_dir: Path, num_samples: int, image_size: int, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    metadata = []
    for idx in range(num_samples):
        with_crack = idx % 2 == 0
        image, mask = generate_sample(image_size, with_crack)
        name = f"sample_{idx:04d}.png"
        cv2.imwrite(str(images_dir / name), image)
        cv2.imwrite(str(masks_dir / name), mask)
        metadata.append(
            {
                "file": name,
                "label": 1 if with_crack else 0,
                "crack_pixels": int(mask.sum() // 255),
            }
        )

    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    stats = {
        "total": num_samples,
        "with_crack": sum(1 for m in metadata if m["label"] == 1),
        "without_crack": sum(1 for m in metadata if m["label"] == 0),
        "image_size": image_size,
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = generate_dataset(args.output, args.num_samples, args.image_size, args.seed)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

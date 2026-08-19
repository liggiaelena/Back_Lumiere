"""Create leakage-safe synthetic vitiligo faces for training only.

Human-labelled validation and test samples are copied unchanged.  Training
masks are sampled from human-labelled face crops, mildly transformed in the
same face coordinate system, and used to depigment unrelated normal faces.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def find_image(directory: Path, item: str) -> Path:
    candidate = directory / item
    if candidate.exists():
        return candidate
    stem = Path(item).stem
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(item)


def read_split(dataset: Path, split: str) -> list[str]:
    return [line.strip() for line in (dataset / "splits" / f"{split}.txt").read_text().splitlines() if line.strip()]


def transformed_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = mask.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rng.uniform(-8, 8), rng.uniform(0.88, 1.12))
    matrix[:, 2] += (rng.uniform(-0.04, 0.04) * width, rng.uniform(-0.04, 0.04) * height)
    result = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    kernel_size = rng.choice((1, 3, 5))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.dilate(result, kernel) if rng.random() < 0.5 else cv2.erode(result, kernel)
    return result > 0


def depigment(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    target_lab = lab.copy()
    target_lab[..., 0] = np.clip(target_lab[..., 0] + rng.uniform(20, 48), 0, 255)
    target_lab[..., 1] = 128 + (target_lab[..., 1] - 128) * rng.uniform(0.25, 0.65)
    target_lab[..., 2] = 128 + (target_lab[..., 2] - 128) * rng.uniform(0.25, 0.65)
    pale = cv2.cvtColor(target_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    # Preserve some local texture and avoid synthetic pure-white patches.
    pale_hsv = cv2.cvtColor(pale.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    pale_hsv[..., 1] = np.minimum(pale_hsv[..., 1], hsv[..., 1] * rng.uniform(0.25, 0.65))
    pale = cv2.cvtColor(pale_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    feather = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=rng.uniform(1.2, 2.8))
    feather = np.clip(feather * rng.uniform(0.72, 0.94), 0, 1)[..., None]
    return np.clip(image.astype(np.float32) * (1 - feather) + pale * feather, 0, 255).astype(np.uint8)


def prepare(source: Path, normal: Path, output: Path, count: int, seed: int) -> dict:
    rng = random.Random(seed)
    for name in ("images", "masks", "splits"):
        (output / name).mkdir(parents=True, exist_ok=True)

    split_items: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        items = read_split(source, split)
        split_items[split] = []
        for item in items:
            image_path = find_image(source / "images", item)
            stem = Path(item).stem
            destination_name = f"human_{stem}{image_path.suffix.lower()}"
            shutil.copy2(image_path, output / "images" / destination_name)
            shutil.copy2(source / "masks" / f"{stem}.png", output / "masks" / f"human_{stem}.png")
            split_items[split].append(destination_name)

    donors = read_split(source, "train")
    normal_items = read_split(normal, "train")
    areas: list[float] = []
    made = 0
    attempts = 0
    while made < count and attempts < count * 10:
        attempts += 1
        donor = rng.choice(donors)
        target = rng.choice(normal_items)
        donor_mask = cv2.imread(str(source / "masks" / f"{Path(donor).stem}.png"), cv2.IMREAD_GRAYSCALE)
        target_path = find_image(normal / "images", target)
        image = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if donor_mask is None or image is None:
            continue
        donor_mask = cv2.resize(donor_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = transformed_mask(donor_mask, rng)
        area = float(mask.mean())
        if not 0.002 <= area <= 0.35:
            continue
        synthetic = depigment(image, mask, rng)
        name = f"synthetic_{made:05d}.jpg"
        cv2.imwrite(str(output / "images" / name), synthetic, [cv2.IMWRITE_JPEG_QUALITY, 94])
        cv2.imwrite(str(output / "masks" / f"synthetic_{made:05d}.png"), mask.astype(np.uint8) * 255)
        split_items["train"].append(name)
        areas.append(area)
        made += 1

    for split, items in split_items.items():
        (output / "splits" / f"{split}.txt").write_text("\n".join(items) + "\n")
    report = {
        "seed": seed,
        "human": {split: len(read_split(source, split)) for split in ("train", "val", "test")},
        "synthetic_train": made,
        "mask_area_median": float(np.median(areas)) if areas else 0.0,
        "mask_area_p95": float(np.percentile(areas, 95)) if areas else 0.0,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.normal, args.output, args.count, args.seed), indent=2))


if __name__ == "__main__":
    main()

"""Evaluate disease-specific SegFormer checkpoints without empty-mask inflation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import SegformerForSemanticSegmentation


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def find_image(image_dir: Path, item: str) -> Path:
    path = Path(item)
    if path.suffix.lower() in EXTENSIONS and (image_dir / path.name).exists():
        return image_dir / path.name
    stem = path.stem if path.suffix.lower() in EXTENSIONS else item
    for extension in EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found: {item} in {image_dir}")


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def filter_components(prediction: np.ndarray, minimum_percent: float) -> np.ndarray:
    if minimum_percent <= 0 or not prediction.any():
        return prediction
    count, labels, stats, _ = cv2.connectedComponentsWithStats(prediction.astype(np.uint8), 8)
    minimum_pixels = prediction.size * minimum_percent / 100.0
    output = np.zeros_like(prediction, dtype=bool)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum_pixels:
            output[labels == index] = True
    return output


@torch.inference_mode()
def evaluate(checkpoint: Path, target_label: str, target_dataset: str, datasets: dict[str, Path], image_size: int, split: str, erosion_kernel: int, minimum_component_percent: float, lightness_percentile: float, saturation_percentile: float, device: str = "cpu") -> dict:
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint).to(device).eval()
    labels = {int(key): value for key, value in model.config.id2label.items()}
    target_ids = [idx for idx, label in labels.items() if label == target_label]
    if not target_ids:
        raise ValueError(f"{target_label!r} not present in {labels}")
    target_id = target_ids[0]

    thresholds = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
    totals = {threshold: {"tp": 0, "fp": 0, "fn": 0, "pixels": 0, "predicted": 0} for threshold in thresholds}
    per_dataset = {}
    for disease, dataset_dir in datasets.items():
        items = [line.strip() for line in (dataset_dir / "splits" / f"{split}.txt").read_text().splitlines() if line.strip()]
        stats_by_threshold = {threshold: {"tp": 0, "fp": 0, "fn": 0, "pixels": 0, "predicted": 0, "images": len(items)} for threshold in thresholds}
        for item in items:
            image_path = find_image(dataset_dir / "images", item)
            item_path = Path(item)
            stem = item_path.stem if item_path.suffix.lower() in EXTENSIONS else item
            mask_path = dataset_dir / "masks" / f"{stem}.png"
            image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
            pixel_values = TF.normalize(TF.to_tensor(image), MEAN, STD).unsqueeze(0).to(device)
            logits = model(pixel_values=pixel_values).logits
            logits = F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=False)
            probabilities = logits.softmax(1)
            argmax = logits.argmax(1).squeeze(0).cpu().numpy() == target_id
            target_probability = probabilities[0, target_id].cpu().numpy()
            image_rgb = np.asarray(image)
            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            color_gate = np.ones((image_size, image_size), dtype=bool)
            if lightness_percentile >= 0:
                color_gate &= lab[:, :, 0] >= np.percentile(lab[:, :, 0], lightness_percentile)
            if saturation_percentile >= 0:
                color_gate &= hsv[:, :, 1] <= np.percentile(hsv[:, :, 1], saturation_percentile)
            binary_model = int(model.config.num_labels) == 2
            truth = (np.asarray(mask) > 0) if disease == target_dataset else np.zeros_like(argmax)
            for threshold in thresholds:
                prediction = target_probability >= threshold
                if not binary_model:
                    prediction &= argmax
                prediction &= color_gate
                if erosion_kernel > 1:
                    kernel = np.ones((erosion_kernel, erosion_kernel), dtype=np.uint8)
                    prediction = cv2.erode(prediction.astype(np.uint8), kernel, iterations=1).astype(bool)
                prediction = filter_components(prediction, minimum_component_percent)
                tp = int(np.logical_and(prediction, truth).sum())
                fp = int(np.logical_and(prediction, ~truth).sum())
                fn = int(np.logical_and(~prediction, truth).sum())
                for bucket in (stats_by_threshold[threshold], totals[threshold]):
                    bucket["tp"] += tp
                    bucket["fp"] += fp
                    bucket["fn"] += fn
                    bucket["pixels"] += prediction.size
                    bucket["predicted"] += int(prediction.sum())
        per_dataset[disease] = {str(threshold): summarize(stats) for threshold, stats in stats_by_threshold.items()}
    return {"checkpoint": str(checkpoint), "target_label": target_label, "class_id": target_id,
            "overall": {str(threshold): summarize(stats) for threshold, stats in totals.items()}, "datasets": per_dataset}


def summarize(stats: dict) -> dict:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    result = dict(stats)
    result.update({
        "iou": safe_ratio(tp, tp + fp + fn),
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "predicted_area_percent": 100 * safe_ratio(stats["predicted"], stats["pixels"]),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--melasma-dataset", type=Path)
    parser.add_argument("--vitiligo-dataset", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--erosion-kernel", type=int, default=0)
    parser.add_argument("--minimum-component-percent", type=float, default=0.0)
    parser.add_argument("--lightness-percentile", type=float, default=-1)
    parser.add_argument("--saturation-percentile", type=float, default=-1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    datasets = {
        "melasma_like_hyperpigmentation": args.dataset_root / "melasma",
        "vitiligo": args.dataset_root / "vitiligo",
        "port_wine_stain": args.dataset_root / "port_wine_stain" / "processed",
    }
    if args.vitiligo_dataset is not None:
        datasets["vitiligo"] = args.vitiligo_dataset
    if args.melasma_dataset is not None:
        datasets["melasma_like_hyperpigmentation"] = args.melasma_dataset
    result = evaluate(args.checkpoint, args.target_label, args.target_dataset, datasets, args.image_size, args.split, args.erosion_kernel, args.minimum_component_percent, args.lightness_percentile, args.saturation_percentile, args.device)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()

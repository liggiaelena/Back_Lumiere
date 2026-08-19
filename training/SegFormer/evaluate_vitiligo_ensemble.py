"""Select and test a two-checkpoint vitiligo probability ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import SegformerForSemanticSegmentation

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def find_image(root: Path, item: str) -> Path:
    path = Path(item)
    if path.is_absolute() and path.exists():
        return path
    if path.suffix.lower() in EXTENSIONS and (root / "images" / path.name).exists():
        return root / "images" / path.name
    stem = path.stem if path.suffix.lower() in EXTENSIONS else item
    for extension in EXTENSIONS:
        candidate = root / "images" / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(item)


def probability(model, tensor, size):
    logits = F.interpolate(model(pixel_values=tensor).logits, (size, size), mode="bilinear", align_corners=False)
    return logits.softmax(1)[0, 1].cpu().numpy()


def components(mask: np.ndarray, minimum_percent: float) -> np.ndarray:
    if minimum_percent <= 0 or not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    output = np.zeros_like(mask, dtype=bool)
    minimum = mask.size * minimum_percent / 100.0
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum:
            output[labels == index] = True
    return output


@torch.inference_mode()
def collect(models, datasets, split, size, device):
    rows = []
    for group, root in datasets.items():
        items = [x.strip() for x in (root / "splits" / f"{split}.txt").read_text().splitlines() if x.strip()]
        for item in items:
            path = find_image(root, item)
            image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            tensor = TF.normalize(TF.to_tensor(image), MEAN, STD).unsqueeze(0).to(device)
            probs = [probability(model, tensor, size) for model in models]
            stem = Path(item).stem if Path(item).suffix.lower() in EXTENSIONS else item
            if group == "vitiligo":
                truth = np.asarray(Image.open(root / "masks" / f"{stem}.png").convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
            else:
                truth = np.zeros((size, size), dtype=bool)
            rows.append((group, probs, truth))
    return rows


def summarize(rows, mode, threshold, minimum_component):
    counts = {"tp": 0, "fp": 0, "fn": 0, "positive_images": 0, "detected_positive_images": 0,
              "negative_images": 0, "false_positive_images": 0}
    for _, probs, truth in rows:
        first, second = probs
        combined = {
            "baseline": first,
            "mean": (first + second) / 2,
            "geometric": np.sqrt(first * second),
            "product": first * second,
            "minimum": np.minimum(first, second),
            "patch_confirmed": first * (0.5 + 0.5 * second),
        }[mode]
        prediction = components(combined >= threshold, minimum_component)
        counts["tp"] += int((prediction & truth).sum())
        counts["fp"] += int((prediction & ~truth).sum())
        counts["fn"] += int((~prediction & truth).sum())
        positive = bool(truth.any()); detected = bool(prediction.any())
        counts["positive_images"] += int(positive)
        counts["detected_positive_images"] += int(positive and detected)
        counts["negative_images"] += int(not positive)
        counts["false_positive_images"] += int(not positive and detected)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {**counts, "iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "image_recall": counts["detected_positive_images"] / max(counts["positive_images"], 1),
            "image_false_positive_rate": counts["false_positive_images"] / max(counts["negative_images"], 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--vitiligo", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--melasma", type=Path, required=True)
    parser.add_argument("--port-wine-stain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    models = [SegformerForSemanticSegmentation.from_pretrained(path).to(args.device).eval() for path in (args.baseline, args.patch)]
    datasets = {"vitiligo": args.vitiligo, "normal": args.normal, "melasma": args.melasma, "wine_stain": args.port_wine_stain}
    validation = collect(models, datasets, "val", args.image_size, args.device)
    candidates = []
    for mode in ("baseline", "mean", "geometric", "product", "minimum", "patch_confirmed"):
        for component in (0.0, 0.025, 0.05, 0.1):
            for threshold in np.arange(0.05, 0.951, 0.025):
                result = summarize(validation, mode, float(threshold), component)
                candidates.append({"mode": mode, "threshold": float(threshold), "minimum_component_percent": component, **result})
    eligible = [x for x in candidates if x["precision"] >= .60 and x["recall"] >= .50 and x["image_recall"] >= .90 and x["image_false_positive_rate"] <= .02]
    selected = max(eligible or candidates, key=lambda x: (x["iou"], x["precision"], x["recall"]))
    test = collect(models, datasets, "test", args.image_size, args.device)
    test_result = summarize(test, selected["mode"], selected["threshold"], selected["minimum_component_percent"])
    report = {"validation_selected": selected, "validation_gate_met": bool(eligible), "test": test_result,
              "baseline": str(args.baseline), "patch": str(args.patch), "image_size": args.image_size}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

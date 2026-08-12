"""Train one disease-vs-rest SegFormer with cross-disease negative samples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import SegformerConfig, SegformerForSemanticSegmentation

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DISEASE_DATASET_NAMES = ("melasma", "vitiligo", "port_wine_stain")


def dataset_paths(root: Path) -> dict[str, Path]:
    return {
        "melasma": root / "melasma",
        "vitiligo": root / "vitiligo",
        "port_wine_stain": root / "port_wine_stain" / "processed",
    }


def validate_mask_integrity(dataset_dir: Path) -> None:
    """Reject masks with scanline corruption before starting expensive training."""
    ratios, maximum_transitions = [], []
    for path in (dataset_dir / "masks").glob("*.png"):
        mask = np.asarray(Image.open(path).convert("L")) > 0
        if min(mask.shape) < 2:
            continue
        horizontal = float(np.mean(mask[:, 1:] != mask[:, :-1]))
        vertical = float(np.mean(mask[1:] != mask[:-1]))
        lower = min(horizontal, vertical)
        upper = max(horizontal, vertical)
        if upper > 0:
            ratios.append(upper / (lower + 1e-6))
            maximum_transitions.append(upper)
    if not ratios:
        raise ValueError(f"No non-empty masks found in {dataset_dir}")
    median_ratio = float(np.median(ratios))
    median_transition = float(np.median(maximum_transitions))
    if median_ratio > 10 and median_transition > 0.10:
        raise ValueError(
            "Mask integrity check failed: likely scanline/reshape corruption in "
            f"{dataset_dir} (median directional ratio={median_ratio:.2f}, "
            f"transition={median_transition:.3f}). Regenerate masks from source annotations."
        )


class OneVsRestDataset(Dataset):
    def __init__(self, datasets: dict[str, Path], split: str, target: str, size: int, augment: bool, lesion_crop_probability: float = 0.0, low_contrast_probability: float = 0.0):
        self.datasets = datasets
        self.target = target
        self.size = size
        self.augment = augment
        self.lesion_crop_probability = lesion_crop_probability
        self.low_contrast_probability = low_contrast_probability
        self.jitter = transforms.ColorJitter(0.15, 0.15, 0.1, 0.04)
        self.items = []
        self.groups = []
        for group, root in datasets.items():
            names = [line.strip() for line in (root / "splits" / f"{split}.txt").read_text().splitlines() if line.strip()]
            self.items.extend((group, root, name) for name in names)
            self.groups.extend([group] * len(names))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def stem(name: str) -> str:
        path = Path(name)
        return path.stem if path.suffix.lower() in EXTENSIONS else name

    @staticmethod
    def image_path(root: Path, name: str) -> Path:
        path = Path(name)
        if path.is_absolute() and path.exists():
            return path
        if path.suffix.lower() in EXTENSIONS and (root / "images" / path.name).exists():
            return root / "images" / path.name
        stem = OneVsRestDataset.stem(name)
        for extension in EXTENSIONS:
            candidate = root / "images" / f"{stem}{extension}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(name)

    def __getitem__(self, index):
        group, root, name = self.items[index]
        image = Image.open(self.image_path(root, name)).convert("RGB")
        if group == self.target:
            mask = Image.open(root / "masks" / f"{self.stem(name)}.png").convert("L")
            if self.augment and random.random() < self.lesion_crop_probability:
                image, mask = self._lesion_crop(image, mask)
        else:
            mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        label = Image.fromarray((np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8))
        if self.augment:
            if random.random() < 0.5:
                image, label = TF.hflip(image), TF.hflip(label)
            image = self.jitter(image)
            # Diffuse melasma is often photographed with weak local contrast,
            # uneven illumination, JPEG softness, or makeup. Simulating those
            # conditions reduces the all-or-nothing misses seen in production.
            if random.random() < self.low_contrast_probability:
                image = TF.adjust_contrast(image, random.uniform(0.55, 0.85))
                image = TF.adjust_gamma(image, random.uniform(0.8, 1.25))
                if random.random() < 0.5:
                    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        return TF.normalize(TF.to_tensor(image), MEAN, STD), torch.from_numpy(np.asarray(label, dtype=np.int64).copy()), group

    @staticmethod
    def _lesion_crop(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        binary = np.asarray(mask) > 0
        ys, xs = np.where(binary)
        if not len(xs):
            return image, mask
        lesion_width = int(xs.max() - xs.min() + 1)
        lesion_height = int(ys.max() - ys.min() + 1)
        minimum = max(lesion_width, lesion_height, int(min(image.size) * 0.35))
        crop_size = min(min(image.size), int(minimum * random.uniform(1.2, 2.0)))
        center_x = int((xs.min() + xs.max()) / 2 + random.uniform(-0.1, 0.1) * crop_size)
        center_y = int((ys.min() + ys.max()) / 2 + random.uniform(-0.1, 0.1) * crop_size)
        left = max(0, min(image.width - crop_size, center_x - crop_size // 2))
        top = max(0, min(image.height - crop_size, center_y - crop_size // 2))
        box = (left, top, left + crop_size, top + crop_size)
        return image.crop(box), mask.crop(box)

    def balanced_weights(self, positive_ratio: float):
        dataset_names = tuple(dict.fromkeys(self.groups))
        counts = {name: self.groups.count(name) for name in dataset_names}
        negative_names = [name for name in dataset_names if name != self.target]
        if not negative_names:
            raise ValueError("At least one negative dataset is required")
        negative_ratio = (1.0 - positive_ratio) / len(negative_names)
        ratios = {
            name: (positive_ratio if name == self.target else negative_ratio)
            for name in dataset_names
        }
        return [ratios[group] / counts[group] for group in self.groups]


def loss_fn(logits, labels, positive_weight, dice_weight, focal_gamma):
    weights = logits.new_tensor([1.0, positive_weight])
    ce_pixels = F.cross_entropy(logits, labels, weight=weights, reduction="none")
    ce = (((1.0 - torch.exp(-ce_pixels)) ** focal_gamma) * ce_pixels).mean() if focal_gamma > 0 else ce_pixels.mean()
    probability = logits.softmax(1)[:, 1]
    target = labels.float()
    intersection = (probability * target).sum()
    dice = 1 - (2 * intersection + 1) / (probability.sum() + target.sum() + 1)
    return ce + dice_weight * dice


@torch.inference_mode()
def evaluate(model, loader, device, thresholds):
    counts = {threshold: {"tp": 0, "fp": 0, "fn": 0, "positive_images": 0, "detected_positive_images": 0, "negative_images": 0, "false_positive_images": 0} for threshold in thresholds}
    model.eval()
    for pixels, labels, _ in loader:
        pixels, labels = pixels.to(device), labels.to(device)
        logits = F.interpolate(model(pixel_values=pixels).logits, labels.shape[-2:], mode="bilinear", align_corners=False)
        probabilities = logits.softmax(1)[:, 1]
        truth = labels == 1
        for threshold in thresholds:
            prediction = probabilities >= threshold
            counts[threshold]["tp"] += int((prediction & truth).sum())
            counts[threshold]["fp"] += int((prediction & ~truth).sum())
            counts[threshold]["fn"] += int((~prediction & truth).sum())
            positive_images = truth.flatten(1).any(1)
            predicted_images = prediction.flatten(1).any(1)
            counts[threshold]["positive_images"] += int(positive_images.sum())
            counts[threshold]["detected_positive_images"] += int((positive_images & predicted_images).sum())
            counts[threshold]["negative_images"] += int((~positive_images).sum())
            counts[threshold]["false_positive_images"] += int(((~positive_images) & predicted_images).sum())
    metrics = {}
    for threshold, value in counts.items():
        tp, fp, fn = value["tp"], value["fp"], value["fn"]
        metrics[str(threshold)] = {
            **value,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "image_recall": value["detected_positive_images"] / value["positive_images"] if value["positive_images"] else 0.0,
            "image_false_positive_rate": value["false_positive_images"] / value["negative_images"] if value["negative_images"] else 0.0,
        }
    return metrics


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    paths = dataset_paths(args.dataset_root)
    if args.melasma_dataset is not None:
        paths["melasma"] = args.melasma_dataset
    if args.vitiligo_dataset is not None:
        paths["vitiligo"] = args.vitiligo_dataset
    if args.normal_dataset is not None:
        paths["normal"] = args.normal_dataset
    validate_mask_integrity(paths[args.target])
    train_set = OneVsRestDataset(paths, "train", args.target, args.image_size, True, args.lesion_crop_probability, args.low_contrast_probability)
    val_set = OneVsRestDataset(paths, "val", args.target, args.image_size, False)
    sampler = WeightedRandomSampler(train_set.balanced_weights(args.positive_ratio), args.samples_per_epoch, replacement=True)
    train_loader = DataLoader(train_set, args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    config = SegformerConfig.from_pretrained(args.checkpoint)
    config.num_labels = 2
    config.image_size = args.image_size
    config.id2label = {0: "background", 1: args.target}
    config.label2id = {"background": 0, args.target: 1}
    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint, config=config, ignore_mismatched_sizes=True).to(device)
    if args.reset_classifier:
        torch.nn.init.normal_(
            model.decode_head.classifier.weight,
            mean=0.0,
            std=float(config.initializer_range),
        )
        if model.decode_head.classifier.bias is not None:
            torch.nn.init.zeros_(model.decode_head.classifier.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate / 20)
    thresholds = tuple(args.thresholds)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    history, best_score, stale = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for pixels, labels, _ in train_loader:
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = F.interpolate(model(pixel_values=pixels).logits, labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss_fn(logits, labels, args.positive_weight, args.dice_weight, args.focal_gamma)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        metrics = evaluate(model, val_loader, device, thresholds)
        eligible = [(float(t), m) for t, m in metrics.items() if m["precision"] >= args.min_precision and m["recall"] >= args.min_recall and m["image_recall"] >= args.min_image_recall]
        if eligible:
            threshold, selected = max(eligible, key=lambda item: item[1]["iou"])
        else:
            threshold, selected = max(((float(t), m) for t, m in metrics.items()), key=lambda item: item[1]["iou"])
        meets_deployment_gate = (
            selected["precision"] >= args.min_precision
            and selected["recall"] >= args.min_recall
            and selected["image_recall"] >= args.min_image_recall
        )
        # Never let a higher pixel IoU overwrite a deployable checkpoint when
        # it completely misses too many positive images.
        score = selected["iou"] if meets_deployment_gate else selected["iou"] * 0.5
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "threshold": threshold, **selected, "all_thresholds": metrics}
        history.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "all_thresholds"}), flush=True)
        if score > best_score + 1e-4:
            best_score, stale = score, 0
            model.save_pretrained(output / "best")
            (output / "best" / "deployment.json").write_text(json.dumps({"target": args.target, "threshold": threshold, "image_size": args.image_size, "metrics": selected}, indent=2))
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, output / "last_state.pt")
        if epoch >= args.min_epochs and stale >= args.patience:
            break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=DISEASE_DATASET_NAMES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--melasma-dataset", type=Path)
    parser.add_argument("--vitiligo-dataset", type=Path)
    parser.add_argument("--normal-dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-epoch", type=int, default=480)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--lesion-crop-probability", type=float, default=0.0)
    parser.add_argument("--low-contrast-probability", type=float, default=0.0)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-precision", type=float, default=0.6)
    parser.add_argument("--min-recall", type=float, default=0.5)
    parser.add_argument("--min-image-recall", type=float, default=0.8)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-classifier", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

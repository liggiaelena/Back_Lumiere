import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import SegformerConfig, SegformerForSemanticSegmentation


ID2LABEL = {
    0: "background",
    1: "vitiligo",
    2: "melasma_like_hyperpigmentation",
    3: "port_wine_stain",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
DISEASE_CLASS_IDS = {
    "vitiligo": 1,
    "melasma": 2,
    "port_wine_stain": 3,
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


class DiseaseSegmentationDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: Path,
        split: str,
        image_size: int,
        class_id: int,
        augment: bool = False,
    ):
        self.name = name
        self.dataset_dir = dataset_dir
        self.image_dir = dataset_dir / "images"
        self.mask_dir = dataset_dir / "masks"
        split_path = dataset_dir / "splits" / f"{split}.txt"
        self.items = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        self.image_size = image_size
        self.class_id = class_id
        self.augment = augment
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        stem = self._item_stem(item)
        image_path = self._find_image(item)
        mask_path = self.mask_dir / f"{stem}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            image = self.color_jitter(image)

        pixel_values = TF.normalize(TF.to_tensor(image), mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        labels_np = np.array(mask, dtype=np.int64)
        labels_np = np.where(labels_np > 0, self.class_id, 0)
        labels = torch.from_numpy(labels_np)
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "disease": self.name,
            "stem": stem,
        }

    def _item_stem(self, item: str) -> str:
        item_path = Path(item)
        if item_path.suffix.lower() in IMAGE_EXTENSIONS:
            return item_path.stem
        return item

    def _find_image(self, item: str) -> Path:
        item_path = Path(item)
        if item_path.suffix:
            path = self.image_dir / item_path.name
            if path.exists():
                return path
        for ext in IMAGE_EXTENSIONS:
            path = self.image_dir / f"{item}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"No image found for {item} in {self.image_dir}")


class UnifiedDataset(Dataset):
    def __init__(self, datasets: list[DiseaseSegmentationDataset]):
        self.datasets = datasets
        self.index: list[tuple[int, int]] = []
        for dataset_idx, dataset in enumerate(datasets):
            self.index.extend((dataset_idx, item_idx) for item_idx in range(len(dataset)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dataset_idx, item_idx = self.index[idx]
        return self.datasets[dataset_idx][item_idx]

    def balanced_weights(self, ratios: dict[str, float] | None = None) -> list[float]:
        dataset_lengths = [len(dataset) for dataset in self.datasets]
        weights = []
        for dataset_idx, _ in self.index:
            dataset = self.datasets[dataset_idx]
            ratio = ratios.get(dataset.name, 1.0) if ratios else 1.0
            weights.append(ratio / dataset_lengths[dataset_idx])
        return weights


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([row["pixel_values"] for row in batch]),
        "labels": torch.stack([row["labels"] for row in batch]),
        "disease": [row["disease"] for row in batch],
        "stem": [row["stem"] for row in batch],
    }


def get_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_device(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch environment has no CUDA device.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but this PyTorch environment has no MPS device.")


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Explicit flattening avoids MPS backward issues with non-contiguous CE inputs.
    num_classes = logits.shape[1]
    flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
    flat_labels = labels.reshape(-1)
    return F.cross_entropy(flat_logits, flat_labels, weight=weight)


def foreground_dice_loss(logits: torch.Tensor, labels: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    losses = []
    for class_id in DISEASE_CLASS_IDS.values():
        target = (labels == class_id).float()
        target_sum = target.sum()
        if target_sum == 0:
            continue
        pred = probs[:, class_id]
        intersection = (pred * target).sum()
        denominator = pred.sum() + target_sum
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        losses.append(1.0 - dice)
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def segmentation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    dice_weight: float = 0.0,
) -> torch.Tensor:
    ce_loss = cross_entropy_loss(logits, labels, weight)
    if dice_weight <= 0:
        return ce_loss
    dice_loss = foreground_dice_loss(logits, labels)
    return ce_loss + dice_weight * dice_loss


def compute_class_iou(preds: torch.Tensor, labels: torch.Tensor, class_id: int) -> float:
    pred_pos = preds == class_id
    label_pos = labels == class_id
    intersection = (pred_pos & label_pos).sum().item()
    union = (pred_pos | label_pos).sum().item()
    if union == 0:
        return 1.0
    return intersection / union


def make_datasets(args, split: str, augment: bool) -> dict[str, DiseaseSegmentationDataset]:
    return {
        "melasma": DiseaseSegmentationDataset(
            "melasma",
            Path(args.melasma_dataset),
            split,
            args.image_size,
            DISEASE_CLASS_IDS["melasma"],
            augment=augment,
        ),
        "port_wine_stain": DiseaseSegmentationDataset(
            "port_wine_stain",
            Path(args.port_wine_stain_dataset),
            split,
            args.image_size,
            DISEASE_CLASS_IDS["port_wine_stain"],
            augment=augment,
        ),
        "vitiligo": DiseaseSegmentationDataset(
            "vitiligo",
            Path(args.vitiligo_dataset),
            split,
            args.image_size,
            DISEASE_CLASS_IDS["vitiligo"],
            augment=augment,
        ),
    }


def make_train_loader(args, datasets: dict[str, DiseaseSegmentationDataset]) -> DataLoader:
    unified_dataset = UnifiedDataset(list(datasets.values()))
    if args.sampling == "balanced":
        sampling_ratios = parse_sampling_ratios(args.sampling_ratios)
        samples_per_epoch = args.samples_per_epoch or max(len(dataset) for dataset in datasets.values()) * len(datasets)
        sampler = WeightedRandomSampler(
            weights=unified_dataset.balanced_weights(sampling_ratios),
            num_samples=samples_per_epoch,
            replacement=True,
        )
        return DataLoader(
            unified_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=args.pin_memory,
        )

    return DataLoader(
        unified_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=args.pin_memory,
    )


def parse_sampling_ratios(values: list[str] | None) -> dict[str, float]:
    if not values:
        return {}
    ratios = {}
    valid_names = set(DISEASE_CLASS_IDS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid sampling ratio '{value}'. Use name=value, e.g. vitiligo=3.")
        name, raw_ratio = value.split("=", 1)
        if name not in valid_names:
            raise ValueError(f"Unknown disease '{name}'. Expected one of: {', '.join(sorted(valid_names))}.")
        ratio = float(raw_ratio)
        if ratio <= 0:
            raise ValueError(f"Sampling ratio for {name} must be positive.")
        ratios[name] = ratio
    return ratios


def make_val_loaders(args, datasets: dict[str, DiseaseSegmentationDataset]) -> dict[str, DataLoader]:
    return {
        name: DataLoader(
            dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=args.pin_memory,
        )
        for name, dataset in datasets.items()
    }


@torch.no_grad()
def evaluate_one(
    model,
    loader: DataLoader,
    device: torch.device,
    weight: torch.Tensor,
    class_id: int,
    dice_weight: float,
) -> dict[str, float]:
    model.eval()
    losses = []
    ious = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values)
        logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = segmentation_loss(logits, labels, weight, dice_weight=dice_weight)
        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        ious.append(compute_class_iou(preds.cpu(), labels.cpu(), class_id=class_id))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
    }


def evaluate_all(
    model,
    loaders: dict[str, DataLoader],
    device: torch.device,
    weight: torch.Tensor,
    dice_weight: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    ious = []
    losses = []
    for name, loader in loaders.items():
        result = evaluate_one(model, loader, device, weight, DISEASE_CLASS_IDS[name], dice_weight)
        metrics[f"val_{name}_loss"] = result["loss"]
        metrics[f"val_{name}_iou"] = result["iou"]
        ious.append(result["iou"])
        losses.append(result["loss"])
    metrics["val_mean_iou"] = float(np.mean(ious)) if ious else 0.0
    metrics["val_min_iou"] = float(np.min(ious)) if ious else 0.0
    metrics["val_mean_loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def load_training_state(state_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(state_path, map_location=device, weights_only=False)


def save_training_state(
    state_path: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_score: float,
    history: list[dict[str, Any]],
    args,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_score": best_score,
            "history": history,
            "args": vars(args),
        },
        state_path,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_metadata(output_dir: Path, args, train_datasets: dict[str, DiseaseSegmentationDataset]) -> None:
    metadata = {
        "model_name": "lumiere_segformer_unified_joint",
        "training_mode": "joint_multitask",
        "class_mapping": {str(key): value for key, value in ID2LABEL.items()},
        "datasets": {
            name: {
                "path": dataset.dataset_dir.as_posix(),
                "train_samples": len(dataset),
                "class_id": dataset.class_id,
            }
            for name, dataset in train_datasets.items()
        },
        "hyperparameters": vars(args),
    }
    (output_dir / "lumiere_unified_model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "label_mapping.json").write_text(
        json.dumps({"id2label": ID2LABEL, "label2id": LABEL2ID}, indent=2) + "\n"
    )


def train(args) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_datasets = make_datasets(args, "train", augment=not args.no_augment)
    val_datasets = make_datasets(args, "val", augment=False)
    train_loader = make_train_loader(args, train_datasets)
    val_loaders = make_val_loaders(args, val_datasets)

    device = get_device(args.device)
    validate_device(device)
    print(f"using_device={device}")
    print(
        "train_samples="
        + json.dumps({name: len(dataset) for name, dataset in train_datasets.items()}, sort_keys=True)
    )

    config = SegformerConfig.from_pretrained(args.checkpoint)
    config.num_labels = 4
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    class_weight = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_score = -1.0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    training_state_path = Path(args.resume_state) if args.resume_state else output_dir / "training_state.pt"

    if args.resume:
        if training_state_path.exists():
            state = load_training_state(training_state_path, device)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if scheduler and state.get("scheduler_state_dict"):
                scheduler.load_state_dict(state["scheduler_state_dict"])
            best_score = float(state.get("best_score", best_score))
            history = state.get("history", history)
            start_epoch = int(state.get("epoch", 0)) + 1
            print(f"resumed_training_state={training_state_path} start_epoch={start_epoch}")
        else:
            print(f"resume_state_not_found={training_state_path}; starting fresh")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=pixel_values)
            logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = segmentation_loss(logits, labels, class_weight, dice_weight=args.dice_weight)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_losses.append(loss.item())

        if scheduler:
            scheduler.step()

        metrics = evaluate_all(model, val_loaders, device, class_weight, dice_weight=args.dice_weight)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": current_lr,
            **metrics,
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_mean_iou={metrics['val_mean_iou']:.4f} "
            f"val_min_iou={metrics['val_min_iou']:.4f} "
            f"val_melasma_iou={metrics['val_melasma_iou']:.4f} "
            f"val_pws_iou={metrics['val_port_wine_stain_iou']:.4f} "
            f"val_vitiligo_iou={metrics['val_vitiligo_iou']:.4f} "
            f"lr={current_lr:.2e}"
        )

        score = metrics["val_mean_iou"]
        if args.min_iou_weight > 0:
            score = (1.0 - args.min_iou_weight) * metrics["val_mean_iou"] + args.min_iou_weight * metrics["val_min_iou"]
        row["selection_score"] = score

        if score > best_score:
            best_score = score
            model.save_pretrained(output_dir / "best")
            save_metadata(output_dir / "best", args, train_datasets)

        model.save_pretrained(output_dir / "last")
        save_metadata(output_dir / "last", args, train_datasets)
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        save_training_state(training_state_path, epoch, model, optimizer, scheduler, best_score, history, args)

    print(f"best_selection_score={best_score:.4f}")
    print(f"saved_best={output_dir / 'best'}")
    print(f"saved_last={output_dir / 'last'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Jointly train a 4-class Lumiere SegFormer model.")
    parser.add_argument(
        "--checkpoint",
        default="training/checkpoints/SegFormer/segformer_b2_4class_port_wine_stain_finetune/best",
        help="Starting checkpoint. Defaults to the existing 4-class checkpoint to preserve the segmentation head.",
    )
    parser.add_argument("--output-dir", default="training/checkpoints/SegFormer/unified/joint")
    parser.add_argument("--melasma-dataset", default="data-collection/melasma")
    parser.add_argument("--port-wine-stain-dataset", default="data-collection/port_wine_stain/processed")
    parser.add_argument("--vitiligo-dataset", default="data-collection/vitiligo")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--class-weights", type=float, nargs=4, default=[0.3, 2.0, 2.0, 2.0])
    parser.add_argument("--dice-weight", type=float, default=0.0)
    parser.add_argument("--min-iou-weight", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--sampling", choices=["balanced", "natural"], default="balanced")
    parser.add_argument(
        "--sampling-ratios",
        nargs="*",
        help="Optional balanced sampler multipliers, e.g. melasma=1 port_wine_stain=1 vitiligo=3.",
    )
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from training_state.pt if it exists.")
    parser.add_argument("--resume-state", help="Path to a saved training_state.pt.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

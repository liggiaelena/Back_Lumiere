import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
VITILIGO_CLASS_ID = 1
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


class VilitigoDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, image_size: int, augment: bool = False):
        self.dataset_dir = dataset_dir
        self.image_dir = dataset_dir / "images"
        self.mask_dir = dataset_dir / "masks"
        split_path = dataset_dir / "splits" / f"{split}.txt"
        self.items = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        self.image_size = image_size
        self.augment = augment
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem = self.items[idx]
        image_path = self._find_image(stem)
        mask_path = self.mask_dir / f"{stem}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize both image and mask together
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.augment:
            # Random horizontal flip applied to both image and mask
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # Color jitter applied only to image
            image = self.color_jitter(image)

        pixel_values = TF.normalize(
            TF.to_tensor(image),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        labels_np = np.array(mask, dtype=np.int64)
        labels_np = np.where(labels_np > 0, VITILIGO_CLASS_ID, 0)
        labels = torch.from_numpy(labels_np)
        return {"pixel_values": pixel_values, "labels": labels, "stem": stem}

    def _find_image(self, stem: str) -> Path:
        for ext in IMAGE_EXTENSIONS:
            path = self.image_dir / f"{stem}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"No image found for {stem} in {self.image_dir}")


def compute_class_iou(preds: torch.Tensor, labels: torch.Tensor, class_id: int) -> float:
    pred_pos = preds == class_id
    label_pos = labels == class_id
    intersection = (pred_pos & label_pos).sum().item()
    union = (pred_pos | label_pos).sum().item()
    if union == 0:
        return 1.0
    return intersection / union


def segmentation_loss(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Weighted cross-entropy: higher weight on vitiligo class to counter class imbalance
    # Flatten explicitly to avoid PyTorch/MPS cross_entropy backward view/stride issues.
    num_classes = logits.shape[1]
    flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
    flat_labels = labels.reshape(-1)
    return F.cross_entropy(flat_logits, flat_labels, weight=weight)


def get_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_training_state(state_path: Path, device: torch.device) -> dict:
    return torch.load(state_path, map_location=device, weights_only=False)


def save_training_state(
    state_path: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_iou: float,
    history: list[dict],
    args,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_iou": best_iou,
            "history": history,
            "args": vars(args),
        },
        state_path,
    )


@torch.no_grad()
def evaluate(model, loader, device, weight):
    model.eval()
    losses = []
    ious = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values)
        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = segmentation_loss(logits, labels, weight)
        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        ious.append(compute_class_iou(preds.cpu(), labels.cpu(), class_id=VITILIGO_CLASS_ID))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "vitiligo_iou": float(np.mean(ious)) if ious else 0.0,
    }


def train(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Augmentation only on train split
    train_dataset = VilitigoDataset(dataset_dir, "train", args.image_size, augment=True)
    val_dataset   = VilitigoDataset(dataset_dir, "val",   args.image_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = get_device(args.device)
    print(f"using_device={device}")

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

    # Class weights: vitiligo (1) gets 3x weight to counter background dominance
    class_weight = torch.tensor([0.3, 3.0, 1.0, 1.0], device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Cosine scheduler gradually reduces LR, stabilising training in later epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_iou = -1.0
    history = []
    start_epoch = 1
    training_state_path = Path(args.resume_state) if args.resume_state else output_dir / "training_state.pt"

    if args.resume:
        if training_state_path.exists():
            state = load_training_state(training_state_path, device)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if "scheduler_state_dict" in state:
                scheduler.load_state_dict(state["scheduler_state_dict"])
            best_iou = float(state.get("best_iou", best_iou))
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
            logits = F.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = segmentation_loss(logits, labels, class_weight)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        metrics = evaluate(model, val_loader, device, class_weight)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        current_lr = scheduler.get_last_lr()[0]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_vitiligo_iou": metrics["vitiligo_iou"],
            "lr": current_lr,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_vitiligo_iou={metrics['vitiligo_iou']:.4f} "
            f"lr={current_lr:.2e}"
        )

        if metrics["vitiligo_iou"] > best_iou:
            best_iou = metrics["vitiligo_iou"]
            model.save_pretrained(output_dir / "best")

        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        save_training_state(training_state_path, epoch, model, optimizer, scheduler, best_iou, history, args)

    model.save_pretrained(output_dir / "last")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output_dir / "label_mapping.json").write_text(
        json.dumps({"id2label": ID2LABEL, "label2id": LABEL2ID}, indent=2) + "\n"
    )
    print(f"best_val_vitiligo_iou={best_iou:.4f}")
    print(f"saved_best={output_dir / 'best'}")
    print(f"saved_last={output_dir / 'last'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune SegFormer-B2 on vitiligo segmentation masks.")
    parser.add_argument(
        "--dataset-dir",
        default="data-collection/vitiligo",
        help="Processed dataset directory with images, masks, and splits.",
    )
    parser.add_argument(
        "--checkpoint",
        default="training/checkpoints/SegFormer/unified/tmp/port_wine_stain/best",
        help="Starting checkpoint directory. Use the port wine stain 4-class checkpoint so the full class architecture is preserved.",
    )
    parser.add_argument(
        "--output-dir",
        default="training/checkpoints/SegFormer/unified/tmp/vitiligo",
        help="Directory for the finetuned model.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Training device. auto uses CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from training_state.pt if it exists.")
    parser.add_argument(
        "--resume-state",
        help="Path to a saved training_state.pt. Defaults to <output-dir>/training_state.pt.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

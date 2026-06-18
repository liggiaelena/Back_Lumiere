import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from transformers import SegformerConfig, SegformerForSemanticSegmentation


ID2LABEL = {
    0: "background",
    1: "vitiligo",
    2: "melasma_like_hyperpigmentation",
    3: "port_wine_stain",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
MELASMA_CLASS_ID = 2
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


class MelasmaDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, image_size: int, augment: bool = False):
        self.dataset_dir = dataset_dir
        self.image_dir = dataset_dir / "images"
        self.mask_dir = dataset_dir / "masks"
        split_path = dataset_dir / "splits" / f"{split}.txt"
        self.items = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = self._find_image(item)
        mask_path = self.mask_dir / f"{Path(item).stem}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        if self.augment and torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        pixel_values = TF.normalize(
            TF.to_tensor(image),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        labels_np = np.array(mask, dtype=np.int64)
        labels_np = np.where(labels_np > 0, MELASMA_CLASS_ID, 0)
        labels = torch.from_numpy(labels_np)
        return {"pixel_values": pixel_values, "labels": labels, "stem": Path(item).stem}

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


def compute_class_iou(preds: torch.Tensor, labels: torch.Tensor, class_id: int) -> float:
    pred_pos = preds == class_id
    label_pos = labels == class_id
    intersection = (pred_pos & label_pos).sum().item()
    union = (pred_pos | label_pos).sum().item()
    if union == 0:
        return 1.0
    return intersection / union


def segmentation_loss(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.contiguous(), labels.contiguous(), weight=weight)


def get_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, weight):
    model.eval()
    losses = []
    ious = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values)
        logits = F.interpolate(
            outputs.logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = segmentation_loss(logits, labels, weight)
        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        ious.append(compute_class_iou(preds.cpu(), labels.cpu(), class_id=MELASMA_CLASS_ID))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "melasma_iou": float(np.mean(ious)) if ious else 0.0,
    }


def train(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = MelasmaDataset(dataset_dir, "train", args.image_size, augment=True)
    val_dataset = MelasmaDataset(dataset_dir, "val", args.image_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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

    class_weight = torch.tensor([0.3, 1.0, 3.0, 1.0], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_iou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
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

        metrics = evaluate(model, val_loader, device, class_weight)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_melasma_iou": metrics["melasma_iou"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_melasma_iou={metrics['melasma_iou']:.4f}"
        )

        if metrics["melasma_iou"] > best_iou:
            best_iou = metrics["melasma_iou"]
            model.save_pretrained(output_dir / "best")

    model.save_pretrained(output_dir / "last")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output_dir / "label_mapping.json").write_text(
        json.dumps({"id2label": ID2LABEL, "label2id": LABEL2ID}, indent=2) + "\n"
    )
    print(f"best_val_melasma_iou={best_iou:.4f}")
    print(f"saved_best={output_dir / 'best'}")
    print(f"saved_last={output_dir / 'last'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune SegFormer-B2 on melasma segmentation masks.")
    parser.add_argument("--dataset-dir", default="data-collection/melasma")
    parser.add_argument(
        "--checkpoint",
        default="training/checkpoints/SegFormer/segformer_b2_melasma_colab_export",
        help="Starting checkpoint directory. Existing melasma checkpoint is used as the local base.",
    )
    parser.add_argument(
        "--output-dir",
        default="training/checkpoints/SegFormer/melasma_v2",
        help="Directory for the finetuned model.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

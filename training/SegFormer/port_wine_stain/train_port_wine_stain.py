import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import SegformerConfig, SegformerForSemanticSegmentation


ID2LABEL = {
    0: "background",
    1: "vitiligo",
    2: "melasma_like_hyperpigmentation",
    3: "port_wine_stain",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


class PortWineStainDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, image_size: int):
        self.dataset_dir = dataset_dir
        self.image_dir = dataset_dir / "images"
        self.mask_dir = dataset_dir / "masks"
        split_path = dataset_dir / "splits" / f"{split}.txt"
        self.items = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        self.image_size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem = self.items[idx]
        image_path = self._find_image(stem)
        mask_path = self.mask_dir / f"{stem}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), resample=Image.Resampling.NEAREST)

        pixel_values = self.image_transform(image)
        labels = torch.from_numpy(np.array(mask, dtype=np.int64))
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


def segmentation_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.contiguous()
    labels = labels.contiguous()
    log_probs = F.log_softmax(logits, dim=1)
    picked = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    return -picked.mean()


def get_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
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
        loss = segmentation_loss(logits, labels)
        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        ious.append(compute_class_iou(preds.cpu(), labels.cpu(), class_id=3))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "port_wine_stain_iou": float(np.mean(ious)) if ious else 0.0,
    }


def train(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PortWineStainDataset(dataset_dir, "train", args.image_size)
    val_dataset = PortWineStainDataset(dataset_dir, "val", args.image_size)
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
            loss = segmentation_loss(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        metrics = evaluate(model, val_loader, device)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_port_wine_stain_iou": metrics["port_wine_stain_iou"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_pws_iou={metrics['port_wine_stain_iou']:.4f}"
        )

        if metrics["port_wine_stain_iou"] > best_iou:
            best_iou = metrics["port_wine_stain_iou"]
            model.save_pretrained(output_dir / "best")

    model.save_pretrained(output_dir / "last")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output_dir / "label_mapping.json").write_text(
        json.dumps({"id2label": ID2LABEL, "label2id": LABEL2ID}, indent=2) + "\n"
    )
    print(f"best_val_port_wine_stain_iou={best_iou:.4f}")
    print(f"saved_best={output_dir / 'best'}")
    print(f"saved_last={output_dir / 'last'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune SegFormer-B2 on port wine stain segmentation masks.")
    parser.add_argument(
        "--dataset-dir",
        default="training/datasets/port_wine_stain/processed",
        help="Processed dataset directory with images, masks, and splits.",
    )
    parser.add_argument(
        "--checkpoint",
        default="training/checkpoints/SegFormer/segformer_b2_melasma_colab_export",
        help="Starting checkpoint directory. Prefer the Task 6 ISIC checkpoint when available.",
    )
    parser.add_argument(
        "--output-dir",
        default="training/checkpoints/SegFormer/segformer_b2_port_wine_stain_export",
        help="Directory for the finetuned model.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Training device. auto uses CUDA, then Apple MPS, then CPU.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

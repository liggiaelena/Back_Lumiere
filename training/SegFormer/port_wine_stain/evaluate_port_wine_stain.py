import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
PORT_WINE_STAIN_CLASS_ID = 3


def get_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_image(image_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image found for {stem} in {image_dir}")


def compute_iou(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    pred_pos = pred == class_id
    target_pos = target == class_id
    union = np.logical_or(pred_pos, target_pos).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred_pos, target_pos).sum()
    return float(intersection / union)


def make_overlay(image: Image.Image, pred: np.ndarray, target: np.ndarray) -> Image.Image:
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    pred_pos = pred == PORT_WINE_STAIN_CLASS_ID
    target_pos = target == PORT_WINE_STAIN_CLASS_ID

    overlay = rgb.copy()
    overlay[target_pos] = overlay[target_pos] * 0.45 + np.array([0, 220, 0]) * 0.55
    overlay[pred_pos] = overlay[pred_pos] * 0.45 + np.array([230, 0, 180]) * 0.55

    both = pred_pos & target_pos
    overlay[both] = overlay[both] * 0.35 + np.array([255, 255, 0]) * 0.65
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


@torch.no_grad()
def evaluate(args):
    dataset_dir = Path(args.dataset_dir)
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    split_path = dataset_dir / "splits" / f"{args.split}.txt"
    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    names = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    device = get_device(args.device)
    print(f"using_device={device}")

    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    image_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    rows = []
    for idx, stem in enumerate(names):
        image_path = find_image(image_dir, stem)
        mask_path = mask_dir / f"{stem}.png"

        image = Image.open(image_path).convert("RGB")
        target = np.array(Image.open(mask_path).convert("L"))
        input_tensor = image_transform(image).unsqueeze(0).to(device)

        outputs = model(pixel_values=input_tensor)
        logits = F.interpolate(outputs.logits, size=target.shape, mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        iou = compute_iou(pred, target, PORT_WINE_STAIN_CLASS_ID)
        rows.append({"stem": stem, "port_wine_stain_iou": iou})

        if idx < args.max_overlays:
            overlay = make_overlay(image, pred, target)
            overlay.save(overlay_dir / f"{stem}_overlay.png")

    mean_iou = float(np.mean([row["port_wine_stain_iou"] for row in rows])) if rows else 0.0
    summary = {
        "checkpoint": args.checkpoint,
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "samples": len(rows),
        "image_size": args.image_size,
        "mean_port_wine_stain_iou": mean_iou,
        "overlay_legend": {
            "green": "ground truth",
            "magenta": "prediction",
            "yellow": "prediction and ground truth overlap",
        },
        "per_sample": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.split}_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, indent=2))
    print(f"metrics={output_dir / f'{args.split}_metrics.json'}")
    print(f"overlays={overlay_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the port wine stain SegFormer checkpoint.")
    parser.add_argument("--dataset-dir", default="data-collection/port_wine_stain/processed")
    parser.add_argument(
        "--checkpoint",
        default="training/checkpoints/SegFormer/port_wine_stain_v2/best",
    )
    parser.add_argument(
        "--output-dir",
        default="training/checkpoints/SegFormer/port_wine_stain_v2/evaluation",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-overlays", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

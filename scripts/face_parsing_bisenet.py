# face_parsing_bisenet.py

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.models.model import BiSeNet

BASE_DIR = Path(__file__).resolve().parent

INPUT_FACE = BASE_DIR / "outputs" / "face_crop.jpg"
OUTPUT_DIR = BASE_DIR / "outputs"

BEST_MODEL = BASE_DIR / "checkpoints" / "bisenet_best.pth"
DEFAULT_MODEL = BASE_DIR / "checkpoints" / "79999_iter.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = (512, 512)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_checkpoint_and_num_classes():
    """
    If trained model exists, use it with 14 classes.
    Otherwise use pretrained model with 19 classes.
    """
    if BEST_MODEL.exists():
        print("Using trained model:", BEST_MODEL)
        return BEST_MODEL, 14
    elif DEFAULT_MODEL.exists():
        print("Using pretrained model:", DEFAULT_MODEL)
        return DEFAULT_MODEL, 19
    else:
        raise FileNotFoundError(
            f"No checkpoint found.\n"
            f"Expected one of:\n"
            f" - {BEST_MODEL}\n"
            f" - {DEFAULT_MODEL}"
        )


CHECKPOINT, NUM_CLASSES = get_checkpoint_and_num_classes()


def get_palette(num_classes):
    palette = []
    for i in range(num_classes):
        palette.append(((i * 37) % 255, (i * 67) % 255, (i * 97) % 255))
    return palette


def load_model():
    net = BiSeNet(n_classes=NUM_CLASSES)
    state = torch.load(str(CHECKPOINT), map_location=DEVICE)

    if isinstance(state, dict) and "model_state_dict" in state:
        net.load_state_dict(state["model_state_dict"])
    else:
        net.load_state_dict(state)

    net.to(DEVICE)
    net.eval()
    return net


def main():
    ensure_dir(OUTPUT_DIR)

    print("INPUT_FACE =", INPUT_FACE)
    print("OUTPUT_DIR =", OUTPUT_DIR)
    print("CHECKPOINT =", CHECKPOINT)
    print("NUM_CLASSES =", NUM_CLASSES)
    print("FACE exists:", INPUT_FACE.exists())
    print("CHECKPOINT exists:", CHECKPOINT.exists())
    print("DEVICE =", DEVICE)

    img_bgr = cv2.imread(str(INPUT_FACE))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {INPUT_FACE}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    tfm = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    x = tfm(pil_img).unsqueeze(0).to(DEVICE)

    net = load_model()

    with torch.no_grad():
        out = net(x)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)

    mask_path = OUTPUT_DIR / "parsing_mask.png"
    cv2.imwrite(str(mask_path), parsing)

    palette = get_palette(NUM_CLASSES)
    color_mask = np.zeros((parsing.shape[0], parsing.shape[1], 3), dtype=np.uint8)

    for cls_id, color in enumerate(palette):
        color_mask[parsing == cls_id] = color

    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    color_path = OUTPUT_DIR / "parsing_color.png"
    cv2.imwrite(str(color_path), color_mask_bgr)

    meta = {
        "input_face": str(INPUT_FACE),
        "checkpoint": str(CHECKPOINT),
        "num_classes": NUM_CLASSES,
        "mask_path": str(mask_path),
        "color_visualization_path": str(color_path),
        "input_size": INPUT_SIZE,
        "device": DEVICE,
    }

    json_path = OUTPUT_DIR / "parsing_meta.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:")
    print(mask_path)
    print(color_path)
    print(json_path)


if __name__ == "__main__":
    main()
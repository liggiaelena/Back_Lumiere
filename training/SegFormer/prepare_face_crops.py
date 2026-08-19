"""Create a face-domain view of a prepared segmentation dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "dev"))
from app.face_detection_mediapipe import detect_and_zoom_face  # noqa: E402


def prepare(source: Path, output: Path, padding_ratio: float) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    for name in ("images", "masks", "splits"):
        (output / name).mkdir(parents=True, exist_ok=True)

    manifest = {"source": str(source), "padding_ratio": padding_ratio, "splits": {}}
    for split in ("train", "val", "test"):
        items = [line.strip() for line in (source / "splits" / f"{split}.txt").read_text().splitlines() if line.strip()]
        kept, no_face, positives, negatives = [], 0, 0, 0
        for item in items:
            image_path = source / "images" / Path(item).name
            mask_path = source / "masks" / f"{Path(item).stem}.png"
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None or mask is None or bgr.shape[:2] != mask.shape:
                raise ValueError(f"Invalid image/mask pair: {item}")
            try:
                _, info = detect_and_zoom_face(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    padding_ratio=padding_ratio,
                )
            except ValueError:
                no_face += 1
                continue
            box = info["crop_bbox"]
            image_crop = bgr[box["y1"]:box["y2"], box["x1"]:box["x2"]]
            mask_crop = mask[box["y1"]:box["y2"], box["x1"]:box["x2"]]
            if image_crop.size == 0:
                no_face += 1
                continue
            destination_image = output / "images" / Path(item).name
            destination_mask = output / "masks" / f"{Path(item).stem}.png"
            if not cv2.imwrite(str(destination_image), image_crop):
                raise OSError(destination_image)
            if not cv2.imwrite(str(destination_mask), (mask_crop > 0).astype(np.uint8)):
                raise OSError(destination_mask)
            kept.append(Path(item).name)
            if mask_crop.any():
                positives += 1
            else:
                negatives += 1
        (output / "splits" / f"{split}.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
        manifest["splits"][split] = {
            "input": len(items),
            "kept": len(kept),
            "no_face": no_face,
            "positive_images": positives,
            "negative_images": negatives,
        }
    source_manifest = source / "manifest.json"
    if source_manifest.exists():
        shutil.copy2(source_manifest, output / "source_manifest.json")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padding-ratio", type=float, default=0.35)
    args = parser.parse_args()
    prepare(args.source, args.output, args.padding_ratio)


if __name__ == "__main__":
    main()

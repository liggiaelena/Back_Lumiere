"""Convert Roboflow COCO melasma annotations into the independent trainer layout.

The source COCO files contain compressed RLE masks. Their ``area`` and ``bbox``
metadata are intentionally ignored because they are inconsistent with the masks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def decode_compressed_rle(counts: str, height: int, width: int) -> np.ndarray:
    runs: list[int] = []
    position = 0
    while position < len(counts):
        value = 0
        shift = 0
        while True:
            code = ord(counts[position]) - 48
            position += 1
            value |= (code & 0x1F) << (5 * shift)
            shift += 1
            if not (code & 0x20):
                if code & 0x10:
                    value |= -1 << (5 * shift)
                break
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise ValueError("Negative RLE run length")
        runs.append(value)

    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    foreground = False
    for length in runs:
        end = offset + length
        if foreground:
            flat[offset:end] = 255
        offset = end
        foreground = not foreground
    if offset != flat.size:
        raise ValueError(f"RLE length {offset} does not match {flat.size}")
    return flat.reshape((height, width), order="F")


def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        counts = segmentation["counts"]
        if not isinstance(counts, str):
            raise ValueError("Only compressed COCO RLE is supported")
        return decode_compressed_rle(counts, height, width)

    # A few source annotations use polygons.
    from PIL import ImageDraw

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in segmentation:
        points = list(zip(polygon[0::2], polygon[1::2]))
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    return np.asarray(canvas, dtype=np.uint8)


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def convert(source: Path, output: Path, old_dataset: Path | None) -> dict:
    image_output = output / "images"
    mask_output = output / "masks"
    split_output = output / "splits"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)
    split_output.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    for source_split, target_split in SPLIT_MAP.items():
        split_dir = source / source_split
        coco = json.loads((split_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
        annotations = defaultdict(list)
        for annotation in coco["annotations"]:
            if annotation.get("category_id") == 1:
                annotations[annotation["image_id"]].append(annotation)

        names: list[str] = []
        skipped_empty: list[str] = []
        areas: list[float] = []
        for item in coco["images"]:
            source_image = split_dir / item["file_name"]
            mask = np.zeros((item["height"], item["width"]), dtype=np.uint8)
            for annotation in annotations[item["id"]]:
                mask = np.maximum(mask, decode_segmentation(annotation["segmentation"], item["height"], item["width"]))
            if not mask.any():
                skipped_empty.append(item["file_name"])
                continue
            name = f"coco_{source_split}_{item['file_name']}"
            link_or_copy(source_image, image_output / name)
            Image.fromarray(mask).save(mask_output / f"{Path(name).stem}.png")
            names.append(name)
            areas.append(float((mask > 0).mean()))

        # Retain only non-empty legacy masks. Known legacy empty masks include
        # missed melasma cases and are therefore unsafe training negatives.
        old_added = 0
        if old_dataset is not None:
            old_split = old_dataset / "splits" / f"{target_split}.txt"
            for raw_name in old_split.read_text(encoding="utf-8").splitlines():
                raw_name = raw_name.strip()
                if not raw_name:
                    continue
                stem = Path(raw_name).stem
                old_mask = np.asarray(Image.open(old_dataset / "masks" / f"{stem}.png").convert("L"))
                if not (old_mask > 0).any():
                    continue
                old_image = next(
                    old_dataset / "images" / f"{stem}{ext}"
                    for ext in (".jpg", ".jpeg", ".png", ".webp")
                    if (old_dataset / "images" / f"{stem}{ext}").exists()
                )
                name = f"legacy_{target_split}_{old_image.name}"
                link_or_copy(old_image, image_output / name)
                Image.fromarray(np.where(old_mask > 0, 255, 0).astype(np.uint8)).save(mask_output / f"{Path(name).stem}.png")
                names.append(name)
                old_added += 1

        (split_output / f"{target_split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        report[target_split] = {
            "total": len(names),
            "coco_non_empty": len(areas),
            "legacy_non_empty": old_added,
            "skipped_coco_empty": skipped_empty,
            "coco_mask_area_median": float(np.median(areas)) if areas else 0.0,
            "coco_mask_area_max": max(areas, default=0.0),
        }

    (output / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-dataset", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(convert(args.source, args.output, args.old_dataset), indent=2))

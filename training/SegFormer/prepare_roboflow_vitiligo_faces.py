"""Clean a Roboflow COCO vitiligo export into source-group-safe face crops."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "dev"))
from app.face_detection_mediapipe import detect_and_zoom_face  # noqa: E402


SOURCE_SUFFIX = re.compile(r"_(?:jpg|jpeg|png)\.rf\.[^.]+$", re.IGNORECASE)


def source_group(filename: str) -> str:
    return SOURCE_SUFFIX.sub("", Path(filename).stem).lower()


def rasterize(image: dict, annotations: list[dict], minimum: float, maximum: float) -> np.ndarray:
    height, width = int(image["height"]), int(image["width"])
    mask = np.zeros((height, width), dtype=np.uint8)
    for annotation in annotations:
        fraction = float(annotation.get("area", 0.0)) / max(height * width, 1)
        if not minimum <= fraction <= maximum:
            continue
        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list):
            continue
        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask


def load_candidates(source: Path, minimum: float, maximum: float, padding: float) -> tuple[dict[str, list[dict]], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    audit = defaultdict(int)
    for original_split in ("train", "valid", "test"):
        split_dir = source / original_split
        coco = json.loads((split_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
        annotations: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations[int(annotation["image_id"])].append(annotation)
        for metadata in coco["images"]:
            audit["input_images"] += 1
            mask = rasterize(metadata, annotations[int(metadata["id"])], minimum, maximum)
            if not mask.any():
                audit["no_eligible_annotation"] += 1
                continue
            image_path = split_dir / metadata["file_name"]
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                audit["unreadable"] += 1
                continue
            if mask.shape != bgr.shape[:2]:
                mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            try:
                _, info = detect_and_zoom_face(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), padding_ratio=padding)
            except ValueError:
                audit["no_face"] += 1
                continue
            box = info["crop_bbox"]
            image_crop = bgr[box["y1"] : box["y2"], box["x1"] : box["x2"]]
            mask_crop = mask[box["y1"] : box["y2"], box["x1"] : box["x2"]]
            if not mask_crop.any():
                audit["lesion_outside_face"] += 1
                continue
            fraction = float((mask_crop > 0).mean())
            if fraction > 0.45:
                audit["crop_mask_too_large"] += 1
                continue
            groups[source_group(metadata["file_name"])].append(
                {"name": metadata["file_name"], "image": image_crop, "mask": mask_crop, "fraction": fraction}
            )
            audit["eligible_face_variants"] += 1
    audit["eligible_source_groups"] = len(groups)
    return groups, dict(audit)


def prepare(source: Path, output: Path, minimum: float, maximum: float, padding: float, seed: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    for name in ("images", "masks", "splits"):
        (output / name).mkdir(parents=True, exist_ok=True)

    groups, audit = load_candidates(source, minimum, maximum, padding)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    test_count = max(1, round(len(keys) * 0.15))
    val_count = max(1, round(len(keys) * 0.15))
    assignments = {
        "test": keys[:test_count],
        "val": keys[test_count : test_count + val_count],
        "train": keys[test_count + val_count :],
    }
    manifests: dict[str, list[str]] = {name: [] for name in assignments}
    mask_fractions: dict[str, list[float]] = {name: [] for name in assignments}
    for split, split_groups in assignments.items():
        for group in split_groups:
            variants = groups[group] if split == "train" else [min(groups[group], key=lambda row: row["fraction"])]
            for index, row in enumerate(variants):
                stem = re.sub(r"[^a-zA-Z0-9_-]", "_", group)[:100]
                name = f"{stem}_{index:02d}.jpg"
                cv2.imwrite(str(output / "images" / name), row["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(output / "masks" / f"{Path(name).stem}.png"), (row["mask"] > 0).astype(np.uint8) * 255)
                manifests[split].append(name)
                mask_fractions[split].append(row["fraction"])
        (output / "splits" / f"{split}.txt").write_text("\n".join(manifests[split]) + "\n", encoding="utf-8")

    report = {
        "source": str(source),
        "license": "CC BY 4.0",
        "annotation_fraction_range": [minimum, maximum],
        "padding_ratio": padding,
        "seed": seed,
        "audit": audit,
        "splits": {
            split: {
                "source_groups": len(assignments[split]),
                "images": len(manifests[split]),
                "mask_fraction_median": float(np.median(mask_fractions[split])) if mask_fractions[split] else 0.0,
                "mask_fraction_max": max(mask_fractions[split], default=0.0),
            }
            for split in assignments
        },
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-annotation-fraction", type=float, default=0.0005)
    parser.add_argument("--maximum-annotation-fraction", type=float, default=0.4)
    parser.add_argument("--padding-ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    prepare(
        args.source,
        args.output,
        args.minimum_annotation_fraction,
        args.maximum_annotation_fraction,
        args.padding_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()

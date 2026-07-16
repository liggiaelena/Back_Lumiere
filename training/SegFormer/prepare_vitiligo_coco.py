"""Convert Roboflow COCO vitiligo polygons into leakage-safe binary masks."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def source_key(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"(?:_jpe?g|_png)?\.rf\.[0-9a-f]+$", "", stem)


def load_coco(split_dir: Path) -> tuple[dict, dict[int, list[dict]]]:
    json_files = list(split_dir.glob("*.json"))
    if len(json_files) != 1:
        raise ValueError(f"Expected one COCO JSON in {split_dir}, found {len(json_files)}")
    document = json.loads(json_files[0].read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in document["images"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in document["annotations"]:
        if int(annotation["category_id"]) != 1:
            continue
        annotations[int(annotation["image_id"])].append(annotation)
    return images, annotations


def rasterize(meta: dict, annotations: list[dict]) -> np.ndarray:
    mask = np.zeros((int(meta["height"]), int(meta["width"])), dtype=np.uint8)
    for annotation in annotations:
        for polygon in annotation.get("segmentation", []):
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, mask.shape[1] - 1)
            points[:, 1] = np.clip(points[:, 1], 0, mask.shape[0] - 1)
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask


def evenly_spaced_groups(groups: dict[str, list[dict]], count: int) -> set[str]:
    scored = []
    for key, rows in groups.items():
        fractions = [row["mask_fraction"] for row in rows]
        scored.append((float(np.median(fractions)), key))
    scored.sort()
    if count >= len(scored):
        return {key for _, key in scored}
    indices = np.linspace(0, len(scored) - 1, count, dtype=int)
    return {scored[int(index)][1] for index in indices}


def prepare(source: Path, output: Path, test_groups: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    image_output = output / "images"
    mask_output = output / "masks"
    split_output = output / "splits"
    for directory in (image_output, mask_output, split_output):
        directory.mkdir(parents=True, exist_ok=True)

    records: dict[str, list[dict]] = {"train": [], "valid": []}
    source_groups: dict[str, list[dict]] = defaultdict(list)
    for split in ("train", "valid"):
        split_dir = source / split
        images, annotations = load_coco(split_dir)
        for image_id, meta in images.items():
            source_path = split_dir / meta["file_name"]
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (meta["height"], meta["width"]):
                raise ValueError(f"Invalid image or COCO dimensions: {source_path}")
            mask = rasterize(meta, annotations.get(image_id, []))
            row = {
                "split": split,
                "source_path": source_path,
                "filename": meta["file_name"],
                "stem": Path(meta["file_name"]).stem,
                "source_key": source_key(meta["file_name"]),
                "mask": mask,
                "mask_fraction": float(mask.mean()),
                "has_vitiligo": bool(mask.any()),
            }
            records[split].append(row)
            if split == "train":
                source_groups[row["source_key"]].append(row)

    held_out_groups = evenly_spaced_groups(source_groups, test_groups)
    split_rows = {"train": [], "val": [], "test": []}
    excluded_test_augmentations = 0
    for row in records["train"]:
        if row["source_key"] in held_out_groups:
            continue
        split_rows["train"].append(row)
    for row in records["valid"]:
        split_rows["val"].append(row)
    for key in sorted(held_out_groups):
        variants = sorted(source_groups[key], key=lambda row: row["filename"])
        split_rows["test"].append(variants[0])
        excluded_test_augmentations += len(variants) - 1

    used_stems: set[str] = set()
    manifest = {"source": str(source), "splits": {}, "test_source_groups": sorted(held_out_groups)}
    for split, rows in split_rows.items():
        names = []
        group_keys = set()
        for row in rows:
            stem = row["stem"]
            if stem in used_stems:
                raise ValueError(f"Duplicate output stem: {stem}")
            used_stems.add(stem)
            shutil.copy2(row["source_path"], image_output / row["filename"])
            if not cv2.imwrite(str(mask_output / f"{stem}.png"), row["mask"]):
                raise OSError(f"Could not write mask for {stem}")
            names.append(row["filename"])
            group_keys.add(row["source_key"])
        (split_output / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        manifest["splits"][split] = {
            "images": len(rows),
            "source_groups": len(group_keys),
            "positive_images": sum(row["has_vitiligo"] for row in rows),
            "negative_images": sum(not row["has_vitiligo"] for row in rows),
            "mask_fraction_min": min(row["mask_fraction"] for row in rows),
            "mask_fraction_median": float(np.median([row["mask_fraction"] for row in rows])),
            "mask_fraction_max": max(row["mask_fraction"] for row in rows),
        }
    manifest["excluded_test_augmentations"] = excluded_test_augmentations
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-groups", type=int, default=20)
    args = parser.parse_args()
    prepare(args.source, args.output, args.test_groups)


if __name__ == "__main__":
    main()

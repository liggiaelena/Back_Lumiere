"""Add face crops as training-only augmentations without changing val/test."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def merge(base: Path, face: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    for name in ("images", "masks", "splits"):
        (output / name).mkdir(parents=True, exist_ok=True)

    manifest = {"base": str(base), "face_augmentations": str(face), "splits": {}}
    for split in ("train", "val", "test"):
        base_items = [line.strip() for line in (base / "splits" / f"{split}.txt").read_text().splitlines() if line.strip()]
        output_items = []
        for item in base_items:
            shutil.copy2(base / "images" / item, output / "images" / item)
            shutil.copy2(base / "masks" / f"{Path(item).stem}.png", output / "masks" / f"{Path(item).stem}.png")
            output_items.append(item)
        face_count = 0
        if split == "train":
            face_items = [line.strip() for line in (face / "splits" / "train.txt").read_text().splitlines() if line.strip()]
            for item in face_items:
                output_name = f"face__{item}"
                output_stem = Path(output_name).stem
                shutil.copy2(face / "images" / item, output / "images" / output_name)
                shutil.copy2(face / "masks" / f"{Path(item).stem}.png", output / "masks" / f"{output_stem}.png")
                output_items.append(output_name)
                face_count += 1
        (output / "splits" / f"{split}.txt").write_text("\n".join(output_items) + "\n", encoding="utf-8")
        manifest["splits"][split] = {
            "base_images": len(base_items),
            "face_augmentations": face_count,
            "total": len(output_items),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--face", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.base, args.face, args.output)


if __name__ == "__main__":
    main()

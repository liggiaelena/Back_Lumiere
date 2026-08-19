"""Create deterministic train/val/test splits for normal-face negatives.

The split entries are absolute image paths, so large source images do not need
to be copied. Disease-vs-rest training creates empty target masks for this
dataset automatically.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def prepare(source: Path, output: Path, train: int, val: int, test: int, seed: int) -> None:
    images = sorted(
        path.resolve()
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    requested = train + val + test
    if len(images) < requested:
        raise ValueError(f"Requested {requested} images, but only {len(images)} are available")

    random.Random(seed).shuffle(images)
    selected = {
        "train": images[:train],
        "val": images[train:train + val],
        "test": images[train + val:requested],
    }
    splits_dir = output / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, paths in selected.items():
        (splits_dir / f"{split}.txt").write_text(
            "\n".join(str(path) for path in paths) + "\n",
            encoding="utf-8",
        )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "seed": seed,
                "normal_images": {name: len(paths) for name, paths in selected.items()},
                "note": "Assumed normal negatives; manually audit deployment evaluation subsets.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()

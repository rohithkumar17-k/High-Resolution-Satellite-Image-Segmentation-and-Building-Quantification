from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split prepared image/mask tiles into train/validation/test sets.")
    parser.add_argument("--tiles", required=True, help="Directory containing images/ and masks/.")
    parser.add_argument("--output", required=True, help="Output directory for train/val/test image and mask folders.")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.val < 0 or args.test < 0 or args.val + args.test >= 1:
        raise ValueError("val and test fractions must be non-negative and sum to less than 1")

    root = Path(args.tiles)
    images = sorted((root / "images").glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG tiles found in {root / 'images'}")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    n = len(images)
    n_test = int(n * args.test)
    n_val = int(n * args.val)
    groups = {"test": images[:n_test], "val": images[n_test:n_test + n_val], "train": images[n_test + n_val:]}

    for split, files in groups.items():
        image_out = Path(args.output) / split / "images"
        mask_out = Path(args.output) / split / "masks"
        image_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        for image in files:
            mask = root / "masks" / image.name
            if not mask.exists():
                raise FileNotFoundError(f"Missing mask for {image.name}")
            shutil.copy2(image, image_out / image.name)
            shutil.copy2(mask, mask_out / mask.name)
        print(f"{split}: {len(files)} tiles")


if __name__ == "__main__":
    main()

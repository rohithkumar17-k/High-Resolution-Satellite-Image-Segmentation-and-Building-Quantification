from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


def source_scene_name(path: Path) -> str:
    """Recover the source-scene identifier from a tiled PNG filename."""
    return re.sub(r"_x\d+_y\d+\.png$", "", path.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split prepared image/mask tiles by source scene to prevent spatial leakage."
    )
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

    # Group all overlapping tiles from the same source scene together.
    scenes: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        scenes[source_scene_name(image)].append(image)

    scene_names = sorted(scenes)
    rng = random.Random(args.seed)
    rng.shuffle(scene_names)

    n = len(scene_names)
    n_test = int(n * args.test)
    n_val = int(n * args.val)
    groups = {
        "test": scene_names[:n_test],
        "val": scene_names[n_test:n_test + n_val],
        "train": scene_names[n_test + n_val:],
    }

    for split, split_scenes in groups.items():
        image_out = Path(args.output) / split / "images"
        mask_out = Path(args.output) / split / "masks"
        image_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        tile_count = 0
        for scene in split_scenes:
            for image in scenes[scene]:
                mask = root / "masks" / image.name
                if not mask.exists():
                    raise FileNotFoundError(f"Missing mask for {image.name}")
                shutil.copy2(image, image_out / image.name)
                shutil.copy2(mask, mask_out / mask.name)
                tile_count += 1

        print(f"{split}: {len(split_scenes)} source scenes, {tile_count} tiles")


if __name__ == "__main__":
    main()

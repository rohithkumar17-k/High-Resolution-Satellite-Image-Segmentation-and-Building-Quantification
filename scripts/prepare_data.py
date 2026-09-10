from __future__ import annotations

import argparse
from pathlib import Path

from src.tiling import tile_directory
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize SpaceNet GeoJSON labels and create overlapping training tiles.")
    parser.add_argument("--images", required=True, help="Directory containing RGB GeoTIFF scenes.")
    parser.add_argument("--annotations", required=True, help="Directory containing matching GeoJSON building labels.")
    parser.add_argument("--output", default=None, help="Tile output directory. Defaults to paths.tiles from config.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tiling = config["tiling"]
    output = args.output or config["paths"]["tiles"]
    count = tile_directory(
        args.images,
        args.annotations,
        output,
        tile_size=tiling["tile_size"],
        overlap=tiling["overlap"],
        min_foreground_fraction=tiling["min_foreground_fraction"],
    )
    print(f"Created {count} image/mask tile pairs under {Path(output)}")


if __name__ == "__main__":
    main()

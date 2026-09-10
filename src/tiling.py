from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from tqdm import tqdm


@dataclass(frozen=True)
class TileWindow:
    x: int
    y: int
    width: int
    height: int


def make_windows(height: int, width: int, tile_size: int = 512, overlap: float = 0.25) -> list[TileWindow]:
    """Create full-coverage windows, anchoring final tiles to each image edge."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")
    stride = max(1, int(tile_size * (1 - overlap)))
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if xs[-1] != max(0, width - tile_size): xs.append(max(0, width - tile_size))
    if ys[-1] != max(0, height - tile_size): ys.append(max(0, height - tile_size))
    return [TileWindow(x, y, min(tile_size, width - x), min(tile_size, height - y)) for y in ys for x in xs]


def read_rgb(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        image = src.read([1, 2, 3]).transpose(1, 2, 0)
        profile = src.profile.copy()
    # uint16 SpaceNet images are robustly displayed/trained after percentile scaling.
    if image.dtype != np.uint8:
        lo, hi = np.percentile(image, (1, 99))
        image = np.clip((image - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    return image, profile


def geojson_mask(annotation_path: str | Path, image_shape: tuple[int, int], transform) -> np.ndarray:
    import json
    with open(annotation_path, encoding="utf-8") as handle:
        features = json.load(handle).get("features", [])
    geometries = [(shape(item["geometry"]), 1) for item in features if item.get("geometry")]
    return rasterize(geometries, out_shape=image_shape, transform=transform, fill=0, dtype=np.uint8)


def tile_image_and_mask(image: np.ndarray, mask: np.ndarray, tile_size: int, overlap: float):
    height, width = mask.shape
    for window in make_windows(height, width, tile_size, overlap):
        image_tile = image[window.y:window.y + window.height, window.x:window.x + window.width]
        mask_tile = mask[window.y:window.y + window.height, window.x:window.x + window.width]
        pad_h, pad_w = tile_size - window.height, tile_size - window.width
        if pad_h or pad_w:
            image_tile = cv2.copyMakeBorder(image_tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask_tile = cv2.copyMakeBorder(mask_tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        yield window, image_tile, mask_tile


def create_tiles(image_path: str | Path, annotation_path: str | Path, output_dir: str | Path,
                 tile_size: int = 512, overlap: float = 0.25, min_foreground_fraction: float = 0.0) -> int:
    output_dir = Path(output_dir); images_dir = output_dir / "images"; masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True); masks_dir.mkdir(parents=True, exist_ok=True)
    image, _ = read_rgb(image_path)
    with rasterio.open(image_path) as src:
        mask = geojson_mask(annotation_path, image.shape[:2], src.transform)
    stem, count = Path(image_path).stem, 0
    for tile_index, (window, image_tile, mask_tile) in enumerate(tile_image_and_mask(image, mask, tile_size, overlap)):
        if mask_tile.mean() < min_foreground_fraction and tile_index % 3 != 0:
            continue  # retain background, but avoid overwhelming class imbalance
        name = f"{stem}_x{window.x}_y{window.y}.png"
        cv2.imwrite(str(images_dir / name), cv2.cvtColor(image_tile, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(masks_dir / name), mask_tile * 255)
        count += 1
    return count


def tile_directory(images_dir: str | Path, annotations_dir: str | Path, output_dir: str | Path, **kwargs) -> int:
    images = sorted(Path(images_dir).glob("*.tif")) + sorted(Path(images_dir).glob("*.tiff"))
    annotations = Path(annotations_dir); total = 0
    for image_path in tqdm(images, desc="Tiling images"):
        candidates = list(annotations.glob(f"*{image_path.stem}*.geojson"))
        if not candidates:
            raise FileNotFoundError(f"No GeoJSON annotation matching {image_path.name}")
        total += create_tiles(image_path, candidates[0], output_dir, **kwargs)
    return total

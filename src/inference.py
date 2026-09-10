from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import rasterio
import torch

from .tiling import make_windows, read_rgb


@torch.no_grad()
def predict_large_image(model, image_path: str | Path, device, tile_size=512, overlap=0.25) -> tuple[np.ndarray, np.ndarray, float]:
    """Blend overlapping tile probabilities to preserve seamless whole-image output."""
    image, _ = read_rgb(image_path); height, width = image.shape[:2]
    total, weights = np.zeros((height, width), np.float32), np.zeros((height, width), np.float32)
    model.eval(); start = time.perf_counter()
    for window in make_windows(height, width, tile_size, overlap):
        tile = image[window.y:window.y + window.height, window.x:window.x + window.width]
        padded = cv2.copyMakeBorder(tile, 0, tile_size - window.height, 0, tile_size - window.width, cv2.BORDER_REFLECT_101)
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()[:window.height, :window.width]
        total[window.y:window.y + window.height, window.x:window.x + window.width] += probability
        weights[window.y:window.y + window.height, window.x:window.x + window.width] += 1
    return image, total / np.maximum(weights, 1), time.perf_counter() - start


def save_geotiff(mask: np.ndarray, reference_path: str | Path, output_path: str | Path) -> None:
    with rasterio.open(reference_path) as source:
        profile = source.profile.copy(); profile.update(count=1, dtype=rasterio.uint8, nodata=0)
    with rasterio.open(output_path, "w", **profile) as destination: destination.write(mask.astype(np.uint8), 1)

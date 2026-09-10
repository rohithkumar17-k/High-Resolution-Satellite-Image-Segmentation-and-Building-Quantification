from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def postprocess(probability: np.ndarray, threshold: float = 0.5, min_area_px: int = 20,
                opening_kernel: int = 0, closing_kernel: int = 3) -> np.ndarray:
    mask = (probability >= threshold).astype(np.uint8)
    if opening_kernel > 1:
        k = np.ones((opening_kernel, opening_kernel), np.uint8); mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if closing_kernel > 1:
        k = np.ones((closing_kernel, closing_kernel), np.uint8); mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area_px: clean[labels == label] = 1
    return clean


def quantify_buildings(mask: np.ndarray, pixel_size_m: float | None = None) -> tuple[pd.DataFrame, dict]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    rows = []
    for label in range(1, count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        row = {"building_id": label, "area_px": area_px, "centroid_x_px": float(centroids[label][0]), "centroid_y_px": float(centroids[label][1])}
        if pixel_size_m: row["area_m2"] = area_px * pixel_size_m**2
        rows.append(row)
    buildings = pd.DataFrame(rows)
    summary = {"building_count": len(rows), "building_coverage_percent": float(mask.mean() * 100), "total_building_area_px": int(mask.sum())}
    if pixel_size_m: summary["total_building_area_m2"] = float(mask.sum() * pixel_size_m**2)
    return buildings, summary

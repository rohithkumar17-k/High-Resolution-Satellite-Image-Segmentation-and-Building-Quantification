from __future__ import annotations

import numpy as np
import torch


def binary_metrics(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = np.asarray(prediction >= threshold, dtype=bool); truth = np.asarray(target >= 0.5, dtype=bool)
    tp = np.logical_and(pred, truth).sum(); fp = np.logical_and(pred, ~truth).sum(); fn = np.logical_and(~pred, truth).sum()
    eps = 1e-7
    precision = tp / (tp + fp + eps); recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps); iou = tp / (tp + fp + fn + eps)
    return {"iou": float(iou), "dice": float(dice), "precision": float(precision), "recall": float(recall), "f1": float(dice)}


@torch.no_grad()
def evaluate_loader(model, loader, device, threshold=0.5) -> dict[str, float]:
    model.eval(); probabilities, targets = [], []
    for images, masks, _ in loader:
        probabilities.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
        targets.append(masks.numpy())
    return binary_metrics(np.concatenate(probabilities), np.concatenate(targets), threshold)

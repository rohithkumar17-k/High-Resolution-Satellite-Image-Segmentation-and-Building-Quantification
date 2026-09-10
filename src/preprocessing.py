from __future__ import annotations

import albumentations as A


def train_transform(size: int = 512) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3), A.Normalize(),
    ])


def validation_transform() -> A.Compose:
    return A.Compose([A.Normalize()])

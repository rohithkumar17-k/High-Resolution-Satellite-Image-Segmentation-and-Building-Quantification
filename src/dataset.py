from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BuildingTileDataset(Dataset):
    def __init__(self, image_paths: list[str | Path], transform=None):
        self.images = [Path(value) for value in image_paths]
        self.transform = transform
        if not self.images:
            raise ValueError("Dataset is empty.")

    def __len__(self): return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(image_path.parent.parent / "masks" / image_path.name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None: raise FileNotFoundError(image_path)
        if self.transform: transformed = self.transform(image=image, mask=mask); image, mask = transformed["image"], transformed["mask"]
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask[None] / 255.0)).float()
        return image, mask, image_path.name

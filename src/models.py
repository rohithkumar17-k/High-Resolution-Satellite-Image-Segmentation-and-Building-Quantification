from __future__ import annotations

import segmentation_models_pytorch as smp


def build_model(architecture: str, encoder: str, encoder_weights: str | None = "imagenet"):
    common = dict(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)
    normalized = architecture.lower().replace("+", "").replace("_", "")
    if normalized == "unet": return smp.Unet(**common)
    if normalized in {"deeplabv3", "deeplabv3plus"}: return smp.DeepLabV3Plus(**common)
    raise ValueError("architecture must be 'unet' or 'deeplabv3plus'.")

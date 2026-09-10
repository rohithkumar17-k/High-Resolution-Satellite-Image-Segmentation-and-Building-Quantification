from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.inference import predict_large_image, save_geotiff
from src.models import build_model
from src.quantification import postprocess, quantify_buildings
from src.utils import load_config, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run overlapping-tile whole-scene inference and building quantification.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--pixel-size-m", type=float, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(args.config or "configs/default.yaml")
    model_cfg = checkpoint.get("model", config["model"])
    model = build_model(**model_cfg)
    model.load_state_dict(checkpoint["state_dict"])
    device = resolve_device(config["training"]["device"])
    model.to(device)

    tiling = config["tiling"]
    image, probability, seconds = predict_large_image(
        model, args.image, device,
        tile_size=tiling["tile_size"], overlap=tiling["overlap"]
    )
    post_cfg = config["postprocess"]
    mask = postprocess(
        probability,
        threshold=config["training"]["threshold"],
        min_area_px=post_cfg["min_area_px"],
        opening_kernel=post_cfg["opening_kernel"],
        closing_kernel=post_cfg["closing_kernel"],
    )
    buildings, summary = quantify_buildings(mask, args.pixel_size_m)
    summary["inference_seconds"] = seconds

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_geotiff(mask, args.image, out / "building_mask.tif")
    buildings.to_csv(out / "building_instances.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image); axes[0].set_title("Satellite image"); axes[0].axis("off")
    axes[1].imshow(probability, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Blended probability"); axes[1].axis("off")
    axes[2].imshow(image); axes[2].imshow(mask, alpha=0.45, cmap="Reds"); axes[2].set_title("Building mask"); axes[2].axis("off")
    fig.tight_layout(); fig.savefig(out / "overlay.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

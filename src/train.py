from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import BuildingTileDataset
from .evaluate import evaluate_loader
from .models import build_model
from .preprocessing import train_transform, validation_transform
from .utils import ensure_dir, load_config, resolve_device, seed_everything


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5): super().__init__(); self.bce_weight = bce_weight; self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, target):
        bce = self.bce(logits, target); probs = torch.sigmoid(logits)
        dims = (1, 2, 3); dice = (2 * (probs * target).sum(dims) + 1) / (probs.sum(dims) + target.sum(dims) + 1)
        return self.bce_weight * bce + (1 - self.bce_weight) * (1 - dice.mean())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/default.yaml"); parser.add_argument("--train-dir", required=True); parser.add_argument("--val-dir", required=True); args = parser.parse_args()
    config = load_config(args.config); seed_everything(config["seed"]); device = resolve_device(config["training"]["device"])
    train_paths = sorted(Path(args.train_dir).glob("*.png")); val_paths = sorted(Path(args.val_dir).glob("*.png"))
    train_set = BuildingTileDataset(train_paths, train_transform()); val_set = BuildingTileDataset(val_paths, validation_transform())
    workers = config["training"]["num_workers"]; batch = config["training"]["batch_size"]
    train_loader = DataLoader(train_set, batch, shuffle=True, num_workers=workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch, num_workers=workers, pin_memory=device.type == "cuda")
    model = build_model(**config["model"]).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    loss_fn = BCEDiceLoss(config["training"]["bce_weight"]); checkpoint_dir = ensure_dir(config["paths"]["checkpoints"]); best = -1.0
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train(); running = 0.0
        for images, masks, _ in train_loader:
            optimizer.zero_grad(); loss = loss_fn(model(images.to(device)), masks.to(device)); loss.backward(); optimizer.step(); running += loss.item()
        metrics = evaluate_loader(model, val_loader, device, config["training"]["threshold"])
        print(f"epoch={epoch} loss={running / len(train_loader):.4f} " + " ".join(f"{k}={v:.4f}" for k,v in metrics.items()))
        if metrics["iou"] > best:
            best = metrics["iou"]
            torch.save({"state_dict": model.state_dict(), "model": config["model"], "metrics": metrics}, checkpoint_dir / "best_model.pth")
            (checkpoint_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__": main()

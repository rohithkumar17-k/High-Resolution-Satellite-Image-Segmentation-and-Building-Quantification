# High-Resolution Satellite Building Segmentation

An end-to-end, reproducible computer-vision project for extracting and quantifying buildings from large RGB satellite scenes. It tiles high-resolution imagery into overlapping 512×512 patches, trains semantic-segmentation models, blends patch predictions into a whole-image map, extracts individual building instances, and reports coverage and area statistics.

## Highlights

- SpaceNet 2-ready GeoTIFF + GeoJSON data preparation.
- Overlapping tiled training and whole-image reconstruction—no destructive resizing of large scenes.
- Baseline: **U-Net + ResNet18**. Stronger experiment: **DeepLabV3+ + ResNet50**.
- BCE + Dice objective with IoU, Dice, precision, recall, and F1 evaluation.
- Connected-component instance extraction and per-building pixel (and optional m²) areas.
- GeoTIFF prediction output, CSV instance inventory, JSON summary, and overlay visualization.

## Repository layout

```text
├── configs/default.yaml
├── data/                         # ignored; place downloaded SpaceNet data here
├── scripts/
│   ├── prepare_data.py
│   ├── split_tiles.py
│   └── infer_large_image.py
├── src/
│   ├── tiling.py                 # rasterization, tiled extraction
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py              # overlap-blended reconstruction
│   └── quantification.py
└── tests/
```

## Setup

Use Python 3.10+ and install a PyTorch build appropriate for your CPU/CUDA setup, then:

```bash
git clone <your-repository-url>
cd high-resolution-satellite-segmentation
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset preparation

Download the SpaceNet 2 building-detection RGB images and GeoJSON building labels. A common local layout is:

```text
data/raw/
├── RGB-PanSharpen/           # .tif / .tiff images
└── geojson/buildings/        # matching .geojson labels
```

Tile the scenes. Each source image stays spatially intact; tiles are only used as model inputs.

```bash
python scripts/prepare_data.py --images data/raw/RGB-PanSharpen --annotations data/raw/geojson/buildings
python scripts/split_tiles.py --tiles data/processed/tiles --output data/splits
```

For leakage-safe research results, split at the **source-scene/AOI level** before tiling rather than random tile level. `split_tiles.py` is a convenient starter split; use separate source folders when reporting final metrics.

## Train the baseline

```bash
python -m src.train --train-dir data/splits/train/images --val-dir data/splits/val/images
```

The best checkpoint (by validation IoU) is saved to `outputs/checkpoints/best_model.pth`.

## DeepLabV3+ experiment

Copy `configs/default.yaml`, set `architecture: deeplabv3plus` and `encoder: resnet50`, then train with that config:

```bash
python -m src.train --config configs/deeplabv3plus_resnet50.yaml --train-dir data/splits/train/images --val-dir data/splits/val/images
```

Report each model's IoU, Dice, precision, recall, F1, parameter count, and end-to-end whole-image inference time. Keep the same validation scenes and threshold for a fair comparison.

## Whole-image inference and quantification

```bash
python scripts/infer_large_image.py --image data/raw/RGB-PanSharpen/<scene>.tif --checkpoint outputs/checkpoints/best_model.pth --pixel-size-m 0.3
```

Outputs include a binary GeoTIFF/PNG mask, probability-overlay figure, building-instance CSV, and JSON summary. Only pass `--pixel-size-m` when it is verified for the image product; otherwise report pixel areas.

The quantification summary reports the following fields:

- `building_count`: number of retained connected components.
- `building_coverage_percent`: percentage of scene pixels classified as buildings.
- `total_building_area_px`: total building footprint in pixels.
- `total_building_area_m2`: included only when a verified `--pixel-size-m` is supplied.
- `inference_seconds`: tiled whole-scene prediction time, excluding file export.

`configs/default.yaml` uses a 20-pixel minimum connected-component area to remove tiny prediction artifacts. Tune this value to the imagery resolution and smallest building you need to count.

## Method

```text
Large RGB scene → overlapping 512px tiles → segmentation model → probability blending
    → whole-image mask → morphology + connected components → building count / coverage / areas
```

The overlap blend averages every tile prediction at each pixel, reducing seams. Connected components form building instances; nearby/merged buildings remain a known limitation of semantic segmentation and should be discussed when interpreting counts.

## Reproducibility and responsible reporting

- `seed: 42` is configured by default.
- Do not claim a scene size or physical resolution you did not actually process/verify.
- Keep raw data, model checkpoints, and generated outputs out of Git; `.gitignore` already does this.
- Evaluate on held-out scenes, not just held-out overlapping tiles from the same scene.

## Tests

```bash
pytest -q
```

## License

MIT. Check SpaceNet's dataset terms before distributing data or derived artifacts.

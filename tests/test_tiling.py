import numpy as np

from src.tiling import make_windows, tile_image_and_mask


def test_windows_cover_image_edges():
    windows = make_windows(1000, 1200, tile_size=512, overlap=0.25)
    assert windows
    assert max(w.x + w.width for w in windows) == 1200
    assert max(w.y + w.height for w in windows) == 1000


def test_tile_padding_keeps_requested_size():
    image = np.zeros((600, 700, 3), dtype=np.uint8)
    mask = np.zeros((600, 700), dtype=np.uint8)
    for _, image_tile, mask_tile in tile_image_and_mask(image, mask, 512, 0.25):
        assert image_tile.shape == (512, 512, 3)
        assert mask_tile.shape == (512, 512)

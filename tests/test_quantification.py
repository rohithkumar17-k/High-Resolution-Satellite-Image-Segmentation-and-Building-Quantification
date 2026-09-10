import numpy as np

from src.quantification import postprocess, quantify_buildings


def test_small_components_are_removed():
    probability = np.zeros((20, 20), dtype=np.float32)
    probability[2:6, 2:6] = 1
    probability[10, 10] = 1
    mask = postprocess(probability, threshold=0.5, min_area_px=4, closing_kernel=0)
    assert mask[3, 3] == 1
    assert mask[10, 10] == 0


def test_quantification_reports_count_area_and_coverage():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[6:9, 6:8] = 1
    buildings, summary = quantify_buildings(mask, pixel_size_m=0.5)
    assert len(buildings) == 2
    assert summary["building_count"] == 2
    assert summary["total_building_area_px"] == 10
    assert summary["building_coverage_percent"] == 10.0
    assert summary["total_building_area_m2"] == 2.5

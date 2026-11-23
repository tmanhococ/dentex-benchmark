# tests/augment/test_classical_masks.py
from pathlib import Path
import numpy as np
import pytest

from dentex_bench.augment.masks import classical_mask, compute_dice_score, evaluate_mask_methods


def test_dice_score_perfect_and_zero():
    # 1. Identical non-empty masks -> 1.0
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m2 = np.zeros((100, 100), dtype=np.uint8)
    m1[30:70, 30:70] = 255
    m2[30:70, 30:70] = 255
    dice = compute_dice_score(m1, m2)
    assert abs(dice - 1.0) < 1e-4

    # 2. Disjoint masks -> 0.0
    m3 = np.zeros((100, 100), dtype=np.uint8)
    m3[0:20, 0:20] = 255
    dice_disjoint = compute_dice_score(m1, m3)
    assert abs(dice_disjoint - 0.0) < 1e-4

    # 3. Empty masks -> 1.0
    empty1 = np.zeros((100, 100), dtype=np.uint8)
    empty2 = np.zeros((100, 100), dtype=np.uint8)
    assert abs(compute_dice_score(empty1, empty2) - 1.0) < 1e-4

    # 4. One empty, one non-empty -> 0.0
    assert abs(compute_dice_score(m1, empty1) - 0.0) < 1e-4


def test_classical_mask_segmentation():
    # Synthetic tooth crop: dark background (50), bright tooth (200)
    img = np.full((100, 100), 50, dtype=np.uint8)
    img[30:70, 40:60] = 200

    mask = classical_mask(img, bbox=[40, 30, 20, 40])
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    # Center of tooth should be foreground (255)
    assert mask[50, 50] == 255
    # Distant background should be background (0)
    assert mask[10, 10] == 0
    # Values should strictly be binary 0 or 255
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255})


def test_classical_mask_grayscale_and_color():
    # 2D grayscale input
    img_gray = np.full((80, 80), 40, dtype=np.uint8)
    img_gray[25:55, 25:55] = 220
    mask_gray = classical_mask(img_gray, bbox=[25, 25, 30, 30])
    assert mask_gray.shape == (80, 80)
    assert mask_gray[40, 40] == 255
    assert mask_gray[5, 5] == 0

    # 3D 3-channel color input (RGB)
    img_color = np.full((80, 80, 3), 40, dtype=np.uint8)
    img_color[25:55, 25:55, :] = 220
    mask_color = classical_mask(img_color, bbox=[25, 25, 30, 30])
    assert mask_color.shape == (80, 80)
    assert mask_color[40, 40] == 255
    assert mask_color[5, 5] == 0

    # Without bbox (bbox=None)
    mask_no_bbox = classical_mask(img_color)
    assert mask_no_bbox.shape == (80, 80)
    assert mask_no_bbox[40, 40] == 255


def test_evaluate_mask_methods(tmp_path):
    # Create synthetic samples
    crops = []
    gts = []
    for _ in range(3):
        img = np.full((80, 80), 40, dtype=np.uint8)
        img[20:60, 20:60] = 220
        gt = np.zeros((80, 80), dtype=np.uint8)
        gt[20:60, 20:60] = 255
        crops.append(img)
        gts.append(gt)

    out_csv = tmp_path / "mask_eval.csv"
    metrics = evaluate_mask_methods(crops, gts, out_csv=out_csv)

    assert isinstance(metrics, dict)
    assert "mean_dice" in metrics
    assert "median_dice" in metrics
    assert metrics["mean_dice"] > 0.8
    assert metrics["median_dice"] > 0.8
    assert metrics["num_samples"] == 3.0
    assert out_csv.exists()


def test_classical_mask_edge_cases():
    # Uniform image -> returns empty mask
    uniform = np.full((50, 50), 100, dtype=np.uint8)
    mask = classical_mask(uniform)
    assert mask.shape == (50, 50)
    assert np.all(mask == 0)

    # Very small crop (< 5px) -> returns zeros
    tiny = np.ones((4, 4), dtype=np.uint8) * 200
    mask_tiny = classical_mask(tiny)
    assert mask_tiny.shape == (4, 4)
    assert np.all(mask_tiny == 0)


def test_evaluate_mask_methods_polygon_and_errors(tmp_path):
    crop = np.full((60, 60), 30, dtype=np.uint8)
    crop[15:45, 15:45] = 230
    poly = [[15, 15, 45, 15, 45, 45, 15, 45]]

    metrics = evaluate_mask_methods([crop], [poly])
    assert metrics["mean_dice"] > 0.8
    assert metrics["num_samples"] == 1.0

    # Empty list
    empty_metrics = evaluate_mask_methods([], [])
    assert empty_metrics["mean_dice"] == 0.0
    assert empty_metrics["num_samples"] == 0.0

    # Length mismatch
    with pytest.raises(ValueError, match="equal length"):
        evaluate_mask_methods([crop], [])

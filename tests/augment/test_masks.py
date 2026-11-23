# tests/augment/test_masks.py
import numpy as np
from dentex_bench.augment.config import ROIConfig
from dentex_bench.augment.masks import polygon_to_mask, create_class_roi, compute_ring_mask


def test_polygon_to_mask():
    poly = [[10, 10, 50, 10, 50, 60, 10, 60]]
    mask = polygon_to_mask(poly, 100, 100)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert mask[30, 30] == 255
    assert mask[5, 5] == 0


def test_polygon_to_mask_empty():
    mask = polygon_to_mask([], 100, 100)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert np.all(mask == 0)


def test_class_roi_caries_dilation():
    tooth_mask = np.zeros((100, 100), dtype=np.uint8)
    tooth_mask[40:60, 40:60] = 255
    roi_cfg = ROIConfig(caries_dilate_px=4)
    roi_mask = create_class_roi(tooth_mask, "caries", None, roi_cfg)
    
    assert roi_mask[38, 40] == 255
    assert np.sum(roi_mask) > np.sum(tooth_mask)


def test_class_roi_deep_caries():
    tooth_mask = np.zeros((100, 100), dtype=np.uint8)
    tooth_mask[40:60, 40:60] = 255
    roi_cfg = ROIConfig(caries_dilate_px=4)
    roi_mask = create_class_roi(tooth_mask, "deep_caries", None, roi_cfg)
    assert np.sum(roi_mask) > np.sum(tooth_mask)


def test_class_roi_periapical():
    tooth_mask = np.zeros((100, 100), dtype=np.uint8)
    tooth_mask[40:60, 40:60] = 255
    roi_cfg = ROIConfig(periapical_bbox_expand=0.15)
    periapical_bbox = [35.0, 55.0, 30.0, 20.0]  # [x, y, w, h]
    roi_mask = create_class_roi(tooth_mask, "periapical", periapical_bbox, roi_cfg)
    
    assert roi_mask.dtype == np.uint8
    assert np.sum(roi_mask) > np.sum(tooth_mask)
    # Check that region inside periapical_bbox is covered
    assert roi_mask[60, 45] == 255


def test_class_roi_periapical_no_bbox():
    tooth_mask = np.zeros((100, 100), dtype=np.uint8)
    tooth_mask[40:60, 40:60] = 255
    roi_cfg = ROIConfig()
    roi_mask = create_class_roi(tooth_mask, "periapical", None, roi_cfg)
    assert roi_mask.dtype == np.uint8
    assert np.sum(roi_mask) >= np.sum(tooth_mask)


def test_class_roi_impacted():
    tooth_mask = np.zeros((100, 100), dtype=np.uint8)
    tooth_mask[40:60, 40:60] = 255
    roi_cfg = ROIConfig(impacted_dilate_px=9)
    roi_mask = create_class_roi(tooth_mask, "impacted", None, roi_cfg)
    assert np.sum(roi_mask) > np.sum(tooth_mask)
    # Dilation should be wider than caries (9px vs 4px)
    caries_mask = create_class_roi(tooth_mask, "caries", None, ROIConfig(caries_dilate_px=4))
    assert np.sum(roi_mask) > np.sum(caries_mask)


def test_class_roi_healthy_and_fallback():
    tooth_mask = np.zeros((100, 100), dtype=np.uint8)
    tooth_mask[40:60, 40:60] = 255
    roi_cfg = ROIConfig(healthy_dilate_px=4)
    roi_mask_healthy = create_class_roi(tooth_mask, "healthy", None, roi_cfg)
    roi_mask_fallback = create_class_roi(tooth_mask, "unknown_class", None, roi_cfg)
    assert np.array_equal(roi_mask_healthy, roi_mask_fallback)
    assert np.sum(roi_mask_healthy) > np.sum(tooth_mask)


def test_ring_mask_outer():
    roi_mask = np.zeros((100, 100), dtype=np.uint8)
    roi_mask[40:60, 40:60] = 255
    ring = compute_ring_mask(roi_mask, ring_width_px=5)
    # Ring should not intersect with original ROI
    assert np.all(ring[roi_mask == 255] == 0)
    assert np.any(ring == 255)

# tests/augment/test_extras.py
import numpy as np
import pytest

from dentex_bench.augment.extras import apply_photometric, apply_patch_crop


def test_apply_photometric_preserves_shape():
    # 3-channel RGB image
    img_rgb = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    out_rgb = apply_photometric(img_rgb, clahe_clip=2.0, gamma=1.1)
    assert out_rgb.shape == img_rgb.shape
    assert out_rgb.dtype == np.uint8

    # 2D Grayscale image
    img_gray = (np.random.rand(180, 240) * 255).astype(np.uint8)
    out_gray = apply_photometric(img_gray, clahe_clip=2.0, gamma=1.1)
    assert out_gray.shape == img_gray.shape
    assert out_gray.dtype == np.uint8


def test_apply_photometric_grayscale_and_rgb():
    # Grayscale
    img_gray = np.full((100, 100), 100, dtype=np.uint8)
    img_gray[20:80, 20:80] = 180
    out_gray = apply_photometric(img_gray, clahe_clip=3.0, gamma=1.0)
    assert out_gray.shape == (100, 100)
    assert out_gray.dtype == np.uint8
    # CLAHE should alter contrast
    assert not np.array_equal(out_gray, img_gray)

    # RGB
    img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    img_rgb[20:80, 20:80, 0] = 150
    img_rgb[30:70, 30:70, 1] = 200
    img_rgb[10:90, 10:90, 2] = 80
    out_rgb = apply_photometric(img_rgb, clahe_clip=2.5, gamma=1.0)
    assert out_rgb.shape == (100, 100, 3)
    assert out_rgb.dtype == np.uint8
    assert not np.array_equal(out_rgb, img_rgb)


def test_apply_photometric_gamma():
    # Mid-tones image
    img = np.full((50, 50, 3), 128, dtype=np.uint8)

    # Gamma > 1.0 (brightening / expansion of dark-to-mid tones)
    out_bright = apply_photometric(img, clahe_clip=0.0, gamma=1.5)
    assert np.all(out_bright > 128)

    # Gamma < 1.0 (darkening / compression)
    out_dark = apply_photometric(img, clahe_clip=0.0, gamma=0.7)
    assert np.all(out_dark < 128)

    # Gamma == 1.0 with clahe_clip=0.0 should be no-op identity
    out_ident = apply_photometric(img, clahe_clip=0.0, gamma=1.0)
    np.testing.assert_array_equal(out_ident, img)


def test_apply_patch_crop_basic():
    img = np.zeros((1500, 2000, 3), dtype=np.uint8)
    anns = [
        {"id": 1, "category_id": 3, "bbox": [500.0, 500.0, 80.0, 80.0], "area": 6400.0}
    ]
    crop_img, crop_anns = apply_patch_crop(img, anns, crop_size=1024, min_visible=0.6)

    # Output image should have exact crop size
    assert crop_img.shape == (1024, 1024, 3)
    assert len(crop_anns) == 1

    # Anchor center is at (540, 540)
    # x1 = clip(540 - 512, 0, 2000 - 1024) = 28
    # y1 = clip(540 - 512, 0, 1500 - 1024) = 28
    # Shifted bbox should be [500 - 28, 500 - 28, 80, 80] = [472, 472, 80, 80]
    bx, by, bw, bh = crop_anns[0]["bbox"]
    assert pytest.approx(bx) == 472.0
    assert pytest.approx(by) == 472.0
    assert pytest.approx(bw) == 80.0
    assert pytest.approx(bh) == 80.0
    assert pytest.approx(crop_anns[0]["area"]) == 6400.0


def test_apply_patch_crop_filters_truncated_annotations():
    img = np.zeros((1500, 2000, 3), dtype=np.uint8)
    # Anchor at (500, 500) -> crop will be [28:1052, 28:1052]
    anns = [
        {"id": 1, "category_id": 3, "bbox": [500.0, 500.0, 80.0, 80.0], "area": 6400.0},
        # Far outside crop window:
        {"id": 2, "category_id": 2, "bbox": [1800.0, 1300.0, 100.0, 100.0], "area": 10000.0},
        # Straddling right edge: x from 1030 to 1130. Inside crop: 1030 to 1052 (width 22).
        # Visible ratio = (22 * 100) / 10000 = 0.22 < 0.6 -> should be filtered
        {"id": 3, "category_id": 1, "bbox": [1030.0, 500.0, 100.0, 100.0], "area": 10000.0},
        # Straddling top edge: y from 10 to 110. Inside crop: 28 to 110 (height 82).
        # Visible ratio = (100 * 82) / 10000 = 0.82 >= 0.6 -> should be preserved
        {"id": 4, "category_id": 1, "bbox": [500.0, 10.0, 100.0, 100.0], "area": 10000.0},
    ]

    crop_img, crop_anns = apply_patch_crop(img, anns, crop_size=1024, min_visible=0.6)
    kept_ids = [a["id"] for a in crop_anns]

    assert 1 in kept_ids
    assert 4 in kept_ids
    assert 2 not in kept_ids
    assert 3 not in kept_ids

    # For ann 4, check that bbox is clipped properly
    ann_4 = next(a for a in crop_anns if a["id"] == 4)
    # y1 of crop is 28. Box original y was 10, y2 was 110. Inside crop y goes from 28 to 110 -> h=82, y_new=0
    assert pytest.approx(ann_4["bbox"][1]) == 0.0
    assert pytest.approx(ann_4["bbox"][3]) == 82.0
    assert pytest.approx(ann_4["area"]) == 8200.0


def test_apply_patch_crop_updates_segmentation():
    img = np.zeros((1500, 2000, 3), dtype=np.uint8)
    # Anchor at (500, 500) -> crop origin is (28, 28)
    poly = [500.0, 500.0, 580.0, 500.0, 580.0, 580.0, 500.0, 580.0]
    anns = [
        {
            "id": 1,
            "category_id": 3,
            "bbox": [500.0, 500.0, 80.0, 80.0],
            "area": 6400.0,
            "segmentation": [poly],
        }
    ]

    crop_img, crop_anns = apply_patch_crop(img, anns, crop_size=1024, min_visible=0.6)
    assert len(crop_anns) == 1
    new_seg = crop_anns[0]["segmentation"]
    assert len(new_seg) == 1
    new_poly = new_seg[0]

    # Origin offset is (28, 28)
    expected_poly = [
        500.0 - 28.0, 500.0 - 28.0,
        580.0 - 28.0, 500.0 - 28.0,
        580.0 - 28.0, 580.0 - 28.0,
        500.0 - 28.0, 580.0 - 28.0,
    ]
    np.testing.assert_allclose(new_poly, expected_poly)


def test_apply_patch_crop_smaller_image():
    # When image is smaller than crop_size, returns copy unchanged
    img = np.ones((500, 600, 3), dtype=np.uint8)
    anns = [{"id": 1, "bbox": [10.0, 10.0, 50.0, 50.0], "area": 2500.0}]
    crop_img, crop_anns = apply_patch_crop(img, anns, crop_size=1024)
    assert crop_img.shape == (500, 600, 3)
    assert crop_anns == anns


def test_apply_patch_crop_prioritizes_synthetic_anchor():
    img = np.zeros((1500, 2000, 3), dtype=np.uint8)
    anns = [
        {"id": 1, "synthetic": False, "bbox": [100.0, 100.0, 50.0, 50.0], "area": 2500.0},
        {"id": 2, "synthetic": True, "bbox": [1200.0, 800.0, 60.0, 60.0], "area": 3600.0},
    ]
    # Crop should center around synthetic annotation (id 2) at (1230, 830)
    crop_img, crop_anns = apply_patch_crop(img, anns, crop_size=1024)
    kept_ids = [a["id"] for a in crop_anns]
    assert 2 in kept_ids
    # ann 1 at (100, 100) will be far outside
    assert 1 not in kept_ids

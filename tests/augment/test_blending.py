# tests/augment/test_blending.py
import numpy as np
import pytest
from dentex_bench.augment.blending import blend_images


def test_blend_hard():
    src = np.full((100, 100, 3), 200, dtype=np.uint8)
    dst = np.full((100, 100, 3), 50, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    out = blend_images(src, dst, mask, mode="hard")
    assert np.all(out[50, 50] == 200)
    assert np.all(out[10, 10] == 50)


def test_blend_core_preserving():
    src = np.full((100, 100, 3), 220, dtype=np.uint8)
    dst = np.full((100, 100, 3), 30, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    out = blend_images(src, dst, mask, mode="core_preserving", core_erode_px=6, ring_mode="feather")
    # Inside core (eroded mask), pixels MUST be bit-exact source pixels
    assert np.all(out[50, 50] == 220)
    assert np.all(out[10, 10] == 30)


def test_blend_feather():
    src = np.full((100, 100, 3), 200, dtype=np.uint8)
    dst = np.full((100, 100, 3), 50, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    out = blend_images(src, dst, mask, mode="feather", feather_sigma=3.0)
    assert out.shape == dst.shape
    # Center should be close to 200, far outside close to 50
    assert np.all(out[50, 50] > 180)
    assert np.all(out[10, 10] == 50)


def test_blend_poisson_modes():
    src = np.full((100, 100, 3), 180, dtype=np.uint8)
    dst = np.full((100, 100, 3), 60, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    out_norm = blend_images(src, dst, mask, mode="poisson_normal")
    assert out_norm.shape == dst.shape

    out_mixed = blend_images(src, dst, mask, mode="poisson_mixed")
    assert out_mixed.shape == dst.shape

    # Test empty mask fallback for poisson
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    out_empty = blend_images(src, dst, empty_mask, mode="poisson_normal")
    np.testing.assert_array_equal(out_empty, dst)


def test_blend_unknown_mode():
    src = np.full((50, 50, 3), 200, dtype=np.uint8)
    dst = np.full((50, 50, 3), 50, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unknown blending mode"):
        blend_images(src, dst, mask, mode="invalid_mode")


def test_blend_grayscale_2d():
    src = np.full((100, 100), 210, dtype=np.uint8)
    dst = np.full((100, 100), 40, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    for mode in ["hard", "feather", "poisson_normal"]:
        out = blend_images(src, dst, mask, mode=mode)
        assert out.shape == (100, 100)
        assert out.dtype == np.uint8

    out_core = blend_images(src, dst, mask, mode="core_preserving", core_erode_px=6, ring_mode="poisson_normal")
    assert out_core.shape == (100, 100)
    assert out_core.dtype == np.uint8
    # Inside core, exact src pixel
    assert out_core[50, 50] == 210
    # Far outside mask, exact dst pixel
    assert out_core[10, 10] == 40


def test_blend_single_channel_3d():
    src = np.full((100, 100, 1), 210, dtype=np.uint8)
    dst = np.full((100, 100, 1), 40, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    for mode in ["hard", "feather", "poisson_normal"]:
        out = blend_images(src, dst, mask, mode=mode)
        assert out.shape == (100, 100, 1)

    out_core = blend_images(src, dst, mask, mode="core_preserving", core_erode_px=6, ring_mode="poisson_normal")
    assert out_core.shape == (100, 100, 1)
    assert out_core[50, 50, 0] == 210
    assert out_core[10, 10, 0] == 40


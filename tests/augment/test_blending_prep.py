import numpy as np
from dentex_bench.augment.blending import (
    inpaint_target_tooth,
    compute_intensity_match_params,
    apply_intensity_match,
)


def test_inpaint_target_tooth():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Put a black spot
    img[40:60, 40:60] = 0
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255

    inpainted = inpaint_target_tooth(img, mask, radius=5)
    # The black spot should be filled with neighbor intensity ~128
    assert np.all(inpainted[45:55, 45:55] > 50)


def test_inpaint_target_tooth_grayscale():
    img = np.full((100, 100), 128, dtype=np.uint8)
    img[40:60, 40:60] = 0
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255

    inpainted = inpaint_target_tooth(img, mask, radius=5)
    assert np.all(inpainted[45:55, 45:55] > 50)


def test_intensity_matching_linear():
    src = np.full((100, 100, 3), 100, dtype=np.uint8)
    dst = np.full((100, 100, 3), 150, dtype=np.uint8)
    ring_src = np.ones((100, 100), dtype=np.uint8) * 255
    ring_dst = np.ones((100, 100), dtype=np.uint8) * 255

    params = compute_intensity_match_params(src, dst, ring_src, ring_dst, mode="meanstd")
    assert params["mode"] == "meanstd"

    matched = apply_intensity_match(src, params)
    # Matched mean should be close to 150
    assert abs(np.mean(matched) - 150.0) < 5.0


def test_intensity_matching_histogram():
    src = np.full((100, 100, 3), 80, dtype=np.uint8)
    dst = np.full((100, 100, 3), 180, dtype=np.uint8)
    ring_src = np.ones((100, 100), dtype=np.uint8) * 255
    ring_dst = np.ones((100, 100), dtype=np.uint8) * 255

    params = compute_intensity_match_params(src, dst, ring_src, ring_dst, mode="histogram")
    assert params["mode"] == "histogram"
    assert "lut" in params
    matched = apply_intensity_match(src, params)
    assert np.mean(matched) > 100


def test_intensity_matching_none_or_empty():
    src = np.full((100, 100, 3), 80, dtype=np.uint8)
    dst = np.full((100, 100, 3), 180, dtype=np.uint8)
    ring_src = np.zeros((100, 100), dtype=np.uint8)
    ring_dst = np.zeros((100, 100), dtype=np.uint8)

    params = compute_intensity_match_params(src, dst, ring_src, ring_dst, mode="none")
    assert params["mode"] == "none"
    matched = apply_intensity_match(src, params)
    assert np.array_equal(matched, src)

    # Empty rings with meanstd should fallback to none
    params_empty = compute_intensity_match_params(src, dst, ring_src, ring_dst, mode="meanstd")
    assert params_empty["mode"] == "none"
    matched_empty = apply_intensity_match(src, params_empty)
    assert np.array_equal(matched_empty, src)

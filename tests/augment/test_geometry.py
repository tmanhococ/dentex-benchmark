# tests/augment/test_geometry.py
import numpy as np
import cv2
from dentex_bench.augment.geometry import (
    compute_tooth_pca,
    compute_affine_matrix,
    warp_image,
    warp_mask,
    warp_bbox,
    warp_polygons,
)


def test_compute_tooth_pca_vertical_ellipse():
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Draw vertical ellipse: axis lengths 60 (y) and 20 (x)
    cv2.ellipse(mask, (100, 100), (20, 60), 0, 0, 360, 255, -1)
    angle, major, minor = compute_tooth_pca(mask)
    # Near 90 or -90 degrees
    assert abs(abs(angle) - 90.0) < 5.0
    assert major > minor


def test_compute_tooth_pca_horizontal_ellipse():
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Draw horizontal ellipse: axis lengths 60 (x) and 20 (y)
    cv2.ellipse(mask, (100, 100), (60, 20), 0, 0, 360, 255, -1)
    angle, major, minor = compute_tooth_pca(mask)
    # Near 0 or 180 degrees
    assert abs(angle) < 5.0 or abs(abs(angle) - 180.0) < 5.0
    assert major > minor


def test_compute_tooth_pca_empty_or_tiny_mask():
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    angle, major, minor = compute_tooth_pca(empty_mask)
    assert angle == 0.0
    assert major == 1.0
    assert minor == 1.0

    tiny_mask = np.zeros((100, 100), dtype=np.uint8)
    tiny_mask[50, 50] = 255
    angle, major, minor = compute_tooth_pca(tiny_mask)
    assert angle == 0.0
    assert major == 1.0
    assert minor == 1.0


def test_affine_warp_bbox_and_mask_consistency():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    bbox = [40.0, 40.0, 20.0, 20.0]

    # Translate by (+10, +20)
    M = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float32)
    warped_mask = warp_mask(mask, M, (100, 100))
    warped_bbox = warp_bbox(bbox, M)

    assert warped_mask[65, 55] == 255
    assert warped_bbox == [50.0, 60.0, 20.0, 20.0]


def test_compute_affine_matrix_identity():
    # Identity transform: same center, same angle, scale 1.0, no mirror
    M = compute_affine_matrix(
        src_center=(50.0, 50.0),
        dst_center=(50.0, 50.0),
        src_angle=0.0,
        dst_angle=0.0,
        scale_x=1.0,
        scale_y=1.0,
        mirror=False,
    )
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    assert np.allclose(M, expected, atol=1e-5)


def test_compute_affine_matrix_mirror():
    # Mirror transform horizontally around center
    M = compute_affine_matrix(
        src_center=(50.0, 50.0),
        dst_center=(50.0, 50.0),
        src_angle=0.0,
        dst_angle=0.0,
        scale_x=1.0,
        scale_y=1.0,
        mirror=True,
    )
    # Point (60, 50) is 10 units right of center -> mirrored should be (40, 50)
    pt = np.array([60.0, 50.0, 1.0], dtype=np.float32)
    warped_pt = M @ pt
    assert np.allclose(warped_pt, [40.0, 50.0], atol=1e-5)
    # Y-coordinate unchanged (no vertical flip, INV-3)
    pt_y = np.array([50.0, 70.0, 1.0], dtype=np.float32)
    warped_pt_y = M @ pt_y
    assert np.isclose(warped_pt_y[1], 70.0, atol=1e-5)


def test_warp_image_grayscale_and_color():
    gray = np.full((100, 100), 128, dtype=np.uint8)
    color = np.full((100, 100, 3), 128, dtype=np.uint8)

    M = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 10.0]], dtype=np.float32)
    warped_gray = warp_image(gray, M, (100, 100))
    warped_color = warp_image(color, M, (100, 100))

    assert warped_gray.shape == (100, 100)
    assert warped_color.shape == (100, 100, 3)
    assert warped_gray[50, 50] == 128
    assert np.all(warped_color[50, 50] == [128, 128, 128])
    # Borders should be 0
    assert warped_gray[5, 5] == 0
    assert np.all(warped_color[5, 5] == [0, 0, 0])


def test_warp_polygons():
    polys = [[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]]
    # Shift +5 x, +10 y
    M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 10.0]], dtype=np.float32)
    warped = warp_polygons(polys, M)

    expected = [[15.0, 20.0, 25.0, 20.0, 25.0, 30.0, 15.0, 30.0]]
    assert len(warped) == 1
    assert np.allclose(warped[0], expected[0], atol=1e-5)

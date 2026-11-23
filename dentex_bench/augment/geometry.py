from typing import List, Tuple
import cv2
import numpy as np


def compute_tooth_pca(mask: np.ndarray) -> Tuple[float, float, float]:
    """Compute principal axis orientation (degrees) and major/minor lengths via PCA."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0, 1.0, 1.0

    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=np.empty((0)))

    # Principal axis vector (largest eigenvalue)
    v1 = eigenvectors[0]
    angle_rad = np.arctan2(v1[1], v1[0])
    angle_deg = float(np.degrees(angle_rad))

    # 4 * sqrt(eigenvalue) approximate dimensions (2 standard deviations each side)
    major = float(4.0 * np.sqrt(max(1e-4, eigenvalues[0, 0])))
    minor = float(4.0 * np.sqrt(max(1e-4, eigenvalues[1, 0])))
    return angle_deg, major, minor


def compute_affine_matrix(
    src_center: Tuple[float, float],
    dst_center: Tuple[float, float],
    src_angle: float,
    dst_angle: float,
    scale_x: float,
    scale_y: float,
    mirror: bool = False,
) -> np.ndarray:
    """Build unified 2x3 affine matrix aligning source patch to target position."""
    sx, sy = src_center
    dx, dy = dst_center
    d_theta = np.radians(dst_angle - src_angle)

    # 1. Translate src_center to origin
    T1 = np.array([[1, 0, -sx], [0, 1, -sy], [0, 0, 1]], dtype=np.float32)
    # 2. Optional horizontal mirror across tooth axis (INV-3: no vertical flip)
    M_mir = np.array([[-1 if mirror else 1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    # 3. Scale
    S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)
    # 4. Rotate
    cos_t, sin_t = np.cos(d_theta), np.sin(d_theta)
    R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
    # 5. Translate to dst_center
    T2 = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)

    H = T2 @ R @ S @ M_mir @ T1
    return H[:2, :]


def warp_image(img: np.ndarray, M: np.ndarray, dsize: Tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(
        img, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def warp_mask(mask: np.ndarray, M: np.ndarray, dsize: Tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(
        mask, M, dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def warp_bbox(bbox: List[float], M: np.ndarray) -> List[float]:
    x, y, w, h = bbox
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    warped_corners = (M @ np.hstack([corners, ones]).T).T

    min_x, min_y = np.min(warped_corners, axis=0)
    max_x, max_y = np.max(warped_corners, axis=0)
    return [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]


def warp_polygons(polygons: List[List[float]], M: np.ndarray) -> List[List[float]]:
    warped_polys = []
    for poly in polygons:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        ones = np.ones((len(pts), 1), dtype=np.float32)
        warped_pts = (M @ np.hstack([pts, ones]).T).T
        warped_polys.append(warped_pts.reshape(-1).tolist())
    return warped_polys

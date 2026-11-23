# dentex_bench/augment/masks.py
"""Mask extraction and class-specific ROI processing for dental images."""

from typing import List, Optional
import cv2
import numpy as np
from dentex_bench.augment.config import ROIConfig


def polygon_to_mask(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    """Render list of polygon coordinates [[x1, y1, x2, y2, ...]] to uint8 binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        if len(poly) >= 6:  # Need at least 3 points (x, y)
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def create_class_roi(
    tooth_mask: np.ndarray,
    roi_class: str,
    periapical_bbox: Optional[List[float]],
    roi_cfg: ROIConfig,
) -> np.ndarray:
    """Generate class-specific ROI mask according to Section 4.2 of SPEC."""
    h, w = tooth_mask.shape
    roi_class = roi_class.lower()

    if roi_class in ("caries", "deep_caries"):
        k = max(1, roi_cfg.caries_dilate_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(tooth_mask, kernel)

    elif roi_class == "periapical":
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = tooth_mask.copy()
        if periapical_bbox and len(periapical_bbox) == 4:
            bx, by, bw, bh = periapical_bbox
            expand = roi_cfg.periapical_bbox_expand
            ex1 = max(0, int(bx - bw * expand))
            ey1 = max(0, int(by - bh * expand))
            ex2 = min(w, int(bx + bw * (1.0 + expand)))
            ey2 = min(h, int(by + bh * (1.0 + expand)))
            combined[ey1:ey2, ex1:ex2] = 255
        return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    elif roi_class == "impacted":
        k = max(1, roi_cfg.impacted_dilate_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(tooth_mask, kernel)

    else:  # healthy or fallback
        k = max(1, roi_cfg.healthy_dilate_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(tooth_mask, kernel)


def compute_ring_mask(roi_mask: np.ndarray, ring_width_px: int = 15) -> np.ndarray:
    """Compute outer ring of width ring_width_px around roi_mask: dilate(M, w) \\ M."""
    k = max(1, ring_width_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(roi_mask, kernel)
    ring = cv2.subtract(dilated, roi_mask)
    return ring

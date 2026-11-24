# dentex_bench/augment/masks.py
"""Mask extraction and class-specific ROI processing for dental images."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
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


def compute_dice_score(m1: np.ndarray, m2: np.ndarray) -> float:
    """Compute Dice similarity coefficient between two binary masks.

    Formula: 2 * |m1 & m2| / (|m1| + |m2|).
    Returns 1.0 when both masks are empty.
    """
    b1 = m1 > 0
    b2 = m2 > 0
    intersection = np.logical_and(b1, b2).sum()
    total = b1.sum() + b2.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def classical_mask(img_crop: np.ndarray, bbox: Optional[List[float]] = None) -> np.ndarray:
    """Classical image processing mask generator following Section 4.3 of SPEC:

    1. CLAHE enhancement (clipLimit=2.0, tileGridSize=(8, 8))
    2. Gaussian blur (ksize=(5, 5), sigma=1.0)
    3. Otsu thresholding
    4. Morphology opening (k=3) & closing (k=7) + largest connected component + fill holes
    5. GrabCut (5 iterations, initializing with foreground/background seeds, border as background)

    Returns clean binary uint8 mask (0 or 255).
    """
    # 0. Handle empty or degenerate crops safely
    if img_crop is None or img_crop.size == 0:
        return np.zeros((10, 10), dtype=np.uint8)
    if img_crop.ndim < 2 or img_crop.shape[0] < 5 or img_crop.shape[1] < 5:
        h = max(1, img_crop.shape[0]) if img_crop.ndim >= 1 else 10
        w = max(1, img_crop.shape[1]) if img_crop.ndim >= 2 else 10
        return np.zeros((h, w), dtype=np.uint8)

    # Handle grayscale/color and type normalization
    if img_crop.ndim == 3:
        if img_crop.shape[2] == 4:
            gray = cv2.cvtColor(img_crop, cv2.COLOR_RGBA2GRAY)
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif img_crop.shape[2] == 3:
            gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
            color = img_crop.copy()
        else:
            gray = img_crop[:, :, 0]
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray = img_crop.copy()
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]
    if h < 5 or w < 5 or gray.max() == gray.min():
        return np.zeros((h, w), dtype=np.uint8)

    # 1. CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)

    # 3. Otsu thresholding
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Morphology opening & closing + largest CC + fill holes
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels > 1:
        max_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest_cc = (labels == max_label).astype(np.uint8) * 255
    else:
        largest_cc = closed

    contours, _ = cv2.findContours(largest_cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((h, w), dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # 5. GrabCut initialization
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Outer border (3px) is definite background
    gc_mask[:3, :] = cv2.GC_BGD
    gc_mask[-3:, :] = cv2.GC_BGD
    gc_mask[:, :3] = cv2.GC_BGD
    gc_mask[:, -3:] = cv2.GC_BGD

    # If bbox provided, outside bbox is definite background
    if bbox is not None and len(bbox) == 4:
        bx, by, bw, bh = bbox
        x1 = max(0, int(round(bx)))
        y1 = max(0, int(round(by)))
        x2 = min(w, int(round(bx + bw)))
        y2 = min(h, int(round(by + bh)))
        if y1 > 0:
            gc_mask[:y1, :] = cv2.GC_BGD
        if y2 < h:
            gc_mask[y2:, :] = cv2.GC_BGD
        if x1 > 0:
            gc_mask[:, :x1] = cv2.GC_BGD
        if x2 < w:
            gc_mask[:, x2:] = cv2.GC_BGD

    # Probable and definite foreground seeds
    gc_mask[(filled == 255) & (gc_mask != cv2.GC_BGD)] = cv2.GC_PR_FGD
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded = cv2.erode(filled, k_erode)
    gc_mask[(eroded == 255) & (gc_mask != cv2.GC_BGD)] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    has_fg = np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))
    has_bg = np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD))

    if has_fg and has_bg:
        try:
            cv2.grabCut(color, gc_mask, None, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)
            final_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            final_mask = filled
    else:
        final_mask = filled

    # Clean up final mask: keep largest connected component & fill holes if foreground exists
    if np.any(final_mask > 0):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        if num_labels > 1:
            max_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            clean_mask = (labels == max_label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                final_clean = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(final_clean, contours, -1, 255, thickness=cv2.FILLED)
                final_mask = final_clean
            else:
                final_mask = clean_mask

    return final_mask


def evaluate_mask_methods(
    image_crops: List[np.ndarray],
    gt_masks: List[Union[np.ndarray, List[List[float]]]],
    out_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """Evaluate classical mask segmentation vs ground truth masks across crops.

    Computes mean and median Dice scores, and optionally exports results to CSV.
    """
    if len(image_crops) != len(gt_masks):
        raise ValueError(f"image_crops ({len(image_crops)}) and gt_masks ({len(gt_masks)}) must have equal length")

    dice_scores = []
    for crop, gt in zip(image_crops, gt_masks):
        pred = classical_mask(crop)
        if isinstance(gt, (list, tuple)):
            h, w = crop.shape[:2]
            gt_arr = polygon_to_mask(gt, h, w)
        else:
            gt_arr = gt
        dice = compute_dice_score(pred, gt_arr)
        dice_scores.append(dice)

    if dice_scores:
        mean_dice = float(np.mean(dice_scores))
        median_dice = float(np.median(dice_scores))
        min_dice = float(np.min(dice_scores))
        max_dice = float(np.max(dice_scores))
    else:
        mean_dice = 0.0
        median_dice = 0.0
        min_dice = 0.0
        max_dice = 0.0

    metrics = {
        "mean_dice": mean_dice,
        "median_dice": median_dice,
        "dice_mean": mean_dice,
        "dice_median": median_dice,
        "min_dice": min_dice,
        "max_dice": max_dice,
        "num_samples": float(len(dice_scores)),
    }

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("crop_idx,dice_score\n")
            for idx, score in enumerate(dice_scores):
                f.write(f"{idx},{score:.6f}\n")

    return metrics

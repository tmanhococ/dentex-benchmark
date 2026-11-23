# dentex_bench/augment/qc.py
"""Quality Control & Diagnostic Visualization for DENTEX Tooth-Swap Augmentation.

Enforces:
- Section 7.2 of SPEC: Automated boundary gradient seam energy ratio analysis.
- Section 7.3 of SPEC: 3-panel side-by-side diagnostic image visualization.
"""

from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_seam_score(img: np.ndarray, M: np.ndarray) -> float:
    r"""Compute seam gradient energy ratio according to Section 7.2 of SPEC.

    Formula:
        G = |Sobel_x| + |Sobel_y| trên ảnh kết quả
        boundary = dilate(M, 1) XOR erode(M, 1)
        ring_in = M \ erode(M, 3)
        ring_out = dilate(M, 3) \ M
        seam = mean(G[boundary]) / (0.5 * (mean(G[ring_in]) + mean(G[ring_out])) + 1e-6)

    Args:
        img: Input image as RGB (H, W, 3) or Grayscale (H, W).
        M: Binary ROI mask of the pasted/swapped region (H, W).

    Returns:
        float: Seam score (ratio). Values close to 1.0 or lower indicate seamless blending.
    """
    if M is None or not np.any(M > 0):
        return 0.0

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Convert to float32 Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    G = np.abs(gx) + np.abs(gy)

    # Standardize mask to uint8 binary (0 or 255)
    M_bin = (M > 0).astype(np.uint8) * 255
    if M_bin.ndim == 3:
        M_bin = M_bin[:, :, 0]

    # Structuring elements: radius 1 -> 3x3, radius 3 -> 7x7
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    dil1 = cv2.dilate(M_bin, k1, iterations=1)
    ero1 = cv2.erode(M_bin, k1, iterations=1)
    boundary = cv2.bitwise_xor(dil1, ero1)

    ero3 = cv2.erode(M_bin, k3, iterations=1)
    dil3 = cv2.dilate(M_bin, k3, iterations=1)
    ring_in = cv2.subtract(M_bin, ero3)
    ring_out = cv2.subtract(dil3, M_bin)

    boundary_mask = boundary > 0
    ring_in_mask = ring_in > 0
    ring_out_mask = ring_out > 0

    mean_boundary = float(np.mean(G[boundary_mask])) if np.any(boundary_mask) else 0.0
    mean_in = float(np.mean(G[ring_in_mask])) if np.any(ring_in_mask) else 0.0
    mean_out = float(np.mean(G[ring_out_mask])) if np.any(ring_out_mask) else 0.0

    denom = 0.5 * (mean_in + mean_out) + 1e-6
    return float(mean_boundary / denom)


def save_debug_composite(
    dst_orig: np.ndarray,
    syn_img: np.ndarray,
    M: np.ndarray,
    new_boxes: Sequence[Any],
    out_path: Union[str, Path],
) -> None:
    """Save 3-panel side-by-side diagnostic image: [orig | syn | syn + contours + boxes].

    Args:
        dst_orig: Original target image before augmentation.
        syn_img: Synthesized image after augmentation.
        M: Binary ROI mask of the pasted/swapped region.
        new_boxes: List of new/swapped bounding boxes [x, y, w, h] or annotation dicts with 'bbox'.
        out_path: Output file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert grayscale to RGB if necessary
    if dst_orig.ndim == 2:
        orig_rgb = cv2.cvtColor(dst_orig, cv2.COLOR_GRAY2RGB)
    elif dst_orig.ndim == 3 and dst_orig.shape[2] == 4:
        orig_rgb = cv2.cvtColor(dst_orig, cv2.COLOR_RGBA2RGB)
    else:
        orig_rgb = dst_orig.copy()

    if syn_img.ndim == 2:
        syn_rgb = cv2.cvtColor(syn_img, cv2.COLOR_GRAY2RGB)
    elif syn_img.ndim == 3 and syn_img.shape[2] == 4:
        syn_rgb = cv2.cvtColor(syn_img, cv2.COLOR_RGBA2RGB)
    else:
        syn_rgb = syn_img.copy()

    overlay = syn_rgb.copy()

    # Draw ROI contour on overlay
    if M is not None and np.any(M > 0):
        M_bin = (M > 0).astype(np.uint8) * 255
        if M_bin.ndim == 3:
            M_bin = M_bin[:, :, 0]
        contours, _ = cv2.findContours(M_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Draw new boxes on overlay (Red in RGB)
    for box in new_boxes:
        if isinstance(box, dict) and "bbox" in box:
            b = box["bbox"]
        else:
            b = box
        if len(b) >= 4:
            bx, by, bw, bh = b[:4]
            x1, y1 = int(round(bx)), int(round(by))
            x2, y2 = int(round(bx + bw)), int(round(by + bh))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    composite = np.hstack([orig_rgb, syn_rgb, overlay])
    plt.imsave(str(out_path), composite)


__all__ = ["compute_seam_score", "save_debug_composite"]

# dentex_bench/augment/blending.py
"""Target inpainting and boundary intensity matching for tooth swap augmentation."""

from typing import Any, Dict
import cv2
import numpy as np


def inpaint_target_tooth(img: np.ndarray, target_mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Inpaint target hole before pasting source tooth."""
    if target_mask.dtype != np.uint8:
        target_mask = target_mask.astype(np.uint8)
    if np.max(target_mask) == 1:
        target_mask = (target_mask * 255).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(target_mask, k)

    if img.ndim == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(bgr, dilated_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        inpainted_2d = cv2.inpaint(img[:, :, 0], dilated_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        return inpainted_2d[:, :, np.newaxis]
    else:
        return cv2.inpaint(img, dilated_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def compute_intensity_match_params(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    src_ring_mask: np.ndarray,
    dst_ring_mask: np.ndarray,
    mode: str = "meanstd",
) -> Dict[str, Any]:
    """Compute deterministic intensity match parameters (linear a,b or histogram LUT)."""
    if mode == "none":
        return {"mode": "none"}

    src_pixels = src_img[src_ring_mask > 0]
    dst_pixels = dst_img[dst_ring_mask > 0]

    if len(src_pixels) == 0 or len(dst_pixels) == 0:
        return {"mode": "none"}

    if mode == "meanstd":
        src_mean = float(np.mean(src_pixels))
        src_std = float(np.std(src_pixels)) + 1e-6
        dst_mean = float(np.mean(dst_pixels))
        dst_std = float(np.std(dst_pixels))

        a = float(dst_std / src_std)
        # Constrain gain to prevent severe blowout
        a = float(np.clip(a, 0.5, 2.0))
        b = float(dst_mean - a * src_mean)
        return {"mode": "meanstd", "a": a, "b": b}

    elif mode == "histogram":
        # Compute 256-bin lookup table from src to dst cumulative distributions
        src_hist, _ = np.histogram(src_pixels, bins=256, range=(0, 256), density=True)
        dst_hist, _ = np.histogram(dst_pixels, bins=256, range=(0, 256), density=True)
        src_cdf = src_hist.cumsum()
        dst_cdf = dst_hist.cumsum()
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            diff = np.abs(dst_cdf - src_cdf[i])
            lut[i] = np.argmin(diff)
        return {"mode": "histogram", "lut": lut.tolist()}

    return {"mode": "none"}


def apply_intensity_match(src_img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply precomputed intensity adjustment."""
    mode = params.get("mode", "none")
    if mode == "meanstd":
        a = params["a"]
        b = params["b"]
        adjusted = a * src_img.astype(np.float32) + b
        return np.clip(adjusted, 0, 255).astype(src_img.dtype)
    elif mode == "histogram" and "lut" in params:
        lut = np.array(params["lut"], dtype=np.uint8)
        if src_img.dtype != np.uint8:
            src_uint8 = np.clip(src_img, 0, 255).astype(np.uint8)
            return cv2.LUT(src_uint8, lut)
        return cv2.LUT(src_img, lut)
    return src_img

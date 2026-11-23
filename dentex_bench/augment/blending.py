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


def blend_images(
    src: np.ndarray,
    dst: np.ndarray,
    M: np.ndarray,
    mode: str = "core_preserving",
    feather_sigma: float = 5.0,
    core_erode_px: int = 6,
    ring_mode: str = "feather",
    **kwargs: Any,
) -> np.ndarray:
    """Multi-mode image compositor for dental X-ray tooth paste."""
    h, w = dst.shape[:2]

    if M.dtype != np.uint8:
        M = M.astype(np.uint8)
    if M.size > 0 and np.max(M) == 1:
        M = M * 255

    is_3d = (dst.ndim == 3)

    if mode == "hard":
        alpha = (M > 0)[:, :, None].astype(np.float32) if is_3d else (M > 0).astype(np.float32)
        out = alpha * src.astype(np.float32) + (1.0 - alpha) * dst.astype(np.float32)
        return np.clip(out, 0, 255).astype(dst.dtype)

    elif mode == "feather":
        k = int(feather_sigma * 4 + 1) | 1
        alpha_2d = cv2.GaussianBlur(M.astype(np.float32) / 255.0, (k, k), feather_sigma)
        alpha = alpha_2d[:, :, None] if is_3d else alpha_2d
        out = alpha * src.astype(np.float32) + (1.0 - alpha) * dst.astype(np.float32)
        return np.clip(out, 0, 255).astype(dst.dtype)

    elif mode in ("poisson_normal", "poisson_mixed"):
        rx, ry, rw, rh = cv2.boundingRect(M)
        if rw == 0 or rh == 0:
            return dst.copy()
        center = (rx + rw // 2, ry + rh // 2)

        # Pad 2px to avoid border touch error in seamlessClone
        p_src = cv2.copyMakeBorder(src, 2, 2, 2, 2, cv2.BORDER_REFLECT)
        p_dst = cv2.copyMakeBorder(dst, 2, 2, 2, 2, cv2.BORDER_REFLECT)
        p_mask = cv2.copyMakeBorder(M, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        p_center = (center[0] + 2, center[1] + 2)

        is_gray = (dst.ndim == 2)
        is_single_ch = (dst.ndim == 3 and dst.shape[2] == 1)

        if is_gray or is_single_ch:
            p_src_2d = p_src[:, :, 0] if p_src.ndim == 3 else p_src
            p_dst_2d = p_dst[:, :, 0] if p_dst.ndim == 3 else p_dst
            clone_src = cv2.cvtColor(p_src_2d, cv2.COLOR_GRAY2BGR)
            clone_dst = cv2.cvtColor(p_dst_2d, cv2.COLOR_GRAY2BGR)
        else:
            clone_src = p_src
            clone_dst = p_dst

        if clone_src.dtype != np.uint8:
            clone_src = np.clip(clone_src, 0, 255).astype(np.uint8)
        if clone_dst.dtype != np.uint8:
            clone_dst = np.clip(clone_dst, 0, 255).astype(np.uint8)

        clone_mode = cv2.NORMAL_CLONE if mode == "poisson_normal" else cv2.MIXED_CLONE
        cloned = cv2.seamlessClone(clone_src, clone_dst, p_mask, p_center, clone_mode)

        if is_gray:
            return cloned[2 : h + 2, 2 : w + 2, 0].astype(dst.dtype)
        elif is_single_ch:
            return cloned[2 : h + 2, 2 : w + 2, :1].astype(dst.dtype)
        return cloned[2 : h + 2, 2 : w + 2].astype(dst.dtype)

    elif mode == "core_preserving":
        # 1. Erode M to create untouched core
        k = max(1, core_erode_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        core_mask = cv2.erode(M, kernel)

        # 2. Blend background using ring_mode
        ring_blended = blend_images(
            src, dst, M, mode=ring_mode, feather_sigma=feather_sigma, core_erode_px=core_erode_px, **kwargs
        )

        # 3. Paste exact src into core
        alpha_core = (core_mask > 0)[:, :, None].astype(np.float32) if is_3d else (core_mask > 0).astype(np.float32)
        out = alpha_core * src.astype(np.float32) + (1.0 - alpha_core) * ring_blended.astype(np.float32)
        return np.clip(out, 0, 255).astype(dst.dtype)

    raise ValueError(f"Unknown blending mode: {mode}")


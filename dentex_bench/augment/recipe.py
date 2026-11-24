# dentex_bench/augment/recipe.py
"""Recipe schema, deterministic resolution, and pure execution engine.

Enforces:
- INV-6: Label annotation synchronization via update_annotations_for_swap.
- INV-8: execute() is a pure function that never calls any RNG. All stochastic
  choices are resolved deterministically inside resolve().
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from dentex_bench.augment.config import AugmentConfig
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.locator import ToothLocator
from dentex_bench.augment.placement import generate_swap_plan, SwapOp
from dentex_bench.augment.rng import rng_for
from dentex_bench.augment.geometry import compute_affine_matrix, warp_image, warp_mask
from dentex_bench.augment.blending import (
    inpaint_target_tooth,
    compute_intensity_match_params,
    apply_intensity_match,
    blend_images,
)
from dentex_bench.augment.labels import update_annotations_for_swap
from dentex_bench.augment.masks import compute_ring_mask, polygon_to_mask
from dentex_bench.augment.qc import compute_seam_score
from dentex_bench.augment.io_utils import read_image_lossless


@dataclass
class Recipe:
    """Fully resolved augmentation recipe for a target image."""

    recipe_id: str
    schema_version: int
    rng_key: List[Any]
    target_image_id: int
    target_file: str
    target_file_sha256: str
    ops: List[Dict[str, Any]]
    post: Dict[str, Any] = field(default_factory=dict)
    rejections: List[Dict[str, Any]] = field(default_factory=list)
    output: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Recipe":
        return cls(**d)


def resolve(
    target_image_id: int,
    key: Tuple[Any, ...],
    cfg: AugmentConfig,
    bank: ToothBank,
    locator: ToothLocator,
    target_file: str = "",
) -> Optional[Recipe]:
    """Resolve stochastic choices into fully-specified Recipe parameters.

    All RNG calls must occur strictly within this resolve function (INV-8).
    """
    rng = rng_for(cfg.global_seed, *key)
    plan = generate_swap_plan(target_image_id, cfg, bank, locator, rng)
    if not plan.ops:
        return None

    recipe_ops = []
    for op in plan.ops:
        src_entry = bank.get(op.source_entry_id)
        if not src_entry:
            continue

        # Resolve geometry jitter
        j_scale = float(rng.uniform(1.0 - cfg.geometry.jitter_scale, 1.0 + cfg.geometry.jitter_scale))
        j_rot = float(rng.uniform(-cfg.geometry.jitter_rotate_deg, cfg.geometry.jitter_rotate_deg))

        # Determine blender choice
        blend_mode = cfg.blend.mode
        if blend_mode == "random_choice":
            blend_mode = str(rng.choice(cfg.blend.choices))

        noise_seed = int(rng.integers(0, 2**31 - 1))

        op_dict = {
            "op": op.op,
            "source_entry_id": op.source_entry_id,
            "target_fdi": op.target_fdi,
            "target_tooth_ref": op.target_tooth_ref,
            "mirror": op.mirror,
            "scale_jitter": j_scale,
            "rotate_jitter": j_rot,
            "blend_mode": blend_mode,
            "feather_sigma": float(cfg.blend.feather_sigma),
            "core_erode_px": int(cfg.blend.core_erode_px),
            "noise_seed": noise_seed,
            "inpaint": {
                "enabled": cfg.blend.inpaint_target,
                "radius": cfg.blend.inpaint_radius,
            },
            "intensity": {
                "mode": cfg.blend.intensity_match,
            },
        }
        recipe_ops.append(op_dict)

    if not recipe_ops:
        return None

    variant_idx = key[-1] if key else 0
    return Recipe(
        recipe_id=f"rec_{target_image_id}_{variant_idx}",
        schema_version=1,
        rng_key=list(key),
        target_image_id=target_image_id,
        target_file=target_file,
        target_file_sha256="",
        ops=recipe_ops,
        rejections=plan.rejections,
    )


def _target_tooth_mask(target_ref: Dict[str, Any], h: int, w: int) -> np.ndarray:
    """Outline of the tooth being replaced: its DENTEX polygon, or the ellipse inscribed in its box."""
    seg = target_ref.get("segmentation")
    if seg:
        mask = polygon_to_mask(seg, h, w)
        if np.any(mask):
            return mask
    mask = np.zeros((h, w), dtype=np.uint8)
    bx, by, bw, bh = [float(v) for v in target_ref.get("bbox", [0, 0, 0, 0])]
    if bw > 0 and bh > 0:
        cv2.ellipse(mask, (int(round(bx + bw / 2)), int(round(by + bh / 2))),
                    (max(1, int(round(bw / 2))), max(1, int(round(bh / 2)))), 0, 0, 360, 255, -1)
    return mask


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)


def _match_layout(img: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Converts img to the channel layout of like (gray / RGB)."""
    if like.ndim == 2:
        return _gray(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if like.shape[2] == 1:
        return _gray(img)[:, :, None]
    return img[:, :, :3]


def execute(
    recipe: Recipe,
    image_loader: Callable[[int], Tuple[np.ndarray, List[Dict[str, Any]]]],
    bank: ToothBank,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Pure execution of a recipe without any RNG calls (INV-8).

    For each op the target tooth outline is inpainted, the source tooth patch (read from the bank,
    never substituted) is warped onto the target box, its brightness is matched to the bone/soft
    tissue ring around the target, and it is blended in with an untouched core. Labels follow INV-6.
    Raises FileNotFoundError / KeyError when a bank entry or file is missing.
    """
    dst_img, dst_anns = image_loader(recipe.target_image_id)
    out_img = dst_img.copy()
    out_anns = [dict(a) for a in dst_anns]
    h, w = out_img.shape[:2]
    seam_scores: List[float] = []

    for op_data in recipe.ops:
        src_entry = bank.get(op_data["source_entry_id"])
        if src_entry is None:
            raise KeyError(f"Bank entry '{op_data['source_entry_id']}' used by {recipe.recipe_id} is not in the bank")

        target_ref = op_data.get("target_tooth_ref", {})
        dst_bbox = [float(v) for v in target_ref.get("bbox", [0.0, 0.0, 0.0, 0.0])]
        if len(dst_bbox) < 4 or dst_bbox[2] <= 0 or dst_bbox[3] <= 0:
            continue

        patch = _match_layout(read_image_lossless(bank.resolve_path(src_entry.patch_path)), out_img)
        roi_src = read_image_lossless(bank.resolve_path(src_entry.roi_mask_path))
        roi_src = (_gray(roi_src) > 0).astype(np.uint8) * 255

        # Affine: source tooth box centre -> target tooth box centre, scaled to the target box
        src_roi = src_entry.roi_xywh
        src_bw = max(1.0, float(src_entry.bbox_xywh[2]))
        src_bh = max(1.0, float(src_entry.bbox_xywh[3]))
        src_cx = (src_entry.bbox_xywh[0] - src_roi[0]) + src_bw / 2.0
        src_cy = (src_entry.bbox_xywh[1] - src_roi[1]) + src_bh / 2.0
        dst_cx = dst_bbox[0] + dst_bbox[2] / 2.0
        dst_cy = dst_bbox[1] + dst_bbox[3] / 2.0
        scale_jitter = float(op_data.get("scale_jitter", 1.0))
        M = compute_affine_matrix(
            (src_cx, src_cy),
            (dst_cx, dst_cy),
            0.0,
            float(op_data.get("rotate_jitter", 0.0)),
            (dst_bbox[2] / src_bw) * scale_jitter,
            (dst_bbox[3] / src_bh) * scale_jitter,
            mirror=bool(op_data.get("mirror", False)),
        )

        warped_patch = warp_image(patch, M, (w, h))
        warped_roi = warp_mask(roi_src, M, (w, h))
        valid = warp_mask(np.full(patch.shape[:2], 255, dtype=np.uint8), M, (w, h))
        if not np.any(warped_roi):
            continue

        # Brightness: match the source ring (around the pasted region) to the target ring
        intensity_cfg = op_data.get("intensity") or {}
        mode = intensity_cfg.get("mode", "meanstd") if isinstance(intensity_cfg, dict) else "meanstd"
        if mode != "none" and isinstance(intensity_cfg, dict) and ("a" in intensity_cfg or "lut" in intensity_cfg):
            # Parameters pinned in the recipe (e.g. by a replay override) are applied as given
            warped_patch = apply_intensity_match(warped_patch, intensity_cfg)
        elif mode == "meanstd":
            ring = compute_ring_mask(warped_roi, ring_width_px=15)
            src_px = _gray(warped_patch)[cv2.bitwise_and(ring, valid) > 0].astype(np.float64)
            dst_px = _gray(out_img)[ring > 0].astype(np.float64)
            if src_px.size and dst_px.size:
                # Gain kept near 1 so the lesion contrast inside the tooth is not stretched
                if src_px.std() < 1.0 or dst_px.std() < 1.0:
                    a = 1.0  # flat ring: a gain ratio would be meaningless, shift only
                else:
                    a = float(np.clip(dst_px.std() / src_px.std(), 0.8, 1.25))
                b = float(dst_px.mean() - a * src_px.mean())
                warped_patch = apply_intensity_match(warped_patch, {"mode": "meanstd", "a": a, "b": b})
        elif mode == "histogram":
            ring = compute_ring_mask(warped_roi, ring_width_px=15)
            params = compute_intensity_match_params(
                _gray(warped_patch), _gray(out_img), cv2.bitwise_and(ring, valid), ring, mode="histogram"
            )
            warped_patch = apply_intensity_match(warped_patch, params)

        # Remove the target tooth (outline + a few px) before pasting, so no ghost tooth shows through
        inpaint_cfg = op_data.get("inpaint", {})
        if not isinstance(inpaint_cfg, dict) or inpaint_cfg.get("enabled", True):
            grow = max(3, int(round(0.04 * dst_bbox[2])))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
            target_mask = cv2.dilate(_target_tooth_mask(target_ref, h, w), k)
            radius = inpaint_cfg.get("radius", 5) if isinstance(inpaint_cfg, dict) else 5
            out_img = inpaint_target_tooth(out_img, target_mask, radius=radius)

        paste_mask = cv2.bitwise_and(warped_roi, valid)
        out_img = blend_images(
            warped_patch,
            out_img,
            paste_mask,
            mode=op_data.get("blend_mode", "core_preserving"),
            feather_sigma=float(op_data.get("feather_sigma", 5.0)),
            core_erode_px=int(op_data.get("core_erode_px", 6)),
        )
        seam_scores.append(round(float(compute_seam_score(out_img, paste_mask)), 4))

        # Synchronize labels (INV-6)
        op = SwapOp(
            op=op_data.get("op", "swap"),
            source_entry_id=op_data["source_entry_id"],
            target_image_id=recipe.target_image_id,
            target_fdi=op_data["target_fdi"],
            target_tooth_ref=target_ref,
            mirror=op_data.get("mirror", False),
        )
        out_anns = update_annotations_for_swap(out_anns, op, src_entry, M, w, h)

    return out_img, out_anns, {"status": "success", "seam_scores": seam_scores}

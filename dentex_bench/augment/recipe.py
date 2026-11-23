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
from dentex_bench.augment.masks import compute_ring_mask
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


def execute(
    recipe: Recipe,
    image_loader: Callable[[int], Tuple[np.ndarray, List[Dict[str, Any]]]],
    bank: ToothBank,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Pure execution of a recipe without any non-seeded RNG calls (INV-8).

    Guarantees deterministic bit-exact output for identical inputs and enforces
    label synchronization respecting INV-6.
    """
    dst_img, dst_anns = image_loader(recipe.target_image_id)
    out_img = dst_img.copy()
    out_anns = [dict(a) for a in dst_anns]
    h, w = out_img.shape[:2]

    for op_data in recipe.ops:
        src_entry = bank.get(op_data["source_entry_id"])
        if not src_entry:
            continue

        target_ref = op_data.get("target_tooth_ref", {})
        dst_bbox = target_ref.get("bbox", [0.0, 0.0, 0.0, 0.0])

        # Target inpainting
        inpaint_cfg = op_data.get("inpaint", {})
        do_inpaint = inpaint_cfg.get("enabled", True) if isinstance(inpaint_cfg, dict) else True
        if do_inpaint and len(dst_bbox) >= 4 and dst_bbox[2] > 0 and dst_bbox[3] > 0:
            target_mask = np.zeros((h, w), dtype=np.uint8)
            bx, by, bw, bh = dst_bbox
            y1, y2 = max(0, int(by)), min(h, int(by + bh))
            x1, x2 = max(0, int(bx)), min(w, int(bx + bw))
            if y2 > y1 and x2 > x1:
                target_mask[y1:y2, x1:x2] = 255
                inpaint_radius = inpaint_cfg.get("radius", 5) if isinstance(inpaint_cfg, dict) else 5
                out_img = inpaint_target_tooth(out_img, target_mask, radius=inpaint_radius)

        # Affine matrix alignment
        src_roi = src_entry.roi_xywh
        src_bw = max(1.0, float(src_entry.bbox_xywh[2]))
        src_bh = max(1.0, float(src_entry.bbox_xywh[3]))
        src_cx = (src_entry.bbox_xywh[0] - src_roi[0]) + src_entry.bbox_xywh[2] / 2.0
        src_cy = (src_entry.bbox_xywh[1] - src_roi[1]) + src_entry.bbox_xywh[3] / 2.0
        dst_cx = dst_bbox[0] + dst_bbox[2] / 2.0
        dst_cy = dst_bbox[1] + dst_bbox[3] / 2.0

        scale_jitter = op_data.get("scale_jitter", 1.0)
        scale_x = (dst_bbox[2] / src_bw) * scale_jitter
        scale_y = (dst_bbox[3] / src_bh) * scale_jitter

        rot_jitter = op_data.get("rotate_jitter", 0.0)
        M = compute_affine_matrix(
            (src_cx, src_cy),
            (dst_cx, dst_cy),
            src_entry.axis_angle_deg,
            rot_jitter,
            scale_x,
            scale_y,
            mirror=op_data.get("mirror", False),
        )

        # Warp and blend
        has_patch = bool(src_entry.patch_path and Path(src_entry.patch_path).is_file())
        has_mask = bool(src_entry.roi_mask_path and Path(src_entry.roi_mask_path).is_file())

        intensity_cfg = op_data.get("intensity")
        if has_patch:
            patch = read_image_lossless(src_entry.patch_path)
            if intensity_cfg and isinstance(intensity_cfg, dict) and intensity_cfg.get("mode", "none") != "none":
                patch = apply_intensity_match(patch, intensity_cfg)

            if has_mask:
                mask = read_image_lossless(src_entry.roi_mask_path)
            else:
                mask = np.ones(patch.shape[:2], dtype=np.uint8) * 255
            warped_patch = warp_image(patch, M, (w, h))
            warped_mask = warp_mask(mask, M, (w, h))
            if warped_patch.ndim == 2 and out_img.ndim == 3:
                warped_patch = cv2.cvtColor(warped_patch, cv2.COLOR_GRAY2RGB)
            elif warped_patch.ndim == 3 and out_img.ndim == 2:
                warped_patch = cv2.cvtColor(warped_patch, cv2.COLOR_RGB2GRAY)
            out_img = blend_images(
                warped_patch, out_img, warped_mask, mode=op_data.get("blend_mode", "core_preserving")
            )
        else:
            dummy_patch = np.full_like(out_img, 180)
            if intensity_cfg and isinstance(intensity_cfg, dict) and intensity_cfg.get("mode", "none") != "none":
                dummy_patch = apply_intensity_match(dummy_patch, intensity_cfg)
            dummy_mask = np.zeros((h, w), dtype=np.uint8)
            y1, y2 = max(0, int(dst_cy - 20)), min(h, int(dst_cy + 20))
            x1, x2 = max(0, int(dst_cx - 20)), min(w, int(dst_cx + 20))
            if y2 > y1 and x2 > x1:
                dummy_mask[y1:y2, x1:x2] = 255
            out_img = blend_images(
                dummy_patch, out_img, dummy_mask, mode=op_data.get("blend_mode", "core_preserving")
            )

        # Synchronize labels (INV-6)
        dummy_op = SwapOp(
            op=op_data.get("op", "swap"),
            source_entry_id=op_data["source_entry_id"],
            target_image_id=recipe.target_image_id,
            target_fdi=op_data["target_fdi"],
            target_tooth_ref=op_data["target_tooth_ref"],
            mirror=op_data.get("mirror", False),
        )
        out_anns = update_annotations_for_swap(out_anns, dummy_op, src_entry, M, w, h)

    return out_img, out_anns, {"status": "success"}

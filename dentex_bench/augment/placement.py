"""Placement strategy and anatomical rejection engine (Stage 2).

Defines SwapOp, SwapPlan, and generate_swap_plan, enforcing invariant rules:
- INV-4: FDI swap pair validity across jaws/quadrants (mirroring for contralateral quadrants)
- INV-5: Impacted teeth only placed into tooth position 8
- R_SAME_IMAGE: Disallow source and target from same image unless symmetric mode
- Rejection rule helper functions for scale, angle, and border checks.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from dentex_bench.augment.config import AugmentConfig
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.fdi import is_valid_swap_pair, tooth_number
from dentex_bench.augment.locator import ToothLocator, LocatedTooth


@dataclass
class SwapOp:
    """Represents a planned tooth swap or placement operation."""

    op: str  # "swap", "paste_negative", "erase_disease"
    source_entry_id: str
    target_image_id: int
    target_fdi: int
    target_tooth_ref: Dict[str, Any]
    mirror: bool
    affine: Optional[List[List[float]]] = None
    inpaint: Optional[Dict[str, Any]] = None
    intensity: Optional[Dict[str, Any]] = None
    instance_aug: Optional[Dict[str, Any]] = None
    blend: Optional[Dict[str, Any]] = None
    noise: Optional[Dict[str, Any]] = None
    qc: Optional[Dict[str, Any]] = None


@dataclass
class SwapPlan:
    """Collection of swap operations and rejection logs for a target image."""

    target_image_id: int
    ops: List[SwapOp] = field(default_factory=list)
    rejections: List[Dict[str, Any]] = field(default_factory=list)


def check_scale_rule(
    src_bbox: List[float],
    dst_bbox: List[float],
    scale_min: float,
    scale_max: float,
) -> bool:
    """Validate scale ratio between destination and source bounding boxes.

    Returns False if scale ratio is outside [scale_min, scale_max] on either axis (R_SCALE).
    """
    if len(src_bbox) < 4 or len(dst_bbox) < 4:
        return True
    src_w, src_h = src_bbox[2], src_bbox[3]
    dst_w, dst_h = dst_bbox[2], dst_bbox[3]
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return True
    sx = dst_w / src_w
    sy = dst_h / src_h
    if sx < scale_min or sx > scale_max or sy < scale_min or sy > scale_max:
        return False
    return True


def check_angle_rule(
    src_angle_deg: float,
    dst_angle_deg: float,
    max_rotate_deg: float,
) -> bool:
    """Validate axis orientation angle difference (R_ANGLE).

    Returns False if |src_angle - dst_angle| > max_rotate_deg.
    """
    diff = abs(src_angle_deg - dst_angle_deg)
    diff = min(diff, 360.0 - diff)
    return diff <= max_rotate_deg


def check_border_rule(
    bbox_xywh: List[float],
    img_w: int,
    img_h: int,
    max_out_ratio: float = 0.1,
) -> bool:
    """Check if bounding box / ROI extends outside image boundaries (R_BORDER).

    Returns False if outside area exceeds max_out_ratio of total box area.
    """
    if len(bbox_xywh) < 4 or img_w <= 0 or img_h <= 0:
        return True
    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0:
        return True
    total_area = w * h

    cx1 = max(0.0, float(x))
    cy1 = max(0.0, float(y))
    cx2 = min(float(img_w), float(x + w))
    cy2 = min(float(img_h), float(y + h))

    if cx2 <= cx1 or cy2 <= cy1:
        return False

    inside_area = (cx2 - cx1) * (cy2 - cy1)
    outside_area = max(0.0, total_area - inside_area)
    return (outside_area / total_area) <= max_out_ratio


def generate_swap_plan(
    target_image_id: int,
    cfg: AugmentConfig,
    bank: ToothBank,
    locator: ToothLocator,
    rng: np.random.Generator,
) -> SwapPlan:
    """Generate a valid SwapPlan for target_image_id enforcing anatomical reject rules.

    Constraints enforced:
    - INV-4: FDI swap pair validity across jaws and quadrants.
    - INV-5: Impacted teeth only placed into position 8.
    - R_SAME_IMAGE: Prohibit source from target image unless in symmetric mode.
    - R_SCALE: Reject candidate if scale ratio on either axis is outside [scale_min, scale_max].
    - Multi-op uniqueness: Each destination tooth FDI is swapped at most once.
    - Per-op retry budget: Attempts are tracked per op up to cfg.placement.max_tries.
    """
    plan = SwapPlan(target_image_id=target_image_id)
    target_teeth = _merge_teeth_by_fdi(locator.teeth(target_image_id))
    if not target_teeth:
        return plan

    n_ops = int(rng.integers(cfg.placement.min_ops, cfg.placement.max_ops + 1))
    n_ops = min(n_ops, len(target_teeth))

    if cfg.placement.mode == "symmetric":
        image_pool = [e for e in bank.entries if e.image_id == target_image_id]
    else:
        image_pool = [e for e in bank.entries if e.image_id != target_image_id]

    used_fdis = set()
    for _ in range(n_ops):
        available_teeth = [t for t in target_teeth if t.fdi not in used_fdis]
        if not available_teeth:
            break

        attempts_per_op = 0
        while attempts_per_op < cfg.placement.max_tries and available_teeth:
            attempts_per_op += 1
            dst_tooth = available_teeth[int(rng.integers(0, len(available_teeth)))]

            # Candidates are filtered rule by rule; the rule that empties the pool is logged
            if not image_pool:
                plan.rejections.append({"code": "R_SAME_IMAGE", "tries": attempts_per_op})
                break
            pool = image_pool
            if tooth_number(dst_tooth.fdi) != 8:
                pool = [e for e in pool if e.roi_class != "impacted"]
                if not pool:
                    plan.rejections.append({"code": "R_IMPACTED", "tries": attempts_per_op})
                    available_teeth.remove(dst_tooth)
                    continue

            matched = []
            for e in pool:
                if cfg.placement.mode == "fdi_swap":
                    valid, mirror = is_valid_swap_pair(e.fdi, dst_tooth.fdi, allow_mirror=cfg.placement.allow_mirror)
                elif cfg.placement.mode == "symmetric":
                    valid, mirror = is_valid_swap_pair(e.fdi, dst_tooth.fdi, allow_mirror=True)
                    valid = valid and mirror
                else:
                    valid, mirror = True, False
                if valid:
                    matched.append((e, mirror))
            if not matched:
                plan.rejections.append({"code": "R_FDI_MISMATCH", "tries": attempts_per_op})
                available_teeth.remove(dst_tooth)
                continue

            scaled = [
                (e, m) for e, m in matched
                if check_scale_rule(e.bbox_xywh, dst_tooth.bbox, cfg.geometry.scale_min, cfg.geometry.scale_max)
            ]
            if not scaled:
                plan.rejections.append({"code": "R_SCALE", "tries": attempts_per_op})
                available_teeth.remove(dst_tooth)
                continue

            # Class weighting among the anatomically valid candidates
            if cfg.placement.class_weights:
                weights = np.array([cfg.placement.class_weights.get(e.roi_class, 1.0) for e, _ in scaled], dtype=float)
            elif cfg.placement.class_weight_power is not None:
                counts = Counter(e.roi_class for e, _ in scaled)
                weights = np.array([(1.0 / counts[e.roi_class]) ** cfg.placement.class_weight_power for e, _ in scaled], dtype=float)
            else:
                weights = None
            if weights is not None and weights.sum() > 0:
                idx = int(rng.choice(len(scaled), p=weights / weights.sum()))
            else:
                idx = int(rng.integers(0, len(scaled)))
            src_entry, mirror = scaled[idx]

            op = SwapOp(
                op="swap",
                source_entry_id=src_entry.entry_id,
                target_image_id=target_image_id,
                target_fdi=dst_tooth.fdi,
                target_tooth_ref={
                    "source": dst_tooth.source,
                    "bbox": list(dst_tooth.bbox),
                    "has_mask": dst_tooth.segmentation is not None,
                    "segmentation": dst_tooth.segmentation,
                },
                mirror=mirror,
            )
            plan.ops.append(op)
            used_fdis.add(dst_tooth.fdi)
            break

    return plan


def _merge_teeth_by_fdi(teeth: List[LocatedTooth]) -> List[LocatedTooth]:
    """One target per tooth: DENTEX lists a tooth once per diagnosis, so merge outlines and boxes."""
    by_fdi: Dict[int, List[LocatedTooth]] = {}
    for t in teeth:
        by_fdi.setdefault(int(t.fdi), []).append(t)
    merged = []
    for fdi in sorted(by_fdi):
        group = by_fdi[fdi]
        boxes = [t.bbox for t in group if t.bbox and len(t.bbox) == 4]
        if not boxes:
            continue
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        segs = [p for t in group if t.segmentation for p in t.segmentation if len(p) >= 6]
        if segs:
            # Same reference as the bank entries: the box of the tooth outline
            pts = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in segs], axis=0)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
        merged.append(LocatedTooth(
            fdi=fdi,
            bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            segmentation=segs or None,
            source=group[0].source,
            score=max(t.score for t in group),
        ))
    return merged

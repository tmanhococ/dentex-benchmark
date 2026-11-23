# dentex_bench/augment/extras.py
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


__all__ = [
    "apply_photometric",
    "apply_patch_crop",
    "plan_hard_negative_ops",
    "plan_disease_erasing_ops",
]



def apply_photometric(
    img: np.ndarray,
    clahe_clip: float = 2.0,
    gamma: float = 1.0,
    tile_grid_size: Union[int, Tuple[int, int]] = (8, 8),
) -> np.ndarray:
    """Apply random multi-facility CLAHE and gamma correction.

    Args:
        img: Input image as uint8 numpy array, either 2D (H, W) or 3D (H, W, C).
        clahe_clip: Contrast limit for CLAHE. If > 0, CLAHE is applied.
        gamma: Gamma correction parameter (> 0). If != 1.0, power-law transform is applied.
        tile_grid_size: Grid size for CLAHE tiles, default (8, 8).

    Returns:
        Processed uint8 image preserving shape and dtype.
    """
    out = img.copy()
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    if clahe_clip > 0:
        if isinstance(tile_grid_size, int):
            grid = (tile_grid_size, tile_grid_size)
        else:
            grid = tuple(tile_grid_size)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=grid)
        if out.ndim == 3:
            for c in range(out.shape[2]):
                out[:, :, c] = clahe.apply(out[:, :, c])
        elif out.ndim == 2:
            out = clahe.apply(out)

    if gamma > 0 and abs(gamma - 1.0) > 1e-3:
        inv_gamma = 1.0 / float(gamma)
        table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)]).clip(0, 255).astype(np.uint8)
        out = cv2.LUT(out, table)

    return out


def _update_segmentation(
    seg: Any,
    x1: int,
    y1: int,
    crop_w: int,
    crop_h: int,
) -> Any:
    """Shift and clip segmentation polygons to the cropped window coordinates."""
    if not isinstance(seg, list) or not seg:
        return seg

    # Check whether seg is List[List[float]] or a flat List[float]
    if isinstance(seg[0], (int, float)):
        polygons = [seg]
        single = True
    elif isinstance(seg[0], list):
        polygons = seg
        single = False
    else:
        return seg

    updated_polys = []
    for poly in polygons:
        shifted = []
        for i in range(0, len(poly) - 1, 2):
            px = poly[i] - x1
            py = poly[i + 1] - y1
            px_clipped = float(np.clip(px, 0.0, crop_w))
            py_clipped = float(np.clip(py, 0.0, crop_h))
            shifted.extend([px_clipped, py_clipped])
        if len(shifted) >= 6:  # Must form at least a triangle
            updated_polys.append(shifted)

    if single:
        return updated_polys[0] if updated_polys else []
    return updated_polys


def apply_patch_crop(
    img: np.ndarray,
    annotations: List[Dict[str, Any]],
    crop_size: Union[int, Tuple[int, int]] = 1024,
    min_visible: float = 0.6,
    anchor_idx: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Native resolution patch cropping around target pathology bounding box anchor.

    Args:
        img: Full-resolution image ndarray (H, W) or (H, W, C).
        annotations: List of COCO-format annotation dicts.
        crop_size: Size of crop window, int or (crop_h, crop_w).
        min_visible: Fraction of original annotation area that must remain
            visible after cropping for the annotation to be retained.
        anchor_idx: Optional explicit index of annotation to anchor around.

    Returns:
        (crop_img, updated_annotations)
    """
    h, w = img.shape[:2]
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    if h < crop_h or w < crop_w:
        return img.copy(), [dict(a) for a in annotations]

    # Pick anchor annotation
    if anchor_idx is not None and 0 <= anchor_idx < len(annotations):
        target_ann = annotations[anchor_idx]
    elif annotations:
        # Prioritize synthetic annotations, then first annotation
        target_ann = next(
            (a for a in annotations if a.get("synthetic") or a.get("is_synthetic")),
            annotations[0],
        )
    else:
        target_ann = None

    if target_ann and "bbox" in target_ann and len(target_ann["bbox"]) == 4:
        bx, by, bw, bh = target_ann["bbox"]
        cx = float(bx + bw / 2.0)
        cy = float(by + bh / 2.0)
    else:
        cx = float(w / 2.0)
        cy = float(h / 2.0)

    x1 = int(np.clip(cx - crop_w / 2.0, 0, w - crop_w))
    y1 = int(np.clip(cy - crop_h / 2.0, 0, h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    crop = img[y1:y2, x1:x2].copy()

    updated_anns = []
    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        bx, by, bw, bh = bbox
        ix1 = max(float(x1), float(bx))
        iy1 = max(float(y1), float(by))
        ix2 = min(float(x2), float(bx + bw))
        iy2 = min(float(y2), float(by + bh))

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih

        orig_area = float(ann.get("area", bw * bh))
        if orig_area <= 0:
            orig_area = float(bw * bh)

        if orig_area > 0 and (inter_area / orig_area) >= min_visible:
            new_ann = dict(ann)
            new_ann["bbox"] = [float(ix1 - x1), float(iy1 - y1), float(iw), float(ih)]
            new_ann["area"] = float(inter_area)

            if "segmentation" in ann and ann["segmentation"] is not None:
                new_ann["segmentation"] = _update_segmentation(
                    ann["segmentation"], x1, y1, crop_w, crop_h
                )

            updated_anns.append(new_ann)

    return crop, updated_anns


def plan_hard_negative_ops(
    target_image_id: int,
    healthy_bank_entries: list,
    locator_teeth: list,
    rng: Optional[Any] = None,
    max_per_image: int = 2,
) -> List[Dict[str, Any]]:
    """Generate paste_negative ops matching healthy bank entries to located teeth by FDI.

    Preserves destination labels without adding new pathology labels (INV-6 compliant).
    """
    if hasattr(healthy_bank_entries, "entries"):
        bank_entries = healthy_bank_entries.entries
    else:
        bank_entries = list(healthy_bank_entries)

    if hasattr(locator_teeth, "teeth"):
        target_teeth = locator_teeth.teeth(target_image_id)
    else:
        target_teeth = list(locator_teeth)

    if rng is not None:
        if hasattr(rng, "permutation"):
            indices = rng.permutation(len(target_teeth))
            target_teeth = [target_teeth[i] for i in indices]
        elif hasattr(rng, "shuffle"):
            target_teeth = list(target_teeth)
            rng.shuffle(target_teeth)

    ops: List[Dict[str, Any]] = []
    used_fdis = set()

    for tooth in target_teeth:
        if len(ops) >= max_per_image:
            break

        tooth_fdi = getattr(tooth, "fdi", None) if not isinstance(tooth, dict) else tooth.get("fdi")
        if tooth_fdi is None or tooth_fdi in used_fdis:
            continue

        # Find matching healthy bank entries for this FDI
        matching = []
        for e in bank_entries:
            e_fdi = getattr(e, "fdi", None) if not isinstance(e, dict) else e.get("fdi")
            if e_fdi == tooth_fdi:
                matching.append(e)

        if not matching:
            continue

        # Prefer entries not from the same image
        diff_img = [
            e for e in matching
            if (getattr(e, "image_id", None) if not isinstance(e, dict) else e.get("image_id")) != target_image_id
        ]
        candidates = diff_img if diff_img else matching

        # Prefer entries with is_healthy == True if available
        healthy_only = [
            e for e in candidates
            if (getattr(e, "is_healthy", True) if not isinstance(e, dict) else e.get("is_healthy", True))
        ]
        candidates = healthy_only if healthy_only else candidates

        if not candidates:
            continue

        if rng is not None:
            if hasattr(rng, "integers"):
                chosen = candidates[int(rng.integers(0, len(candidates)))]
            elif hasattr(rng, "choice"):
                chosen = rng.choice(candidates)
            else:
                chosen = candidates[0]
        else:
            chosen = candidates[0]

        entry_id = getattr(chosen, "entry_id", None) if not isinstance(chosen, dict) else chosen.get("entry_id")
        tooth_bbox = getattr(tooth, "bbox", []) if not isinstance(tooth, dict) else tooth.get("bbox", [])
        tooth_seg = getattr(tooth, "segmentation", None) if not isinstance(tooth, dict) else tooth.get("segmentation")
        tooth_source = getattr(tooth, "source", "pseudo_label") if not isinstance(tooth, dict) else tooth.get("source", "pseudo_label")

        ops.append({
            "op": "paste_negative",
            "source_entry_id": entry_id,
            "target_image_id": target_image_id,
            "target_fdi": int(tooth_fdi),
            "target_tooth_ref": {
                "source": tooth_source,
                "bbox": tooth_bbox,
                "has_mask": tooth_seg is not None,
            },
            "mirror": False,
        })
        used_fdis.add(tooth_fdi)

    return ops


def plan_disease_erasing_ops(
    target_image_id: int,
    diseased_target_teeth: list,
    healthy_bank_entries: list,
    rng: Optional[Any] = None,
    max_per_image: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate erase_disease ops replacing diseased teeth with healthy counterparts of matching FDI."""
    if hasattr(healthy_bank_entries, "entries"):
        bank_entries = healthy_bank_entries.entries
    else:
        bank_entries = list(healthy_bank_entries)

    if hasattr(diseased_target_teeth, "teeth"):
        target_teeth = diseased_target_teeth.teeth(target_image_id)
    else:
        target_teeth = list(diseased_target_teeth)

    if rng is not None:
        if hasattr(rng, "permutation"):
            indices = rng.permutation(len(target_teeth))
            target_teeth = [target_teeth[i] for i in indices]
        elif hasattr(rng, "shuffle"):
            target_teeth = list(target_teeth)
            rng.shuffle(target_teeth)

    ops: List[Dict[str, Any]] = []
    used_fdis = set()

    for tooth in target_teeth:
        if max_per_image is not None and len(ops) >= max_per_image:
            break

        tooth_fdi = getattr(tooth, "fdi", None) if not isinstance(tooth, dict) else tooth.get("fdi")
        if tooth_fdi is None or tooth_fdi in used_fdis:
            continue

        matching = []
        for e in bank_entries:
            e_fdi = getattr(e, "fdi", None) if not isinstance(e, dict) else e.get("fdi")
            if e_fdi == tooth_fdi:
                matching.append(e)

        if not matching:
            continue

        diff_img = [
            e for e in matching
            if (getattr(e, "image_id", None) if not isinstance(e, dict) else e.get("image_id")) != target_image_id
        ]
        candidates = diff_img if diff_img else matching

        healthy_only = [
            e for e in candidates
            if (getattr(e, "is_healthy", True) if not isinstance(e, dict) else e.get("is_healthy", True))
        ]
        candidates = healthy_only if healthy_only else candidates

        if not candidates:
            continue

        if rng is not None:
            if hasattr(rng, "integers"):
                chosen = candidates[int(rng.integers(0, len(candidates)))]
            elif hasattr(rng, "choice"):
                chosen = rng.choice(candidates)
            else:
                chosen = candidates[0]
        else:
            chosen = candidates[0]

        entry_id = getattr(chosen, "entry_id", None) if not isinstance(chosen, dict) else chosen.get("entry_id")
        tooth_bbox = getattr(tooth, "bbox", []) if not isinstance(tooth, dict) else tooth.get("bbox", [])
        tooth_seg = getattr(tooth, "segmentation", None) if not isinstance(tooth, dict) else tooth.get("segmentation")
        tooth_source = getattr(tooth, "source", "annotations") if not isinstance(tooth, dict) else tooth.get("source", "annotations")

        ops.append({
            "op": "erase_disease",
            "source_entry_id": entry_id,
            "target_image_id": target_image_id,
            "target_fdi": int(tooth_fdi),
            "target_tooth_ref": {
                "source": tooth_source,
                "bbox": tooth_bbox,
                "has_mask": tooth_seg is not None,
            },
            "mirror": False,
        })
        used_fdis.add(tooth_fdi)

    return ops


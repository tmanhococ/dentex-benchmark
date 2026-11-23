# dentex_bench/augment/labels.py
from typing import Any, Dict, List
import numpy as np

from dentex_bench.augment.bank import ToothEntry
from dentex_bench.augment.geometry import warp_bbox, warp_polygons
from dentex_bench.augment.placement import SwapOp
from dentex_bench.augment.fdi import quadrant, tooth_number


def update_annotations_for_swap(
    target_anns: List[Dict[str, Any]],
    op: SwapOp,
    src_entry: ToothEntry,
    M: np.ndarray,
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """Update annotation list enforcing INV-6 (delete dst annotations, insert transformed src annotations)."""
    # 1. Remove all annotations on target_fdi
    filtered_anns = [a for a in target_anns if a.get("fdi") != op.target_fdi]

    if op.op == "erase_disease":
        return filtered_anns
    elif op.op == "paste_negative":
        return filtered_anns

    # 2. Add transformed source annotations
    q_dst = quadrant(op.target_fdi)
    t_dst = tooth_number(op.target_fdi)

    for ann in src_entry.annotations:
        orig_box = ann.get("bbox", [])
        if len(orig_box) == 4:
            warped_box = warp_bbox(orig_box, M)
            # Clip to image boundaries
            bx1 = max(0.0, warped_box[0])
            by1 = max(0.0, warped_box[1])
            bx2 = min(float(img_w), warped_box[0] + warped_box[2])
            by2 = min(float(img_h), warped_box[1] + warped_box[3])

            w_clipped = max(0.0, bx2 - bx1)
            h_clipped = max(0.0, by2 - by1)
            orig_area = orig_box[2] * orig_box[3]

            # Drop if visible area < 50%
            if orig_area > 0 and (w_clipped * h_clipped) / orig_area < 0.5:
                continue

            new_ann = dict(ann)
            new_ann["bbox"] = [bx1, by1, w_clipped, h_clipped]
            new_ann["area"] = float(w_clipped * h_clipped)
            new_ann["fdi"] = op.target_fdi
            new_ann["category_id_1"] = q_dst
            new_ann["category_id_2"] = t_dst
            new_ann["synthetic"] = True
            if op.target_image_id is not None:
                new_ann["image_id"] = op.target_image_id

            if "segmentation" in ann and ann["segmentation"]:
                new_ann["segmentation"] = warp_polygons(ann["segmentation"], M)

            filtered_anns.append(new_ann)

    return filtered_anns

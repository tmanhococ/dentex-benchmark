# scripts/build_bank.py
"""Build DENTEX Tooth and Lesion Bank from training split."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.augment.config import load_config, ROIConfig
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.fdi import is_upper, parse_fdi
from dentex_bench.augment.masks import (
    polygon_to_mask,
    create_class_roi,
    compute_ring_mask,
    classical_mask,
)
from dentex_bench.augment.io_utils import (
    read_image_lossless,
    write_image_lossless,
    STANDARD_CATEGORIES,
)


CAT_ID_TO_ROI_CLASS = {
    1: "caries",
    2: "deep_caries",
    3: "periapical",
    4: "impacted",
}

CAT_ID_TO_NAME = {c["id"]: c["name"] for c in STANDARD_CATEGORIES}


def infer_fdi_if_missing(ann: Dict[str, Any], img_w: int, img_h: int) -> int:
    """Infer FDI tooth identifier if not explicitly present in annotation."""
    if "fdi" in ann and ann["fdi"] is not None:
        return int(ann["fdi"])

    q = ann.get("category_id_1")
    t = ann.get("category_id_2")
    if q is not None and t is not None:
        if q in (1, 2, 3, 4) and t in range(1, 9):
            return q * 10 + t
        elif q in (0, 1, 2, 3) and t in range(0, 8):
            return (q + 1) * 10 + (t + 1)

    # Spatial fallback based on bounding box
    bx, by, bw, bh = ann.get("bbox", [0, 0, 10, 10])
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    cat_id = ann.get("category_id", 1)

    is_upper_jaw = (cy < img_h / 2.0)
    is_patient_right = (cx < img_w / 2.0)

    if cat_id == 4:
        # Impacted tooth MUST be position 8 (INV-5)
        if is_upper_jaw:
            return 18 if is_patient_right else 28
        else:
            return 48 if is_patient_right else 38

    # Assign quadrant
    if is_upper_jaw:
        quad = 1 if is_patient_right else 2
    else:
        quad = 4 if is_patient_right else 3

    dist_from_center = abs(cx - img_w / 2.0) / (img_w / 2.0)
    t_num = max(1, min(7, int(round(dist_from_center * 7)) + 1))
    return quad * 10 + t_num


def build_bank_from_config(cfg_path: Optional[str] = None, out_bank_dir: Optional[Path] = None) -> ToothBank:
    cfg = load_config(cfg_path or "configs/augment/base.yaml")
    train_ann_path = Path(cfg.paths.train_ann)
    val_ann_path = Path(cfg.paths.val_ann)
    img_dir = Path(cfg.paths.img_dir)
    bank_dir = out_bank_dir or (Path(cfg.paths.out_root) / "bank")

    if not train_ann_path.exists():
        raise FileNotFoundError(f"Training annotations not found at: {train_ann_path}")

    with open(train_ann_path, "r", encoding="utf-8") as f:
        train_coco = json.load(f)

    # Invariant 1: Assert no validation leakage
    train_ids = {img["id"] for img in train_coco.get("images", [])}
    if val_ann_path.exists():
        with open(val_ann_path, "r", encoding="utf-8") as f:
            val_coco = json.load(f)
        val_ids = {img["id"] for img in val_coco.get("images", [])}
        ToothBank.assert_no_val_leakage(train_ids, val_ids)

    patches_dir = bank_dir / "patches"
    masks_dir = bank_dir / "masks"
    patches_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    images_by_id = {img["id"]: img for img in train_coco.get("images", [])}
    annotations_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in train_coco.get("annotations", []):
        annotations_by_img.setdefault(ann["image_id"], []).append(ann)

    entries: List[ToothEntry] = []
    roi_cfg = cfg.bank.roi if hasattr(cfg, "bank") and hasattr(cfg.bank, "roi") else ROIConfig()

    for img_id, img_info in images_by_id.items():
        file_name = img_info["file_name"]
        candidate_paths = [
            img_dir / file_name,
            Path("data/raw") / file_name,
            Path("data") / file_name,
        ]
        img_path = next((p for p in candidate_paths if p.is_file()), None)
        if img_path is None:
            print(f"[Bank] Warning: Image file {file_name} not found, skipping.")
            continue

        img = read_image_lossless(img_path)
        img_h, img_w = img.shape[:2]
        anns = annotations_by_img.get(img_id, [])

        for idx, ann in enumerate(anns):
            fdi = infer_fdi_if_missing(ann, img_w, img_h)
            cat_id = ann.get("category_id", 1)
            roi_class = CAT_ID_TO_ROI_CLASS.get(cat_id, "caries")
            cat_name = CAT_ID_TO_NAME.get(cat_id, roi_class)

            bx, by, bw, bh = ann.get("bbox", [0, 0, 10, 10])
            bx, by, bw, bh = float(bx), float(by), float(bw), float(bh)

            # 20% margin for ROI crop
            pad_x = bw * 0.2
            pad_y = bh * 0.2
            rx1 = max(0, int(bx - pad_x))
            ry1 = max(0, int(by - pad_y))
            rx2 = min(img_w, int(bx + bw + pad_x))
            ry2 = min(img_h, int(by + bh + pad_y))
            rw = max(1, rx2 - rx1)
            rh = max(1, ry2 - ry1)

            patch = img[ry1:ry2, rx1:rx2].copy()
            roi_xywh = [rx1, ry1, rw, rh]

            # Generate tooth mask
            rel_box = [bx - rx1, by - ry1, bw, bh]
            seg = ann.get("segmentation")
            if seg and isinstance(seg, list) and len(seg) > 0 and len(seg[0]) >= 6:
                shifted_seg = []
                for poly in seg:
                    pts = np.array(poly).reshape(-1, 2)
                    pts[:, 0] -= rx1
                    pts[:, 1] -= ry1
                    shifted_seg.append(pts.flatten().tolist())
                tooth_mask = polygon_to_mask(shifted_seg, rh, rw)
                mask_source = "polygon"
            else:
                tooth_mask = classical_mask(patch, bbox=rel_box)
                mask_source = "classical"

            # Create class ROI mask
            p_box = rel_box if roi_class == "periapical" else None
            roi_mask = create_class_roi(tooth_mask, roi_class, p_box, roi_cfg)

            # Ring statistics
            ring_mask = compute_ring_mask(roi_mask, ring_width_px=15)
            if np.any(ring_mask > 0):
                gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) if patch.ndim == 3 else patch
                ring_pixels = gray_patch[ring_mask > 0]
                ring_stats = {"mean": float(np.mean(ring_pixels)), "std": float(np.std(ring_pixels))}
            else:
                ring_stats = {"mean": 128.0, "std": 20.0}

            entry_id = f"{img_id}_{fdi}_{idx}"
            rel_patch_path = f"patches/{entry_id}.png"
            rel_mask_path = f"masks/{entry_id}_tooth.png"
            rel_roi_mask_path = f"masks/{entry_id}_roi.png"

            write_image_lossless(bank_dir / rel_patch_path, patch)
            write_image_lossless(bank_dir / rel_mask_path, tooth_mask)
            write_image_lossless(bank_dir / rel_roi_mask_path, roi_mask)

            # Local annotations in patch frame
            local_ann = dict(ann)
            local_ann["bbox"] = rel_box

            entry = ToothEntry(
                entry_id=entry_id,
                image_id=img_id,
                file_name=file_name,
                fdi=fdi,
                is_upper=is_upper(fdi),
                labels=[cat_name.lower()],
                is_healthy=(cat_id not in (1, 2, 3, 4)),
                roi_class=roi_class,
                bbox_xywh=[bx, by, bw, bh],
                roi_xywh=roi_xywh,
                annotations=[local_ann],
                axis_angle_deg=0.0,
                mask_area_px=int(np.sum(tooth_mask > 0)),
                mask_h=float(bh),
                mask_w=float(bw),
                ring_stats=ring_stats,
                noise_sigma=1.0,
                patch_path=rel_patch_path,
                mask_path=rel_mask_path,
                roi_mask_path=rel_roi_mask_path,
                mask_source=mask_source,
                healthy_score=None,
            )
            entries.append(entry)

    bank = ToothBank(entries)
    bank.save(bank_dir)
    print(f"[Bank] Successfully built bank with {len(entries)} teeth saved to {bank_dir}")
    return bank


def main():
    parser = argparse.ArgumentParser(description="Build DENTEX Tooth and Lesion Bank")
    parser.add_argument("--config", type=str, default="configs/augment/base.yaml")
    parser.add_argument("--out", type=str, default=None, help="Output directory for tooth bank")
    args = parser.parse_args()

    out_bank_dir = Path(args.out) if args.out else None
    build_bank_from_config(args.config, out_bank_dir=out_bank_dir)


if __name__ == "__main__":
    main()

"""Builds the tooth bank from DENTEX training annotations.

One bank entry = one annotated tooth (all diagnoses DENTEX gives that tooth in that image), cut out
with its real polygon outline. Teeth without an exact FDI number or without a polygon are skipped
and counted, never guessed: a wrong FDI or a guessed outline would put a tooth in the wrong place.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.fdi import is_upper
from dentex_bench.augment.geometry import compute_tooth_pca
from dentex_bench.augment.io_utils import read_image_lossless, write_image_lossless
from dentex_bench.augment.masks import compute_ring_mask, polygon_to_mask

CAT_ID_TO_ROI_CLASS = {1: "caries", 2: "deep_caries", 3: "periapical", 4: "impacted"}

# Paste region grows beyond the tooth outline by this fraction of the tooth width, so the
# radiolucency of a periapical lesion (around the apex, outside the root outline) travels with it.
ROI_DILATE_FRAC = {"caries": 0.03, "deep_caries": 0.03, "impacted": 0.05, "periapical": 0.15}
# When a tooth carries several diagnoses its ROI follows the one that reaches furthest
ROI_PRIORITY = ["periapical", "impacted", "deep_caries", "caries"]
RING_WIDTH_PX = 15


def roi_dilate_px(roi_class: str, tooth_w: float) -> int:
    return max(3, int(round(ROI_DILATE_FRAC.get(roi_class, 0.03) * tooth_w)))


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def polygons_bbox(polys: List[List[float]]) -> List[float]:
    pts = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polys], axis=0)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def shift_polygons(polys: List[List[float]], dx: float, dy: float) -> List[List[float]]:
    out = []
    for p in polys:
        pts = np.asarray(p, dtype=np.float64).reshape(-1, 2).copy()
        pts[:, 0] -= dx
        pts[:, 1] -= dy
        out.append(pts.reshape(-1).tolist())
    return out


def _find_image(file_name: str, img_dirs: List[Path]) -> Optional[Path]:
    for d in img_dirs:
        for cand in (d / file_name, d / Path(file_name).name):
            if cand.is_file():
                return cand
    return None


def build_tooth_bank(
    train_ann: Union[str, Path],
    img_dir: Union[str, Path],
    out_dir: Union[str, Path],
    val_ann: Optional[Union[str, Path]] = None,
) -> Tuple[ToothBank, Dict[str, Any]]:
    """
    Cuts every annotated tooth of the training split into a bank under out_dir.

    Returns the bank (root = out_dir) and a report with how many teeth were kept and why others
    were skipped. Raises if an annotated image is missing on disk.
    """
    train_ann, img_dir, out_dir = Path(train_ann), Path(img_dir), Path(out_dir)
    with open(train_ann, "r", encoding="utf-8") as f:
        coco = json.load(f)

    train_ids = {img["id"] for img in coco.get("images", [])}
    if val_ann is not None and Path(val_ann).is_file():
        with open(val_ann, "r", encoding="utf-8") as f:
            val_ids = {img["id"] for img in json.load(f).get("images", [])}
        ToothBank.assert_no_val_leakage(train_ids, val_ids)

    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    anns_by_img: Dict[Any, List[Dict[str, Any]]] = {}
    for a in coco.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)

    skipped: Counter = Counter()
    entries: List[ToothEntry] = []
    img_dirs = [img_dir, Path("data/raw")]

    for img_info in coco.get("images", []):
        img_id = img_info["id"]
        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue
        img_path = _find_image(img_info["file_name"], img_dirs)
        if img_path is None:
            raise FileNotFoundError(f"Image '{img_info['file_name']}' of the training annotations is missing under {img_dir}")
        img = read_image_lossless(img_path)
        img_h, img_w = img.shape[:2]

        # Group the diagnoses of the same tooth
        teeth: Dict[int, List[Dict[str, Any]]] = {}
        for a in anns:
            if a.get("fdi") is None:
                skipped["no_fdi"] += 1
                continue
            if not a.get("segmentation"):
                skipped["no_polygon"] += 1
                continue
            teeth.setdefault(int(a["fdi"]), []).append(a)

        for fdi, tooth_anns in sorted(teeth.items()):
            polys = [p for a in tooth_anns for p in a["segmentation"] if len(p) >= 6]
            if not polys:
                skipped["no_polygon"] += 1
                continue
            tx, ty, tw, th = polygons_bbox(polys)
            if tw < 8 or th < 8:
                skipped["degenerate"] += 1
                continue

            classes = {CAT_ID_TO_ROI_CLASS.get(int(a["category_id"]), "caries") for a in tooth_anns}
            roi_class = next(c for c in ROI_PRIORITY if c in classes)
            dil = roi_dilate_px(roi_class, tw)
            margin = dil + RING_WIDTH_PX + 8

            rx1 = int(max(0, np.floor(tx - margin)))
            ry1 = int(max(0, np.floor(ty - margin)))
            rx2 = int(min(img_w, np.ceil(tx + tw + margin)))
            ry2 = int(min(img_h, np.ceil(ty + th + margin)))
            if rx2 - rx1 < 8 or ry2 - ry1 < 8:
                skipped["degenerate"] += 1
                continue
            patch = img[ry1:ry2, rx1:rx2].copy()
            rh, rw = patch.shape[:2]

            local_polys = shift_polygons(polys, rx1, ry1)
            tooth_mask = polygon_to_mask(local_polys, rh, rw)
            if not np.any(tooth_mask):
                skipped["empty_mask"] += 1
                continue
            roi_mask = dilate(tooth_mask, dil)

            ring = compute_ring_mask(roi_mask, ring_width_px=RING_WIDTH_PX)
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) if patch.ndim == 3 else patch
            ring_px = gray[ring > 0]
            ring_stats = {"mean": float(ring_px.mean()), "std": float(ring_px.std())} if ring_px.size else {}
            angle_deg, _, _ = compute_tooth_pca(tooth_mask)

            local_anns = []
            for a in tooth_anns:
                la = {k: v for k, v in a.items() if k not in ("id", "image_id")}
                bx, by, bw, bh = [float(v) for v in a["bbox"]]
                la["bbox"] = [bx - rx1, by - ry1, bw, bh]
                la["segmentation"] = shift_polygons(a["segmentation"], rx1, ry1)
                local_anns.append(la)

            entry_id = f"{img_id}_{fdi}"
            patch_rel = f"patches/{entry_id}.png"
            mask_rel = f"masks/{entry_id}_tooth.png"
            roi_rel = f"masks/{entry_id}_roi.png"
            write_image_lossless(out_dir / patch_rel, patch)
            write_image_lossless(out_dir / mask_rel, tooth_mask)
            write_image_lossless(out_dir / roi_rel, roi_mask)

            entries.append(ToothEntry(
                entry_id=entry_id,
                image_id=img_id,
                file_name=img_info["file_name"],
                fdi=fdi,
                is_upper=is_upper(fdi),
                labels=sorted(classes),
                is_healthy=False,
                roi_class=roi_class,
                bbox_xywh=[tx, ty, tw, th],
                roi_xywh=[rx1, ry1, rw, rh],
                annotations=local_anns,
                axis_angle_deg=float(angle_deg),
                mask_area_px=int(np.count_nonzero(tooth_mask)),
                mask_h=float(th),
                mask_w=float(tw),
                ring_stats=ring_stats,
                noise_sigma=1.0,
                patch_path=patch_rel,
                mask_path=mask_rel,
                roi_mask_path=roi_rel,
                mask_source="polygon",
            ))

    bank = ToothBank(entries, root=out_dir)
    bank.save(out_dir)
    report = {
        "teeth_in_bank": len(entries),
        "by_class": dict(Counter(e.roi_class for e in entries)),
        "skipped_annotations": dict(skipped),
    }
    with open(out_dir / "bank_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return bank, report

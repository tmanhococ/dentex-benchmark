import argparse
import json
import sys
from pathlib import Path
import numpy as np

def inspect_dataset(ann_path: Path, img_dir: Path | None = None) -> dict:
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    top_keys = list(data.keys()) if isinstance(data, dict) else ["<list_format>"]
    images = data.get("images", []) if isinstance(data, dict) else data
    annotations = data.get("annotations", []) if isinstance(data, dict) else []
    
    # Check polygons
    valid_polygon_count = 0
    total_anns = len(annotations)
    ious = []
    
    for ann in annotations:
        seg = ann.get("segmentation")
        is_poly_valid = False
        if isinstance(seg, list) and len(seg) > 0:
            if isinstance(seg[0], list) and len(seg[0]) >= 6:
                is_poly_valid = True
            elif len(seg) >= 6 and isinstance(seg[0], (int, float)):
                is_poly_valid = True
        if is_poly_valid:
            valid_polygon_count += 1
            
        bbox = ann.get("bbox", [])
        if len(bbox) == 4 and is_poly_valid:
            poly = seg[0] if isinstance(seg[0], list) else seg
            pts = np.array(poly).reshape(-1, 2)
            px_min, py_min = pts.min(axis=0)
            px_max, py_max = pts.max(axis=0)
            poly_w = max(0.0, px_max - px_min)
            poly_h = max(0.0, py_max - py_min)
            
            # Intersection of bbox and poly_rect
            ix1 = max(bbox[0], px_min)
            iy1 = max(bbox[1], py_min)
            ix2 = min(bbox[0] + bbox[2], px_max)
            iy2 = min(bbox[1] + bbox[3], py_max)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = (bbox[2] * bbox[3]) + (poly_w * poly_h) - inter
            ious.append(inter / union if union > 0 else 0.0)

    categories = {}
    for key in ["categories", "categories_1", "categories_2", "categories_3"]:
        if isinstance(data, dict) and key in data:
            categories[key] = data[key]
            
    report = {
        "top_level_keys": top_keys,
        "total_images": len(images),
        "total_annotations": total_anns,
        "valid_polygon_count": valid_polygon_count,
        "polygon_valid_ratio": (valid_polygon_count / total_anns) if total_anns > 0 else 0.0,
        "mean_bbox_polygon_iou": float(np.mean(ious)) if ious else None,
        "categories": categories,
        "use_classical_mask_fallback": (valid_polygon_count / total_anns < 0.9) if total_anns > 0 else True
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="Inspect DENTEX dataset JSON and verify segmentation")
    parser.add_argument("--ann", type=str, required=True, help="Path to DENTEX JSON file")
    parser.add_argument("--out", type=str, default="outputs/augment/_inspect/report.json", help="Path to output report")
    parser.add_argument("--img-dir", type=str, default=None, help="Path to image directory")
    args = parser.parse_args()
    
    ann_path = Path(args.ann)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = inspect_dataset(ann_path, Path(args.img_dir) if args.img_dir else None)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[T0] Inspection completed. Report written to {out_path}")
    print(f"Total images: {report['total_images']} | Valid polygon ratio: {report['polygon_valid_ratio']:.2%}")

if __name__ == "__main__":
    main()

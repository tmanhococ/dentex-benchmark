import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

TARGET_CLASSES: Dict[int, str] = {
    1: "Caries",
    2: "Deep Caries",
    3: "Periapical Lesion",
    4: "Impacted Tooth"
}

NAME_TO_ID = {name.lower().replace(" ", "").replace("_", ""): cid for cid, name in TARGET_CLASSES.items()}

def normalize_category_name(raw_name: Any) -> Optional[int]:
    """Normalize a raw category name or id to 1..4 (or None if not a target class)."""
    if raw_name is None:
        return None
    clean = str(raw_name).strip().lower().replace(" ", "").replace("_", "")
    if "deep" in clean and ("caries" in clean or "carious" in clean):
        return 2
    if "caries" in clean or "carious" in clean:
        return 1
    if "periapical" in clean or "lesion" in clean:
        return 3
    if "impacted" in clean:
        return 4
    if clean in ("1", "2", "3", "4"):
        return int(clean)
    return None


def _category_map(cats: Any) -> Dict[Any, str]:
    """Maps both int and str ids of a DENTEX/COCO category list (or dict) to category names."""
    items = list(cats.values()) if isinstance(cats, dict) else list(cats or [])
    out: Dict[Any, str] = {}
    for c in items:
        if isinstance(c, dict) and "id" in c:
            out[c["id"]] = str(c.get("name", ""))
            out[str(c["id"])] = str(c.get("name", ""))
    return out


def _position_lookup(cats: Any, valid: range) -> Dict[Any, int]:
    """
    DENTEX numbers quadrants (categories_1) and teeth (categories_2) by name ("1".."4", "1".."8")
    while the ids are 0-based; returns id -> clinical number. Falls back to id+1 when names are not numeric.
    """
    names = _category_map(cats)
    lookup: Dict[Any, int] = {}
    for key, name in names.items():
        try:
            num = int(str(name).strip())
        except ValueError:
            num = int(key) + 1 if str(key).isdigit() else None
        if num is not None and num in valid:
            lookup[key] = num
    return lookup


def _fdi_from_ann(ann: Dict[str, Any], quad_lookup: Dict[Any, int], tooth_lookup: Dict[Any, int]) -> Optional[int]:
    """Exact FDI tooth number (11..48) from DENTEX quadrant/enumeration ids, or None if unknown."""
    if ann.get("fdi") is not None:
        fdi = int(ann["fdi"])
        return fdi if fdi // 10 in (1, 2, 3, 4) and fdi % 10 in range(1, 9) else None
    q_id, t_id = ann.get("category_id_1"), ann.get("category_id_2")
    if q_id is None or t_id is None:
        return None
    if quad_lookup and tooth_lookup:
        q, t = quad_lookup.get(q_id), tooth_lookup.get(t_id)
    else:
        # No category tables: DENTEX ids are 0-based
        q, t = int(q_id) + 1, int(t_id) + 1
    if q in (1, 2, 3, 4) and t in range(1, 9):
        return q * 10 + t
    return None


def _clean_segmentation(seg: Any) -> Optional[List[List[float]]]:
    """Keeps polygon segmentations with at least 3 points; returns None otherwise."""
    if not isinstance(seg, list):
        return None
    polys = [list(map(float, p)) for p in seg if isinstance(p, (list, tuple)) and len(p) >= 6]
    return polys or None


def convert_dentex_annotations_to_coco(
    raw_ann_file: Union[str, Path],
    train_out: Union[str, Path],
    val_out: Union[str, Path],
    split_ratio: float = 0.8,
    seed: int = 42
) -> Dict[str, int]:
    """
    Parse DENTEX annotations (list of images with nested annotations or COCO-like format),
    filter to 4 abnormal tooth classes, and generate train/val COCO split.
    """
    raw_ann_file = Path(raw_ann_file)
    train_out = Path(train_out)
    val_out = Path(val_out)
    
    with open(raw_ann_file, "r") as f:
        data = json.load(f)
        
    categories = [{"id": cid, "name": cname} for cid, cname in TARGET_CLASSES.items()]
    
    image_entries: List[Dict[str, Any]] = []
    ann_entries: List[Dict[str, Any]] = []
    ann_id_counter = 1
    
    if isinstance(data, list):
        # Format 1: Array of images with nested annotations
        for img_item in data:
            img_id = img_item.get("image_id", img_item.get("id"))
            file_name = img_item.get("file_name", f"{img_id}.png")
            width = img_item.get("width", 1000)
            height = img_item.get("height", 500)
            
            for ann in img_item.get("annotations", []):
                cat_id = normalize_category_name(ann.get("category", ann.get("category_id", "")))
                if cat_id is not None:
                    bbox = ann["bbox"]  # [x, y, w, h]
                    area = float(ann.get("area", bbox[2] * bbox[3]))
                    entry = {
                        "id": ann_id_counter,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    }
                    fdi = _fdi_from_ann(ann, {}, {})
                    if fdi is not None:
                        entry["fdi"] = fdi
                    seg = _clean_segmentation(ann.get("segmentation"))
                    if seg is not None:
                        entry["segmentation"] = seg
                    ann_entries.append(entry)
                    ann_id_counter += 1
            
            image_entries.append({
                "id": img_id,
                "file_name": file_name,
                "width": width,
                "height": height
            })
    else:
        # Format 2: Standard COCO structure with raw labels or DENTEX Level 3
        raw_items = data.get("images", [])
        raw_anns = data.get("annotations", [])
        
        # DENTEX hierarchical files: categories_1 = quadrant, categories_2 = tooth, categories_3 = diagnosis
        raw_cat_map = _category_map(data.get("categories", []))
        diag_cat_map = _category_map(data.get("categories_3", [])) or raw_cat_map
        quad_lookup = _position_lookup(data.get("categories_1", []), range(1, 5))
        tooth_lookup = _position_lookup(data.get("categories_2", []), range(1, 9))

        for img in raw_items:
            iid = img.get("id", img.get("image_id"))
            image_entries.append({
                "id": iid,
                "file_name": img.get("file_name", f"{iid}.png"),
                "width": img.get("width", 1000),
                "height": img.get("height", 500)
            })
            
        for ann in raw_anns:
            if ann.get("category_id_3") is not None:
                cid_val, cat_map = ann["category_id_3"], diag_cat_map
            else:
                cid_val, cat_map = None, raw_cat_map
                for key in ["category_id", "category", "category_name"]:
                    if key in ann and ann[key] is not None:
                        cid_val = ann[key]
                        break

            if cid_val in cat_map:
                cat_id = normalize_category_name(cat_map[cid_val])
            else:
                cat_id = normalize_category_name(cid_val)

            if cat_id is not None:
                bbox = ann["bbox"]
                area = float(ann.get("area", bbox[2] * bbox[3]))
                entry = {
                    "id": ann_id_counter,
                    "image_id": ann.get("image_id", ann.get("id")),
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                }
                # Keep the tooth identity and outline DENTEX provides: augmentation needs both
                fdi = _fdi_from_ann(ann, quad_lookup, tooth_lookup)
                if fdi is not None:
                    entry["fdi"] = fdi
                seg = _clean_segmentation(ann.get("segmentation"))
                if seg is not None:
                    entry["segmentation"] = seg
                ann_entries.append(entry)
                ann_id_counter += 1

    # Shuffle and split by images
    random.seed(seed)
    image_ids = [img["id"] for img in image_entries]
    duplicates = sorted({i for i in image_ids if image_ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Raw annotations contain duplicate image ids {duplicates[:10]}; the train/val split would be ambiguous.")
    random.shuffle(image_ids)
    
    split_idx = int(len(image_ids) * split_ratio)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])
    
    train_images = [img for img in image_entries if img["id"] in train_ids]
    val_images = [img for img in image_entries if img["id"] in val_ids]
    
    train_anns = [ann for ann in ann_entries if ann["image_id"] in train_ids]
    val_anns = [ann for ann in ann_entries if ann["image_id"] in val_ids]
    
    coco_train = {"images": train_images, "annotations": train_anns, "categories": categories}
    coco_val = {"images": val_images, "annotations": val_anns, "categories": categories}
    
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(train_out, "w") as f:
        json.dump(coco_train, f, indent=2)
    with open(val_out, "w") as f:
        json.dump(coco_val, f, indent=2)
        
    return {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_annotations": len(train_anns),
        "val_annotations": len(val_anns)
    }

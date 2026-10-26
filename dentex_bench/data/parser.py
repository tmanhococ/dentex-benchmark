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
                    ann_entries.append({
                        "id": ann_id_counter,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    })
                    ann_id_counter += 1
            
            image_entries.append({
                "id": img_id,
                "file_name": file_name,
                "width": width,
                "height": height
            })
    else:
        # Format 2: Standard COCO structure with raw labels
        raw_items = data.get("images", [])
        raw_anns = data.get("annotations", [])
        raw_cat_map = {c["id"]: c.get("name", "") for c in data.get("categories", [])}
        
        for img in raw_items:
            iid = img.get("id", img.get("image_id"))
            image_entries.append({
                "id": iid,
                "file_name": img.get("file_name", f"{iid}.png"),
                "width": img.get("width", 1000),
                "height": img.get("height", 500)
            })
            
        for ann in raw_anns:
            raw_cname = raw_cat_map.get(ann.get("category_id"), "")
            cat_id = normalize_category_name(raw_cname)
            if cat_id is None:
                cat_id = normalize_category_name(ann.get("category", ann.get("category_name", ann.get("category_id"))))
            if cat_id is not None:
                bbox = ann["bbox"]
                area = float(ann.get("area", bbox[2] * bbox[3]))
                ann_entries.append({
                    "id": ann_id_counter,
                    "image_id": ann.get("image_id", ann.get("id")),
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                })
                ann_id_counter += 1

    # Shuffle and split by images
    random.seed(seed)
    image_ids = [img["id"] for img in image_entries]
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

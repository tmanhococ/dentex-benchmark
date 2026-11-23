# dentex_bench/augment/io_utils.py
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Union
import cv2
import numpy as np


STANDARD_CATEGORIES = [
    {"id": 1, "name": "Caries"},
    {"id": 2, "name": "Deep Caries"},
    {"id": 3, "name": "Periapical Lesion"},
    {"id": 4, "name": "Impacted Tooth"},
]


def read_image_lossless(path: Union[str, Path]) -> np.ndarray:
    """Read image preserving bit depth and channels."""
    p_str = str(path)
    img = cv2.imread(p_str, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    if img.ndim == 2:
        return img
    elif img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def write_image_lossless(path: Union[str, Path], img: np.ndarray) -> None:
    """Write image to disk as lossless PNG using atomic temporary write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(".tmp.png")

    if img.ndim == 2:
        to_save = img
    elif img.ndim == 3 and img.shape[2] == 3:
        to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        to_save = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:
        to_save = img

    success = cv2.imwrite(str(tmp_path), to_save, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    if not success:
        raise IOError(f"cv2.imwrite failed for {tmp_path}")
    tmp_path.replace(p)


def sha256_file(path: Union[str, Path]) -> str:
    """Compute SHA256 hex digest of file in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_coco_dataset(
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    out_path: Union[str, Path],
) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": STANDARD_CATEGORIES,
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)


def export_yolo_dataset(
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    out_dir: Union[str, Path],
) -> None:
    base = Path(out_dir)
    labels_dir = base / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    anns_by_img: Dict[Any, List[Dict[str, Any]]] = {}
    for a in annotations:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    for img in images:
        img_id = img["id"]
        w = float(img["width"])
        h = float(img["height"])
        stem = Path(img["file_name"]).stem
        label_file = labels_dir / f"{stem}.txt"

        lines = []
        for a in anns_by_img.get(img_id, []):
            cid = a["category_id"] - 1  # 0-indexed for YOLO
            bx, by, bw, bh = a["bbox"]
            cx = (bx + bw / 2.0) / w
            cy = (by + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_file, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
            else:
                f.write("")

    # Write data.yaml
    data_yaml = base / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("names:\n  0: Caries\n  1: Deep Caries\n  2: Periapical Lesion\n  3: Impacted Tooth\n")

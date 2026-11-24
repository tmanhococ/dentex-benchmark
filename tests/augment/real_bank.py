"""Test helpers: banks and target images backed by real files (execute() never fakes pixels)."""

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.io_utils import write_image_lossless


def make_real_bank(entries: Iterable[ToothEntry], root: Optional[Path] = None) -> ToothBank:
    """Writes a bright tooth-shaped patch, tooth mask and ROI mask for every entry under root."""
    root = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="bank_"))
    out = []
    for e in entries:
        rx, ry, rw, rh = [int(v) for v in e.roi_xywh]
        bx, by, bw, bh = e.bbox_xywh
        patch = np.full((rh, rw, 3), 60, dtype=np.uint8)
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cx, cy = int(round(bx - rx + bw / 2)), int(round(by - ry + bh / 2))
        cv2.ellipse(mask, (cx, cy), (max(1, int(bw / 2)), max(1, int(bh / 2))), 0, 0, 360, 255, -1)
        patch[mask > 0] = 200
        roi = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        rel = f"patches/{e.entry_id}.png"
        write_image_lossless(root / rel, patch)
        write_image_lossless(root / f"masks/{e.entry_id}_tooth.png", mask)
        write_image_lossless(root / f"masks/{e.entry_id}_roi.png", roi)
        out.append(dataclasses.replace(
            e, patch_path=rel, mask_path=f"masks/{e.entry_id}_tooth.png", roi_mask_path=f"masks/{e.entry_id}_roi.png"
        ))
    bank = ToothBank(out, root=root)
    bank.save(root)
    return bank


def write_images_for_ann(ann_file: Path, img_dir: Optional[Path] = None) -> Path:
    """Writes a gray radiograph-like PNG for every image listed in a COCO file."""
    img_dir = Path(img_dir) if img_dir is not None else Path(ann_file).parent / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    coco = json.loads(Path(ann_file).read_text(encoding="utf-8"))
    for i, img in enumerate(coco["images"]):
        w, h = int(img.get("width", 800)), int(img.get("height", 400))
        arr = np.full((h, w, 3), 90 + (i % 5) * 5, dtype=np.uint8)
        write_image_lossless(img_dir / img["file_name"], arr)
    return img_dir

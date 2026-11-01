# scripts/create_test_fixtures.py
import json
from pathlib import Path
import sys
from typing import Dict, Any
import numpy as np
import cv2
import torch

# Add project root so it runs standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dentex_bench.config import BenchmarkConfig
from dentex_bench.models import HierarchicalNetRunner, DiffusionNetRunner

CATEGORIES = [
    {"id": 1, "name": "Caries"},
    {"id": 2, "name": "Deep Caries"},
    {"id": 3, "name": "Periapical Lesion"},
    {"id": 4, "name": "Impacted Tooth"}
]

def create_sample_fixtures(data_dir: Path, outputs_dir: Path, num_samples: int = 4) -> Dict[str, int]:
    data_dir = Path(data_dir)
    outputs_dir = Path(outputs_dir)
    
    raw_dir = data_dir / "raw"
    proc_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    images = []
    annotations = []
    ann_id = 1
    
    for idx in range(1, num_samples + 1):
        file_name = f"sample_xray_{idx:03d}.png"
        img_path = raw_dir / file_name
        
        # Synthetic panoramic X-ray: 400x800 with realistic intensity gradient
        h, w = 400, 800
        grad = np.linspace(30, 220, w, dtype=np.uint8)
        img = np.tile(grad, (h, 1))
        # Add dental arch curve
        cv2.ellipse(img, (w // 2, h - 50), (w // 3, h // 2), 0, 180, 360, 240, 15)
        cv2.imwrite(str(img_path), img)
        
        images.append({
            "id": idx,
            "file_name": file_name,
            "width": w,
            "height": h
        })
        
        # Assign 2 distinct pathology boxes per image
        cat_id1 = ((idx - 1) % 4) + 1
        cat_id2 = (idx % 4) + 1
        
        boxes = [
            [150 + idx * 30, 180, 45, 40],
            [450 + idx * 20, 220, 70, 65]
        ]
        for b, cid in zip(boxes, [cat_id1, cat_id2]):
            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": cid,
                "bbox": b,
                "area": float(b[2] * b[3]),
                "iscrowd": 0
            })
            ann_id += 1
            
    split_idx = num_samples // 2
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    train_ids = {im["id"] for im in train_imgs}
    
    train_anns = [a for a in annotations if a["image_id"] in train_ids]
    val_anns = [a for a in annotations if a["image_id"] not in train_ids]
    
    with open(proc_dir / "instances_train.json", "w", encoding="utf-8") as f:
        json.dump({"images": train_imgs, "annotations": train_anns, "categories": CATEGORIES}, f, indent=2)
        
    with open(proc_dir / "instances_val.json", "w", encoding="utf-8") as f:
        json.dump({"images": val_imgs, "annotations": val_anns, "categories": CATEGORIES}, f, indent=2)
        
    # Generate mock checkpoints if not existing
    cfg = BenchmarkConfig.from_auto_env()
    for m_key, runner_cls in [("hierarchical_net", HierarchicalNetRunner), ("diffusion_net", DiffusionNetRunner)]:
        m_dir = outputs_dir / m_key
        m_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = m_dir / "best.pth"
        if not ckpt_path.exists():
            runner = runner_cls(cfg)
            runner.build_model(pretrained=False)
            runner.save_checkpoint(ckpt_path)
            
    return {
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
        "annotations": len(annotations)
    }

if __name__ == "__main__":
    create_sample_fixtures(Path("data"), Path("outputs"), num_samples=6)
    print("Sample fixtures generated successfully in data/ and outputs/.")

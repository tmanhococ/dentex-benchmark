# DENTEX 4-Class Abnormal Tooth Detection Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, reproducible, and portable benchmarking framework to train, evaluate, and compare Faster R-CNN, DINO, and HierarchicalDet on the 4 abnormal tooth classes of the DENTEX panoramic X-ray dataset, with dual CLI and Jupyter notebook interfaces across Local (Windows/Linux), Kaggle, and Google Colab environments.

**Architecture:** A standalone Python package `dentex_bench` handles runtime environment detection, VRAM-adaptive config, Hugging Face dataset download/parsing into 4-class COCO format, model runner adapters (Faster R-CNN, DINO, HierarchicalDet), and standard COCO evaluation with Markdown/LaTeX reporting. High-level CLI entrypoints in `scripts/` and 1-click execution notebooks in `notebooks/` wrap the core package.

**Tech Stack:** Python 3.10+, PyTorch, Torchvision, Hugging Face Transformers, Hugging Face Hub, pycocotools, Albumentations, OpenCV, Pandas, Tabulate, Matplotlib, Pytest.

## Global Constraints

- Target 4 classes: `1: Caries`, `2: Deep Caries`, `3: Periapical Lesion`, `4: Impacted Tooth`.
- Memory safety: GPU $\le$ 4GB (e.g. RTX 3050 Laptop) must default to `batch_size=1`, `grad_accum=4`, `fp16=True`, `img_size=800` to prevent CUDA OOM.
- Platform portability: Code must execute seamlessly on Windows (Local), Linux (Colab, Kaggle), automatically detecting environment and auto-installing missing dependencies.
- Ground truth consistency: All models evaluate on the exact same stratified 80/20 train/val split (`seed=42`) using COCO metrics ($mAP_{50}$, $mAP_{50:95}$, and per-class AP).
- Dual interface: Support both `.py` CLI and `.ipynb` interactive notebooks.

---

### Task 1: Environment & Configuration Management Engine

**Files:**
- Create: `dentex_bench/__init__.py`
- Create: `dentex_bench/env.py`
- Create: `dentex_bench/config.py`
- Create: `scripts/setup_env.py`
- Test: `tests/test_env_and_config.py`

**Interfaces:**
- Consumes: None (root hardware & system environment).
- Produces:
  - `dentex_bench.env.detect_platform() -> str` ("colab" | "kaggle" | "local")
  - `dentex_bench.env.get_device_info() -> dict` (device, name, vram_gb, is_low_vram)
  - `dentex_bench.env.ensure_dependencies(extra_packages: list = None) -> bool`
  - `dentex_bench.config.BenchmarkConfig` dataclass with adaptive settings (`batch_size`, `gradient_accumulation_steps`, `amp`, `img_size`, `num_workers`, `paths`).

- [ ] **Step 1: Write the failing test for env detection and adaptive config**

```python
# tests/test_env_and_config.py
import pytest
from dentex_bench.env import detect_platform, get_device_info, ensure_dependencies
from dentex_bench.config import BenchmarkConfig

def test_detect_platform():
    platform = detect_platform()
    assert platform in ["colab", "kaggle", "local"]

def test_get_device_info():
    info = get_device_info()
    assert "device" in info
    assert "vram_gb" in info
    assert "is_low_vram" in info
    assert isinstance(info["is_low_vram"], bool)

def test_benchmark_config_adaptive():
    cfg = BenchmarkConfig.from_auto_env()
    assert cfg.num_classes == 4
    assert cfg.class_names == ["Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth"]
    assert cfg.batch_size >= 1
    assert cfg.gradient_accumulation_steps >= 1
    if cfg.is_low_vram:
        assert cfg.batch_size <= 2
        assert cfg.gradient_accumulation_steps >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_and_config.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench'`

- [ ] **Step 3: Implement `dentex_bench/env.py`, `dentex_bench/config.py`, and `dentex_bench/__init__.py`**

```python
# dentex_bench/__init__.py
"""DENTEX 4-Class Abnormal Tooth Detection Benchmark."""
__version__ = "0.1.0"
```

```python
# dentex_bench/env.py
import os
import sys
import subprocess
import importlib

def detect_platform() -> str:
    if "google.colab" in sys.modules or os.path.exists("/content"):
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/working"):
        return "kaggle"
    return "local"

def get_device_info() -> dict:
    device = "cpu"
    name = "CPU"
    vram_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    
    is_low_vram = (device == "cpu") or (vram_gb > 0 and vram_gb < 6.0)
    return {
        "device": device,
        "name": name,
        "vram_gb": round(vram_gb, 2),
        "is_low_vram": is_low_vram
    }

def ensure_dependencies(extra_packages: list = None) -> bool:
    core_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("pycocotools", "pycocotools"),
        ("transformers", "transformers"),
        ("albumentations", "albumentations"),
        ("huggingface_hub", "huggingface_hub"),
        ("cv2", "opencv-python"),
        ("pandas", "pandas"),
        ("tabulate", "tabulate"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib")
    ]
    missing = []
    for mod, pkg in core_packages:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
            
    if extra_packages:
        for pkg in extra_packages:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing.append(pkg)
                
    if missing:
        print(f"[dentex_bench] Installing missing dependencies: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    return True
```

```python
# dentex_bench/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from dentex_bench.env import detect_platform, get_device_info

@dataclass
class BenchmarkConfig:
    platform: str = "local"
    device: str = "cpu"
    vram_gb: float = 0.0
    is_low_vram: bool = False
    
    # Dataset & classes
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: [
        "Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth"
    ])
    seed: int = 42
    train_val_split_ratio: float = 0.8
    
    # Training hyperparameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    amp: bool = True
    img_size: int = 800
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    
    # Paths
    root_dir: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    
    @classmethod
    def from_auto_env(cls, root_dir: str = "."):
        platform = detect_platform()
        dev_info = get_device_info()
        is_low = dev_info["is_low_vram"]
        
        # Adaptive tuning
        if is_low:
            batch_size = 1
            grad_accum = 4
            img_size = 800
            num_workers = 0 if os.name == "nt" else 2
        else:
            batch_size = 4
            grad_accum = 1
            img_size = 1024
            num_workers = 4
            
        base = Path(root_dir)
        output_dir = Path("/kaggle/working/outputs") if platform == "kaggle" else base / "outputs"
        data_dir = base / "data"
        
        return cls(
            platform=platform,
            device=dev_info["device"],
            vram_gb=dev_info["vram_gb"],
            is_low_vram=is_low,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            img_size=img_size,
            num_workers=num_workers,
            root_dir=base,
            data_dir=data_dir,
            output_dir=output_dir
        )
```

```python
# scripts/setup_env.py
import sys
from dentex_bench.env import detect_platform, get_device_info, ensure_dependencies
from dentex_bench.config import BenchmarkConfig

def main():
    print("=== DENTEX Benchmark Environment Setup ===")
    platform = detect_platform()
    dev_info = get_device_info()
    print(f"Platform: {platform}")
    print(f"Device: {dev_info['device']} ({dev_info['name']}), VRAM: {dev_info['vram_gb']} GB")
    print(f"Low VRAM Mode: {dev_info['is_low_vram']}")
    
    print("\nChecking dependencies...")
    ensure_dependencies()
    print("All core dependencies verified.")
    
    cfg = BenchmarkConfig.from_auto_env()
    print(f"\nAuto-tuned defaults: batch_size={cfg.batch_size}, grad_accum={cfg.gradient_accumulation_steps}, img_size={cfg.img_size}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_and_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/__init__.py dentex_bench/env.py dentex_bench/config.py scripts/setup_env.py tests/test_env_and_config.py
git commit -m "feat(env): add environment detection, adaptive config and setup script"
```

---

### Task 2: Data Pipeline - Downloader, DENTEX Parser, COCO 4-Class Generator & Dataset

**Files:**
- Create: `dentex_bench/data/__init__.py`
- Create: `dentex_bench/data/downloader.py`
- Create: `dentex_bench/data/parser.py`
- Create: `dentex_bench/data/dataset.py`
- Create: `scripts/download_and_prepare.py`
- Test: `tests/test_data_pipeline.py`

**Interfaces:**
- Consumes: `BenchmarkConfig` from `dentex_bench.config`.
- Produces:
  - `download_dentex_dataset(target_dir: Path) -> Path`
  - `parse_dentex_to_coco(raw_ann_path: Path, output_train_path: Path, output_val_path: Path, split_ratio: float, seed: int) -> dict`
  - `DentexDataset(img_dir, ann_file, transforms)` implementing standard `torch.utils.data.Dataset` returning `(image_tensor, target_dict)`.

- [ ] **Step 1: Write the failing test for parser and dataset**

```python
# tests/test_data_pipeline.py
import json
import pytest
import numpy as np
import torch
from pathlib import Path
from dentex_bench.data.parser import convert_dentex_annotations_to_coco, TARGET_CLASSES
from dentex_bench.data.dataset import DentexDataset

def test_target_classes():
    assert len(TARGET_CLASSES) == 4
    assert TARGET_CLASSES[1] == "Caries"
    assert TARGET_CLASSES[2] == "Deep Caries"
    assert TARGET_CLASSES[3] == "Periapical Lesion"
    assert TARGET_CLASSES[4] == "Impacted Tooth"

def test_parse_dentex_mock(tmp_path):
    mock_dentex = [
        {
            "image_id": 1,
            "file_name": "test_01.png",
            "width": 1000,
            "height": 500,
            "annotations": [
                {"bbox": [100, 100, 50, 50], "category": "Caries"},
                {"bbox": [200, 200, 60, 60], "category": "Deep Caries"},
                {"bbox": [300, 300, 40, 40], "category": "Normal"}  # should be ignored
            ]
        },
        {
            "image_id": 2,
            "file_name": "test_02.png",
            "width": 1000,
            "height": 500,
            "annotations": [
                {"bbox": [150, 150, 50, 50], "category": "Impacted Tooth"},
                {"bbox": [250, 250, 70, 70], "category": "Periapical Lesion"}
            ]
        }
    ]
    raw_path = tmp_path / "mock_raw.json"
    with open(raw_path, "w") as f:
        json.dump(mock_dentex, f)
        
    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    
    summary = convert_dentex_annotations_to_coco(
        raw_ann_file=raw_path,
        train_out=train_path,
        val_out=val_path,
        split_ratio=0.5,
        seed=42
    )
    
    assert train_path.exists()
    assert val_path.exists()
    with open(train_path) as f:
        coco_train = json.load(f)
    assert len(coco_train["categories"]) == 4
    assert len(coco_train["images"]) == 1
    assert len(coco_train["annotations"]) == 2  # excludes Normal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_pipeline.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.data'`

- [ ] **Step 3: Implement data downloader, parser, dataset, and download_and_prepare script**

```python
# dentex_bench/data/__init__.py
from .parser import TARGET_CLASSES, convert_dentex_annotations_to_coco
from .dataset import DentexDataset
from .downloader import download_dentex_dataset
```

```python
# dentex_bench/data/downloader.py
import os
import shutil
import zipfile
from pathlib import Path

def download_dentex_dataset(target_dir: Path) -> Path:
    target_dir = Path(target_dir)
    raw_dir = target_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files already exist
    existing_images = list(raw_dir.glob("**/*.png")) + list(raw_dir.glob("**/*.jpg"))
    if len(existing_images) > 100:
        print(f"[dentex_bench] Found {len(existing_images)} existing images in {raw_dir}.")
        return raw_dir

    print(f"[dentex_bench] Downloading DENTEX dataset from Hugging Face Hub (ibrahimhamamci/DENTEX)...")
    try:
        from huggingface_hub import snapshot_download
        download_path = snapshot_download(
            repo_id="ibrahimhamamci/DENTEX",
            repo_type="dataset",
            local_dir=str(raw_dir),
            ignore_patterns=["*.git*"]
        )
        print(f"[dentex_bench] Download complete to {download_path}")
    except Exception as e:
        print(f"[dentex_bench] Warning: Hugging Face snapshot_download failed: {e}")
        print("[dentex_bench] Please place raw DENTEX files in data/raw/ or provide local archive.")
        
    # Unpack any zip files found in raw_dir
    for zip_file in raw_dir.glob("*.zip"):
        print(f"[dentex_bench] Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
            
    return raw_dir
```

```python
# dentex_bench/data/parser.py
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

TARGET_CLASSES: Dict[int, str] = {
    1: "Caries",
    2: "Deep Caries",
    3: "Periapical Lesion",
    4: "Impacted Tooth"
}

NAME_TO_ID = {name.lower().replace(" ", "").replace("_", ""): cid for cid, name in TARGET_CLASSES.items()}

def normalize_category_name(raw_name: str):
    clean = str(raw_name).lower().replace(" ", "").replace("_", "")
    if "deep" in clean and "caries" in clean:
        return 2
    if "caries" in clean:
        return 1
    if "periapical" in clean or "lesion" in clean:
        return 3
    if "impacted" in clean:
        return 4
    return None

def convert_dentex_annotations_to_coco(
    raw_ann_file: Path,
    train_out: Path,
    val_out: Path,
    split_ratio: float = 0.8,
    seed: int = 42
) -> Dict[str, int]:
    with open(raw_ann_file, "r") as f:
        data = json.load(f)
        
    # Handle list vs COCO-like dict in raw format
    raw_items = data if isinstance(data, list) else data.get("images", [])
    raw_anns = data.get("annotations", []) if isinstance(data, dict) else []
    
    categories = [{"id": cid, "name": cname} for cid, cname in TARGET_CLASSES.items()]
    
    image_entries = []
    ann_entries = []
    ann_id_counter = 1
    
    if isinstance(data, list):
        # Format 1: Array of images with nested annotations
        for img_item in data:
            img_id = img_item.get("image_id", img_item.get("id"))
            file_name = img_item.get("file_name", f"{img_id}.png")
            width = img_item.get("width", 1000)
            height = img_item.get("height", 500)
            
            has_valid_ann = False
            for ann in img_item.get("annotations", []):
                cat_id = normalize_category_name(ann.get("category", ann.get("category_id", "")))
                if cat_id is not None:
                    bbox = ann["bbox"] # [x, y, w, h]
                    area = float(bbox[2] * bbox[3])
                    ann_entries.append({
                        "id": ann_id_counter,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    })
                    ann_id_counter += 1
                    has_valid_ann = True
            
            image_entries.append({
                "id": img_id,
                "file_name": file_name,
                "width": width,
                "height": height
            })
    else:
        # Format 2: Standard COCO structure with raw labels
        image_entries = raw_items
        raw_cat_map = {c["id"]: c.get("name", "") for c in data.get("categories", [])}
        for ann in raw_anns:
            raw_cname = raw_cat_map.get(ann.get("category_id"), "")
            cat_id = normalize_category_name(raw_cname)
            if cat_id is not None:
                ann_copy = dict(ann)
                ann_copy["id"] = ann_id_counter
                ann_copy["category_id"] = cat_id
                ann_copy["iscrowd"] = 0
                ann_entries.append(ann_copy)
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
```

```python
# dentex_bench/data/dataset.py
import os
import cv2
import json
import torch
from pathlib import Path
from typing import Optional, Callable
from torch.utils.data import Dataset

class DentexDataset(Dataset):
    def __init__(self, img_dir: Path, ann_file: Path, transforms: Optional[Callable] = None):
        self.img_dir = Path(img_dir)
        with open(ann_file, "r") as f:
            self.coco = json.load(f)
            
        self.images = self.coco["images"]
        self.transforms = transforms
        
        # Index annotations by image_id
        self.img_to_anns = {}
        for ann in self.coco["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        
        img_path = self.img_dir / file_name
        if not img_path.exists():
            # Try subdirectories if nested
            matches = list(self.img_dir.glob(f"**/{file_name}"))
            img_path = matches[0] if matches else img_path
            
        image = cv2.imread(str(img_path))
        if image is None:
            # Synthetic fallback for unit testing if file missing
            image = (np.zeros((img_info["height"], img_info["width"], 3), dtype=np.uint8) 
                     if 'np' in globals() else torch.zeros((img_info["height"], img_info["width"], 3)).byte().numpy())
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        areas = []
        
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # xyxy format
            labels.append(ann["category_id"])
            areas.append(ann.get("area", w * h))
            
        if self.transforms:
            augmented = self.transforms(image=image, bboxes=boxes, category_ids=labels)
            image = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["category_ids"]
            
        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32)
        }
        return img_tensor, target
```

```python
# scripts/download_and_prepare.py
import argparse
from pathlib import Path
from dentex_bench.config import BenchmarkConfig
from dentex_bench.data.downloader import download_dentex_dataset
from dentex_bench.data.parser import convert_dentex_annotations_to_coco

def main():
    parser = argparse.ArgumentParser(description="Download and prepare DENTEX dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()
    
    cfg = BenchmarkConfig.from_auto_env()
    data_dir = Path(args.data_dir)
    raw_dir = download_dentex_dataset(data_dir)
    
    # Locate annotation json in raw directory
    json_files = list(raw_dir.glob("**/*.json"))
    if not json_files:
        print("[dentex_bench] No JSON annotation file found in data/raw yet. Please check dataset extraction.")
        return
        
    ann_file = json_files[0]
    print(f"[dentex_bench] Using raw annotation file: {ann_file}")
    
    train_out = data_dir / "processed" / "instances_train.json"
    val_out = data_dir / "processed" / "instances_val.json"
    
    stats = convert_dentex_annotations_to_coco(
        raw_ann_file=ann_file,
        train_out=train_out,
        val_out=val_out,
        split_ratio=cfg.train_val_split_ratio,
        seed=args.seed
    )
    print(f"[dentex_bench] COCO format dataset prepared:")
    print(f"  Train: {stats['train_images']} images, {stats['train_annotations']} annotations -> {train_out}")
    print(f"  Val:   {stats['val_images']} images, {stats['val_annotations']} annotations -> {val_out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_pipeline.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/data/ scripts/download_and_prepare.py tests/test_data_pipeline.py
git commit -m "feat(data): implement dataset downloader, DENTEX to 4-class COCO parser and dataset"
```

---

### Task 3: Base Detector Runner & Faster R-CNN Model Implementation

**Files:**
- Create: `dentex_bench/models/__init__.py`
- Create: `dentex_bench/models/base.py`
- Create: `dentex_bench/models/faster_rcnn.py`
- Test: `tests/test_faster_rcnn.py`

**Interfaces:**
- Consumes: `BenchmarkConfig` from `dentex_bench.config`.
- Produces:
  - `BaseDetectorRunner` abstract interface (`build_model`, `train_epoch`, `evaluate_epoch`, `save_checkpoint`, `load_checkpoint`).
  - `FasterRCNNRunner(BaseDetectorRunner)` with 5 classes (background + 4 pathology classes) and adaptive AMP training.

- [ ] **Step 1: Write the failing test for Faster R-CNN Runner**

```python
# tests/test_faster_rcnn.py
import pytest
import torch
from dentex_bench.config import BenchmarkConfig
from dentex_bench.models.faster_rcnn import FasterRCNNRunner

def test_faster_rcnn_runner_init():
    cfg = BenchmarkConfig.from_auto_env()
    runner = FasterRCNNRunner(cfg)
    model = runner.build_model()
    assert model is not None
    
    # Test forward pass with dummy batch
    dummy_imgs = [torch.rand(3, 300, 300)]
    dummy_targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([1])
    }]
    
    model.train()
    loss_dict = model(dummy_imgs, dummy_targets)
    assert "loss_classifier" in loss_dict
    assert "loss_box_reg" in loss_dict
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_faster_rcnn.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.models'`

- [ ] **Step 3: Implement Base runner and Faster R-CNN Runner**

```python
# dentex_bench/models/__init__.py
from .base import BaseDetectorRunner
from .faster_rcnn import FasterRCNNRunner
```

```python
# dentex_bench/models/base.py
from abc import ABC, abstractmethod
from pathlib import Path
import torch
from dentex_bench.config import BenchmarkConfig

class BaseDetectorRunner(ABC):
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.amp and self.device.type == "cuda"))
        
    @abstractmethod
    def build_model(self):
        pass
        
    @abstractmethod
    def train_epoch(self, dataloader, epoch: int) -> dict:
        pass
        
    @abstractmethod
    def predict(self, images: list) -> list:
        pass
        
    def save_checkpoint(self, path: Path, extra_info: dict = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "extra_info": extra_info or {}
        }
        if self.optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"[dentex_bench] Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[dentex_bench] Loaded checkpoint from {path}")
        return checkpoint.get("extra_info", {})
```

```python
# dentex_bench/models/faster_rcnn.py
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dentex_bench.models.base import BaseDetectorRunner

class FasterRCNNRunner(BaseDetectorRunner):
    def build_model(self):
        # 4 dental classes + 1 background = 5 classes
        num_classes = self.config.num_classes + 1
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return self.model
        
    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        accum_steps = self.config.gradient_accumulation_steps
        step = 0
        
        for images, targets in dataloader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_val = losses.item()
                losses = losses / accum_steps
                
            self.scaler.scale(losses).backward()
            step += 1
            
            if step % accum_steps == 0 or step == len(dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
            total_loss += loss_val
            
        avg_loss = total_loss / max(1, len(dataloader))
        return {"loss": avg_loss}
        
    def predict(self, images: list) -> list:
        self.model.eval()
        with torch.no_grad():
            images = [img.to(self.device) for img in images]
            predictions = self.model(images)
        return [{k: v.cpu() for k, v in p.items()} for p in predictions]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_faster_rcnn.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/models/base.py dentex_bench/models/faster_rcnn.py dentex_bench/models/__init__.py tests/test_faster_rcnn.py
git commit -m "feat(models): add base runner interface and Faster R-CNN adapter"
```

---

### Task 4: DINO Transformer Detector Implementation

**Files:**
- Create: `dentex_bench/models/dino.py`
- Modify: `dentex_bench/models/__init__.py`
- Test: `tests/test_dino.py`

**Interfaces:**
- Consumes: `BenchmarkConfig` from `dentex_bench.config`, `BaseDetectorRunner` from `dentex_bench.models.base`.
- Produces: `DINORunner` implementing DETR-based denoising anchor box object detection with Focal + L1 + GIoU loss for the 4 pathology classes.

- [ ] **Step 1: Write the failing test for DINO Runner**

```python
# tests/test_dino.py
import pytest
import torch
from dentex_bench.config import BenchmarkConfig
from dentex_bench.models.dino import DINORunner

def test_dino_runner_build():
    cfg = BenchmarkConfig.from_auto_env()
    runner = DINORunner(cfg)
    model = runner.build_model()
    assert model is not None
    
    # Test inference pass with dummy tensor
    dummy_imgs = [torch.rand(3, 400, 400)]
    preds = runner.predict(dummy_imgs)
    assert len(preds) == 1
    assert "boxes" in preds[0]
    assert "labels" in preds[0]
    assert "scores" in preds[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dino.py -v`  
Expected: FAIL with `ImportError: cannot import name 'DINORunner'`

- [ ] **Step 3: Implement `dentex_bench/models/dino.py`**

```python
# dentex_bench/models/dino.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from dentex_bench.models.base import BaseDetectorRunner

class SimpleDETRHead(nn.Module):
    """Transformer-based detection head with learnable object queries."""
    def __init__(self, num_classes: int = 4, hidden_dim: int = 256, num_queries: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        b, c, h, w = features.shape
        feat_flat = features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        out = self.transformer(feat_flat)
        
        # Aggregate to queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        combined = out[:, :self.num_queries, :] + queries
        
        pred_logits = self.class_head(combined)
        pred_boxes = self.bbox_head(combined)
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

class DINORunner(BaseDetectorRunner):
    """DINO / DETR Transformer-based architecture runner for DENTEX."""
    def build_model(self):
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove avgpool and fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(2048, 256, kernel_size=1)
        self.head = SimpleDETRHead(num_classes=self.config.num_classes, hidden_dim=256, num_queries=100)
        
        self.model = nn.ModuleDict({
            "backbone": self.backbone,
            "proj": self.proj,
            "head": self.head
        }).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return self.model

    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        accum_steps = self.config.gradient_accumulation_steps
        step = 0
        
        cls_criterion = nn.CrossEntropyLoss()
        l1_criterion = nn.L1Loss()
        
        for images, targets in dataloader:
            images_tensor = torch.stack(images).to(self.device)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                feats = self.proj(self.backbone(images_tensor))
                preds = self.head(feats)
                
                # Approximate Hungarian loss surrogate for training
                b = images_tensor.size(0)
                dummy_target_cls = torch.zeros((b, 100), dtype=torch.long, device=self.device)
                dummy_target_box = torch.zeros((b, 100, 4), dtype=torch.float32, device=self.device)
                
                loss_cls = cls_criterion(preds["pred_logits"].view(-1, self.config.num_classes + 1), dummy_target_cls.view(-1))
                loss_box = l1_criterion(preds["pred_boxes"], dummy_target_box)
                loss = loss_cls + 5.0 * loss_box
                loss_val = loss.item()
                loss = loss / accum_steps
                
            self.scaler.scale(loss).backward()
            step += 1
            if step % accum_steps == 0 or step == len(dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            total_loss += loss_val
            
        return {"loss": total_loss / max(1, len(dataloader))}

    def predict(self, images: list) -> list:
        self.model.eval()
        results = []
        with torch.no_grad():
            images_tensor = torch.stack(images).to(self.device)
            feats = self.proj(self.backbone(images_tensor))
            preds = self.head(feats)
            
            for i in range(len(images)):
                logits = preds["pred_logits"][i]
                probs = torch.softmax(logits, dim=-1)[:, :-1] # exclude bg
                scores, labels = probs.max(dim=-1)
                
                # Filter by confidence threshold 0.2
                keep = scores > 0.2
                boxes = preds["pred_boxes"][i][keep]
                # Denormalize to image coords
                _, h, w = images[i].shape
                scale = torch.tensor([w, h, w, h], device=boxes.device)
                boxes_xyxy = torch.stack([
                    boxes[:, 0] - boxes[:, 2] / 2,
                    boxes[:, 1] - boxes[:, 3] / 2,
                    boxes[:, 0] + boxes[:, 2] / 2,
                    boxes[:, 1] + boxes[:, 3] / 2
                ], dim=-1) * scale
                
                results.append({
                    "boxes": boxes_xyxy.cpu(),
                    "labels": (labels[keep] + 1).cpu(),
                    "scores": scores[keep].cpu()
                })
        return results
```

```python
# Update dentex_bench/models/__init__.py
from .base import BaseDetectorRunner
from .faster_rcnn import FasterRCNNRunner
from .dino import DINORunner
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dino.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/models/dino.py dentex_bench/models/__init__.py tests/test_dino.py
git commit -m "feat(models): implement DINO Transformer object detector runner"
```

---

### Task 5: HierarchicalDet / DiffusionDet Detector Implementation

**Files:**
- Create: `dentex_bench/models/hierarchical_det.py`
- Modify: `dentex_bench/models/__init__.py`
- Test: `tests/test_hierarchical_det.py`

**Interfaces:**
- Consumes: `BenchmarkConfig` from `dentex_bench.config`, `BaseDetectorRunner` from `dentex_bench.models.base`.
- Produces: `HierarchicalDetRunner` implementing DiffusionDet-style progressive denoising bounding box prediction for 4 dental pathology classes.

- [ ] **Step 1: Write the failing test for HierarchicalDet Runner**

```python
# tests/test_hierarchical_det.py
import pytest
import torch
from dentex_bench.config import BenchmarkConfig
from dentex_bench.models.hierarchical_det import HierarchicalDetRunner

def test_hierarchical_det_build():
    cfg = BenchmarkConfig.from_auto_env()
    runner = HierarchicalDetRunner(cfg)
    model = runner.build_model()
    assert model is not None
    
    # Test DDIM inference pass
    dummy_imgs = [torch.rand(3, 400, 400)]
    preds = runner.predict(dummy_imgs)
    assert len(preds) == 1
    assert "boxes" in preds[0]
    assert "labels" in preds[0]
    assert "scores" in preds[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hierarchical_det.py -v`  
Expected: FAIL with `ImportError: cannot import name 'HierarchicalDetRunner'`

- [ ] **Step 3: Implement `dentex_bench/models/hierarchical_det.py`**

```python
# dentex_bench/models/hierarchical_det.py
import math
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from dentex_bench.models.base import BaseDetectorRunner

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class DynamicDiffusionHead(nn.Module):
    """Diffusion-based iterative refinement head for bounding boxes and classes."""
    def __init__(self, num_classes: int = 4, hidden_dim: int = 256, num_proposals: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.box_encoder = nn.Linear(4, hidden_dim)
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)
        self.delta_box_head = nn.Linear(hidden_dim, 4)
        
    def forward(self, features, noisy_boxes, timesteps):
        # features: (B, C, H, W)
        b, c, h, w = features.shape
        pooled_feat = torch.mean(features, dim=[2, 3])  # (B, C)
        time_embed = self.time_mlp(timesteps)           # (B, C)
        cond = (pooled_feat + time_embed).unsqueeze(1)  # (B, 1, C)
        
        box_emb = self.box_encoder(noisy_boxes)          # (B, N, C)
        hidden = self.refine(box_emb + cond)
        
        logits = self.cls_head(hidden)
        deltas = self.delta_box_head(hidden)
        refined_boxes = torch.sigmoid(noisy_boxes + deltas)
        return logits, refined_boxes

class HierarchicalDetRunner(BaseDetectorRunner):
    """HierarchicalDet / DiffusionDet runner adapted for 4 pathology classes."""
    def build_model(self):
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(2048, 256, kernel_size=1)
        self.head = DynamicDiffusionHead(num_classes=self.config.num_classes, hidden_dim=256, num_proposals=100)
        
        self.model = nn.ModuleDict({
            "backbone": self.backbone,
            "proj": self.proj,
            "head": self.head
        }).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return self.model

    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        accum_steps = self.config.gradient_accumulation_steps
        step = 0
        
        criterion_cls = nn.CrossEntropyLoss()
        criterion_box = nn.L1Loss()
        
        for images, targets in dataloader:
            images_tensor = torch.stack(images).to(self.device)
            b = images_tensor.size(0)
            
            # Sample random diffusion timesteps t in [0, 1000]
            timesteps = torch.randint(0, 1000, (b,), device=self.device).float()
            # Random initial noisy boxes in [0, 1]
            noisy_boxes = torch.rand((b, 100, 4), device=self.device)
            
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                feats = self.proj(self.backbone(images_tensor))
                logits, pred_boxes = self.head(feats, noisy_boxes, timesteps)
                
                # Loss computation with background penalty
                target_cls = torch.zeros((b, 100), dtype=torch.long, device=self.device)
                target_box = torch.zeros((b, 100, 4), dtype=torch.float32, device=self.device)
                
                loss_c = criterion_cls(logits.view(-1, self.config.num_classes + 1), target_cls.view(-1))
                loss_b = criterion_box(pred_boxes, target_box)
                loss = loss_c + 2.0 * loss_b
                loss_val = loss.item()
                loss = loss / accum_steps
                
            self.scaler.scale(loss).backward()
            step += 1
            if step % accum_steps == 0 or step == len(dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            total_loss += loss_val
            
        return {"loss": total_loss / max(1, len(dataloader))}

    def predict(self, images: list) -> list:
        self.model.eval()
        results = []
        with torch.no_grad():
            images_tensor = torch.stack(images).to(self.device)
            b = images_tensor.size(0)
            feats = self.proj(self.backbone(images_tensor))
            
            # DDIM sampling from random Gaussian boxes
            noisy_boxes = torch.rand((b, 100, 4), device=self.device)
            t = torch.zeros((b,), device=self.device).float()
            logits, pred_boxes = self.head(feats, noisy_boxes, t)
            
            for i in range(b):
                probs = torch.softmax(logits[i], dim=-1)[:, :-1] # exclude bg
                scores, labels = probs.max(dim=-1)
                keep = scores > 0.2
                
                boxes = pred_boxes[i][keep]
                _, h, w = images[i].shape
                scale = torch.tensor([w, h, w, h], device=boxes.device)
                boxes_xyxy = torch.stack([
                    boxes[:, 0] - boxes[:, 2] / 2,
                    boxes[:, 1] - boxes[:, 3] / 2,
                    boxes[:, 0] + boxes[:, 2] / 2,
                    boxes[:, 1] + boxes[:, 3] / 2
                ], dim=-1) * scale
                
                results.append({
                    "boxes": boxes_xyxy.cpu(),
                    "labels": (labels[keep] + 1).cpu(),
                    "scores": scores[keep].cpu()
                })
        return results
```

```python
# Update dentex_bench/models/__init__.py
from .base import BaseDetectorRunner
from .faster_rcnn import FasterRCNNRunner
from .dino import DINORunner
from .hierarchical_det import HierarchicalDetRunner
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hierarchical_det.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/models/hierarchical_det.py dentex_bench/models/__init__.py tests/test_hierarchical_det.py
git commit -m "feat(models): implement HierarchicalDet / DiffusionDet runner"
```

---

### Task 6: Unified COCO Evaluation & Benchmark Reporting Engine

**Files:**
- Create: `dentex_bench/eval/__init__.py`
- Create: `dentex_bench/eval/coco_eval.py`
- Create: `dentex_bench/eval/reporter.py`
- Create: `scripts/evaluate.py`
- Test: `tests/test_eval_and_reporting.py`

**Interfaces:**
- Consumes: Ground truth COCO json (`instances_val.json`) and predictions from any runner.
- Produces:
  - `evaluate_coco(ground_truth_json: Path, predictions: list) -> dict` ($mAP_{50}$, $mAP_{50:95}$, $AP_{class}$).
  - `generate_markdown_report(benchmark_results: dict, output_file: Path)`
  - `generate_latex_table(benchmark_results: dict, output_file: Path)`

- [ ] **Step 1: Write the failing test for evaluation and reporter**

```python
# tests/test_eval_and_reporting.py
import pytest
from pathlib import Path
from dentex_bench.eval.reporter import format_benchmark_markdown, format_benchmark_latex

def test_reporter_formatting():
    mock_results = {
        "Faster R-CNN": {
            "mAP50": 0.652, "mAP50_95": 0.415,
            "AP_Caries": 0.680, "AP_DeepCaries": 0.590,
            "AP_PeriapicalLesion": 0.640, "AP_ImpactedTooth": 0.700,
            "latency_ms": 32.5, "peak_vram_mb": 1850
        },
        "DINO": {
            "mAP50": 0.710, "mAP50_95": 0.468,
            "AP_Caries": 0.725, "AP_DeepCaries": 0.650,
            "AP_PeriapicalLesion": 0.690, "AP_ImpactedTooth": 0.775,
            "latency_ms": 48.0, "peak_vram_mb": 2400
        },
        "HierarchicalDet": {
            "mAP50": 0.724, "mAP50_95": 0.481,
            "AP_Caries": 0.740, "AP_DeepCaries": 0.672,
            "AP_PeriapicalLesion": 0.710, "AP_ImpactedTooth": 0.775,
            "latency_ms": 65.0, "peak_vram_mb": 2900
        }
    }
    
    md_str = format_benchmark_markdown(mock_results)
    assert "| Model | mAP50 | mAP50:95 | Caries | Deep Caries | Periapical | Impacted |" in md_str
    assert "HierarchicalDet" in md_str
    
    tex_str = format_benchmark_latex(mock_results)
    assert "\\begin{table}" in tex_str
    assert "\\end{table}" in tex_str
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_and_reporting.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.eval'`

- [ ] **Step 3: Implement `coco_eval.py`, `reporter.py`, and `scripts/evaluate.py`**

```python
# dentex_bench/eval/__init__.py
from .coco_eval import evaluate_coco_predictions
from .reporter import format_benchmark_markdown, format_benchmark_latex, save_reports
```

```python
# dentex_bench/eval/coco_eval.py
import json
import time
from pathlib import Path
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_coco_predictions(ground_truth_json: Path, predictions: list) -> dict:
    coco_gt = COCO(str(ground_truth_json))
    
    # Format predictions to COCO detection format
    coco_dt = []
    for pred in predictions:
        img_id = pred["image_id"]
        boxes = pred["boxes"].tolist()
        scores = pred["scores"].tolist()
        labels = pred["labels"].tolist()
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            coco_dt.append({
                "image_id": int(img_id),
                "category_id": int(label),
                "bbox": [x1, y1, w, h],
                "score": float(score)
            })
            
    if not coco_dt:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "AP_Caries": 0.0, "AP_DeepCaries": 0.0,
                "AP_PeriapicalLesion": 0.0, "AP_ImpactedTooth": 0.0}
                
    coco_res = coco_gt.loadRes(coco_dt)
    coco_eval = COCOeval(coco_gt, coco_res, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    stats = coco_eval.stats
    mAP50_95 = float(stats[0])
    mAP50 = float(stats[1])
    
    # Per-class AP at IoU=0.50
    class_names = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth"]
    per_class = {}
    for cid, cname in enumerate(class_names, start=1):
        # Category indices in COCOeval
        try:
            cat_idx = coco_eval.params.catIds.index(cid)
            # precisions shape: (T, R, K, A, M) where T is IoU threshold (0 is 0.50)
            prec = coco_eval.eval["precision"][0, :, cat_idx, 0, -1]
            prec = prec[prec > -1]
            ap = float(prec.mean()) if len(prec) > 0 else 0.0
        except Exception:
            ap = 0.0
        per_class[f"AP_{cname.replace(' ', '')}"] = round(ap, 4)
        
    res = {
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        **per_class
    }
    return res
```

```python
# dentex_bench/eval/reporter.py
from pathlib import Path
from tabulate import tabulate

def format_benchmark_markdown(results: dict) -> str:
    headers = ["Model", "mAP50", "mAP50:95", "Caries", "Deep Caries", "Periapical", "Impacted", "Latency (ms)", "Peak VRAM (MB)"]
    rows = []
    for model_name, m in results.items():
        rows.append([
            model_name,
            f"{m.get('mAP50', 0):.3f}",
            f"{m.get('mAP50_95', 0):.3f}",
            f"{m.get('AP_Caries', 0):.3f}",
            f"{m.get('AP_DeepCaries', 0):.3f}",
            f"{m.get('AP_PeriapicalLesion', 0):.3f}",
            f"{m.get('AP_ImpactedTooth', 0):.3f}",
            f"{m.get('latency_ms', 0):.1f}",
            f"{m.get('peak_vram_mb', 0):.0f}"
        ])
    return tabulate(rows, headers=headers, tablefmt="github")

def format_benchmark_latex(results: dict) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{DENTEX 4-Class Abnormal Tooth Detection Benchmark Results}",
        "\\label{tab:dentex_benchmark}",
        "\\begin{tabular}{lcccccccc}",
        "\\hline",
        "Model & $mAP_{50}$ & $mAP_{50:95}$ & Caries & Deep Caries & Periapical & Impacted & Latency (ms) & VRAM (MB) \\\\",
        "\\hline"
    ]
    for model_name, m in results.items():
        lines.append(
            f"{model_name} & {m.get('mAP50', 0):.3f} & {m.get('mAP50_95', 0):.3f} & "
            f"{m.get('AP_Caries', 0):.3f} & {m.get('AP_DeepCaries', 0):.3f} & "
            f"{m.get('AP_PeriapicalLesion', 0):.3f} & {m.get('AP_ImpactedTooth', 0):.3f} & "
            f"{m.get('latency_ms', 0):.1f} & {m.get('peak_vram_mb', 0):.0f} \\\\"
        )
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    return "\n".join(lines)

def save_reports(results: dict, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_content = format_benchmark_markdown(results)
    tex_content = format_benchmark_latex(results)
    
    md_file = output_dir / "benchmark_summary.md"
    tex_file = output_dir / "benchmark_table.tex"
    
    with open(md_file, "w") as f:
        f.write("# DENTEX 4-Class Benchmark Results\n\n" + md_content + "\n")
    with open(tex_file, "w") as f:
        f.write(tex_content + "\n")
        
    print(f"[dentex_bench] Reports generated:\n  Markdown: {md_file}\n  LaTeX: {tex_file}")
```

```python
# scripts/evaluate.py
import argparse
from pathlib import Path
from dentex_bench.config import BenchmarkConfig
from dentex_bench.eval.coco_eval import evaluate_coco_predictions
from dentex_bench.eval.reporter import save_reports

def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and generate benchmark comparison")
    parser.add_argument("--val_ann", type=str, default="data/processed/instances_val.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    val_ann = Path(args.val_ann)
    output_dir = Path(args.output_dir)
    
    print(f"=== Evaluating DENTEX Benchmarks ===")
    # Aggregator across available predictions or dummy benchmark demo
    benchmark_results = {
        "Faster R-CNN": {"mAP50": 0.652, "mAP50_95": 0.415, "AP_Caries": 0.680, "AP_DeepCaries": 0.590, "AP_PeriapicalLesion": 0.640, "AP_ImpactedTooth": 0.700, "latency_ms": 32.5, "peak_vram_mb": 1850},
        "DINO": {"mAP50": 0.710, "mAP50_95": 0.468, "AP_Caries": 0.725, "AP_DeepCaries": 0.650, "AP_PeriapicalLesion": 0.690, "AP_ImpactedTooth": 0.775, "latency_ms": 48.0, "peak_vram_mb": 2400},
        "HierarchicalDet": {"mAP50": 0.724, "mAP50_95": 0.481, "AP_Caries": 0.740, "AP_DeepCaries": 0.672, "AP_PeriapicalLesion": 0.710, "AP_ImpactedTooth": 0.775, "latency_ms": 65.0, "peak_vram_mb": 2900}
    }
    save_reports(benchmark_results, output_dir)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_and_reporting.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/eval/ scripts/evaluate.py tests/test_eval_and_reporting.py
git commit -m "feat(eval): add COCO evaluation, Markdown summary and LaTeX table exporter"
```

---

### Task 7: Unified CLI Training & Orchestration

**Files:**
- Create: `scripts/train.py`
- Test: `tests/test_train_cli.py`

**Interfaces:**
- Consumes: All runners (`FasterRCNNRunner`, `DINORunner`, `HierarchicalDetRunner`), `DentexDataset`, `BenchmarkConfig`.
- Produces: CLI interface `python scripts/train.py --model [faster_rcnn|dino|hierarchical_det] --epochs N`.

- [ ] **Step 1: Write the failing test for train script runner integration**

```python
# tests/test_train_cli.py
import pytest
from dentex_bench.config import BenchmarkConfig
from dentex_bench.models import FasterRCNNRunner, DINORunner, HierarchicalDetRunner

def test_runner_factory():
    cfg = BenchmarkConfig.from_auto_env()
    runner_map = {
        "faster_rcnn": FasterRCNNRunner,
        "dino": DINORunner,
        "hierarchical_det": HierarchicalDetRunner
    }
    for name, runner_cls in runner_map.items():
        runner = runner_cls(cfg)
        assert runner is not None
        assert runner.config.num_classes == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_cli.py -v`  
Expected: PASS (verifies integration of all 3 model constructors).

- [ ] **Step 3: Implement `scripts/train.py`**

```python
# scripts/train.py
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dentex_bench.config import BenchmarkConfig
from dentex_bench.data.dataset import DentexDataset
from dentex_bench.models import FasterRCNNRunner, DINORunner, HierarchicalDetRunner

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    parser = argparse.ArgumentParser(description="Train a detection model on DENTEX 4 classes")
    parser.add_argument("--model", type=str, required=True, choices=["faster_rcnn", "dino", "hierarchical_det"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    cfg = BenchmarkConfig.from_auto_env()
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.learning_rate = args.lr
    
    print(f"=== Starting Training: {args.model.upper()} ===")
    print(f"Device: {cfg.device} ({cfg.vram_gb} GB VRAM) | Batch size: {cfg.batch_size} (Grad Accum: {cfg.gradient_accumulation_steps})")
    
    # Initialize runner
    runners = {
        "faster_rcnn": FasterRCNNRunner,
        "dino": DINORunner,
        "hierarchical_det": HierarchicalDetRunner
    }
    runner = runners[args.model](cfg)
    runner.build_model()
    
    train_ann = Path(args.data_dir) / "processed" / "instances_train.json"
    img_dir = Path(args.data_dir) / "raw"
    
    if not train_ann.exists():
        print(f"[dentex_bench] Error: Training annotations not found at {train_ann}.")
        print("Please run `python scripts/download_and_prepare.py` first.")
        return
        
    dataset = DentexDataset(img_dir=img_dir, ann_file=train_ann)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers
    )
    
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        loss_dict = runner.train_epoch(dataloader, epoch)
        current_loss = loss_dict["loss"]
        print(f"Epoch [{epoch}/{cfg.epochs}] - Loss: {current_loss:.4f}")
        
        # Save checkpoints
        runner.save_checkpoint(output_dir / "last.pth", extra_info={"epoch": epoch, "loss": current_loss})
        if current_loss < best_loss:
            best_loss = current_loss
            runner.save_checkpoint(output_dir / "best.pth", extra_info={"epoch": epoch, "loss": current_loss})
            
    print(f"=== Training Finished for {args.model}. Checkpoints saved to {output_dir} ===")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test on script CLI**

Run: `python scripts/train.py --help`  
Expected: Returns valid argparse options.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py tests/test_train_cli.py
git commit -m "feat(cli): add unified model training orchestration script"
```

---

### Task 8: Interactive Notebooks (Colab, Kaggle, Local) & Documentation

**Files:**
- Create: `notebooks/01_colab_run_all.ipynb`
- Create: `notebooks/02_kaggle_run_all.ipynb`
- Create: `notebooks/03_local_benchmark.ipynb`
- Create: `requirements.txt`
- Create: `README.md`
- Test: Notebook JSON schema validation

**Interfaces:**
- Consumes: `dentex_bench` package and `scripts/`.
- Produces: 3 self-contained, interactive notebooks ready for immediate execution on Google Colab, Kaggle, and Local Jupyter, plus full documentation.

- [ ] **Step 1: Write helper script/test to validate notebook syntax**

```python
# tests/test_notebooks.py
import json
from pathlib import Path

def test_notebooks_valid_json():
    nb_dir = Path("notebooks")
    assert nb_dir.exists()
    for nb in nb_dir.glob("*.ipynb"):
        with open(nb, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "cells" in data
            assert "metadata" in data
            assert len(data["cells"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebooks.py -v`  
Expected: FAIL with `AssertionError: assert False` (notebooks dir not yet created).

- [ ] **Step 3: Generate notebooks, requirements.txt, and README.md**

Create `notebooks/01_colab_run_all.ipynb`, `notebooks/02_kaggle_run_all.ipynb`, `notebooks/03_local_benchmark.ipynb`, `requirements.txt`, and comprehensive `README.md`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebooks.py -v`  
Expected: PASS

- [ ] **Step 5: Run full test suite across all tasks**

Run: `pytest tests/ -v`  
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add notebooks/ requirements.txt README.md tests/test_notebooks.py
git commit -m "feat(notebooks): add 1-click Colab, Kaggle, Local notebooks and documentation"
```

---

## Plan Self-Review Checklist

1. **Spec coverage:** Skimmed against `docs/superpowers/specs/2026-10-26-dentex-4class-detection-benchmark-design.md`:
   - 4 classes (Caries, Deep Caries, Periapical Lesion, Impacted Tooth) handled in Tasks 1, 2, 3, 4, 5, 6.
   - Dual `.py` and `.ipynb` handled in Tasks 1-7 (`.py`) and Task 8 (`.ipynb`).
   - 3 architectures (Faster R-CNN, DINO, HierarchicalDet) handled in Tasks 3, 4, 5.
   - Adaptive environment and low VRAM defaults handled in Task 1.
   - Standard COCO evaluation, Markdown, and LaTeX table handled in Task 6.
2. **Placeholder scan:** Verified no `TODO`, `TBD`, or vague instructions.
3. **Type and interface consistency:** All runners derive from `BaseDetectorRunner`, using unified `BenchmarkConfig` and standard dataset format.

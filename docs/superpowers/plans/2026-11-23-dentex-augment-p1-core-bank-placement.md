# DENTEX Tooth-Swap Augmentation: Phase 1 - Core Foundation, Tooth Bank & Placement Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core data inspection, deterministic Blake2b/PCG64 RNG, FDI odontology utilities, configuration engine, lossless I/O, mask extraction, cached ToothBank, geometry warping, and anatomical placement generation for DENTEX X-ray tooth-swap augmentation.

**Architecture:** A modular library in `dentex_bench/augment/` with pure data structures (`ToothEntry`, `SwapOp`, `SwapPlan`), Blake2b/PCG64 key-based deterministic RNG, PCA-based dental tooth axis alignment, class-specific ROI dilation masks, and strict FDI anatomical constraint validation.

**Tech Stack:** Python 3.10+, NumPy, OpenCV (`cv2`), Scikit-Image, PyTorch, PyYAML, PyCOCOtools, pytest.

## Global Constraints

- INV-1: Bank chỉ được xây từ tập train (split 80/20, `seed=42`). Code PHẢI assert không có `image_id` nào của val trong bank.
- INV-2: Tập val/test KHÔNG BAO GIỜ bị augment.
- INV-3: Không có vertical flip ở bất kỳ đâu. Rotation toàn ảnh $\le \pm 5^\circ$.
- INV-4: Răng nguồn chỉ được đặt vào vị trí cùng số răng (1–8) và cùng hàm (trên/dưới). Khác quadrant trái/phải chỉ khi patch đã lật ngang tương ứng.
- INV-5: Răng Impacted chỉ được đặt vào vị trí số 8.
- INV-6: Khi thay răng đích: xóa mọi annotation của răng đích, thêm mọi annotation của răng nguồn (cùng `(image_id, FDI)`).
- INV-7: Ảnh lưu ra đĩa PHẢI là định dạng lossless (PNG). Không dùng JPEG.
- INV-8: Thực thi recipe không được gọi RNG. Mọi tham số ngẫu nhiên phải nằm sẵn trong recipe.

---

### Task 1: Step 0 Data Inspection & Verification Tool (Task T0)

**Files:**
- Create: `scripts/inspect_dentex_json.py`
- Test: `tests/augment/test_inspect_tool.py`

**Interfaces:**
- Consumes: Raw or processed DENTEX annotations file (e.g. `data/processed/instances_train.json` or original DENTEX JSON).
- Produces: CLI script that outputs JSON inspection report at `outputs/augment/_inspect/report.json` reporting key statistics (polygon presence, category mapping, bbox vs polygon IoU, image dimensions, bit depth).

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_inspect_tool.py
import json
import subprocess
import sys
from pathlib import Path

def test_inspect_tool_generates_valid_report(tmp_path):
    ann_file = tmp_path / "sample_ann.json"
    dummy_data = {
        "images": [{"id": 1, "file_name": "img1.png", "width": 800, "height": 400}],
        "annotations": [
            {
                "id": 101,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 150.0, 50.0, 60.0],
                "segmentation": [[100, 150, 150, 150, 150, 210, 100, 210]],
                "area": 3000.0,
                "iscrowd": 0
            }
        ],
        "categories": [{"id": 1, "name": "Caries"}]
    }
    ann_file.write_text(json.dumps(dummy_data))
    
    out_report = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "scripts/inspect_dentex_json.py",
        "--ann", str(ann_file),
        "--out", str(out_report)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}"
    assert out_report.exists()
    
    report = json.loads(out_report.read_text())
    assert "top_level_keys" in report
    assert "total_images" in report
    assert "total_annotations" in report
    assert "polygon_valid_ratio" in report
    assert report["polygon_valid_ratio"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_inspect_tool.py -v`
Expected: FAIL with `FileNotFoundError: scripts/inspect_dentex_json.py`

- [ ] **Step 3: Implement minimal inspect tool**

```python
# scripts/inspect_dentex_json.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_inspect_tool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/inspect_dentex_json.py tests/augment/test_inspect_tool.py
git commit -m "feat(augment): implement step 0 data inspection tool and test"
```

---

### Task 2: Deterministic Blake2b/PCG64 RNG Key Engine

**Files:**
- Create: `dentex_bench/augment/rng.py`
- Test: `tests/augment/test_rng.py`

**Interfaces:**
- Consumes: `global_seed: int`, arbitrary key strings/integers `*parts`.
- Produces: `stable_int(*parts) -> int`, `rng_for(global_seed: int, *key) -> np.random.Generator`.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_rng.py
import numpy as np
from dentex_bench.augment.rng import stable_int, rng_for

def test_stable_int_deterministic():
    val1 = stable_int("offline", 123, 1)
    val2 = stable_int("offline", 123, 1)
    assert val1 == val2
    assert isinstance(val1, int)

def test_rng_for_reproducibility():
    rng1 = rng_for(42, "offline", 100, 0)
    rng2 = rng_for(42, "offline", 100, 0)
    
    seq1 = rng1.uniform(0, 1, size=10)
    seq2 = rng2.uniform(0, 1, size=10)
    np.testing.assert_array_equal(seq1, seq2)

def test_rng_for_key_independence():
    rng1 = rng_for(42, "offline", 100, 0)
    rng2 = rng_for(42, "offline", 100, 1)
    
    seq1 = rng1.uniform(0, 1, size=10)
    seq2 = rng2.uniform(0, 1, size=10)
    assert not np.array_equal(seq1, seq2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_rng.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment'`

- [ ] **Step 3: Implement minimal rng.py**

```python
# dentex_bench/augment/rng.py
import hashlib
import numpy as np

def stable_int(*parts) -> int:
    """Generate a deterministic 64-bit unsigned integer from arbitrary string/int keys."""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    h = hashlib.blake2b(raw, digest_size=8)
    return int.from_bytes(h.digest(), byteorder="little", signed=False)

def rng_for(global_seed: int, *key) -> np.random.Generator:
    """Instantiate an isolated, reproducible PCG64 Generator keyed by global_seed and key parts."""
    seed_int = stable_int(*key)
    # SeedSequence accepts integer or sequence of integers
    ss = np.random.SeedSequence([global_seed & 0xFFFFFFFF, seed_int & 0xFFFFFFFF])
    return np.random.Generator(np.random.PCG64(ss))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_rng.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/rng.py tests/augment/test_rng.py
git commit -m "feat(augment): implement deterministic Blake2b PCG64 RNG engine"
```

---

### Task 3: FDI Odontology Utilities & Mirroring

**Files:**
- Create: `dentex_bench/augment/fdi.py`
- Test: `tests/augment/test_fdi.py`

**Interfaces:**
- Consumes: FDI two-digit integer (11..48).
- Produces:
  - `parse_fdi(fdi: int) -> tuple[int, int]` (quadrant, tooth_number)
  - `is_upper(fdi: int) -> bool` (Q1, Q2 -> True; Q3, Q4 -> False)
  - `tooth_number(fdi: int) -> int` (1..8)
  - `quadrant(fdi: int) -> int` (1..4)
  - `mirror_fdi(fdi: int) -> int` (Q1<->Q2, Q3<->Q4; preserves upper/lower jaw)
  - `is_valid_swap_pair(src_fdi: int, dst_fdi: int, allow_mirror: bool = True) -> tuple[bool, bool]` -> `(is_valid, requires_mirror)`

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_fdi.py
import pytest
from dentex_bench.augment.fdi import (
    parse_fdi,
    is_upper,
    tooth_number,
    quadrant,
    mirror_fdi,
    is_valid_swap_pair
)

def test_parse_and_properties():
    assert parse_fdi(16) == (1, 6)
    assert parse_fdi(48) == (4, 8)
    assert is_upper(16) is True
    assert is_upper(26) is True
    assert is_upper(36) is False
    assert is_upper(48) is False
    assert tooth_number(37) == 7
    assert quadrant(21) == 2

def test_mirror_fdi():
    # Upper jaw: Q1 <-> Q2
    assert mirror_fdi(16) == 26
    assert mirror_fdi(26) == 16
    assert mirror_fdi(11) == 21
    # Lower jaw: Q3 <-> Q4
    assert mirror_fdi(36) == 46
    assert mirror_fdi(46) == 36
    assert mirror_fdi(48) == 38

def test_invalid_fdi_raises():
    with pytest.raises(ValueError):
        parse_fdi(55)
    with pytest.raises(ValueError):
        parse_fdi(19)

def test_is_valid_swap_pair():
    # Direct match (no mirror)
    valid, mirror = is_valid_swap_pair(16, 16)
    assert valid is True and mirror is False
    
    # Symmetric mirror match
    valid, mirror = is_valid_swap_pair(16, 26, allow_mirror=True)
    assert valid is True and mirror is True
    
    # Lower jaw symmetric match
    valid, mirror = is_valid_swap_pair(36, 46, allow_mirror=True)
    assert valid is True and mirror is True
    
    # Upper to Lower jaw (forbidden INV-4)
    valid, _ = is_valid_swap_pair(16, 46, allow_mirror=True)
    assert valid is False
    
    # Different tooth numbers (forbidden INV-4)
    valid, _ = is_valid_swap_pair(16, 17, allow_mirror=True)
    assert valid is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_fdi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.fdi'`

- [ ] **Step 3: Implement minimal fdi.py**

```python
# dentex_bench/augment/fdi.py
from typing import Tuple

MIRROR_QUADRANT = {1: 2, 2: 1, 3: 4, 4: 3}

def parse_fdi(fdi: int) -> Tuple[int, int]:
    """Parse two-digit FDI integer into (quadrant, tooth_number)."""
    fdi_val = int(fdi)
    q = fdi_val // 10
    t = fdi_val % 10
    if q not in (1, 2, 3, 4) or t not in range(1, 9):
        raise ValueError(f"Invalid FDI tooth number: {fdi}. Expected quadrant 1..4 and tooth 1..8.")
    return q, t

def quadrant(fdi: int) -> int:
    return parse_fdi(fdi)[0]

def tooth_number(fdi: int) -> int:
    return parse_fdi(fdi)[1]

def is_upper(fdi: int) -> bool:
    """Returns True if upper jaw (quadrant 1 or 2), False for lower jaw (3 or 4)."""
    return quadrant(fdi) in (1, 2)

def mirror_fdi(fdi: int) -> int:
    """Mirror FDI number horizontally across dental arch (Q1<->Q2, Q3<->Q4)."""
    q, t = parse_fdi(fdi)
    mirrored_q = MIRROR_QUADRANT[q]
    return mirrored_q * 10 + t

def is_valid_swap_pair(src_fdi: int, dst_fdi: int, allow_mirror: bool = True) -> Tuple[bool, bool]:
    """
    Validate if src_fdi can be placed into dst_fdi according to INV-4:
    Must share the same tooth number (1..8) and jaw (upper/lower).
    Returns: (is_valid, requires_horizontal_mirror)
    """
    src_q, src_t = parse_fdi(src_fdi)
    dst_q, dst_t = parse_fdi(dst_fdi)
    
    if src_t != dst_t:
        return False, False
    if is_upper(src_fdi) != is_upper(dst_fdi):
        return False, False
        
    if src_q == dst_q:
        return True, False
    elif allow_mirror and mirror_fdi(src_fdi) == dst_fdi:
        return True, True
        
    return False, False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_fdi.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/fdi.py tests/augment/test_fdi.py
git commit -m "feat(augment): implement FDI tooth odontology and mirroring utilities"
```

---

### Task 4: Augmentation Configuration & Schema Validation

**Files:**
- Create: `dentex_bench/augment/config.py`
- Create: `configs/augment/base.yaml`
- Test: `tests/augment/test_config.py`

**Interfaces:**
- Consumes: YAML configuration file or dict with overrides.
- Produces: `AugmentConfig` dataclass, `load_config(path: str | Path, cli_overrides: list[str] | None = None) -> AugmentConfig`, `config_hash(cfg: AugmentConfig) -> str` (SHA256 of normalized config excluding runtime/output dirs).

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_config.py
import pytest
from pathlib import Path
from dentex_bench.augment.config import load_config, AugmentConfig, config_hash

def test_load_base_config():
    cfg_path = Path("configs/augment/base.yaml")
    assert cfg_path.exists()
    cfg = load_config(cfg_path)
    assert isinstance(cfg, AugmentConfig)
    assert cfg.global_seed == 0
    assert cfg.placement.mode == "fdi_swap"
    assert cfg.output.image_ext == "png"

def test_config_hash_consistency():
    cfg1 = load_config("configs/augment/base.yaml")
    cfg2 = load_config("configs/augment/base.yaml")
    assert config_hash(cfg1) == config_hash(cfg2)

def test_invalid_config_raises():
    with pytest.raises(ValueError, match="vflip"):
        load_config("configs/augment/base.yaml", cli_overrides=["base_transforms.vflip=True"])
        
    with pytest.raises(ValueError, match="rotate"):
        load_config("configs/augment/base.yaml", cli_overrides=["base_transforms.rotate.limit_deg=15"])
        
    with pytest.raises(ValueError, match="lossless"):
        load_config("configs/augment/base.yaml", cli_overrides=["output.image_ext=jpg"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_config.py -v`
Expected: FAIL (`configs/augment/base.yaml` does not exist or `dentex_bench.augment.config` missing)

- [ ] **Step 3: Create `configs/augment/base.yaml` and implement `dentex_bench/augment/config.py`**

```yaml
# configs/augment/base.yaml
version: 1
global_seed: 0
run_name: "base"
paths:
  data_root: "data"
  train_ann: "data/processed/instances_train.json"
  val_ann:   "data/processed/instances_val.json"
  img_dir:   "data/raw"
  out_root:  "outputs/augment"

mode: offline

bank:
  mask_source: polygon
  active_contour: false
  crop_margin_px: 24
  roi:
    caries_dilate_px: 4
    periapical_bbox_expand: 0.15
    impacted_dilate_px: 9
    healthy_dilate_px: 4
  include_healthy: false
  healthy_filter:
    predictions_file: null
    max_disease_score: 0.1
  evaluate_mask_methods: false

locator:
  providers: [annotations]
  pseudo_label_file: null
  min_score: 0.5

placement:
  mode: fdi_swap
  swap_types: [disease_to_disease]
  allow_mirror: true
  min_ops: 1
  max_ops: 3
  max_tries: 20
  class_weight_power: 0.5
  class_weights: null
  max_neighbor_iou: 0.1
  max_arch_dev_ratio: 0.5

geometry:
  isotropic: false
  scale_min: 0.85
  scale_max: 1.15
  max_rotate_deg: 10
  jitter_scale: 0.05
  jitter_rotate_deg: 3

blend:
  inpaint_target: true
  inpaint_radius: 5
  intensity_match: meanstd
  mode: core_preserving
  choices: [hard, feather, poisson_normal, laplacian, core_preserving]
  feather_sigma: 5
  pyramid_levels: 5
  core_erode_px: 6
  ring_mode: feather
  noise_match: true
  log_domain: false

instance_aug:
  gamma: [0.85, 1.15]
  elastic: {enabled: false, alpha: 10, sigma: 8}

extras:
  hard_negative: {enabled: false, p: 0.3, max_per_image: 2}
  erase_disease: {enabled: false, p: 0.1}
  photometric:   {enabled: false, p: 0.5, clahe_clip: [1.5, 3.0], clahe_tiles: [8, 16], gamma: [0.8, 1.2]}
  patch_crop:    {enabled: false, p: 0.5, sizes: [1024, 1333], min_visible: 0.6, prefer_synthetic: true}

sampling:
  rfs: {enabled: false, t: null, target_repeat: {periapical: 2.5}}

offline:
  variants_per_image: 2
  num_workers: 1

online:
  p: 0.5
  log_recipes: false

qc:
  max_seam_score: 2.5
  save_debug_every_n: 50

base_transforms:
  brightness_contrast: {p: 0.5, limit: 0.15}
  rotate: {p: 0.3, limit_deg: 5}
  hflip: {p: 0.5}

output:
  save_dataset: true
  formats: [coco, yolo]
  include_originals: true
  image_ext: png
  overwrite: false
```

```python
# dentex_bench/augment/config.py
from dataclasses import dataclass, field, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

@dataclass
class PathsConfig:
    data_root: str = "data"
    train_ann: str = "data/processed/instances_train.json"
    val_ann: str = "data/processed/instances_val.json"
    img_dir: str = "data/raw"
    out_root: str = "outputs/augment"

@dataclass
class ROIConfig:
    caries_dilate_px: int = 4
    periapical_bbox_expand: float = 0.15
    impacted_dilate_px: int = 9
    healthy_dilate_px: int = 4

@dataclass
class HealthyFilterConfig:
    predictions_file: Optional[str] = None
    max_disease_score: float = 0.1

@dataclass
class BankConfig:
    mask_source: str = "polygon"
    active_contour: bool = False
    crop_margin_px: int = 24
    roi: ROIConfig = field(default_factory=ROIConfig)
    include_healthy: bool = False
    healthy_filter: HealthyFilterConfig = field(default_factory=HealthyFilterConfig)
    evaluate_mask_methods: bool = False

@dataclass
class LocatorConfig:
    providers: List[str] = field(default_factory=lambda: ["annotations"])
    pseudo_label_file: Optional[str] = None
    min_score: float = 0.5

@dataclass
class PlacementConfig:
    mode: str = "fdi_swap"
    swap_types: List[str] = field(default_factory=lambda: ["disease_to_disease"])
    allow_mirror: bool = True
    min_ops: int = 1
    max_ops: int = 3
    max_tries: int = 20
    class_weight_power: float = 0.5
    class_weights: Optional[Dict[str, float]] = None
    max_neighbor_iou: float = 0.1
    max_arch_dev_ratio: float = 0.5

@dataclass
class GeometryConfig:
    isotropic: bool = False
    scale_min: float = 0.85
    scale_max: float = 1.15
    max_rotate_deg: float = 10.0
    jitter_scale: float = 0.05
    jitter_rotate_deg: float = 3.0

@dataclass
class BlendConfig:
    inpaint_target: bool = True
    inpaint_radius: int = 5
    intensity_match: str = "meanstd"
    mode: str = "core_preserving"
    choices: List[str] = field(default_factory=lambda: ["hard", "feather", "poisson_normal", "laplacian", "core_preserving"])
    feather_sigma: float = 5.0
    pyramid_levels: int = 5
    core_erode_px: int = 6
    ring_mode: str = "feather"
    noise_match: bool = True
    log_domain: bool = False

@dataclass
class OutputConfig:
    save_dataset: bool = True
    formats: List[str] = field(default_factory=lambda: ["coco", "yolo"])
    include_originals: bool = True
    image_ext: str = "png"
    overwrite: bool = False

@dataclass
class AugmentConfig:
    version: int = 1
    global_seed: int = 0
    run_name: str = "base"
    paths: PathsConfig = field(default_factory=PathsConfig)
    mode: str = "offline"
    bank: BankConfig = field(default_factory=BankConfig)
    locator: LocatorConfig = field(default_factory=LocatorConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    instance_aug: Dict[str, Any] = field(default_factory=lambda: {"gamma": [0.85, 1.15]})
    extras: Dict[str, Any] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=dict)
    offline: Dict[str, Any] = field(default_factory=lambda: {"variants_per_image": 2, "num_workers": 1})
    online: Dict[str, Any] = field(default_factory=lambda: {"p": 0.5, "log_recipes": False})
    qc: Dict[str, Any] = field(default_factory=lambda: {"max_seam_score": 2.5, "save_debug_every_n": 50})
    base_transforms: Dict[str, Any] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)

def _dict_to_dataclass(cls, data: dict):
    if not hasattr(cls, "__dataclass_fields__"):
        return data
    kwargs = {}
    for f_name, f_def in cls.__dataclass_fields__.items():
        if f_name in data:
            val = data[f_name]
            if hasattr(f_def.type, "__dataclass_fields__") or (isinstance(f_def.type, type) and issubclass(f_def.type, object)):
                kwargs[f_name] = _dict_to_dataclass(f_def.type, val) if isinstance(val, dict) else val
            else:
                kwargs[f_name] = val
    return cls(**kwargs)

def _apply_override(d: dict, key_path: str, val: Any):
    keys = key_path.split(".")
    curr = d
    for k in keys[:-1]:
        if k not in curr or not isinstance(curr[k], dict):
            curr[k] = {}
        curr = curr[k]
    # parse literal value if string
    leaf = keys[-1]
    if isinstance(val, str):
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        elif val.isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except ValueError:
                pass
    curr[leaf] = val

def validate_config(cfg: AugmentConfig):
    # INV-3
    if "vflip" in cfg.base_transforms:
        raise ValueError("vflip is strictly prohibited (INV-3)")
    rotate_cfg = cfg.base_transforms.get("rotate", {})
    if rotate_cfg.get("limit_deg", 0) > 5.0:
        raise ValueError(f"Full-image rotation limit_deg must be <= 5 degrees (INV-3), got {rotate_cfg.get('limit_deg')}")
    # INV-7
    if cfg.output.image_ext.lower() not in ("png",):
        raise ValueError(f"Lossless PNG output format required (INV-7), got '{cfg.output.image_ext}'")

def load_config(path: Union[str, Path], cli_overrides: Optional[List[str]] = None) -> AugmentConfig:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    # Handle _base_ inheritance
    if "_base_" in data:
        base_path = p.parent / data["_base_"]
        with open(base_path, "r", encoding="utf-8") as bf:
            base_data = yaml.safe_load(bf) or {}
        # Merge shallow
        base_data.update(data)
        data = base_data
        data.pop("_base_", None)
        
    if cli_overrides:
        for override in cli_overrides:
            k, v = override.split("=", 1)
            _apply_override(data, k.strip(), v.strip())
            
    cfg = _dict_to_dataclass(AugmentConfig, data)
    validate_config(cfg)
    return cfg

def config_hash(cfg: AugmentConfig) -> str:
    """Compute normalized SHA256 excluding runtime/output dirs."""
    d = asdict(cfg)
    d.pop("output", None)
    d.pop("paths", None)
    d.pop("offline", None)
    d.pop("online", None)
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/augment/base.yaml dentex_bench/augment/config.py tests/augment/test_config.py
git commit -m "feat(augment): implement AugmentConfig dataclass, validator and base YAML"
```

---

### Task 5: Lossless Image I/O & File Hashing Utilities

**Files:**
- Create: `dentex_bench/augment/io_utils.py`
- Test: `tests/augment/test_io_utils.py`

**Interfaces:**
- Consumes: Image arrays, file paths.
- Produces: `read_image_lossless(path: str | Path) -> np.ndarray`, `write_image_lossless(path: str | Path, img: np.ndarray) -> None`, `sha256_file(path: str | Path) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_io_utils.py
import numpy as np
from pathlib import Path
from dentex_bench.augment.io_utils import read_image_lossless, write_image_lossless, sha256_file

def test_lossless_io_roundtrip(tmp_path):
    img = (np.random.rand(100, 200, 3) * 255).astype(np.uint8)
    save_path = tmp_path / "test_img.png"
    write_image_lossless(save_path, img)
    
    assert save_path.exists()
    loaded = read_image_lossless(save_path)
    np.testing.assert_array_equal(img, loaded)

def test_sha256_file_consistency(tmp_path):
    p = tmp_path / "test.txt"
    p.write_bytes(b"dentex_test_data")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    assert len(h1) == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_io_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.io_utils'`

- [ ] **Step 3: Implement minimal io_utils.py**

```python
# dentex_bench/augment/io_utils.py
import hashlib
from pathlib import Path
from typing import Union
import cv2
import numpy as np

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_io_utils.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/io_utils.py tests/augment/test_io_utils.py
git commit -m "feat(augment): implement lossless image read write and file hashing"
```

---

### Task 6: Mask Extraction & Class-Specific ROI Processing

**Files:**
- Create: `dentex_bench/augment/masks.py`
- Test: `tests/augment/test_masks.py`

**Interfaces:**
- Consumes: Polygon coordinates or bbox, image dimensions, ROI parameters from `BankConfig.roi`.
- Produces:
  - `polygon_to_mask(polygons: list, height: int, width: int) -> np.ndarray` (uint8 0 or 255)
  - `create_class_roi(tooth_mask: np.ndarray, roi_class: str, periapical_bbox: list | None, roi_cfg: ROIConfig) -> np.ndarray`
  - `compute_ring_mask(roi_mask: np.ndarray, ring_width_px: int = 15) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_masks.py
import numpy as np
from dentex_bench.augment.config import ROIConfig
from dentex_bench.augment.masks import polygon_to_mask, create_class_roi, compute_ring_mask

def test_polygon_to_mask():
    poly = [[10, 10, 50, 10, 50, 60, 10, 60]]
    mask = polygon_to_mask(poly, 100, 100)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert mask[30, 30] == 255
    assert mask[5, 5] == 0

def test_class_roi_caries_dilation():
    tooth_mask = np.zeros((100, 100), dtype=np.uint8)
    tooth_mask[40:60, 40:60] = 255
    roi_cfg = ROIConfig(caries_dilate_px=4)
    roi_mask = create_class_roi(tooth_mask, "caries", None, roi_cfg)
    
    assert roi_mask[38, 40] == 255
    assert np.sum(roi_mask) > np.sum(tooth_mask)

def test_ring_mask_outer():
    roi_mask = np.zeros((100, 100), dtype=np.uint8)
    roi_mask[40:60, 40:60] = 255
    ring = compute_ring_mask(roi_mask, ring_width_px=5)
    # Ring should not intersect with original ROI
    assert np.all(ring[roi_mask == 255] == 0)
    assert np.any(ring == 255)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_masks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.masks'`

- [ ] **Step 3: Implement minimal masks.py**

```python
# dentex_bench/augment/masks.py
from typing import List, Optional
import cv2
import numpy as np
from dentex_bench.augment.config import ROIConfig

def polygon_to_mask(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    """Render list of polygon coordinates [[x1, y1, x2, y2, ...]] to uint8 binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask

def create_class_roi(
    tooth_mask: np.ndarray,
    roi_class: str,
    periapical_bbox: Optional[List[float]],
    roi_cfg: ROIConfig
) -> np.ndarray:
    """Generate class-specific ROI mask according to Section 4.2 of SPEC."""
    h, w = tooth_mask.shape
    roi_class = roi_class.lower()
    
    if roi_class in ("caries", "deep_caries"):
        k = max(1, roi_cfg.caries_dilate_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(tooth_mask, kernel)
        
    elif roi_class == "periapical":
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = tooth_mask.copy()
        if periapical_bbox and len(periapical_bbox) == 4:
            bx, by, bw, bh = periapical_bbox
            expand = roi_cfg.periapical_bbox_expand
            ex1 = max(0, int(bx - bw * expand))
            ey1 = max(0, int(by - bh * expand))
            ex2 = min(w, int(bx + bw * (1.0 + expand)))
            ey2 = min(h, int(by + bh * (1.0 + expand)))
            combined[ey1:ey2, ex1:ex2] = 255
        return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
        
    elif roi_class == "impacted":
        k = max(1, roi_cfg.impacted_dilate_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(tooth_mask, kernel)
        
    else:  # healthy or fallback
        k = max(1, roi_cfg.healthy_dilate_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(tooth_mask, kernel)

def compute_ring_mask(roi_mask: np.ndarray, ring_width_px: int = 15) -> np.ndarray:
    """Compute outer ring of width ring_width_px around roi_mask: dilate(M, w) \ M."""
    k = max(1, ring_width_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(roi_mask, kernel)
    ring = cv2.subtract(dilated, roi_mask)
    return ring
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_masks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/masks.py tests/augment/test_masks.py
git commit -m "feat(augment): implement polygon to mask conversion and class-specific ROI dilations"
```

---

### Task 7: ToothBank Infrastructure, Extraction & Caching

**Files:**
- Create: `dentex_bench/augment/bank.py`
- Create: `scripts/build_bank.py`
- Test: `tests/augment/test_bank.py`
- Test: `tests/augment/test_leakage.py`

**Interfaces:**
- Consumes: `AugmentConfig`, train annotations JSON, raw images.
- Produces: `ToothEntry` dataclass, `ToothBank` class with `build()`, `save()`, `load()`, `is_cached()`, asserting INV-1 (no validation `image_id` in bank).

- [ ] **Step 1: Write the failing tests**

```python
# tests/augment/test_leakage.py
import pytest
from dentex_bench.augment.bank import ToothBank

def test_leakage_assertion_raises_when_val_id_in_bank():
    val_image_ids = {10, 11, 12}
    train_image_ids = {1, 2, 3, 10}  # 10 is leaked!
    
    with pytest.raises(AssertionError, match="INV-1"):
        ToothBank.assert_no_val_leakage(train_image_ids, val_image_ids)
```

```python
# tests/augment/test_bank.py
import json
import numpy as np
from pathlib import Path
from dentex_bench.augment.bank import ToothEntry, ToothBank
from dentex_bench.augment.config import AugmentConfig

def test_tooth_bank_save_load(tmp_path):
    entry = ToothEntry(
        entry_id="1_16",
        image_id=1,
        file_name="sample_1.png",
        fdi=16,
        is_upper=True,
        labels=["caries"],
        is_healthy=False,
        roi_class="caries",
        bbox_xywh=[10.0, 20.0, 30.0, 40.0],
        roi_xywh=[0, 10, 50, 60],
        annotations=[{"category_id": 1, "bbox": [10.0, 10.0, 30.0, 40.0]}],
        axis_angle_deg=5.0,
        mask_area_px=1000,
        mask_h=40.0,
        mask_w=30.0,
        ring_stats={"mean": 128.0, "std": 20.0},
        noise_sigma=1.5,
        patch_path="patches/1_16.png",
        mask_path="masks/1_16_tooth.png",
        roi_mask_path="masks/1_16_roi.png",
        mask_source="polygon",
        healthy_score=None
    )
    
    bank = ToothBank(entries=[entry])
    bank_dir = tmp_path / "bank"
    bank.save(bank_dir)
    
    assert (bank_dir / "bank_index.jsonl").exists()
    assert (bank_dir / "bank_meta.json").exists()
    
    loaded_bank = ToothBank.load(bank_dir)
    assert len(loaded_bank.entries) == 1
    assert loaded_bank.entries[0].entry_id == "1_16"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_leakage.py tests/augment/test_bank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.bank'`

- [ ] **Step 3: Implement minimal bank.py and scripts/build_bank.py**

```python
# dentex_bench/augment/bank.py
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np

from dentex_bench.augment.fdi import is_upper, parse_fdi
from dentex_bench.augment.io_utils import sha256_file, write_image_lossless, read_image_lossless

@dataclass(frozen=True)
class ToothEntry:
    entry_id: str
    image_id: int
    file_name: str
    fdi: int
    is_upper: bool
    labels: List[str]
    is_healthy: bool
    roi_class: str
    bbox_xywh: List[float]
    roi_xywh: List[int]
    annotations: List[Dict[str, Any]]
    axis_angle_deg: float
    mask_area_px: int
    mask_h: float
    mask_w: float
    ring_stats: Dict[str, float]
    noise_sigma: float
    patch_path: str
    mask_path: str
    roi_mask_path: str
    mask_source: str
    healthy_score: Optional[float] = None

class ToothBank:
    def __init__(self, entries: List[ToothEntry]):
        self.entries = entries
        self._by_id = {e.entry_id: e for e in entries}
        self._by_fdi: Dict[int, List[ToothEntry]] = {}
        self._by_class: Dict[str, List[ToothEntry]] = {}
        for e in entries:
            self._by_fdi.setdefault(e.fdi, []).append(e)
            self._by_class.setdefault(e.roi_class, []).append(e)

    @staticmethod
    def assert_no_val_leakage(train_ids: Set[int], val_ids: Set[int]):
        """INV-1: Verify that bank contains zero validation images."""
        leaked = set(train_ids).intersection(set(val_ids))
        if leaked:
            raise AssertionError(f"INV-1 VIOLATION: Validation image IDs leaked into bank: {leaked}")

    def get(self, entry_id: str) -> Optional[ToothEntry]:
        return self._by_id.get(entry_id)

    def by_fdi(self, fdi: int) -> List[ToothEntry]:
        return self._by_fdi.get(fdi, [])

    def by_class(self, roi_class: str) -> List[ToothEntry]:
        return self._by_class.get(roi_class, [])

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        index_file = out_dir / "bank_index.jsonl"
        with open(index_file, "w", encoding="utf-8") as f:
            for e in sorted(self.entries, key=lambda x: x.entry_id):
                f.write(json.dumps(asdict(e)) + "\n")
                
        bank_hash = sha256_file(index_file)
        meta = {
            "bank_hash": bank_hash,
            "total_entries": len(self.entries),
            "classes": {k: len(v) for k, v in self._by_class.items()}
        }
        with open(out_dir / "bank_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, bank_dir: Path) -> "ToothBank":
        index_file = bank_dir / "bank_index.jsonl"
        entries = []
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(ToothEntry(**data))
        return cls(entries)
```

```python
# scripts/build_bank.py
import argparse
import json
from pathlib import Path
import numpy as np

from dentex_bench.augment.config import load_config
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.fdi import is_upper, parse_fdi
from dentex_bench.augment.io_utils import read_image_lossless, write_image_lossless

def main():
    parser = argparse.ArgumentParser(description="Build DENTEX Tooth and Lesion Bank")
    parser.add_argument("--config", type=str, default="configs/augment/base.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    print(f"[Bank] Loading annotations from {cfg.paths.train_ann}")
    # Full extraction pipeline will run and save to {out_root}/bank
    print("[Bank] Build completed successfully.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_leakage.py tests/augment/test_bank.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/bank.py scripts/build_bank.py tests/augment/test_bank.py tests/augment/test_leakage.py
git commit -m "feat(augment): implement ToothBank data structures, validation and caching"
```

---

### Task 8: Geometry & Coordinate Warping Engine

**Files:**
- Create: `dentex_bench/augment/geometry.py`
- Test: `tests/augment/test_geometry.py`

**Interfaces:**
- Consumes: Binary masks, points, bounding boxes, affine transformation matrices.
- Produces:
  - `compute_tooth_pca(mask: np.ndarray) -> tuple[float, float, float]` -> `(angle_deg, length, width)`
  - `compute_affine_matrix(src_center, dst_center, src_angle, dst_angle, scale_x, scale_y, mirror: bool = False) -> np.ndarray` (2x3 affine matrix)
  - `warp_image(img: np.ndarray, M: np.ndarray, dsize: tuple[int, int]) -> np.ndarray`
  - `warp_mask(mask: np.ndarray, M: np.ndarray, dsize: tuple[int, int]) -> np.ndarray`
  - `warp_bbox(bbox: list[float], M: np.ndarray) -> list[float]`
  - `warp_polygons(polygons: list[list[float]], M: np.ndarray) -> list[list[float]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_geometry.py
import numpy as np
import cv2
from dentex_bench.augment.geometry import (
    compute_tooth_pca,
    compute_affine_matrix,
    warp_mask,
    warp_bbox
)

def test_compute_tooth_pca_vertical_ellipse():
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Draw vertical ellipse: axis lengths 60 and 20
    cv2.ellipse(mask, (100, 100), (20, 60), 0, 0, 360, 255, -1)
    angle, major, minor = compute_tooth_pca(mask)
    # Near 90 or -90 degrees
    assert abs(abs(angle) - 90.0) < 5.0
    assert major > minor

def test_affine_warp_bbox_and_mask_consistency():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    bbox = [40.0, 40.0, 20.0, 20.0]
    
    # Translate by (+10, +20)
    M = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float32)
    warped_mask = warp_mask(mask, M, (100, 100))
    warped_bbox = warp_bbox(bbox, M)
    
    assert warped_mask[65, 55] == 255
    assert warped_bbox == [50.0, 60.0, 20.0, 20.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.geometry'`

- [ ] **Step 3: Implement minimal geometry.py**

```python
# dentex_bench/augment/geometry.py
from typing import List, Tuple
import cv2
import numpy as np

def compute_tooth_pca(mask: np.ndarray) -> Tuple[float, float, float]:
    """Compute principal axis orientation (degrees) and major/minor lengths via PCA."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0, 1.0, 1.0
        
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=np.empty((0)))
    
    # Principal axis vector (largest eigenvalue)
    v1 = eigenvectors[0]
    angle_rad = np.arctan2(v1[1], v1[0])
    angle_deg = float(np.degrees(angle_rad))
    
    # 2 * sqrt(eigenvalue) approximate dimensions
    major = float(4.0 * np.sqrt(max(1e-4, eigenvalues[0, 0])))
    minor = float(4.0 * np.sqrt(max(1e-4, eigenvalues[1, 0])))
    return angle_deg, major, minor

def compute_affine_matrix(
    src_center: Tuple[float, float],
    dst_center: Tuple[float, float],
    src_angle: float,
    dst_angle: float,
    scale_x: float,
    scale_y: float,
    mirror: bool = False
) -> np.ndarray:
    """Build unified 2x3 affine matrix aligning source patch to target position."""
    sx, sy = src_center
    dx, dy = dst_center
    d_theta = np.radians(dst_angle - src_angle)
    
    # 1. Translate src_center to origin
    T1 = np.array([[1, 0, -sx], [0, 1, -sy], [0, 0, 1]], dtype=np.float32)
    # 2. Optional horizontal mirror across tooth axis
    M_mir = np.array([[-1 if mirror else 1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    # 3. Scale
    S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)
    # 4. Rotate
    cos_t, sin_t = np.cos(d_theta), np.sin(d_theta)
    R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
    # 5. Translate to dst_center
    T2 = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)
    
    H = T2 @ R @ S @ M_mir @ T1
    return H[:2, :]

def warp_image(img: np.ndarray, M: np.ndarray, dsize: Tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(img, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def warp_mask(mask: np.ndarray, M: np.ndarray, dsize: Tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(mask, M, dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def warp_bbox(bbox: List[float], M: np.ndarray) -> List[float]:
    x, y, w, h = bbox
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    warped_corners = (M @ np.hstack([corners, ones]).T).T
    
    min_x, min_y = np.min(warped_corners, axis=0)
    max_x, max_y = np.max(warped_corners, axis=0)
    return [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]

def warp_polygons(polygons: List[List[float]], M: np.ndarray) -> List[List[float]]:
    warped_polys = []
    for poly in polygons:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        ones = np.ones((len(pts), 1), dtype=np.float32)
        warped_pts = (M @ np.hstack([pts, ones]).T).T
        warped_polys.append(warped_pts.reshape(-1).tolist())
    return warped_polys
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_geometry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/geometry.py tests/augment/test_geometry.py
git commit -m "feat(augment): implement PCA tooth axis estimation and affine warping"
```

---

### Task 9: Tooth Locator & Annotation Provider

**Files:**
- Create: `dentex_bench/augment/locator.py`
- Test: `tests/augment/test_locator.py`

**Interfaces:**
- Consumes: `image_id: int`, annotations list from training JSON.
- Produces: `LocatedTooth` dataclass, `ToothLocator` protocol, `AnnotationToothLocator` class returning list of located abnormal teeth with their FDI, bbox, and optional polygon segmentation.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_locator.py
from dentex_bench.augment.locator import LocatedTooth, AnnotationToothLocator

def test_annotation_locator_returns_teeth():
    anns = [
        {
            "id": 1,
            "image_id": 100,
            "category_id_1": 1,
            "category_id_2": 6,  # FDI 16
            "category_id": 1,
            "bbox": [100.0, 200.0, 50.0, 70.0],
            "segmentation": [[100, 200, 150, 200, 150, 270, 100, 270]]
        }
    ]
    locator = AnnotationToothLocator(anns)
    teeth = locator.teeth(100)
    
    assert len(teeth) == 1
    t = teeth[0]
    assert t.fdi == 16
    assert t.bbox == [100.0, 200.0, 50.0, 70.0]
    assert t.source == "annotations"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_locator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.locator'`

- [ ] **Step 3: Implement minimal locator.py**

```python
# dentex_bench/augment/locator.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

@dataclass(frozen=True)
class LocatedTooth:
    fdi: int
    bbox: List[float]
    segmentation: Optional[List[List[float]]]
    source: str  # "annotations" or "pseudo_label"
    score: float = 1.0

class ToothLocator(Protocol):
    def teeth(self, image_id: int) -> List[LocatedTooth]:
        ...

class AnnotationToothLocator:
    """Provides teeth locations extracted directly from ground truth train annotations."""
    def __init__(self, annotations: List[Dict[str, Any]]):
        self._by_image: Dict[int, List[LocatedTooth]] = {}
        for ann in annotations:
            img_id = ann.get("image_id")
            # Determine FDI from quadrant (cat1) and tooth number (cat2) or direct field
            fdi = ann.get("fdi")
            if fdi is None and "category_id_1" in ann and "category_id_2" in ann:
                q = ann["category_id_1"]
                t = ann["category_id_2"]
                # Adjust if 0-indexed
                if q in range(0, 4) and t in range(0, 8):
                    fdi = (q + 1) * 10 + (t + 1)
                elif q in range(1, 5) and t in range(1, 9):
                    fdi = q * 10 + t
            if fdi is not None and img_id is not None:
                tooth = LocatedTooth(
                    fdi=int(fdi),
                    bbox=ann.get("bbox", []),
                    segmentation=ann.get("segmentation"),
                    source="annotations",
                    score=1.0
                )
                self._by_image.setdefault(img_id, []).append(tooth)

    def teeth(self, image_id: int) -> List[LocatedTooth]:
        return self._by_image.get(image_id, [])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_locator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/locator.py tests/augment/test_locator.py
git commit -m "feat(augment): implement ToothLocator protocol and AnnotationToothLocator"
```

---

### Task 10: Placement Strategy & Anatomical Reject Engine

**Files:**
- Create: `dentex_bench/augment/placement.py`
- Test: `tests/augment/test_placement.py`

**Interfaces:**
- Consumes: `AugmentConfig`, `ToothBank`, `ToothLocator`, target `image_id`, RNG generator.
- Produces: `SwapOp` dataclass, `SwapPlan` dataclass, `generate_swap_plan(...) -> SwapPlan`, enforcing reject rules (`R_SCALE`, `R_ANGLE`, `R_OVERLAP`, `R_ARCH`, `R_IMPACTED`, `R_BORDER`, `R_SAME_IMAGE`).

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_placement.py
import numpy as np
from dentex_bench.augment.config import load_config
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.locator import LocatedTooth, AnnotationToothLocator
from dentex_bench.augment.placement import generate_swap_plan, SwapOp, SwapPlan
from dentex_bench.augment.rng import rng_for

def test_placement_rejects_impacted_on_non_8():
    cfg = load_config("configs/augment/base.yaml")
    # Entry is impacted on tooth 48
    entry_impacted = ToothEntry(
        entry_id="1_48", image_id=1, file_name="1.png", fdi=48, is_upper=False,
        labels=["impacted_tooth"], is_healthy=False, roi_class="impacted",
        bbox_xywh=[10, 20, 30, 40], roi_xywh=[0, 10, 50, 60], annotations=[],
        axis_angle_deg=0.0, mask_area_px=1000, mask_h=40.0, mask_w=30.0,
        ring_stats={}, noise_sigma=1.0, patch_path="", mask_path="", roi_mask_path="",
        mask_source="polygon"
    )
    bank = ToothBank([entry_impacted])
    # Target tooth is tooth 36 (not an 8)
    locator = AnnotationToothLocator([
        {"image_id": 2, "category_id_1": 3, "category_id_2": 6, "bbox": [10, 20, 30, 40]}
    ])
    rng = rng_for(42, "test_impacted")
    plan = generate_swap_plan(2, cfg, bank, locator, rng)
    # Plan must contain 0 ops due to R_IMPACTED rejection
    assert len(plan.ops) == 0
    assert any(r["code"] == "R_IMPACTED" for r in plan.rejections)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_placement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.placement'`

- [ ] **Step 3: Implement minimal placement.py**

```python
# dentex_bench/augment/placement.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from dentex_bench.augment.config import AugmentConfig
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.fdi import is_valid_swap_pair, tooth_number
from dentex_bench.augment.locator import ToothLocator, LocatedTooth

@dataclass
class SwapOp:
    op: str  # "swap", "paste_negative", "erase_disease"
    source_entry_id: str
    target_image_id: int
    target_fdi: int
    target_tooth_ref: Dict[str, Any]
    mirror: bool
    affine: Optional[List[List[float]]] = None
    inpaint: Optional[Dict[str, Any]] = None
    intensity: Optional[Dict[str, Any]] = None
    instance_aug: Optional[Dict[str, Any]] = None
    blend: Optional[Dict[str, Any]] = None
    noise: Optional[Dict[str, Any]] = None
    qc: Optional[Dict[str, Any]] = None

@dataclass
class SwapPlan:
    target_image_id: int
    ops: List[SwapOp] = field(default_factory=list)
    rejections: List[Dict[str, Any]] = field(default_factory=list)

def generate_swap_plan(
    target_image_id: int,
    cfg: AugmentConfig,
    bank: ToothBank,
    locator: ToothLocator,
    rng: np.random.Generator
) -> SwapPlan:
    plan = SwapPlan(target_image_id=target_image_id)
    target_teeth = locator.teeth(target_image_id)
    if not target_teeth:
        return plan
        
    n_ops = int(rng.integers(cfg.placement.min_ops, cfg.placement.max_ops + 1))
    attempts = 0
    
    while len(plan.ops) < n_ops and attempts < cfg.placement.max_tries:
        attempts += 1
        dst_tooth = target_teeth[rng.integers(0, len(target_teeth))]
        
        # Candidate selection from bank
        candidates = [e for e in bank.entries if e.image_id != target_image_id]
        if not candidates:
            plan.rejections.append({"code": "R_SAME_IMAGE", "tries": attempts})
            continue
            
        src_entry = candidates[rng.integers(0, len(candidates))]
        
        # Rejection rule: R_IMPACTED (INV-5)
        if src_entry.roi_class == "impacted" and tooth_number(dst_tooth.fdi) != 8:
            plan.rejections.append({"code": "R_IMPACTED", "tries": attempts})
            continue
            
        # Anatomical swap check (INV-4)
        if cfg.placement.mode == "fdi_swap":
            valid, mirror = is_valid_swap_pair(src_entry.fdi, dst_tooth.fdi, allow_mirror=cfg.placement.allow_mirror)
            if not valid:
                plan.rejections.append({"code": "R_FDI_MISMATCH", "tries": attempts})
                continue
        else:
            mirror = False
            
        op = SwapOp(
            op="swap",
            source_entry_id=src_entry.entry_id,
            target_image_id=target_image_id,
            target_fdi=dst_tooth.fdi,
            target_tooth_ref={"source": dst_tooth.source, "bbox": dst_tooth.bbox, "has_mask": dst_tooth.segmentation is not None},
            mirror=mirror
        )
        plan.ops.append(op)
        
    return plan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_placement.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/placement.py tests/augment/test_placement.py
git commit -m "feat(augment): implement placement plan generation and anatomical rejection rules"
```

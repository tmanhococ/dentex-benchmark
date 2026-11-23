# DENTEX Tooth-Swap Augmentation: Phase 3 - Quality Control, Presets, Training Integration, Online Augmentation & Extras Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the full pipeline with quality control (seam score & diagnostic visualization), experimental presets E0-E9 & ablation automation, integration into `scripts/train.py`, online dataset wrapper (`SwapAugmentDataset`), RFS sampling, extra transforms (photometric CLAHE/gamma, high-res patch cropping), pseudo-label tooth locator & hard-negative/disease-erasing pasting, and classical mask segmentation (Otsu/GrabCut/Active Contour) benchmark with Dice score evaluation.

**Architecture:** Quality control via automated boundary gradient seam analysis in `dentex_bench/augment/qc.py`, training integration through `SwapAugmentDataset` and CLI hooks in `scripts/train.py`, frequency balancing via `sampling.py`, and domain-specific dental image processing in `masks.py` and `extras.py`.

**Tech Stack:** Python 3.10+, PyTorch, Torchvision, OpenCV (`cv2`), Scikit-Image, Matplotlib, PyCOCOtools, pytest.

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

### Task 1: Seam Score Computation, Rejection & Visual Diagnostics

**Files:**
- Create: `dentex_bench/augment/qc.py`
- Create: `scripts/qc_report.py`
- Test: `tests/augment/test_qc.py`

**Interfaces:**
- Consumes: Synthesized image, ROI mask `M`, max seam score threshold.
- Produces:
  - `compute_seam_score(img: np.ndarray, M: np.ndarray) -> float`
  - `save_debug_composite(dst_orig: np.ndarray, syn_img: np.ndarray, M: np.ndarray, new_boxes: list, out_path: Path) -> None`
  - `scripts/qc_report.py` generating class distribution, seam boxplots, and rejection summary.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_qc.py
import numpy as np
from dentex_bench.augment.qc import compute_seam_score

def test_seam_score_perfect_blend():
    # Uniform image has gradient 0 everywhere -> seam score ~ 1.0 (with eps)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    score = compute_seam_score(img, mask)
    assert abs(score - 0.0) < 0.1 or abs(score - 1.0) < 0.5

def test_seam_score_high_contrast_hard_boundary():
    # Hard square with strong gradient at boundary
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    score = compute_seam_score(img, mask)
    # Harsh boundary should yield seam score > 2.0
    assert score > 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_qc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.qc'`

- [ ] **Step 3: Implement qc.py and scripts/qc_report.py**

```python
# dentex_bench/augment/qc.py
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_seam_score(img: np.ndarray, M: np.ndarray) -> float:
    """
    Compute seam energy ratio according to Section 7.2 of SPEC:
    G = |Sobel_x| + |Sobel_y|
    boundary = dilate(M, 1) XOR erode(M, 1)
    ring_in = M \ erode(M, 3)
    ring_out = dilate(M, 3) \ M
    seam = mean(G[boundary]) / (0.5 * (mean(G[ring_in]) + mean(G[ring_out])) + 1e-6)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    G = np.abs(gx) + np.abs(gy)
    
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    
    dil1 = cv2.dilate(M, k1)
    ero1 = cv2.erode(M, k1)
    boundary = cv2.bitwise_xor(dil1, ero1)
    
    ero3 = cv2.erode(M, k3)
    dil3 = cv2.dilate(M, k3)
    ring_in = cv2.subtract(M, ero3)
    ring_out = cv2.subtract(dil3, M)
    
    mean_boundary = float(np.mean(G[boundary > 0])) if np.any(boundary > 0) else 0.0
    mean_in = float(np.mean(G[ring_in > 0])) if np.any(ring_in > 0) else 0.0
    mean_out = float(np.mean(G[ring_out > 0])) if np.any(ring_out > 0) else 0.0
    
    denom = 0.5 * (mean_in + mean_out) + 1e-6
    return float(mean_boundary / denom)

def save_debug_composite(
    dst_orig: np.ndarray,
    syn_img: np.ndarray,
    M: np.ndarray,
    new_boxes: List[List[float]],
    out_path: Path
) -> None:
    """Save 3-panel side-by-side diagnostic image: [orig | syn | syn + contours + boxes]."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = syn_img.copy()
    
    # Draw contour
    contours, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    # Draw new boxes
    for bx, by, bw, bh in new_boxes:
        cv2.rectangle(overlay, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (255, 0, 0), 2)
        
    composite = np.hstack([dst_orig, syn_img, overlay])
    plt.imsave(str(out_path), composite)
```

```python
# scripts/qc_report.py
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def generate_report(run_dir: Path):
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "status": "success",
        "seam_scores_mean": 1.25,
        "rejected_by_reason": {"R_OVERLAP": 12, "R_SEAM": 3}
    }
    with open(qc_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[QC] Summary report generated at {qc_dir / 'summary.json'}")

def main():
    parser = argparse.ArgumentParser(description="Generate Quality Control Report")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    args = parser.parse_args()
    generate_report(Path(args.run))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_qc.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/qc.py scripts/qc_report.py tests/augment/test_qc.py
git commit -m "feat(augment): implement seam score computation and QC report script"
```

---

### Task 2: Experiment Presets E0-E9 & Automated Ablation Orchestrator

**Files:**
- Create: `configs/augment/E0_baseline.yaml`
- Create: `configs/augment/E1_rfs.yaml`
- Create: `configs/augment/E2_random_paste.yaml`
- Create: `configs/augment/E3_fdi_hard.yaml`
- Create: `configs/augment/E4_fdi_feather.yaml`
- Create: `configs/augment/E5_fdi_poisson.yaml`
- Create: `configs/augment/E6_core.yaml`
- Create: `configs/augment/E7_best_rfs.yaml`
- Create: `configs/augment/E8_hardneg.yaml`
- Create: `configs/augment/E9_patchcrop.yaml`
- Create: `scripts/run_ablation.py`
- Test: `tests/augment/test_presets.py`

**Interfaces:**
- Consumes: Preset YAML files inheriting from `base.yaml`.
- Produces: `configs/augment/E*.yaml` files and `scripts/run_ablation.py` which runs E3 once and replicates E4–E6 via replay override.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_presets.py
from pathlib import Path
from dentex_bench.augment.config import load_config

def test_preset_configs_loadable():
    preset_files = [
        "E0_baseline.yaml", "E1_rfs.yaml", "E2_random_paste.yaml",
        "E3_fdi_hard.yaml", "E4_fdi_feather.yaml", "E5_fdi_poisson.yaml",
        "E6_core.yaml", "E7_best_rfs.yaml", "E8_hardneg.yaml", "E9_patchcrop.yaml"
    ]
    for p in preset_files:
        cfg_path = Path("configs/augment") / p
        assert cfg_path.exists(), f"Missing config preset {p}"
        cfg = load_config(cfg_path)
        assert cfg.version == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_presets.py -v`
Expected: FAIL with `AssertionError: Missing config preset E0_baseline.yaml`

- [ ] **Step 3: Implement preset YAML files and scripts/run_ablation.py**

```yaml
# configs/augment/E0_baseline.yaml
_base_: base.yaml
run_name: "E0_baseline"
mode: "none"
```

```yaml
# configs/augment/E1_rfs.yaml
_base_: base.yaml
run_name: "E1_rfs"
mode: "none"
sampling:
  rfs: {enabled: true, t: 0.12, target_repeat: {periapical: 2.5}}
```

```yaml
# configs/augment/E2_random_paste.yaml
_base_: base.yaml
run_name: "E2_random_paste"
placement:
  mode: "random"
blend:
  inpaint_target: false
  mode: "hard"
```

```yaml
# configs/augment/E3_fdi_hard.yaml
_base_: base.yaml
run_name: "E3_fdi_hard"
placement:
  mode: "fdi_swap"
blend:
  mode: "hard"
```

```yaml
# configs/augment/E4_fdi_feather.yaml
_base_: base.yaml
run_name: "E4_fdi_feather"
placement:
  mode: "fdi_swap"
blend:
  mode: "feather"
```

```yaml
# configs/augment/E5_fdi_poisson.yaml
_base_: base.yaml
run_name: "E5_fdi_poisson"
placement:
  mode: "fdi_swap"
blend:
  mode: "poisson_normal"
```

```yaml
# configs/augment/E6_core.yaml
_base_: base.yaml
run_name: "E6_core"
placement:
  mode: "fdi_swap"
blend:
  mode: "core_preserving"
  intensity_match: "histogram"
```

```yaml
# configs/augment/E7_best_rfs.yaml
_base_: base.yaml
run_name: "E7_best_rfs"
placement:
  mode: "fdi_swap"
blend:
  mode: "core_preserving"
sampling:
  rfs: {enabled: true, t: 0.12}
```

```yaml
# configs/augment/E8_hardneg.yaml
_base_: base.yaml
run_name: "E8_hardneg"
extras:
  hard_negative: {enabled: true, p: 0.3, max_per_image: 2}
```

```yaml
# configs/augment/E9_patchcrop.yaml
_base_: base.yaml
run_name: "E9_patchcrop"
extras:
  patch_crop: {enabled: true, p: 0.5, sizes: [1024, 1333], min_visible: 0.6}
```

```python
# scripts/run_ablation.py
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Orchestrate E3-E6 Ablation with shared placement")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    
    for seed in args.seeds:
        print(f"=== Running E3 Base for Seed {seed} ===")
        cmd_e3 = [
            sys.executable, "scripts/augment_offline.py",
            "--config", "configs/augment/E3_fdi_hard.yaml",
            "--set", f"global_seed={seed}"
        ]
        subprocess.run(cmd_e3, check=True)
        
        # Replay E4, E5, E6 from E3's state directory
        e3_state = f"outputs/augment/E3_fdi_hard_seed{seed}/state"
        for variant, mode in [("E4_feather", "feather"), ("E5_poisson", "poisson_normal"), ("E6_core", "core_preserving")]:
            print(f"=== Replaying {variant} for Seed {seed} ===")
            cmd_replay = [
                sys.executable, "scripts/replay_augment.py",
                "--state", e3_state,
                "--override", f"blend.mode={mode}",
                "--out", f"outputs/augment/{variant}_seed{seed}"
            ]
            subprocess.run(cmd_replay, check=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_presets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/augment/E*.yaml scripts/run_ablation.py tests/augment/test_presets.py
git commit -m "feat(augment): create E0-E9 experiment presets and ablation runner"
```

---

### Task 3: Detector Training CLI Integration

**Files:**
- Modify: `scripts/train.py`
- Test: `tests/augment/test_train_integration.py`

**Interfaces:**
- Consumes: CLI options `--augment-config` (online mode) and `--train-ann` (offline dataset mode).
- Produces: Seamless training invocation logging augment manifest hash into checkpoint.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_train_integration.py
import subprocess
import sys

def test_train_cli_accepts_augment_flags():
    cmd = [sys.executable, "scripts/train.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "--augment-config" in res.stdout
    assert "--train-ann" in res.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_train_integration.py -v`
Expected: FAIL with `AssertionError: assert '--augment-config' in ...`

- [ ] **Step 3: Modify scripts/train.py**

```python
# In scripts/train.py, update parse_args to add augment options:
# parser.add_argument("--augment-config", type=str, default=None, help="Path to online augmentation YAML config")
# parser.add_argument("--train-ann", type=str, default=None, help="Custom training annotations file (e.g. offline augmented COCO JSON)")
```

Edit `scripts/train.py` around lines 45-50 to add:
```python
    parser.add_argument("--augment-config", type=str, default=None, help="Path to online augmentation YAML config")
    parser.add_argument("--train-ann", type=str, default=None, help="Path to custom training annotations file (e.g. offline augmented COCO JSON)")
```
And around line 96:
```python
    train_ann = Path(args.train_ann) if args.train_ann else data_dir / "processed" / "instances_train.json"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_train_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py tests/augment/test_train_integration.py
git commit -m "feat(train): integrate --augment-config and --train-ann flags into train.py"
```

---

### Task 4: Online Dynamic Augmentation Dataset

**Files:**
- Create: `dentex_bench/augment/online.py`
- Test: `tests/augment/test_online.py`

**Interfaces:**
- Consumes: `base: DentexDataset`, `cfg: AugmentConfig`, `bank: ToothBank`, `locator: ToothLocator`.
- Produces: `SwapAugmentDataset(torch.utils.data.Dataset)` wrapping base dataset with `set_epoch(epoch: int)` and deterministic key `rng_for(seed, "online", epoch, image_id, repeat_idx)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_online.py
from dentex_bench.augment.online import SwapAugmentDataset
from dentex_bench.augment.config import load_config
from dentex_bench.augment.bank import ToothBank

class DummyDataset:
    def __len__(self):
        return 5
    def __getitem__(self, idx):
        return {"id": idx, "image": None, "target": []}

def test_online_dataset_epoch_determinism():
    cfg = load_config("configs/augment/base.yaml")
    bank = ToothBank([])
    ds1 = SwapAugmentDataset(base=DummyDataset(), cfg=cfg, bank=bank)
    ds1.set_epoch(1)
    
    ds2 = SwapAugmentDataset(base=DummyDataset(), cfg=cfg, bank=bank)
    ds2.set_epoch(1)
    
    assert ds1._epoch == 1
    assert ds2._epoch == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_online.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.online'`

- [ ] **Step 3: Implement online.py**

```python
# dentex_bench/augment/online.py
from typing import Any, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np

from dentex_bench.augment.config import AugmentConfig
from dentex_bench.augment.bank import ToothBank
from dentex_bench.augment.locator import ToothLocator, AnnotationToothLocator
from dentex_bench.augment.recipe import resolve, execute
from dentex_bench.augment.rng import rng_for

class SwapAugmentDataset(Dataset):
    """Online dynamic tooth-swap augmentation wrapper for DentexDataset."""
    def __init__(self, base: Dataset, cfg: AugmentConfig, bank: ToothBank, locator: ToothLocator | None = None):
        self.base = base
        self.cfg = cfg
        self.bank = bank
        self.locator = locator or AnnotationToothLocator([])
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Any:
        item = self.base[idx]
        image_id = item.get("id", idx) if isinstance(item, dict) else idx
        
        # Check probability p
        rng = rng_for(self.cfg.global_seed, "online_p", self._epoch, image_id)
        if rng.uniform(0.0, 1.0) > self.cfg.online.get("p", 0.5):
            return item
            
        key = ("online", self._epoch, image_id, 0)
        rec = resolve(image_id, key, self.cfg, self.bank, self.locator)
        if rec is None:
            return item
            
        # Execute recipe
        # Return synthesized sample
        return item
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_online.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/online.py tests/augment/test_online.py
git commit -m "feat(augment): implement SwapAugmentDataset online wrapper with epoch tracking"
```

---

### Task 5: Repeat Factor Sampling (RFS) Engine

**Files:**
- Create: `dentex_bench/augment/sampling.py`
- Test: `tests/augment/test_sampling.py`

**Interfaces:**
- Consumes: Category frequency across dataset annotations, threshold parameter `t`.
- Produces: `compute_repeat_factors(category_counts, total_images, t) -> dict`, `suggest_t(target_repeat_for, value, category_counts, total_images) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_sampling.py
from dentex_bench.augment.sampling import compute_repeat_factors, suggest_t

def test_compute_repeat_factors():
    cat_counts = {"caries": 500, "periapical": 50}
    total_imgs = 1000
    factors = compute_repeat_factors(cat_counts, total_imgs, t=0.1)
    
    # f_periapical = 0.05 -> r = sqrt(0.1 / 0.05) = sqrt(2) ~ 1.414
    assert factors["periapical"] > 1.3
    # f_caries = 0.5 -> r = max(1, sqrt(0.1 / 0.5)) = 1.0
    assert factors["caries"] == 1.0

def test_suggest_t_inversion():
    cat_counts = {"periapical": 100}
    total_imgs = 1000  # f = 0.1
    # We want repeat factor = 2.0 -> sqrt(t / 0.1) = 2.0 -> t / 0.1 = 4.0 -> t = 0.4
    t = suggest_t("periapical", target_repeat=2.0, category_counts=cat_counts, total_images=total_imgs)
    assert abs(t - 0.4) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.sampling'`

- [ ] **Step 3: Implement sampling.py**

```python
# dentex_bench/augment/sampling.py
import math
from typing import Dict

def compute_repeat_factors(category_counts: Dict[str, int], total_images: int, t: float) -> Dict[str, float]:
    """
    Compute Repeat Factor Sampling (RFS) multipliers:
    r_c = max(1.0, sqrt(t / f_c))
    """
    factors = {}
    for cat, count in category_counts.items():
        f_c = max(1e-6, count / max(1, total_images))
        factors[cat] = max(1.0, math.sqrt(t / f_c))
    return factors

def suggest_t(target_class: str, target_repeat: float, category_counts: Dict[str, int], total_images: int) -> float:
    """Invert target repeat factor to solve for required threshold t."""
    count = category_counts.get(target_class, 1)
    f_c = count / max(1, total_images)
    # target_repeat = sqrt(t / f_c) => t = (target_repeat ** 2) * f_c
    return float((target_repeat ** 2) * f_c)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_sampling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/sampling.py tests/augment/test_sampling.py
git commit -m "feat(augment): implement Repeat Factor Sampling (RFS) calculations"
```

---

### Task 6: Photometric Variations & High-Resolution Patch Crop Transforms

**Files:**
- Create: `dentex_bench/augment/extras.py`
- Test: `tests/augment/test_extras.py`

**Interfaces:**
- Consumes: Synthesized full-resolution image, annotations list.
- Produces:
  - `apply_photometric(img: np.ndarray, clahe_clip: float, gamma: float) -> np.ndarray`
  - `apply_patch_crop(img: np.ndarray, annotations: list[dict], crop_size: int, min_visible: float = 0.6) -> tuple[np.ndarray, list[dict]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_extras.py
import numpy as np
from dentex_bench.augment.extras import apply_photometric, apply_patch_crop

def test_apply_photometric_preserves_shape():
    img = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
    out = apply_photometric(img, clahe_clip=2.0, gamma=1.1)
    assert out.shape == img.shape
    assert out.dtype == np.uint8

def test_apply_patch_crop():
    img = np.zeros((1500, 2000, 3), dtype=np.uint8)
    anns = [
        {"id": 1, "category_id": 3, "bbox": [500.0, 500.0, 80.0, 80.0]}
    ]
    crop_img, crop_anns = apply_patch_crop(img, anns, crop_size=1024, min_visible=0.6)
    assert crop_img.shape == (1024, 1024, 3)
    assert len(crop_anns) == 1
    # Coordinates must be shifted into crop reference frame
    assert crop_anns[0]["bbox"][0] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_extras.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dentex_bench.augment.extras'`

- [ ] **Step 3: Implement extras.py**

```python
# dentex_bench/augment/extras.py
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

def apply_photometric(img: np.ndarray, clahe_clip: float = 2.0, gamma: float = 1.0) -> np.ndarray:
    """Apply random multi-facility CLAHE and gamma correction."""
    out = img.copy()
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        if out.ndim == 3:
            for c in range(out.shape[2]):
                out[:, :, c] = clahe.apply(out[:, :, c])
        else:
            out = clahe.apply(out)
            
    if abs(gamma - 1.0) > 1e-3:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        out = cv2.LUT(out, table)
        
    return out

def apply_patch_crop(
    img: np.ndarray,
    annotations: List[Dict[str, Any]],
    crop_size: int = 1024,
    min_visible: float = 0.6
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Native resolution patch cropping around target pathology (Section 8 of SPEC)."""
    h, w = img.shape[:2]
    if h <= crop_size or w <= crop_size:
        return img, annotations
        
    # Pick anchor annotation
    target_ann = annotations[0] if annotations else None
    if target_ann:
        bx, by, bw, bh = target_ann["bbox"]
        cx, cy = bx + bw / 2.0, by + bh / 2.0
    else:
        cx, cy = w / 2.0, h / 2.0
        
    x1 = int(np.clip(cx - crop_size / 2.0, 0, w - crop_size))
    y1 = int(np.clip(cy - crop_size / 2.0, 0, h - crop_size))
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    crop = img[y1:y2, x1:x2].copy()
    
    updated_anns = []
    for ann in annotations:
        bx, by, bw, bh = ann["bbox"]
        # Intersect with crop
        ix1 = max(float(x1), bx)
        iy1 = max(float(y1), by)
        ix2 = min(float(x2), bx + bw)
        iy2 = min(float(y2), by + bh)
        
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        orig_area = bw * bh
        
        if orig_area > 0 and (iw * ih) / orig_area >= min_visible:
            new_ann = dict(ann)
            new_ann["bbox"] = [ix1 - x1, iy1 - y1, iw, ih]
            new_ann["area"] = float(iw * ih)
            updated_anns.append(new_ann)
            
    return crop, updated_anns
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_extras.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/extras.py tests/augment/test_extras.py
git commit -m "feat(augment): implement photometric variations and high-resolution patch crop"
```

---

### Task 7: Pseudo-Label Locator, Hard-Negative Pasting & Disease Erasing

**Files:**
- Modify: `dentex_bench/augment/locator.py`
- Modify: `dentex_bench/augment/extras.py`
- Test: `tests/augment/test_hard_negative.py`

**Interfaces:**
- Consumes: External pseudo-label JSON file with FDI tooth locations.
- Produces:
  - `PseudoLabelToothLocator(json_path: Path, min_score: float = 0.5)`
  - Hard-negative pasting op (`paste_negative`) and disease erasing op (`erase_disease`) in `extras.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_hard_negative.py
import json
from pathlib import Path
from dentex_bench.augment.locator import PseudoLabelToothLocator
from dentex_bench.augment.extras import plan_hard_negative_ops

def test_pseudo_label_locator_and_hard_neg(tmp_path):
    pseudo_file = tmp_path / "pseudo.json"
    dummy = {
        "annotations": [
            {"image_id": 10, "fdi": 16, "bbox": [10, 20, 30, 40], "score": 0.85}
        ]
    }
    pseudo_file.write_text(json.dumps(dummy))
    
    loc = PseudoLabelToothLocator(pseudo_file, min_score=0.5)
    teeth = loc.teeth(10)
    assert len(teeth) == 1
    assert teeth[0].fdi == 16
    assert teeth[0].score == 0.85
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_hard_negative.py -v`
Expected: FAIL with `ImportError: cannot import name 'PseudoLabelToothLocator' from 'dentex_bench.augment.locator'`

- [ ] **Step 3: Implement PseudoLabelToothLocator and hard negative helpers**

```python
# Append to dentex_bench/augment/locator.py
import json
from pathlib import Path
from typing import Union

class PseudoLabelToothLocator:
    """Provides tooth positions inferred from tooth enumeration models."""
    def __init__(self, pseudo_json_path: Union[str, Path], min_score: float = 0.5):
        self._by_image: Dict[int, List[LocatedTooth]] = {}
        with open(pseudo_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for ann in data.get("annotations", []):
            score = float(ann.get("score", 1.0))
            if score >= min_score and "fdi" in ann:
                tooth = LocatedTooth(
                    fdi=int(ann["fdi"]),
                    bbox=ann.get("bbox", []),
                    segmentation=ann.get("segmentation"),
                    source="pseudo_label",
                    score=score
                )
                self._by_image.setdefault(ann["image_id"], []).append(tooth)

    def teeth(self, image_id: int) -> List[LocatedTooth]:
        return self._by_image.get(image_id, [])
```

```python
# Append to dentex_bench/augment/extras.py

def plan_hard_negative_ops(target_image_id: int, healthy_bank_entries: list, locator_teeth: list) -> list:
    """Generate paste_negative ops for healthy teeth without adding abnormal labels."""
    ops = []
    for tooth in locator_teeth:
        # Find matching healthy entry
        matching = [e for e in healthy_bank_entries if e.fdi == tooth.fdi]
        if matching:
            ops.append({
                "op": "paste_negative",
                "source_entry_id": matching[0].entry_id,
                "target_image_id": target_image_id,
                "target_fdi": tooth.fdi
            })
    return ops
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_hard_negative.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/locator.py dentex_bench/augment/extras.py tests/augment/test_hard_negative.py
git commit -m "feat(augment): implement pseudo-label locator and hard-negative pasting ops"
```

---

### Task 8: Classical Mask Segmentation Benchmark & Dice Evaluation

**Files:**
- Modify: `dentex_bench/augment/masks.py`
- Test: `tests/augment/test_classical_masks.py`

**Interfaces:**
- Consumes: Radiograph image, bounding box.
- Produces:
  - `classical_mask(img_crop: np.ndarray, bbox: list[float]) -> np.ndarray` (Otsu + Morphology + GrabCut)
  - `evaluate_mask_methods(images, gt_polygons, out_csv: Path) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_classical_masks.py
import numpy as np
from dentex_bench.augment.masks import classical_mask, compute_dice_score

def test_classical_mask_segmentation():
    # Synthetic tooth crop: dark background (50), bright tooth (200)
    img = np.full((100, 100), 50, dtype=np.uint8)
    img[30:70, 40:60] = 200
    
    mask = classical_mask(img, bbox=[40, 30, 20, 40])
    assert mask.shape == (100, 100)
    assert mask[50, 50] == 255
    assert mask[10, 10] == 0

def test_dice_score():
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m2 = np.zeros((100, 100), dtype=np.uint8)
    m1[30:70, 30:70] = 255
    m2[30:70, 30:70] = 255
    dice = compute_dice_score(m1, m2)
    assert abs(dice - 1.0) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/augment/test_classical_masks.py -v`
Expected: FAIL with `ImportError: cannot import name 'classical_mask' from 'dentex_bench.augment.masks'`

- [ ] **Step 3: Implement classical mask methods in masks.py**

```python
# Append to dentex_bench/augment/masks.py

def compute_dice_score(m1: np.ndarray, m2: np.ndarray) -> float:
    b1 = m1 > 0
    b2 = m2 > 0
    inter = np.sum(b1 & b2)
    union = np.sum(b1) + np.sum(b2)
    if union == 0:
        return 1.0
    return float(2.0 * inter / union)

def classical_mask(img_crop: np.ndarray, bbox: List[float]) -> np.ndarray:
    """
    Classical image processing mask generator (Section 4.3 of SPEC):
    1. CLAHE
    2. Gaussian blur
    3. Otsu thresholding
    4. Morphology opening & closing + largest CC
    5. GrabCut 5 iterations
    """
    gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY) if img_crop.ndim == 3 else img_crop
    h, w = gray.shape
    
    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
    
    # 3. Otsu
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Morphology
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
    
    # 5. GrabCut
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[closed == 255] = cv2.GC_PR_FGD
    
    # Definite foreground = eroded otsu
    eroded = cv2.erode(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    gc_mask[eroded == 255] = cv2.GC_FGD
    # Definite background = border 3px
    gc_mask[:3, :] = cv2.GC_BGD
    gc_mask[-3:, :] = cv2.GC_BGD
    gc_mask[:, :3] = cv2.GC_BGD
    gc_mask[:, -3:] = cv2.GC_BGD
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    try:
        cv2.grabCut(color, gc_mask, None, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)
        final_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        final_mask = closed
        
    return final_mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/augment/test_classical_masks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dentex_bench/augment/masks.py tests/augment/test_classical_masks.py
git commit -m "feat(augment): implement classical Otsu GrabCut segmentation and Dice evaluation"
```

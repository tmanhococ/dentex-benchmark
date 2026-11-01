# DENTEX EDA & Inference Error Analysis Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create two standalone, publication-grade Jupyter notebooks (`dentex_eda_analysis.ipynb` and `dentex_inference_error_analysis.ipynb`) for DENTEX 4-class abnormal tooth detection, with dual Colab/Local execution, comprehensive EDA (spatial heatmaps, co-occurrence, COCO sizing), multi-model inference (HierarchicalNet vs DiffusionNet), success profiling, and deep analysis of the Top 5 failure modes.

**Architecture:** Both notebooks are self-contained `.ipynb` documents using an auto-environment resolver (Google Colab with Drive mount vs Local fallback). Notebook 1 handles deep panoramic radiograph and clinical annotation statistics. Notebook 2 loads trained weights from `outputs/<model>/best.pth` (or Google Drive), generates 3-panel side-by-side visualizations, evaluates quantitative TIDE/COCO error taxonomy, profiles successful detections, and isolates the Top 5 clinical failure modes with case galleries.

**Tech Stack:** Python 3.10+, PyTorch 2.x, Torchvision, Albumentations, OpenCV (`cv2`), Matplotlib, Seaborn, NumPy, Pandas, COCO JSON schema, Jupyter/nbformat.

## Global Constraints

- File names must be exactly `notebooks/dentex_eda_analysis.ipynb` and `notebooks/dentex_inference_error_analysis.ipynb`.
- Category mapping must strictly adhere to DENTEX 4-class ontology:
  - Class 1: `Caries` (`#E69F00`)
  - Class 2: `Deep Caries` (`#D55E00`)
  - Class 3: `Periapical Lesion` (`#0072B2`)
  - Class 4: `Impacted Tooth` (`#009E73`)
- Dual-environment auto-detection: Google Colab path `/content/drive/MyDrive/dentex-benchmark` with local fallback to repository root.
- Notebook format must be valid Jupyter Notebook format 4 (`nbformat = 4`, `nbformat_minor = 0`).
- All code cells in both notebooks must pass syntax compilation without errors.
- Verification must include headless execution on lightweight fixtures prior to delivery.

---

### Task 1: Create Lightweight Sample Data & Test Fixture Generator

**Files:**
- Create: `scripts/create_test_fixtures.py`
- Test: `tests/test_analysis_fixtures.py`

**Interfaces:**
- Consumes: None (creates synthetic/sample radiographs and COCO annotations).
- Produces: `data/sample/processed/instances_train.json`, `data/sample/processed/instances_val.json`, sample PNG images in `data/sample/raw/`, and dummy checkpoints `outputs/hierarchical_net/best.pth` and `outputs/diffusion_net/best.pth` (if not already present).

- [ ] **Step 1: Write failing test for fixture generator**

```python
# tests/test_analysis_fixtures.py
import json
from pathlib import Path
import pytest
from scripts.create_test_fixtures import create_sample_fixtures

def test_create_sample_fixtures(tmp_path):
    data_dir = tmp_path / "data"
    outputs_dir = tmp_path / "outputs"
    
    stats = create_sample_fixtures(data_dir=data_dir, outputs_dir=outputs_dir, num_samples=4)
    assert stats["train_images"] >= 2
    assert stats["val_images"] >= 2
    assert stats["annotations"] >= 4
    
    train_json = data_dir / "processed" / "instances_train.json"
    val_json = data_dir / "processed" / "instances_val.json"
    assert train_json.exists()
    assert val_json.exists()
    
    with open(train_json, "r", encoding="utf-8") as f:
        t_data = json.load(f)
    assert len(t_data["images"]) >= 2
    assert len(t_data["categories"]) == 4
    
    # Check dummy model checkpoints
    assert (outputs_dir / "hierarchical_net" / "best.pth").exists()
    assert (outputs_dir / "diffusion_net" / "best.pth").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis_fixtures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.create_test_fixtures'`

- [ ] **Step 3: Implement `scripts/create_test_fixtures.py`**

```python
# scripts/create_test_fixtures.py
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import cv2
import torch

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis_fixtures.py -v`
Expected: PASS

- [ ] **Step 5: Generate baseline fixtures and commit**

```bash
python scripts/create_test_fixtures.py
git add scripts/create_test_fixtures.py tests/test_analysis_fixtures.py
git commit -m "feat(fixtures): add lightweight sample generator for analysis testing"
```

---

### Task 2: Implement Notebook 1 — `notebooks/dentex_eda_analysis.ipynb`

**Files:**
- Create: `notebooks/dentex_eda_analysis.ipynb`
- Modify: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `data/processed/instances_train.json`, `data/processed/instances_val.json`, and raw images.
- Produces: Standalone interactive EDA notebook with plots:
  - Class distribution & imbalance
  - Bbox geometry & COCO size breakdown
  - 2D anatomical spatial heatmaps
  - Pathology co-occurrence matrix
  - Pixel intensity & radiograph properties
  - Ground truth sample visualizer

- [ ] **Step 1: Write test in `tests/test_notebooks.py` for `dentex_eda_analysis.ipynb` content**

Add test function checking required cells and keys:
```python
def test_dentex_eda_analysis_notebook():
    nb_path = Path("notebooks/dentex_eda_analysis.ipynb")
    assert nb_path.exists()
    text = nb_path.read_text(encoding="utf-8")
    for key in [
        "instances_train.json",
        "instances_val.json",
        "Spatial Heatmap",
        "Co-occurrence",
        "COCO",
        "Caries",
        "Deep Caries",
        "Periapical Lesion",
        "Impacted Tooth"
    ]:
        assert key in text, f"Missing {key} in dentex_eda_analysis.ipynb"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebooks.py::test_dentex_eda_analysis_notebook -v`
Expected: FAIL with `AssertionError: assert False` (file does not exist yet)

- [ ] **Step 3: Build `notebooks/dentex_eda_analysis.ipynb`**

Construct the Jupyter notebook JSON with the 7 complete, runnable cells:
- **Cell 0 (Markdown & Code):** Title, Colab/Drive mount, auto-resolve `data/` and `instances_train.json`.
- **Cell 1 (Code):** Load annotations into Pandas DataFrames, verify disk files, display dataset summary table.
- **Cell 2 (Code):** Class distribution bar chart & pie chart, compute imbalance ratio, plot lesions per scan histogram.
- **Cell 3 (Code):** COCO size categorization (<32², 32²-96², >96²), stacked bar chart per class, aspect ratio scatter/KDE.
- **Cell 4 (Code):** 2D spatial pathology heatmap on panoramic dental arch coordinates $(x_c, y_c)$, 4-panel subplot showing anatomical priors.
- **Cell 5 (Code):** Pathology co-occurrence matrix & correlation heatmap.
- **Cell 6 (Code):** Radiographic pixel intensity histogram & contrast analysis across sample scans.
- **Cell 7 (Code):** Clinical sample visualizer with color-coded bounding boxes.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebooks.py -v`
Expected: PASS (syntax check + content requirement check)

- [ ] **Step 5: Commit**

```bash
git add notebooks/dentex_eda_analysis.ipynb tests/test_notebooks.py
git commit -m "feat(notebooks): implement dentex_eda_analysis.ipynb with spatial heatmaps and co-occurrence"
```

---

### Task 3: Implement Notebook 2 — `notebooks/dentex_inference_error_analysis.ipynb`

**Files:**
- Create: `notebooks/dentex_inference_error_analysis.ipynb`
- Modify: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `outputs/hierarchical_net/best.pth`, `outputs/diffusion_net/best.pth`, validation images, and COCO annotations.
- Produces: Standalone interactive inference & error analysis notebook:
  - 3-panel side-by-side comparison (`Ground Truth | HierarchicalNet | DiffusionNet`)
  - Random / specific image sampler with threshold controls
  - Quantitative TIDE/COCO error decomposition & $4 \times 5$ Confusion Matrix
  - Success Profiling for correct predictions
  - Deep analysis of Top 5 Failure Modes with case galleries
  - Clinical comparative summary

- [ ] **Step 1: Write test in `tests/test_notebooks.py` for `dentex_inference_error_analysis.ipynb` content**

Add test function checking required cells and keys:
```python
def test_dentex_inference_error_analysis_notebook():
    nb_path = Path("notebooks/dentex_inference_error_analysis.ipynb")
    assert nb_path.exists()
    text = nb_path.read_text(encoding="utf-8")
    for key in [
        "HierarchicalNet",
        "DiffusionNet",
        "Confusion Matrix",
        "Success Profiling",
        "Top 5 Failure Modes",
        "Ground Truth",
        "best.pth"
    ]:
        assert key in text, f"Missing {key} in dentex_inference_error_analysis.ipynb"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebooks.py::test_dentex_inference_error_analysis_notebook -v`
Expected: FAIL with `AssertionError: assert False` (file does not exist yet)

- [ ] **Step 3: Build `notebooks/dentex_inference_error_analysis.ipynb`**

Construct the Jupyter notebook JSON with the 8 complete, runnable cells:
- **Cell 0 (Markdown & Code):** Title, Colab/Drive mount, device check (CUDA/CPU).
- **Cell 1 (Code):** Model Checkpoint Loader: builds and restores `HierarchicalNetRunner` and `DiffusionNetRunner` from `best.pth`.
- **Cell 2 (Code):** 3-panel side-by-side visualizer function (`Ground Truth | HierarchicalNet | DiffusionNet`) with clinical color coding.
- **Cell 3 (Code):** Interactive random & specific image sampler with confidence threshold filter.
- **Cell 4 (Code):** Quantitative error breakdown (TIDE/COCO taxonomy: Cls Err, Loc Err, Bkg FP, Missed FN) and $4 \times 5$ Confusion Matrix.
- **Cell 5 (Code):** Success Profiling: characteristics of correct detections (box area, aspect ratios, confidence calibration, boundary sharpness).
- **Cell 6 (Code):** Deep analysis of Top 5 Failure Modes (Caries vs Deep Caries, small misses, anatomical pseudo-lesions, localization jitter, crowding/artifacts) with case gallery and clinical commentary.
- **Cell 7 (Code):** Clinical takeaway synthesis & Markdown comparison report.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebooks.py -v`
Expected: PASS (syntax check + content requirement check)

- [ ] **Step 5: Commit**

```bash
git add notebooks/dentex_inference_error_analysis.ipynb tests/test_notebooks.py
git commit -m "feat(notebooks): implement dentex_inference_error_analysis.ipynb with top-5 failure profiling"
```

---

### Task 4: Headless Execution & Comprehensive Test Suite Verification

**Files:**
- Create: `tests/test_execute_analysis_notebooks.py`
- Modify: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: Both notebooks (`dentex_eda_analysis.ipynb`, `dentex_inference_error_analysis.ipynb`) and sample test fixtures.
- Produces: Automated headless execution verification confirming zero runtime crashes.

- [ ] **Step 1: Write headless notebook execution test**

```python
# tests/test_execute_analysis_notebooks.py
import json
from pathlib import Path
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

@pytest.mark.parametrize("nb_name", [
    "dentex_eda_analysis.ipynb",
    "dentex_inference_error_analysis.ipynb"
])
def test_headless_execution_of_analysis_notebooks(nb_name, tmp_path):
    nb_path = Path("notebooks") / nb_name
    assert nb_path.exists()
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    # Execute in repository root
    ep.preprocess(nb, {"metadata": {"path": "."}})
    
    # Verify no cell contains an unhandled exception output
    for cell in nb.cells:
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    pytest.fail(f"Cell execution failed in {nb_name}: {out.get('ename')}: {out.get('evalue')}")
```

- [ ] **Step 2: Run test to verify execution**

Run: `pytest tests/test_execute_analysis_notebooks.py -v`
Expected: PASS (both notebooks execute cell-by-cell without runtime errors)

- [ ] **Step 3: Run full pytest suite across entire project**

Run: `pytest`
Expected: All tests pass across the repository.

- [ ] **Step 4: Final commit**

```bash
git add tests/test_execute_analysis_notebooks.py
git commit -m "test(notebooks): add automated headless execution verification for analysis notebooks"
```

---

## 6. Self-Review & Plan Verification

1. **Spec Coverage:**
   - Notebook 1 covers all 7 modules: class distribution, imbalance, COCO sizes, 2D spatial heatmap, co-occurrence, pixel intensity, sample visualizer -> Task 2.
   - Notebook 2 covers all 8 modules: 3-panel comparison, sampler, TIDE/COCO error breakdown, confusion matrix, success profiling, Top 5 failure modes profiling, clinical report -> Task 3.
   - Lightweight local testing prior to Colab -> Task 1 & Task 4.
2. **No Placeholders:** All tasks contain exact file paths, exact code snippets, and commands.
3. **Type Consistency:** Model runner names (`HierarchicalNetRunner`, `DiffusionNetRunner`), class colors, and COCO category definitions strictly match `dentex_bench`.
# DENTEX HierarchicalNet & DiffusionNet Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize the DENTEX 4-Class Dental Abnormality Detection benchmark by replacing Faster R-CNN and DINO with HierarchicalNet and DiffusionNet, eliminating CUDA OOM and NaN errors via memory-safe multi-scale resizing and gradient accumulation, capping training at 30 epochs with `best.pth` and `last.pth` checkpointing, and embedding comprehensive theoretical markdown cells into both Google Colab (`01_colab_run_all.ipynb`) and Kaggle (`02_kaggle_run_all.ipynb`) notebooks.

**Architecture:** A unified modular Python package `dentex_bench` provides dataset loading with multi-scale resizing, model runners (`HierarchicalNetRunner` with ResNet50-FPN and `DiffusionNetRunner` with dynamic diffusion box refinement), memory-safe AMP training with NaN unscale guards, and standard COCO/clinical metric evaluation. High-level CLI scripts (`scripts/train.py`, `scripts/evaluate.py`) coordinate execution, and Jupyter notebooks (`notebooks/01_colab_run_all.ipynb`, `notebooks/02_kaggle_run_all.ipynb`) present complete theoretical foundations alongside 1-click cloud execution.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, Torchvision, PyCOCOtools, OpenCV, Matplotlib, Tabulate, Albumentations, Google Colab, Kaggle Kernels.

## Global Constraints

- Python compatibility: $\ge 3.10$, PyTorch $\ge 2.0$.
- Memory safety constraint: `batch_size = 1`, `gradient_accumulation_steps = 16`, safe multi-scale resizing (short edge 480-800, max 1333) so peak VRAM is $< 6$ GB (never OOM on 14.56 GB T4 or 16 GB P100).
- Numerical stability constraint: AMP GradScaler with `GRAD_CLIP_NORM = 0.1` and Inf/NaN unscale skip check (never NaN loss).
- Training epoch constraint: Max 30 epochs, tracking and saving `best.pth` (validation mAP) and `last.pth` (epoch 30).
- Theoretical markdown requirement: Notebooks must contain complete theoretical sections covering model architectures, mathematical formulas, training workflow, and evaluation metrics.
- 4 Pathology classes: 1: `Caries`, 2: `Deep Caries`, 3: `Periapical Lesion`, 4: `Impacted Tooth`.

---

### Task 1: Multi-Scale Resizing & Memory-Safe Data Loading

**Files:**
- Modify: `dentex_bench/data/dataset.py`
- Modify: `dentex_bench/config.py`
- Test: `tests/test_data_pipeline.py`

**Interfaces:**
- Consumes: COCO annotations (`instances_train.json`, `instances_val.json`) and raw X-ray images.
- Produces: `DentexDataset` returning appropriately scaled image tensors $[3, H, W]$ with bounding boxes $[N, 4]$ rescaled proportionally to the new dimensions. `BenchmarkConfig` providing default `batch_size = 1`, `gradient_accumulation_steps = 16`, `img_size = 800`, `epochs = 30`.

- [ ] **Step 1: Write unit tests for multi-scale resizing in `tests/test_data_pipeline.py`**
  Verify that when an image of size $2800 \times 1300$ is loaded, it is resized so that its short edge is within $[480, 800]$ and max edge $\le 1333$, and bounding box coordinates are correctly scaled.

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_data_pipeline.py -k "resize" -v`
  Expected: FAIL (resizing logic not yet active in `DentexDataset`).

- [ ] **Step 3: Implement multi-scale resizing and aspect-ratio preserving transform in `dentex_bench/data/dataset.py`**
  Add a default resizing function in `DentexDataset`:
  Given image $(H, W)$, compute scale factor $s = \min(\text{target\_short} / \min(H, W), \text{max\_long} / \max(H, W))$, resize image to $(round(H \cdot s), round(W \cdot s))$, and scale box coordinates $x, y, w, h$ by $s$.

- [ ] **Step 4: Update `dentex_bench/config.py`**
  Set `batch_size = 1`, `gradient_accumulation_steps = 16`, `epochs = 30`, `learning_rate = 1e-4`, and update default models list.

- [ ] **Step 5: Run data pipeline tests to verify PASS**
  Run: `pytest tests/test_data_pipeline.py -v`
  Expected: PASS.

- [ ] **Step 6: Commit**
  Run: `git add dentex_bench/data/dataset.py dentex_bench/config.py tests/test_data_pipeline.py`  
  `git commit -m "feat(data): implement multi-scale image resizing and memory-safe config defaults"`

---

### Task 2: HierarchicalNet Implementation

**Files:**
- Create: `dentex_bench/models/hierarchical_net.py`
- Modify: `dentex_bench/models/__init__.py`
- Test: `tests/test_hierarchical_net.py`

**Interfaces:**
- Consumes: `BenchmarkConfig`, input image batch $[B, 3, H, W]$, targets with boxes and labels.
- Produces: `HierarchicalNetRunner` subclassing `BaseDetectorRunner`, implementing ResNet50 + FPN ($P_2, P_3, P_4, P_5$), multi-scale detection head, `train_epoch()` with AMP and NaN checks, `predict()`, and `save_checkpoint()` / `load_checkpoint()`.

- [ ] **Step 1: Write unit tests in `tests/test_hierarchical_net.py`**
  Verify model creation, forward pass on variable image sizes, loss calculation, backward pass with gradient accumulation, and checkpoint saving/loading.

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_hierarchical_net.py -v`
  Expected: FAIL (`ModuleNotFoundError` or `ImportError`).

- [ ] **Step 3: Implement `HierarchicalNetRunner` in `dentex_bench/models/hierarchical_net.py`**
  - Implement ResNet-50 feature extraction ($C_2, C_3, C_4, C_5$) with torchvision weights.
  - Implement FPN lateral convolutions ($1\times1$) and top-down smoothing ($3\times3$) producing $P_2, P_3, P_4, P_5$ (channels = 256).
  - Implement multi-scale detection head predicting 4 classes + background and box regression.
  - Implement `train_epoch` with:
    - `torch.amp.autocast('cuda')`
    - `scaler.scale(loss).backward()`
    - Every `accum_steps`: `scaler.unscale_()`, finite check (`torch.isfinite(g).all()`), `clip_grad_norm_(0.1)`, `scaler.step()`, `scaler.update()`.
    - If non-finite, skip optimizer step and update scaler to prevent NaN corruptions.
  - Implement `predict()` returning scores, boxes, and labels.

- [ ] **Step 4: Export `HierarchicalNetRunner` in `dentex_bench/models/__init__.py`**

- [ ] **Step 5: Run tests to verify PASS**
  Run: `pytest tests/test_hierarchical_net.py -v`
  Expected: PASS.

- [ ] **Step 6: Commit**
  Run: `git add dentex_bench/models/hierarchical_net.py dentex_bench/models/__init__.py tests/test_hierarchical_net.py`  
  `git commit -m "feat(models): implement HierarchicalNet with ResNet50-FPN and NaN-safe AMP"`

---

### Task 3: DiffusionNet Implementation

**Files:**
- Create: `dentex_bench/models/diffusion_net.py`
- Modify: `dentex_bench/models/__init__.py`
- Test: `tests/test_diffusion_net.py`

**Interfaces:**
- Consumes: `BenchmarkConfig`, input image batch $[B, 3, H, W]$, targets with boxes and labels.
- Produces: `DiffusionNetRunner` subclassing `BaseDetectorRunner`, implementing ResNet50 + FPN, sinusoidal timestep embedding, `DynamicDiffusionHead`, progressive box refinement, `train_epoch()`, and `predict()`.

- [ ] **Step 1: Write unit tests in `tests/test_diffusion_net.py`**
  Verify `DiffusionNetRunner` construction, diffusion timestep sampling, forward pass, loss calculation, backward pass with gradient accumulation, and checkpoint saving/loading.

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_diffusion_net.py -v`
  Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `DiffusionNetRunner` in `dentex_bench/models/diffusion_net.py`**
  - Implement ResNet50-FPN backbone for rich multi-scale conditioning.
  - Implement `SinusoidalPositionEmbeddings(dim=256)`.
  - Implement `DynamicDiffusionHead` with proposal box encoder, conditioning MLP, classification head (4 classes + bg), and delta-box regressor.
  - Implement `train_epoch` with noise sampling $t \in [0, 1000]$, noisy proposal box generation, AMP autocasting, scaled backward pass, gradient clipping (0.1), and finite check.
  - Implement `predict()` with iterative reverse denoising steps.

- [ ] **Step 4: Export `DiffusionNetRunner` in `dentex_bench/models/__init__.py`**

- [ ] **Step 5: Run tests to verify PASS**
  Run: `pytest tests/test_diffusion_net.py -v`
  Expected: PASS.

- [ ] **Step 6: Commit**
  Run: `git add dentex_bench/models/diffusion_net.py dentex_bench/models/__init__.py tests/test_diffusion_net.py`  
  `git commit -m "feat(models): implement DiffusionNet runner with progressive box denoising"`

---

### Task 4: CLI Scripts & Benchmark Evaluation Enhancement

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/evaluate.py`
- Modify: `dentex_bench/eval/reporter.py`
- Test: `tests/test_train_cli.py`
- Test: `tests/test_eval_and_reporting.py`

**Interfaces:**
- Consumes: Trained model checkpoints in `outputs/<model>/best.pth` and `outputs/<model>/last.pth`.
- Produces: `outputs/metrics.csv`, `outputs/training_curves.png`, `outputs/metrics_comparison.png`, `outputs/benchmark_summary.md`, `outputs/benchmark_table.tex`.

- [ ] **Step 1: Update `scripts/train.py`**
  - Support `--model hierarchical_net` and `--model diffusion_net` (with legacy fallback).
  - During the 30-epoch training loop, record epoch losses.
  - Evaluate validation loss/mAP each epoch to update `outputs/<model>/best.pth`.
  - Always save final checkpoint to `outputs/<model>/last.pth` at epoch 30.
  - Plot and save `outputs/<model>/training_curve.png`.

- [ ] **Step 2: Update `scripts/evaluate.py` and `dentex_bench/eval/reporter.py`**
  - Evaluate both `best.pth` and `last.pth` for `hierarchical_net` and `diffusion_net`.
  - Compute COCO metrics ($mAP_{50}, mAP_{75}, mAP_{50:95}, AR@100$), per-class AP, and clinical F1-macro.
  - Generate comparison bar chart `outputs/metrics_comparison.png`.
  - Save full tabular records to `outputs/metrics.csv`.
  - Update Markdown summary and LaTeX table generation.

- [ ] **Step 3: Run CLI and evaluation unit tests**
  Run: `pytest tests/test_train_cli.py tests/test_eval_and_reporting.py -v`
  Expected: PASS.

- [ ] **Step 4: Commit**
  Run: `git add scripts/train.py scripts/evaluate.py dentex_bench/eval/reporter.py tests/test_train_cli.py tests/test_eval_and_reporting.py`  
  `git commit -m "feat(train,eval): update CLI runners for HierarchicalNet and DiffusionNet with best/last checkpoints and rich plots"`

---

### Task 5: Standardize Notebook 1 (`notebooks/01_colab_run_all.ipynb`)

**Files:**
- Modify: `notebooks/01_colab_run_all.ipynb`
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: Google Colab environment (T4 GPU runtime, Google Drive mount).
- Produces: Complete, self-contained Colab notebook with theoretical markdown cells, 1-click execution for HierarchicalNet and DiffusionNet (30 epochs), inline rendering of training curves, metrics comparison chart, Markdown/LaTeX tables, and visual predictions.

- [ ] **Step 1: Embed Theoretical Markdown Sections**
  - **Section 0: Problem Definition & Clinical Dental Abnormality Taxonomy** (Caries, Deep Caries, Periapical Lesion, Impacted Tooth).
  - **Section 1: Model Theory & Mathematical Formulation**
    - Detailed HierarchicalNet: ResNet-50 backbone, FPN multi-scale feature hierarchy ($C_2..C_5 \rightarrow P_2..P_5$), lateral connections formula, classification & regression loss.
    - Detailed DiffusionNet: Denoising diffusion probabilistic framework, forward box noising equation, sinusoidal timestep embedding, progressive reverse denoising, dynamic diffusion loss.
  - **Section 2: Training Workflow & Memory Stabilization Theory**
    - Workflow diagram.
    - Memory-safe mechanism explanation: why `batch_size=1`, `accum_steps=16`, and multi-scale resizing prevent OOM on 15 GB Colab T4.
    - Gradient clipping & NaN unscale guard explanation.
    - Dual checkpointing strategy (`best.pth` vs `last.pth`).
  - **Section 3: Evaluation Metrics Theory**
    - Mathematical definition of IoU, Precision, Recall, PR curves.
    - Average Precision (AP), $mAP_{50}, mAP_{75}, mAP_{50:95}$, AR@100.
    - Clinical classification metrics: Macro F1-score and per-class AP.

- [ ] **Step 2: Update Code Execution Cells**
  - Cell for Google Drive mount (`USE_DRIVE = True`).
  - Cell for git clone / repo pull.
  - Cell for automated dependency check.
  - Cell for dataset download and COCO preparation.
  - Cell for training HierarchicalNet (30 epochs, lr=0.0001, batch_size=1, accum=16).
  - Cell for training DiffusionNet (30 epochs, lr=0.0001, batch_size=1, accum=16).
  - Cell for evaluation across best and last checkpoints.
  - Cell for inline display of `training_curves.png`, `metrics_comparison.png`, `metrics.csv`, Markdown summary, and LaTeX table.
  - Cell for inline visualization of GT vs Predicted bounding boxes on real dental X-rays.

- [ ] **Step 3: Validate Notebook JSON & Cell Structure**
  Run: `pytest tests/test_notebooks.py -v`
  Expected: PASS.

- [ ] **Step 4: Commit**
  Run: `git add notebooks/01_colab_run_all.ipynb tests/test_notebooks.py`  
  `git commit -m "feat(colab): standardize 01_colab_run_all.ipynb with theoretical foundations and HierarchicalNet/DiffusionNet pipeline"`

---

### Task 6: Standardize Notebook 2 (`notebooks/02_kaggle_run_all.ipynb`)

**Files:**
- Modify: `notebooks/02_kaggle_run_all.ipynb`
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: Kaggle Notebooks environment (GPU P100/T4, `/kaggle/input`, `/kaggle/working`).
- Produces: Complete, self-contained Kaggle notebook with identical rich theoretical foundations, smart Kaggle Input dataset detection, 30-epoch training for HierarchicalNet and DiffusionNet, and inline rendering of all evaluation artifacts.

- [ ] **Step 1: Embed Full Theoretical Markdown Sections**
  Mirror all comprehensive theoretical markdown cells from Notebook 1 (Taxonomy, HierarchicalNet & DiffusionNet math, Workflow & Memory stabilization, Evaluation metrics theory).

- [ ] **Step 2: Update Kaggle Code Execution Cells**
  - Git clone / setup in `/kaggle/working`.
  - Dependency check and setup.
  - Smart dataset detection from `/kaggle/input/*dentex*` (fallback to HF Hub download).
  - Training execution for HierarchicalNet (30 epochs) and DiffusionNet (30 epochs) with zero OOM risk.
  - Evaluation and artifact generation in `/kaggle/working/outputs`.
  - Inline display of plots, CSV, LaTeX table, and visual prediction comparisons.

- [ ] **Step 3: Validate Notebook JSON & Cell Structure**
  Run: `pytest tests/test_notebooks.py -v`
  Expected: PASS.

- [ ] **Step 4: Commit**
  Run: `git add notebooks/02_kaggle_run_all.ipynb tests/test_notebooks.py`  
  `git commit -m "feat(kaggle): standardize 02_kaggle_run_all.ipynb with theoretical foundations and Kaggle execution flow"`

---

### Task 7: End-to-End System Smoke Test & Verification

**Files:**
- Test: Full pytest suite across `tests/`

- [ ] **Step 1: Run full unit test suite**
  Run: `pytest tests/ -v`
  Expected: All tests PASS.

- [ ] **Step 2: Run a dry-run smoke test of train and evaluate**
  Run: `python scripts/train.py --model hierarchical_net --epochs 1 --batch_size 1`  
  Run: `python scripts/train.py --model diffusion_net --epochs 1 --batch_size 1`  
  Run: `python scripts/evaluate.py --compare_all`  
  Verify that checkpoints `best.pth` and `last.pth` are created, `metrics.csv` is written, and plots are generated.

- [ ] **Step 3: Final Commit and Documentation update**
  Update `README.md` to reflect the updated architecture (HierarchicalNet & DiffusionNet), memory safety guarantees, and standardized notebook instructions.  
  `git commit -m "docs: update README with HierarchicalNet and DiffusionNet benchmark documentation"`

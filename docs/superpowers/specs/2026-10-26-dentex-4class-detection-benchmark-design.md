# DENTEX 4-Class Abnormal Tooth Detection Benchmark - Design Specification

**Date:** 2026-10-26  
**Status:** Approved  
**Topic:** Reproducible Benchmark for 4-Class Dental Abnormality Detection on DENTEX (Faster R-CNN, DINO, HierarchicalDet)

---

## 1. Overview & Objective

This project establishes a standardized, reproducible, and multi-environment benchmarking framework for detecting 4 types of dental pathologies on panoramic dental X-rays using the **DENTEX** dataset (MICCAI 2023 Challenge, [arXiv:2303.06500](https://arxiv.org/pdf/2303.06500), [arXiv:2406.13963](https://arxiv.org/pdf/2406.13963)).

### 1.1 Target Problem & Classes
The benchmark focuses on the 4 abnormal tooth categories:
1. `Caries` (Tooth decay, Category ID: 1)
2. `Deep Caries` (Severe decay extending near or into pulp chamber, Category ID: 2)
3. `Periapical Lesion` (Periapical inflammatory/cyst lesion at root tip, Category ID: 3)
4. `Impacted Tooth` (Unerupted or mechanically blocked tooth, Category ID: 4)

Healthy / normal teeth function as background instances in this pathology detection task.

### 1.2 Target Model Architectures
A unified pipeline evaluates three representative object detection paradigms under identical splits, augmentations, and metrics:
1. **Faster R-CNN** (Two-stage anchor-based CNN baseline with ResNet50-FPN)
2. **DINO** (Vision Transformer DETR-based state-of-the-art with Improved DeNoising anchor boxes)
3. **HierarchicalDet / DiffusionDet** (Official DENTEX benchmark baseline utilizing a denoising diffusion process for bounding box prediction)

### 1.3 Key Operational Constraints
- **Dual Interface:** Full modular Python package (`.py`) for reusable logic, testability, and CLI scripting + interactive Jupyter notebooks (`.ipynb`) for 1-click cloud training (Google Colab, Kaggle).
- **Multi-Environment Auto-Detection:** Automatically senses runtime environment (Local Windows/Linux, Kaggle Notebooks, Google Colab) and available hardware (e.g. RTX 3050 4GB local vs 16GB T4/P100 on Cloud).
- **Auto-Dependency Resolution:** Automatically detects missing packages and installs appropriate wheels dynamically.
- **Memory-Safe Defaults:** Automatically throttles batch size, resolution, and applies gradient accumulation with FP16 mixed precision on low-VRAM GPUs ($\le$ 4GB) to eliminate Out-Of-Memory (OOM) errors.

---

## 2. System Architecture & Directory Layout

The project follows a **Core Engine + Multi-Interface** pattern:

```
Dentex/
├── dentex_bench/                  # Core library package (.py)
│   ├── __init__.py
│   ├── env.py                     # Environment detection (Local/Colab/Kaggle), VRAM probe, auto-installer
│   ├── config.py                  # Hyperparameters, paths, seeds, memory-safe defaults
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py          # Auto-download from HuggingFace Hub (ibrahimhamamci/DENTEX) / Kaggle
│   │   ├── parser.py              # Parse DENTEX annotations into COCO format for 4 disease classes
│   │   └── dataset.py             # PyTorch Dataset, Albumentations transforms, Stratified train/val split
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # Base detector runner interface
│   │   ├── faster_rcnn.py         # Faster R-CNN (ResNet50-FPN) adapter
│   │   ├── dino.py                # DINO DETR-based adapter
│   │   └── hierarchical_det.py    # HierarchicalDet / DiffusionDet adapted for 4 classes
│   └── eval/
│       ├── __init__.py
│       ├── coco_eval.py           # COCO evaluation: mAP50, mAP50:95, per-class AP, latency, VRAM
│       └── reporter.py            # Generates Markdown summary, LaTeX table, and visual prediction overlays
│
├── scripts/                       # CLI entrypoints (.py)
│   ├── setup_env.py               # Dependency checker and environment diagnostics
│   ├── download_and_prepare.py    # Download raw data and generate COCO 4-class annotations
│   ├── train.py                   # Unified training script (--model faster_rcnn|dino|hierarchical_det)
│   └── evaluate.py                # Unified evaluation script across checkpoints
│
├── notebooks/                     # Interactive Jupyter Notebooks (.ipynb)
│   ├── 01_colab_run_all.ipynb     # 1-click execution for Google Colab (Drive mount, train, benchmark)
│   ├── 02_kaggle_run_all.ipynb    # 1-click execution for Kaggle (GPU P100/T4)
│   └── 03_local_benchmark.ipynb   # Interactive local testing, inference, and visualization
│
├── data/                          # Data directory (git-ignored)
│   ├── raw/                       # Raw downloads from HuggingFace / Kaggle
│   └── processed/                 # Prepared images and instances_{train,val}.json
│
├── outputs/                       # Checkpoints, predictions, and benchmark tables
├── requirements.txt
└── README.md
```

---

## 3. Detailed Component Specifications

### 3.1 Environment Management & Auto-Setup (`dentex_bench/env.py`)
- **Runtime Detection:**
  - `is_colab()`: Checks for `google.colab` module.
  - `is_kaggle()`: Checks for `KAGGLE_KERNEL_RUN_TYPE` or `/kaggle/working`.
  - `is_local()`: Returns true if neither Colab nor Kaggle.
- **Hardware & Memory Probe:**
  - Detects CUDA availability, device name, and total VRAM.
  - Returns recommended runtime configuration:
    - Low VRAM ($\le$ 4GB, e.g. local RTX 3050): `batch_size = 1`, `grad_accum = 4`, `amp = True`, `img_size = 800`, `num_workers = 2`.
    - Standard/High VRAM ($\ge$ 15GB, e.g. T4, P100, A100): `batch_size = 4` or `8`, `grad_accum = 1`, `amp = True`, `img_size = 1024`, `num_workers = 4`.
- **Auto-Dependency Installer:**
  - Checks required packages: `torch`, `torchvision`, `transformers`, `pycocotools`, `albumentations`, `huggingface_hub`, `opencv-python`, `pandas`, `tabulate`, `tqdm`.
  - Automatically runs `pip install` for missing packages.
  - For HierarchicalDet: On Linux/Colab, resolves prebuilt wheels; on Windows local, provides pure-PyTorch implementation of diffusion sampling/head to avoid MSVC C++ compilation failures.

### 3.2 Data Pipeline (`dentex_bench/data/`)
- **Downloader (`downloader.py`):**
  - Uses `huggingface_hub.snapshot_download` to fetch `ibrahimhamamci/DENTEX` repository files.
  - Fallback support for local archive extraction or Kaggle dataset path.
- **Parser (`parser.py`):**
  - Parses DENTEX Level 3 annotation JSONs.
  - Maps classes to 4 standard targets:
    - `1`: `Caries`
    - `2`: `Deep Caries`
    - `3`: `Periapical Lesion`
    - `4`: `Impacted Tooth`
  - Generates standard COCO format: `images`, `annotations` (`[x, y, width, height]`), `categories`.
- **Dataset & Split (`dataset.py`):**
  - Stratified 80/20 train/val split using a fixed seed (`seed = 42`) to guarantee identical evaluation subsets.
  - Exports `instances_train.json` and `instances_val.json`.
  - PyTorch `DentexDataset` with Albumentations (HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate).

### 3.3 Model Runners (`dentex_bench/models/`)
All models implement `BaseDetectorRunner`:
- **`FasterRCNNRunner` (`faster_rcnn.py`):**
  - PyTorch `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`.
  - Head replaced with `FastRCNNPredictor(in_features, num_classes=5)`.
  - Native SGD/AdamW optimizer with linear warmup and cosine annealing.
- **`DINORunner` (`dino.py`):**
  - Hugging Face / DETR-based DINO architecture.
  - Hungarian matcher with Focal loss + L1 box loss + GIoU loss.
  - Bounding box denoiser for stable Transformer convergence.
- **`HierarchicalDetRunner` (`hierarchical_det.py`):**
  - DiffusionDet-based framework (official DENTEX baseline).
  - ResNet-50 FPN backbone.
  - Denoising diffusion head with time-step embedding and DDIM inference sampling.
  - Configured for the 4 dental pathology classes.

### 3.4 Evaluation & Reporting (`dentex_bench/eval/`)
- **`COCOEvaluator` (`coco_eval.py`):**
  - Evaluates predictions using `pycocotools.cocoeval.COCOeval`.
  - Computes $mAP_{50}$, $mAP_{50:95}$, $mAP_{75}$, Average Recall ($AR$).
  - Computes per-class AP: $AP_{Caries}$, $AP_{DeepCaries}$, $AP_{PeriapicalLesion}$, $AP_{ImpactedTooth}$.
  - Records inference latency (ms/image) and peak GPU VRAM usage.
- **`BenchmarkReporter` (`reporter.py`):**
  - Generates `outputs/benchmark_summary.md` with comparison tables.
  - Generates `outputs/benchmark_table.tex` formatted for academic publication (MICCAI/IEEE/Springer).
  - Generates overlay images of ground-truth vs predictions with color-coded bounding boxes.

### 3.5 Multi-Interface Workflow (Scripts & Notebooks)
- **CLI Scripts (`scripts/`):**
  - `python scripts/setup_env.py`
  - `python scripts/download_and_prepare.py --seed 42`
  - `python scripts/train.py --model faster_rcnn --epochs 30`
  - `python scripts/train.py --model dino --epochs 30`
  - `python scripts/train.py --model hierarchical_det --epochs 30`
  - `python scripts/evaluate.py --compare_all`
- **Jupyter Notebooks (`notebooks/`):**
  - Self-contained, structured cell sequences importing `dentex_bench`.
  - Support execution in Google Colab (with Drive mounting) and Kaggle (with GPU acceleration).

---

## 4. Error Handling & Edge Cases

1. **Windows Multiprocessing (`BrokenPipeError` / `spawn` errors):**
   - In `dentex_bench/config.py`, when `os.name == 'nt'`, workers are capped at 2 and DataLoader `persistent_workers` is configured safely.
2. **GPU Out-Of-Memory (OOM):**
   - Dynamic batch scaling and gradient accumulation prevent OOM on 4GB VRAM.
   - Catch `torch.cuda.OutOfMemoryError`, clear cache with `torch.cuda.empty_cache()`, and log suggested configuration adjustments.
3. **Network Download Interruption:**
   - Hugging Face download verifies file hashes and supports resume on reconnect.
4. **Extreme Class Imbalance:**
   - Stratified train/val splitting ensures all 4 classes are present in both subsets.
   - Evaluator handles classes with zero ground-truth instances in a mini-batch without crashing.

---

## 5. Verification & Testing Strategy

1. **Environment & Dependency Test:**
   - Run `setup_env.py` on local machine to confirm environment detection and package availability.
2. **Data Pipeline Unit Test:**
   - Test `parser.py` on synthetic and sample DENTEX annotations; verify COCO schema validity via `pycocotools.coco.COCO`.
3. **Smoke Train Test:**
   - Run 1-epoch mini-batch training for each model (`faster_rcnn`, `dino`, `hierarchical_det`) to verify loss computation, backward pass, and checkpoint saving.
4. **Evaluation Benchmark Test:**
   - Run `evaluate.py` to produce validation metrics, generating `benchmark_summary.md` and `benchmark_table.tex`.
5. **Notebook Validation:**
   - Verify that notebooks execute sequentially without syntax errors.

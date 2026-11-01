# DENTEX 4-Class Abnormal Tooth Detection Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MICCAI DENTEX](https://img.shields.io/badge/Challenge-MICCAI%20DENTEX%202023-purple.svg)](https://dentex.grand-challenge.org/)

A standardized, multi-environment benchmarking framework for detecting 4 abnormal tooth categories on panoramic dental radiographs using the **DENTEX** dataset ([arXiv:2303.06500](https://arxiv.org/pdf/2303.06500), [arXiv:2406.13963](https://arxiv.org/pdf/2406.13963)).

---

## 1. Overview & Abnormal Pathology Classes

The benchmark standardizes detection across the **4 clinical dental abnormality classes** from DENTEX Level 3 annotations:

| Category ID | Class Name | Description | Visualization Color |
|:-----------:|:-----------|:------------|:-------------------:|
| **1** | `Caries` | Enamel/dentin tooth decay lesions | `#E69F00` (Orange) |
| **2** | `Deep Caries` | Advanced decay encroaching into or abutting dental pulp | `#D55E00` (Vermilion/Red) |
| **3** | `Periapical Lesion` | Periapical inflammatory/cyst radiolucency at the root apex | `#0072B2` (Blue) |
| **4** | `Impacted Tooth` | Unerupted or mechanically blocked tooth (wisdom teeth, etc.) | `#009E73` (Green) |

> **Note:** Normal / healthy teeth are treated as background instances to isolate pathology localization and severity discrimination.

---

## 2. Evaluated Model Architectures

The benchmark centers on two advanced detection paradigms for 4-class dental abnormality detection, evaluated alongside classical baselines under identical train/validation splits, multi-scale augmentations, and evaluation protocols:

### Primary Benchmark Architectures
1. **HierarchicalNet (`hierarchical_net`)**: Multi-scale feature hierarchy with ResNet-50 backbone and Feature Pyramid Network (FPN, $P_2 \rightarrow P_5$). Designed to effectively detect pathologies with extreme spatial scale differences (from tiny enamel caries to large impacted third molars).
2. **DiffusionNet (`diffusion_net`)**: Denoising diffusion probabilistic model (DiffusionDet paradigm) with `DynamicDiffusionHead`. Utilizes sinusoidal timestep embeddings and iterative progressive box denoising for pathology localization and classification.

### Classical Detection Baselines
3. **Faster R-CNN (`faster_rcnn`)**: Two-stage anchor-based CNN baseline with ResNet50-FPN backbone and Fast R-CNN box regression head.
4. **DINO (`dino`)**: Modern Vision Transformer (DETR-based) detector with contrastive DeNoising anchor boxes and bipartite Hungarian matching.
5. **HierarchicalDet (`hierarchical_det`)**: Diffusion-based object detection framework (official DENTEX baseline adapted for 4 abnormality classes) with iterative reverse-diffusion box generation and ResNet50-FPN feature backbone.

---

## 3. Hardware-Adaptive & Multi-Platform Design

The framework automatically senses its runtime environment:
- **Local PC / Workstation**: Optimizes for local GPUs such as RTX 3050 (4GB VRAM). Automatically enables FP16 mixed precision, scales down batch size to 1, sets gradient accumulation steps to 4, and bounds DataLoader worker threads to prevent Windows multiprocessing `BrokenPipeError` or OOM crashes.
- **Google Colab**: Senses cloud environment, enables Google Drive checkpoint persistence, and tunes batch size for 16GB T4 / A100 GPUs.
- **Kaggle Notebooks**: Detects Kaggle environment, paths working artifacts to `/kaggle/working`, and enables GPU acceleration.

---

## 4. Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/dentex-benchmark.git
cd dentex-benchmark
pip install -r requirements.txt
```

Verify your environment and hardware:

```bash
python scripts/setup_env.py
```

---

## 5. CLI Pipeline Workflow

### Step 1: Download & Prepare Dataset
Downloads data from HuggingFace Hub (`ibrahimhamamci/DENTEX`) or parses local raw archives, producing standard COCO annotations (`instances_train.json`, `instances_val.json`) with an 80/20 stratified split:

```bash
python scripts/download_and_prepare.py --seed 42
```

### Step 2: Model Training
Train each model using the unified CLI runner:

```bash
# Train HierarchicalNet (30 epochs, memory-safe)
python scripts/train.py --model hierarchical_net --epochs 30 --lr 0.0001

# Train DiffusionNet (30 epochs, memory-safe)
python scripts/train.py --model diffusion_net --epochs 30 --lr 0.0001

# Classical Baselines (Optional)
python scripts/train.py --model faster_rcnn --epochs 30 --lr 0.005
python scripts/train.py --model dino --epochs 30 --lr 0.0001
```

### Step 3: Comprehensive Evaluation & Benchmarking
Evaluate all trained models or an individual checkpoint on validation data:

```bash
# Evaluate and compare all available checkpoints
python scripts/evaluate.py --compare_all

# Or evaluate a specific model checkpoint
python scripts/evaluate.py --model faster_rcnn --checkpoint outputs/faster_rcnn_best.pth
```

Evaluation outputs are exported to:
- `outputs/benchmark_summary.md` (Markdown summary table for GitHub)
- `outputs/benchmark_table.tex` (LaTeX table formatted for academic papers)
- `outputs/predictions_<model>.json` (Standard COCO detection format)

---

## 6. Interactive Jupyter Notebooks

Three plug-and-play notebooks are provided in the [`notebooks/`](notebooks/) directory:

- **[`notebooks/01_colab_run_all.ipynb`](notebooks/01_colab_run_all.ipynb)**: 1-click execution for Google Colab with Google Drive mount, automated environment setup, full training of all 3 architectures, evaluation, and inline Markdown report rendering.
- **[`notebooks/02_kaggle_run_all.ipynb`](notebooks/02_kaggle_run_all.ipynb)**: Complete Kaggle workflow targeting P100/T4 GPUs, working directory isolation, and artifact packaging.
- **[`notebooks/03_local_benchmark.ipynb`](notebooks/03_local_benchmark.ipynb)**: Interactive local testing, inference on custom panoramic X-rays, memory-safe execution, and color-coded bounding box visualization.

---

## 7. Metrics & Benchmark Reporting

The benchmark reports standard COCO metrics:
- Overall: $mAP_{50}$, $mAP_{50:95}$, $mAP_{75}$, Average Recall ($AR$)
- Per-Class AP: $AP_{\text{Caries}}$, $AP_{\text{Deep Caries}}$, $AP_{\text{Periapical Lesion}}$, $AP_{\text{Impacted Tooth}}$
- Efficiency: Latency (ms per image) and Peak GPU VRAM (GB)

---

## 8. Running Unit Tests

The test suite validates data preprocessing, all 3 model runners, evaluation pipelines, CLI entrypoints, and notebook schemas:

```bash
pytest tests/ -v
```

---

## 9. References & Citation

- **DENTEX Challenge**: [MICCAI 2023 DENTEX Challenge](https://dentex.grand-challenge.org/)
- **DENTEX Dataset**: [arXiv:2303.06500](https://arxiv.org/pdf/2303.06500)
- **Hierarchical Tooth Abnormality Detection**: [arXiv:2406.13963](https://arxiv.org/pdf/2406.13963)

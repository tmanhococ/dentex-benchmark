# DENTEX Benchmark: HierarchicalNet & DiffusionNet 4-Class Detection Design

**Date:** 2026-11-01  
**Status:** In Review / Approved by User  
**Topic:** Standardization of Notebook 1 (Colab) & Notebook 2 (Kaggle) with HierarchicalNet and DiffusionNet for 4-Class Dental Abnormality Detection

---

## 1. Executive Summary & Objective

This project standardizes the DENTEX 4-Class Dental Abnormality Detection benchmark across Google Colab and Kaggle environments, resolving previous Out-of-Memory (CUDA OOM) and numerical instability (`loss: nan`) failures observed in `notebooks/01_colab_run_all.ipynb`.

The revised pipeline:
1. Replaces Faster R-CNN and vanilla DINO with two dedicated target architectures:
   - **HierarchicalNet**: ResNet-50 backbone with Feature Pyramid Network (FPN) multi-scale feature hierarchy for 4 pathology classes.
   - **DiffusionNet**: Denoising diffusion probabilistic model (DiffusionDet paradigm) using iterative progressive box denoising for 4 pathology classes.
2. Adopts the memory and numerical stability standards proven in `notebooks/cv-dinoresnet.ipynb`:
   - Multi-scale image resizing (short edge $\in [480, 800]$, max long edge $= 1333$).
   - `batch_size = 1` with `gradient_accumulation_steps = 16` (effective batch 16, VRAM consumption $< 6$ GB).
   - Mixed Precision (AMP `GradScaler`) + `clip_grad_norm_ = 0.1` + dynamic NaN/Inf unscale check.
3. Limits training to **30 epochs**, capturing both `best.pth` (peak validation mAP) and `last.pth` (epoch 30).
4. Embeds full theoretical markdown cells into both notebooks:
   - Model mathematical formulations and network architecture diagrams.
   - End-to-end training workflow explanations.
   - Comprehensive metric formulations (IoU, AP, mAP50, mAP50:95, AR, F1-macro).
5. Provides rich reporting: `metrics.csv`, `training_curves.png`, `metrics_comparison.png`, Markdown and LaTeX summary tables, and GT vs Prediction visual overlays.

---

## 2. Theoretical Formulations

### 2.1 4-Class Dental Abnormality Taxonomy
The task localizes and classifies 4 clinically vital dental pathologies on panoramic radiographs:
1. $c = 1$: **Caries** (Tooth decay lesions in enamel/dentin)
2. $c = 2$: **Deep Caries** (Severe decay encroaching upon or exposing the dental pulp chamber)
3. $c = 3$: **Periapical Lesion** (Inflammatory periapical radiolucency / cyst at the tooth root apex)
4. $c = 4$: **Impacted Tooth** (Unerupted or mechanically blocked tooth, e.g. third molar)
Healthy/normal teeth serve as background.

---

### 2.2 HierarchicalNet Formulation
Dental pathologies exhibit high scale variance: Caries lesions are tiny ($< 30$ px), whereas Impacted Teeth and large Periapical Cysts span large spatial areas ($> 300$ px). HierarchicalNet addresses this through a multi-scale Feature Pyramid Network (FPN).

#### 1. Bottom-Up Pathway (ResNet-50 Backbone)
Given an input dental X-ray $I \in \mathbb{R}^{3 \times H \times W}$, feature representations are extracted across 4 residual stages at successive stride $\{4, 8, 16, 32\}$:
$$C_2, C_3, C_4, C_5 = \text{ResNet50}(I)$$

#### 2. Top-Down Pathway & Lateral Connections
Feature maps with strong semantic representation are fused top-down:
$$P_5 = \text{Conv}_{3\times3}(\text{Conv}_{1\times1}(C_5))$$
$$P_i = \text{Conv}_{3\times3}\left(\text{Conv}_{1\times1}(C_i) + \text{Upsample}_{2\times}(P_{i+1})\right), \quad i \in \{4, 3, 2\}$$
Where all pyramid levels $\{P_2, P_3, P_4, P_5\}$ share a uniform hidden channel dimension $d = 256$.

#### 3. Hierarchical Detection Head & Loss Function
Each pyramid level predicts classification logits $\hat{p} \in \mathbb{R}^{K+1}$ and bounding box offsets $\hat{t} = (\hat{t}_x, \hat{t}_y, \hat{t}_w, \hat{t}_h)$. The total loss is:
$$\mathcal{L}_{HierarchicalNet} = \mathcal{L}_{cls}(\hat{p}, p^*) + \lambda_{reg} \cdot \mathbf{1}_{[p^* \ge 1]} \mathcal{L}_{box}(\hat{t}, t^*)$$
where $\mathcal{L}_{cls}$ is Cross-Entropy / Focal Loss, and $\mathcal{L}_{box}$ is GIoU + Smooth L1 loss ($\lambda_{reg} = 2.0$).

---

### 2.3 DiffusionNet Formulation
DiffusionNet views object detection as a progressive denoising diffusion process, mapping random Gaussian bounding boxes into calibrated pathology bounding boxes and classifications.

#### 1. Forward Diffusion Process (Box Noising)
Let ground-truth bounding box coordinates be normalized to $z_0 \in [0, 1]^4$. At arbitrary continuous diffusion timestep $t \in [0, T]$ (where $T = 1000$):
$$q(z_t | z_0) = \mathcal{N}\left(z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$
where $\alpha_t = 1 - \beta_t$ follows a cosine/linear noise schedule, and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

#### 2. Conditioning and Timestep Embedding
The timestep $t$ is embedded via sinusoidal positional encodings:
$$e_k(t) = \begin{cases} \sin\left(\frac{t}{10000^{2k/d}}\right), & k \text{ even} \\ \cos\left(\frac{t}{10000^{2k/d}}\right), & k \text{ odd} \end{cases}$$
The conditioning representation merges spatial image features $F = \text{FPN}(\text{Backbone}(I))$ and time embedding:
$$Cond = \text{GlobalPool}(F) + \text{MLP}(e(t))$$

#### 3. Reverse Progressive Denoising & Head
Given noisy boxes $z_t$ and condition $Cond$, the dynamic diffusion head predicts clean box reconstructions $\hat{z}_0$ and category logits $\hat{c}$:
$$\hat{z}_0, \hat{c} = \text{DynamicDiffusionHead}(z_t, Cond)$$
$$\mathcal{L}_{DiffusionNet} = \mathcal{L}_{cls}(\hat{c}, c^*) + \lambda_{L1} \|\hat{z}_0 - z_0\|_1 + \lambda_{giou} \mathcal{L}_{giou}(\hat{z}_0, z_0)$$

---

### 2.4 Evaluation Metrics Formulations

#### 1. Intersection over Union (IoU)
$$\text{IoU}(B_p, B_{gt}) = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$

#### 2. Average Precision (AP)
For a specific class $c$, given precision $P(r)$ as a function of recall $r$:
$$\text{AP}_c = \int_0^1 P_c(r) \, dr \approx \sum_{k=1}^N (R_k - R_{k-1}) \max_{\tilde{r} \ge R_k} P(\tilde{r})$$

#### 3. COCO Benchmark Metrics
- **$mAP_{50}$**: Mean AP across all 4 classes at $\text{IoU} \ge 0.50$:
  $$mAP_{50} = \frac{1}{4} \sum_{c=1}^4 \text{AP}_c^{IoU=0.50}$$
- **$mAP_{75}$**: Strict AP at $\text{IoU} \ge 0.75$.
- **$mAP_{50:95}$**: Mean AP averaged across 10 IoU thresholds $\{0.50, 0.55, \dots, 0.95\}$:
  $$mAP_{50:95} = \frac{1}{10} \sum_{\tau \in \{0.50, 0.55, \dots, 0.95\}} mAP_\tau$$
- **Average Recall (AR@100)**: Maximum recall obtained with at most 100 detected bounding boxes per image.

#### 4. Clinical Classification Metrics
Matching predicted boxes to ground truth with $\text{IoU} \ge 0.50$ yields True Positives ($TP_c$), False Positives ($FP_c$), and False Negatives ($FN_c$):
- Precision: $P_c = \frac{TP_c}{TP_c + FP_c}$
- Recall: $R_c = \frac{TP_c}{TP_c + FN_c}$
- Macro F1:
  $$\text{F1}_{macro} = \frac{1}{4} \sum_{c=1}^4 \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

---

## 3. Training Workflow & Memory Stabilization

```
[ Input Dental Panoramic X-Ray ]
              │
              ▼
[ Multi-Scale Resize: Short Edge 480..800, Max Long Edge 1333 ]
              │
              ▼
[ DataLoader: batch_size = 1, num_workers = 2 ]
              │
              ▼
[ Mixed Precision Autocast: torch.amp.autocast('cuda') ]
              │
              ▼
[ Forward Pass: HierarchicalNet / DiffusionNet ]
              │
              ▼
[ Loss Calculation / Gradient Accumulation Steps (16) ]
              │
              ▼
[ Backward Pass: scaler.scale(loss).backward() ]
              │
              ▼
     [ Step % 16 == 0 ? ]
     ├── YES ──► [ scaler.unscale_(optimizer) ]
     │                │
     │                ▼
     │           [ Check Finite Gradients ]
     │           ├── Inf/NaN ──► [ Skip optimizer.step, scaler.update(), drop batch ]
     │           └── Finite  ──► [ clip_grad_norm_(0.1) ──► optimizer.step() ──► scaler.update() ]
     └── NO  ──► Continue accumulating
              │
              ▼
[ Epoch End: Evaluate Validation Loss / mAP ]
              │
              ├── If val_mAP > best_mAP ──► Save outputs/<model>/best.pth
              └── Always at epoch 30   ──► Save outputs/<model>/last.pth
```

---

## 4. Codebase Architecture & File Updates

1. **`dentex_bench/models/`**:
   - `hierarchical_net.py`: Implements `HierarchicalNetRunner` with ResNet-50 + FPN multi-scale feature hierarchy and safe multi-scale forward pass.
   - `diffusion_net.py`: Implements `DiffusionNetRunner` with ResNet-50 + FPN + `DynamicDiffusionHead` (denoising diffusion process).
   - `__init__.py`: Exports `HierarchicalNetRunner` and `DiffusionNetRunner`.
   - Deprecated models (`faster_rcnn.py`, `dino.py`, old `hierarchical_det.py`) retained for backwards compatibility or archived cleanly.
2. **`dentex_bench/config.py`**:
   - Updates model choices to `["hierarchical_net", "diffusion_net"]`.
   - Sets safe defaults: `batch_size = 1`, `gradient_accumulation_steps = 16`, `img_size = 800`, `epochs = 30`, `amp = True`.
3. **`dentex_bench/data/dataset.py`**:
   - Integrates multi-scale resizing transform (short edge 480-800, max 1333) so large original images never enter backbone unscaled.
4. **`scripts/train.py` & `scripts/evaluate.py`**:
   - Updates CLI choices to `hierarchical_net` and `diffusion_net`.
   - Tracks validation metric each epoch, saving `best.pth` and `last.pth`.
   - Evaluates both `best.pth` and `last.pth` on validation set, saving `metrics.csv`, `training_curves.png`, `metrics_comparison.png`, Markdown summary, and LaTeX table.
5. **Notebooks**:
   - **`notebooks/01_colab_run_all.ipynb`**:
     - Includes rich theoretical markdown cells (Architecture, Mathematical Formulas, Training Workflow, Evaluation Metrics).
     - Colab Google Drive mount (`USE_DRIVE = True`).
     - Trains `HierarchicalNet` (30 ep) and `DiffusionNet` (30 ep).
     - Inline visualization of training curves, metrics comparison, LaTeX code, and prediction overlays.
   - **`notebooks/02_kaggle_run_all.ipynb`**:
     - Includes identical rich theoretical markdown cells.
     - Auto-detects Kaggle Input dataset or fallback downloads.
     - Trains both models for 30 epochs without OOM.
     - Renders inline evaluation plots and tables.

---

## 5. Verification Plan

1. **Unit Tests**:
   - Test `HierarchicalNetRunner` build, forward pass, multi-scale input handling, train step, and checkpoint saving.
   - Test `DiffusionNetRunner` build, forward pass, timestep embedding, train step, and checkpoint saving.
   - Verify `scripts/train.py` runs with `--model hierarchical_net` and `--model diffusion_net`.
2. **Notebook Syntax & Structure Verification**:
   - Verify all notebook JSON structures, cells, and markdown math formatting.
   - Run smoke test on training loop with dummy/small batches to guarantee 0 OOM and 0 NaN occurrences.

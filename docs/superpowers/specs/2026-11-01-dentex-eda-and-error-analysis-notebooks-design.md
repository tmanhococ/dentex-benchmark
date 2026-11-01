# Design Spec: DENTEX EDA & Inference Error Analysis Notebooks

**Date:** 2026-11-01  
**Author:** AI Research Assistant & User  
**Status:** Approved by User  
**Target Runtime:** Google Colab (GPU T4/V100/A100 + Google Drive) & Local Workstation (Windows/Linux)

---

## 1. Overview & Objectives

The goal is to develop two standalone, publication-grade Jupyter notebooks for in-depth exploration of the DENTEX panoramic radiograph dataset and detailed error analysis of the trained object detection models (`HierarchicalNet` and `DiffusionNet`):

1. **`notebooks/dentex_eda_analysis.ipynb`**: Comprehensive Exploratory Data Analysis (EDA) covering both standard computer vision metrics and clinical dental anatomy characteristics (class distributions, bounding box geometries, COCO size categorization, 2D anatomical spatial heatmaps, multi-lesion co-occurrence, and radiographic intensity profiles).
2. **`notebooks/dentex_inference_error_analysis.ipynb`**: Multi-model inference, 3-panel side-by-side qualitative inspection (Ground Truth vs HierarchicalNet vs DiffusionNet), quantitative error decomposition (TIDE/COCO error taxonomy), success profiling (characteristics of correct predictions), and deep failure analysis focusing on the Top 5 most frequent failure modes in abnormal tooth detection.

Both notebooks are designed with dual-environment support: they run out-of-the-box on Google Colab with Google Drive integration, while also executing smoothly on a local machine for testing and offline verification.

---

## 2. System Architecture & Environment Handling

### 2.1 Dual-Environment Detection Engine
Both notebooks implement an automated environment resolver:
- **Google Colab Mode:**
  - Detects `google.colab`.
  - Mounts Google Drive at `/content/drive`.
  - Automatically locates pre-downloaded dataset files at `/content/drive/MyDrive/dentex-benchmark/data` or `/content/dentex-benchmark/data`.
  - Automatically loads trained checkpoints from `/content/drive/MyDrive/dentex-benchmark/outputs/<model>/best.pth` or local `outputs/<model>/best.pth`.
  - Activates CUDA FP16 autocasting on T4/V100/A100 GPUs for low-latency inference.
- **Local Machine Mode:**
  - Falls back to local repository paths (`data/`, `outputs/`).
  - Supports CPU fallback if CUDA is unavailable, enabling fast lightweight dry-runs.

### 2.2 Pathology Class Schema
Both notebooks adhere to the official DENTEX 4-class pathology ontology:
- **Class 1 (Caries):** Early/superficial enamel and dentin carious lesions. Color: Amber/Orange (`#E69F00`).
- **Class 2 (Deep Caries):** Advanced caries approaching or invading the pulp chamber. Color: Vermilion/Red-Orange (`#D55E00`).
- **Class 3 (Periapical Lesion):** Radiolucent inflammatory periapical lesion / cyst at root apex. Color: Blue (`#0072B2`).
- **Class 4 (Impacted Tooth):** Mechanically impacted / unerupted wisdom or ectopic tooth. Color: Green (`#009E73`).

---

## 3. Detailed Specification: Notebook 1 (`dentex_eda_analysis.ipynb`)

### Cell Workflow & Modules:

1. **Cell 0 - Environment Setup & Drive Mounting:**
   - Detects execution environment (Colab vs Local).
   - Mounts Google Drive if running in Colab.
   - Sets deterministic seeds (`SEED = 42`).
   - Imports standard libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `cv2`, `json`, `pathlib`).

2. **Cell 1 - Dataset Loader & Integrity Verification:**
   - Parses `instances_train.json` and `instances_val.json`.
   - Generates summary statistics: Total images, total annotations per split, split ratio.
   - Performs disk integrity check: Verifies that image paths in JSON match actual files on disk.

3. **Cell 2 - Class Distribution & Imbalance Analysis:**
   - Bar chart & Pie chart of the 4 pathology classes across Train and Validation sets.
   - Calculates the imbalance ratio between dominant and minority classes.
   - Histogram and boxplot of pathology counts per panoramic image (single vs multi-lesion patients).

4. **Cell 3 - Bounding Box Geometry & COCO Size Taxonomy:**
   - Evaluates object dimensions ($w, h$, aspect ratio $w/h$, area in $\text{px}^2$).
   - Categorizes lesions according to standard COCO object size scales:
     - **Small:** $\text{Area} < 32^2$ ($< 1,024 \text{ px}^2$).
     - **Medium:** $32^2 \le \text{Area} < 96^2$ ($1,024 - 9,216 \text{ px}^2$).
     - **Large:** $\text{Area} \ge 96^2$ ($> 9,216 \text{ px}^2$).
   - Stacked bar chart showing the small/medium/large composition per pathology class (e.g., highlighting that Caries are overwhelmingly small while Impacted Teeth are large).

5. **Cell 4 - Spatial Pathology Heatmap (2D Anatomic Coordinates):**
   - Normalizes bbox center coordinates $(x_c, y_c) \in [0, 1] \times [0, 1]$.
   - Generates 2D Kernel Density Estimation (KDE) and heatmaps for each class:
     - Reveals spatial anatomical priors: Impacted teeth at jaw angles/corners, Periapical lesions at root apex line, Caries along occlusal planes.

6. **Cell 5 - Pathology Co-occurrence Analysis:**
   - Computes the co-occurrence matrix across all panoramic radiographs.
   - Plots a correlation heatmap identifying clinical associations (e.g., co-existence of Caries and Deep Caries/Periapical lesions).

7. **Cell 6 - Radiographic Image Properties & Contrast Analysis:**
   - Distribution of raw panoramic image resolutions and aspect ratios.
   - Pixel intensity histogram: Dynamic range analysis, radiolucent (dark) vs radiopaque (light) distribution.

8. **Cell 7 - Clinical Sample Gallery:**
   - Displays sample panoramic radiographs with color-coded ground truth annotations and clinical captions.

---

## 4. Detailed Specification: Notebook 2 (`dentex_inference_error_analysis.ipynb`)

### Cell Workflow & Modules:

1. **Cell 0 - Environment Setup & Drive Mounting:**
   - Mounts Google Drive, verifies GPU availability, sets up paths.

2. **Cell 1 - Model Checkpoint Loader & Config Setup:**
   - Initializes `HierarchicalNetRunner` and `DiffusionNetRunner`.
   - Automatically loads `best.pth` from `outputs/<model>/` or Drive.
   - Sets models to `.eval()` mode, reports parameter count and inference device.

3. **Cell 2 - 3-Panel Side-by-Side Inference Visualizer:**
   - Renders 3 synchronized panels on a single row:
     $$\mathbf{Ground\ Truth} \quad \Big| \quad \mathbf{HierarchicalNet} \quad \Big| \quad \mathbf{DiffusionNet}$$
   - Displays bounding boxes with clinical color coding, predicted class names, confidence scores, and IoU with ground truth.

4. **Cell 3 - Interactive & Random Sample Selector:**
   - Random selector: Draws $N$ random images from the validation set with seed control.
   - Direct selector: Allows specifying an `image_id` or filename.
   - Threshold control: Configurable `conf_threshold` (default $0.30$) and `iou_threshold` (default $0.50$).

5. **Cell 4 - Quantitative Error Breakdown (TIDE / COCO Taxonomy) & Confusion Matrix:**
   - Matches predictions to ground truth at $\text{IoU} \ge 0.5$:
     - **True Positives (TP):** Correct class and $\text{IoU} \ge 0.5$.
     - **Classification Error (Cls Err):** $\text{IoU} \ge 0.5$ with ground truth, but incorrect class predicted.
     - **Localization Error (Loc Err):** Correct class, but $0.1 \le \text{IoU} < 0.5$.
     - **Background False Positive (Bkg FP):** Predicted box with $\text{IoU} < 0.1$ with any ground truth box.
     - **Missed / False Negative (FN):** Ground truth lesion not detected.
   - Computes and plots the $4 \times 5$ Confusion Matrix (4 pathology classes + Background/Missed) for both models.
   - Comparative bar chart of error proportions between HierarchicalNet and DiffusionNet.

6. **Cell 5 - Success Profiling (Characteristics of Correct Predictions):**
   - Profiles True Positive detections:
     - Area and aspect ratio distribution of successfully detected lesions.
     - Spatial distribution on the dental arch.
     - Confidence score distribution (calibration) of HierarchicalNet vs DiffusionNet on correct predictions.
     - Visual clarity: Highlights how sharp boundaries and clear radiolucency contribute to successful detections.

7. **Cell 6 - Deep Analysis of Top 5 Failure Modes:**
   - Ranks and extracts the Top 5 most frequent failure modes:
     1. **Top 1: Caries $\longleftrightarrow$ Deep Caries Confusion:** Borderline lesion depth at the dentin-pulp junction.
     2. **Top 2: Missed Small Lesions (False Negatives):** Tiny occlusal caries or early-stage periapical radiolucencies ($< 32^2\text{ px}$).
     3. **Top 3: Anatomical Pseudo-Lesions (Background False Positives):** Normal structures (mental foramen, maxillary sinus boundary, dental follicle) misclassified as periapical lesions or cysts.
     4. **Top 4: Localization Jitter ($0.1 \le \text{IoU} < 0.5$):** Box alignment discrepancies on diffuse, non-distinct lesion borders.
     5. **Top 5: Dental Crowding & Radiopaque Artifacts:** Overlapping teeth, metallic restorations (amalgam, crowns, orthodontic brackets) degrading lesion boundaries.
   - For each failure mode:
     - Displays 2-3 representative case images with ground truth vs model predictions.
     - Compares failure counts between HierarchicalNet and DiffusionNet.
     - Provides clinical and architectural commentary explaining why the model failed.

8. **Cell 7 - Comparative Clinical Synthesis:**
   - Synthesizes findings into a Markdown report:
     - Multi-scale FPN benefits vs Diffusion iterative refinement.
     - Recommendations for clinical deployment and future model improvements.

---

## 5. Verification & Testing Strategy

Before delivery, local end-to-end execution testing will be performed:

1. **Lightweight Test Fixture:**
   - A small sample dataset (2-4 images + valid COCO `instances_train.json` and `instances_val.json`) is prepared locally.
   - Lightweight checkpoint states are created for HierarchicalNet and DiffusionNet to allow zero-delay local execution.
2. **Headless Execution Verification:**
   - Run both notebooks end-to-end using an automated runner (`pytest` or `nbconvert`).
   - Verify that all cells execute with exit code 0, all figures render, dataframes compute, and no missing symbol or CUDA errors occur.
3. **Colab Compatibility Check:**
   - Ensure all path operations use cross-platform `pathlib.Path` and handle Google Drive fallback gracefully.
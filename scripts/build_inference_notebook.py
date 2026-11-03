"""Programmatic generator for notebooks/dentex_inference_error_analysis.ipynb.

Every number, figure and statement in the generated notebook is computed from the trained
checkpoints' predictions on the validation split. Nothing is simulated: a missing checkpoint or
image raises instead of being replaced by a synthetic stand-in.

Regenerate with:
    python scripts/build_inference_notebook.py
"""
import json
from pathlib import Path


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


CELLS = [
    md(r"""
# DENTEX 4-Class Abnormal Tooth Detection — Inference & Error Analysis

Inference and error analysis of the two benchmark detectors on the **validation split** of DENTEX
(4 classes: Caries, Deep Caries, Periapical Lesion, Impacted Tooth).

1. **HierarchicalNet** — ResNet-50 + FPN feature hierarchy, region proposals on every pyramid level, multi-level RoIAlign box head.
2. **DiffusionNet** — DiffusionDet-style detector: Gaussian boxes are denoised by RoI-feature refinement stages (DDIM sampling with box renewal).

**Integrity rules of this notebook**
- Every metric, figure, gallery and statement below is computed from the checkpoints' predictions on real validation radiographs.
- A missing checkpoint, a checkpoint from an older architecture, or a missing / mismatched image stops the notebook with an error — nothing is replaced by synthetic data.
- The operating confidence threshold is chosen per model by maximising F1 on this same validation split, so P/R/F1 at that threshold are optimistic. COCO mAP does not depend on any threshold.

> Regenerate with `python scripts/build_inference_notebook.py`.

### Contents
1. Setup & validation data
2. Checkpoint loading (`best.pth`)
3. Cohort inference & COCO metrics
4. Confidence-threshold sweep (P / R / F1)
5. Ground Truth vs HierarchicalNet vs DiffusionNet on a scan
6. Error taxonomy & Confusion Matrix
7. Success Profiling
8. Top 5 Failure Modes (real cases)
9. Summary generated from the measurements
"""),
    code(r"""
import os
import sys
import json
import time
import random
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import torch

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REPO_URL = 'https://github.com/tmanhococ/dentex-benchmark.git'
BRANCH = 'main'

if IN_COLAB:
    import subprocess
    from google.colab import drive
    drive.mount('/content/drive')
    repo_root = Path('/content/dentex-benchmark')
    if (repo_root / '.git').exists():
        # Always analyse with the latest code; a stale clone would run old model definitions
        subprocess.run(['git', '-C', str(repo_root), 'fetch', 'origin', BRANCH], check=True)
        subprocess.run(['git', '-C', str(repo_root), 'reset', '--hard', f'origin/{BRANCH}'], check=True)
    else:
        subprocess.run(['git', 'clone', '-b', BRANCH, REPO_URL, str(repo_root)], check=True)
    os.chdir(repo_root)
    drive_base = Path('/content/drive/MyDrive/dentex-benchmark')
    DATA_DIR = drive_base / 'data' if (drive_base / 'data').exists() else repo_root / 'data'
    OUTPUTS_DIR = drive_base / 'outputs' if (drive_base / 'outputs').exists() else repo_root / 'outputs'
else:
    repo_root = Path.cwd().resolve()
    if repo_root.name == 'notebooks':
        repo_root = repo_root.parent
        os.chdir(repo_root)
    DATA_DIR = Path('data')
    OUTPUTS_DIR = Path('outputs')
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dentex_bench.config import BenchmarkConfig
from dentex_bench.data.dataset import build_image_index, load_annotated_image
from dentex_bench.eval.coco_eval import evaluate_coco_predictions
from dentex_bench.models import HierarchicalNetRunner, DiffusionNetRunner

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Platform: {"Google Colab" if IN_COLAB else "Local"} | device: {DEVICE}' + (f' ({torch.cuda.get_device_name(0)})' if DEVICE == 'cuda' else ''))
print(f'Data: {DATA_DIR.resolve()} | Outputs: {OUTPUTS_DIR.resolve()}')

CLASS_METADATA = {
    1: {'name': 'Caries', 'color': '#E69F00'},
    2: {'name': 'Deep Caries', 'color': '#D55E00'},
    3: {'name': 'Periapical Lesion', 'color': '#0072B2'},
    4: {'name': 'Impacted Tooth', 'color': '#009E73'},
}
CLASS_NAMES = [CLASS_METADATA[i]['name'] for i in range(1, 5)]

VAL_JSON = DATA_DIR / 'processed' / 'instances_val.json'
RAW_IMG_DIR = DATA_DIR / 'raw'
if not VAL_JSON.exists():
    raise FileNotFoundError(f'{VAL_JSON} not found. Run scripts/download_and_prepare.py first.')
with open(VAL_JSON, encoding='utf-8') as f:
    val_coco = json.load(f)

val_images = {im['id']: im for im in val_coco['images']}
gt_by_img = defaultdict(list)
for a in val_coco['annotations']:
    gt_by_img[a['image_id']].append(a)

def gt_arrays(img_id):
    anns = gt_by_img[img_id]
    boxes = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns], dtype=float).reshape(-1, 4)
    return boxes, np.array([a['category_id'] for a in anns], dtype=int)

print(f'Validation: {len(val_images)} scans, {len(val_coco["annotations"])} lesions, '
      f'{sum(1 for i in val_images if not gt_by_img[i])} scans without any of the 4 pathologies')
for cid, meta in CLASS_METADATA.items():
    print(f'  {meta["name"]}: {sum(1 for a in val_coco["annotations"] if a["category_id"] == cid)}')

IMAGE_INDEX = build_image_index(RAW_IMG_DIR)
print(f'Indexed {sum(len(v) for v in IMAGE_INDEX.values())} image files ({len(IMAGE_INDEX)} distinct names) under {RAW_IMG_DIR}')

def load_scan(img_meta):
    # Raises if the radiograph is missing or if no file with that name has the annotated size
    rgb, _ = load_annotated_image(img_meta['file_name'], img_meta.get('width'), img_meta.get('height'), RAW_IMG_DIR, IMAGE_INDEX)
    return rgb
"""),
    md(r"""
## 2. Checkpoint loading (`best.pth`)
`best.pth` is the epoch with the highest validation mAP50 recorded by `scripts/train.py`.
"""),
    code(r"""
cfg = BenchmarkConfig.from_auto_env()
cfg.device = DEVICE
MODEL_SPECS = {'HierarchicalNet': ('hierarchical_net', HierarchicalNetRunner),
               'DiffusionNet': ('diffusion_net', DiffusionNetRunner)}
MODEL_COLORS = {'HierarchicalNet': '#4C72B0', 'DiffusionNet': '#55A868'}

runners, ckpt_info = {}, {}
for name, (key, runner_cls) in MODEL_SPECS.items():
    ckpt = OUTPUTS_DIR / key / 'best.pth'
    if not ckpt.exists():
        raise FileNotFoundError(f'{ckpt} not found. Train it first: python scripts/train.py --model {key}')
    runner = runner_cls(cfg)
    runner.build_model(pretrained=False)
    # Raises ValueError for checkpoints written by an older architecture
    ckpt_info[name] = runner.load_checkpoint(ckpt)
    runners[name] = runner
    print(f'{name}: epoch {ckpt_info[name].get("epoch")}, train loss {ckpt_info[name].get("loss")}, '
          f'val mAP50 when selected {ckpt_info[name].get("val_mAP50")}')
"""),
    md(r"""
## 3. Cohort inference & COCO metrics
Each model runs once on every validation scan at original resolution (the runners resize and normalise internally and
return boxes in original pixels). All predictions are kept (up to 100 per scan) for COCO mAP; later sections apply a
confidence threshold.
"""),
    code(r"""
PREDICTIONS = {name: {} for name in runners}
LATENCIES = {name: [] for name in runners}

for img_id, meta in val_images.items():
    tensor = torch.from_numpy(load_scan(meta)).permute(2, 0, 1).float() / 255.0
    for name, runner in runners.items():
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = runner.predict([tensor])[0]
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        LATENCIES[name].append((time.perf_counter() - t0) * 1000.0)
        PREDICTIONS[name][img_id] = {
            'boxes': out['boxes'].numpy().reshape(-1, 4),
            'labels': out['labels'].numpy().astype(int),
            'scores': out['scores'].numpy(),
        }

coco_rows = {}
for name in runners:
    preds = [dict(image_id=i, **p) for i, p in PREDICTIONS[name].items()]
    metrics = evaluate_coco_predictions(VAL_JSON, preds)
    # The first call includes CUDA warm-up, so the median is reported
    metrics['latency_ms_median'] = round(float(np.median(LATENCIES[name])), 1)
    coco_rows[name] = metrics
COCO_TABLE = pd.DataFrame(coco_rows).T
print('=== COCO metrics on the validation split ===')
display(COCO_TABLE)
"""),
    md(r"""
## 4. Confidence-threshold sweep
Detections are matched to ground truth at IoU ≥ 0.5 (greedy by score, a prediction can only match a lesion of its own class).
The F1-maximising threshold becomes each model's **operating threshold** for the rest of the notebook. The row at 0.30
shows what the earlier fixed threshold gives.
"""),
    code(r"""
EVAL_IOU = 0.5

def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)

def analyse_scan(img_id, name, thresh, iou_thresh=EVAL_IOU):
    # Classifies every prediction above thresh (TP / Cls / Loc / Bkg / Dup) and lists missed lesions
    gt_boxes, gt_labels = gt_arrays(img_id)
    p = PREDICTIONS[name][img_id]
    keep = p['scores'] >= thresh
    pb, pl, ps = p['boxes'][keep], p['labels'][keep], p['scores'][keep]
    ious = iou_matrix(pb, gt_boxes)
    tp_of_gt, records = {}, []
    for pi in np.argsort(-ps):
        row = ious[pi] if len(gt_boxes) else np.zeros(0)
        same = [g for g in range(len(gt_boxes)) if gt_labels[g] == pl[pi] and row[g] >= iou_thresh]
        free = [g for g in same if g not in tp_of_gt]
        if free:
            g = max(free, key=lambda k: row[k]); kind = 'TP'; tp_of_gt[g] = pi
        elif same:
            g = max(same, key=lambda k: row[k]); kind = 'Dup'
        elif len(row) and row.max() >= iou_thresh:
            g = int(row.argmax()); kind = 'Cls'
        elif len(row) and row.max() >= 0.1:
            g = int(row.argmax()); kind = 'Loc'
        else:
            g = -1; kind = 'Bkg'
        records.append({'image_id': img_id, 'kind': kind, 'pred_box': pb[pi], 'pred_label': int(pl[pi]), 'score': float(ps[pi]),
                        'gt': g, 'gt_label': int(gt_labels[g]) if g >= 0 else None, 'gt_box': gt_boxes[g] if g >= 0 else None,
                        'iou': float(row[g]) if g >= 0 else 0.0})
    missed = [{'image_id': img_id, 'gt': g, 'gt_label': int(gt_labels[g]), 'gt_box': gt_boxes[g]}
              for g in range(len(gt_boxes)) if g not in tp_of_gt]
    return records, missed

def analyse_cohort(name, thresh):
    records, missed = [], []
    for img_id in val_images:
        r, m = analyse_scan(img_id, name, thresh)
        records += r; missed += m
    return records, missed

def prf(records, missed):
    tp = sum(r['kind'] == 'TP' for r in records); fp = len(records) - tp; fn = len(missed)
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {'TP': tp, 'FP': fp, 'FN': fn, 'precision': p, 'recall': r, 'f1': 2 * p * r / (p + r) if p + r else 0.0}

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
sweep = {name: pd.DataFrame([dict(threshold=t, **prf(*analyse_cohort(name, t))) for t in THRESHOLDS]) for name in runners}
OPERATING_THRESH = {name: float(df.loc[df['f1'].idxmax(), 'threshold']) for name, df in sweep.items()}

fig, axes = plt.subplots(1, len(runners), figsize=(7 * len(runners), 4.5), squeeze=False)
for ax, (name, df) in zip(axes[0], sweep.items()):
    for col, style in [('precision', '--'), ('recall', ':'), ('f1', '-')]:
        ax.plot(df['threshold'], df[col], style, color=MODEL_COLORS[name], label=col)
    ax.axvline(OPERATING_THRESH[name], color='black', alpha=0.5, label=f'max F1 @ {OPERATING_THRESH[name]:.2f}')
    ax.set_title(f'{name}: P / R / F1 vs confidence threshold', fontweight='bold')
    ax.set_xlabel('confidence threshold'); ax.set_ylim(0, 1); ax.legend()
plt.tight_layout(); plt.show()

for name, df in sweep.items():
    print(f'\n{name}: operating threshold {OPERATING_THRESH[name]:.2f}')
    display(df[df['threshold'].isin([0.3, OPERATING_THRESH[name]])].round(3))
"""),
    md(r"""
## 5. Ground Truth vs HierarchicalNet vs DiffusionNet on a scan
Set `SAMPLE_IMAGE_ID` to inspect a specific scan; `None` picks one at random.
"""),
    code(r"""
def draw_boxes(ax, boxes, labels, scores=None, dashed=False):
    for i, (b, c) in enumerate(zip(boxes, labels)):
        meta = CLASS_METADATA[int(c)]
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=2.2,
                                       edgecolor=meta['color'], facecolor='none', linestyle='--' if dashed else '-'))
        text = meta['name'] + (f' {scores[i]:.2f}' if scores is not None else '')
        ax.text(b[0], max(0.0, b[1] - 5), text, color='white', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.2', facecolor=meta['color'], alpha=0.9, edgecolor='none'))

def show_scan(img_id):
    meta = val_images[img_id]
    rgb = load_scan(meta)
    gt_boxes, gt_labels = gt_arrays(img_id)
    fig, axes = plt.subplots(1, 1 + len(runners), figsize=(8 * (1 + len(runners)), 5))
    axes[0].imshow(rgb); draw_boxes(axes[0], gt_boxes, gt_labels)
    axes[0].set_title(f'Ground Truth ({len(gt_boxes)} lesions)', fontweight='bold')
    for ax, name in zip(axes[1:], runners):
        thr = OPERATING_THRESH[name]
        p = PREDICTIONS[name][img_id]; keep = p['scores'] >= thr
        stats = prf(*analyse_scan(img_id, name, thr))
        ax.imshow(rgb); draw_boxes(ax, p['boxes'][keep], p['labels'][keep], p['scores'][keep])
        ax.set_title(f'{name} (conf ≥ {thr:.2f}): {int(keep.sum())} boxes | TP {stats["TP"]} FP {stats["FP"]} FN {stats["FN"]}', fontweight='bold')
    for ax in axes:
        ax.axis('off')
    handles = [patches.Patch(color=CLASS_METADATA[c]['color'], label=CLASS_METADATA[c]['name']) for c in CLASS_METADATA]
    fig.legend(handles=handles, loc='lower center', ncol=4)
    fig.suptitle(f'Scan {img_id} ({meta["file_name"]}, {meta["width"]}x{meta["height"]})', fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]); plt.show()

SAMPLE_IMAGE_ID = None
show_scan(SAMPLE_IMAGE_ID if SAMPLE_IMAGE_ID in val_images else random.Random(0).choice(list(val_images)))
"""),
    md(r"""
## 6. Error taxonomy & Confusion Matrix (at each model's operating threshold)
- **Cls**: overlaps a lesion (IoU ≥ 0.5) but with the wrong class · **Loc**: right area, 0.1 ≤ IoU < 0.5 · **Bkg**: IoU < 0.1 with every lesion
- **Dup**: second detection of an already-found lesion · **Missed**: lesion with no correct detection

Confusion-matrix rows are ground-truth lesions: a missed lesion is counted under the class of the highest-scoring
wrong-class detection overlapping it, otherwise under *Missed*.
"""),
    code(r"""
COHORT = {name: analyse_cohort(name, OPERATING_THRESH[name]) for name in runners}
KINDS = ['Cls', 'Loc', 'Bkg', 'Dup']

error_counts = pd.DataFrame({
    name: {**{k: sum(r['kind'] == k for r in recs) for k in ['TP'] + KINDS}, 'Missed': len(missed)}
    for name, (recs, missed) in COHORT.items()
}).T
display(error_counts)

ax = error_counts[KINDS + ['Missed']].T.plot.bar(figsize=(11, 4.5), color=[MODEL_COLORS[n] for n in error_counts.index], edgecolor='black')
ax.set_title('Error taxonomy at the operating threshold', fontweight='bold'); ax.set_ylabel('count')
for c in ax.containers:
    ax.bar_label(c, fontsize=8)
plt.xticks(rotation=0); plt.tight_layout(); plt.show()

def confusion(recs, missed):
    cm = np.zeros((4, 5), dtype=int)
    cls_err = defaultdict(list)
    for r in recs:
        if r['kind'] == 'TP':
            cm[r['gt_label'] - 1, r['gt_label'] - 1] += 1
        elif r['kind'] == 'Cls':
            cls_err[(r['image_id'], r['gt'])].append(r)
    for m in missed:
        wrong = cls_err.get((m['image_id'], m['gt']))
        col = max(wrong, key=lambda r: r['score'])['pred_label'] - 1 if wrong else 4
        cm[m['gt_label'] - 1, col] += 1
    return cm

fig, axes = plt.subplots(1, len(runners), figsize=(9 * len(runners), 6.5), squeeze=False)
for ax, (name, (recs, missed)) in zip(axes[0], COHORT.items()):
    cm = confusion(recs, missed)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    annot = np.array([[f'{cm[r, c]}\n({norm[r, c]:.1f}%)' for c in range(5)] for r in range(4)])
    sns.heatmap(norm, annot=annot, fmt='', cmap='Blues', vmin=0, vmax=100, ax=ax,
                xticklabels=CLASS_NAMES + ['Missed'], yticklabels=CLASS_NAMES)
    ax.set_title(f'Confusion Matrix: {name} (conf ≥ {OPERATING_THRESH[name]:.2f})', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Ground Truth')
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 7. Success Profiling
Almost every DENTEX lesion is COCO-"large" (≥ 96² px), so lesion size is split into validation-set terciles instead.
"""),
    code(r"""
all_areas = np.array([a['bbox'][2] * a['bbox'][3] for a in val_coco['annotations']], dtype=float)
edges = np.quantile(all_areas, [0, 1 / 3, 2 / 3, 1]) if len(all_areas) else np.array([0, 1, 2, 3])
size_names = [f'{"small" if i == 0 else "medium" if i == 1 else "large"} tercile\n({edges[i]:.0f}–{edges[i + 1]:.0f} px²)' for i in range(3)]

def size_bin(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return int(np.clip(np.searchsorted(edges, area, side='right') - 1, 0, 2))

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
width = 0.8 / len(runners)
for k, (name, (recs, missed)) in enumerate(COHORT.items()):
    tps = [r for r in recs if r['kind'] == 'TP']
    color = MODEL_COLORS[name]
    # A: recall per class
    rec_cls = [sum(r['gt_label'] == c for r in tps) / max(1, sum(r['gt_label'] == c for r in tps) + sum(m['gt_label'] == c for m in missed)) for c in range(1, 5)]
    axes[0, 0].bar(np.arange(4) + k * width, rec_cls, width, label=name, color=color, edgecolor='black')
    # B: recall per size tercile
    hit = Counter(size_bin(r['gt_box']) for r in tps); miss = Counter(size_bin(m['gt_box']) for m in missed)
    axes[0, 1].bar(np.arange(3) + k * width, [hit[i] / max(1, hit[i] + miss[i]) for i in range(3)], width, label=name, color=color, edgecolor='black')
    # C: score distribution of TPs vs false positives
    axes[1, 0].hist([r['score'] for r in tps], bins=np.linspace(0, 1, 21), alpha=0.6, color=color, label=f'{name} TP (n={len(tps)})')
    axes[1, 0].hist([r['score'] for r in recs if r['kind'] != 'TP'], bins=np.linspace(0, 1, 21), histtype='step', linewidth=2,
                    color=color, label=f'{name} FP (n={len(recs) - len(tps)})')
    # D: localisation quality of TPs
    if tps:
        axes[1, 1].hist([r['iou'] for r in tps], bins=np.linspace(0.5, 1, 11), alpha=0.6, color=color, label=f'{name} (mean IoU {np.mean([r["iou"] for r in tps]):.2f})')

axes[0, 0].set_xticks(np.arange(4) + width / 2); axes[0, 0].set_xticklabels(CLASS_NAMES); axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_title('A: Recall per pathology', fontweight='bold')
axes[0, 1].set_xticks(np.arange(3) + width / 2); axes[0, 1].set_xticklabels(size_names); axes[0, 1].set_ylim(0, 1)
axes[0, 1].set_title('B: Recall per lesion-size tercile', fontweight='bold')
axes[1, 0].set_title('C: Confidence of true vs false positives', fontweight='bold'); axes[1, 0].set_xlabel('score')
axes[1, 1].set_title('D: IoU of true positives', fontweight='bold'); axes[1, 1].set_xlabel('IoU')
for ax in axes.flat:
    ax.legend(fontsize=9)
plt.suptitle('Success Profiling (operating thresholds)', fontweight='bold'); plt.tight_layout(); plt.show()
"""),
    md(r"""
## 8. Top 5 Failure Modes (real validation cases)
Failure modes are defined from the error records — missed lesions per class, class confusions per (true → predicted)
pair, poorly localised detections per class, background detections per predicted class and duplicates — and ranked by
their total count over both models. Each row shows real crops: solid = ground truth, dashed = prediction.
"""),
    code(r"""
def mode_key(item):
    if 'kind' not in item:
        return f'Missed {CLASS_METADATA[item["gt_label"]]["name"]}'
    if item['kind'] == 'Cls':
        return f'{CLASS_METADATA[item["gt_label"]]["name"]} predicted as {CLASS_METADATA[item["pred_label"]]["name"]}'
    if item['kind'] == 'Loc':
        return f'Poorly localised {CLASS_METADATA[item["pred_label"]]["name"]}'
    if item['kind'] == 'Bkg':
        return f'Background detected as {CLASS_METADATA[item["pred_label"]]["name"]}'
    return 'Duplicate detection'

failures = defaultdict(lambda: defaultdict(list))
for name, (recs, missed) in COHORT.items():
    for item in [r for r in recs if r['kind'] != 'TP'] + missed:
        failures[mode_key(item)][name].append(item)

ranked = sorted(failures, key=lambda k: -sum(len(v) for v in failures[k].values()))
TOP_MODES = ranked[:5]
print(pd.DataFrame({name: {m: len(failures[m][name]) for m in TOP_MODES} for name in runners}).to_string())

def crop_example(ax, item, name):
    rgb = load_scan(val_images[item['image_id']])
    boxes = [b for b in (item.get('gt_box'), item.get('pred_box')) if b is not None]
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes); x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    pad = 0.6 * max(x2 - x1, y2 - y1)
    cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    cx2, cy2 = int(min(rgb.shape[1], x2 + pad)), int(min(rgb.shape[0], y2 + pad))
    ax.imshow(rgb[cy1:cy2, cx1:cx2])
    shift = np.array([cx1, cy1, cx1, cy1])
    if item.get('gt_box') is not None:
        draw_boxes(ax, [item['gt_box'] - shift], [item['gt_label']])
    if item.get('pred_box') is not None:
        draw_boxes(ax, [item['pred_box'] - shift], [item['pred_label']], [item['score']], dashed=True)
    ax.set_title(f'{name} | scan {item["image_id"]}' + (f' | IoU {item["iou"]:.2f}' if 'iou' in item else ''), fontsize=9)

if TOP_MODES:
    fig, axes = plt.subplots(len(TOP_MODES), 3, figsize=(15, 4.2 * len(TOP_MODES)), squeeze=False)
    for row, mode in enumerate(TOP_MODES):
        # Highest-confidence false positives, or the largest missed lesions
        examples = [(name, it) for name, items in failures[mode].items() for it in items]
        examples.sort(key=lambda e: -(e[1]['score'] if 'score' in e[1] else
                                      (e[1]['gt_box'][2] - e[1]['gt_box'][0]) * (e[1]['gt_box'][3] - e[1]['gt_box'][1])))
        for col in range(3):
            ax = axes[row, col]
            ax.axis('off')
            if col < len(examples):
                crop_example(ax, examples[col][1], examples[col][0])
        axes[row, 0].text(0, 1.18, f'#{row + 1} {mode}', transform=axes[row, 0].transAxes, fontsize=12, fontweight='bold')
    plt.suptitle('Top 5 Failure Modes — real validation cases (solid = Ground Truth, dashed = prediction)', fontweight='bold')
    plt.tight_layout(); plt.show()
else:
    print('No failures at the operating thresholds.')
"""),
    md(r"""
## 9. Summary generated from the measurements
"""),
    code(r"""
from IPython.display import Markdown

op = {name: prf(*COHORT[name]) for name in runners}
names = list(runners)
lines = ['| Metric | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
for label, fn in [
    ('COCO mAP50', lambda n: f'{COCO_TABLE.loc[n, "mAP50"]:.3f}'),
    ('COCO mAP50:95', lambda n: f'{COCO_TABLE.loc[n, "mAP50_95"]:.3f}'),
    *[(f'AP50 {c}', (lambda key: lambda n: f'{COCO_TABLE.loc[n, key]:.3f}')(f'AP_{c.replace(" ", "")}')) for c in CLASS_NAMES],
    ('Operating threshold (max F1 on val)', lambda n: f'{OPERATING_THRESH[n]:.2f}'),
    ('Precision / Recall / F1 @ operating threshold', lambda n: f'{op[n]["precision"]:.3f} / {op[n]["recall"]:.3f} / {op[n]["f1"]:.3f}'),
    ('Median latency per scan', lambda n: f'{COCO_TABLE.loc[n, "latency_ms_median"]:.0f} ms'),
    ('Most frequent error', lambda n: error_counts.loc[n, KINDS + ['Missed']].idxmax() if error_counts.loc[n, KINDS + ['Missed']].sum() else 'none'),
]:
    lines.append(f'| {label} | ' + ' | '.join(fn(n) for n in names) + ' |')

best = max(names, key=lambda n: COCO_TABLE.loc[n, 'mAP50'])
notes = [f'- Higher COCO mAP50: **{best}** ({COCO_TABLE.loc[best, "mAP50"]:.3f}).']
for n in names:
    # Classes without validation lesions have an undefined (NaN) AP and are left out
    ap = {c: COCO_TABLE.loc[n, f'AP_{c.replace(" ", "")}'] for c in CLASS_NAMES if not np.isnan(COCO_TABLE.loc[n, f'AP_{c.replace(" ", "")}'])}
    if not ap:
        continue
    notes.append(f'- {n}: weakest class {min(ap, key=ap.get)} (AP50 {min(ap.values()):.3f}), strongest {max(ap, key=ap.get)} (AP50 {max(ap.values()):.3f}).')
notes.append(f'- Top failure mode overall: {TOP_MODES[0] if TOP_MODES else "none"}.')
notes += [
    f'- Caveats: one split of {len(val_images)} scans; Periapical Lesion has only '
    f'{sum(1 for a in val_coco["annotations"] if a["category_id"] == 3)} validation lesions, so its AP is noisy; '
    'operating thresholds were tuned on this split.',
]
display(Markdown('\n'.join(lines) + '\n\n' + '\n'.join(notes)))
"""),
]


def build_notebook():
    notebook_data = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out_path = Path("notebooks/dentex_inference_error_analysis.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2, ensure_ascii=False)
    print(f"Built {out_path} with {len(CELLS)} cells.")


if __name__ == "__main__":
    build_notebook()

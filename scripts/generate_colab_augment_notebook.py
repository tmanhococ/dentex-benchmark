"""
scripts/generate_colab_augment_notebook.py
Generates notebooks/04_colab_dentex_augment_run_all.ipynb: DENTEX tooth-swap augmentation (real
polygons, exact FDI) and a baseline-vs-augmented benchmark of three detectors, runnable with Run all.
"""

import json
from pathlib import Path


def create_notebook():
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().split("\n")]})

    def code(text):
        cells.append({
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")],
        })

    md(r"""# 🦷 DENTEX v2: Ghép Răng Thật (Polygon + FDI) & Benchmark 3 Mô Hình

Notebook chạy **Runtime → Run all** từ đầu đến cuối:
1. Nạp file nhãn DENTEX gốc `quadrant-enumeration-disease` từ Drive, **giữ nguyên polygon và số răng FDI** (không đoán).
2. Tạo Tooth Bank: mỗi răng bệnh được cắt theo đúng đường viền polygon.
3. Ghép răng: thay một răng bệnh bằng răng bệnh **cùng số FDI** (hoặc đối xứng) từ ảnh khác; xóa răng cũ theo đường viền, cân sáng theo nền xung quanh, hòa trộn biên.
4. QC trực quan: so sánh trước/sau phóng to từng răng được ghép.
5. Train 3 kiến trúc (Faster R-CNN, HierarchicalNet, DiffusionNet) × (Baseline, Augmented), mỗi lần `EPOCHS` epoch.
6. Đánh giá COCO trên tập validation và lập bảng so sánh.

**An toàn khi mất kết nối:** checkpoint `last.pth`, `best.pth`, `history.json` và log được ghi **thẳng lên Drive sau mỗi epoch**. Nếu Colab ngắt, chỉ cần Run all lại: dữ liệu được sinh lại y hệt (cùng seed) và việc train tiếp tục từ epoch đã lưu.

⚠️ Kết quả của các lần chạy trước (`augmented_data/`, checkpoint cũ) **không hợp lệ** vì lỗi ghép ô vuông xám. Notebook này lưu vào thư mục mới `RUN_TAG`.""")

    md("## ⚙️ Cell 0: Cấu hình")
    code(r"""# Thư mục chứa dữ liệu DENTEX trên Drive (ảnh + file json gốc của tập quadrant-enumeration-disease)
DRIVE_DATA_DIR = '/content/drive/MyDrive/DENTEX'
# Thư mục lưu kết quả trên Drive
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/DENTEX_AUGMENT_OUTPUT'
RUN_TAG = 'v2_realswap'          # kết quả nằm trong DRIVE_OUTPUT_DIR/RUN_TAG

REPO_URL = 'https://github.com/tmanhococ/dentex-benchmark.git'
BRANCH = 'main'

MODELS = ['faster_rcnn', 'hierarchical_net', 'diffusion_net']
DATASETS = ['baseline', 'augmented']
EPOCHS = 5
BATCH_SIZE = 2                   # an toàn cho T4 16GB ở 800x1333 với AMP
GRAD_ACCUM = 2                   # batch hiệu dụng = BATCH_SIZE * GRAD_ACCUM = 4
LEARNING_RATE = 1e-4
EVAL_INTERVAL = 1                # đánh giá val mỗi epoch (chọn best.pth theo val mAP50)
NUM_AUGMENT_PER_IMAGE = 2        # số ảnh ghép sinh ra cho mỗi ảnh train
VAL_SPLIT = 0.2
SEED = 42
FORCE_REGENERATE = False         # True: sinh lại dữ liệu augment dù đã có
QC_PAIRS = 8                     # số cặp trước/sau hiển thị ở bước QC
""")

    md("## 💻 Cell 1: GPU & Google Drive")
    code(r"""import os, sys
from pathlib import Path
import torch

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB | CUDA {torch.version.cuda}')
else:
    print('⚠️ Không có GPU: train sẽ rất chậm. Vào Runtime → Change runtime type → GPU.')

if IN_COLAB:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive')
OUT = Path(DRIVE_OUTPUT_DIR) / RUN_TAG
OUT.mkdir(parents=True, exist_ok=True)
print('Kết quả sẽ lưu tại:', OUT)
""")

    md("## 📥 Cell 2: Lấy mã nguồn mới nhất")
    code(r"""import subprocess

REPO_DIR = Path('/content/dentex-benchmark') if IN_COLAB else Path('.').resolve()
if IN_COLAB:
    if not (REPO_DIR / 'dentex_bench').is_dir():
        subprocess.run(['git', 'clone', '-b', BRANCH, REPO_URL, str(REPO_DIR)], check=True)
    else:
        # Luôn khớp đúng nhánh trên GitHub (bỏ thay đổi cục bộ trên VM)
        subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH], check=True)
        subprocess.run(['git', '-C', str(REPO_DIR), 'reset', '--hard', f'origin/{BRANCH}'], check=True)
    os.chdir(str(REPO_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
print('Thư mục làm việc:', Path.cwd(), '| commit:', commit)
""")

    md("## 📦 Cell 3: Thư viện")
    code(r"""import subprocess
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pycocotools', 'pyyaml', 'tabulate', 'scipy'], check=True)
import cv2
cv2.setNumThreads(0)
print('OK')
""")

    md(r"""## 🔍 Cell 4: Nạp DENTEX gốc (giữ polygon + FDI)

Tìm file json có `categories_3` (nhãn bệnh) — ưu tiên đường dẫn chứa `disease`. Chỉ copy đúng các ảnh mà file json đó liệt kê, và **kiểm tra kích thước từng ảnh khớp với nhãn** (DENTEX có nhiều tập dùng trùng tên `train_0.png`...).""")
    code(r"""import json, shutil
from collections import Counter
from PIL import Image
from dentex_bench.data.parser import convert_dentex_annotations_to_coco

drive_path = Path(DRIVE_DATA_DIR)
if not drive_path.exists():
    raise FileNotFoundError(f'Không thấy {drive_path}. Sửa DRIVE_DATA_DIR ở Cell 0.')

# 1. File nhãn bệnh DENTEX gốc
candidates = []
for j in drive_path.rglob('*.json'):
    if 'instances' in j.name or j.stat().st_size < 1_000:
        continue
    try:
        with open(j, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except Exception:
        continue
    if isinstance(d, dict) and 'categories_3' in d and d.get('annotations'):
        candidates.append(j)
if not candidates:
    raise FileNotFoundError('Không tìm thấy file json DENTEX có categories_3 (vd train_quadrant_enumeration_disease.json) trong DRIVE_DATA_DIR.')
candidates.sort(key=lambda p: (0 if 'disease' in str(p).lower() else 1, -p.stat().st_size))
RAW_JSON = candidates[0]
print('File nhãn:', RAW_JSON)
with open(RAW_JSON, 'r', encoding='utf-8') as f:
    raw = json.load(f)
raw_images = {im['file_name']: im for im in raw['images']}

# 2. Thư mục ảnh: thư mục chứa nhiều ảnh trùng tên nhất, ưu tiên có 'disease'
by_dir = Counter()
for p in drive_path.rglob('*'):
    if p.suffix.lower() in ('.png', '.jpg', '.jpeg') and p.name in raw_images:
        by_dir[p.parent] += 1
if not by_dir:
    raise FileNotFoundError('Không tìm thấy ảnh nào trùng tên với file nhãn.')
IMG_SRC = sorted(by_dir, key=lambda d: (-by_dir[d], 0 if 'disease' in str(d).lower() else 1))[0]
print(f'Thư mục ảnh: {IMG_SRC} ({by_dir[IMG_SRC]}/{len(raw_images)} ảnh)')

# 3. Copy đúng các ảnh đó vào data/raw (làm sạch trước để không lẫn ảnh tập khác)
local_raw, local_proc = Path('data/raw'), Path('data/processed')
if local_raw.exists():
    shutil.rmtree(local_raw)
local_raw.mkdir(parents=True)
local_proc.mkdir(parents=True, exist_ok=True)
missing, mismatched = [], []
for name, im in raw_images.items():
    src = IMG_SRC / name
    if not src.is_file():
        missing.append(name)
        continue
    with Image.open(src) as pim:
        if (pim.width, pim.height) != (im['width'], im['height']):
            mismatched.append((name, pim.size, (im['width'], im['height'])))
            continue
    shutil.copy2(src, local_raw / name)
if mismatched:
    raise ValueError(f'{len(mismatched)} ảnh có kích thước khác nhãn (sai tập?), ví dụ: {mismatched[:3]}')
if missing:
    print(f'⚠️ Thiếu {len(missing)} ảnh, ví dụ {missing[:5]} — các ảnh này sẽ bị loại.')
    raw['images'] = [im for im in raw['images'] if im['file_name'] not in set(missing)]
    keep = {im['id'] for im in raw['images']}
    raw['annotations'] = [a for a in raw['annotations'] if a['image_id'] in keep]
    RAW_JSON_USED = local_proc / 'raw_filtered.json'
    RAW_JSON_USED.write_text(json.dumps(raw), encoding='utf-8')
else:
    RAW_JSON_USED = RAW_JSON

TRAIN_ANN, VAL_ANN = local_proc / 'instances_train.json', local_proc / 'instances_val.json'
split = convert_dentex_annotations_to_coco(RAW_JSON_USED, TRAIN_ANN, VAL_ANN, split_ratio=1.0 - VAL_SPLIT, seed=SEED)
print('Chia tập:', split)

CLASS_NAMES = {1: 'Caries', 2: 'Deep Caries', 3: 'Periapical Lesion', 4: 'Impacted Tooth'}
for name, p in [('train', TRAIN_ANN), ('val', VAL_ANN)]:
    c = json.loads(p.read_text(encoding='utf-8'))
    anns = c['annotations']
    per_cls = Counter(CLASS_NAMES[a['category_id']] for a in anns)
    fdi_cov = sum(1 for a in anns if a.get('fdi')) / max(1, len(anns))
    seg_cov = sum(1 for a in anns if a.get('segmentation')) / max(1, len(anns))
    print(f'{name}: {len(c["images"])} ảnh, {len(anns)} nhãn {dict(per_cls)} | có FDI: {fdi_cov:.1%} | có polygon: {seg_cov:.1%}')
    if fdi_cov < 0.9 or seg_cov < 0.9:
        raise ValueError(f'Tập {name} thiếu FDI/polygon ({fdi_cov:.1%}/{seg_cov:.1%}): file nhãn không phải DENTEX disease gốc.')
""")

    md("### Kiểm tra nhãn bằng mắt: polygon + số FDI vẽ trên 2 ảnh train")
    code(r"""import numpy as np
import matplotlib.pyplot as plt

train_coco = json.loads(TRAIN_ANN.read_text(encoding='utf-8'))
anns_by_img = {}
for a in train_coco['annotations']:
    anns_by_img.setdefault(a['image_id'], []).append(a)
COLORS = {1: (255, 200, 0), 2: (255, 80, 0), 3: (255, 0, 200), 4: (0, 200, 255)}
fig, axes = plt.subplots(1, 2, figsize=(22, 6))
for ax, im in zip(axes, train_coco['images'][:2]):
    img = cv2.cvtColor(cv2.imread(str(local_raw / im['file_name'])), cv2.COLOR_BGR2RGB)
    for a in anns_by_img.get(im['id'], []):
        for poly in a.get('segmentation', []):
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(img, [pts], True, COLORS[a['category_id']], 3)
        x, y = int(a['bbox'][0]), int(a['bbox'][1])
        cv2.putText(img, f"{a.get('fdi')} {CLASS_NAMES[a['category_id']]}", (x, max(30, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS[a['category_id']], 3)
    ax.imshow(img); ax.set_title(im['file_name']); ax.axis('off')
plt.tight_layout(); plt.show()
""")

    md("## 🦷 Cell 5: Tooth Bank (mỗi răng bệnh cắt theo đúng polygon)")
    code(r"""from dentex_bench.augment.bank_builder import build_tooth_bank

BANK_DIR = Path('outputs/augment/bank')
if BANK_DIR.exists():
    shutil.rmtree(BANK_DIR)
bank, bank_report = build_tooth_bank(TRAIN_ANN, local_raw, BANK_DIR, val_ann=VAL_ANN)
print(json.dumps(bank_report, indent=2, ensure_ascii=False))
(OUT / 'bank_report.json').write_text(json.dumps(bank_report, indent=2), encoding='utf-8')

from dentex_bench.augment.io_utils import read_image_lossless
show = bank.entries[:: max(1, len(bank.entries) // 8)][:8]
fig, axes = plt.subplots(1, len(show), figsize=(3 * len(show), 4))
for ax, e in zip(np.atleast_1d(axes), show):
    patch = read_image_lossless(bank.resolve_path(e.patch_path))
    patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB) if patch.ndim == 2 else patch.copy()
    roi = read_image_lossless(bank.resolve_path(e.roi_mask_path))
    contours, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(patch, contours, -1, (255, 0, 0), 2)
    ax.imshow(patch); ax.set_title(f'FDI {e.fdi}\n{",".join(e.labels)}', fontsize=9); ax.axis('off')
plt.tight_layout(); plt.show()
""")

    md("## 🧬 Cell 6: Sinh dữ liệu ghép răng")
    code(r"""from dentex_bench.augment.config import load_config
from dentex_bench.augment.offline import run_offline

aug_cfg = load_config('configs/augment/base.yaml', cli_overrides=[
    f'paths.train_ann={TRAIN_ANN}', f'paths.val_ann={VAL_ANN}', f'paths.img_dir={local_raw}',
    'paths.out_root=outputs/augment', f'global_seed={SEED}',
    f'offline.variants_per_image={NUM_AUGMENT_PER_IMAGE}', 'offline.num_workers=1',
    'output.formats=[coco]', 'output.include_originals=true',
])
aug_cfg.output.overwrite = bool(FORCE_REGENERATE)
RUN_DIR = run_offline(aug_cfg, bank=bank)
AUG_TRAIN_ANN = RUN_DIR / 'dataset' / 'coco' / 'instances_train_aug.json'
aug_coco = json.loads(AUG_TRAIN_ANN.read_text(encoding='utf-8'))
qc_summary = json.loads((RUN_DIR / 'state' / 'qc_summary.json').read_text(encoding='utf-8')) if (RUN_DIR / 'state' / 'qc_summary.json').exists() else {}
n_syn = sum(1 for im in aug_coco['images'] if im['file_name'].startswith('syn_'))
print(f'Tập augmented: {len(aug_coco["images"])} ảnh ({n_syn} ảnh ghép), {len(aug_coco["annotations"])} nhãn')
print('QC:', qc_summary)

# Lưu nhãn, recipes và QC lên Drive (không copy ảnh: có thể sinh lại y hệt bằng cùng seed)
aug_drive = OUT / 'augmented_data'
aug_drive.mkdir(parents=True, exist_ok=True)
shutil.copy2(AUG_TRAIN_ANN, aug_drive / 'instances_train_aug.json')
shutil.copytree(RUN_DIR / 'state', aug_drive / 'state', dirs_exist_ok=True)
shutil.copy2(TRAIN_ANN, aug_drive / 'instances_train.json')
shutil.copy2(VAL_ANN, aug_drive / 'instances_val.json')
print('Đã lưu nhãn/recipes lên', aug_drive)
""")

    md(r"""## 🔬 Cell 7: QC trực quan trước/sau

Mỗi hàng: răng nguồn (từ bank) | vùng răng đích **trước** (viền xanh = răng bị thay) | **sau** khi ghép (viền đỏ = nhãn mới) | toàn ảnh sau. Hãy nhìn kỹ cột 3: nếu còn thấy vệt chéo, ô vuông, hay răng "ma" thì báo lại.""")
    code(r"""recipes = [json.loads(l) for l in (RUN_DIR / 'state' / 'recipes.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
syn_by_file = {im['file_name']: im['id'] for im in aug_coco['images']}
aug_anns_by_img = {}
for a in aug_coco['annotations']:
    aug_anns_by_img.setdefault(a['image_id'], []).append(a)

def to_rgb(x):
    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB) if x.ndim == 2 else x.copy()

pairs = recipes[:: max(1, len(recipes) // QC_PAIRS)][:QC_PAIRS]
fig, axes = plt.subplots(len(pairs), 4, figsize=(20, 4.5 * len(pairs)))
axes = np.atleast_2d(axes)
for row, rec in zip(axes, pairs):
    op = rec['ops'][0]
    syn_name = f"syn_{rec['target_image_id']}_{rec['rng_key'][-1]}.png"
    before = to_rgb(read_image_lossless(local_raw / rec['target_file']))
    after = to_rgb(read_image_lossless(RUN_DIR / 'synthetic' / syn_name))
    x, y, w, h = [int(v) for v in op['target_tooth_ref']['bbox']]
    for poly in op['target_tooth_ref'].get('segmentation') or []:
        cv2.polylines(before, [np.array(poly, dtype=np.int32).reshape(-1, 2)], True, (0, 255, 0), 2)
    for a in aug_anns_by_img.get(syn_by_file.get(syn_name), []):
        if a.get('synthetic'):
            for poly in a.get('segmentation') or []:
                cv2.polylines(after, [np.array(poly, dtype=np.int32).reshape(-1, 2)], True, (255, 0, 0), 2)
    pad = int(0.6 * max(w, h))
    y1, y2, x1, x2 = max(0, y - pad), y + h + pad, max(0, x - pad), x + w + pad
    src = bank.get(op['source_entry_id'])
    row[0].imshow(to_rgb(read_image_lossless(bank.resolve_path(src.patch_path))))
    row[0].set_title(f"Nguồn {src.entry_id} ({','.join(src.labels)})" + (' [lật]' if op['mirror'] else ''), fontsize=9)
    row[1].imshow(before[y1:y2, x1:x2]); row[1].set_title(f"Trước: FDI {op['target_fdi']} ({rec['target_file']})", fontsize=9)
    row[2].imshow(after[y1:y2, x1:x2]); row[2].set_title('Sau khi ghép', fontsize=9)
    small = after.copy(); cv2.rectangle(small, (x, y), (x + w, y + h), (255, 0, 0), 6)
    row[3].imshow(small); row[3].set_title(syn_name, fontsize=9)
    for ax in row:
        ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'qc_before_after.png', dpi=80)
plt.show()
""")

    md(r"""## 🏋️ Cell 8: Train 3 mô hình × (Baseline, Augmented)

Log từng lần train hiện trực tiếp ở đây và được ghi vào `OUT/logs/`. Checkpoint ghi thẳng vào `OUT/checkpoints/<dataset>/<model>/` sau mỗi epoch; lần train đã đủ `EPOCHS` epoch sẽ được bỏ qua. Một lần train lỗi không làm dừng các lần còn lại — lỗi được tổng kết ở cuối cell.

Lưu ý so sánh: với cùng số epoch, tập augmented có nhiều ảnh hơn (gấp ~1 + NUM_AUGMENT_PER_IMAGE lần) nên cũng được train nhiều bước hơn.""")
    code(r"""import time

def run_streaming(cmd, log_path):
    # Stream child output into the notebook and a log file on Drive
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['TQDM_MININTERVAL'] = '5'
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(map(str, cmd))}\n")
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        while True:
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            text = chunk.decode('utf-8', errors='replace')
            sys.stdout.write(text); sys.stdout.flush()
            log.write(text); log.flush()
        return proc.wait()

def epochs_done(run_dir):
    h = run_dir / 'history.json'
    if not h.is_file():
        return 0
    try:
        return len(json.loads(h.read_text(encoding='utf-8')))
    except Exception:
        return 0

CKPT_ROOT = OUT / 'checkpoints'
train_status = {}
for model in MODELS:
    for ds in DATASETS:
        run_dir = CKPT_ROOT / ds / model
        key = f'{ds}/{model}'
        if epochs_done(run_dir) >= EPOCHS:
            print(f'✅ {key}: đã đủ {EPOCHS} epoch, bỏ qua.')
            train_status[key] = 'done'
            continue
        cmd = [sys.executable, '-u', 'scripts/train.py', '--model', model, '--epochs', EPOCHS,
               '--batch-size', BATCH_SIZE, '--grad-accum', GRAD_ACCUM, '--lr', LEARNING_RATE,
               '--eval-interval', EVAL_INTERVAL, '--resume', '--output_dir', CKPT_ROOT / ds]
        if ds == 'augmented':
            cmd += ['--train-ann', AUG_TRAIN_ANN]
        print(f'\n🏋️ {key} (đã có {epochs_done(run_dir)}/{EPOCHS} epoch)')
        t0 = time.time()
        rc = run_streaming(cmd, OUT / 'logs' / f'train_{ds}_{model}.log')
        train_status[key] = 'done' if rc == 0 and epochs_done(run_dir) >= EPOCHS else f'FAILED (exit {rc})'
        print(f'⏱️ {key}: {(time.time() - t0) / 60:.1f} phút → {train_status[key]}')

print('\n=== Tổng kết train ===')
for k, v in train_status.items():
    print(f'{k:32s} {v}')
(OUT / 'train_status.json').write_text(json.dumps(train_status, indent=2), encoding='utf-8')
""")

    md(r"""## 📊 Cell 9: Đánh giá trên validation & bảng so sánh

Mọi con số dưới đây đọc từ file kết quả của `scripts/evaluate.py` (COCO eval) — không có số nào được điền tay. Lưu ý: `best.pth` được chọn theo val mAP50, rồi lại báo cáo trên chính tập val đó, nên số liệu hơi lạc quan; dùng để **so sánh tương đối** baseline vs augmented.""")
    code(r"""import csv
import pandas as pd

rows = []
for model in MODELS:
    for ds in DATASETS:
        run_dir = CKPT_ROOT / ds / model
        ckpt = run_dir / 'best.pth'
        if not ckpt.is_file():
            print(f'⚠️ {ds}/{model}: không có best.pth, bỏ qua.')
            continue
        eval_dir = OUT / 'eval' / ds / model
        rc = run_streaming([sys.executable, '-u', 'scripts/evaluate.py', '--model', model, '--checkpoint', ckpt,
                            '--val_ann', VAL_ANN, '--img_dir', local_raw, '--output_dir', eval_dir],
                           OUT / 'logs' / f'eval_{ds}_{model}.log')
        metrics_csv = eval_dir / 'metrics.csv'
        if rc != 0 or not metrics_csv.is_file():
            print(f'❌ {ds}/{model}: đánh giá lỗi (exit {rc})')
            continue
        with open(metrics_csv, newline='', encoding='utf-8') as f:
            m = next(csv.DictReader(f))
        hist = json.loads((run_dir / 'history.json').read_text(encoding='utf-8'))
        best = max(hist, key=lambda h: h.get('val_mAP50', float('-inf')))
        row = {'model': model, 'data': ds, 'best_epoch': best['epoch'], 'epochs': len(hist)}
        for k in ['mAP50', 'mAP50_95', 'AP_Caries', 'AP_DeepCaries', 'AP_PeriapicalLesion', 'AP_ImpactedTooth']:
            row[k] = float(m[k])
        rows.append(row)

if rows:
    table = pd.DataFrame(rows).sort_values(['model', 'data']).reset_index(drop=True)
    table.to_csv(OUT / 'results_table.csv', index=False)
    display(table.style.format({c: '{:.4f}' for c in table.columns if c.startswith(('mAP', 'AP_'))}))
    print('Đã lưu', OUT / 'results_table.csv')
else:
    print('Chưa có mô hình nào được đánh giá.')

# Đường cong val mAP50 theo epoch
fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 4), squeeze=False)
for ax, model in zip(axes[0], MODELS):
    for ds, style in zip(DATASETS, ['r-o', 'g-s']):
        hp = CKPT_ROOT / ds / model / 'history.json'
        if hp.is_file():
            hist = json.loads(hp.read_text(encoding='utf-8'))
            ax.plot([h['epoch'] for h in hist], [h.get('val_mAP50', float('nan')) for h in hist], style, label=ds)
    ax.set_title(model); ax.set_xlabel('epoch'); ax.set_ylabel('val mAP50'); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(OUT / 'val_map50_curves.png', dpi=100); plt.show()
""")

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out_file = Path(__file__).resolve().parent.parent / "notebooks" / "04_colab_dentex_augment_run_all.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook at: {out_file}")


if __name__ == "__main__":
    create_notebook()

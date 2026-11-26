"""
scripts/generate_kaggle_augment_notebook.py
Generates notebooks/05_kaggle_dentex_augment_run_all.ipynb: the Kaggle version of notebook 04.
DENTEX is read from /kaggle/input, tooth-swap augmentation (real polygons, exact FDI) is generated,
three detectors are trained on baseline and augmented data and evaluated, all in one notebook.
Checkpoints go to /kaggle/working after every epoch, a wall-clock budget keeps the run inside
Kaggle's 12 h limit, and a later version resumes from the output of a previous one.
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

    md(r"""# 🦷 DENTEX v2 trên Kaggle: Ghép Răng Thật (Polygon + FDI) → Train → Đánh giá

Một notebook làm hết: đọc DENTEX từ `/kaggle/input` → tạo Tooth Bank → sinh dữ liệu ghép răng → QC trước/sau → train 3 mô hình × (Baseline, Augmented) → đánh giá COCO và lập bảng so sánh.

### Cách chạy (đọc 1 lần)
1. **Add Input**: thêm dataset DENTEX có file nhãn gốc `train_quadrant_enumeration_disease.json` và ảnh của nó.
2. **Settings** (thanh bên phải): *Accelerator* = **GPU T4 x2** (đừng chọn P100: bản PyTorch mới không còn chạy trên P100); *Internet* = **On**; *Persistence* = **Files only** nếu có tùy chọn này.
3. Bấm **Save Version → Save & Run All (Commit)**. Notebook chạy nền tối đa 12 giờ, **tắt trình duyệt cũng không sao**. Kết quả nằm ở tab *Output* của version đó.
   (Chạy kiểu tương tác bằng nút ▶ Run All cũng được, nhưng phiên sẽ chết khi bạn rời trang lâu.)

### An toàn khi bị die / hết giờ
- `last.pth`, `best.pth`, `history.json` được ghi vào `/kaggle/working/<RUN_TAG>` **sau mỗi epoch**.
- `TIME_BUDGET_HOURS` (mặc định 11h): train tự dừng sạch trước khi chạm giới hạn 12h, dành thời gian cho bước đánh giá, để version luôn kết thúc bình thường và lưu được output.
- **Chạy tiếp ở version sau**: *Add Input → Your Work / Notebooks →* chọn chính notebook này (output của version trước), rồi Save & Run All lần nữa. Cell 1 tự tìm `<RUN_TAG>/checkpoints` trong `/kaggle/input`, copy về và train tiếp từ epoch đã lưu. Dữ liệu augment được sinh lại y hệt (cùng seed).
- Nếu kernel restart giữa chừng trong cùng một phiên: chạy lại từ đầu, các bước đã xong được bỏ qua.""")

    md("## ⚙️ Cell 0: Cấu hình")
    code(r"""DATA_ROOT = '/kaggle/input'          # nơi Kaggle gắn các dataset (tự tìm file nhãn DENTEX trong đây)
OUTPUT_ROOT = '/kaggle/working'      # được lưu thành Output của version
RUN_TAG = 'v2_realswap'              # kết quả nằm trong OUTPUT_ROOT/RUN_TAG
WORK_DIR = '/tmp/dentex_work'        # mã nguồn, ảnh copy và ảnh augment (lớn, không cần lưu: sinh lại được)

REPO_URL = 'https://github.com/tmanhococ/dentex-benchmark.git'
BRANCH = 'main'

MODELS = ['faster_rcnn', 'hierarchical_net', 'diffusion_net']
DATASETS = ['baseline', 'augmented']
EPOCHS = 5
BATCH_SIZE = 2                       # an toàn cho T4 16GB ở 800x1333 với AMP
GRAD_ACCUM = 2                       # batch hiệu dụng = BATCH_SIZE * GRAD_ACCUM = 4
LEARNING_RATE = 1e-4
EVAL_INTERVAL = 1                    # đánh giá val mỗi epoch (chọn best.pth theo val mAP50)
NUM_AUGMENT_PER_IMAGE = 2            # số ảnh ghép sinh ra cho mỗi ảnh train
VAL_SPLIT = 0.2
SEED = 42
QC_PAIRS = 8                         # số cặp trước/sau hiển thị ở bước QC

TIME_BUDGET_HOURS = 11.0             # Kaggle giới hạn 12h/phiên; train dừng sạch trước mốc này
EVAL_RESERVE_MIN = 30                # phút giữ lại cho bước đánh giá cuối
""")

    md("## 💻 Cell 1: GPU, thư mục output, khôi phục checkpoint từ version trước")
    code(r"""import os, sys, json, time, shutil, socket
from pathlib import Path
import torch

NB_START = time.time()
IN_KAGGLE = Path('/kaggle/working').is_dir()

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB | CUDA {torch.version.cuda} | torch {torch.__version__}')
    try:
        (torch.ones(8, device='cuda') * 2).sum().item()
    except Exception as e:
        raise RuntimeError(f'GPU {props.name} không chạy được với torch {torch.__version__} ({e}). '
                           'Vào Settings → Accelerator → chọn GPU T4 x2 rồi chạy lại.')
    if 'P100' in props.name:
        print('⚠️ Đang dùng P100: nếu gặp lỗi "no kernel image", đổi Accelerator sang GPU T4 x2.')
else:
    print('⚠️ Không có GPU: train sẽ rất chậm. Settings → Accelerator → GPU T4 x2.')

OUT = Path(OUTPUT_ROOT) / RUN_TAG
OUT.mkdir(parents=True, exist_ok=True)
print('Kết quả lưu tại:', OUT)

def history_len(run_dir):
    h = Path(run_dir) / 'history.json'
    try:
        return len(json.loads(h.read_text(encoding='utf-8'))) if h.is_file() else 0
    except Exception:
        return 0

def find_previous_outputs(root, tag, max_depth=5):
    # Output của version trước được gắn vào /kaggle/input/<...>/<RUN_TAG>/
    found, root = [], Path(root)
    if not root.is_dir():
        return found
    for cur, dirs, _ in os.walk(root):
        depth = len(Path(cur).relative_to(root).parts)
        if depth >= max_depth:
            dirs[:] = []
            continue
        if Path(cur).name == tag and (Path(cur) / 'checkpoints').is_dir():
            found.append(Path(cur))
            dirs[:] = []
    return found

PREVIOUS = [p for p in find_previous_outputs(DATA_ROOT, RUN_TAG) if p.resolve() != OUT.resolve()]
restored = []
for prev in PREVIOUS:
    for run_dir in sorted((prev / 'checkpoints').glob('*/*')):
        if not (run_dir / 'last.pth').is_file():
            continue
        ds, model = run_dir.parent.name, run_dir.name
        dst = OUT / 'checkpoints' / ds / model
        if history_len(run_dir) > history_len(dst):
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(run_dir, dst)
            restored.append(f'{ds}/{model}: {history_len(dst)} epoch (từ {prev})')
    if (prev / 'logs').is_dir():
        shutil.copytree(prev / 'logs', OUT / 'logs', dirs_exist_ok=True)
print('Khôi phục checkpoint:', *(restored or ['không có (chạy mới từ đầu)']), sep='\n  ')
""")

    md("## 📥 Cell 2: Lấy mã nguồn mới nhất (cần Internet = On)")
    code(r"""import subprocess

def internet_ok(host='github.com', port=443, timeout=5):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False

if IN_KAGGLE:
    if not internet_ok():
        raise RuntimeError('Không có Internet. Settings → Internet → On (cần xác minh số điện thoại tài khoản Kaggle), rồi chạy lại.')
    REPO_DIR = Path(WORK_DIR) / 'dentex-benchmark'
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not (REPO_DIR / 'dentex_bench').is_dir():
        subprocess.run(['git', 'clone', '-q', '-b', BRANCH, REPO_URL, str(REPO_DIR)], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', '-q', 'origin', BRANCH], check=True)
        subprocess.run(['git', '-C', str(REPO_DIR), 'reset', '-q', '--hard', f'origin/{BRANCH}'], check=True)
    os.chdir(str(REPO_DIR))
else:
    REPO_DIR = Path('.').resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
print('Thư mục làm việc:', Path.cwd(), '| commit:', commit or '(không phải git repo)')
(OUT / 'code_commit.txt').write_text(commit + '\n', encoding='utf-8')
""")

    md("## 📦 Cell 3: Thư viện")
    code(r"""if IN_KAGGLE:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pycocotools', 'pyyaml', 'tabulate', 'scipy'], check=True)
import cv2
cv2.setNumThreads(0)
from tqdm import tqdm
print('OK | opencv', cv2.__version__)
""")

    md(r"""## 🔍 Cell 4: Tìm & nạp DENTEX gốc trong `/kaggle/input` (giữ polygon + FDI)

Tìm file json có `categories_3` (nhãn bệnh), ưu tiên đường dẫn chứa `disease`. Chỉ copy đúng các ảnh mà file json liệt kê, và **kiểm tra kích thước từng ảnh khớp với nhãn** (DENTEX có nhiều tập trùng tên `train_0.png`...). Output của version trước (thư mục `RUN_TAG`) được bỏ qua khi tìm.""")
    code(r"""from collections import Counter
from PIL import Image
from dentex_bench.data.parser import convert_dentex_annotations_to_coco

data_root = Path(DATA_ROOT)
if not data_root.exists():
    raise FileNotFoundError(f'Không thấy {data_root}. Hãy Add Input dataset DENTEX.')

def skip(p):
    return RUN_TAG in p.parts

# 1. File nhãn bệnh DENTEX gốc
candidates, seen_json = [], []
for j in data_root.rglob('*.json'):
    if skip(j) or 'instances' in j.name or j.stat().st_size < 1_000:
        continue
    seen_json.append(str(j))
    try:
        with open(j, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except Exception:
        continue
    if isinstance(d, dict) and 'categories_3' in d and d.get('annotations'):
        candidates.append(j)
if not candidates:
    raise FileNotFoundError('Không tìm thấy file json DENTEX có categories_3 (vd train_quadrant_enumeration_disease.json) '
                            f'trong {data_root}. Các json đã thấy: {seen_json[:10]}')
candidates.sort(key=lambda p: (0 if 'disease' in str(p).lower() else 1, -p.stat().st_size))
RAW_JSON = candidates[0]
print('File nhãn:', RAW_JSON)
with open(RAW_JSON, 'r', encoding='utf-8') as f:
    raw = json.load(f)
raw_images = {im['file_name']: im for im in raw['images']}

# 2. Thư mục ảnh: thư mục chứa nhiều ảnh trùng tên nhất, ưu tiên có 'disease'
by_dir = Counter()
for p in data_root.rglob('*'):
    if p.suffix.lower() in ('.png', '.jpg', '.jpeg') and p.name in raw_images and not skip(p):
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
for name, im in tqdm(raw_images.items(), desc='copy ảnh', file=sys.stdout, mininterval=10, ncols=100):
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
shutil.copy2(TRAIN_ANN, OUT / 'instances_train.json')
shutil.copy2(VAL_ANN, OUT / 'instances_val.json')
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
plt.tight_layout(); plt.savefig(OUT / 'labels_check.png', dpi=60); plt.show()
""")

    md("## 🦷 Cell 5: Tooth Bank (mỗi răng bệnh cắt theo đúng polygon)")
    code(r"""from dentex_bench.augment.bank_builder import build_tooth_bank
from dentex_bench.augment.io_utils import read_image_lossless

BANK_DIR = Path('outputs/augment/bank')
if BANK_DIR.exists():
    shutil.rmtree(BANK_DIR)
t0 = time.time()
bank, bank_report = build_tooth_bank(TRAIN_ANN, local_raw, BANK_DIR, val_ann=VAL_ANN)
print(json.dumps(bank_report, indent=2, ensure_ascii=False))
print(f'⏱️ Tooth bank: {time.time() - t0:.0f}s')
(OUT / 'bank_report.json').write_text(json.dumps(bank_report, indent=2), encoding='utf-8')

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

    md(r"""## 🧬 Cell 6: Sinh dữ liệu ghép răng (có thanh tiến trình)

Tiến độ được ghi sau mỗi ảnh; nếu kernel restart, chạy lại cell này sẽ tiếp tục từ ảnh dở dang. Ảnh augment nằm trong `WORK_DIR` (không lưu vào Output vì rất nặng và sinh lại y hệt được); nhãn, recipes và QC được lưu vào `OUT/augmented_data`.""")
    code(r"""from dentex_bench.augment.config import load_config
from dentex_bench.augment.offline import run_offline

aug_cfg = load_config('configs/augment/base.yaml', cli_overrides=[
    f'paths.train_ann={TRAIN_ANN}', f'paths.val_ann={VAL_ANN}', f'paths.img_dir={local_raw}',
    'paths.out_root=outputs/augment', f'global_seed={SEED}',
    f'offline.variants_per_image={NUM_AUGMENT_PER_IMAGE}', 'offline.num_workers=1',
    'output.formats=[coco]', 'output.include_originals=true',
])
aug_cfg.output.overwrite = False
t0 = time.time()
RUN_DIR = run_offline(aug_cfg, bank=bank)
print(f'⏱️ Augment: {(time.time() - t0) / 60:.1f} phút')
AUG_TRAIN_ANN = RUN_DIR / 'dataset' / 'coco' / 'instances_train_aug.json'
aug_coco = json.loads(AUG_TRAIN_ANN.read_text(encoding='utf-8'))
qc_path = RUN_DIR / 'state' / 'qc_summary.json'
qc_summary = json.loads(qc_path.read_text(encoding='utf-8')) if qc_path.exists() else {}
n_syn = sum(1 for im in aug_coco['images'] if im['file_name'].startswith('syn_'))
print(f'Tập augmented: {len(aug_coco["images"])} ảnh ({n_syn} ảnh ghép), {len(aug_coco["annotations"])} nhãn')
print('QC:', qc_summary)

aug_out = OUT / 'augmented_data'
aug_out.mkdir(parents=True, exist_ok=True)
shutil.copy2(AUG_TRAIN_ANN, aug_out / 'instances_train_aug.json')
shutil.copytree(RUN_DIR / 'state', aug_out / 'state', dirs_exist_ok=True)
print('Đã lưu nhãn/recipes vào', aug_out)
""")

    md(r"""## 🔬 Cell 7: QC trực quan trước/sau

Mỗi hàng: răng nguồn (từ bank) | vùng răng đích **trước** (viền xanh = răng bị thay) | **sau** khi ghép (viền đỏ = nhãn mới) | toàn ảnh sau. Ảnh được lưu thành `OUT/qc_before_after.png`.""")
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

    md(r"""## 🏋️ Cell 8: Train 3 mô hình × (Baseline, Augmented) — có checkpoint & giới hạn thời gian

- Log hiện trực tiếp (tqdm theo batch) và ghi vào `OUT/logs/`. Checkpoint ghi vào `OUT/checkpoints/<dataset>/<model>/` sau mỗi epoch.
- Lần train đã đủ `EPOCHS` epoch được bỏ qua; lần dở dang được train tiếp (`--resume`).
- Mỗi lần train nhận phần thời gian còn lại của `TIME_BUDGET_HOURS` (trừ `EVAL_RESERVE_MIN`). Nếu epoch kế tiếp không kịp xong, nó dừng sạch và được đánh dấu `partial`: chạy version sau (gắn output version này làm input) để train nốt.
- Một lần train lỗi không làm dừng các lần còn lại.""")
    code(r"""def run_streaming(cmd, log_path):
    # Stream child output into the notebook and a log file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['TQDM_MININTERVAL'] = '10'
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

def read_history(run_dir):
    h = run_dir / 'history.json'
    try:
        return json.loads(h.read_text(encoding='utf-8')) if h.is_file() else []
    except Exception:
        return []

def minutes_left():
    return TIME_BUDGET_HOURS * 60 - (time.time() - NB_START) / 60 - EVAL_RESERVE_MIN

def epoch_minutes_estimate(ds):
    # Slowest epoch seen so far on this dataset (any model), used before a run has its own timing
    secs = [h['seconds'] for m in MODELS for h in read_history(CKPT_ROOT / ds / m) if 'seconds' in h]
    return max(secs) / 60 if secs else 0.0

CKPT_ROOT = OUT / 'checkpoints'
status_file = OUT / 'train_status.json'
train_status = {}
runs = [(model, ds) for model in MODELS for ds in DATASETS]
for i, (model, ds) in enumerate(runs, 1):
    run_dir = CKPT_ROOT / ds / model
    key = f'{ds}/{model}'
    done = len(read_history(run_dir))
    left = minutes_left()
    print(f'\n[{i}/{len(runs)}] {key}: đã có {done}/{EPOCHS} epoch | còn {left:.0f} phút ngân sách')
    if done >= EPOCHS:
        train_status[key] = 'done'
        print('✅ đã đủ, bỏ qua.')
    elif left < max(5.0, 1.1 * epoch_minutes_estimate(ds)):
        train_status[key] = f'partial {done}/{EPOCHS} (hết thời gian, chạy version sau để train tiếp)'
        print('⏸️ không đủ thời gian cho 1 epoch nữa, để version sau.')
    else:
        cmd = [sys.executable, '-u', 'scripts/train.py', '--model', model, '--epochs', EPOCHS,
               '--batch-size', BATCH_SIZE, '--grad-accum', GRAD_ACCUM, '--lr', LEARNING_RATE,
               '--eval-interval', EVAL_INTERVAL, '--resume', '--output_dir', CKPT_ROOT / ds,
               '--time-limit-min', f'{left:.1f}']
        if ds == 'augmented':
            cmd += ['--train-ann', AUG_TRAIN_ANN]
        t0 = time.time()
        rc = run_streaming(cmd, OUT / 'logs' / f'train_{ds}_{model}.log')
        done = len(read_history(run_dir))
        if rc != 0:
            train_status[key] = f'FAILED (exit {rc}, {done}/{EPOCHS} epoch) - xem logs/train_{ds}_{model}.log'
        elif done >= EPOCHS:
            train_status[key] = 'done'
        else:
            train_status[key] = f'partial {done}/{EPOCHS} (hết thời gian, chạy version sau để train tiếp)'
        print(f'⏱️ {key}: {(time.time() - t0) / 60:.1f} phút → {train_status[key]}')
    status_file.write_text(json.dumps(train_status, indent=2, ensure_ascii=False), encoding='utf-8')

print('\n=== Tổng kết train ===')
for k, v in train_status.items():
    print(f'{k:32s} {v}')
""")

    md(r"""## 📊 Cell 9: Đánh giá trên validation & bảng so sánh

Mọi con số đọc từ kết quả `scripts/evaluate.py` (COCO eval), không có số nào điền tay. Lần train `partial` vẫn được đánh giá (cột `epochs` cho biết đã train bao nhiêu). `best.pth` được chọn theo val mAP50 rồi báo cáo trên chính tập val đó, nên số liệu hơi lạc quan; dùng để **so sánh tương đối** baseline vs augmented.""")
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
        hist = read_history(run_dir)
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

fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 4), squeeze=False)
for ax, model in zip(axes[0], MODELS):
    for ds, style in zip(DATASETS, ['r-o', 'g-s']):
        hist = read_history(CKPT_ROOT / ds / model)
        if hist:
            ax.plot([h['epoch'] for h in hist], [h.get('val_mAP50', float('nan')) for h in hist], style, label=ds)
    ax.set_title(model); ax.set_xlabel('epoch'); ax.set_ylabel('val mAP50'); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(OUT / 'val_map50_curves.png', dpi=100); plt.show()
""")

    md(r"""## 📦 Cell 10: Gói kết quả nhẹ để tải về

`results_small.zip` chứa bảng kết quả, ảnh QC, đường cong, trạng thái train, history và log (không có file `.pth`). Tải ở tab **Output** của version.""")
    code(r"""import zipfile

small = Path(OUTPUT_ROOT) / 'results_small.zip'
with zipfile.ZipFile(small, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.suffix != '.pth':
            z.write(p, p.relative_to(OUT.parent))
total_gb = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) / 1024**3
print(f'{small} ({small.stat().st_size / 1024**2:.1f} MB) | toàn bộ {OUT}: {total_gb:.2f} GB')
print(f'Tổng thời gian notebook: {(time.time() - NB_START) / 3600:.2f} giờ')
print(json.dumps(json.loads(status_file.read_text(encoding='utf-8')) if status_file.exists() else {}, indent=2, ensure_ascii=False))
""")

    nb = {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True,
                       "language": "python", "sourceType": "notebook"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out_file = Path(__file__).resolve().parent.parent / "notebooks" / "05_kaggle_dentex_augment_run_all.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook at: {out_file}")


if __name__ == "__main__":
    create_notebook()

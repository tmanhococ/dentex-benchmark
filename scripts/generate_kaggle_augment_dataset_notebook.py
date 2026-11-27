"""
scripts/generate_kaggle_augment_dataset_notebook.py
Generates notebooks/06_kaggle_dentex_augment_dataset_export.ipynb: the data half of notebook 05
(load DENTEX, train/val split, tooth bank, tooth-swap augmentation, QC) with no training. Every
artefact is written to /kaggle/working so the version's Output can be turned into a Kaggle Dataset.

The data, bank, augment and QC cells are taken verbatim from generate_kaggle_augment_notebook.py,
so the augmented data is identical to what notebook 05 trains on (same code, same seed).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_kaggle_augment_notebook import build_cells as build_training_cells  # noqa: E402


def _src(cell):
    return "".join(cell["source"])


def _shared_cells():
    """Cells of notebook 05 from 'Cell 2' (clone) through the QC code cell, unchanged."""
    cells = build_training_cells()
    start = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and "Cell 2:" in _src(c))
    end = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "qc_before_after.png" in _src(c))
    return cells[start:end + 1]


def create_notebook():
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().split("\n")]})

    def code(text):
        cells.append({
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")],
        })

    md(r"""# 🦷 DENTEX v2: Xuất bộ dữ liệu Augmented (không train) để tạo Kaggle Dataset

Notebook này chạy **đúng phần dữ liệu của notebook 05** (cùng mã, cùng seed nên ra dữ liệu y hệt): nạp DENTEX gốc → chia train/val → Tooth Bank → ghép răng thật (polygon + FDI) → QC. Sau đó **lưu toàn bộ vào Output** và dừng, không train.

### Cách chạy
1. **Add Input**: dataset DENTEX có `train_quadrant_enumeration_disease.json` và ảnh.
2. **Settings**: *Internet* = **On** (để clone code). Không cần GPU: Accelerator = **None** là đủ (augment chạy trên CPU, khoảng 40 phút).
3. **Save Version → Save & Run All (Commit)**.
4. Khi xong: mở version → tab **Output** → **New Dataset** (hoặc *Create Dataset*) để biến Output thành Kaggle Dataset.

### Output
Output chỉ có `<RUN_TAG>.zip` (toàn bộ dataset) và `<RUN_TAG>_info/` (README, dataset_info, ảnh QC để xem nhanh). Kaggle có thể tự giải nén zip khi tạo Dataset; nếu không, notebook dùng dataset chỉ cần giải nén vào `/tmp`:
```python
import zipfile; zipfile.ZipFile('/kaggle/input/<dataset>/dentex_v2_augmented.zip').extractall('/tmp')
```

### Nội dung bên trong zip (`<RUN_TAG>/`)
```
images/train/          ảnh gốc của tập train
images/val/            ảnh gốc của tập val (không bao giờ bị augment)
images/synthetic/      ảnh đã ghép răng syn_<image_id>_<variant>.png
annotations/instances_train.json       nhãn train gốc (COCO, có fdi + polygon)
annotations/instances_train_aug.json   nhãn train gốc + ảnh ghép (dùng để train augmented)
annotations/instances_val.json         nhãn val
tooth_bank/            patch, mask, roi mask từng răng bệnh + bank.json
augment_state/         recipes.jsonl (tái lập từng ảnh ghép), manifest, qc_summary
qc_before_after.png, labels_check.png, bank_report.json, dataset_info.json, README.md
```
Mỗi `file_name` trong file nhãn là tên file trần; ảnh nằm ở một trong ba thư mục con của `images/` (loader của repo tìm đệ quy theo tên).""")

    md("## ⚙️ Cell 0: Cấu hình")
    code(r"""DATA_ROOT = '/kaggle/input'          # nơi Kaggle gắn các dataset (tự tìm file nhãn DENTEX trong đây)
OUTPUT_ROOT = '/kaggle/working'      # được lưu thành Output của version
RUN_TAG = 'dentex_v2_augmented'      # tên thư mục dataset trong OUTPUT_ROOT
WORK_DIR = '/tmp/dentex_work'        # mã nguồn và file tạm

REPO_URL = 'https://github.com/tmanhococ/dentex-benchmark.git'
BRANCH = 'main'

NUM_AUGMENT_PER_IMAGE = 2            # số ảnh ghép sinh ra cho mỗi ảnh train (giống notebook 05)
VAL_SPLIT = 0.2                      # giống notebook 05 để ra đúng cùng tập train/val
SEED = 42
QC_PAIRS = 8                         # số cặp trước/sau hiển thị ở bước QC
ZIP_OUTPUT = True                    # True: nén cả dataset thành 1 file zip rồi xoá thư mục (Kaggle không lưu trọn hàng nghìn file)
""")

    md("## 💻 Cell 1: Môi trường & thư mục output")
    code(r"""import os, sys, json, time, shutil, socket
from pathlib import Path

NB_START = time.time()
IN_KAGGLE = Path('/kaggle/working').is_dir()
OUT = Path(OUTPUT_ROOT) / RUN_TAG
if OUT.exists():
    shutil.rmtree(OUT)  # luôn xuất lại từ đầu để dataset không lẫn file cũ
OUT.mkdir(parents=True)
print('Dataset sẽ lưu tại:', OUT)
""")

    cells.extend(_shared_cells())

    md(r"""## 💾 Cell 8: Ghi bộ dữ liệu vào Output

Copy ảnh (có thanh tiến trình), file nhãn, tooth bank và recipes vào `OUT`, rồi **kiểm tra mọi ảnh mà file nhãn nhắc tới đều có mặt**.""")
    code(r"""from collections import Counter

IMG_OUT = OUT / 'images'
ANN_OUT = OUT / 'annotations'
for d in [IMG_OUT / 'train', IMG_OUT / 'val', IMG_OUT / 'synthetic', ANN_OUT]:
    d.mkdir(parents=True, exist_ok=True)

def copy_many(names, src_dir, dst_dir, desc):
    for name in tqdm(names, desc=desc, file=sys.stdout, mininterval=10, ncols=100):
        src = Path(src_dir) / name
        if not src.is_file():
            raise FileNotFoundError(f'{desc}: thiếu {src}')
        shutil.copy2(src, Path(dst_dir) / name)

train_orig = json.loads(TRAIN_ANN.read_text(encoding='utf-8'))
val_coco = json.loads(VAL_ANN.read_text(encoding='utf-8'))
syn_names = [im['file_name'] for im in aug_coco['images'] if im['file_name'].startswith('syn_')]
copy_many([im['file_name'] for im in train_orig['images']], local_raw, IMG_OUT / 'train', 'ảnh train')
copy_many([im['file_name'] for im in val_coco['images']], local_raw, IMG_OUT / 'val', 'ảnh val')
copy_many(syn_names, RUN_DIR / 'synthetic', IMG_OUT / 'synthetic', 'ảnh ghép')

shutil.copy2(TRAIN_ANN, ANN_OUT / 'instances_train.json')
shutil.copy2(AUG_TRAIN_ANN, ANN_OUT / 'instances_train_aug.json')
shutil.copy2(VAL_ANN, ANN_OUT / 'instances_val.json')
shutil.copytree(BANK_DIR, OUT / 'tooth_bank')
shutil.copytree(RUN_DIR / 'state', OUT / 'augment_state')
# Bản sao tạm do các cell dùng chung tạo ra; nội dung đã nằm trong annotations/ và augment_state/
shutil.rmtree(OUT / 'augmented_data', ignore_errors=True)
for f in ['instances_train.json', 'instances_val.json']:
    (OUT / f).unlink(missing_ok=True)

# Kiểm tra: mọi ảnh trong nhãn đều có file, và train/val không trùng ảnh
on_disk = {p.name for p in IMG_OUT.rglob('*') if p.is_file()}
for ann_name in ['instances_train.json', 'instances_train_aug.json', 'instances_val.json']:
    c = json.loads((ANN_OUT / ann_name).read_text(encoding='utf-8'))
    lost = [im['file_name'] for im in c['images'] if im['file_name'] not in on_disk]
    if lost:
        raise FileNotFoundError(f'{ann_name}: {len(lost)} ảnh không có trong images/, ví dụ {lost[:3]}')
overlap = {im['file_name'] for im in train_orig['images']} & {im['file_name'] for im in val_coco['images']}
if overlap:
    raise ValueError(f'Train và val trùng {len(overlap)} ảnh: {sorted(overlap)[:3]}')
print('✅ Mọi ảnh trong 3 file nhãn đều có mặt; train/val không trùng ảnh.')
""")

    md("## 🧾 Cell 9: Thông tin dataset, README, kiểm tra dung lượng")
    code(r"""CLASS_NAMES = {1: 'Caries', 2: 'Deep Caries', 3: 'Periapical Lesion', 4: 'Impacted Tooth'}

def summarize(c):
    return {'images': len(c['images']), 'annotations': len(c['annotations']),
            'per_class': dict(Counter(CLASS_NAMES[a['category_id']] for a in c['annotations']))}

def dir_size_mb(d):
    return round(sum(p.stat().st_size for p in Path(d).rglob('*') if p.is_file()) / 1024**2, 1)

info = {
    'source_label_file': str(RAW_JSON),
    'code_commit': commit,
    'seed': SEED, 'val_split': VAL_SPLIT, 'variants_per_image': NUM_AUGMENT_PER_IMAGE,
    'splits': {'train': summarize(train_orig), 'train_aug': summarize(aug_coco), 'val': summarize(val_coco)},
    'synthetic_images': len(syn_names),
    'tooth_bank_entries': len(bank.entries),
    'augment_qc': qc_summary,
    'size_mb': {d: dir_size_mb(OUT / d) for d in ['images/train', 'images/val', 'images/synthetic', 'annotations', 'tooth_bank', 'augment_state']},
    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
}
(OUT / 'dataset_info.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')

readme = f'''# DENTEX v2 tooth-swap augmented dataset

Built from `{RAW_JSON.name}` (DENTEX quadrant-enumeration-disease) with dentex-benchmark commit `{commit}`,
seed {SEED}, train/val split {1 - VAL_SPLIT:.0%}/{VAL_SPLIT:.0%}, {NUM_AUGMENT_PER_IMAGE} synthetic variants per train image.

Each synthetic image replaces one diseased tooth with a diseased tooth of the same FDI number (or its mirror
counterpart) cut along its DENTEX polygon from another train image. Validation images are never augmented.

| split | images | annotations |
|---|---|---|
| train (original) | {info['splits']['train']['images']} | {info['splits']['train']['annotations']} |
| train_aug (original + synthetic) | {info['splits']['train_aug']['images']} | {info['splits']['train_aug']['annotations']} |
| val | {info['splits']['val']['images']} | {info['splits']['val']['annotations']} |

Layout: `images/{{train,val,synthetic}}/`, `annotations/instances_{{train,train_aug,val}}.json` (COCO, category ids
1 Caries, 2 Deep Caries, 3 Periapical Lesion, 4 Impacted Tooth; each annotation keeps `fdi` and `segmentation`),
`tooth_bank/` (per-tooth patches and masks), `augment_state/recipes.jsonl` (exact parameters of every synthetic image).
`file_name` is a bare name; look images up under `images/` recursively.
'''
(OUT / 'README.md').write_text(readme, encoding='utf-8')

n_files = sum(1 for p in OUT.rglob('*') if p.is_file())
print(json.dumps(info, indent=2, ensure_ascii=False))
print(f'\n📦 {OUT}: {n_files} file, {dir_size_mb(OUT) / 1024:.2f} GB')
""")

    md(r"""## 🗜️ Cell 10: Nén thành 1 file zip

Kaggle không lưu trọn Output có hàng nghìn file trong nhiều thư mục con (lần chạy trước: 10136 file, 5.55 GB nhưng Output chỉ còn 50 file, 117 MB). Vì vậy dataset được nén thành **một file** `<RUN_TAG>.zip` rồi xoá thư mục gốc. Ảnh PNG đã nén sẵn nên zip dùng chế độ *store* (không nén lại): nhanh và dung lượng gần như không đổi.

Một vài file nhỏ (`README.md`, `dataset_info.json`, ảnh QC) được để ngoài zip trong `<RUN_TAG>_info/` để xem nhanh trên Kaggle.""")
    code(r"""import zipfile

ZIP_PATH = Path(OUTPUT_ROOT) / f'{RUN_TAG}.zip'
INFO_DIR = Path(OUTPUT_ROOT) / f'{RUN_TAG}_info'
if not ZIP_OUTPUT:
    print('ZIP_OUTPUT = False: giữ nguyên thư mục, không nén.')
else:
    free_gb = shutil.disk_usage(OUTPUT_ROOT).free / 1024**3
    need_gb = dir_size_mb(OUT) / 1024
    print(f'Cần {need_gb:.2f} GB để nén, còn trống {free_gb:.2f} GB')
    if free_gb < need_gb * 1.05:
        raise OSError('Không đủ chỗ trống để tạo file zip')

    files = sorted(p for p in OUT.rglob('*') if p.is_file())
    ZIP_PATH.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in tqdm(files, desc='nén zip', file=sys.stdout, mininterval=10, ncols=100):
            zf.write(p, arcname=str(Path(RUN_TAG) / p.relative_to(OUT)))

    # Kiểm tra zip trước khi xoá thư mục gốc
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = [n for n in zf.namelist() if not n.endswith('/')]
    if len(names) != len(files):
        raise RuntimeError(f'Zip có {len(names)} file, thư mục có {len(files)} file')

    if INFO_DIR.exists():
        shutil.rmtree(INFO_DIR)
    INFO_DIR.mkdir(parents=True)
    for f in ['README.md', 'dataset_info.json', 'bank_report.json', 'code_commit.txt',
              'qc_before_after.png', 'labels_check.png']:
        if (OUT / f).is_file():
            shutil.copy2(OUT / f, INFO_DIR / f)
    shutil.rmtree(OUT)
    print(f'✅ {ZIP_PATH.name}: {len(names)} file, {ZIP_PATH.stat().st_size / 1024**3:.2f} GB')

left = [p for p in Path(OUTPUT_ROOT).rglob('*') if p.is_file()]
print(f'Output còn {len(left)} file:', sorted(str(p.relative_to(OUTPUT_ROOT)) for p in left)[:12])
print(f'Tổng thời gian: {(time.time() - NB_START) / 60:.1f} phút. Xong: vào tab Output → New Dataset.')
""")

    nb = {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False,
                       "language": "python", "sourceType": "notebook"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out_file = Path(__file__).resolve().parent.parent / "notebooks" / "06_kaggle_dentex_augment_dataset_export.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook at: {out_file}")


if __name__ == "__main__":
    create_notebook()

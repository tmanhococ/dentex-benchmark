"""
scripts/generate_colab_augment_notebook.py
Generates the comprehensive 1-Click Google Colab Notebook for DENTEX Tooth-Swap Augmentation & Benchmark.
"""

import json
from pathlib import Path


def create_notebook():
    cells = []

    def md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header Markdown
    md("""# 🦷 DENTEX: Pipeline Tăng Cường Dữ Liệu Ghép Răng & Huấn Luyện Benchmark (1-Click Google Colab)

Notebook này tự động hóa **toàn bộ quy trình 1-Click** từ dữ liệu X-quang răng trên Google Drive:
1. **Nạp & Xác thực dữ liệu X-quang thật** từ thư mục Google Drive của bạn (tự động nhận diện cấu trúc thư mục, kích thước ảnh và gán nhãn giải phẫu FDI).
2. **Khởi tạo Tooth Bank:** Trích xuất các răng sâu, răng mọc ngầm từ tập Train kèm mask phân đoạn Otsu GrabCut.
3. **Thực hiện ghép răng (Tooth-Swap Augmentation):** Sinh ảnh X-quang tăng cường tuân thủ nghiêm ngặt 8 invariants giải phẫu nha khoa (khớp số răng FDI 1-8, bảo toàn vị trí răng khôn 8, đối xứng qua hàm đối bên, hòa trộn biên Poisson/Gaussian).
4. **Hiển thị trực quan so sánh Trước vs Sau:** Hiển thị ảnh răng trước và sau khi ghép kèm ảnh zoom cận cảnh từng chân răng/thân răng và vết ghép.
5. **Lưu toàn bộ dữ liệu đã xử lý vào Google Drive:** Tự động sao lưu ảnh đã ghép, nhãn COCO/YOLO, Tooth Bank và file nén zip về Drive.
6. **Huấn luyện mô hình:** Train cả 2 mô hình **Baseline** và **Augmented** (Faster R-CNN / HierarchicalNet, 20-30 epochs).
7. **Đánh giá Benchmark chuẩn:** Đo lường các chỉ số detection chuẩn COCO ($mAP_{50}$, $mAP_{50:95}$, $AR$, per-class AP, độ trễ và VRAM).
8. **Đóng gói & Xuất bản Kaggle:** Tự động tổng hợp thành dataset đa định dạng (COCO + YOLO + Tooth Bank + Recipes), nén zip lưu về Drive và hỗ trợ push thẳng lên Kaggle Public.

---
👉 **Hướng dẫn:** Chỉ cần sửa đường dẫn `DRIVE_DATA_DIR` ở Cell 0 dưới đây thành đường dẫn thư mục dữ liệu trên Drive của bạn, sau đó chọn **Runtime → Run all** (Ctrl+F9).""")

    # Cell 0: Config
    md("## ⚙️ Cell 0: Cấu Hình Thực Thi (Người Dùng Tùy Chỉnh)")
    code("""# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU TRÊN GOOGLE DRIVE
# ==============================================================================
# 1. Đường dẫn đến thư mục chứa dữ liệu X-quang của bạn trên Google Drive:
# Thư mục này có thể chứa:
# - Ảnh X-quang (.png, .jpg) và file nhãn .json (COCO hoặc DENTEX gốc)
# - Hoặc thư mục raw/ chứa ảnh và thư mục processed/ chứa instances_train.json
DRIVE_DATA_DIR = '/content/drive/MyDrive/DENTEX'

# 2. Thư mục trên Google Drive để tự động lưu lại toàn bộ dữ liệu đã xử lý,
# checkpoints, logs, và dataset hoàn chỉnh đã đóng gói (kèm file .zip):
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/DENTEX_AUGMENT_OUTPUT'

# Cấu hình GitHub Repository
REPO_URL = 'https://github.com/tmanhococ/dentex-benchmark.git'
BRANCH = 'main'

# Cấu hình Thực nghiệm
MODEL_TYPE = 'faster_rcnn'       # 'faster_rcnn' hoặc 'hierarchical_net'
EPOCHS = 30                      # Số epoch huấn luyện (khuyến nghị: 25 - 30)
LEARNING_RATE = 0.0002           # Learning rate
BATCH_SIZE = 2                   # Batch size per step (FP16 tự động kích hoạt)
NUM_AUGMENT_PER_IMAGE = 2        # Số lượng ảnh nhân tạo sinh ra trên mỗi ảnh train gốc
SEED = 42                        # Seed cố định bảo đảm tính tái lập 100%

# Tùy chọn Kaggle (Nếu muốn tự động upload lên Kaggle Dataset)
AUTO_UPLOAD_KAGGLE = False       # Đổi thành True nếu bạn muốn push tự động
KAGGLE_DATASET_ID = 'your_username/dentex-augmented-tooth-swap'
""")

    # Cell 1: Environment & Drive
    md("## 💻 Cell 1: Kết Nối Google Drive & Kiểm Tra Phần Cứng GPU")
    code("""import os, sys
from pathlib import Path
import torch

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print('=== 1. KIỂM TRA PHẦN CỨNG GPU ===')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'✅ GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GB | CUDA: {torch.version.cuda}')
else:
    print('ℹ️ Không có GPU, đang chạy trên CPU (Tự động tối ưu số bước test).')

print('\\n=== 2. KẾT NỐI GOOGLE DRIVE ===')
if IN_COLAB:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive')
    print('✅ Google Drive đã kết nối thành công.')
    Path(DRIVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
else:
    print('ℹ️ Đang chạy môi trường ngoài Colab (Local/Headless).')
""")

    # Cell 2: Clone Repo
    md("## 📥 Cell 2: Tải Mã Nguồn Pipeline Từ GitHub")
    code("""import subprocess

REPO_DIR = Path('/content/dentex-benchmark') if IN_COLAB else Path('.').resolve()

if IN_COLAB:
    if not (REPO_DIR / 'dentex_bench').is_dir():
        print(f'📥 Đang clone repository từ {REPO_URL} (nhánh {BRANCH})...')
        subprocess.run(['git', 'clone', '-b', BRANCH, REPO_URL, str(REPO_DIR)], check=True)
    else:
        print('🔄 Repository đã tồn tại, tiến hành git pull cập nhật...')
        subprocess.run(['git', '-C', str(REPO_DIR), 'pull'], check=False)
    os.chdir(str(REPO_DIR))

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

print(f'✅ Thư mục làm việc hiện tại: {Path.cwd()}')
""")

    # Cell 3: Dependencies
    md("## 📦 Cell 3: Cài Đặt Các Gói Thư Viện Cần Thiết")
    code("""import subprocess

REQUIRED_PACKAGES = [
    'albumentations>=1.4.0',
    'pyyaml>=6.0',
    'tabulate>=0.9.0',
    'scikit-image>=0.20.0',
    'scipy>=1.10.0',
    'kaggle>=1.6.0',
    'pillow>=9.0.0'
]

print('📦 Đang kiểm tra và cài đặt thư viện phụ thuộc...')
cmd = [sys.executable, '-m', 'pip', 'install', '-q'] + REQUIRED_PACKAGES
subprocess.run(cmd, check=True)
print('✅ Hoàn tất cài đặt thư viện!')
""")

    # Cell 4: Verify Real Data
    md("## 🔍 Cell 4: Nạp & Xác Thực Dữ Liệu X-Quang Thật Từ Google Drive")
    code("""import json, shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dentex_bench.data.parser import convert_dentex_annotations_to_coco

drive_path = Path(DRIVE_DATA_DIR)
local_data = Path('data')
local_raw = local_data / 'raw'
local_processed = local_data / 'processed'
local_raw.mkdir(parents=True, exist_ok=True)
local_processed.mkdir(parents=True, exist_ok=True)

# 1. Tìm kiếm thông minh nếu người dùng đặt tên thư mục hơi khác trên Drive
if not drive_path.exists():
    print(f'⚠️ Không tìm thấy chính xác {drive_path}, đang tìm kiếm các thư mục X-quang liên quan...')
    candidates = []
    if Path('/content/drive/MyDrive').exists():
        for p in Path('/content/drive/MyDrive').iterdir():
            if p.is_dir() and ('dentex' in p.name.lower() or 'xray' in p.name.lower() or 'dental' in p.name.lower()):
                candidates.append(p)
    if candidates:
        print(f'💡 Đã tự động phát hiện thư mục dữ liệu: {candidates[0]}')
        drive_path = candidates[0]
    elif Path('data').exists() and (list(Path('data').glob('**/*.png')) or list(Path('data').glob('**/*.jpg'))):
        print('💡 Sử dụng thư mục data cục bộ có sẵn: data')
        drive_path = Path('data')
    else:
        raise FileNotFoundError(f'❌ Không tìm thấy thư mục {drive_path}. Hãy kiểm tra lại đường dẫn DRIVE_DATA_DIR trong Cell 0.')

print(f'🔎 Đang quét dữ liệu từ: {drive_path} ...')
all_drive_imgs = [p for p in drive_path.rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg') and p.is_file()]
all_drive_jsons = [p for p in drive_path.rglob('*.json') if p.is_file()]

print(f'📊 Tìm thấy {len(all_drive_imgs)} file ảnh và {len(all_drive_jsons)} file JSON.')
if len(all_drive_imgs) == 0:
    raise RuntimeError(f'❌ Không tìm thấy ảnh X-quang nào trong {drive_path}. Vui lòng kiểm tra lại đường dẫn.')

# 2. Đồng bộ ảnh sang local data/raw an toàn (tránh copy cùng file)
print('📥 Đang đồng bộ ảnh vào môi trường thực thi...')
for img_p in all_drive_imgs:
    dest = local_raw / img_p.name
    if img_p.resolve() != dest.resolve() and not dest.exists():
        shutil.copy2(img_p, dest)

train_ann_file = local_processed / 'instances_train.json'
val_ann_file = local_processed / 'instances_val.json'

# 3. Phân tích hoặc nạp nhãn COCO
existing_train = [j for j in all_drive_jsons if 'instances_train' in j.name]
existing_val = [j for j in all_drive_jsons if 'instances_val' in j.name]

if existing_train and existing_val:
    print('📦 Sử dụng các file COCO annotations có sẵn từ Drive...')
    if existing_train[0].resolve() != train_ann_file.resolve():
        shutil.copy2(existing_train[0], train_ann_file)
    if existing_val[0].resolve() != val_ann_file.resolve():
        shutil.copy2(existing_val[0], val_ann_file)
else:
    raw_jsons = [j for j in all_drive_jsons if 'instances' not in j.name and j.stat().st_size > 3000]
    if not raw_jsons: raw_jsons = all_drive_jsons
    best_ann = raw_jsons[0] if raw_jsons else None
    if best_ann is None:
        raise FileNotFoundError('❌ Không tìm thấy file nhãn JSON nào trong thư mục data.')
    print(f'⚙️ Tự động chuyển đổi nhãn DENTEX từ {best_ann.name} sang định dạng COCO...')
    convert_dentex_annotations_to_coco(
        raw_ann_file=best_ann,
        train_out=train_ann_file,
        val_out=val_ann_file,
        split_ratio=0.8,
        seed=SEED
    )

# 4. TỰ ĐỘNG CHUẨN HÓA KÍCH THƯỚC ẢNH & GÁN NHÃN GIẢI PHẪU FDI
def infer_fdi_robust(ann, img_w, img_h):
    if 'fdi' in ann and ann['fdi']: return int(ann['fdi'])
    q = ann.get('category_id_1')
    t = ann.get('category_id_2')
    if q is not None and t is not None:
        if q in (1, 2, 3, 4) and t in range(1, 9): return q * 10 + t
        elif q in (0, 1, 2, 3) and t in range(0, 8): return (q + 1) * 10 + (t + 1)
    bx, by, bw, bh = ann.get('bbox', [0, 0, 10, 10])
    cx, cy = bx + bw / 2.0, by + bh / 2.0
    is_upper, is_right = (cy < img_h / 2.0), (cx < img_w / 2.0)
    cat_id = ann.get('category_id', 1)
    if cat_id == 4:
        return 18 if (is_upper and is_right) else (28 if is_upper else (48 if is_right else 38))
    quad = (1 if is_right else 2) if is_upper else (4 if is_right else 3)
    dist_ratio = abs(cx - img_w / 2.0) / max(1.0, img_w / 2.0)
    pos = min(7, max(1, int(dist_ratio * 7) + 1))
    return quad * 10 + pos

for ann_path in [train_ann_file, val_ann_file]:
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco_d = json.load(f)
    img_map = {im['id']: im for im in coco_d.get('images', [])}
    for im in coco_d.get('images', []):
        r_path = local_raw / Path(im['file_name']).name
        if r_path.is_file():
            try:
                with Image.open(r_path) as pimg:
                    im['width'], im['height'] = pimg.size
            except Exception: pass
    for a in coco_d.get('annotations', []):
        im_info = img_map.get(a['image_id'], {})
        iw, ih = im_info.get('width', 1000), im_info.get('height', 500)
        a['fdi'] = infer_fdi_robust(a, iw, ih)
    with open(ann_path, 'w', encoding='utf-8') as f:
        json.dump(coco_d, f, indent=2)

with open(train_ann_file, 'r', encoding='utf-8') as f: train_coco = json.load(f)
with open(val_ann_file, 'r', encoding='utf-8') as f: val_coco = json.load(f)

print(f'\\n✅ XÁC THỰC DỮ LIỆU HOÀN TẤT:')
print(f'  - Tập Train: {len(train_coco[\"images\"])} ảnh | {len(train_coco[\"annotations\"])} nhãn tổn thương.')
print(f'  - Tập Val:   {len(val_coco[\"images\"])} ảnh | {len(val_coco[\"annotations\"])} nhãn tổn thương.')

sample_imgs = train_coco[\"images\"][:2]
fig, axes = plt.subplots(1, len(sample_imgs), figsize=(16, 6))
if len(sample_imgs) == 1: axes = [axes]

CLASS_NAMES = {1: 'Caries', 2: 'Deep Caries', 3: 'Periapical', 4: 'Impacted'}

for ax, im_info in zip(axes, sample_imgs):
    img_path = local_raw / im_info['file_name']
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None: continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    anns = [a for a in train_coco['annotations'] if a['image_id'] == im_info['id']]
    for a in anns:
        bx, by, bw, bh = [int(v) for v in a['bbox']]
        c_name = CLASS_NAMES.get(a['category_id'], str(a['category_id']))
        fdi_tag = f\" [FDI {a.get('fdi', '')}]\" if 'fdi' in a else ''
        cv2.rectangle(img_rgb, (bx, by), (bx + bw, by + bh), (255, 255, 0), 3)
        cv2.putText(img_rgb, f\"{c_name}{fdi_tag}\", (bx, max(20, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
    ax.imshow(img_rgb)
    ax.set_title(f\"{im_info['file_name']} ({w}x{h}) - {len(anns)} lesions\")
    ax.axis('off')

plt.tight_layout()
plt.show()
""")

    # Cell 5: Tooth Bank Extraction
    md("## 🦷 Cell 5: Trích Xuất Tooth Bank (Các Răng Bệnh Lý & Masks Phân Đoạn)")
    code("""import json, os, shutil, cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from dentex_bench.augment.config import load_config, ROIConfig
from dentex_bench.augment.bank import ToothBank, ToothEntry
from dentex_bench.augment.fdi import is_upper
from dentex_bench.augment.masks import (
    polygon_to_mask,
    create_class_roi,
    compute_ring_mask,
    classical_mask,
)
from dentex_bench.augment.io_utils import (
    read_image_lossless,
    write_image_lossless,
)

CAT_ID_TO_ROI_CLASS = {
    1: 'caries',
    2: 'deep_caries',
    3: 'periapical',
    4: 'impacted',
}

def safe_infer_fdi(ann, img_w, img_h):
    if 'fdi' in ann and ann['fdi'] is not None:
        return int(ann['fdi'])
    q = ann.get('category_id_1')
    t = ann.get('category_id_2')
    if q is not None and t is not None:
        if q in (1, 2, 3, 4) and t in range(1, 9): return q * 10 + t
        elif q in (0, 1, 2, 3) and t in range(0, 8): return (q + 1) * 10 + (t + 1)
    bx, by, bw, bh = ann.get('bbox', [0, 0, 10, 10])
    cx, cy = bx + bw / 2.0, by + bh / 2.0
    cat_id = ann.get('category_id', 1)
    is_upper_jaw = (cy < img_h / 2.0)
    is_patient_right = (cx < img_w / 2.0)
    if cat_id == 4:
        return 18 if (is_upper_jaw and is_patient_right) else (28 if is_upper_jaw else (48 if is_patient_right else 38))
    quad = (1 if is_patient_right else 2) if is_upper_jaw else (4 if is_patient_right else 3)
    dist = abs(cx - img_w / 2.0) / max(1.0, img_w / 2.0)
    pos = max(1, min(7, int(round(dist * 7)) + 1))
    return quad * 10 + pos

def safe_classical_mask(img_crop, bbox=None):
    if img_crop is None or img_crop.size == 0 or img_crop.shape[0] < 5 or img_crop.shape[1] < 5:
        h = max(1, img_crop.shape[0]) if (img_crop is not None and img_crop.ndim >= 1) else 10
        w = max(1, img_crop.shape[1]) if (img_crop is not None and img_crop.ndim >= 2) else 10
        return np.zeros((h, w), dtype=np.uint8)
    try:
        return classical_mask(img_crop, bbox=bbox)
    except Exception:
        h, w = img_crop.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

def robust_build_bank(cfg_path='configs/augment/base.yaml', train_ann='data/processed/instances_train.json', out_bank_dir='outputs/augment/bank'):
    print('⚙️ Đang trích xuất Tooth Bank từ tập huấn luyện (Tuân thủ INV-1: zero val leakage)...')
    cfg = load_config(cfg_path)
    train_ann_path = Path(train_ann)
    val_ann_path = Path(cfg.paths.val_ann)
    img_dir = Path(cfg.paths.img_dir)
    bank_dir = Path(out_bank_dir)

    if not train_ann_path.exists():
        raise FileNotFoundError(f'Training annotations not found at: {train_ann_path}')

    with open(train_ann_path, 'r', encoding='utf-8') as f:
        train_coco = json.load(f)

    # Invariant 1: Assert zero validation leakage
    train_ids = {img['id'] for img in train_coco.get('images', [])}
    if val_ann_path.exists():
        with open(val_ann_path, 'r', encoding='utf-8') as f:
            val_coco = json.load(f)
        val_ids = {img['id'] for img in val_coco.get('images', [])}
        ToothBank.assert_no_val_leakage(train_ids, val_ids)

    patches_dir = bank_dir / 'patches'
    masks_dir = bank_dir / 'masks'
    patches_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    images_by_id = {img['id']: img for img in train_coco.get('images', [])}
    annotations_by_img = {}
    for ann in train_coco.get('annotations', []):
        annotations_by_img.setdefault(ann['image_id'], []).append(ann)

    entries = []
    roi_cfg = cfg.bank.roi if hasattr(cfg, 'bank') and hasattr(cfg.bank, 'roi') else ROIConfig()

    for img_id, img_info in images_by_id.items():
        file_name = img_info['file_name']
        candidate_paths = [
            img_dir / file_name,
            Path('data/raw') / Path(file_name).name,
            Path('data/raw') / file_name,
            Path('data') / file_name,
        ]
        img_path = next((p for p in candidate_paths if p.is_file()), None)
        if img_path is None:
            continue

        img = read_image_lossless(img_path)
        img_h, img_w = img.shape[:2]
        anns = annotations_by_img.get(img_id, [])

        for idx, ann in enumerate(anns):
            cat_id = ann.get('category_id', 1)
            roi_class = CAT_ID_TO_ROI_CLASS.get(cat_id, 'caries')
            cat_name = roi_class

            bx, by, bw, bh = ann.get('bbox', [0, 0, 10, 10])
            bx, by, bw, bh = float(bx), float(by), float(bw), float(bh)

            # Bỏ qua bounding box suy biến hoặc kích thước không hợp lệ
            if bw < 5 or bh < 5 or bx + bw <= 0 or by + bh <= 0 or bx >= img_w or by >= img_h:
                continue

            fdi = safe_infer_fdi(ann, img_w, img_h)

            pad_x = max(1.0, bw * 0.2)
            pad_y = max(1.0, bh * 0.2)
            rx1 = max(0, min(img_w - 5, int(bx - pad_x)))
            ry1 = max(0, min(img_h - 5, int(by - pad_y)))
            rx2 = min(img_w, max(rx1 + 5, int(bx + bw + pad_x)))
            ry2 = min(img_h, max(ry1 + 5, int(by + bh + pad_y)))
            rw = rx2 - rx1
            rh = ry2 - ry1

            patch = img[ry1:ry2, rx1:rx2].copy()
            if patch is None or patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
                continue
            roi_xywh = [rx1, ry1, rw, rh]

            # Tạo tooth mask an toàn
            rel_box = [bx - rx1, by - ry1, bw, bh]
            seg = ann.get('segmentation')
            if seg and isinstance(seg, list) and len(seg) > 0 and len(seg[0]) >= 6:
                shifted_seg = []
                for poly in seg:
                    pts = np.array(poly).reshape(-1, 2)
                    pts[:, 0] -= rx1
                    pts[:, 1] -= ry1
                    shifted_seg.append(pts.flatten().tolist())
                tooth_mask = polygon_to_mask(shifted_seg, rh, rw)
                mask_source = 'polygon'
            else:
                tooth_mask = safe_classical_mask(patch, bbox=rel_box)
                mask_source = 'classical'

            if tooth_mask is None or tooth_mask.size == 0:
                tooth_mask = np.zeros((rh, rw), dtype=np.uint8)

            p_box = rel_box if roi_class == 'periapical' else None
            roi_mask = create_class_roi(tooth_mask, roi_class, p_box, roi_cfg)
            if roi_mask is None or roi_mask.size == 0:
                roi_mask = tooth_mask.copy()

            ring_mask = compute_ring_mask(roi_mask, ring_width_px=15)
            if np.any(ring_mask > 0):
                gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) if (patch.ndim == 3 and patch.shape[2] in (3, 4)) else patch
                ring_pixels = gray_patch[ring_mask > 0]
                ring_stats = {'mean': float(np.mean(ring_pixels)), 'std': float(np.std(ring_pixels))}
            else:
                ring_stats = {'mean': 128.0, 'std': 20.0}

            entry_id = f'{img_id}_{fdi}_{idx}'
            rel_patch_path = f'patches/{entry_id}.png'
            rel_mask_path = f'masks/{entry_id}_tooth.png'
            rel_roi_mask_path = f'masks/{entry_id}_roi.png'

            write_image_lossless(bank_dir / rel_patch_path, patch)
            write_image_lossless(bank_dir / rel_mask_path, tooth_mask)
            write_image_lossless(bank_dir / rel_roi_mask_path, roi_mask)

            local_ann = dict(ann)
            local_ann['bbox'] = rel_box

            entry = ToothEntry(
                entry_id=entry_id,
                image_id=img_id,
                file_name=file_name,
                fdi=fdi,
                is_upper=is_upper(fdi),
                labels=[cat_name.lower()],
                is_healthy=(cat_id not in (1, 2, 3, 4)),
                roi_class=roi_class,
                bbox_xywh=[bx, by, bw, bh],
                roi_xywh=roi_xywh,
                annotations=[local_ann],
                axis_angle_deg=0.0,
                mask_area_px=int(np.sum(tooth_mask > 0)),
                mask_h=float(bh),
                mask_w=float(bw),
                ring_stats=ring_stats,
                noise_sigma=1.0,
                patch_path=rel_patch_path,
                mask_path=rel_mask_path,
                roi_mask_path=rel_roi_mask_path,
                mask_source=mask_source,
                healthy_score=None,
            )
            entries.append(entry)

    bank = ToothBank(entries)
    bank.save(bank_dir)
    print(f'✅ Đã trích xuất thành công {len(entries)} răng bệnh lý vào Tooth Bank ({bank_dir})')
    return bank

bank = robust_build_bank(
    cfg_path='configs/augment/base.yaml',
    train_ann='data/processed/instances_train.json',
    out_bank_dir='outputs/augment/bank'
)

bank_dir = Path('outputs/augment/bank')
bank_idx_p = bank_dir / 'bank_index.jsonl'
bank_items = []
if bank_idx_p.exists():
    with open(bank_idx_p, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): bank_items.append(json.loads(line))

# Trực quan hóa một số răng mẫu và mask Otsu GrabCut
valid_show = []
for item in bank_items:
    patch_p = bank_dir / item['patch_path'] if not Path(item['patch_path']).is_file() else Path(item['patch_path'])
    mask_p = bank_dir / item['mask_path'] if not Path(item['mask_path']).is_file() else Path(item['mask_path'])
    if patch_p.is_file() and mask_p.is_file():
        p_img = cv2.imread(str(patch_p))
        m_img = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if p_img is not None and m_img is not None:
            valid_show.append((item, p_img, m_img))
    if len(valid_show) >= 5: break

if valid_show:
    fig, axes = plt.subplots(2, len(valid_show), figsize=(15, 6))
    if len(valid_show) == 1: axes = np.expand_dims(axes, axis=1)
    
    for col_idx, (item, p_img, m_img) in enumerate(valid_show):
        lbl = item.get('roi_class') or (item.get('labels', ['Tooth'])[0] if item.get('labels') else 'Tooth')
        axes[0, col_idx].imshow(cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB))
        axes[0, col_idx].set_title(f"FDI: {item.get('fdi', '')}\\n({lbl.upper()})")
        axes[0, col_idx].axis('off')
        
        axes[1, col_idx].imshow(m_img, cmap='gray')
        axes[1, col_idx].set_title('GrabCut Mask')
        axes[1, col_idx].axis('off')
        
    plt.suptitle('🔍 Mẫu Răng Bệnh Lý & Mặt Nạ Phân Đoạn Trong Tooth Bank', fontsize=14)
    plt.tight_layout()
    plt.show()
""")

    # Cell 6: Offline Augmentation + Save to Drive + Visual Comparison
    md("## 🧬 Cell 6: Thực Hiện Ghép Răng, Hiển Thị So Sánh Trước/Sau & Lưu Dữ Liệu Vào Google Drive")
    code("""from scripts.augment_offline import main as augment_offline_main
import json, shutil, cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

print(f'⚙️ Đang sinh dữ liệu ghép răng nhân tạo (seed={SEED}, n={NUM_AUGMENT_PER_IMAGE} per image)...')
augment_offline_main([
    '--config', 'configs/augment/base.yaml',
    '--train-ann', 'data/processed/instances_train.json',
    '--output-dir', 'outputs/augment',
    '--seed', str(SEED),
    '--n-samples-per-image', str(NUM_AUGMENT_PER_IMAGE),
    '--overwrite'
])

run_dir = Path('outputs/augment') / f'base_seed{SEED}'
aug_coco_p = run_dir / 'dataset/coco/instances_train_aug.json'
with open(aug_coco_p, 'r', encoding='utf-8') as f:
    aug_coco = json.load(f)

print(f'\\n✅ TỔNG HỢP DỮ LIỆU AUGMENT THÀNH CÔNG:')
print(f'  - Tổng số ảnh train sau augment: {len(aug_coco[\"images\"])}')
print(f'  - Tổng số nhãn tổn thương: {len(aug_coco[\"annotations\"])}')

# ==============================================================================
# 💾 LƯU TOÀN BỘ DỮ LIỆU ĐÃ XỬ LÝ VÀO GOOGLE DRIVE NGAY LẬP TỨC
# ==============================================================================
drive_processed_dir = Path(DRIVE_OUTPUT_DIR) / 'augmented_data'
if IN_COLAB or Path(DRIVE_OUTPUT_DIR).exists():
    drive_processed_dir.mkdir(parents=True, exist_ok=True)
    print(f'\\n💾 Đang lưu dữ liệu đã xử lý vào Google Drive: {drive_processed_dir} ...')
    
    # 1. Lưu toàn bộ thư mục dataset (images, coco, yolo)
    if (run_dir / 'dataset').resolve() != (drive_processed_dir / 'dataset').resolve():
        shutil.copytree(run_dir / 'dataset', drive_processed_dir / 'dataset', dirs_exist_ok=True)
    
    # 2. Lưu Tooth Bank (patches, masks, index)
    if Path('outputs/augment/bank').exists() and Path('outputs/augment/bank').resolve() != (drive_processed_dir / 'tooth_bank').resolve():
        shutil.copytree('outputs/augment/bank', drive_processed_dir / 'tooth_bank', dirs_exist_ok=True)
        
    # 3. Lưu Provenance state (manifest.json, recipes.jsonl)
    if (run_dir / 'state').exists() and (run_dir / 'state').resolve() != (drive_processed_dir / 'state').resolve():
        shutil.copytree(run_dir / 'state', drive_processed_dir / 'state', dirs_exist_ok=True)
        
    all_saved = list(drive_processed_dir.glob('**/*'))
    print(f'✅ ĐÃ LƯU THÀNH CÔNG VÀO GOOGLE DRIVE: {len([f for f in all_saved if f.is_file()])} files!')
    print(f'📁 Vị trí trên Drive: {drive_processed_dir}')

# ==============================================================================
# 🔍 HIỂN THỊ SO SÁNH TRỰC QUAN: ẢNH GỐC (TRƯỚC) VS ẢNH TĂNG CƯỜNG (SAU KHI GHÉP RĂNG)
# ==============================================================================
print('\\n🖼️ Đang hiển thị trực quan so sánh ảnh trước và sau khi ghép răng...')

recipes_p = run_dir / 'state/recipes.jsonl'
recipe_pairs = []
if recipes_p.exists():
    with open(recipes_p, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recipe_pairs.append(r)

display_pairs = recipe_pairs[:min(3, len(recipe_pairs))]
CLASS_NAMES = {1: 'Caries', 2: 'Deep Caries', 3: 'Periapical', 4: 'Impacted'}

for p_idx, rec in enumerate(display_pairs, 1):
    host_file = rec.get('target_file')
    rec_id = rec.get('recipe_id', '')
    tokens = rec_id.replace('rec_', '').split('_')
    syn_file = f'syn_{tokens[0]}_{tokens[1]}.png' if len(tokens) >= 2 else None
    
    host_path = Path('data/raw') / host_file if host_file else None
    syn_path = run_dir / 'dataset/images' / syn_file if syn_file else None
    
    if not (host_path and host_path.exists() and syn_path and syn_path.exists()):
        continue
        
    img_orig = cv2.imread(str(host_path))
    img_syn = cv2.imread(str(syn_path))
    if img_orig is None or img_syn is None: continue
    
    rgb_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    rgb_syn = cv2.cvtColor(img_syn, cv2.COLOR_BGR2RGB)
    
    orig_im_info = next((im for im in aug_coco['images'] if im['file_name'] == host_file), None)
    syn_im_info = next((im for im in aug_coco['images'] if im['file_name'] == syn_file), None)
    
    orig_anns = [a for a in aug_coco['annotations'] if orig_im_info and a['image_id'] == orig_im_info['id']]
    syn_anns = [a for a in aug_coco['annotations'] if syn_im_info and a['image_id'] == syn_im_info['id']]
    
    # 1. Vẽ bboxes trên ảnh gốc (màu xanh lá)
    for a in orig_anns:
        bx, by, bw, bh = [int(v) for v in a['bbox']]
        cv2.rectangle(rgb_orig, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
        cv2.putText(rgb_orig, CLASS_NAMES.get(a['category_id'], 'Lesion'), (bx, max(20, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
    # 2. Vẽ bboxes trên ảnh sau ghép (Đỏ cho răng mới ghép, Xanh cho răng gốc)
    swapped_anns = []
    for a in syn_anns:
        bx, by, bw, bh = [int(v) for v in a['bbox']]
        is_syn = a.get('synthetic', False)
        color = (255, 50, 50) if is_syn else (0, 255, 0)
        tag = ' [GHÉP]' if is_syn else ''
        cv2.rectangle(rgb_syn, (bx, by), (bx + bw, by + bh), color, 3)
        cv2.putText(rgb_syn, f\"{CLASS_NAMES.get(a['category_id'], '')}{tag}\", (bx, max(20, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if is_syn: swapped_anns.append(a)
        
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), gridspec_kw={'width_ratios': [3, 3, 1.5, 1.5]})
    
    # Cột 1: Toàn cảnh ảnh gốc
    axes[0].imshow(rgb_orig)
    axes[0].set_title(f'1. TRƯỚC TĂNG CƯỜNG (GỐC)\\n{host_file} (Viền Xanh: Tổn thương gốc)', fontsize=11, color='darkgreen')
    axes[0].axis('off')
    
    # Cột 2: Toàn cảnh ảnh synthetic
    axes[1].imshow(rgb_syn)
    axes[1].set_title(f'2. SAU TĂNG CƯỜNG (GHÉP RĂNG)\\n{syn_file} (Viền Đỏ: Răng nhân tạo vừa ghép)', fontsize=11, color='darkred')
    axes[1].axis('off')
    
    # Cột 3 & 4: Zoom cận cảnh vị trí răng ghép
    target_ann = swapped_anns[0] if swapped_anns else (syn_anns[0] if syn_anns else None)
    if target_ann:
        zx, zy, zw, zh = [int(v) for v in target_ann['bbox']]
        pad = 40
        y1, y2 = max(0, zy - pad), min(rgb_syn.shape[0], zy + zh + pad)
        x1, x2 = max(0, zx - pad), min(rgb_syn.shape[1], zx + zw + pad)
        
        crop_orig = rgb_orig[y1:y2, x1:x2]
        crop_syn = rgb_syn[y1:y2, x1:x2]
        
        axes[2].imshow(crop_orig)
        axes[2].set_title('3. CẬN CẢNH TRƯỚC KHI GHÉP\\n(Vùng nhận răng gốc)', fontsize=10)
        axes[2].axis('off')
        
        axes[3].imshow(crop_syn)
        axes[3].set_title('4. CẬN CẢNH SAU KHI GHÉP\\n(Hòa trộn Poisson viền răng)', fontsize=10)
        axes[3].axis('off')
    else:
        axes[2].axis('off'); axes[3].axis('off')
        
    plt.suptitle(f'🦷 CẶP MẪU #{p_idx}: SO SÁNH TRỰC QUAN TRƯỚC VÀ SAU KHI TĂNG CƯỜNG DỮ LIỆU', fontsize=13, weight='bold')
    plt.tight_layout()
    plt.show()
""")

    # Cell 6.1: Display K pairs of before & after augmentation
    md("## 🖼️ Cell 6.1: Hiển Thị K Cặp Ảnh So Sánh Trước & Sau Khi Tăng Cường (Mặc định K = 10)")
    code("""import json, cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# ⚙️ CẤU HÌNH SỐ LƯỢNG CẶP ẢNH MUỐN HIỂN THỊ
# ==============================================================================
K_PAIRS = 10  # Số lượng cặp ảnh Trước/Sau muốn hiển thị (có thể sửa thành 5, 10, 15...)
CLASS_NAMES = {1: 'Caries', 2: 'Deep Caries', 3: 'Periapical', 4: 'Impacted'}

# 1. Xác định đường dẫn thư mục kết quả an toàn
if 'run_dir' not in globals() or not Path(str(run_dir)).exists():
    candidate_runs = sorted(Path('outputs/augment').glob('base_seed*'))
    if candidate_runs:
        run_dir = candidate_runs[-1]
    elif Path('outputs/augment').exists():
        run_dir = Path('outputs/augment')
    elif 'DRIVE_OUTPUT_DIR' in globals() and (Path(DRIVE_OUTPUT_DIR) / 'augmented_data').exists():
        run_dir = Path(DRIVE_OUTPUT_DIR) / 'augmented_data'
    else:
        run_dir = Path('outputs/augment')

# 2. Nạp COCO Annotations đã augment nếu chưa có trong bộ nhớ
if 'aug_coco' not in globals() or aug_coco is None:
    aug_coco_candidates = [
        run_dir / 'dataset/coco/instances_train_aug.json',
        run_dir / 'dataset/coco/instances.json',
        Path('outputs/augment/dataset/coco/instances_train_aug.json'),
    ]
    if 'DRIVE_OUTPUT_DIR' in globals():
        aug_coco_candidates.append(Path(DRIVE_OUTPUT_DIR) / 'augmented_data/dataset/coco/instances_train_aug.json')
    aug_coco_p = next((p for p in aug_coco_candidates if p.is_file()), None)
    if aug_coco_p:
        with open(aug_coco_p, 'r', encoding='utf-8') as f:
            aug_coco = json.load(f)
    else:
        raise FileNotFoundError("❌ Không tìm thấy file nhãn COCO instances_train_aug.json. Hãy chắc chắn Cell 6 đã chạy xong.")

# 3. Thu thập danh sách các cặp ảnh (Gốc -> Ghép)
recipes_p = run_dir / 'state/recipes.jsonl'
if not recipes_p.exists() and 'DRIVE_OUTPUT_DIR' in globals():
    alt_p = Path(DRIVE_OUTPUT_DIR) / 'augmented_data/state/recipes.jsonl'
    if alt_p.exists():
        recipes_p = alt_p

recipe_list = []
if recipes_p.exists():
    with open(recipes_p, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                recipe_list.append(json.loads(line))

all_pairs = []
if recipe_list:
    for rec in recipe_list:
        host_file = rec.get('target_file')
        rec_id = rec.get('recipe_id', '')
        tokens = rec_id.replace('rec_', '').split('_')
        syn_file = f'syn_{tokens[0]}_{tokens[1]}.png' if len(tokens) >= 2 else None
        if host_file and syn_file:
            all_pairs.append((host_file, syn_file, rec))
else:
    # Fallback tự động ghép cặp theo tên file trong COCO
    syn_imgs = [im for im in aug_coco.get('images', []) if im['file_name'].startswith('syn_')]
    for s_im in syn_imgs:
        syn_fn = s_im['file_name']
        parts = syn_fn.replace('syn_', '').replace('.png', '').split('_')
        if len(parts) >= 1:
            try:
                orig_id = int(parts[0])
                orig_im = next((im for im in aug_coco['images'] if im['id'] == orig_id), None)
                if orig_im:
                    all_pairs.append((orig_im['file_name'], syn_fn, None))
            except ValueError:
                pass

print(f"📊 Tìm thấy tổng cộng {len(all_pairs)} cặp ảnh sinh ra từ quá trình ghép răng.")
display_count = min(K_PAIRS, len(all_pairs))
print(f"🖼️ Đang hiển thị {display_count} cặp ảnh so sánh chi tiết trước và sau khi tăng cường...\\n")

def resolve_img_path(fn):
    candidates = [
        run_dir / 'dataset/images' / fn,
        run_dir / 'synthetic' / fn,
        Path('data/raw') / fn,
        Path('data/processed') / fn,
        Path('data') / fn,
    ]
    if 'DRIVE_DATA_DIR' in globals():
        candidates.append(Path(DRIVE_DATA_DIR) / fn)
    if 'DRIVE_OUTPUT_DIR' in globals():
        candidates.append(Path(DRIVE_OUTPUT_DIR) / 'augmented_data/dataset/images' / fn)
    for c in candidates:
        if c.is_file():
            return c
    return None

shown_idx = 0
for host_file, syn_file, rec in all_pairs:
    if shown_idx >= K_PAIRS:
        break

    host_path = resolve_img_path(host_file)
    syn_path = resolve_img_path(syn_file)

    if not host_path or not syn_path:
        continue

    img_orig = cv2.imread(str(host_path))
    img_syn = cv2.imread(str(syn_path))
    if img_orig is None or img_syn is None:
        continue

    rgb_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    rgb_syn = cv2.cvtColor(img_syn, cv2.COLOR_BGR2RGB)

    orig_im_info = next((im for im in aug_coco['images'] if im['file_name'] == host_file), None)
    syn_im_info = next((im for im in aug_coco['images'] if im['file_name'] == syn_file), None)

    orig_anns = [a for a in aug_coco['annotations'] if orig_im_info and a['image_id'] == orig_im_info['id']]
    syn_anns = [a for a in aug_coco['annotations'] if syn_im_info and a['image_id'] == syn_im_info['id']]

    # 1. Vẽ bounding boxes trên ảnh gốc (Màu xanh lá)
    for a in orig_anns:
        bx, by, bw, bh = [int(v) for v in a['bbox']]
        cname = CLASS_NAMES.get(a.get('category_id'), 'Lesion')
        cv2.rectangle(rgb_orig, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
        cv2.putText(rgb_orig, cname, (bx, max(20, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2. Vẽ bounding boxes trên ảnh sau ghép (Đỏ cho răng mới ghép, Xanh cho răng gốc)
    swapped_anns = []
    for a in syn_anns:
        bx, by, bw, bh = [int(v) for v in a['bbox']]
        is_syn = a.get('synthetic', False)
        cname = CLASS_NAMES.get(a.get('category_id'), 'Lesion')
        color = (255, 50, 50) if is_syn else (0, 255, 0)
        tag = ' [GHÉP]' if is_syn else ''
        cv2.rectangle(rgb_syn, (bx, by), (bx + bw, by + bh), color, 3)
        cv2.putText(rgb_syn, f"{cname}{tag}", (bx, max(20, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if is_syn:
            swapped_anns.append(a)

    shown_idx += 1
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), gridspec_kw={'width_ratios': [3, 3, 1.5, 1.5]})

    # Cột 1: Toàn cảnh ảnh gốc (Trước)
    axes[0].imshow(rgb_orig)
    axes[0].set_title(f'1. TRƯỚC TĂNG CƯỜNG (GỐC)\\n{host_file} (Viền Xanh: Tổn thương gốc)', fontsize=11, color='darkgreen')
    axes[0].axis('off')

    # Cột 2: Toàn cảnh ảnh synthetic (Sau)
    axes[1].imshow(rgb_syn)
    axes[1].set_title(f'2. SAU TĂNG CƯỜNG (GHÉP RĂNG)\\n{syn_file} (Viền Đỏ: Răng nhân tạo vừa ghép)', fontsize=11, color='darkred')
    axes[1].axis('off')

    # Cột 3 & 4: Zoom cận cảnh vị trí răng ghép
    target_ann = swapped_anns[0] if swapped_anns else (syn_anns[0] if syn_anns else None)
    if target_ann:
        zx, zy, zw, zh = [int(v) for v in target_ann['bbox']]
        pad = 40
        h_max = min(rgb_orig.shape[0], rgb_syn.shape[0])
        w_max = min(rgb_orig.shape[1], rgb_syn.shape[1])
        y1, y2 = max(0, zy - pad), min(h_max, zy + zh + pad)
        x1, x2 = max(0, zx - pad), min(w_max, zx + zw + pad)

        crop_orig = rgb_orig[y1:y2, x1:x2]
        crop_syn = rgb_syn[y1:y2, x1:x2]

        axes[2].imshow(crop_orig)
        axes[2].set_title('3. CẬN CẢNH TRƯỚC KHI GHÉP\\n(Vùng nhận răng gốc)', fontsize=10)
        axes[2].axis('off')

        axes[3].imshow(crop_syn)
        axes[3].set_title('4. CẬN CẢNH SAU KHI GHÉP\\n(Hòa trộn Poisson viền răng)', fontsize=10)
        axes[3].axis('off')
    else:
        axes[2].axis('off')
        axes[3].axis('off')

    extra_info = ''
    if rec and 'ops' in rec and rec['ops']:
        first_op = rec['ops'][0]
        extra_info = f" | Răng ghép: FDI {first_op.get('target_fdi', '')} ({first_op.get('source_entry_id', '')})"

    plt.suptitle(f'🦷 CẶP MẪU #{shown_idx}/{display_count}: SO SÁNH TRỰC QUAN TRƯỚC VÀ SAU KHI TĂNG CƯỜNG{extra_info}', fontsize=13, weight='bold')
    plt.tight_layout()
    plt.show()
""")

    # Cell 7: Train Baseline & Augmented
    md("## 🏋️ Cell 7: Huấn Luyện Cả 2 Mô Hình: Baseline vs Augmented (30 Epochs)")
    code("""import subprocess, shutil

print(f'=============================================================')
print(f'🏋️ HUẤN LUYỆN BENCHMARK: {MODEL_TYPE.upper()} ({EPOCHS} EPOCHS)')
print(f'=============================================================')

# 1. Huấn luyện Baseline
baseline_out = Path('outputs/benchmark_runs/baseline') / MODEL_TYPE
print(f'\\n[1/2] Đang huấn luyện BASELINE {MODEL_TYPE} ...')
subprocess.run([
    sys.executable, 'scripts/train.py',
    '--model', MODEL_TYPE,
    '--epochs', str(EPOCHS),
    '--lr', str(LEARNING_RATE),
    '--output_dir', 'outputs/benchmark_runs/baseline'
], check=True)

# 2. Huấn luyện Augmented
augmented_out = Path('outputs/benchmark_runs/augmented') / MODEL_TYPE
print(f'\\n[2/2] Đang huấn luyện AUGMENTED {MODEL_TYPE} (với tập dữ liệu răng ghép) ...')
aug_train_ann = Path('outputs/augment') / f'base_seed{SEED}' / 'dataset/coco/instances_train_aug.json'
subprocess.run([
    sys.executable, 'scripts/train.py',
    '--model', MODEL_TYPE,
    '--epochs', str(EPOCHS),
    '--lr', str(LEARNING_RATE),
    '--train-ann', str(aug_train_ann),
    '--output_dir', 'outputs/benchmark_runs/augmented'
], check=True)

# Sao lưu checkpoint về Google Drive an toàn
if (IN_COLAB or Path(DRIVE_OUTPUT_DIR).exists()) and Path('outputs/benchmark_runs').exists():
    drive_ckpts = Path(DRIVE_OUTPUT_DIR) / 'checkpoints'
    if Path('outputs/benchmark_runs').resolve() != drive_ckpts.resolve():
        print(f'\\n💾 Đang sao lưu toàn bộ checkpoints về Google Drive: {DRIVE_OUTPUT_DIR} ...')
        drive_ckpts.mkdir(parents=True, exist_ok=True)
        shutil.copytree('outputs/benchmark_runs', drive_ckpts, dirs_exist_ok=True)
        print('✅ Sao lưu checkpoint về Google Drive hoàn tất!')
""")

    # Cell 8: Evaluate & Compare
    md("## 📊 Cell 8: Đánh Giá Đối Soát Độc Lập Trên Validation Set & Báo Cáo")
    code("""import subprocess
from pathlib import Path
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import json

eval_base_dir = Path('outputs/eval_results/baseline')
eval_aug_dir = Path('outputs/eval_results/augmented')
eval_base_dir.mkdir(parents=True, exist_ok=True)
eval_aug_dir.mkdir(parents=True, exist_ok=True)

# 1. Đánh giá Baseline Checkpoint
print('📊 Đang đánh giá mô hình Baseline...')
subprocess.run([
    sys.executable, 'scripts/evaluate.py',
    '--model', MODEL_TYPE,
    '--checkpoint', str(baseline_out / 'best.pth'),
    '--output_dir', str(eval_base_dir)
], check=True)

# 2. Đánh giá Augmented Checkpoint
print('📊 Đang đánh giá mô hình Augmented...')
subprocess.run([
    sys.executable, 'scripts/evaluate.py',
    '--model', MODEL_TYPE,
    '--checkpoint', str(augmented_out / 'best.pth'),
    '--output_dir', str(eval_aug_dir)
], check=True)

# Đọc kết quả từ summary
base_summary = (eval_base_dir / 'benchmark_summary.md').read_text(encoding='utf-8') if (eval_base_dir / 'benchmark_summary.md').exists() else ''
aug_summary = (eval_aug_dir / 'benchmark_summary.md').read_text(encoding='utf-8') if (eval_aug_dir / 'benchmark_summary.md').exists() else ''

print('\\n=============================================================')
print('🏆 BẢNG KẾT QUẢ ĐỐI SOÁT BENCHMARK CHI TIẾT')
print('=============================================================\\n')

# Hiển thị biểu đồ Loss Curves
b_hist_p = baseline_out / 'history.json'
a_hist_p = augmented_out / 'history.json'

if b_hist_p.exists() and a_hist_p.exists():
    with open(b_hist_p) as f: b_h = json.load(f)
    with open(a_hist_p) as f: a_h = json.load(f)
    
    epochs_range = range(1, len(b_h) + 1)
    b_loss = [x['loss'] for x in b_h]
    a_loss = [x['loss'] for x in a_h]
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, b_loss, 'r-o', label='Baseline Loss')
    plt.plot(epochs_range, a_loss, 'g-s', label='Tooth-Swap Augmented Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'So Sánh Đường Cong Hội Tụ: Baseline vs Tooth-Swap Augmented ({MODEL_TYPE})')
    plt.grid(True)
    plt.legend()
    plt.show()

display(Markdown(f\"\"\"
### 📌 Baseline Report:
{base_summary}

### 🚀 Augmented (Ghép răng) Report:
{aug_summary}
\"\"\"))
""")

    # Cell 9: Package Dataset & Zip
    md("## 📦 Cell 9: Đóng Gói Dataset Đầy Đủ & Đồng Bộ Toàn Bộ Về Google Drive")
    code("""from scripts.package_kaggle_dataset import main as package_dataset_main
import shutil
from pathlib import Path

print('📦 Đang đóng gói dataset thành chuẩn COCO + YOLO + Tooth Bank + Lineage Recipes...')
package_dataset_main()

pkg_dir = Path('outputs/kaggle_dataset/dentex-augmented-tooth-swap')
zip_path = Path('outputs/kaggle_dataset/dentex_augmented_tooth_swap.zip')

print(f'✅ Đã tạo gói dataset tại: {pkg_dir}')
print(f'✅ File zip nén: {zip_path} ({zip_path.stat().st_size / (1024**2):.2f} MB)')

if (IN_COLAB or Path(DRIVE_OUTPUT_DIR).exists()) and pkg_dir.exists():
    drive_final_dir = Path(DRIVE_OUTPUT_DIR) / 'final_dataset'
    drive_final_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'\\n💾 Đang đồng bộ toàn bộ dataset đóng gói và file zip về Google Drive ({drive_final_dir})...')
    if zip_path.resolve() != (Path(DRIVE_OUTPUT_DIR) / zip_path.name).resolve():
        shutil.copy2(zip_path, Path(DRIVE_OUTPUT_DIR) / zip_path.name)
    if pkg_dir.resolve() != drive_final_dir.resolve():
        shutil.copytree(pkg_dir, drive_final_dir, dirs_exist_ok=True)
    
    print('\\n=============================================================')
    print('🎉 TẤT CẢ DỮ LIỆU ĐÃ ĐƯỢC LƯU AN TOÀN TRÊN GOOGLE DRIVE:')
    print(f'  1. Thư mục dữ liệu xử lý: {Path(DRIVE_OUTPUT_DIR) / \"augmented_data\"}')
    print(f'  2. Checkpoints huấn luyện: {Path(DRIVE_OUTPUT_DIR) / \"checkpoints\"}')
    print(f'  3. File Zip Dataset nén:   {Path(DRIVE_OUTPUT_DIR) / zip_path.name}')
    print(f'  4. Dataset giải nén:       {drive_final_dir}')
    print('=============================================================')
""")

    # Cell 10: Kaggle Upload
    md("## 🚀 Cell 10: Tự Động Tải Dataset Lên Kaggle (Tùy Chọn)")
    code("""import subprocess

if AUTO_UPLOAD_KAGGLE:
    print('🚀 Đang kiểm tra cấu hình Kaggle API...')
    kaggle_cfg = Path.home() / '.kaggle/kaggle.json'
    
    # Kiểm tra nếu có file kaggle.json trên Google Drive
    drive_kaggle_json = Path('/content/drive/MyDrive/kaggle.json')
    if drive_kaggle_json.exists():
        kaggle_cfg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_kaggle_json, kaggle_cfg)
        kaggle_cfg.chmod(0o600)
        print('✅ Đã nạp thông tin Kaggle API từ Google Drive.')

    if kaggle_cfg.exists() or 'KAGGLE_API_TOKEN' in os.environ or 'KAGGLE_KEY' in os.environ:
        print(f'📤 Đang đẩy dataset lên Kaggle ({KAGGLE_DATASET_ID}) với quyền Public...')
        try:
            res = subprocess.run([
                sys.executable, '-m', 'kaggle', 'datasets', 'create',
                '-p', str(pkg_dir),
                '--public'
            ], capture_output=True, text=True)
            print(res.stdout)
            if res.returncode == 0:
                print('🎉 Chúc mừng! Dataset đã được publish public thành công trên Kaggle!')
            else:
                print(f'ℹ️ Thông báo từ Kaggle: {res.stderr}')
        except Exception as e:
            print(f'❌ Lỗi upload Kaggle: {e}')
    else:
        print('⚠️ Chưa tìm thấy Kaggle credentials.')
        print('👉 Bạn chỉ cần tải file zip từ Google Drive lên https://www.kaggle.com/datasets/new để mở public.')
else:
    print('ℹ️ Tùy chọn AUTO_UPLOAD_KAGGLE đang tắt.')
    print('👉 Bạn có thể tải file zip từ Google Drive để upload lên Kaggle bất cứ lúc nào.')
""")

    nb = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_file = Path(__file__).resolve().parent.parent / "notebooks" / "04_colab_dentex_augment_run_all.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"Generated notebook at: {out_file}")


if __name__ == "__main__":
    create_notebook()

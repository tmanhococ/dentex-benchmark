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
1. **Nạp & Xác thực dữ liệu X-quang thật** từ thư mục Google Drive của bạn.
2. **Khởi tạo Tooth Bank:** Trích xuất các răng sâu, răng mọc ngầm từ tập Train kèm mask phân đoạn Otsu GrabCut.
3. **Thực hiện ghép răng (Tooth-Swap Augmentation):** Sinh ảnh X-quang tăng cường tuân thủ nghiêm ngặt 8 invariants giải phẫu nha khoa (khớp số răng FDI 1-8, bảo toàn vị trí răng khôn 8, đối xứng qua hàm đối bên, hòa trộn biên Poisson/Gaussian).
4. **Hiển thị trực quan so sánh Trước vs Sau:** Hiển thị ảnh răng trước và sau khi ghép kèm ảnh zoom cận cảnh từng chân răng/thân răng.
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
    print('⚠️ Cảnh báo: Không tìm thấy GPU. Hãy vào Runtime → Change runtime type → Chọn T4 GPU.')

print('\\n=== 2. KẾT NỐI GOOGLE DRIVE ===')
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print(f'✅ Google Drive đã kết nối thành công.')
    Path(DRIVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
else:
    print('ℹ️ Đang chạy ngoài Google Colab (Local environment).')
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
    'kaggle>=1.6.0'
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
from dentex_bench.data.parser import convert_dentex_annotations_to_coco

drive_path = Path(DRIVE_DATA_DIR)
local_data = Path('data')
local_raw = local_data / 'raw'
local_processed = local_data / 'processed'
local_raw.mkdir(parents=True, exist_ok=True)
local_processed.mkdir(parents=True, exist_ok=True)

print(f'🔎 Đang quét dữ liệu từ Google Drive: {drive_path} ...')
if not drive_path.exists():
    raise FileNotFoundError(f'❌ Không tìm thấy thư mục {drive_path}. Hãy kiểm tra lại tên thư mục trên Google Drive của bạn.')

# Quét ảnh
all_drive_imgs = list(drive_path.glob('**/*.png')) + list(drive_path.glob('**/*.jpg'))
all_drive_jsons = list(drive_path.glob('**/*.json'))
print(f'📊 Tìm thấy {len(all_drive_imgs)} file ảnh và {len(all_drive_jsons)} file JSON.')

if len(all_drive_imgs) == 0:
    raise RuntimeError(f'❌ Không tìm thấy ảnh X-quang nào trong {drive_path}. Vui lòng đặt ảnh vào thư mục.')

# Copy ảnh sang local data/raw nếu chưa có
print('📥 Đang đồng bộ ảnh vào môi trường thực thi...')
for img_p in all_drive_imgs:
    dest = local_raw / img_p.name
    if not dest.exists():
        shutil.copy2(img_p, dest)

train_ann_file = local_processed / 'instances_train.json'
val_ann_file = local_processed / 'instances_val.json'

# Kiểm tra nếu đã có sẵn instances_train.json trên Drive
existing_train = [j for j in all_drive_jsons if 'instances_train' in j.name]
existing_val = [j for j in all_drive_jsons if 'instances_val' in j.name]

if existing_train and existing_val:
    print('📦 Sử dụng các file COCO annotations có sẵn từ Drive...')
    shutil.copy2(existing_train[0], train_ann_file)
    shutil.copy2(existing_val[0], val_ann_file)
else:
    # Tự động parse từ raw DENTEX annotation json
    raw_jsons = [j for j in all_drive_jsons if 'instances' not in j.name and j.stat().st_size > 5000]
    if not raw_jsons:
        raw_jsons = all_drive_jsons
    best_ann = raw_jsons[0]
    print(f'⚙️ Tự động chuyển đổi nhãn DENTEX từ {best_ann.name} sang định dạng COCO...')
    convert_dentex_annotations_to_coco(
        raw_ann_file=best_ann,
        train_out=train_ann_file,
        val_out=val_ann_file,
        split_ratio=0.8,
        seed=SEED
    )

with open(train_ann_file, 'r', encoding='utf-8') as f:
    train_coco = json.load(f)
with open(val_ann_file, 'r', encoding='utf-8') as f:
    val_coco = json.load(f)

print(f'\\n✅ XÁC THỰC DỮ LIỆU HOÀN TẤT:')
print(f'  - Tập Train: {len(train_coco[\"images\"])} ảnh | {len(train_coco[\"annotations\"])} nhãn tổn thương.')
print(f'  - Tập Val:   {len(val_coco[\"images\"])} ảnh | {len(val_coco[\"annotations\"])} nhãn tổn thương.')

# Hiển thị trực quan 2 ảnh X-quang thật đầu tiên để xác nhận
sample_imgs = train_coco[\"images\"][:2]
fig, axes = plt.subplots(1, len(sample_imgs), figsize=(16, 6))
if len(sample_imgs) == 1: axes = [axes]

CLASS_NAMES = {1: 'Caries', 2: 'Deep Caries', 3: 'Periapical', 4: 'Impacted'}
COLORS = {1: (0, 255, 0), 2: (255, 165, 0), 3: (255, 0, 0), 4: (0, 0, 255)}

for ax, im_info in zip(axes, sample_imgs):
    img_path = local_raw / im_info['file_name']
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None: continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Vẽ bboxes
    anns = [a for a in train_coco['annotations'] if a['image_id'] == im_info['id']]
    for a in anns:
        bx, by, bw, bh = [int(v) for v in a['bbox']]
        c_id = a['category_id']
        c_name = CLASS_NAMES.get(c_id, str(c_id))
        cv2.rectangle(img_rgb, (bx, by), (bx + bw, by + bh), (255, 255, 0), 3)
        cv2.putText(img_rgb, c_name, (bx, max(20, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
    ax.imshow(img_rgb)
    ax.set_title(f\"{im_info['file_name']} ({w}x{h}) - {len(anns)} lesions\")
    ax.axis('off')

plt.tight_layout()
plt.show()
""")

    # Cell 5: Tooth Bank Extraction
    md("## 🦷 Cell 5: Trích Xuất Tooth Bank (Các Răng Bệnh Lý & Masks Phân Đoạn)")
    code("""from scripts.build_bank import main as build_bank_main
import cv2
import matplotlib.pyplot as plt

print('⚙️ Đang trích xuất Tooth Bank từ tập huấn luyện (Tuân thủ INV-1: zero val leakage)...')
build_bank_main([
    '--config', 'configs/augment/base.yaml',
    '--train-ann', 'data/processed/instances_train.json',
    '--output', 'outputs/augment/bank',
    '--overwrite'
])

bank_idx_p = Path('outputs/augment/bank/bank_index.jsonl')
bank_items = []
if bank_idx_p.exists():
    with open(bank_idx_p, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): bank_items.append(json.loads(line))

print(f'✅ Đã trích xuất thành công {len(bank_items)} răng bệnh lý vào Tooth Bank.')

# Trực quan hóa một số răng mẫu và mask Otsu GrabCut
if bank_items:
    show_items = bank_items[:min(5, len(bank_items))]
    fig, axes = plt.subplots(2, len(show_items), figsize=(15, 6))
    if len(show_items) == 1: axes = np.expand_dims(axes, axis=1)
    
    for col_idx, item in enumerate(show_items):
        p_img = cv2.imread(item['patch_path'])
        m_img = cv2.imread(item['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        axes[0, col_idx].imshow(cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB))
        axes[0, col_idx].set_title(f\"FDI: {item['fdi']} | Cat: {item['category_id']}\\n({CLASS_NAMES.get(item['category_id'], 'Tooth')})\")
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
    '--output-dir', 'outputs/augment/base_seed0',
    '--seed', str(SEED),
    '--n-samples-per-image', str(NUM_AUGMENT_PER_IMAGE),
    '--overwrite'
])

aug_coco_p = Path('outputs/augment/base_seed0/dataset/coco/instances_train_aug.json')
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
    shutil.copytree('outputs/augment/base_seed0/dataset', drive_processed_dir / 'dataset', dirs_exist_ok=True)
    
    # 2. Lưu Tooth Bank (patches, masks, index)
    if Path('outputs/augment/bank').exists():
        shutil.copytree('outputs/augment/bank', drive_processed_dir / 'tooth_bank', dirs_exist_ok=True)
        
    # 3. Lưu Provenance state (manifest.json, recipes.jsonl)
    if Path('outputs/augment/base_seed0/state').exists():
        shutil.copytree('outputs/augment/base_seed0/state', drive_processed_dir / 'state', dirs_exist_ok=True)
        
    all_saved = list(drive_processed_dir.glob('**/*'))
    print(f'✅ ĐÃ LƯU THÀNH CÔNG VÀO GOOGLE DRIVE: {len([f for f in all_saved if f.is_file()])} files!')
    print(f'📁 Vị trí trên Drive: {drive_processed_dir}')

# ==============================================================================
# 🔍 HIỂN THỊ SO SÁNH TRỰC QUAN: ẢNH GỐC (TRƯỚC) VS ẢNH TĂNG CƯỜNG (SAU KHI GHÉP RĂNG)
# ==============================================================================
print('\\n🖼️ Đang hiển thị trực quan so sánh ảnh trước và sau khi ghép răng...')

# Đọc recipes để bắt cặp chính xác ảnh gốc và ảnh synthetic
recipes_p = Path('outputs/augment/base_seed0/state/recipes.jsonl')
recipe_pairs = []
if recipes_p.exists():
    with open(recipes_p, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recipe_pairs.append(r)

# Chọn tối đa 3 cặp để hiển thị chi tiết
display_pairs = recipe_pairs[:min(3, len(recipe_pairs))]

CLASS_NAMES = {1: 'Caries', 2: 'Deep Caries', 3: 'Periapical', 4: 'Impacted'}

for p_idx, rec in enumerate(display_pairs, 1):
    host_file = rec.get('target_file')
    rec_id = rec.get('recipe_id', '')
    # Tên file synthetic tương ứng (vd syn_1_0.png)
    tokens = rec_id.replace('rec_', '').split('_')
    syn_file = f'syn_{tokens[0]}_{tokens[1]}.png' if len(tokens) >= 2 else None
    
    host_path = Path('data/raw') / host_file if host_file else None
    syn_path = Path('outputs/augment/base_seed0/dataset/images') / syn_file if syn_file else None
    
    if not (host_path and host_path.exists() and syn_path and syn_path.exists()):
        continue
        
    img_orig = cv2.imread(str(host_path))
    img_syn = cv2.imread(str(syn_path))
    if img_orig is None or img_syn is None: continue
    
    rgb_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    rgb_syn = cv2.cvtColor(img_syn, cv2.COLOR_BGR2RGB)
    
    # Tìm thông tin ảnh và annotations
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
        
    # Tạo bảng hiển thị 4 cột: [Ảnh gốc toàn cảnh] | [Ảnh sau ghép toàn cảnh] | [Zoom gốc] | [Zoom sau ghép]
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
subprocess.run([
    sys.executable, 'scripts/train.py',
    '--model', MODEL_TYPE,
    '--epochs', str(EPOCHS),
    '--lr', str(LEARNING_RATE),
    '--train-ann', 'outputs/augment/base_seed0/dataset/coco/instances_train_aug.json',
    '--output_dir', 'outputs/benchmark_runs/augmented'
], check=True)

# Sao lưu checkpoint về Google Drive
if IN_COLAB and Path(DRIVE_OUTPUT_DIR).exists():
    print(f'\\n💾 Đang sao lưu toàn bộ checkpoints về Google Drive: {DRIVE_OUTPUT_DIR} ...')
    drive_ckpts = Path(DRIVE_OUTPUT_DIR) / 'checkpoints'
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

if IN_COLAB and Path(DRIVE_OUTPUT_DIR).exists():
    drive_final_dir = Path(DRIVE_OUTPUT_DIR) / 'final_dataset'
    drive_final_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'\\n💾 Đang đồng bộ toàn bộ dataset đóng gói và file zip về Google Drive ({drive_final_dir})...')
    # Copy file zip vào thư mục chính của DRIVE_OUTPUT_DIR để dễ tải về
    shutil.copy2(zip_path, Path(DRIVE_OUTPUT_DIR) / zip_path.name)
    # Copy thư mục dataset mở rộng
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

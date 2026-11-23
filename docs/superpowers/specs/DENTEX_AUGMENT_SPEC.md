# SPEC: DENTEX Tooth-Swap Augmentation Pipeline (`dentex_bench.augment`)

> **Mục đích tài liệu:** Đặc tả kỹ thuật để một AI/kỹ sư implement pipeline tăng cường dữ liệu dạng **FDI tooth-swap / lesion copy-paste** cho repo `dentex_bench`.
> **Phiên bản spec:** 1.0 · 23/11/2026
> **Đối tượng đọc:** AI coding agent. Mọi yêu cầu có từ khóa **PHẢI / KHÔNG ĐƯỢC / NÊN** theo nghĩa RFC 2119.

---

## 0. Tóm tắt (TL;DR cho người implement)

Xây một module `dentex_bench/augment/` gồm 4 stage chạy nối tiếp:

```
[COCO train JSON + ảnh gốc]
        │
        ▼
 Stage 1: BANK        → trích răng/tổn thương (patch + mask + metadata) từ TẬP TRAIN
        │
        ▼
 Stage 2: PLACEMENT   → chọn (răng nguồn, ảnh đích, vị trí đích) theo ràng buộc FDI/giải phẫu
        │
        ▼
 Stage 3: BLEND       → inpaint lỗ đích → căn hình học → khớp cường độ → hòa trộn → khớp nhiễu
        │
        ▼
 Stage 4: LABEL + QC  → cập nhật nhãn, tính seam score, loại mẫu lỗi
        │
        ▼
 [Ảnh tổng hợp + nhãn COCO/YOLO]  +  [STATE: manifest + recipes để tái lập]
```

Hai yêu cầu bắt buộc của người dùng:

1. **Config tùy chọn lưu hay không lưu bộ dữ liệu đã tăng cường** (`output.save_dataset`).
2. **Luôn lưu state** (manifest + recipe từng mẫu) để lần sau **tái lập bit-exact** hoặc **transform lại** mà không cần chạy lại random.

Nguyên tắc cốt lõi để đạt (2): **mọi ngẫu nhiên được "resolve" thành tham số cụ thể và ghi vào recipe trước khi thực thi.** Thực thi một recipe là hàm thuần (pure function) không dùng RNG.

---

## 1. Phạm vi

### 1.1. Trong phạm vi (In scope)
- Stage 1–4 như trên, chạy offline (sinh dataset ra đĩa) và online (trong `Dataset.__getitem__`).
- Các chế độ placement: `fdi_swap`, `symmetric`, `random` (baseline), `heatmap`.
- Các chế độ blending: `hard`, `feather`, `poisson_normal`, `poisson_mixed`, `laplacian`, `core_preserving`, `random_choice`.
- Hard-negative pasting (dán răng khỏe không nhãn) và "disease erasing" (thay răng bệnh bằng răng khỏe cùng FDI).
- Các augmentation phụ: CLAHE/gamma ngẫu nhiên, patch crop độ phân giải gốc, Repeat Factor Sampling (RFS).
- Hệ thống state/replay, CLI, export COCO + YOLO, công cụ QC (seam score, ảnh debug).
- Preset cấu hình cho các thí nghiệm E0–E9.

### 1.2. Ngoài phạm vi (Non-goals)
- Không implement mô hình sinh (diffusion/GAN inpainting).
- Không huấn luyện mô hình đánh số răng (tooth enumeration). Chỉ định nghĩa **định dạng file pseudo-label** mà pipeline sẽ đọc (mục 4.4).
- Không sửa kiến trúc detector. Chỉ thêm điểm tích hợp vào `scripts/train.py`.

### 1.3. Bất biến (Invariants) — KHÔNG ĐƯỢC vi phạm
| ID | Bất biến |
|---|---|
| INV-1 | Bank chỉ được xây từ **tập train** (split 80/20, `seed=42`). Code PHẢI assert không có `image_id` nào của val trong bank. |
| INV-2 | Tập val/test KHÔNG BAO GIỜ bị augment. |
| INV-3 | Không có vertical flip ở bất kỳ đâu. Rotation toàn ảnh ≤ ±5°. |
| INV-4 | Răng nguồn chỉ được đặt vào vị trí cùng **số răng** (1–8) và cùng **hàm** (trên/dưới). Khác quadrant trái/phải chỉ khi patch đã lật ngang tương ứng. |
| INV-5 | Răng Impacted chỉ được đặt vào vị trí số 8. |
| INV-6 | Khi thay răng đích: **xóa mọi annotation** của răng đích, **thêm mọi annotation** của răng nguồn (cùng `(image_id, FDI)`). |
| INV-7 | Ảnh lưu ra đĩa PHẢI là định dạng lossless (PNG). Không dùng JPEG. |
| INV-8 | Thực thi recipe không được gọi RNG. Mọi tham số ngẫu nhiên phải nằm sẵn trong recipe. |

---

## 2. Bước 0 bắt buộc: Xác minh dữ liệu (Task T0)

Trước khi code stage nào, PHẢI viết `scripts/inspect_dentex_json.py` để in ra và lưu `outputs/augment/_inspect/report.json`:

- Danh sách key top-level của JSON (`images`, `annotations`, `categories_1/2/3`…).
- Với annotation: có `segmentation` không? Kiểu polygon hay RLE? Tỷ lệ annotation có polygon hợp lệ (≥ 3 điểm).
- Ánh xạ trường `category_id_1` (quadrant), `category_id_2` (số răng), `category_id_3` (bệnh) → tên. **Lưu ý:** id trong DENTEX có thể bắt đầu từ 0; PHẢI đọc từ `categories_*` chứ không hard-code.
- Bbox của annotation bệnh là bbox **của răng** hay **của vùng tổn thương**? (So sánh bbox với bounding rect của polygon: IoU trung bình.)
- Số răng bệnh / ảnh, số annotation cùng `(image_id, FDI)` (đa nhãn).
- Kích thước ảnh (min/mean/max), bit depth, số kênh.

**Quy tắc:** nếu `segmentation` không có hoặc < 90% hợp lệ → bật nhánh tạo mask cổ điển (mục 4.3) làm mặc định và log cảnh báo. Các mục sau giả định đã có polygon; nếu thực tế khác, agent PHẢI ghi lại khác biệt vào `docs/augment_assumptions.md`.

---

## 3. Cấu trúc mã nguồn

```
dentex_bench/augment/
├── __init__.py
├── config.py          # Dataclass AugmentConfig + load/validate YAML + config_hash()
├── rng.py             # Sinh RNG xác định theo key (mục 9.1)
├── io_utils.py        # Đọc/ghi ảnh lossless, hash file, COCO/YOLO writer
├── fdi.py             # Tiện ích FDI: parse, mirror (Q1↔Q2, Q3↔Q4), is_upper, tooth_number
├── bank.py            # Stage 1: ToothEntry, ToothBank (build/save/load)
├── masks.py           # polygon→mask, mask cổ điển (Otsu/GrabCut/morphology), ROI theo lớp
├── locator.py         # ToothLocator: tìm vị trí các răng (kể cả răng khỏe) trong ảnh đích
├── placement.py       # Stage 2: các Placement strategy → sinh SwapPlan
├── geometry.py        # PCA trục răng, affine (scale/rotate/translate), warp mask/bbox/polygon
├── blending.py        # Stage 3: inpaint, intensity match, các blender, noise match
├── labels.py          # Stage 4a: cập nhật annotation
├── qc.py              # Stage 4b: seam score, reject rules, ảnh debug
├── extras.py          # CLAHE/gamma, patch crop, hard-negative, disease-erasing
├── sampling.py        # Repeat Factor Sampling
├── recipe.py          # Schema Recipe + resolve (random→params) + execute (params→ảnh)
├── state.py           # Manifest, lưu/đọc state, kiểm tra tương thích, resume
├── offline.py         # Runner sinh dataset offline
└── online.py          # SwapAugmentDataset bọc DentexDataset

configs/augment/
├── base.yaml
├── E0_baseline.yaml … E9_patchcrop.yaml   # preset thí nghiệm (mục 13)

scripts/
├── inspect_dentex_json.py   # T0
├── build_bank.py
├── augment_offline.py
├── replay_augment.py
└── qc_report.py

tests/augment/
└── test_*.py                # mục 15
```

---

## 4. Stage 1 — Tooth & Lesion Bank

### 4.1. Đơn vị dữ liệu `ToothEntry`

Gom annotation theo khóa `(image_id, fdi)` với `fdi = quadrant*10 + tooth_number` (vd. 36, 48).

```python
@dataclass(frozen=True)
class ToothEntry:
    entry_id: str            # f"{image_id}_{fdi}" — ổn định, dùng trong recipe
    image_id: int
    file_name: str
    fdi: int                 # 11..48
    is_upper: bool           # quadrant 1,2
    labels: list[str]        # ["caries","periapical_lesion"] — rỗng nếu răng khỏe
    is_healthy: bool
    roi_class: str           # "caries" | "deep_caries" | "periapical" | "impacted" | "healthy" (lớp quyết định cách tạo ROI, ưu tiên: impacted > periapical > deep_caries > caries)
    bbox_xywh: list[float]   # bbox răng ở toạ độ ảnh gốc
    roi_xywh: list[int]      # vùng crop đã lưu (gồm margin)
    annotations: list[dict]  # các annotation gốc, toạ độ ĐÃ dời về hệ trục patch
    axis_angle_deg: float    # góc trục răng (PCA trên mask)
    mask_area_px: int
    mask_h: float; mask_w: float   # kích thước theo trục chính/phụ
    ring_stats: dict         # mean/std/percentile 1,50,99 của vành 15px quanh mask (ảnh gốc)
    noise_sigma: float       # ước lượng nhiễu (mục 6.6)
    patch_path: str          # bank/patches/{entry_id}.png  (uint8/uint16 giữ nguyên bit depth)
    mask_path: str           # bank/masks/{entry_id}_tooth.png
    roi_mask_path: str       # bank/masks/{entry_id}_roi.png
    mask_source: str         # "polygon" | "otsu_grabcut" | "medsam" | "pseudo_label"
    healthy_score: float | None  # điểm bệnh tối đa từ mô hình lọc (nếu có), dùng để lọc label noise
```

### 4.2. Tạo ROI theo lớp (đọc tham số từ config `bank.roi`)

| roi_class | ROI mask |
|---|---|
| caries, deep_caries | `dilate(tooth_mask, caries_dilate_px)` (mặc định 4) |
| periapical | `closing(tooth_mask ∪ expand(bbox_periapical, periapical_bbox_expand), k=15)`; `expand` = giãn bbox theo tỉ lệ (mặc định 0.15) |
| impacted | `dilate(tooth_mask, impacted_dilate_px)` (mặc định 9) |
| healthy | `dilate(tooth_mask, healthy_dilate_px)` (mặc định 4) |

Patch lưu = crop `roi_xywh` = bounding rect của ROI + `bank.crop_margin_px` (mặc định 24, để có ngữ cảnh cho blending). Lưu ở **độ phân giải gốc**, không resize.

### 4.3. Nhánh tạo mask cổ điển (`masks.classical_mask`)

Dùng khi không có polygon, hoặc khi `bank.mask_source: classical` (để làm thí nghiệm so sánh trong tiểu luận). Thuật toán trong bbox răng (+margin 8px):

1. CLAHE (`clipLimit=2.0`, `tileGridSize=8`).
2. Gaussian blur σ=1.
3. Otsu threshold → foreground ban đầu (răng sáng hơn mô quanh).
4. Morphology: opening k=3, closing k=7; giữ thành phần liên thông lớn nhất; fill holes.
5. GrabCut 5 vòng, khởi tạo: `GC_FGD` = erode(otsu, 7), `GC_PR_FGD` = otsu, `GC_PR_BGD` = phần còn lại trong bbox, `GC_BGD` = viền 3px của bbox.
6. (Tùy chọn `bank.active_contour: true`) skimage `active_contour` khởi tạo từ contour GrabCut.

Hàm `evaluate_mask_methods()` PHẢI tính Dice của từng phương pháp so với polygon gốc trên các răng có polygon và ghi `bank/mask_eval.csv` (đây là thí nghiệm xử lý ảnh cho tiểu luận). Interface PHẢI cho phép thêm provider `medsam` sau này (không bắt buộc implement).

### 4.4. ToothLocator — vị trí răng khỏe ở ảnh đích

Vấn đề: tập disease của DENTEX chỉ gán nhãn răng bệnh, nên **không biết vị trí răng khỏe** trong ảnh đích. `locator.py` định nghĩa interface:

```python
class ToothLocator(Protocol):
    def teeth(self, image_id: int) -> list[LocatedTooth]  # fdi, bbox, mask(optional), source, score
```

Providers (chọn bằng `locator.providers`, theo thứ tự ưu tiên, gộp kết quả, annotation thật thắng pseudo-label khi trùng FDI):

| Provider | Nguồn | Ghi chú |
|---|---|---|
| `annotations` | Annotation bệnh trong JSON train | Luôn có. Chỉ cho vị trí răng **bệnh**. |
| `pseudo_label_file` | File COCO JSON ngoài, mỗi annotation có `fdi`, `bbox`, tùy chọn `segmentation`, `score` | Do một mô hình đánh số răng sinh ra (huấn luyện trên tập `quadrant-enumeration` của DENTEX). Lọc `score ≥ locator.min_score` (mặc định 0.5). |

Hệ quả cho placement (PHẢI xử lý):
- Nếu chỉ có provider `annotations`: chế độ `fdi_swap` chỉ thực hiện được kiểu **bệnh→bệnh** (thay răng bệnh FDI=k ở ảnh đích bằng răng bệnh FDI=k ở ảnh nguồn). Vẫn hữu ích (đổi ngữ cảnh), log cảnh báo `LOCATOR_LIMITED`.
- Có `pseudo_label_file`: thêm kiểu **khỏe→bệnh** (thêm tổn thương vào răng khỏe) và **bệnh→khỏe** (disease erasing), và bank răng khỏe.

### 4.5. Lọc label noise cho răng khỏe

Răng "khỏe" (từ pseudo-label) có thể mang bệnh chưa gán nhãn. Nếu `bank.healthy_filter.predictions_file` được cung cấp (COCO results của mô hình baseline trên tập train), loại răng khỏe có detection bệnh với `IoU ≥ 0.3` và `score ≥ bank.healthy_filter.max_disease_score` (mặc định 0.1). Ghi số răng bị loại vào `bank/stats.json`.

### 4.6. Đầu ra bank

```
{state_dir}/bank/
├── bank_index.jsonl     # mỗi dòng một ToothEntry
├── patches/  masks/
├── stats.json           # số entry theo (roi_class, fdi), số bị loại, lý do
├── mask_eval.csv
└── bank_meta.json       # bank_hash, config con dùng để build, train_image_ids_hash
```

`bank_hash` = SHA256 của `bank_index.jsonl` (đã sort theo `entry_id`). Bank PHẢI được cache: nếu `bank_meta.json` khớp (cùng config con `bank.*`, `locator.*`, cùng `train_image_ids_hash`) thì không build lại.

---

## 5. Stage 2 — Placement

### 5.1. Đầu ra: `SwapPlan` (một phần của recipe)

```python
@dataclass
class SwapOp:
    op: str                  # "swap" | "paste_negative" | "erase_disease"
    source_entry_id: str
    target_image_id: int
    target_fdi: int
    target_tooth_ref: dict   # {"source":"annotations"|"pseudo_label", "bbox":[...], "has_mask":bool}
    mirror: bool             # lật ngang patch (khi chuyển quadrant trái↔phải)
    # tham số hình học/blending được resolve ở Stage 3 nhưng ghi chung vào op
```

### 5.2. Chọn ảnh đích & số op

- Offline: duyệt ảnh train; mỗi ảnh đích sinh `offline.variants_per_image` biến thể (mặc định 2). Mỗi biến thể có `n_ops ~ UniformInt(placement.min_ops, placement.max_ops)` (mặc định 1–3).
- Online: với xác suất `online.p` (mặc định 0.5) ảnh được augment; còn lại trả ảnh gốc.

### 5.3. Chọn răng nguồn

Lấy mẫu `roi_class` theo trọng số tỉ lệ nghịch tần suất (làm mịn):

```
w_c = (1 / count_c) ** placement.class_weight_power     # mặc định power = 0.5
```
có thể ghi đè bằng `placement.class_weights` (vd. `{periapical: 4, impacted: 2, deep_caries: 1.5, caries: 0.5}`). Răng nguồn KHÔNG ĐƯỢC trùng `image_id` với ảnh đích (trừ chế độ `symmetric`).

### 5.4. Các strategy

| `placement.mode` | Hành vi |
|---|---|
| `fdi_swap` (chính) | Chọn răng đích có cùng `tooth_number` và cùng hàm. Ưu tiên cùng FDI; nếu `placement.allow_mirror` và chỉ có FDI đối xứng (vd. 36↔46 KHÔNG hợp lệ vì khác hàm; 36↔46 sai, 36↔26 sai; hợp lệ: 36↔46? → **không**. Hợp lệ là **16↔26** và **36↔46**? Xem quy tắc dưới). |
| `symmetric` | Như van Nistelrooij 2024: dán răng sang vị trí đối xứng trái–phải trong **cùng ảnh**, patch lật ngang. |
| `random` (baseline E2) | Dán ở vị trí ngẫu nhiên trong vùng cung răng (bbox hợp của mọi răng đã biết, giãn 5%), không ràng buộc FDI. Chỉ dùng làm baseline. |
| `heatmap` | Lấy tâm đặt từ heatmap 2D (KDE trên tâm bbox chuẩn hóa của lớp đó trong train), dùng cho lớp không gắn với một răng. |

**Quy tắc mirror (fdi.py, PHẢI có unit test):** cặp trái–phải cùng hàm là Q1↔Q2 (hàm trên) và Q4↔Q3 (hàm dưới), tức `mirror_fdi(16) = 26`, `mirror_fdi(36) = 46`, `mirror_fdi(46) = 36`. Hàm trên/dưới: Q1,Q2 = trên; Q3,Q4 = dưới. Khi nguồn và đích khác bên, `mirror=True` (lật ngang patch, mask, annotation trong patch).

### 5.5. Luật loại bỏ (reject) — ghi lý do vào log

| Mã | Điều kiện |
|---|---|
| `R_SCALE` | Tỷ lệ kích thước đích/nguồn ngoài `[geometry.scale_min, scale_max]` (mặc định 0.85–1.15) trên **cả hai** trục. |
| `R_ANGLE` | Chênh lệch trục \|Δθ\| > `geometry.max_rotate_deg` (mặc định 10°). |
| `R_OVERLAP` | IoU mask sau đặt với răng lân cận đã biết > `placement.max_neighbor_iou` (mặc định 0.1). |
| `R_ARCH` | Tâm răng lệch đường cung (đa thức bậc 2 fit qua tâm các răng đã biết của ảnh đích, cần ≥ 5 răng) > `placement.max_arch_dev_ratio` × chiều cao răng (mặc định 0.5). Bỏ qua luật nếu < 5 răng. |
| `R_IMPACTED` | Nguồn Impacted nhưng đích không phải số 8. |
| `R_BORDER` | ROI sau đặt vượt biên ảnh > 10% diện tích. |
| `R_SAME_IMAGE` | Nguồn và đích cùng ảnh khi mode ≠ symmetric. |

Mỗi op thử tối đa `placement.max_tries` (mặc định 20) lần; hết lượt thì bỏ op (không phải lỗi). Nếu một biến thể có 0 op hợp lệ → không sinh biến thể đó (offline) / trả ảnh gốc (online).

---

## 6. Stage 3 — Blending (thứ tự bắt buộc)

### 6.1. Lấp lỗ răng đích (`blend.inpaint_target`)
Nếu răng đích có mask: `cv2.inpaint(img, dilate(target_mask, 5), radius=blend.inpaint_radius, flags=INPAINT_TELEA)`. Nếu chỉ có bbox: dùng mask cổ điển 4.3 trong bbox đó. Có thể tắt (`inpaint_target: false`) cho ablation.

### 6.2. Căn hình học (`geometry.py`)
- `scale_x = target_mask_w / src_mask_w`, `scale_y = target_mask_h / src_mask_h` (đo theo trục chính/phụ PCA); nếu `geometry.isotropic: true` dùng trung bình hình học.
- `rotate = target_axis - src_axis` (sau mirror nếu có).
- Căn tâm mask nguồn vào tâm mask đích.
- Thêm jitter mức instance (resolve vào recipe): `scale *= U(1-j_s, 1+j_s)` (j_s=0.05), `rotate += U(-j_r, j_r)` (j_r=3°).
- Một ma trận affine 2×3 duy nhất áp cho patch, tooth mask, ROI mask, bbox, polygon. Interpolation: ảnh `INTER_LINEAR`, mask `INTER_NEAREST`.

### 6.3. Khớp cường độ (`blend.intensity_match`)
| Giá trị | Hành vi |
|---|---|
| `none` | Không làm gì. |
| `meanstd` (mặc định) | Biến đổi tuyến tính patch sao cho mean/std của vành nguồn = mean/std vành đích (vành 15px ngoài ROI). |
| `histogram` | `skimage.exposure.match_histograms` với tham chiếu = pixel vành đích, áp cho toàn patch. |

Hệ số (a, b) hoặc LUT histogram PHẢI được tính khi resolve và lưu vào recipe (LUT 256 giá trị, hoặc 65536 nếu uint16 → lưu dạng nén `.npy` trong `state/luts/`).

### 6.4. Blender (`blend.mode`)
Alpha dùng chung: `M` = ROI mask đã warp.

| mode | Công thức / API |
|---|---|
| `hard` | `out = M ? src : dst` |
| `feather` | `α = GaussianBlur(M, σ=blend.feather_sigma)` (mặc định 5), `out = α·src + (1-α)·dst` |
| `poisson_normal` | `cv2.seamlessClone(src, dst, M, center, NORMAL_CLONE)` |
| `poisson_mixed` | như trên với `MIXED_CLONE` |
| `laplacian` | Laplacian pyramid blending `blend.pyramid_levels` mức (mặc định 5) |
| `core_preserving` | `core = erode(M, blend.core_erode_px)` (mặc định 6). Pixel trong `core` = src nguyên vẹn (sau intensity match). Vành `M \ core` hòa trộn bằng `blend.ring_mode` ∈ {`feather`,`poisson_normal`}. Mục đích: không làm nhạt độ thấu quang của tổn thương. |
| `random_choice` | Chọn ngẫu nhiên từ `blend.choices` (mặc định tất cả trừ `random_choice`); lựa chọn PHẢI được resolve vào recipe. |

**Lưu ý triển khai:** `seamlessClone` yêu cầu ảnh 3 kênh uint8 và mask không chạm biên → chuyển gray→BGR, pad 2px, rồi chuyển lại. Với ảnh uint16, chuẩn hóa về uint8 chỉ cho bước Poisson và log cảnh báo `BITDEPTH_REDUCED`.

**Tùy chọn miền log (`blend.log_domain: false` mặc định):** nếu bật, chuyển `I → -log((I+1)/(Imax+1))` trước blending và đổi ngược sau, mô phỏng suy giảm X-quang cộng tính. Đây là tùy chọn thí nghiệm.

### 6.5. Augment mức instance (tùy chọn, `instance_aug`)
Gamma `U(0.85, 1.15)` chỉ trên patch; elastic rất nhẹ (`alpha ≤ 10, sigma ≥ 8`) mặc định **tắt**. Tham số resolve vào recipe.

### 6.6. Khớp nhiễu (`blend.noise_match: true`)
`σ_noise = 1.4826 · median(|I - medianBlur(I,3)|)` tính trên vành. Nếu `σ_dst > σ_src`, thêm nhiễu Gaussian `σ = sqrt(σ_dst² - σ_src²)` vào vùng M. **Nhiễu phải tái lập được:** sinh bằng RNG với seed `noise_seed` lưu trong recipe (đây là ngoại lệ duy nhất của INV-8: RNG được khởi tạo từ seed cố định trong recipe nên vẫn xác định).

---

## 7. Stage 4 — Nhãn & QC

### 7.1. Cập nhật nhãn (`labels.py`)
1. Xóa mọi annotation của ảnh đích có cùng `target_fdi` (INV-6).
2. Với mỗi annotation của răng nguồn: biến đổi bbox bằng affine (lấy 4 góc → bounding rect) và polygon; clip vào ảnh; bỏ nếu diện tích sau clip < 50% ban đầu.
3. Annotation mới giữ `category_id_1/2/3` theo **FDI đích** (quadrant/số răng của đích, bệnh của nguồn). Thêm trường `"synthetic": true, "recipe_id": ...`.
4. `paste_negative`: không thêm annotation. `erase_disease`: xóa annotation đích, không thêm.

### 7.2. Seam score (`qc.seam_score`)
```
G = |Sobel_x| + |Sobel_y| trên ảnh kết quả
boundary = dilate(M,1) XOR erode(M,1)           # đường biên ghép
ring_in  = M \ erode(M, 3);  ring_out = dilate(M, 3) \ M
seam = mean(G[boundary]) / (0.5·(mean(G[ring_in]) + mean(G[ring_out])) + 1e-6)
```
Gần 1 là tốt. Reject nếu `seam > qc.max_seam_score` (mặc định 2.5, mã `R_SEAM`). Lưu seam của mọi op vào recipe để vẽ biểu đồ so sánh các blender.

### 7.3. Ảnh debug
`qc.save_debug_every_n` (mặc định 50): lưu ảnh 3 ô [đích gốc | kết quả | kết quả + contour ROI + bbox mới] vào `state/debug/`.

---

## 8. Augmentation phụ (`extras.py`, `sampling.py`)

| Tên | Config | Hành vi |
|---|---|---|
| Hard-negative paste | `extras.hard_negative.p`, `max_per_image` | Op `paste_negative`: dán răng khỏe từ bank vào vị trí răng cùng FDI bằng **cùng blender** với swap (để seam không tương quan với nhãn). Yêu cầu provider `pseudo_label_file`. |
| Disease erasing | `extras.erase_disease.p` | Op `erase_disease`: thay răng bệnh bằng răng khỏe cùng FDI. |
| CLAHE/gamma đa máy | `extras.photometric.*` | Toàn ảnh: với xác suất p áp CLAHE (`clip ∈ U(1.5,3.0)`, tile ∈ {8,16}) và/hoặc gamma `U(0.8,1.2)`. Áp **sau** swap. |
| Patch crop độ phân giải gốc | `extras.patch_crop.*` | Sau swap, với xác suất p crop vùng `size ∈ {1024,1333}` chứa ≥ 1 annotation (ưu tiên annotation synthetic/lớp hiếm), không resize xuống. Nhãn clip theo crop, bỏ box còn < `min_visible` (0.6) diện tích. |
| RFS | `sampling.rfs.enabled`, `t` | `r_c = max(1, sqrt(t/f_c))`, `f_c` = tỷ lệ ảnh train chứa lớp c; `r_img = max_c r_c`. Offline: nhân bản danh sách ảnh theo `floor(r)` + Bernoulli phần lẻ (resolve vào manifest). Online: cung cấp `WeightedRandomSampler` xác định theo epoch. Hàm `suggest_t(target_repeat_for="periapical", value=2.5)` giải ngược t. |

**Augmentation hiện có** (sáng/tối, xoay ≤ ±5°, lật ngang) giữ nguyên trong `base_transforms` và áp **cuối cùng**, online, sau khi đọc ảnh (kể cả ảnh tổng hợp đã lưu). Lật ngang toàn ảnh: bài toán 4 lớp không cần đổi nhãn, nhưng nếu annotation còn trường quadrant thì PHẢI hoán Q1↔Q2, Q3↔Q4.

---

## 9. Reproducibility & State (yêu cầu cốt lõi)

### 9.1. RNG xác định theo key (`rng.py`)

KHÔNG dùng `random`, `np.random.*` toàn cục, hay `hash()` của Python. Dùng:

```python
def stable_int(*parts) -> int:
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")

def rng_for(global_seed: int, *key) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([global_seed, stable_int(*key)])))
```

Key chuẩn:
- Offline: `rng_for(seed, "offline", image_id, variant_idx)`.
- Online: `rng_for(seed, "online", epoch, image_id, repeat_idx)`.
- RFS: `rng_for(seed, "rfs", epoch)`.

Hệ quả: kết quả **không phụ thuộc** số `num_workers`, thứ tự DataLoader, hay việc resume giữa chừng. Resume training ở epoch k chỉ cần biết `epoch`, không cần lưu RNG state.

### 9.2. Recipe (đơn vị tái lập)

Hai pha tách bạch:
- `resolve(target_image_id, key, cfg, bank, locator) -> Recipe | None` — **chỉ** chỗ này dùng RNG.
- `execute(recipe, image_loader, bank) -> (image, annotations, qc_info)` — hàm thuần, xác định.

Schema (JSON, một dòng/recipe trong `recipes.jsonl`):

```json
{
  "recipe_id": "off_000123_v1",
  "schema_version": 1,
  "rng_key": ["offline", 123, 1],
  "target_image_id": 123,
  "target_file": "train_123.png",
  "target_file_sha256": "…",
  "ops": [
    {
      "op": "swap",
      "source_entry_id": "45_36",
      "target_fdi": 36,
      "target_tooth_ref": {"source": "pseudo_label", "bbox": [1520.0, 820.5, 140.2, 260.0], "has_mask": true},
      "mirror": false,
      "affine": [[1.03, -0.05, 12.4], [0.05, 1.03, -8.1]],
      "inpaint": {"enabled": true, "radius": 5, "dilate_px": 5},
      "intensity": {"mode": "meanstd", "a": 0.94, "b": 7.3},
      "instance_aug": {"gamma": 1.07},
      "blend": {"mode": "core_preserving", "core_erode_px": 6, "ring_mode": "feather", "feather_sigma": 5},
      "noise": {"enabled": true, "sigma": 2.1, "noise_seed": 918273645},
      "qc": {"seam_score": 1.32}
    }
  ],
  "post": {
    "photometric": {"clahe": {"clip": 2.2, "tile": 8}, "gamma": null},
    "patch_crop": null
  },
  "rejections": [{"code": "R_OVERLAP", "tries": 3}],
  "output": {"file": "images/off_000123_v1.png", "sha256": "…", "n_annotations": 7}
}
```

LUT histogram (nếu `intensity.mode = histogram`) lưu ở `state/luts/{recipe_id}_{op_idx}.npy` và recipe trỏ tới đường dẫn.

### 9.3. Manifest (`state/manifest.json`)

```json
{
  "run_id": "E6_core_seed0_20261123-1530",
  "created_at": "2026-11-23T15:30:00+07:00",
  "spec_version": "1.0",
  "schema_version": 1,
  "mode": "offline",
  "global_seed": 0,
  "config": { "...": "toàn bộ config đã resolve (sau merge + default)" },
  "config_hash": "sha256 của config chuẩn hóa (sort keys, bỏ trường output.* và runtime.*)",
  "split": {"seed": 42, "train_ids_hash": "…", "val_ids_hash": "…"},
  "source_ann_file": {"path": "…/instances_train.json", "sha256": "…"},
  "bank": {"path": "bank/", "bank_hash": "…"},
  "locator": {"providers": ["annotations", "pseudo_label_file"], "pseudo_label_sha256": "…"},
  "environment": {
    "python": "3.11.9", "numpy": "…", "opencv": "…", "scikit-image": "…",
    "torch": "…", "platform": "…", "git_commit": "… hoặc null", "git_dirty": false
  },
  "rfs": {"enabled": true, "t": 0.12, "repeat_factors": {"periapical": 2.5, "...": 1.0}},
  "counts": {"recipes": 1012, "ops": 1840, "rejected_ops": 311, "by_reason": {"R_OVERLAP": 120}},
  "status": "complete",
  "progress": {"last_done_index": 563}
}
```

### 9.4. Chế độ lưu (yêu cầu số 1)

```yaml
output:
  save_dataset: true        # true: ghi ảnh + nhãn ra đĩa. false: CHỈ ghi state (manifest + recipes + bank)
  formats: [coco, yolo]     # khi save_dataset = true
  include_originals: true   # bộ dataset đầu ra gồm cả ảnh gốc train (để train trực tiếp)
  image_ext: png            # bắt buộc lossless (INV-7)
  overwrite: false          # true mới được ghi đè run_dir đã tồn tại
```

Bảng hành vi:

| mode | save_dataset | Ghi ra đĩa | Dùng khi |
|---|---|---|---|
| offline | true | bank + manifest + recipes + **images + labels** | Muốn dataset cố định dùng chung cho YOLO/MMDet, hoặc nộp kèm tiểu luận. |
| offline | false | bank + manifest + recipes | Tiết kiệm đĩa; tái tạo ảnh khi cần bằng `replay`. |
| online | (bỏ qua) | bank + manifest; recipes theo epoch nếu `online.log_recipes: true` | Augment mới mỗi epoch, vẫn tái lập được. |

**State luôn được ghi**, không có cờ tắt (`save_recipes` không tồn tại; nếu người dùng đặt `false` → lỗi validate với thông điệp giải thích).

### 9.5. Replay & Transform lại

`scripts/replay_augment.py`:

```bash
# 1) Tái lập y hệt (bit-exact) từ state
python scripts/replay_augment.py --state outputs/augment/E6_core_seed0/state --out outputs/augment/E6_replay

# 2) Tái lập một phần
python scripts/replay_augment.py --state … --recipe-ids off_000123_v1,off_000200_v0

# 3) Transform lại với override (giữ nguyên placement/hình học, đổi blender)
python scripts/replay_augment.py --state … --override blend.mode=poisson_normal --out …_poisson
```

Quy tắc:
- Replay thường: dùng recipe y nguyên → PHẢI tạo ảnh có SHA256 khớp `output.sha256` (nếu có). Nếu khác phiên bản OpenCV/numpy so với manifest → cảnh báo `ENV_MISMATCH` và chuyển sang kiểm tra dung sai: `max|Δ| ≤ 1` mức xám và ≥ 99.9% pixel trùng; báo cáo vào `replay_report.json`.
- Replay có `--override`: chỉ cho phép override nhóm **"execution-only"**: `blend.*`, `intensity_match` (tính lại a,b/LUT một cách xác định), `noise_match`, `extras.photometric.*`, `extras.patch_crop.*`. Override nhóm placement/geometry (`placement.*`, `geometry.*`, `bank.*`) → lỗi, yêu cầu chạy `augment_offline` mới. Tham số ngẫu nhiên mới phát sinh từ override (vd. chọn blender) được resolve bằng `rng_for(seed, "override", recipe_id, op_idx)`. Recipe mới ghi `parent_recipe_id`.
- Đây là cơ chế giúp **ablation blending E3–E6 dùng chung đúng một tập placement**, loại bỏ nhiễu do khác placement — PHẢI dùng cho các thí nghiệm này.

### 9.6. Resume & cache
- Offline runner ghi recipe theo từng ảnh (append + flush) và cập nhật `progress.last_done_index`. Khi chạy lại cùng `run_dir` với `status != complete` và cùng `config_hash` → tiếp tục từ chỗ dừng. Khác `config_hash` → lỗi trừ khi `--force-new`.
- Bank cache theo mục 4.6.
- Ghi file theo kiểu atomic (ghi `*.tmp` rồi `os.replace`).

---

## 10. Cấu hình đầy đủ (`configs/augment/base.yaml`)

```yaml
version: 1
global_seed: 0
run_name: "base"
paths:
  data_root: "data/dentex"
  train_ann: "data/dentex/annotations/instances_train.json"
  val_ann:   "data/dentex/annotations/instances_val.json"   # chỉ để assert INV-1
  img_dir:   "data/dentex/images"
  out_root:  "outputs/augment"         # run_dir = out_root/{run_name}_seed{global_seed}

mode: offline                          # offline | online

bank:
  mask_source: polygon                 # polygon | classical | auto (polygon nếu có, không thì classical)
  active_contour: false
  crop_margin_px: 24
  roi:
    caries_dilate_px: 4
    periapical_bbox_expand: 0.15
    impacted_dilate_px: 9
    healthy_dilate_px: 4
  include_healthy: true                # cần locator pseudo_label_file
  healthy_filter:
    predictions_file: null
    max_disease_score: 0.1
  evaluate_mask_methods: true

locator:
  providers: [annotations]             # thêm pseudo_label_file khi có
  pseudo_label_file: null
  min_score: 0.5

placement:
  mode: fdi_swap                       # fdi_swap | symmetric | random | heatmap
  swap_types: [disease_to_disease, disease_to_healthy]   # disease_to_healthy cần pseudo-label
  allow_mirror: true
  min_ops: 1
  max_ops: 3
  max_tries: 20
  class_weight_power: 0.5
  class_weights: null                  # vd {periapical: 4, impacted: 2, deep_caries: 1.5, caries: 0.5}
  max_neighbor_iou: 0.1
  max_arch_dev_ratio: 0.5

geometry:
  isotropic: false
  scale_min: 0.85
  scale_max: 1.15
  max_rotate_deg: 10
  jitter_scale: 0.05
  jitter_rotate_deg: 3

blend:
  inpaint_target: true
  inpaint_radius: 5
  intensity_match: meanstd             # none | meanstd | histogram
  mode: core_preserving                # hard | feather | poisson_normal | poisson_mixed | laplacian | core_preserving | random_choice
  choices: [hard, feather, poisson_normal, laplacian, core_preserving]
  feather_sigma: 5
  pyramid_levels: 5
  core_erode_px: 6
  ring_mode: feather
  noise_match: true
  log_domain: false

instance_aug:
  gamma: [0.85, 1.15]
  elastic: {enabled: false, alpha: 10, sigma: 8}

extras:
  hard_negative: {enabled: false, p: 0.3, max_per_image: 2}
  erase_disease: {enabled: false, p: 0.1}
  photometric:   {enabled: false, p: 0.5, clahe_clip: [1.5, 3.0], clahe_tiles: [8, 16], gamma: [0.8, 1.2]}
  patch_crop:    {enabled: false, p: 0.5, sizes: [1024, 1333], min_visible: 0.6, prefer_synthetic: true}

sampling:
  rfs: {enabled: false, t: null, target_repeat: {periapical: 2.5}}   # t=null → tự giải từ target_repeat

offline:
  variants_per_image: 2
  num_workers: 4                       # đa tiến trình; kết quả không phụ thuộc giá trị này

online:
  p: 0.5
  log_recipes: false

qc:
  max_seam_score: 2.5
  save_debug_every_n: 50

base_transforms:                       # augmentation đang dùng, áp online sau cùng
  brightness_contrast: {p: 0.5, limit: 0.15}
  rotate: {p: 0.3, limit_deg: 5}
  hflip: {p: 0.5}

output:
  save_dataset: true
  formats: [coco, yolo]
  include_originals: true
  image_ext: png
  overwrite: false
```

**Validate (PHẢI có, `config.py`):** khoảng giá trị hợp lệ; `rotate.limit_deg ≤ 5`; không có khóa `vflip`; `image_ext == png`; `swap_types` chứa loại cần pseudo-label mà `locator` không có provider → lỗi rõ ràng; khóa lạ → lỗi (chống gõ sai). Hỗ trợ `--set a.b=c` từ CLI và kế thừa preset bằng khóa `_base_: base.yaml`.

---

## 11. Bố cục thư mục đầu ra

```
outputs/augment/{run_name}_seed{seed}/
├── state/
│   ├── manifest.json
│   ├── config.resolved.yaml
│   ├── recipes.jsonl                 # offline
│   ├── recipes/epoch_000.jsonl …     # online (nếu log_recipes)
│   ├── rejections.jsonl
│   ├── luts/
│   ├── debug/
│   └── bank/  → symlink hoặc copy tới bank cache (outputs/augment/_bank_cache/{bank_hash}/)
├── dataset/                          # chỉ khi save_dataset = true
│   ├── images/                       # ảnh gốc (nếu include_originals, dạng symlink nếu OS cho phép) + ảnh tổng hợp
│   ├── coco/instances_train_aug.json
│   └── yolo/{images,labels}/ + data.yaml   # val trỏ tới ảnh val gốc
└── qc/
    ├── seam_scores.csv
    └── summary.json                  # phân bố lớp trước/sau, số op theo lớp, seam theo blender
```

YOLO export: 4 lớp theo thứ tự cố định `[caries, deep_caries, periapical_lesion, impacted_tooth]`, toạ độ chuẩn hóa theo kích thước ảnh **sau** khi lưu (không resize).

---

## 12. CLI

```bash
python scripts/inspect_dentex_json.py --ann data/dentex/annotations/instances_train.json
python scripts/build_bank.py        --config configs/augment/base.yaml
python scripts/augment_offline.py   --config configs/augment/E6_core.yaml --set global_seed=1
python scripts/replay_augment.py    --state outputs/augment/E6_core_seed1/state [--override …]
python scripts/qc_report.py         --run outputs/augment/E6_core_seed1
python scripts/train.py --model faster_rcnn --augment-config configs/augment/E6_core.yaml   # online
python scripts/train.py --model faster_rcnn --train-ann outputs/augment/E6_core_seed1/dataset/coco/instances_train_aug.json
```

Mọi script in ra `run_dir` và đường dẫn `manifest.json` ở cuối.

### 12.1. Tích hợp `scripts/train.py`
- Thêm tham số `--augment-config` (online) và `--train-ann` (dùng dataset offline).
- Online: `SwapAugmentDataset(base=DentexDataset(...), cfg, bank, locator)`; gọi `dataset.set_epoch(epoch)` đầu mỗi epoch; `__getitem__(idx)` dùng `rng_for(seed, "online", epoch, image_id, repeat_idx)`.
- Synthesis chạy **ở độ phân giải gốc trước** bước resize hiện có (`target_short=800, max_long=1333`).
- Ghi `augment_manifest_path` + `config_hash` vào checkpoint và log kết quả đánh giá, để nối kết quả mAP với đúng state augmentation.

---

## 13. Preset thí nghiệm (`configs/augment/E*.yaml`)

Mỗi preset kế thừa `base.yaml`, chỉ ghi khác biệt.

| File | Khác biệt so với base | Ghi chú |
|---|---|---|
| `E0_baseline.yaml` | Không swap (`mode: none` đặc biệt: chỉ base_transforms) | Mốc |
| `E1_rfs.yaml` | E0 + `sampling.rfs.enabled: true` | |
| `E2_random_paste.yaml` | `placement.mode: random`, `blend.mode: hard`, `inpaint_target: false` | Baseline copy-paste kiểu Ghiasi |
| `E3_fdi_hard.yaml` | `fdi_swap`, `blend.mode: hard` | **Placement gốc** cho E4–E6 |
| `E4_fdi_feather.yaml` | Replay từ state E3 với `--override blend.mode=feather` | Dùng chung placement |
| `E5_fdi_poisson.yaml` | Replay E3, `blend.mode=poisson_normal` | |
| `E6_core.yaml` | Replay E3, `core_preserving` + `intensity_match=histogram` | |
| `E7_best_rfs.yaml` | Tốt nhất(E3–E6) + RFS | |
| `E8_hardneg.yaml` | E7 + `hard_negative.enabled` | Cần pseudo-label |
| `E9_patchcrop.yaml` | E8 + `patch_crop.enabled` | |

Script tiện ích `scripts/run_ablation.sh` (hoặc `.py`): chạy E3 một lần mỗi seed, rồi sinh E4–E6 bằng replay. Seeds mặc định `[0, 1, 2]`.

---

## 14. QC report (`scripts/qc_report.py`)

Xuất `qc/summary.json` và hình PNG:
1. Phân bố lớp trước/sau augmentation (số box theo lớp, theo FDI).
2. Boxplot seam score theo blender.
3. Lưới 6 ví dụ: cùng một op với các blender khác nhau, có phóng to vùng biên (dùng replay override để tạo).
4. Bảng lý do reject.
5. Dice các phương pháp tạo mask (từ `mask_eval.csv`).

---

## 15. Kiểm thử & tiêu chí nghiệm thu

### 15.1. Unit tests (`tests/augment/`)
| Test | Kiểm tra |
|---|---|
| `test_fdi.py` | `mirror_fdi`, `is_upper`, hợp lệ cặp swap; không cho trên↔dưới. |
| `test_rng.py` | Cùng key → cùng chuỗi số; khác key → khác; không phụ thuộc `PYTHONHASHSEED`. |
| `test_geometry.py` | Affine áp lên bbox/polygon/mask nhất quán (IoU mask warp vs polygon warp ≥ 0.95); mirror đúng. |
| `test_blending.py` | `hard` với M=0 trả ảnh đích; `core_preserving` giữ nguyên pixel lõi (so với src sau intensity match, sai khác 0); `feather` trong [min,max]; dtype/shape giữ nguyên. |
| `test_labels.py` | INV-6: annotation đích bị xóa, nguồn được thêm, `category_id_1/2` theo FDI đích. |
| `test_leakage.py` | INV-1: bank không chứa id val; assert kích hoạt khi cố tình truyền val. |
| `test_recipe_determinism.py` | `execute(recipe)` gọi 2 lần → SHA256 giống; monkeypatch `np.random`/`random` ném lỗi trong `execute` (trừ RNG noise có seed) → không lỗi. |
| `test_replay.py` | Offline 5 ảnh với `save_dataset=true` → replay → SHA256 khớp 100%. Với `save_dataset=false` → replay ra ảnh, chạy lại offline `true` cùng seed → khớp. |
| `test_workers_invariance.py` | Offline `num_workers=1` vs `4` → `recipes.jsonl` giống hệt (sau sort). |
| `test_resume.py` | Dừng giữa chừng (giả lập exception ở ảnh thứ 3) → chạy lại → kết quả giống chạy một mạch. |
| `test_override.py` | Override `blend.mode` giữ nguyên `affine`, `source_entry_id`; override `placement.mode` → lỗi. |
| `test_config.py` | Khóa lạ → lỗi; `vflip` → lỗi; `rotate > 5` → lỗi; `image_ext: jpg` → lỗi. |
| `test_online.py` | `SwapAugmentDataset` epoch e, idx i trả cùng kết quả qua 2 lần tạo dataset mới. |

Dùng ảnh tổng hợp nhỏ (fixture 512×256 vẽ "răng" là elip sáng) để test chạy < 60 s trên CPU, không cần tải DENTEX.

### 15.2. Tiêu chí nghiệm thu (Definition of Done)
- [ ] T0 chạy được trên DENTEX thật và `docs/augment_assumptions.md` ghi rõ kết quả.
- [ ] `build_bank.py` tạo bank với ≥ 90% răng bệnh train (không tính bị loại có lý do).
- [ ] `augment_offline.py` với `E3` chạy hết 564 ảnh train trên CPU 4 nhân trong < 30 phút; mỗi ảnh tổng hợp có ít nhất 1 op hợp lệ hoặc được bỏ qua có log.
- [ ] Replay bit-exact 100% trên cùng môi trường.
- [ ] Tất cả unit test pass; `pytest tests/augment -q` < 2 phút.
- [ ] Mở 20 ảnh debug ngẫu nhiên: nhãn khớp vị trí răng dán, không có răng ngược hàm, không có răng Impacted ngoài vị trí 8.
- [ ] Dataset YOLO load được bằng Ultralytics (`yolo detect val data=.../data.yaml` chạy không lỗi) và COCO load được bằng `pycocotools`.
- [ ] `qc_report.py` sinh đủ 5 nhóm hình/bảng mục 14.

---

## 16. Thứ tự implement đề xuất (milestones)

| M | Nội dung | Phụ thuộc |
|---|---|---|
| M0 | T0 inspect JSON + `fdi.py` + `rng.py` + `config.py` + test | — |
| M1 | `masks.py` (polygon→mask, ROI), `bank.py`, `build_bank.py`, cache | M0 |
| M2 | `locator.py` (provider annotations), `placement.py` (fdi_swap bệnh→bệnh, random, symmetric), `geometry.py` | M1 |
| M3 | `blending.py` (hard, feather, poisson, core_preserving; meanstd), `labels.py`, `recipe.py` resolve/execute | M2 |
| M4 | `state.py`, `offline.py`, export COCO/YOLO, `replay_augment.py`, test tái lập | M3 |
| M5 | `qc.py`, `qc_report.py`, preset E0–E6, tích hợp `train.py --train-ann` → **MVP đủ cho tiểu luận** | M4 |
| M6 | `online.py`, RFS, histogram/laplacian/noise match, extras (photometric, patch crop) | M5 |
| M7 | Provider `pseudo_label_file`, bank răng khỏe, hard-negative, disease erasing, mask cổ điển + đánh giá Dice | M5 |

---

## 17. Rủi ro đã biết & cách xử lý trong code

| Rủi ro | Biện pháp trong spec |
|---|---|
| Seam shortcut (mô hình học "có viền ghép = có bệnh") | Seam score + reject; hard-negative dán bằng cùng blender (M7); ablation blender dùng chung placement. |
| Poisson làm nhạt độ thấu quang tổn thương | `core_preserving`; so sánh E5 vs E6. |
| Label noise răng "khỏe" | `healthy_filter`; mặc định chỉ swap bệnh→bệnh khi không có pseudo-label. |
| Rò rỉ val | INV-1 + `test_leakage.py` + hash id split trong manifest. |
| Không tái lập do khác phiên bản thư viện | Lưu phiên bản vào manifest; replay kiểm tra dung sai + `ENV_MISMATCH`. |
| Học thuộc instance cực hiếm | Jitter mức instance, giới hạn số lần một `source_entry_id` được dùng: `placement.max_uses_per_source` (thêm vào config, mặc định 20 cho offline). |
| Chồng lấp răng kề | `R_OVERLAP`; tùy chọn `log_domain`. |
| Ảnh quá lớn tốn RAM khi đa tiến trình | Đọc ảnh lười (lazy), không giữ cả dataset trong RAM; mỗi worker xử lý một ảnh. |

---

## 18. Câu hỏi mở (agent ghi lại câu trả lời vào `docs/augment_assumptions.md`)

1. Bbox bệnh trong DENTEX là bbox răng hay bbox tổn thương? (ảnh hưởng ROI periapical).
2. Polygon có bao phủ chân răng đến chóp không, hay chỉ thân răng?
3. Ảnh DENTEX là 8-bit hay 16-bit? (ảnh hưởng Poisson và LUT).
4. Có ảnh trùng/gần trùng giữa tập disease và tập quadrant-enumeration không? (kiểm tra bằng SHA256 + tương quan ảnh thu nhỏ 64×32 > 0.98) — nếu dùng tập enumeration để huấn luyện locator, PHẢI loại ảnh trùng với val.

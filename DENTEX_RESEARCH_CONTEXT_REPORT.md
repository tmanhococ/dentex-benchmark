# BÁO CÁO TỔNG QUAN HIỆN TRẠNG TOÀN DIỆN: DỮ LIỆU, MÔ HÌNH, KẾT QUẢ & MÃ NGUỒN DENTEX BENCHMARK
> **Tài liệu nền tảng phục vụ nghiên cứu các phương pháp Tăng Cường Dữ Liệu (Data Augmentation) cho bài toán Phát Hiện Tổn Thương Răng trên Ảnh X-quang Toàn Cảnh (Panoramic Dental Radiographs)**  
> **Dự án:** DENTEX 4-Class Abnormal Tooth Detection Benchmark  
> **Ngày cập nhật:** 23/11/2026 | **Trạng thái:** Phiên bản v2 (Đã chuẩn hóa RoIAlign, loại bỏ kết quả ảo, hỗ trợ đa nền tảng)

---

## MỤC LỤC
1. [Giới Thiệu & Mục Tiêu Nghiên Cứu](#1-giới-thiệu--mục-tiêu-nghiên-cứu)
2. [Hệ Thống Phân Loại Bệnh Lý & Bài Toán Mục Tiêu](#2-hệ-thống-phân-loại-bệnh-lý--bài-toán-mục-tiêu)
3. [Hiện Trạng Dữ Liệu (Dataset Deep-Dive)](#3-hiện-trạng-dữ-liệu-dataset-deep-dive)
   - 3.1. Nguồn gốc, Bản quyền & Phương thức Tải
   - 3.2. Cấu trúc Phân chia Tập dữ liệu (Train/Val Split)
   - 3.3. Phân bố Lớp & Hiện tượng Mất cân bằng Cực đoan (Class Imbalance)
   - 3.4. Thang phân loại Kích thước Bounding Box (COCO Size Taxonomy)
   - 3.5. Ma trận Đồng xuất hiện & Tiến trình Bệnh lý (Co-occurrence)
   - 3.6. Phân bố Không gian Giải phẫu 2D (Spatial Heatmap)
   - 3.7. Đặc trưng Quang phổ & Xảo ảnh X-quang (Radiographic Properties & Artifacts)
4. [Hiện Trạng Các Kiến Trúc Mô Hình (Model Architectures)](#4-hiện-trạng-các-kiến-trúc-mô-hình-model-architectures)
   - 4.1. HierarchicalNet (v2 - Feature Pyramid Network + RoIAlign)
   - 4.2. DiffusionNet (v2 - DiffusionDet Paradigm + Dynamic Refinement)
   - 4.3. DINO DETR (Vision Transformer Baseline)
   - 4.4. Faster R-CNN & HierarchicalDet (Các mô hình đối chứng)
   - 4.5. Cơ chế Tự thích ứng Phần cứng & Chống tràn VRAM (Hardware-Adaptive Engine)
5. [Hiện Trạng Kết Quả Thực Nghiệm & Phân Tích Lỗi (Results & Error Analysis)](#5-hiện-trạng-kết-quả-thực-nghiệm--phân-tích-lỗi-results--error-analysis)
   - 5.1. Bảng Kết quả Thực nghiệm Định lượng
   - 5.2. Điểm nghẽn Định vị (Localization Bottleneck)
   - 5.3. Hồ sơ Top 5 Thất bại Lâm sàng (Top 5 Clinical Failure Modes)
6. [Hiện Trạng Cấu Trúc Mã Nguồn & Quy Trình Thực Thi (Codebase & Pipelines)](#6-hiện-trạng-cấu-trúc-mã-nguồn--quy-trình-thực-thi-codebase--pipelines)
   - 6.1. Cấu trúc Cây Thư mục
   - 6.2. Quy trình Thực thi Dòng lệnh (CLI Workflows)
   - 6.3. Hệ thống Notebook Tương tác & Tự động Sinh mã
   - 6.4. Lịch sử Nâng cấp Quan trọng (v1 sang v2)
7. [Khoảng Trống Hiện Tại & Đề Xuất Nghiên Cứu Tăng Cường Dữ Liệu (Data Augmentation Gap)](#7-khoảng-trống-hiện-tại--đề-xuất-nghiên-cứu-tăng-cường-dữ-liệu-data-augmentation-gap)
   - 7.1. Đánh giá Hiện trạng Pipeline Augmentation trong Repo
   - 7.2. Các Ràng buộc Giải phẫu & Vật lý X-quang (Invariance Rules)
   - 7.3. Các Hướng Tăng cường Dữ liệu Đề xuất Cho Agent Research
8. [Danh Mục Tài Liệu Tham Khảo, Trích Dẫn & Nguồn Gốc (References & Citations)](#8-danh-mục-tài-liệu-tham-khảo-trích-dẫn--nguồn-gốc-references--citations)

---

## 1. GIỚI THIỆU & MỤC TIÊU NGHIÊN CÚU

Tài liệu này cung cấp toàn bộ bức tranh hiện trạng về dữ liệu, mô hình phát hiện đối tượng, kết quả thử nghiệm và cấu trúc repo tại không gian làm việc `d:\Research\Dentex`.

### Mục đích chính:
Cung cấp **ngữ cảnh kỹ thuật và thực nghiệm đầy đủ nhất** để gửi tới các **Research Agent** (hoặc nhóm nghiên cứu) nhằm định hướng, thiết kế và phát triển các **chiến lược Tăng Cường Dữ Liệu (Data Augmentation)** chuyên biệt cho ảnh X-quang toàn cảnh nha khoa. Các phương pháp tăng cường dữ liệu mới cần giải quyết triệt để 3 bài toán nan giải:
1. **Thiếu hụt dữ liệu & Mất cân bằng lớp trầm trọng**: Lớp tổn thương quanh chóp (`Periapical Lesion`) chỉ chiếm ~4.5% tổng số mẫu.
2. **Điểm nghẽn định vị tổn thương nhỏ**: Hơn 50% tổn thương thực tế bị bỏ sót (Missed GT) do các vết sâu răng (`Caries`) quá nhỏ so với độ phân giải lớn của toàn bức ảnh ($>2800 \times 1300$ px).
3. **Nhập nhằng ranh giới bệnh học và xảo ảnh**: Độ tương phản thấp, xảo ảnh kim loại và chồng lấp thân răng dẫn đến tỷ lệ dương tính giả (False Positives) rất cao.

---

## 2. HỆ THỐNG PHÂN LOẠI BỆNH LÝ & BÀI TOÁN MỤC TIÊU

Thử thách quốc tế **MICCAI DENTEX Challenge 2023** chia dữ liệu thành 3 cấp độ (Levels):
- **Level 1**: Phân vùng 4 cung hàm (Quadrant: 1..4).
- **Level 2**: Đánh số thứ tự răng theo chuẩn FDI (Enumeration: 1..8 cho mỗi quadrant, tổng 32 răng vĩnh viễn).
- **Level 3**: Chẩn đoán bệnh lý bất thường của răng (Abnormal Tooth Diagnosis).

Kho mã nguồn này chuẩn hóa nhiệm vụ trọng tâm: **Phát hiện và định vị 4 lớp bệnh lý bất thường (4-Class Dental Abnormality Detection)** dựa trên nhãn Level 3 của DENTEX:

| Class ID | Tên Lớp Bệnh Lý | Mã Màu Hiển Thị | Định Nghĩa & Đặc Điểm Lâm Sàng |
| :---: | :--- | :---: | :--- |
| **1** | `Caries` | `#E69F00` (Cam) | **Sâu răng**: Tổn thương mất khoáng, tiêu hủy mô cứng ở lớp men và lớp ngà nông. Thường có kích thước rất nhỏ, biểu hiện bằng vùng thấu quang (tối) cục bộ ở mặt nhai hoặc kẽ răng. |
| **2** | `Deep Caries` | `#D55E00` (Đỏ cam) | **Sâu răng sâu**: Tổn thương sâu răng tiến triển nặng, ăn sâu qua toàn bộ bề dày lớp ngà và tiếp cận sát hoặc làm thủng trần buồng tủy răng. |
| **3** | `Periapical Lesion` | `#0072B2` (Xanh dương) | **Tổn thương quanh chóp**: Vùng thấu quang do viêm nhiễm, u hạt hoặc nang quanh chóp răng ở vùng chóp chân răng, tiêu xương xốp xung quanh lỗ chóp. |
| **4** | `Impacted Tooth` | `#009E73` (Xanh lá) | **Răng ngầm / Răng mọc kẹt**: Răng không thể mọc lên vị trí giải phẫu bình thường do bị cản trở cơ học bởi xương hoặc răng bên cạnh (phổ biến nhất là răng khôn hàm dưới / hàm trên số 8). |

> **Quy tắc giải phẫu quan trọng:** Toàn bộ răng khỏe mạnh / răng bình thường được coi là **Background** (lớp nền) nhằm cô lập hoàn toàn bài toán vào việc khoanh vùng tổn thương bệnh lý và phân biệt mức độ trầm trọng.

---

## 3. HIỆN TRẠNG DỮ LIỆU (DATASET DEEP-DIVE)

### 3.1. Nguồn gốc, Bản quyền & Phương thức Tải
- **Tên tập dữ liệu gốc:** DENTEX (Dental Enumeration and Diagnosis on Panoramic X-rays).
- **Tổ chức chủ quản:** Ban tổ chức MICCAI DENTEX 2023 Challenge ([dentex.grand-challenge.org](https://dentex.grand-challenge.org/)).
- **Kho lưu trữ chính thức:** Hugging Face Hub tại `ibrahimhamamci/DENTEX` ([huggingface.co/datasets/ibrahimhamamci/DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)).
- **Cơ chế tải tự động trong Repo (`dentex_bench/data/downloader.py`):**
  - Tích hợp sẵn hàm `selective_extract_dentex_zip`: Chỉ tải và trích xuất đúng tập con `training_data` thuộc thư mục `quadrant-enumeration-disease` (~705 ảnh có nhãn bệnh lý).
  - Bỏ qua hơn 5,000 ảnh chỉ có nhãn Quadrant/Enumeration đơn thuần, giúp tiết kiệm hơn 10 GB bộ nhớ đĩa, chống lỗi đầy ổ cứng 20GB trên môi trường Kaggle/Colab.
  - Script chuyển đổi: `python scripts/download_and_prepare.py --seed 42`.

### 3.2. Cấu trúc Phân chia Tập dữ liệu (Train/Val Split)
- **Tổng số ảnh X-quang toàn cảnh có nhãn bệnh:** 705 ảnh (sau khi loại trừ ảnh lỗi hoặc ảnh không chứa tổn thương, tập train/val thực tế nạp 678–705 ảnh).
- **Quy tắc phân chia:** Tỷ lệ **80% Train / 20% Validation**, cố định `seed = 42`.
  - Phân chia theo cấp độ bức ảnh (`image_id` độc lập hoàn toàn, không có rò rỉ dữ liệu).
  - Kiểm tra tính toàn vẹn: Script `parser.py` chủ động từ chối nếu phát hiện trùng lặp `image_id`.
- **Thống kê phân chia cụ thể:**
  - **Tập Huấn luyện (Train):** 564 ảnh scan | 2,838 bounding boxes.
  - **Tập Kiểm định (Validation):** 141 ảnh scan | 691 bounding boxes.
  - **Tổng số nhãn tổn thương:** 3,529 bounding boxes.

### 3.3. Phân bố Lớp & Hiện tượng Mất cân bằng Cực đoan (Class Imbalance)
Dữ liệu thực tế cho thấy sự chênh lệch số lượng mẫu cực kỳ nghiêm trọng giữa các loại bệnh lý:

```
[Phân bố Số lượng Bounding Box theo Lớp Bệnh Lý]

Caries             ████████████████████████████████████████ 2,189 (62.0%)
Impacted Tooth     ███████████ 604 (17.1%)
Deep Caries        ██████████ 578 (16.4%)
Periapical Lesion  ███ 158 (4.5%)
```

| Tên Lớp Bệnh Lý | Số lượng Train | Số lượng Val | Tổng Cộng | Tỷ Lệ (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Caries** (Sâu răng) | 1,721 | 468 | **2,189** | 62.0% |
| **Impacted Tooth** (Răng ngầm) | 496 | 108 | **604** | 17.1% |
| **Deep Caries** (Sâu ngà sâu) | 493 | 85 | **578** | 16.4% |
| **Periapical Lesion** (Tổn thương chóp) | 128 | 30 | **158** | 4.5% |
| **Tổng cộng** | **2,838** | **691** | **3,529** | **100.0%** |

> ⚠️ **Thách thức cốt lõi:**  
> **Tỷ số mất cân bằng (Imbalance Ratio):** $2189 / 158 \approx \mathbf{13.85 : 1}$.  
> Lớp `Periapical Lesion` trong tập train chỉ có đúng 128 mẫu tổn thương trên toàn bộ 564 bệnh nhân! Đây là nguyên nhân trực tiếp khiến các mô hình học sâu rất khó học được đặc trưng vùng tiêu xương quanh chóp nếu không có tăng cường dữ liệu chuyên biệt (như Targeted Copy-Paste hoặc Generative Augmentation).

### 3.4. Thang phân loại Kích thước Bounding Box (COCO Size Taxonomy)
- **Độ phân giải ảnh gốc:** Rất lớn, trung bình $2800 \times 1300$ pixel, tỷ lệ khung hình ngang $\approx 2.15 : 1$.
- **Phân loại theo chuẩn COCO gốc ($Area = w \times h$):**
  - Small: $Area < 32^2$ ($< 1,024 \text{ px}^2$)
  - Medium: $32^2 \le Area < 96^2$ ($1,024 - 9,216 \text{ px}^2$)
  - Large: $Area \ge 96^2$ ($> 9,216 \text{ px}^2$)
- **Nghịch lý đo lường:** Do độ phân giải ảnh gốc quá cao, hầu như toàn bộ bounding box khi tính trên pixel gốc đều rơi vào diện tích $> 96^2$ px ($99.7\%$ Caries, $97.8\%$ Deep Caries, $100\%$ Periapical Lesion). Tuy nhiên, trên góc nhìn lâm sàng và tỷ lệ diện tích so với toàn ảnh ($\frac{Area_{box}}{Area_{image}}$):
  - `Caries` và `Deep Caries` chỉ chiếm $< 0.1\% - 0.5\%$ diện tích toàn bức ảnh. Khi ảnh bị thu nhỏ (resize) về kích thước đầu vào của mô hình ($800 \times 400$), các tổn thương sâu kẽ răng biến thành các chấm cực nhỏ ($10 \times 10$ px), dễ dàng bị tiêu biến qua các tầng pooling và downsampling của mạng nơ-ron tích chập.

### 3.5. Ma trận Đồng xuất hiện & Tiến trình Bệnh lý (Co-occurrence)
Khảo sát từ notebook `ran/dentex_eda_analysis.ipynb` cho thấy ma trận xác suất có điều kiện $P(\text{Cột} \mid \text{Hàng})$:

| Điều Kiện (Hàng) | Caries | Deep Caries | Periapical Lesion | Impacted Tooth |
| :--- | :---: | :---: | :---: | :---: |
| **Caries** | $100.0\%$ | $47.2\%$ | $15.2\%$ | $36.8\%$ |
| **Deep Caries** | $\mathbf{91.6\%}$ | $100.0\%$ | $24.0\%$ | $31.8\%$ |
| **Periapical Lesion** | $\mathbf{81.9\%}$ | $\mathbf{66.4\%}$ | $100.0\%$ | $37.9\%$ |
| **Impacted Tooth** | $90.2\%$ | $40.2\%$ | $17.3\%$ | $100.0\%$ |

> 💡 **Ý nghĩa lâm sàng sâu sắc:**  
> - Bệnh nhân có `Periapical Lesion` thì có tới **$66.4\%$** khả năng đồng xuất hiện `Deep Caries` và **$81.9\%$** đồng xuất hiện `Caries`.  
> - Đây là minh chứng thực nghiệm cho **tiến trình bệnh lý tự nhiên**: Sâu men/ngà nông $\rightarrow$ Sâu ngà sâu sát tủy $\rightarrow$ Hoại tử tủy $\rightarrow$ Tổn thương viêm lan xuống vùng chóp chân răng.  
> - Các kỹ thuật Tăng cường Dữ liệu tổng hợp (như MixUp hoặc Copy-Paste) phải tuân thủ mối quan hệ ngữ cảnh này: không thể dán một tổn thương chóp vào một răng hoàn toàn lành lặn không có sâu hoặc chưa từng điều trị tủy.

### 3.6. Phân bố Không gian Giải phẫu 2D (Spatial Heatmap)
Phân tích tọa độ tâm hộp chuẩn hóa $(x_c, y_c) \in [0, 1] \times [0, 1]$ trên toàn bộ cung hàm:
- `Impacted Tooth`: Tập trung đối xứng ở 4 góc của cung hàm (vùng góc hàm dưới và lồi củ xương hàm trên), tương ứng vị trí các răng số 8 (răng khôn).
- `Periapical Lesion`: Nằm phân bố dọc theo đường cung chân răng (root apex line), cách xa đường mặt nhai.
- `Caries` & `Deep Caries`: Phân bố dọc theo đường cong mặt nhai (occlusal plane) và các mặt tiếp cận (interproximal spaces) ở cả hai hàm.

### 3.7. Đặc trưng Quang phổ & Xảo ảnh X-quang (Radiographic Properties & Artifacts)
1. **Dải động tương phản (Dynamic Range):** Ảnh chụp X-quang toàn cảnh có dải giá trị xám tập trung mạnh vào các cấu trúc giải phẫu dày đặc (xương vỏ, răng) và thấu quang ở mô mềm/khoang khí.
2. **Cervical Burnout (Cháy cổ răng):** Hiện tượng thấu quang quang học sinh lý tại vùng cổ răng (nơi tiếp giáp giữa men răng và mào xương ổ răng), thường xuyên bị các mô hình thị giác máy tính nhận nhầm là `Caries` mặt bên.
3. **Photon Starvation Streak Artifacts:** Vệt sáng/tối hình nan hoa do các vật liệu phục hình kim loại cản quang mạnh (trám amalgam, mão chụp kim loại, trụ Implant titanium) gây nhiễu tán xạ trên toàn bộ cung răng.

---

## 4. HIỆN TRẠNG CÁC KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURES)

Mã nguồn tại `dentex_bench/models/` cung cấp một giao diện trừu tượng thống nhất (`BaseDetectorRunner`), bao bọc các kiến trúc từ kinh điển đến hiện đại:

```
                      ┌─────────────────────────────────────────┐
                      │    dentex_bench.models.base             │
                      │       BaseDetectorRunner                │
                      └────────────────────┬────────────────────┘
                                           │
         ┌───────────────────┬─────────────┴───────┬────────────────────┐
         ▼                   ▼                     ▼                    ▼
┌──────────────────┐┌──────────────────┐┌──────────────────┐┌──────────────────┐
│ HierarchicalNet  ││   DiffusionNet   ││    DINO DETR     ││   Faster R-CNN   │
│  (ResNet50+FPN)  ││  (DiffusionDet)  ││ (Transformer v2) ││  (ResNet50 v2)   │
└──────────────────┘└──────────────────┘└──────────────────┘└──────────────────┘
```

### 4.1. HierarchicalNet (v2 - Feature Pyramid Network + RoIAlign)
- **Mã nguồn:** `dentex_bench/models/hierarchical_net.py`
- **ID Kiến trúc:** `hierarchical_net/resnet50-fpn-rpn-roialign/v2`
- **Cơ chế hoạt động:**
  - **Backbone:** ResNet-50 trích xuất đặc trưng đa tầng $C_2, C_3, C_4, C_5$. Đóng băng stem và layer 1 (`freeze_backbone_stem`). Khởi tạo từ trọng số COCO Faster R-CNN v2.
  - **Feature Pyramid Network (FPN):** Xây dựng các tầng kim tự tháp đặc trưng $P_2, P_3, P_4, P_5, P_6$ với kênh ẩn đồng nhất $d = 256$.
  - **Region Proposal Network (RPN):** Sinh các vùng đề xuất (proposals) độc lập trên từng tầng kim tự tháp.
  - **MultiScaleRoIAlign:** Điểm mấu chốt của v2: Mỗi RoI được ánh xạ và trích xuất đặc trưng đúng từ tầng kim tự tháp tương ứng với kích thước của nó:
    $$k = \left\lfloor k_0 + \log_2 \left(\frac{\sqrt{w \cdot h}}{224}\right) \right\rfloor$$
    Nhờ đó, tổn thương nhỏ như Caries được trích xuất từ tầng độ phân giải cao ($P_2, P_3$), còn răng ngầm to được trích xuất từ tầng ngữ nghĩa sâu ($P_4, P_5$).
  - **Head:** Fast R-CNN Predictor dự đoán 4 lớp bệnh lý + 1 lớp background, đi kèm hộp hồi quy tọa độ.
  - **Tối ưu hóa:** AdamW, $lr = 10^{-4}$, weight decay $= 10^{-4}$.

### 4.2. DiffusionNet (v2 - DiffusionDet Paradigm + Dynamic Refinement)
- **Mã nguồn:** `dentex_bench/models/diffusion_net.py`
- **ID Kiến trúc:** `diffusion_net/diffusiondet-roialign-fpn/v2`
- **Triết lý:** Xem việc phát hiện hộp tổn thương nha khoa như một quá trình khử nhiễu sinh mẫu từ phân bố Gauss chuẩn (dựa trên DiffusionDet - Chen et al., ICCV 2023).
- **Cơ chế hoạt động:**
  - **Quá trình khuếch tán thuận (Forward Box Noising):** Chuẩn hóa tọa độ hộp ground-truth về $z_0 \in [0, 1]^4$. Tại bước khuếch tán ngẫu nhiên $t \in [0, 1000]$:
    $$q(z_t \mid z_0) = \mathcal{N}\left(z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$
    với lịch trình cosine $\bar{\alpha}_t$.
  - **Điều kiện hóa & Nhúng thời gian (Timestep FiLM):** Vector bước thời gian $t$ được mã hóa qua Sinusoidal Position Embeddings và MLP, điều chế đặc trưng RoI qua phép biến đổi FiLM (Scale & Shift).
  - **Cascade 3-Stage Dynamic Refinement:** Gồm 3 tầng tinh chỉnh liên tiếp:
    1. MultiScaleRoIAlign trích xuất đặc trưng tại chính vị trí các hộp đang bị nhiễu.
    2. Proposal Self-Attention (8 heads) tính toán sự tương tác giữa các hộp bệnh lý trên cùng cung hàm.
    3. Hồi quy gia lượng tọa độ $(\Delta x, \Delta y, \Delta w, \Delta h)$ và phân loại đa lớp.
  - **Loss Function:** Khớp cặp một-một (Hungarian matching) giữa dự đoán và nhãn thật:
    $$\mathcal{L}_{total} = \lambda_{cls} \mathcal{L}_{focal}(\hat{p}, c^*) + \lambda_{L1} \|\hat{z}_0 - z_0\|_1 + \lambda_{giou} \mathcal{L}_{giou}(\hat{z}_0, z_0)$$
  - **Suy luận (Reverse DDIM Sampling):** Khởi tạo 100 hộp ngẫu nhiên từ phân bố Gauss $\mathcal{N}(0, \mathbf{I})$, thực hiện khử nhiễu từng bước kèm cơ chế đổi mới hộp (box renewal) để lọc các hộp xác suất thấp.

### 4.3. DINO DETR (Vision Transformer Baseline)
- **Mã nguồn:** Đã thử nghiệm hoàn chỉnh trong `notebooks/cv-dinoresnet.ipynb`.
- **Đặc điểm:** Sử dụng MMDetection 3.3.0, mô hình DINO-R50-4scale với Contrastive DeNoising Training, bipartite Hungarian matching, xử lý đồng thời đa nhiệm: Quadrant (4), Enumeration (8), Diagnosis (4) và Tooth FDI (32 lớp).

### 4.4. Faster R-CNN & HierarchicalDet (Các mô hình đối chứng)
- `dentex_bench/models/faster_rcnn.py`: Mô hình 2 giai đoạn kinh điển làm mốc so sánh cơ sở.
- `dentex_bench/models/hierarchical_det.py`: Mô hình nguyên bản của tác giả DENTEX (v1), dùng Global Average Pooling toàn cục thay vì RoIAlign. Đã được thay thế bởi phiên bản v2 để loại trừ việc hộp không phụ thuộc vị trí tổn thương.

### 4.5. Cơ chế Tự thích ứng Phần cứng & Chống tràn VRAM (Hardware-Adaptive Engine)
Được triển khai trong `dentex_bench/env.py` và `dentex_bench/config.py`:
- Tự động phát hiện môi trường: Google Colab, Kaggle Notebooks, hoặc Máy trạm cục bộ (Windows / Linux).
- Thăm dò phần cứng GPU:
  - Nếu VRAM $\le 4$ GB (ví dụ RTX 3050 Laptop): Tự động đặt `batch_size = 1`, `gradient_accumulation_steps = 16`, `img_size = 800`, `num_workers = 0` (tránh lỗi Windows multiprocessing `BrokenPipeError`).
  - Nếu VRAM $\ge 15$ GB (Tesla T4 / P100 / A100): Kích hoạt FP16 mixed precision (`torch.amp.autocast`), kiểm tra và unscale gradient với `clip_grad_norm_ = 0.1`, bỏ qua các bước gradient có giá trị `NaN/Inf` để đảm bảo không bao giờ bị nổ loss (`loss: nan`).

---

## 5. HIỆN TRẠNG KẾT QUẢ THỰC NGHIỆM & PHÂN TÍCH LỖI (RESULTS & ERROR ANALYSIS)

### 5.1. Bảng Kết quả Thực nghiệm Định lượng

Dưới đây là số liệu thực tế được tổng hợp từ các lần chạy đã thực thi trong repo (`ran/dentex_inference_error_analysis.ipynb` và `notebooks/cv-dinoresnet.ipynb`):

| Chỉ Số Đánh Giá | DINO-R50 (MMDet - 81 epochs, DINO Flip/Crop) | HierarchicalNet (v2 - 30 epochs, No Augmentation) | DiffusionNet (v2 - 30 epochs, No Augmentation) |
| :--- | :---: | :---: | :---: |
| **Tổng số Epochs** | 81 (Early Stop tại ep 66) | 30 | 30 |
| **Data Augmentation** | DINO Random Flip + MultiScale Crop | **Không có (None)** | **Không có (None)** |
| **$mAP_{50}$ (Diagnosis)** | **$0.4427$ ($44.3\%$)** | N/A (F1 Max: $3.2\%$) | $0.000$ (Chưa hội tụ) |
| **$mAP_{50:95}$ (Diagnosis)**| **$0.2971$ ($29.7\%$)** | N/A | $0.000$ |
| **$mAP_{75}$ (Diagnosis)** | **$0.3756$ ($37.6\%$)** | N/A | $0.000$ |
| **Average Recall (AR)** | **$0.6920$ ($69.2\%$)** | N/A | $0.000$ |
| **Matched Accuracy (trên box khớp)** | **$92.0\%$** | N/A | N/A |
| **End-to-End Precision** | $54.11\%$ | $2.6\%$ | $0.0\%$ |
| **End-to-End Recall** | $48.08\%$ | $3.9\%$ | $0.0\%$ |
| **End-to-End F1 Score** | **$50.92\%$** | $3.2\%$ | $0.0\%$ |
| **Mean Matched IoU** | **$0.855$** | $0.59$ | N/A |
| **Số lượng Background FP** | Trung bình | **716** (Rất cao) | **0** (Lọc quá chặt) |
| **Độ trễ suy luận (Latency)** | ~95 ms | ~2849 ms (CPU/eval) | ~141 ms |

### 5.2. Điểm nghẽn Định vị (Localization Bottleneck)
Khảo sát chi tiết trong `notebooks/cv-dinoresnet.ipynb` (khi kiểm thử trên 50 ảnh test độc lập với 260 nhãn ground-truth):
- Tổng số nhãn thật: $260$
- Tổng số dự đoán sinh ra: $231$
- Số dự đoán khớp đúng vị trí ($IoU \ge 0.5$): $125$
- **Số nhãn thật bị mô hình bỏ sót hoàn toàn (Missed Ground-Truth):** $\mathbf{135}$ (chiếm **$51.92\%$**!)
- **Tỷ lệ Recall chỉ đạt $48.08\%$**: Hơn một nửa số bệnh lý trên ảnh X-quang không được mô hình phát hiện.
- **Nguyên nhân cốt lõi:** Khi một hộp đã được mô hình đề xuất đúng vị trí, độ chính xác chẩn đoán bệnh lên tới **$92.0\%$**. Như vậy, **bộ phân loại (Classifier) hoạt động rất tốt, nhưng khâu phát hiện vị trí (Detector / Localization) lại là điểm nghẽn trí mạng!**

### 5.3. Hồ sơ Top 5 Thất bại Lâm sàng (Top 5 Clinical Failure Modes)
Trích xuất từ kết quả chẩn đoán lỗi trong `ran/dentex_inference_error_analysis.ipynb`:

#### 🔴 Failure Mode 1: Nhầm lẫn giữa Caries và Deep Caries (EDJ Ambiguity)
- **Bản chất:** Nhập nhằng ranh giới men - ngà (Enamel-Dentin Junction).
- **Nguyên nhân:** Ảnh chụp 2D nén phẳng thể tích 3D phức tạp của thân răng, khiến ngưỡng phân biệt giữa mất khoáng ngà nông (`Caries` - Class 1) và sâu ngà tiến sát buồng tủy (`Deep Caries` - Class 2) bị mờ nhạt. Nếu không có phim cắn cánh (Bitewing) hoặc 3D CBCT, không thể khẳng định độ dày thành tủy còn lại.

#### 🔴 Failure Mode 2: Bỏ sót Tổn thương Nhỏ / Sơ khởi ($< 32^2$ px)
- **Bản chất:** Bỏ sót sâu men mặt nhai và sâu kẽ răng giai đoạn sớm.
- **Nguyên nhân:** Tổn thương sâu sớm chỉ là một vết thấu quang mờ nhạt trên lớp men. Quá trình downsampling qua các tầng tích chập và pooling làm triệt tiêu hoàn toàn gradient của các tổn thương nhỏ hơn $32 \times 32$ px.

#### 🔴 Failure Mode 3: Giả tổn thương Giải phẫu (Anatomical Pseudo-Lesions)
- **Bản chất:** Nhận nhầm các cấu trúc giải phẫu thấu quang bình thường thành `Periapical Lesion` (Class 3).
- **Nguyên nhân:** Lỗ cằm (mental foramen), ống răng dưới (mandibular canal), lỗ cửa (incisive foramen) và ngách xoang hàm trên (maxillary sinus recess) có hình ảnh thấu quang tròn rất giống nang chóp răng khi chiếu đè lên chóp chân răng cối nhỏ hoặc răng cửa.

#### 🔴 Failure Mode 4: Rung lắc Vị trí & Ranh giới Mờ (Boundary Ambiguity)
- **Bản chất:** Tiêu xương xốp quanh chóp dạng lan tỏa, không có viền vỏ xương rõ ràng.
- **Nguyên nhân:** Viêm quanh chóp mạn tính tạo ra vùng ranh giới chuyển tiếp từ từ vào mô xương lành, khiến tọa độ bounding box dự đoán bị dao động mạnh ($IoU$ chỉ đạt $0.20 - 0.49$, bị đánh trượt theo chuẩn $IoU \ge 0.5$).

#### 🔴 Failure Mode 5: Chồng lấp Thân răng & Xảo ảnh Phục hình Kim loại
- **Bản chất:** Hiệu ứng Cervical Burnout và nhiễu tán xạ kim loại.
- **Nguyên nhân:** Cung răng chen chúc tạo bóng chồng lấp quang học; các miếng trám amalgam hoặc mão răng kim loại gây hiện tượng "đói photon" (photon starvation), tạo các vệt đen nhân tạo cắt ngang qua cổ răng kế cận.

---

## 6. HIỆN TRẠNG CẤU TRÚC MÃ NGUỒN & QUY TRÌNH THỰC THI (CODEBASE & PIPELINES)

### 6.1. Cấu trúc Cây Thư mục
```
D:\Research\Dentex\
├── dentex_bench/                  # Gói thư viện lõi chuẩn hóa
│   ├── env.py                     # Nhận diện GPU, đo VRAM, thiết lập phần cứng
│   ├── config.py                  # Dataclass BenchmarkConfig, siêu tham số
│   ├── data/                      # Xử lý dữ liệu
│   │   ├── downloader.py          # Tải selective từ Hugging Face (tránh tốn 10GB)
│   │   ├── parser.py              # Phân tích cú pháp DENTEX sang chuẩn COCO 4 class
│   │   └── dataset.py             # PyTorch DentexDataset, load ảnh, resize đa tỷ lệ
│   ├── models/                    # Bộ điều khiển các mô hình
│   │   ├── base.py                # Lớp cơ sở BaseDetectorRunner
│   │   ├── hierarchical_net.py    # ResNet-50 + FPN + MultiScaleRoIAlign (v2)
│   │   ├── diffusion_net.py       # Denoising Diffusion Detector (v2)
│   │   ├── dino.py                # Adapter cho kiến trúc DINO DETR
│   │   ├── faster_rcnn.py         # Baseline Faster R-CNN v2
│   │   └── hierarchical_det.py    # Baseline Diffusion v1 (đã deprecated)
│   └── eval/                      # Đánh giá và báo cáo
│       ├── coco_eval.py           # Đánh giá COCO mAP50, mAP50:95, per-class AP
│       └── reporter.py            # Xuất bảng Markdown, LaTeX và đồ thị so sánh
│
├── scripts/                       # Các tập lệnh CLI thực thi độc lập
│   ├── setup_env.py               # Kiểm tra dependencies và môi trường
│   ├── download_and_prepare.py    # Tải dữ liệu và sinh instances_{train,val}.json
│   ├── train.py                   # Script huấn luyện thống nhất (--model ...)
│   ├── evaluate.py                # Script đánh giá checkpoint thống nhất
│   ├── build_eda_notebook.py      # Sinh tự động notebooks/dentex_eda_analysis.ipynb
│   ├── build_inference_notebook.py# Sinh tự động notebooks/dentex_inference_error_analysis.ipynb
│   └── sdd_helpers.py             # Tiện ích phục vụ Subagent-Driven Development
│
├── notebooks/                     # Jupyter Notebooks tương tác
│   ├── 01_colab_run_all.ipynb     # Chạy toàn bộ quy trình trên Google Colab
│   ├── 02_kaggle_run_all.ipynb    # Chạy toàn bộ quy trình trên Kaggle GPU P100/T4
│   ├── 03_local_benchmark.ipynb   # Chạy kiểm thử trên máy trạm cục bộ
│   ├── dentex_eda_analysis.ipynb  # Phân tích khám phá dữ liệu (EDA) chuẩn xuất bản
│   ├── dentex_inference_error_analysis.ipynb # Đánh giá lỗi suy luận chi tiết
│   └── cv-dinoresnet.ipynb        # Bản mẫu đối chiếu huấn luyện DINO đa nhiệm
│
├── ran/                           # Thư mục lưu các notebook đã chạy thực tế có output
│   ├── 01_colab_run_all.ipynb
│   ├── dentex_eda_analysis.ipynb
│   └── dentex_inference_error_analysis.ipynb
│
├── outputs/                       # Lưu checkpoints và kết quả
│   ├── hierarchical_net/best.pth  # Checkpoint tốt nhất của HierarchicalNet (173 MB)
│   └── diffusion_net/best.pth     # Checkpoint tốt nhất của DiffusionNet (289 MB)
│
├── tests/                         # Hệ thống kiểm thử tự động (Unit Tests)
│   ├── test_data_pipeline.py      # Kiểm tra parser, dataset, resize
│   ├── test_hierarchical_net.py   # Kiểm tra forward/backward HierarchicalNet
│   ├── test_diffusion_net.py      # Kiểm tra forward/backward DiffusionNet
│   ├── test_detector_learning.py  # Kiểm tra khả năng overfit trên batch nhỏ
│   └── test_integrity.py          # Kiểm tra tính toàn vẹn, chống kết quả ảo
│
├── requirements.txt               # Danh sách thư viện phụ thuộc
└── README.md                      # Hướng dẫn sử dụng chung
```

### 6.2. Quy trình Thực thi Dòng lệnh (CLI Workflows)

```bash
# Bước 1: Chuẩn bị môi trường
python scripts/setup_env.py

# Bước 2: Tải và chuyển đổi dữ liệu sang COCO format 4 lớp
python scripts/download_and_prepare.py --seed 42

# Bước 3: Huấn luyện mô hình mục tiêu (30 epochs, memory-safe)
python scripts/train.py --model hierarchical_net --epochs 30 --lr 0.0001
python scripts/train.py --model diffusion_net --epochs 30 --lr 0.0001

# Bước 4: Đánh giá độc lập trên tập Validation
python scripts/evaluate.py --compare_all
# Hoặc đánh giá từng checkpoint:
python scripts/evaluate.py --model hierarchical_net --checkpoint outputs/hierarchical_net/best.pth
```

### 6.3. Hệ thống Notebook Tương tác & Tự động Sinh mã
- Bộ đôi notebook `dentex_eda_analysis.ipynb` và `dentex_inference_error_analysis.ipynb` được duy trì bởi 2 generator script (`build_eda_notebook.py` và `build_inference_notebook.py`).
- Mọi con số, đồ thị trong notebook được tính toán trực tiếp từ dữ liệu thực, không sử dụng số liệu giả lập.

### 6.4. Lịch sử Nâng cấp Quan trọng (v1 sang v2 - Commit `374edc8`)
Trong các phiên bản đầu tiên của repo:
- Cả hai mô hình áp dụng Global Average Pooling (GAP) trên toàn bộ bản đồ đặc trưng, dẫn đến việc các tọa độ dự đoán không phụ thuộc vào vị trí thực tế của tổn thương.
- `evaluate.py` in ra bảng kết quả tĩnh cố định thay vì đánh giá checkpoint thật.
- **Nâng cấp tại v2:**
  - Chuyển đổi hoàn toàn sang **MultiScaleRoIAlign** kết hợp pyramid đa tầng cho cả HierarchicalNet và DiffusionNet.
  - Tích hợp `ARCH_ID` trong checkpoint: từ chối nạp checkpoint cũ không tương thích.
  - Chọn `best.pth` dựa trên $mAP_{50}$ thực tế trên tập validation thay vì loss huấn luyện.

---

## 7. KHOẢNG TRỐNG HIỆN TẠI & ĐỀ XUẤT NGHIÊN CÚU TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION GAP)

### 7.1. Đánh giá Hiện trạng Pipeline Augmentation trong Repo
Kiểm tra trực tiếp file `dentex_bench/data/dataset.py` và `scripts/train.py`:
```python
# scripts/train.py (Dòng 106)
dataset = DentexDataset(img_dir=img_dir, ann_file=train_ann, target_short=cfg.img_size, max_long=cfg.max_size)
# LƯU Ý QUAN TRỌNG: transforms KHÔNG ĐƯỢC TRUYỀN VÀO (Mặc định transforms=None)!
```
👉 **Phát hiện quan trọng:** Trong pipeline huấn luyện CLI chính thức (`scripts/train.py`), **chưa hề có bất kỳ kỹ thuật Data Augmentation nào được áp dụng!** Quá trình tiền xử lý hiện tại chỉ bao gồm:
1. Đổi kích thước ảnh giữ nguyên tỷ lệ (`target_short=800, max_long=1333`).
2. Chuyển đổi mảng numpy sang PyTorch float tensor và chuẩn hóa chia cho $255.0$.

Trong khi đó, ở file thử nghiệm `notebooks/cv-dinoresnet.ipynb`, tác giả đã phải tự viết thủ công các hàm `flip_dentex`, `crop_dentex`, `resize_dentex` để giúp DINO hội tụ đạt $mAP_{50} = 44.3\%$.

### 7.2. Các Ràng buộc Giải phẫu & Vật lý X-quang (Invariance Rules)
Khi thiết kế Data Augmentation cho ảnh X-quang toàn cảnh (OPG), **Agent Research cần tuyệt đối tuân thủ các quy tắc bất biến hình học và quang sai sau:**

| Phép Biến Đổi | Được Phép? | Ràng Buộc Kỹ Thuật & Lý Do Lâm Sàng |
| :--- | :---: | :--- |
| **Horizontal Flip** (Lật ngang) |  **Được** | Cung hàm người có tính đối xứng song phương. Tổn thương ở bên trái hoàn toàn có thể xảy ra ở bên phải. Tuy nhiên, nếu là bài toán đa nhiệm có nhãn Cung hàm (Quadrant) thì phải tráo đổi $Q_1 \leftrightarrow Q_2$ và $Q_3 \leftrightarrow Q_4$. Đối với bài toán 4 lớp bệnh lý, lật ngang là an toàn tuyệt đối. |
| **Vertical Flip** (Lật dọc) | ❌ **CẤM** | **Phi lâm sàng hoàn toàn!** Xương hàm trên (Maxilla) cố định với sàn xoang hàm, trong khi xương hàm dưới (Mandible) là xương di động có bờ đáy xương dày. Lật dọc sẽ đảo lộn toàn bộ giải phẫu răng và hướng trọng lực mọc răng. |
| **Rotation** (Xoay góc lớn) | ⚠️ **Hạn chế** | Chỉ cho phép xoay góc nhỏ **$\pm 3^\circ \rightarrow \pm 5^\circ$** mô phỏng tư thế nghiêng đầu của bệnh nhân khi chụp. Cấm xoay góc lớn ($> 15^\circ$) vì tư thế máy chụp OPG quay quanh trục đầu cố định. |
| **Heavy Blur / Gaussian Smoothing** | ❌ **CẤM** | Làm mờ nặng sẽ xóa sổ hoàn toàn các ranh giới thấu quang tinh tế của tổn thương sâu men (`Caries` $< 32^2$ px). |
| **Elastic Distortion** | ⚠️ **Hạn chế** | Biến dạng co giãn mạnh sẽ làm méo mó chân răng và xương ổ răng, tạo ra các tổn thương giả. |

### 7.3. Các Hướng Tăng Cường Dữ Liệu Đề Xuất Cho Agent Research

Dưới đây là 6 hướng nghiên cứu Data Augmentation có tiềm năng đột phá cao nhất dành cho bài toán này:

```
                                  CÁC HƯỚNG DATA AUGMENTATION ĐỀ XUẤT
                                                  │
         ┌───────────────────┬────────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                    ▼                   ▼                   ▼
┌──────────────────┐┌──────────────────┐┌──────────────────┐┌──────────────────┐┌──────────────────┐
│ 1. Cân Bằng Lớp  ││ 2. Đa Tương Phản ││ 3. Cắt Vá Răng   ││ 4. Siêu Phân Giải││ 5. Mô Phỏng Xảo  │
│  Copy-Paste Có   ││ CLAHE, Retinex,  ││ Mosaic / MixUp   ││ Patch-based Zoom ││ Ảnh Kim Loại &   │
│   Chọn Lọc       ││ Unsharp Masking  ││ Theo Cung Hàm    ││ & Crop Cục Bộ    ││ Cervical Burnout │
└──────────────────┘└──────────────────┘└──────────────────┘└──────────────────┘└──────────────────┘
```

#### 1. Targeted Lesion Copy-Paste (Giải quyết Mất cân bằng lớp 13.85:1)
- **Vấn đề:** `Periapical Lesion` chỉ có 128 mẫu train, `Deep Caries` chỉ có 493 mẫu.
- **Giải pháp:** Cắt các mảng tổn thương quanh chóp (kèm phần chân răng lân cận) và dán vào các vị trí cuống răng tương ứng của những ảnh khác; cắt tổn thương sâu ngà dán vào mặt nhai của các răng lành.
- **Ràng buộc:** Phải dùng thuật toán hòa trộn biên (Poisson Blending hoặc Gaussian edge feathering) để không tạo vết cắt nhân tạo sắc nhọn quanh bounding box.

#### 2. Radiographic Contrast & Frequency Enhancement (Làm nổi bật Caries nhỏ)
- **Vấn đề:** Sâu răng sớm có độ tương phản cực kỳ thấp so với ngà răng lành.
- **Giải pháp:**
  - **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Cân bằng lược đồ độ xám cục bộ, làm nổi rõ đường ranh giới thấu quang.
  - **Unsharp Masking & High-Frequency Boost:** Khuếch đại các thành phần tần số cao (edge gradient) trước khi đưa vào backbone.
  - **Multi-vendor Gamma Simulation:** Mô phỏng sự biến thiên gamma giữa các dòng máy chụp X-quang của các hãng sản xuất khác nhau (Carestream, Dentsply Sirona, Planmeca).

#### 3. Anatomically-Consistent Patch Cropping & Zooming (Khắc phục Downsampling)
- **Vấn đề:** Resize ảnh từ $2800$ px về $800$ px làm mất thông tin chi tiết.
- **Giải pháp:** Trong quá trình huấn luyện, bên cạnh ảnh toàn cảnh thu nhỏ, cắt ngẫu nhiên các patch cục bộ kích thước $800 \times 800$ ở độ phân giải gốc quanh các vùng răng có tổn thương, ép mạng học đặc trưng vi mô của tổn thương ở độ phân giải cao nhất.

#### 4. Context-Aware MixUp / Mosaic cho Cung Hàm
- **Vấn đề:** Mô hình dễ bị overfit vào nền giải phẫu của từng bệnh nhân cụ thể (dẫn tới 716 False Positives).
- **Giải pháp:** Cắt ghép nửa hàm trái của bệnh nhân A với nửa hàm phải của bệnh nhân B (ghép đối xứng theo đường giữa xương hàm), tạo ra các biến thể cung hàm phong phú mà vẫn bảo toàn cấu trúc giải phẫu học.

#### 5. Artifact & Noise Simulation (Tăng độ bền vững trước xảo ảnh lâm sàng)
- **Vấn đề:** Xảo ảnh vệt kim loại và hiện tượng cháy cổ răng gây dương tính giả.
- **Giải pháp:** Chủ động tổng hợp các vệt tán xạ photon nhân tạo (streak artifacts) và hiệu ứng giảm sáng tại vùng cổ răng (cervical burnout simulation) để ép mô hình học cách phân biệt giữa xảo ảnh quang học và tổn thương sâu răng thật sự.

#### 6. Generative AI / Diffusion-based Inpainting
- **Vấn đề:** Khó khăn trong việc mô phỏng các tổn thương tiêu xương phức tạp.
- **Giải pháp:** Sử dụng mô hình khuếch tán tạo ảnh (như ControlNet hoặc Stable Diffusion inpainting đã fine-tune trên ảnh X-quang) để sinh tổng hợp các tổn thương sâu răng và nang quanh chóp mới có điều kiện theo mặt nạ giải phẫu.

---

## 8. DANH MỤC TÀI LIỆU THAM KHẢO, TRÍCH DẪN & NGUỒN GỐC (REFERENCES & CITATIONS)

### 8.1. Các Bài Báo Khoa Học Gốc & Benchmark Nền Tảng

1. **Bài báo gốc công bố DENTEX & Mô hình Hierarchical Baseline:**
   - **Tiêu đề:** *Diffusion-Based Hierarchical Multi-Label Object Detection to Analyze Panoramic Dental X-rays*
   - **Tác giả:** Ibrahim Ethem Hamamci, Sezgin Er, Enis Simsar, Anjany Kumar Sekuboyina, Mustafa Gundogar, Bernd Stadlinger, Albert C. Mehl, Bjoern H. Menze.
   - **Hội nghị / Lưu trữ:** MICCAI 2023 / arXiv preprint (2023).
   - **arXiv:** [arXiv:2303.06500](https://arxiv.org/abs/2303.06500)
   - **Mã nguồn chính thức:** [github.com/ibrahimethemhamamci/HierarchicalDet](https://github.com/ibrahimethemhamamci/HierarchicalDet)

2. **Bài báo Benchmark Toàn diện của Thử thách DENTEX:**
   - **Tiêu đề:** *DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-ray*
   - **Tác giả:** Ibrahim Ethem Hamamci et al.
   - **Lưu trữ:** arXiv preprint (2023).
   - **arXiv:** [arXiv:2305.19112](https://arxiv.org/abs/2305.19112)
   - **Trang web chính thức của thử thách:** [dentex.grand-challenge.org](https://dentex.grand-challenge.org/)

3. **Bài báo về Tự giám sát & Khôi phục Đặc trưng Kết cấu Nha khoa (SSAD):**
   - **Tiêu đề:** *SSAD: Self-supervised Auxiliary Detection Framework for Panoramic X-ray based Dental Disease Diagnosis*
   - **Tác giả:** Zijian Cai, Xinquan Yang, Xuguang Li, Xiaoling Luo, Xuechen Li, Linlin Shen, He Meng, Yongqiang Deng.
   - **Lưu trữ:** arXiv preprint (Tháng 6, 2024).
   - **arXiv:** [arXiv:2406.13963](https://arxiv.org/abs/2406.13963)

4. **Kiến trúc Mô hình Khuếch tán Phát hiện Đối tượng (DiffusionDet):**
   - **Tiêu đề:** *DiffusionDet: Diffusion Model for Object Detection*
   - **Tác giả:** Shoufa Chen, Peize Sun, Yida Peng, Wenqi Zhu, Luowei Zhou, Jianshu Li, Zheyuan Liu, Tong Lu, Fuchun Sun, Ping Luo.
   - **Hội nghị:** IEEE/CVF International Conference on Computer Vision (ICCV 2023).
   - **arXiv:** [arXiv:2211.09788](https://arxiv.org/abs/2211.09788)
   - **Mã nguồn:** [github.com/ShoufaChen/DiffusionDet](https://github.com/ShoufaChen/DiffusionDet)

5. **Kiến trúc Vision Transformer Phát hiện Đối tượng (DINO):**
   - **Tiêu đề:** *DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection*
   - **Tác giả:** Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum.
   - **Hội nghị:** International Conference on Learning Representations (ICLR 2023).
   - **arXiv:** [arXiv:2203.03605](https://arxiv.org/abs/2203.03605)
   - **Mã nguồn:** [github.com/IDEA-Research/DINO](https://github.com/IDEA-Research/DINO)

6. **Mô hình Cơ sở Faster R-CNN v2 & FPN:**
   - **Tiêu đề:** *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*
   - **Tác giả:** Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.
   - **Hội nghị:** Advances in Neural Information Processing Systems (NeurIPS 2015).
   - **Triển khai chuẩn:** PyTorch Torchvision Detection Models (`torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`).

### 8.2. Kho Dữ Liệu & Tài Nguyên Trực Tuyến

- **DENTEX Dataset trên Hugging Face:**  
  [https://huggingface.co/datasets/ibrahimhamamci/DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)
- **DENTEX Challenge Grand Challenge Portal:**  
  [https://dentex.grand-challenge.org/](https://dentex.grand-challenge.org/)
- **Kho mã nguồn DENTEX Official GitHub:**  
  [https://github.com/ibrahimethemhamamci/DENTEX](https://github.com/ibrahimethemhamamci/DENTEX)

### 8.3. Mã Trích Dẫn BibTeX

```bibtex
@article{hamamci2023diffusion,
  title={Diffusion-based hierarchical multi-label object detection to analyze panoramic dental X-rays},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Simsar, Enis and Sekuboyina, Anjany Kumar and Gundogar, Mustafa and Stadlinger, Bernd and Mehl, Albert C and Menze, Bjoern H},
  journal={arXiv preprint arXiv:2303.06500},
  year={2023}
}

@article{hamamci2023dentex,
  title={DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-ray},
  author={Hamamci, Ibrahim Ethem and others},
  journal={arXiv preprint arXiv:2305.19112},
  year={2023}
}

@article{cai2024ssad,
  title={SSAD: Self-supervised Auxiliary Detection Framework for Panoramic X-ray based Dental Disease Diagnosis},
  author={Cai, Zijian and Yang, Xinquan and Li, Xuguang and Luo, Xiaoling and Li, Xuechen and Shen, Linlin and Meng, He and Deng, Yongqiang},
  journal={arXiv preprint arXiv:2406.13963},
  year={2024}
}

@inproceedings{chen2023diffusiondet,
  title={DiffusionDet: Diffusion Model for Object Detection},
  author={Chen, Shoufa and Sun, Peize and Peng, Yida and Zhu, Wenqi and Zhou, Luowei and Li, Jianshu and Liu, Zheyuan and Lu, Tong and Sun, Fuchun and Luo, Ping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={19830--19843},
  year={2023}
}

@inproceedings{zhang2023dino,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Zhang, Hao and Li, Feng and Liu, Shilong and Zhang, Lei and Su, Hang and Zhu, Jun and Ni, Lionel M and Shum, Heung-Yeung},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

---
*Báo cáo được biên soạn và chuẩn hóa phục vụ trực tiếp cho quá trình nghiên cứu và thử nghiệm Tăng Cường Dữ Liệu (Data Augmentation) trên mô hình chẩn đoán hình ảnh nha khoa DENTEX.*

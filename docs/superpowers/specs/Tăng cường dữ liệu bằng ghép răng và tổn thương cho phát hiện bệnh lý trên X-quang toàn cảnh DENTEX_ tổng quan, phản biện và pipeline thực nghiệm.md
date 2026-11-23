# Tăng cường dữ liệu bằng ghép răng/tổn thương (tooth & lesion copy-paste) cho phát hiện bệnh lý trên X-quang toàn cảnh DENTEX: tổng quan, phản biện và pipeline thực nghiệm

Ý tưởng của bạn có cơ sở và đã có tiền lệ trực tiếp, nhưng cần đổi trọng tâm: ghép răng **khỏe mạnh** chỉ tạo thêm background (hữu ích để giảm false positive), còn muốn mô hình học tốt ca **hiếm** (Periapical Lesion, Impacted dạng lạ) thì phải ghép **răng mang bệnh** (răng + vùng chóp) vào **đúng vị trí FDI** ở bệnh nhân khác, kèm hòa trộn (blending) có kiểm soát và so sánh với repeat factor sampling.

## TL;DR

- **Đã có công trình tương tự nhưng chưa ai đo riêng hiệu quả cho phát hiện bệnh:** van Nistelrooij et al. (BMC Oral Health 2024, dùng chính DENTEX) đã copy-paste răng sang vị trí đối xứng cùng số FDI; Kadi et al. (arXiv 2402.04408) ghép răng thật vào panoramic "rỗng" tạo bởi Stable Diffusion inpainting; trong ảnh y tế, TumorCP, CarveMix, LesionMix và Hu et al. (CVPR 2023) chứng minh lesion copy-paste/tổng hợp tổn thương hiệu quả. Tuy nhiên chưa tìm thấy bài nào có ablation tách riêng copy-paste cho phát hiện caries/periapical trên panoramic → đây là khoảng trống hợp lý cho tiểu luận (tính mới ở mức "kết hợp + đánh giá có hệ thống", không phải phát minh thuật toán).
- **Phản biện cốt lõi:** răng khỏe là background trong bài toán 4 lớp của bạn, nên ghép răng khỏe không tăng mẫu dương tính cho lớp hiếm; ngoài ra DENTEX chỉ gán nhãn răng bệnh nên "răng khỏe" có thể chứa bệnh chưa gán nhãn (label noise). Điểm nghẽn hiện tại là localization (51.9% GT bị bỏ sót) chứ không phải classification, nên ưu tiên tăng số lượng và ngữ cảnh của tổn thương, không chỉ tăng "độ đẹp" của ảnh.
- **MVP nên làm trước:** (1) tạo ngân hàng răng-bệnh từ polygon có sẵn trong DENTEX (chỉ tập train); (2) "FDI tooth-swap" thay răng cùng số ở ảnh khác, scale theo kích thước răng đích, histogram matching + Gaussian feathering; (3) ablation 5–7 cấu hình (baseline, +RFS, +random paste, +FDI-swap, × 3 kiểu blending) trên YOLOv8/YOLO11 hoặc Faster R-CNN với ≥3 seed, báo cáo AP theo lớp (đặc biệt Periapical), recall và số FP.

## Key Findings

1. **Copy-paste là kỹ thuật đã được kiểm chứng cho lớp hiếm.** Ghiasi et al. (CVPR 2021) cho thấy dán ngẫu nhiên đã đủ mạnh: đạt 49.1 mask AP / 57.3 box AP trên COCO và vượt đội thắng LVIS Challenge 2020 +3.6 mask AP trên nhóm lớp hiếm.\[1\]\[2\] Nhưng Dvornik et al. (ECCV 2018) lại cho thấy dán sai ngữ cảnh có thể làm hại khi dữ liệu ít — và X-quang răng là miền mà ngữ cảnh giải phẫu cực kỳ chặt.\[3\] Hai kết quả này không mâu thuẫn: dữ liệu càng ít và ngữ cảnh càng chặt thì ràng buộc vị trí càng quan trọng.
2. **Trong y tế, "lesion-aware" và "spatially-aware" là chìa khóa.** TumorCP (MICCAI 2021) tăng 7.12% tumor Dice so với baseline mạnh, và tăng 21.87% khi chỉ có 10% dữ liệu có nhãn;\[4\]\[5\] LesionMix chỉ ra CarveMix có thể đặt tổn thương ra ngoài cơ quan, nên đề xuất bản đồ xác suất vị trí tổn thương.\[6\] Đây chính là tương đương của "đặt theo FDI/heatmap" trong bài toán của bạn.
3. **Trong nha khoa, ý tưởng ghép răng đã xuất hiện nhưng chỉ như một thành phần phụ.** van Nistelrooij et al. dùng copy-paste răng sang vị trí đối xứng cùng số FDI khi huấn luyện Mask DINO, nhưng không có ablation riêng;\[7\] Kadi et al. tạo 298 panoramic tổng hợp bằng cách ghép răng vào ảnh "rỗng", đạt mAP 0.76 so với 0.69 khi không augment và 0.82 với augmentation hình học — tức là ảnh ghép giúp nhưng chưa vượt augmentation cổ điển, và tác giả ghi nhận xảo ảnh và chồng lấp do dùng nguyên tọa độ gốc.\[8\]
4. **Các đội top DENTEX 2023 không dùng copy-paste làm vũ khí chính.** Đội hạng 1 (He et al., SJTU) dùng flip/crop/rotation cho mô hình phân đoạn và cấu hình mặc định cho detector, cộng ensemble WBF và bộ lọc tri thức "răng ngầm luôn là răng số 8"; YOLOrtho (Chohotech) dùng "flip mapping" đổi nhãn quadrant khi lật ngang và pseudo-label răng khỏe; Sdent (Choi et al.) pseudo-label ảnh chưa gán nhãn để nhân đôi dữ liệu Periapical và Deep Caries, nâng Disease AP50 nội bộ từ 0.610 lên 0.668.\[9\]\[10\]\[11\] Bài học: tăng **số mẫu dương tính của lớp hiếm** có hiệu quả đo được.
5. **DENTEX có polygon segmentation cho từng răng**, nên bạn không bắt buộc phải tách răng từ bbox bằng GrabCut/SAM — đây là thuận lợi lớn cho tooth-swap.\[10\]\[12\] Kỹ thuật tách răng cổ điển (Otsu, GrabCut, morphology) vẫn đáng đưa vào tiểu luận như một nhánh so sánh hoặc để tinh chỉnh biên polygon.
6. **Có giới hạn tối ưu cho lượng dữ liệu tổng hợp.** Nghiên cứu tổng hợp nốt phổi bằng diffusion trên X-quang ngực (arXiv 2603.01659, preprint 03/2026) ghi nhận "performance improvements plateau or slightly decline beyond an optimal level of augmentation", nên tỷ lệ ảnh ghép phải là một siêu tham số cần quét, không phải "càng nhiều càng tốt".

## Details

### Phần A — Tổng quan tài liệu và mức độ giống ý tưởng của bạn

#### A1. Copy-paste và các phương pháp trộn ảnh tổng quát

| Công trình | Ý chính | Kết quả định lượng nổi bật | Mức giống ý tưởng | Khác biệt chính |
|---|---|---|---|---|
| Cut, Paste and Learn — Dwibedi, Misra, Hebert (ICCV 2017, arXiv 1708.01642) | Cắt instance, dán lên nền; tạo cùng một cảnh với nhiều chế độ blending (none, Gaussian blur, Poisson) để mô hình bất biến với xảo ảnh biên | Thêm distractor giúp mô hình không bám vào xảo ảnh biên và tăng 3 AP\[13\] | Trung bình–cao (ý tưởng "đa blending" áp dụng trực tiếp) | Ảnh tự nhiên, dán ngẫu nhiên, không có ràng buộc giải phẫu |
| Modeling Visual Context — Dvornik, Mairal, Schmid (ECCV 2018, arXiv 1807.07428) | CNN dự đoán vùng phù hợp để đặt vật thể | Không có mô hình ngữ cảnh thì dán ngẫu nhiên "actually hurts"; có ngữ cảnh thì đạt "significant mean average precision improvements" trên VOC'12 khi ít nhãn (abstract không nêu số cụ thể) | Cao về triết lý (đặt theo ngữ cảnh) | Ngữ cảnh học bằng CNN; với răng, ngữ cảnh có thể mã hóa thủ công bằng FDI |
| Simple Copy-Paste — Ghiasi et al. (CVPR 2021, arXiv 2012.07177) | Dán ngẫu nhiên + scale jitter lớn, không cần mô hình ngữ cảnh | COCO 49.1 mask AP / 57.3 box AP; LVIS +3.6 mask AP lớp hiếm\[1\]\[2\]\[14\] | Trung bình | Dữ liệu lớn, đối tượng độc lập ngữ cảnh; không chắc đúng với X-quang |
| X-Paste (arXiv 2212.03863) | Sinh instance bằng Stable Diffusion/CLIP rồi copy-paste | LVIS lớp hiếm: +3.2 box AP so với Copy-Paste\[15\] | Thấp–trung bình | Sinh instance mới thay vì chỉ dùng lại instance có sẵn |
| Mosaic (YOLOv4), CutMix, MixUp | Trộn ảnh/khối chữ nhật | Chuẩn mặc định trong YOLO | Thấp | Không theo đối tượng; mosaic phá vỡ cấu trúc cung hàm |

Nhận xét: InstaBoost và ObjectAug cũng thuộc nhóm "di chuyển/biến đổi instance trong chính ảnh", gần với biến thể "intra-patient" của TumorCP; trong phạm vi thời gian, tôi chưa kiểm tra lại số liệu gốc của hai bài này nên không trích con số.

#### A2. Copy-paste / tổng hợp tổn thương trong ảnh y tế

| Công trình | Miền | Cách làm | Kết quả | Bài học cho bạn |
|---|---|---|---|---|
| TumorCP — Yang et al. (MICCAI 2021, arXiv 2107.09843) | CT thận (KiTS) | Copy-paste khối u intra-patient và inter-patient, kèm biến đổi không gian, gamma, blur | Chỉ intra-patient: +6.24% Dice; đầy đủ: +7.12% Dice; 10% dữ liệu: +21.87%\[16\]\[17\] | Copy-paste tổn thương thật hiệu quả mạnh nhất khi dữ liệu ít — đúng tình huống Periapical (128 mẫu train) |
| CarveMix — Zhang et al. (MICCAI 2021, arXiv 2108.06883) | MRI não | Khắc vùng ROI quanh tổn thương theo hình dạng tổn thương rồi thay vào ảnh khác | Cải thiện Dice so với các augmentation trộn ảnh khác\[18\] | "Răng + vùng chóp" là một ROI lesion-aware tự nhiên |
| LesionMix (arXiv 2308.09026) | MRI não, CT gan | Thêm bản đồ xác suất vị trí, biến đổi hình dạng/cường độ ở mức tổn thương, có thể inpaint tổn thương | Chỉ ra CarveMix có thể đặt tổn thương ra ngoài cơ quan\[6\] | Cần heatmap vị trí / ràng buộc FDI |
| Label-Free Liver Tumor Segmentation — Hu et al. (CVPR 2023, arXiv 2303.14869) | CT gan | Tổng hợp khối u thủ công (hình dạng + texture theo phân phối Gaussian HU) | DSC 59.81% trên khối u thật; vượt visual Turing test với 2 chuyên gia (dưới 50% khối u giả bị nhận ra)\[19\]\[20\] | Mẫu chuẩn cho chương "đánh giá độ chân thật" |
| Tổng hợp nốt phổi trên X-quang ngực (diffusion, arXiv 2603.01659; inpainting MICCAI-W) | X-quang 2D | Chèn nốt tổng hợp vào X-quang | AUC 0.90 (JSRT), 0.93 (ChestX-ray14); hiệu năng chững/giảm nếu augment quá mức\[21\] | X-quang 2D chiếu chồng giống panoramic; phải quét tỷ lệ tổng hợp |

#### A3. Riêng nha khoa

| Công trình | Dữ liệu | Kỹ thuật | Kết quả | Mức giống |
|---|---|---|---|---|
| van Nistelrooij et al., BMC Oral Health 24:387 (2024) | DENTEX + OdontoAI | Copy-paste một răng chồng lên răng đối xứng cùng số FDI (Mask DINO); copy-paste răng sang ảnh phân loại khác cùng FDI (Swin-B) | Tooth segmentation mAP 0.848; F1 chẩn đoán: impacted 0.901, deep caries 0.683, early caries 0.662, periapical 0.603;\[7\] không có ablation riêng cho copy-paste | **Rất cao** (gần nhất với ý tưởng "ghép theo FDI") |
| Kadi et al., arXiv 2402.04408 (2024) | 156 panoramic bệnh răng hiếm, 52 lớp răng | SD2 inpainting tạo panoramic "rỗng" rồi ghép răng thật theo contour; 298 ảnh mới | mAP 0.69 (không augment) → 0.76 (inpainting+ghép) vs 0.82 (hình học)\[8\] | **Rất cao** (đúng "ghép răng tạo panoramic mới"); cảnh báo về xảo ảnh và chồng lấp |
| Wathore & Gorthi, Pattern Recognition Letters 188 (2025) | Panoramic (segmentation) | Augmentation dựa trên đối xứng trái–phải | Tăng tới 5% DSC trung bình so với augmentation cứng/elastic\[22\] | Trung bình (dùng đối xứng giải phẫu) |
| CarDiff, arXiv 2611.00062 (2026) | MLUA-DC1000, caries panoramic | Diffusion có điều kiện mask, anatomy-aware | FID 90.76, LPIPS 0.48; tập train augment 5× tăng Dice tới 6.5%\[23\] | Thấp–trung bình (hướng mở rộng generative) |
| Pano-GAN (J. Imaging 2025, WGAN-GP) | 2,322 panoramic | Sinh panoramic một phần\[24\] | Khả thi nhưng chất lượng còn hạn chế | Thấp |
| Dual Framework răng khôn ngầm (Int. Dent. J. 2026) | Panoramic Mendeley | GAN tổng hợp lớp thiểu số (Inverted, Buccolingual) | +2.9% mAP, đạt 88.5%\[25\] | Trung bình (cùng mục tiêu lớp hiếm, khác phương pháp) |
| Tổng quan có hệ thống Gen-AI trong X-quang răng hàm mặt (PMC 2026) | Nhiều | GAN/diffusion (ví dụ Schoenhof et al. 2024, 54 nha sĩ/bác sĩ + 33 sinh viên đánh giá thật/giả; DentoSMART-LDM tổng hợp 12 bệnh lý)\[26\] | — | Nguồn tham chiếu cho chương "hướng mở rộng" |

**Giải pháp top DENTEX 2023** (24 đội trên bảng xếp hạng cuối, test 250 ảnh):\[27\]

| Đội | Diagnosis AR | AP | AP50 | Augmentation / mẹo dữ liệu |
|---|---|---|---|---|
| He L. (sjtu-seiee-426, hạng 1) | 0.592 | 37.06 | 54.31 | Flip dọc/crop/xoay cho mô hình phân đoạn 256×256; detector để mặc định; ensemble DINO+YOLOv8 bằng WBF; lọc theo tri thức (impacted luôn là răng số 8)\[10\]\[27\] |
| Mei S. (Chohotech/YOLOrtho, hạng 2) | 0.597 | 37.44 | 56.72 | Scale, blur, xoay, tịnh tiến; "flip mapping" đổi quadrant khi lật; pseudo-label răng khỏe; dùng thêm Tufts Dental\[9\]\[27\] |
| Choi K. (Sdent) | 0.759 | 36.48 | 57.30 | Random resize, lật ngang; pseudo-label 1,571 ảnh chưa gán nhãn để nhân đôi Periapical và Deep Caries\[11\]\[27\] |
| HierarchicalDet (baseline ban tổ chức) | 0.691 | 37.6 | 60.2 | Tiền huấn luyện SimMIM trên ảnh chưa gán nhãn\[27\] |

Lưu ý: bài DENTEX v2 xếp baseline HierarchicalDet hạng 3 theo vị trí trung bình (2.92), Choi K. hạng 4 (3.08); nếu chỉ tính đội tham gia thì top 3 là He, Mei, Choi.\[27\] Một điểm đáng chú ý: kết quả DINO-R50 của bạn (mAP50 = 44.3% trên val tự chia) thấp hơn AP50 của các đội top (54–57% trên test chính thức); hai con số không so sánh trực tiếp được (khác tập đánh giá) nhưng cho thấy còn dư địa.

Các bài dùng DENTEX sau 2023 tập trung vào kiến trúc/tự giám sát hơn là augmentation, ví dụ SSAD (arXiv 2406.13963) thêm nhánh tái tạo ảnh tự giám sát "plug-and-play" cho detector.\[28\] Điều này củng cố nhận định: augmentation ở mức instance cho bệnh lý răng trên DENTEX còn ít được khai thác.

#### A4. Long-tail detection

- **Repeat Factor Sampling (LVIS, Gupta et al. 2019):** r_c = max(1, sqrt(t / f_c)), với f_c là tỷ lệ ảnh chứa lớp c; ảnh lặp theo lớp hiếm nhất trong ảnh; t mặc định 0.001 trên LVIS.\[29\]\[30\] Với DENTEX, Periapical xuất hiện ở khoảng một phần nhỏ ảnh train nên cần chọn t lớn hơn nhiều (ví dụ t sao cho Periapical được lặp 2–3 lần).
- **Giới hạn của RFS:** lặp lại ảnh tức là lặp lại cả các đối tượng phổ biến cùng ảnh (Caries đi cùng Periapical 81.9%); Chang et al. (ICML 2021, arXiv 2104.05702) chỉ ra RFS "repeats an image based on the rarest object within that image" và đề xuất resampling mức đối tượng (object-centric memory replay) để khắc phục điều này. Copy-paste chính là "resampling mức đối tượng trong ngữ cảnh mới" — đây là luận điểm lý thuyết mạnh cho tiểu luận.
- Seesaw/Equalization loss là hướng thay đổi hàm mất mát; không bắt buộc cho tiểu luận xử lý ảnh, có thể nêu như hướng mở.

#### A5. Đánh giá độ chân thật ảnh tổng hợp

- FID dùng InceptionV3 tiền huấn luyện ImageNet nên khả năng áp dụng cho ảnh y tế bị tranh luận; tuy nhiên Woodland et al. (arXiv 2311.13717) thấy bộ trích đặc trưng ImageNet (đặc biệt SwAV) cho xếp hạng nhất quán với đánh giá chuyên gia hơn RadImageNet.\[31\] Kết luận thực dụng: báo cáo FID/KID như chỉ số phụ, không dùng làm bằng chứng chính.
- Với tiểu luận, thước đo thuyết phục hơn là **utility** (mAP tăng hay không) + **visual Turing test** nhỏ (ví dụ 40 patch thật/giả cho 2–3 người đánh giá, lý tưởng có nha sĩ) + **chỉ số seam** do bạn tự định nghĩa (xem C5).

**Đánh giá tính mới:** ghép răng theo FDI (van Nistelrooij 2024) và ghép răng tạo panoramic mới (Kadi 2024) đã có; lesion copy-paste trong y tế đã có. Điều chưa thấy công bố là: (i) hoán đổi răng mang bệnh theo FDI để cân bằng lớp hiếm cho detection 4 lớp bệnh, (ii) so sánh có hệ thống các kỹ thuật blending cổ điển trên X-quang răng, (iii) ablation tách riêng hiệu quả copy-paste so với RFS. Với tiểu luận môn học, mức mới "trung bình–khá" là phù hợp; nên viết trung thực là "áp dụng và đánh giá có hệ thống", không tuyên bố "đề xuất thuật toán mới".

### Phần B — Phản biện ý tưởng và các rủi ro

**B1. Mâu thuẫn cốt lõi: răng khỏe không phải lớp dương tính.** Trong bài toán 4 lớp, răng lành là background. Ghép răng khỏe tạo ra ảnh "mới" nhưng số box Periapical vẫn là 128. Ghép răng khỏe chỉ hữu ích cho: (a) đa dạng background, (b) hard negatives — ví dụ dán răng khỏe có cervical burnout vào vị trí mô hình hay báo nhầm, (c) xóa bệnh (thay răng bệnh bằng răng khỏe cùng FDI) để mô hình học "cùng ngữ cảnh nhưng không có bệnh" — một dạng counterfactual có giá trị. Muốn tăng lớp hiếm, phải ghép **răng bệnh**.

**B2. Label noise của "răng khỏe".** Tập disease của DENTEX chỉ gán box cho răng bất thường và không có lớp healthy;\[32\] tập quadrant-enumeration (634 ảnh) gán mọi răng nhưng không có chẩn đoán, nên răng lấy từ đó có thể có caries chưa gán nhãn.\[32\] Dán một răng như vậy vào ảnh đích tạo một tổn thương "không nhãn" → mô hình bị phạt khi phát hiện đúng. Giảm thiểu: lấy răng khỏe từ chính 705 ảnh disease (răng không trùng box bệnh nào) và lọc bằng mô hình baseline (chỉ giữ răng có điểm bệnh < 0.1), hoặc nhờ nha sĩ duyệt nhanh ngân hàng răng khỏe.

**B3. Tính nhất quán giải phẫu.**
- Vị trí FDI: răng hàm trên có chân hướng lên, hàm dưới hướng xuống; răng cửa hẹp, răng cối rộng. Chỉ hoán đổi **cùng số FDI** (hoặc cùng số ở quadrant đối xứng sau khi lật ngang).
- Méo phóng đại của panoramic: độ phóng đại ngang phụ thuộc vị trí trong focal trough; vùng răng cửa thường bị thu hẹp/mờ. Hoán đổi cùng vị trí giúp phóng đại xấp xỉ nhất quán; vẫn nên scale theo tỷ lệ chiều cao/rộng mask đích/nguồn (giới hạn 0.85–1.15).
- Trục răng: ước lượng góc trục bằng PCA trên mask; xoay răng nguồn để khớp trục răng đích (chênh lệch tối đa khoảng ±10°).
- Răng khôn ngầm: thường ở góc ảnh, liên quan cành lên xương hàm và ống thần kinh răng dưới; chỉ dán vào vị trí răng 8 (thay răng 8 có sẵn, hoặc vùng răng 8 trống phía sau răng 7).

**B4. Ngữ cảnh mô quanh răng.** Tổn thương chóp nằm trong xương, không nằm trên răng. Nếu chỉ cắt theo polygon răng, bạn có thể mất một phần vùng thấu quang quanh chóp. Giải pháp: với Periapical, cắt ROI = mask răng ∪ (bbox bệnh giãn 10–20%) ở vùng chóp, rồi hòa trộn biên rộng hơn trong xương.

**B5. Seam và "shortcut learning".** Biên ghép có gradient bất thường; detector dựa vào đặc trưng cục bộ nên có thể học "chỗ có đường viền ghép = có bệnh". Dwibedi et al. đã chỉ ra xảo ảnh biên làm giảm hiệu năng đáng kể và khuyên đa dạng hóa blending.\[13\] Thêm một biện pháp quan trọng: **dán cả răng khỏe với cùng pipeline** (không gán nhãn) để seam không còn tương quan với nhãn.

**B6. Khác biệt histogram/phơi sáng/máy chụp.** DENTEX lấy từ ba cơ sở khác nhau, nên độ tương phản và nhiễu khác nhau giữa các ảnh.\[33\]\[34\] Dùng histogram matching cục bộ (trên vùng vành quanh mask đích) thay vì toàn ảnh; khớp mức nhiễu bằng cách ước lượng độ lệch chuẩn nhiễu (ví dụ trên ảnh sai khác sau median filter) và thêm nhiễu Gaussian cho phần dán.

**B7. Poisson blending có thể làm sai tín hiệu bệnh.** Poisson clone giải phương trình trên miền gradient và kéo cường độ bên trong theo biên đích; tổng quan về image composition ghi nhận Poisson có thể làm biến dạng nội dung tiền cảnh ("seeping").\[35\] Với X-quang, độ thấu quang của tổn thương chính là tín hiệu chẩn đoán → Poisson toàn phần có thể làm nhạt Deep Caries/Periapical. Đề xuất: giữ nguyên lõi (mask co lại 5–8 px) và chỉ hòa trộn ở vành biên (feathering hoặc Poisson trên vành), rồi so sánh định lượng trong ablation.

**B8. Chồng lấp răng.** Panoramic có vùng tiếp xúc chồng lấp (đặc biệt răng cối nhỏ). Khi thay răng, vùng chồng lấp sẽ sai: ở X-quang, các cấu trúc chồng nhau làm tăng suy giảm, xấp xỉ cộng trong miền log, không phải "che phủ" như ảnh RGB. Thực tế đơn giản: tránh vùng chồng lấp bằng cách loại các cặp có IoU mask với răng kề > 0.1, hoặc ghép trong miền log (-log(I)) rồi đổi ngược.

**B9. Rò rỉ dữ liệu.** Chỉ xây ngân hàng răng từ tập train (564 ảnh); không bao giờ lấy từ val. Nên kiểm tra ảnh trùng/gần trùng giữa các tập con DENTEX (Rehan et al., arXiv 2609.21628, đã đối chiếu dữ liệu riêng 1,422 ảnh của họ với DENTEX và Tufts bằng MD5, perceptual hash và tương quan toàn cặp — 1.42 triệu phép so sánh — và nhận thấy perceptual hashing "proved unreliable on this modality").

**B10. Co-occurrence.** P(Caries|Periapical) = 81.9% và P(DeepCaries|Periapical) = 66.4%. Hoán đổi cả răng (răng + chóp) tự động mang theo các tổn thương thân răng — điều này tốt về tính chân thật nhưng **phải mang theo toàn bộ nhãn của răng nguồn** (mọi box bệnh có cùng FDI) và **xóa toàn bộ nhãn của răng đích bị thay**.

**B11. Tỷ lệ ảnh tổng hợp.** Không có con số chuẩn cho bài toán này. Khởi điểm hợp lý: augment online với xác suất p ∈ {0.25, 0.5}, mỗi ảnh tối đa 1–3 răng được thay; quét p trong ablation. Lý do: nghiên cứu nốt phổi tổng hợp bằng diffusion (arXiv 2603.01659) ghi nhận "performance improvements plateau or slightly decline beyond an optimal level of augmentation", và Kadi et al. thấy ảnh ghép đơn thuần chưa vượt augmentation hình học.

**B12. Ca "1–2 trên 1000" thật sự.** Copy-paste chỉ tái sử dụng hình thái đã có; nó tăng **ngữ cảnh** chứ không tạo **hình thái mới**. Với ca cực hiếm (một vài instance), mô hình có thể học thuộc instance đó. Giảm thiểu: biến đổi mức instance (scale ±10%, xoay ±5°, gamma 0.8–1.2, elastic rất nhẹ) như TumorCP; với hình thái mới thật sự cần mô hình sinh (CarDiff, DentoSMART-LDM) — để ở phần hướng mở rộng.

**B13. Ràng buộc augmentation hiện tại.** Lật ngang hợp lệ cho bài toán 4 lớp bệnh (nhãn không chứa FDI), nhưng nếu dùng FDI trong pipeline ghép phải hoán Q1↔Q2, Q3↔Q4 như YOLOrtho đã làm ("flip mapping").\[9\] Một nghiên cứu YOLOv11 trên panoramic trẻ em (Diagnostics 2025;15:2615) đã tắt hẳn lật ngang (fliplr = 0.0) và mosaic (mosaic = 0.0) để giữ đánh số FDI và quan hệ không gian giữa các răng — cũng là lý do không nên dùng mosaic 4 ảnh mặc định của YOLO; nếu dùng, chỉ dùng biến thể ghép nửa hàm trái–phải theo đường giữa.

### Phần C — Pipeline đề xuất

#### C1. Bước 1: Ngân hàng răng/tổn thương (tooth & lesion bank)

1. Đọc JSON COCO của DENTEX (chỉ tập train của split 80/20 seed 42). Với tập disease, mỗi annotation có bbox, trường `segmentation` dạng polygon, và nhãn quadrant/enumeration/diagnosis.\[12\]
2. Gộp annotation theo (image_id, quadrant, tooth number) → mỗi răng bệnh là một entry gồm: FDI = quadrant × 10 + số răng; danh sách chẩn đoán; mask răng (từ polygon); bbox các box bệnh.
3. Tạo ROI:
   - Caries/Deep Caries: ROI = mask răng giãn (dilate) 3–5 px.
   - Periapical: ROI = mask răng ∪ (bbox Periapical giãn 15%), làm mượt bằng closing.
   - Impacted: ROI = mask răng giãn 8–10 px (bao gồm túi quanh thân răng).
4. Lưu patch nguồn (crop độ phân giải gốc ~2800×1300, không resize), mask, trục răng (PCA), kích thước, thống kê cường độ của vành quanh răng.
5. Ngân hàng răng khỏe: răng có polygon trong ảnh train (cần lấy từ tập enumeration hoặc pseudo-label như YOLOrtho) nhưng không trùng box bệnh; lọc bằng mô hình baseline.
6. Nhánh xử lý ảnh cổ điển (nên có trong tiểu luận): khi chỉ có bbox (ví dụ răng khỏe pseudo-label), tạo mask bằng: CLAHE → Gaussian blur nhẹ → Otsu trong bbox → morphology opening/closing → giữ thành phần liên thông lớn nhất → GrabCut khởi tạo bằng kết quả Otsu (vùng chắc chắn tiền cảnh/hậu cảnh) → tùy chọn active contour. So sánh Dice với polygon gốc trên các răng có polygon — đây là một thí nghiệm xử lý ảnh nhỏ, rẻ và rất hợp môn học. MedSAM (Ma et al., Nature Communications 2024) nhận bbox làm prompt và là đường cơ sở deep learning tốt để so sánh.\[36\]

#### C2. Bước 2: Chọn vị trí đặt

- **Chế độ A — FDI tooth-swap (khuyến nghị chính):** chọn răng nguồn có FDI = k (ưu tiên lớp hiếm theo trọng số tỷ lệ nghịch tần suất); chọn ảnh đích có răng FDI = k (tìm từ pseudo-label/enumeration của ảnh đích) mà răng đó không mang bệnh; thay thế.
- **Chế độ B — đối xứng:** dán vào vị trí đối xứng (k ↔ FDI cùng số ở quadrant đối diện) sau khi lật ngang patch, như van Nistelrooij et al.\[7\]
- **Chế độ C — heatmap:** cho lớp không gắn chặt với một răng (ít dùng ở đây), lấy mẫu vị trí từ heatmap tâm box của lớp đó trên tập train.
- Quy tắc loại bỏ: IoU mask mới với răng lân cận > 0.1 → bỏ; tâm răng lệch đường cung hàm (ước lượng bằng đa thức bậc 2 qua tâm các răng) quá ngưỡng → bỏ; răng 8 Impacted chỉ đặt vào vị trí 8.

#### C3. Bước 3: Hòa trộn và hậu xử lý

1. **Lấp lỗ răng đích** trước khi dán (vì răng nguồn có thể nhỏ hơn): cv2.inpaint (Telea) trên mask răng đích giãn, rồi mới dán.
2. **Căn chỉnh hình học:** scale theo tỷ lệ kích thước mask, xoay theo chênh lệch trục PCA, căn tâm.
3. **Khớp cường độ:** histogram matching (skimage.exposure.match_histograms) giữa vành quanh răng nguồn và vành quanh răng đích, áp lên patch nguồn; hoặc khớp mean/std tuyến tính (đơn giản, ít làm méo tổn thương hơn).
4. **Blending — các biến thể để ablation:**
   - B0 Hard paste (cắt dán thẳng).
   - B1 Gaussian feathering: alpha = GaussianBlur(mask, σ = 3–7 px).
   - B2 Poisson: cv2.seamlessClone, NORMAL_CLONE và MIXED_CLONE.
   - B3 Laplacian pyramid blending (4–5 mức).
   - B4 "Core-preserving": lõi giữ nguyên, Poisson/feather chỉ trên vành (đề xuất của báo cáo này).
   - B-rand: chọn ngẫu nhiên một trong các chế độ mỗi lần (theo tinh thần Dwibedi).
5. **Khớp nhiễu/texture:** ước lượng σ nhiễu ở vùng đích, thêm nhiễu Gaussian vào phần dán nếu patch nguồn "mịn" hơn.
6. **Cập nhật nhãn:** xóa nhãn răng đích bị thay; thêm mọi box bệnh của răng nguồn, biến đổi theo cùng phép affine; clip vào ảnh.
7. **Kiểm soát chất lượng tự động:** tính "seam score" (xem C5) và loại mẫu vượt ngưỡng.

#### C4. Bước 4: Các biến thể bổ sung (theo thứ tự ưu tiên)

| Mã | Biến thể | Giải quyết failure mode nào | Ưu tiên |
|---|---|---|---|
| (a) | FDI tooth-swap răng bệnh (Periapical/Deep Caries/Impacted) | Thiếu mẫu lớp hiếm, bỏ sót tổn thương | MVP |
| (b) | Rare-class paste có ràng buộc giải phẫu (răng 8 cho Impacted; ROI chóp cho Periapical) | Periapical AP thấp | MVP |
| (c) | Hard-negative pasting: dán răng khỏe/cervical burnout không nhãn; "xóa bệnh" bằng răng khỏe cùng FDI | Giả tổn thương giải phẫu, nhầm cervical burnout, seam shortcut | MVP (nhẹ) |
| (e) | Patch crop độ phân giải gốc (crop 1024–1333 px quanh vùng có tổn thương) | Caries chỉ còn ~10×10 px sau resize | Cao — có thể là đòn bẩy lớn nhất cho localization |
| (f) | CLAHE/gamma ngẫu nhiên mô phỏng đa máy | Khác biệt cơ sở chụp | Trung bình (rẻ) |
| (d) | Half-arch mosaic theo đường giữa | Đa dạng tổ hợp trái–phải | Thấp–trung bình |
| (g) | Mô phỏng xảo ảnh kim loại (vùng sáng bão hòa + streak) | FP quanh phục hồi kim loại | Thấp |
| (h) | Diffusion inpainting (ControlNet/SD) | Hình thái mới cho ca cực hiếm | Hướng mở rộng; không vừa GPU 4GB |

Nhận xét: với lỗi chính là bỏ sót tổn thương nhỏ, biến thể (e) gần như chắc chắn nên đi kèm (a); kết hợp tốt nhất là cắt patch độ phân giải gốc **sau** khi đã swap, để tổn thương được dán vẫn giữ chi tiết.

#### C5. Bước 5: Thiết kế thực nghiệm (ablation)

**Mô hình:** chọn một detector hội tụ nhanh trên GPU nhỏ làm "ngựa thồ" cho ablation — YOLOv8s/m hoặc YOLO11s (Ultralytics) ở imgsz 1024–1280, hoặc Faster R-CNN R50-FPN (MMDetection/torchvision). Sau đó xác nhận cấu hình tốt nhất trên DINO-R50 (mô hình mạnh nhất hiện tại của bạn). Không dùng HierarchicalNet/DiffusionNet cho ablation vì chúng chưa hội tụ. Với Ultralytics: tắt mosaic (mosaic=0), giữ flipud=0, tắt copy_paste mặc định để không lẫn biến; augment của bạn nên tạo offline (lưu ảnh + nhãn) để dùng chung cho mọi framework.

**Bảng thí nghiệm đề xuất:**

| ID | Cấu hình | Câu hỏi trả lời |
|---|---|---|
| E0 | Baseline hiện tại (sáng/tối + xoay ±5° + lật ngang) | Mốc so sánh |
| E1 | E0 + RFS (t chọn để Periapical lặp ~2–3×) | Resampling đơn thuần được bao nhiêu? |
| E2 | E0 + random copy-paste (vị trí ngẫu nhiên, hard paste) | Copy-paste kiểu Ghiasi có hợp với X-quang? |
| E3 | E0 + FDI tooth-swap, hard paste | Ràng buộc giải phẫu có giúp không? |
| E4 | E3 + feathering | Blending ảnh hưởng thế nào? |
| E5 | E3 + Poisson (NORMAL) | Poisson có làm nhạt tín hiệu bệnh? |
| E6 | E3 + core-preserving + histogram matching | Biến thể đề xuất |
| E7 | Tốt nhất(E3–E6) + RFS | Copy-paste và RFS có cộng dồn? |
| E8 | E7 + hard-negative pasting | Giảm FP? |
| E9 (tùy chọn) | E8 + patch crop độ phân giải gốc | Cải thiện localization tổn thương nhỏ? |
| Phụ | Quét p ∈ {0.25, 0.5, 0.75} cho cấu hình tốt nhất | Tỷ lệ tổng hợp tối ưu |

**Metric:** mAP50, mAP50:95, AP50 theo từng lớp (bắt buộc tách Periapical và Impacted), AR@100, recall end-to-end tại ngưỡng điểm cố định (0.3), số FP/ảnh, và số FP trên vùng giải phẫu dễ nhầm (lỗ cằm, ống răng dưới, xoang hàm — có thể gán nhãn thủ công 20–30 ảnh val). Báo cáo ma trận nhầm Caries vs Deep Caries.

**Thống kê:** tập val chỉ có ~30 box Periapical (158 − 128), nên phương sai rất lớn. Chạy ≥3 seed mỗi cấu hình, báo cáo trung bình ± độ lệch chuẩn, và bootstrap theo ảnh (1,000 lần) để có khoảng tin cậy 95% cho AP Periapical. Nếu chênh lệch nhỏ hơn độ rộng khoảng tin cậy, viết trung thực là "không có ý nghĩa thống kê".

**Kiểm tra seam shortcut (đóng góp xử lý ảnh đẹp cho tiểu luận):**
- *Seam score:* năng lượng gradient (Sobel) trên vành 3 px quanh biên ghép chia cho năng lượng gradient trên vành tương ứng ở ảnh gốc; càng gần 1 càng tốt. So sánh giữa B0–B4.
- *Real-vs-fake probe:* huấn luyện một CNN nhỏ (ResNet18) phân biệt patch thật/patch ghép; độ chính xác càng gần 50% thì ghép càng khó phát hiện.
- *Seam probe trên val:* dán răng khỏe (không nhãn) vào ảnh val bằng cùng pipeline; nếu FP tăng mạnh tại vị trí dán → mô hình đã học shortcut.
- *Phụ:* FID/KID trên patch răng (thật vs ghép) với đặc trưng ImageNet; visual Turing test 40 patch.

**Chi phí tính toán (ước lượng thô, cần tự đo):** sinh ảnh offline bằng OpenCV trên CPU chỉ tốn vài giây mỗi ảnh. Huấn luyện YOLO cỡ nhỏ ở 1024–1280 px trên T4/P100 khả thi với batch nhỏ; trên GPU 4GB laptop chỉ nên dùng mô hình "n/s" với batch 2–4 hoặc huấn luyện trên patch crop. Với khoảng 10 cấu hình × 3 seed = 30 lượt chạy, hãy dồn ablation vào Kaggle/Colab và chỉ chạy DINO cho 1–2 cấu hình cuối cùng.

**Quy tắc bất biến:** chỉ augment tập train; ngân hàng răng chỉ từ train; val/test luôn là ảnh thật; cố định split seed 42.

#### C6. Bước 6: Thư viện và mã có sẵn

- OpenCV: `cv2.seamlessClone` (NORMAL_CLONE, MIXED_CLONE), `cv2.inpaint`, `cv2.createCLAHE`, `cv2.grabCut`, morphology, Otsu.
- scikit-image: `exposure.match_histograms`, `segmentation.active_contour`, `segmentation.watershed`.
- pycocotools: chuyển polygon ↔ mask; Albumentations: augment ảnh + bbox đồng bộ.
- Ultralytics: có `copy_paste` sẵn (chỉ dán khi IoA với mọi đối tượng hiện có < 0.3, chế độ flip mặc định dán trong cùng một ảnh) — dùng làm baseline E2 "random copy-paste" rất nhanh, nhưng cần nhãn segmentation.\[37\]
- MMDetection 3.x: có transform CopyPaste (cần mask) — dùng cho DINO/Faster R-CNN nếu muốn augment online.
- MedSAM (Ma et al., Nature Communications 2024): prompt bằng bbox để tách răng khi không có polygon.\[38\]
- Mã tham khảo: DentexSegAndDet (đội hạng 1, có script chuyển polygon DENTEX sang mask),\[39\] TumorCP (YaoZhang93/TumorCP),\[4\] SyntheticTumors (MrGiovanni/SyntheticTumors),\[40\] context_aug (dvornikita/context_aug).\[39\]\[41\]

#### C7. Bước 7: Cấu trúc tiểu luận gợi ý

1. Mở đầu: bài toán, DENTEX, mất cân bằng lớp (Periapical 4.5%), điểm nghẽn localization.
2. Cơ sở lý thuyết: augmentation hình học/quang học; copy-paste (Dwibedi, Dvornik, Ghiasi); lesion-aware augmentation (TumorCP, CarveMix, LesionMix); kỹ thuật xử lý ảnh dùng trong pipeline (Otsu, GrabCut, morphology, histogram matching, Poisson, Laplacian pyramid, inpainting Telea).
3. Phương pháp: ngân hàng răng; FDI tooth-swap; ràng buộc giải phẫu; các chế độ blending; hard-negative.
4. Thực nghiệm: dữ liệu và split, mô hình, cấu hình, metric, thống kê.
5. Kết quả: bảng ablation; AP theo lớp; seam score; real-vs-fake probe; phân tích lỗi định tính.
6. Thảo luận: khi nào copy-paste giúp/hại; shortcut; label noise; giới hạn với ca cực hiếm.
7. Kết luận và hướng phát triển (diffusion inpainting, pseudo-label ảnh chưa gán nhãn).

**Hình nên có:** (1) sơ đồ pipeline; (2) phân bố lớp và heatmap vị trí theo lớp; (3) ví dụ trích răng: polygon vs Otsu vs GrabCut vs MedSAM; (4) cùng một lần swap với B0–B4 cạnh nhau, có phóng to vùng biên; (5) bản đồ gradient biên (seam); (6) biểu đồ AP theo lớp cho E0–E9 kèm thanh sai số; (7) ví dụ FP/FN trước–sau; (8) biểu đồ quét tỷ lệ p.

## Recommendations

1. **Điều chỉnh ý tưởng:** đổi từ "ghép răng khỏe ngẫu nhiên" sang "**hoán đổi răng mang bệnh theo FDI** + dán răng khỏe làm hard negative". Ghép ngẫu nhiên không ràng buộc chỉ nên giữ làm baseline E2 để chứng minh ràng buộc giải phẫu có giá trị.
2. **MVP trong 2–3 tuần:** (i) script trích ngân hàng răng từ polygon DENTEX (train-only); (ii) FDI-swap với inpaint lỗ + histogram matching + feathering; (iii) chạy E0, E1, E3, E4, E5 với YOLO/Faster R-CNN, 3 seed; (iv) seam score + vài hình minh họa. Như vậy đã đủ một tiểu luận hoàn chỉnh.
3. **Nếu còn thời gian:** thêm E6–E9 (core-preserving, RFS kết hợp, hard negative, patch crop độ phân giải gốc) và xác nhận trên DINO-R50.
4. **Viết kết quả trung thực:** nếu copy-paste không vượt baseline (như Kadi et al. đã thấy với ảnh ghép thuần), đó vẫn là kết quả có giá trị — hãy phân tích bằng seam probe và AP theo lớp xem do shortcut, label noise hay do thiếu đa dạng hình thái.

## Caveats

- Chưa tìm thấy bài nào có ablation tách riêng copy-paste cho phát hiện tổn thương trên panoramic; khẳng định "khoảng trống" dựa trên tìm kiếm có giới hạn, không phải tổng quan hệ thống.
- Thông tin DENTEX có trường `segmentation` cho tập disease dựa trên commit file validation trên Hugging Face và README bên thứ ba; hãy tự mở JSON training để xác nhận trước khi viết code.
- Một số nguồn là preprint (Kadi et al., CarDiff, SSAD, nghiên cứu nốt phổi diffusion) chưa qua bình duyệt đầy đủ; con số của chúng nên được trích kèm ghi chú này.
- Các con số chi phí tính toán trong báo cáo là ước lượng thô; kết quả DINO 44.3% mAP50 của bạn đo trên val tự chia, không so sánh trực tiếp với leaderboard DENTEX.
- Ghép trong miền log-attenuation và "core-preserving blending" là đề xuất kỹ thuật của báo cáo này dựa trên vật lý X-quang và hạn chế đã biết của Poisson, chưa được kiểm chứng trong tài liệu nha khoa.

## Sources

1. [Simple Copy-Paste Is a Strong Data Augmentation Method ...](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf)
2. [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation - Paper Detail](https://deeplearn.org/arxiv/176695/simple-copy-paste-is-a-strong-data-augmentation-method-for-instance-segmentation)
3. [ECCV 2018 papers - CVF Open Access](https://openaccess.thecvf.com/content_ECCV_2018/html/NIKITA_DVORNIK_Modeling_Visual_Context_ECCV_2018_paper.html)
4. [TumorCP: A Simple but Effective Object-Level Data Augmentation for Tumor Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_55)
5. [TumorCP: A Simple but Effective Object-Level Data Augmentation for Tumor Segmentation](https://deepai.org/publication/tumorcp-a-simple-but-effective-object-level-data-augmentation-for-tumor-segmentation)
6. [LesionMix: A Lesion-Level Data Augmentation Method for Medical Image Segmentation](https://arxiv.org/pdf/2308.09026)
7. [Combining public datasets for automated tooth assessment in panoramic radiographs](https://link.springer.com/article/10.1186/s12903-024-04129-5)
8. <https://arxiv.org/pdf/2402.04408>
9. <https://arxiv.org/pdf/2308.05967>
10. [Intergrated Segmentation and Detection Models for Dentex Challenge 2023](https://arxiv.org/pdf/2308.14161)
11. <https://arxiv.org/pdf/2308.14070>
12. [Upload validation\_triple.json · ibrahimhamamci/DENTEX at e210ce9](https://huggingface.co/datasets/ibrahimhamamci/DENTEX/commit/e210ce984776eaf3487ce5aa71cf43cab4b34deb)
13. [Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection](https://arxiv.org/pdf/1708.01642)
14. [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/pdf/2012.07177)
15. [X-Paste: Revisiting Scalable Copy-Paste for Instance Segmentation using CLIP and StableDiffusion](https://arxiv.org/pdf/2212.03863)
16. [TumorCP: A Simple but Effective Object-Level Data Augmentation for Tumor Segmentation](https://arxiv.org/html/2107.09843)
17. [TumorCP: A Simple but Effective Object-Level Data Augmentation for Tumor Segmentation \[Quick Review\]](https://liner.com/review/tumorcp-simple-but-effective-objectlevel-data-augmentation-for-tumor-segmentation)
18. [\[2108.06883\] CarveMix: A Simple Data Augmentation Method for Brain Lesion Segmentation](https://arxiv.org/abs/2108.06883)
19. [Label-Free Liver Tumor Segmentation Qixin Hu1 Yixiong Chen2 Junfei Xiao3](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Label-Free_Liver_Tumor_Segmentation_CVPR_2023_paper.pdf)
20. [\[2303.14869\] Label-Free Liver Tumor Segmentation - ar5iv - arXiv](https://ar5iv.labs.arxiv.org/html/2303.14869)
21. [A Diffusion-Driven Fine-Grained Nodule Synthesis Framework for Enhanced Lung Nodule Detection from Chest Radiographs](https://arxiv.org/html/2603.01659v1)
22. [Feature perturbation augmentation for reliable evaluation of importance estimators in neural networks - ScienceDirect](https://sciencedirect.com/science/article/pii/S0167865523002842?via=ihub%3D)
23. [\[2611.00062\] CarDiff: Anatomically Aware Dental Caries Synthesis using Diffusion](https://arxiv.org/abs/2611.00062)
24. [(PDF) Pano-GAN: A Deep Generative Model for Panoramic Dental Radiographs](https://www.researchgate.net/publication/388642506_Pano-GAN_A_Deep_Generative_Model_for_Panoramic_Dental_Radiographs)
25. [Dual Framework for Classification and Detection of Third Molar Impaction in Panoramic Radiographs - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12907845/)
26. [Current evidence of generative artificial intelligence specifically developed for dental and maxillofacial radiology: a systematic review - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13519356/)
27. [DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-rays](https://arxiv.org/html/2305.19112v2)
28. [\[2406.13963\] SSAD: Self-supervised Auxiliary Detection Framework for Panoramic X-ray based Dental Disease Diagnosis](https://arxiv.org/abs/2406.13963)
29. [LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://arxiv.org/html/1908.03195v2)
30. [LVIS: A Dataset for Large Vocabulary Instance Segmentation Agrim Gupta](https://arxiv.org/pdf/1908.03195)
31. [\[2311.13717\] Feature Extraction for Generative Medical Imaging Evaluation: New Evidence Against an Evolving Trend](https://arxiv.org/abs/2311.13717)
32. [GitHub - SVaishalini1/Dentex · GitHub](https://github.com/SVaishalini1/Dentex)
33. [DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-ray](https://arxiv.org/pdf/2305.19112)
34. <https://arxiv.org/pdf/2305.19112v2>
35. [Making Images Real Again: A Comprehensive Survey on Deep Image Composition](https://arxiv.org/pdf/2106.14490)
36. [Segment anything in medical images](https://www.nature.com/articles/s41467-024-44824-z)
37. [YOLO Data Augmentation Guide - Mosaic, MixUp, HSV, and Previews](https://docs.ultralytics.com/guides/yolo-data-augmentation)
38. [Get Started with Medical Segment Anything Model for Medical Image Segmentation - MATLAB & Simulink](https://www.mathworks.com/help/medical-imaging/ug/get-started-with-medical-segment-anything-model-for-medical-image-segmentation.html)
39. [GitHub - xyzlancehe/DentexSegAndDet · GitHub](https://github.com/xyzlancehe/DentexSegAndDet)
40. [GitHub - MrGiovanni/SyntheticTumors: \[CVPR 2023\] Label-Free Liver Tumor Segmentation](https://github.com/MrGiovanni/SyntheticTumors)
41. [GitHub - dvornikita/context\_aug: Context-driven data augmentation for Object Detection (ECCV'18) · GitHub](https://github.com/dvornikita/context_aug/)

# PLAN — Đồ án AI: Phát hiện & đếm nhà ngập / nhà không ngập từ ảnh UAV (FloodNet)

> File này là bản giao việc cho **Claude Code**. Hãy đọc toàn bộ trước khi viết dòng code đầu tiên.

---

## 0. Quy tắc làm việc (bắt buộc)

Người dùng là **sinh viên năm 3 ngành CNTT**, sẽ phải tự bảo vệ đồ án trước giảng viên. Vì vậy:

1. **Làm tuần tự theo Phase.** Cuối mỗi Phase có một **GATE**: dừng lại, tóm tắt kết quả, chờ người dùng xác nhận rồi mới sang Phase tiếp theo.
2. **Bạn không có GPU.** Viết code, test trên dữ liệu nhỏ/giả lập bằng CPU. Việc huấn luyện chạy trên **Google Colab** do người dùng bấm chạy; sau mỗi lần chạy, người dùng sẽ dán log/kết quả lại cho bạn.
3. **CẤM dùng YOLO** (mọi phiên bản) và package `ultralytics`. Giảng viên chỉ cấm YOLO, các model khác được dùng.
4. **Code đơn giản, dễ đọc**, có comment ngắn giải thích "tại sao". Tránh over-engineering (không dùng framework phức tạp không cần thiết).
5. **Không đoán**: cấu trúc thư mục dataset, tên class/giá trị mask, phiên bản thư viện đều phải **kiểm tra thực tế** (chạy `ls`, đọc file, đọc doc) rồi ghi lại trong `docs/NOTES.md`.
6. Mọi tham số nằm trong file config (YAML) hoặc `argparse`. Không hard-code đường dẫn.
7. `README.md` viết bằng **tiếng Việt**, giải thích từng bước và lý do mỗi quyết định thiết kế để sinh viên hiểu và bảo vệ được.
8. Cố định seed, lưu config vào thư mục kết quả của mỗi lần train để tái lập được.

---

## 1. Mục tiêu

**Đầu vào:** ảnh chụp trên cao (UAV) sau lũ.
**Đầu ra:** các bounding box của **nhà ngập** (`flooded_building`) và **nhà không ngập** (`non_flooded_building`), cùng **số lượng** mỗi loại trên mỗi ảnh.

**Model chốt:** ConvNeXt-Tiny (backbone) + FPN + **Cascade R-CNN 3 tầng** (IoU 0.5 / 0.6 / 0.7).
**Dataset:** FloodNet (phần semantic segmentation).

**Tiêu chí thành công** (chốt lại ở GATE 2 sau khi có baseline):
- Có pipeline chạy end-to-end: dữ liệu → train → đánh giá → suy luận có ảnh minh hoạ.
- Báo cáo được **mAP (COCO)** và **sai số đếm (MAE/RMSE)** trên tập test.
- Có ít nhất 1 ablation có ý nghĩa (ví dụ Cascade vs Faster R-CNN) và phân tích lỗi bằng hình ảnh.

---

## 2. Ràng buộc môi trường

| Hạng mục | Thực tế cần tính đến |
|---|---|
| GPU | Colab thường là T4 16GB (có thể khác; luôn kiểm tra bằng `nvidia-smi`) |
| CPU / RAM | ~2 vCPU, ~12GB RAM → **data loading là điểm nghẽn**, không được dùng augmentation nặng trên CPU |
| Phiên làm việc | Có thể bị ngắt bất kỳ lúc nào → **bắt buộc có checkpoint + resume tự động** |
| Google Drive | Đọc nhiều file nhỏ rất chậm → **không train trực tiếp từ Drive** |
| Dataset gốc | `MyDrive/Flood_House_AI/floodnet_raw.zip` (~12GB) |
| Nơi lưu kết quả | `MyDrive/Flood_House_AI/runs/` (checkpoint, log, kết quả) |

**Quy trình chuẩn mỗi phiên Colab:** mount Drive → copy dataset đã xử lý (nhỏ, dạng zip) từ Drive sang `/content` → giải nén → train với `work_dir` trên Drive → tự resume từ checkpoint mới nhất.

---

## 3. Quyết định kỹ thuật đã chốt

| Vấn đề | Quyết định | Lý do |
|---|---|---|
| Framework | **MMDetection 3.x** (config-driven) | Có sẵn Cascade R-CNN + ConvNeXt-T, AMP, resume, COCO eval |
| Plan B nếu MMDet không cài được | **Detectron2** + ConvNeXt qua `timm` | Cascade R-CNN có sẵn; xem mục 5.1 |
| Số lớp | **2**: `flooded_building`, `non_flooded_building` | Đúng mục tiêu đồ án; các lớp khác của FloodNet bỏ qua |
| Định dạng nhãn | **COCO JSON** (bbox + segmentation) | Chuẩn, dễ đánh giá, dễ dùng lại |
| Tạo box từ mask | Connected components + (tuỳ chọn) tách nhà dính nhau | FloodNet không có nhãn instance |
| Độ phân giải | Resize offline về **cạnh dài ~1536px** làm mặc định; tiling chỉ dùng nếu EDA cho thấy nhà quá nhỏ | Giảm dung lượng, tăng tốc I/O; quyết định dựa trên số liệu (xem 5.2) |
| Tối ưu | AdamW, AMP (fp16), layer-wise LR decay cho ConvNeXt | Theo config chuẩn của MMDet, ổn định |
| Khởi tạo | ConvNeXt-T pretrain ImageNet (mặc định); thử thêm init từ checkpoint COCO | Dữ liệu ít → transfer learning là quan trọng |
| Hậu xử lý | NMS mặc định; thử TTA + **WBF** (`ensemble-boxes`) ở giai đoạn cuối | Đúng hướng ban đầu của người dùng |

---

## 4. Cấu trúc repo đề xuất

```
flood-house-counting/
├── README.md                 # tiếng Việt, hướng dẫn chạy từng bước
├── PLAN.md                   # (file này)
├── docs/NOTES.md             # ghi chép thực tế: cấu trúc data, phiên bản, quyết định
├── configs/
│   ├── data.yaml             # đường dẫn, kích thước resize, min_area, tách nhà...
│   └── mmdet/                # config MMDetection (kế thừa từ config gốc của MMDet)
├── src/floodcount/
│   ├── data/                 # audit.py, mask_to_coco.py, resize.py, visualize.py
│   ├── eval/                 # coco_eval.py, count_eval.py, error_analysis.py
│   └── infer/                # predict.py (ảnh -> box + đếm), tta_wbf.py
├── scripts/                  # CLI mỏng gọi vào src/
├── notebooks/                # notebook MỎNG cho Colab (chỉ gọi script)
│   ├── 00_colab_setup.ipynb
│   ├── 01_data_prep.ipynb
│   ├── 02_train.ipynb
│   └── 03_eval_infer.ipynb
├── tests/                    # test cho mask_to_coco bằng mask giả lập
└── requirements-colab.txt    # phiên bản đã pin sau khi kiểm chứng
```

---

## 5. Các Phase

### Phase 0 — Khởi tạo & kiểm tra môi trường Colab

**Việc cần làm**
1. Tạo cấu trúc repo, `requirements-colab.txt`, notebook `00_colab_setup.ipynb`.
2. Notebook setup: mount Drive, in `nvidia-smi`, phiên bản Python/torch/CUDA có sẵn của Colab.
3. **Smoke test cài đặt MMDetection**: chọn phiên bản `torch` ↔ `mmcv` ↔ `mmdet` tương thích (tra bảng cài đặt chính thức của mmcv/mmdet ở thời điểm làm, **không dựa vào trí nhớ**). Ưu tiên wheel dựng sẵn qua `mim install`. Nếu phải biên dịch `mmcv` từ source (lâu), hãy **build một lần rồi lưu file `.whl` lên Drive** để các phiên sau cài lại nhanh.
4. Chạy thử inference 1 ảnh với model pretrain của MMDet để chắc chắn cả stack hoạt động.
5. Nếu sau 2 hướng thử mà vẫn lỗi → chuyển sang **Plan B (Detectron2 + timm ConvNeXt-T)**, ghi rõ lý do vào `docs/NOTES.md`.

**GATE 0:** stack cài được trên Colab, inference thử chạy, có `requirements-colab.txt` đã pin.

---

### Phase 1 — Khám phá dữ liệu (EDA) & quyết định tiền xử lý

**Việc cần làm** (`src/floodcount/data/audit.py`)
1. Giải nén `floodnet_raw.zip` vào `/content` (hoặc thư mục tạm), **liệt kê cấu trúc thực tế** và tìm đúng thư mục ảnh + mask cho phần segmentation (train/val/test). Kiểm tra xem tập test có nhãn hay không. Nếu test **không** có nhãn: chia val thành val/test, ghi rõ trong NOTES.
2. Xác minh **giá trị mask ↔ tên lớp** (kỳ vọng FloodNet: 1 = building-flooded, 2 = building-non-flooded; phải kiểm tra thực tế trước khi dùng).
3. Thống kê: số ảnh mỗi split, kích thước ảnh, số pixel mỗi lớp, số ảnh có nhà ngập / không ngập.
4. Chạy connected components thử trên toàn bộ train và thống kê: số instance mỗi lớp mỗi ảnh, **phân bố diện tích**, cạnh box nhỏ nhất/percentile 5–50–95 **sau khi resize** về kích thước dự kiến.
5. Xuất ảnh minh hoạ (overlay box lên ảnh) cho ~30 ảnh ngẫu nhiên ra `outputs/eda/`.

**Quy tắc quyết định dựa trên số liệu** (ghi kết quả vào NOTES):
- **Tiling?** Nếu percentile 5 của cạnh box nhỏ nhất sau resize ≥ ~32px → **không cần tiling**. Nếu nhỏ hơn nhiều → bật tiling (ô 1024×1024, chồng lấn 25%).
- **Tách nhà dính nhau?** Nếu nhiều component có diện tích lớn bất thường (> ~3× trung vị) hoặc nhìn overlay thấy nhiều nhà bị gộp → bật tách bằng distance transform + watershed.

**GATE 1:** báo cáo EDA (bảng số liệu + ảnh overlay) và đề xuất tham số (`target_long_side`, `min_area`, tiling, tách nhà). Người dùng xác nhận.

---

### Phase 2 — Chuyển mask → COCO & tiền xử lý offline

**Việc cần làm** (`mask_to_coco.py`, `resize.py`)
1. Resize ảnh (bilinear/INTER_AREA) và mask (**nearest**) về kích thước đã chốt; lưu JPEG chất lượng 95 và PNG mask. Mục tiêu: dataset gọn (~1–2GB thay vì 12GB) để copy nhanh sang `/content`.
2. Với mỗi lớp nhà: `cv2.connectedComponentsWithStats` (connectivity=8) → lọc nhiễu bằng `min_area` → (tuỳ chọn) tách component dính nhau → sinh `bbox` + `segmentation` (polygon) + `area`.
3. Xuất `instances_{train,val,test}.json` theo COCO, với 2 category (`id=1: flooded_building`, `id=2: non_flooded_building`).
4. Sinh file **ground-truth đếm** `counts_{split}.csv` = số instance mỗi lớp mỗi ảnh (lấy từ đúng COCO JSON để nhất quán).
5. **Test đơn vị** (`tests/`) bằng mask giả lập: 2 hình chữ nhật rời nhau → 2 instance; 2 hình dính nhau → kiểm tra hành vi tách; nhiễu nhỏ → bị lọc.
6. **GT audit:** xuất overlay ~30 ảnh ngẫu nhiên từ COCO JSON cuối cùng để người dùng nhìn bằng mắt xem box có hợp lý không.
7. Nén dataset đã xử lý thành `floodnet_coco.zip` và lưu lên `MyDrive/Flood_House_AI/processed/`.

**GATE 2:** COCO JSON hợp lệ (load được bằng `pycocotools`), overlay ổn, số liệu thống kê cuối cùng (số ảnh / số box mỗi lớp mỗi split). Người dùng xác nhận chất lượng GT.

---

### Phase 3 — Cấu hình model & sanity check

**Việc cần làm** (`configs/mmdet/`)
1. Tạo config kế thừa từ config **Cascade (Mask) R-CNN ConvNeXt-T** có sẵn trong MMDetection; chuyển thành **Cascade R-CNN chỉ box** (bỏ mask head) và đổi `num_classes=2` ở cả 3 bbox head.
2. Dataset: đường dẫn `/content/floodnet_coco/...`, `metainfo` 2 lớp, `num_workers=2`, `persistent_workers=True`.
3. Pipeline train (nhẹ cho CPU): resize theo scale đã chốt (multi-scale nhẹ 0.7–1.3), lật ngang/dọc (ảnh trên cao không có "hướng" cố định), xoay 90° nếu Albumentations có sẵn, chỉnh sáng nhẹ. **Không dùng mosaic/mixup nặng.**
4. Cấu hình phù hợp bài toán đếm: `rpn max_per_img` ~1000, `rcnn max_per_img` ~300 (một ảnh có thể có rất nhiều nhà), ngưỡng NMS thử 0.5–0.6.
5. Huấn luyện: AdamW (lr ≈ 1e-4, weight decay 0.05), layer-wise LR decay cho backbone, **AMP bật**, `batch_size` 2 (tăng `accumulative_counts` để đạt effective batch ~8 nếu cần), warmup, cosine hoặc step LR.
6. Checkpoint: lưu mỗi epoch, giữ 3 bản gần nhất + bản tốt nhất theo `bbox_mAP`, **`work_dir` trên Drive**. Viết `scripts/train.py` tự động tìm checkpoint mới nhất để resume.
7. **Sanity check bắt buộc:** train **overfit 20 ảnh** trong ~vài chục epoch. Loss phải giảm rõ và mAP trên chính 20 ảnh đó phải cao. Nếu không → có bug ở dữ liệu/config, **sửa trước khi train đầy đủ.**
8. Đo thời gian 1 epoch và VRAM đỉnh trên Colab, ghi vào NOTES để ước lượng tổng thời gian.

**GATE 3:** overfit test thành công, có ước lượng thời gian/epoch, người dùng đồng ý số epoch.

---

### Phase 4 — Huấn luyện baseline (E1)

- Train ~24 epoch (điều chỉnh theo thời gian/epoch đo được), qua nhiều phiên Colab bằng cơ chế resume.
- Theo dõi bằng TensorBoard (lưu log trên Drive): loss, `bbox_mAP`, `AP50`, AP theo lớp.
- Chọn checkpoint theo mAP trên **val** (tuyệt đối không dùng test để chọn).

**GATE 4:** kết quả baseline trên val (mAP, AP từng lớp) + đồ thị loss. Cùng người dùng đọc kết quả và quyết định thí nghiệm tiếp theo.

---

### Phase 5 — Đánh giá đầy đủ (mAP + đếm) & phân tích lỗi

**Việc cần làm** (`src/floodcount/eval/`)
1. **COCO eval** bằng `pycocotools`: mAP, AP50, AP75, AP theo lớp, APs/APm/APl.
2. **Đánh giá đếm** (`count_eval.py`) — đây là chỉ số quan trọng nhất của đề tài:
   - Với mỗi lớp, đếm số box dự đoán có `score ≥ t` mỗi ảnh; so với `counts_{split}.csv`.
   - Báo cáo **MAE, RMSE** cho: nhà ngập, nhà không ngập, tổng số nhà; thêm sai số của **tỉ lệ nhà ngập** (= ngập / tổng).
   - **Chọn ngưỡng `t` trên val** (tối thiểu hoá MAE), sau đó áp nguyên ngưỡng đó lên test. Báo cáo thêm kết quả ở ngưỡng cố định 0.5 để so sánh.
3. **Phân tích lỗi** (`error_analysis.py`): ghép cặp dự đoán–GT theo IoU ≥ 0.5, thống kê: bỏ sót, dự đoán thừa, **nhầm lớp ngập ↔ không ngập** (confusion matrix). Xuất ~20 ảnh lỗi tiêu biểu (dự đoán màu này, GT màu kia).
4. Vẽ biểu đồ scatter "đếm dự đoán vs đếm GT" cho từng lớp.

**GATE 5:** bảng kết quả val + test, confusion matrix, các ảnh lỗi. Từ đây quyết định hướng cải thiện.

---

### Phase 6 — Thí nghiệm cải thiện & ablation (chọn theo kết quả GATE 5)

Mỗi thí nghiệm chỉ thay **một** yếu tố so với baseline để so sánh công bằng.

| ID | Thí nghiệm | Mục đích | Ưu tiên |
|---|---|---|---|
| E1 | Baseline (ConvNeXt-T + Cascade R-CNN, init ImageNet) | Mốc so sánh | Bắt buộc |
| E2 | Init từ checkpoint COCO của Cascade (Mask) R-CNN ConvNeXt-T (nạp backbone+neck+RPN, bỏ layer phân loại; `strict=False`) | Thường tăng đáng kể khi dữ liệu ít | Cao |
| E3 | **Faster R-CNN** (1 tầng) cùng backbone | Ablation chứng minh giá trị của Cascade — rất hợp để đưa vào báo cáo | Cao |
| E4 | Cân bằng lớp: `ClassBalancedDataset` / oversample ảnh có nhà ngập | Xử lý mất cân bằng lớp | Trung bình |
| E5 | Tiling / thay đổi độ phân giải (chỉ nếu EDA gợi ý nhà nhỏ bị mất) | Cải thiện AP vật thể nhỏ | Theo EDA |
| E6 | TTA (lật ngang/dọc, đa scale) + **WBF** | Đúng hướng ban đầu; thường tăng mAP và giảm sai số đếm | Cao (ở cuối) |
| E7 | (Tuỳ chọn) Cascade **Mask** R-CNN dùng mask đã có | Có thể tách nhà dính nhau tốt hơn | Thấp |

**Nguyên tắc:** ghi kết quả mọi thí nghiệm vào một bảng duy nhất `docs/RESULTS.md` (mAP, AP từng lớp, MAE/RMSE đếm, thời gian train, đường dẫn checkpoint), kể cả thí nghiệm thất bại — kết quả âm vẫn có giá trị cho báo cáo.

**GATE 6:** chốt model cuối cùng dựa trên **val**, sau đó mới chạy **một lần duy nhất** trên test.

---

### Phase 7 — Suy luận, demo & đóng gói báo cáo

1. `scripts/predict.py`: nhận 1 ảnh (hoặc thư mục) → vẽ box (màu khác nhau cho ngập/không ngập) + ghi số đếm lên ảnh + xuất JSON `{flooded: n, non_flooded: m, total: k}`. Hỗ trợ ảnh gốc độ phân giải lớn (tự resize/tiling cho khớp lúc train).
2. `notebooks/03_eval_infer.ipynb`: chạy end-to-end trên vài ảnh mới để demo.
3. (Tuỳ chọn) Demo Gradio trong Colab: upload ảnh → ra ảnh kết quả + số đếm.
4. Hoàn thiện `README.md` (tiếng Việt): cách cài, cách chạy từng Phase, cấu trúc repo, bảng kết quả, hình minh hoạ.
5. Gom vào `report_assets/`: bảng kết quả, đồ thị loss/mAP, scatter đếm, confusion matrix, ảnh lỗi, ảnh demo — để người dùng dùng trực tiếp trong báo cáo/slide.

**GATE 7 (hoàn thành).**

---

## 6. Rủi ro & cách xử lý

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Không cài được mmcv trên Colab | Lỗi build/import, xung đột torch | Pin torch phù hợp wheel có sẵn; build 1 lần rồi cache `.whl` lên Drive; nếu vẫn lỗi → Plan B Detectron2 |
| Nhà dính nhau → đếm thiếu | Component quá lớn, MAE đếm thấp hơn thực tế có hệ thống | Bật tách watershed, kiểm tra bằng overlay; nêu rõ hạn chế của GT trong báo cáo |
| Colab ngắt phiên | Mất tiến trình | Checkpoint mỗi epoch lên Drive + auto-resume |
| Data loading chậm (2 vCPU) | GPU utilization thấp | Resize offline, copy dataset sang `/content`, augmentation nhẹ, `persistent_workers` |
| Hết VRAM | CUDA OOM | Giảm batch xuống 1 + tăng accumulation, giảm scale, bật AMP, giảm số proposal |
| Nhà ngập ít → AP lớp ngập thấp | AP_flooded << AP_non_flooded | E4 (cân bằng lớp), theo dõi AP theo lớp, phân tích confusion |
| Nhầm ngập ↔ không ngập do thiếu ngữ cảnh | Confusion matrix cho thấy nhầm lớn | Nới rộng vùng ngữ cảnh RoI, phân tích ảnh lỗi; cân nhắc bước phân loại phụ có ngữ cảnh (chỉ khi còn thời gian) |
| Rò rỉ test | Chọn ngưỡng/checkpoint bằng test | Mọi lựa chọn dựa trên **val**; test chạy một lần duy nhất ở cuối |

---

## 7. Định nghĩa "Hoàn thành" (Definition of Done)

- [ ] Pipeline chạy lại được từ đầu chỉ bằng README + notebooks trên Colab mới.
- [ ] COCO JSON hợp lệ, có kiểm tra bằng mắt (overlay) và test đơn vị cho bước chuyển mask.
- [ ] Baseline E1 + tối thiểu ablation E3 (và E2 hoặc E6) đã chạy, ghi trong `docs/RESULTS.md`.
- [ ] Báo cáo mAP **và** MAE/RMSE đếm trên test (ngưỡng chọn từ val).
- [ ] Có phân tích lỗi bằng hình ảnh + confusion matrix.
- [ ] Có `predict.py` + demo cho ảnh mới.
- [ ] Không có bất kỳ thành phần YOLO/ultralytics nào trong repo.
- [ ] README tiếng Việt giải thích rõ lý do các quyết định.

---

## 8. Prompt khởi động gợi ý (dán vào Claude Code)

> Hãy đọc `PLAN_DO_AN_DEM_NHA_NGAP.md` ở thư mục gốc và làm theo đúng từng Phase. Bắt đầu với Phase 0: tạo cấu trúc repo và notebook setup Colab. Dừng ở mỗi GATE để báo cáo và chờ tôi xác nhận. Tôi là sinh viên năm 3, nên hãy giải thích ngắn gọn lý do cho các quyết định quan trọng. Nhớ: không dùng YOLO.

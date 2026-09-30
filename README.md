# Đếm nhà ngập / không ngập từ ảnh UAV — ConvNeXt-Tiny + Cascade R-CNN

Đồ án phát hiện và **đếm** hai loại đối tượng trên ảnh chụp từ trên cao (UAV) sau lũ:

| Lớp | Ý nghĩa |
|---|---|
| `flooded_building` | Nhà bị ngập |
| `non_flooded_building` | Nhà không ngập |

**Đầu vào:** một ảnh UAV. **Đầu ra:** các bounding box của hai lớp trên, kèm **số lượng** mỗi loại.

- **Model:** ConvNeXt-Tiny (backbone) + FPN + **Cascade R-CNN 3 tầng** (IoU 0.5 / 0.6 / 0.7)
- **Framework:** MMDetection 3.x — cấu hình tường minh từng thành phần, không dùng
  YOLO/Ultralytics (đề bài cấm)
- **Dataset:** FloodNet (BinaLab) — ảnh UAV, nhãn gốc là *semantic segmentation*,
  phải tự chuyển sang bounding box
- **Huấn luyện:** Google Colab (GPU T4)

Kế hoạch đầy đủ theo từng Phase: [`PLAN_DO_AN_DEM_NHA_NGAP.md`](PLAN_DO_AN_DEM_NHA_NGAP.md).
Số liệu và phát hiện đã kiểm chứng thật: [`docs/NOTES.md`](docs/NOTES.md).

---

## Trạng thái hiện tại

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Khởi tạo repo + kiểm tra môi trường Colab | ✅ **GATE 0 đạt** — đã chạy thật trên Colab T4 (xem `docs/NOTES.md` §1.5) |
| 1 | Khám phá dữ liệu (EDA) & quyết định tiền xử lý | ⏳ chờ xác nhận GATE 0 |
| 2 | Chuyển mask → COCO & tiền xử lý offline | ⏳ |
| 3 | Cấu hình model & sanity check (overfit 20 ảnh) | ⏳ |
| 4 | Huấn luyện baseline (E1) | ⏳ |
| 5 | Đánh giá mAP + sai số đếm & phân tích lỗi | ⏳ |
| 6 | Thí nghiệm cải thiện & ablation (E2–E7) | ⏳ |
| 7 | Suy luận, demo & đóng gói báo cáo | ⏳ |

---

## Cách chạy

Repo: **https://github.com/21xDHoang/flood-house-counting** (public)

Trên Google Colab, mở một notebook trống và clone repo về `/content`:

```python
!git clone https://github.com/21xDHoang/flood-house-counting.git /content/flood-house-counting
```

Repo **public** nên clone không cần token. Sau đó mở
`notebooks/00_colab_setup.ipynb` từ cây thư mục bên trái Colab rồi:

1. `Runtime → Change runtime type → T4 GPU`.
2. Chạy lần lượt từ trên xuống. Notebook tự cài môi trường và **không cần restart
   runtime** (xem lý do ở mục "Quyết định" bên dưới).
3. Ô cuối in ra khối **"BÁO CÁO GATE 0"**.

(Vì sao clone về `/content` mà không chạy thẳng trên Drive — xem mục "Quyết định".)

Dữ liệu và kết quả nằm trên Drive, **không** nằm trong repo:

```
MyDrive/Flood_House_AI/
├── floodnet_raw.zip     # dataset gốc (~12 GB, đã có sẵn)
├── wheels/              # cache wheel mmcv, để phiên sau cài nhanh hơn
└── runs/                # checkpoint, log, kết quả (notebook tự tạo)
```

---

## Cấu trúc repo

| Đường dẫn | Vai trò |
|---|---|
| `PLAN_DO_AN_DEM_NHA_NGAP.md` | Kế hoạch 8 phase, có GATE giữa các phase |
| `docs/NOTES.md` | **Ghi chép thực tế**: phiên bản thư viện, cấu trúc dataset, các bẫy đã kiểm chứng |
| `docs/RESULTS.md` | Bảng kết quả mọi thí nghiệm (sẽ tạo ở Phase 6) |
| `configs/data.yaml` | Tham số dữ liệu (sẽ tạo ở Phase 2) |
| `configs/mmdet/` | Config MMDetection của đồ án, kế thừa config gốc |
| `src/floodcount/data/` | `audit.py` (EDA), `mask_to_coco.py`, `resize.py`, `visualize.py` |
| `src/floodcount/eval/` | `coco_eval.py` (mAP), `count_eval.py` (MAE/RMSE đếm), `error_analysis.py` |
| `src/floodcount/infer/` | `predict.py` (ảnh → box + số đếm), `tta_wbf.py` |
| `scripts/` | Lệnh CLI mỏng gọi vào `src/` |
| `notebooks/` | Notebook **mỏng** cho Colab — chỉ gọi script, không chứa logic |
| `tests/` | Test cho bước mask → COCO bằng mask giả lập |
| `outputs/` | Ảnh minh hoạ, overlay, biểu đồ (không đưa lên git) |
| `requirements-colab.txt` | Bản ghi các gói cài thêm vào Colab (đọc phần đầu file trước khi dùng) |

---

## Các quyết định quan trọng và lý do

*(Phần này để trả lời khi bảo vệ — mỗi dòng là một câu hỏi giảng viên có thể hỏi.)*

**Vì sao không dùng YOLO?**
Đề bài cấm. Ngoài ra MMDetection phù hợp hơn về mặt học thuật: từng thành phần
(backbone, neck, RPN, ROI head, hàm loss, anchor) đều là config tường minh, giải
thích được, thay vì bị đóng gói trong một lệnh `model.train()`.

**Vì sao Cascade R-CNN mà không phải Faster R-CNN?**
Cascade dùng 3 tầng ROI head với ngưỡng IoU tăng dần (0.5 → 0.6 → 0.7). Ảnh UAV
có rất nhiều nhà **dính nhau**, một tầng dễ vừa bỏ sót vừa chồng box. Đây cũng là
lý do chọn Cascade làm model chính và Faster R-CNN làm ablation (E3) để chứng
minh giá trị của nó bằng số liệu.

**Vì sao ConvNeXt-Tiny?**
Backbone hiện đại, nhẹ (≈28M tham số) nên vừa VRAM T4, có checkpoint ImageNet lẫn
COCO sẵn trong MMDetection → transfer learning tốt khi dữ liệu ít. Tiny là mức
cân bằng giữa độ chính xác và thời gian train trong giới hạn của Colab.

**Vì sao chỉ 2 lớp?**
Đúng mục tiêu đồ án là **đếm** nhà ngập / không ngập. FloodNet có 10 lớp nhưng
đường, cây, nước, xe... không phục vụ câu hỏi đếm, thêm vào chỉ làm loãng bài toán
và giảm AP của 2 lớp cần quan tâm.

**Vì sao dùng COCO JSON?**
Là định dạng chuẩn, `pycocotools` đọc được, MMDetection nhận trực tiếp, và COCO
eval là thước đo mAP được công nhận — không phải chỉ số tự chế.

**Vì sao ngưỡng quyết định (chọn checkpoint, chọn ngưỡng đếm) đều lấy từ val?**
Để tránh **rò rỉ test**. Test chỉ được chạy **một lần duy nhất** ở cuối. Nếu chọn
ngưỡng trên test thì con số báo cáo sẽ đẹp giả tạo và không còn ý nghĩa.

**Vì sao không train trực tiếp từ Google Drive?**
Drive gắn vào Colab qua FUSE — đọc nhiều file nhỏ rất chậm. Cách làm: copy dataset
đã xử lý (nhỏ, dạng zip) từ Drive sang `/content` rồi mới train; **chỉ ghi
checkpoint ra Drive** vì `/content` bị xoá sạch khi hết phiên.

**Vì sao không hạ PyTorch cho khớp wheel mmcv chính thức?**
OpenMMLab ngừng phát hành mmcv từ 7/2024, wheel dựng sẵn chỉ tới Python 3.12,
trong khi Colab chạy Python 3.13. Hạ torch cũng không giải quyết được vì torch
2.1.0 không có wheel cho Python 3.13. Vì vậy đồ án dùng **wheel mmcv do cộng đồng
dựng sẵn** cho đúng tổ hợp Python + torch + CUDA của Colab.

> **Nói rõ để bảo vệ:** wheel này **không do OpenMMLab phát hành** mà do repo
> `miropsota/torch_packages_builder` dựng qua GitHub Actions (có build attestation
> để truy nguyên nguồn gốc). Đây là lựa chọn duy nhất còn dùng được trên Python 3.13,
> và **không đụng vào PyTorch của Colab**, nên chỉ cần cài thêm 4 gói và chạy tiếp,
> không phải restart runtime.

**Vì sao phải tắt nhánh "đa phương thức" của mmpretrain?**
Không phải để tối ưu mà để **chạy được**: mmpretrain 1.2.0 nạp BLIP / LLaVA / OFA
khi máy có `transformers>=4.28.0`, mà phần đó viết cho transformers 4.x trong khi
Colab dùng 5.x → `import mmpretrain.models` ném `TypeError`, kéo sập luôn backbone
ConvNeXt (mmdet 3.3.0 không có ConvNeXt riêng). Đồ án chỉ dùng backbone nên tắt
bằng **đúng cờ `WITH_MULTIMODAL` của OpenMMLab** — không vá chắp vá.

**Vì sao độ phân giải phải quyết định bằng số liệu (Phase 1) chứ không chọn trước?**
Có hai hướng trái ngược: resize cả ảnh cho nhẹ, hoặc cắt tile ở độ phân giải gốc để
giữ nhà to. Hướng nào đúng phụ thuộc **phân bố kích thước nhà thật** — nên Phase 1
đo bằng connected components rồi mới chốt, ngưỡng đã định trước trong plan
(percentile 5 của cạnh box nhỏ nhất ≥ ~32px thì không cần tile).

---

## Môi trường

| Thành phần | Phiên bản / ghi chú |
|---|---|
| Python (Colab) | 3.13 (notebook tự in ra và đối chiếu) |
| PyTorch / CUDA | do Colab cấp, **không hạ cấp, không đụng vào** |
| MMDetection | 3.3.0 |
| MMEngine | 0.10.7 (đã sửa lỗi trùng tên `Adafactor` có ở 0.10.4) |
| MMPreTrain | 1.2.0 (bắt buộc — backbone ConvNeXt nằm ở đây); **đã tắt** nhánh `WITH_MULTIMODAL` vì nó đụng transformers 5.x của Colab |
| MMCV | 2.2.0 + hậu tố bản dựng, cài từ index cộng đồng (notebook tự dò) |

Chi tiết nguồn của từng con số và các bẫy đã kiểm chứng: [`docs/NOTES.md`](docs/NOTES.md).

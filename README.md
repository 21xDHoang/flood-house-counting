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
| 0 | Khởi tạo repo + kiểm tra môi trường Colab | ✅ **GATE 0 đạt** — chạy thật trên Colab T4, đã kiểm chứng đủ (`docs/NOTES.md` §1.5–§1.6) |
| 1 | Khám phá dữ liệu (EDA) & quyết định tiền xử lý | ✅ **GATE 1 đạt (30/09/2026)** — 4 tham số đã chốt (`docs/NOTES.md` §2.5) |
| 2 | Chuyển mask → COCO & tiền xử lý offline | ✅ **GATE 2 đạt (30/09/2026)** — 2.343 ảnh, 6.301 box, đối chiếu Phase 1 khớp hoàn toàn (`docs/NOTES.md` §2.7) |
| 3 | Cấu hình model & sanity check (overfit 20 ảnh) | 🟡 **GATE 3 đang chạy trên Colab (01/10/2026)** — `[3.6]`–`[3.9]` xong (số đo `--dry-run`: 866,3 ms/vòng, 10,4 phút/epoch, VRAM 7,81/14,56 GB); `[3.10]` lần đầu hỏng vì cú pháp `{{_base_.}}` hỏng im lặng, đã sửa — chờ chạy lại. Xem `docs/NOTES.md` §3.9 và §3.11 |
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
├── floodnet_raw.zip     # dataset gốc 13 GB — Phase 3 trở đi KHÔNG cần nữa
├── processed/           # floodnet_coco.zip 1,86 GB do Phase 2 dựng ra
├── state/               # audit.jsonl, build.jsonl (để không phải dựng lại từ đầu)
├── wheels/              # cache wheel mmcv, để phiên sau cài nhanh hơn
└── runs/                # checkpoint, log, kết quả (notebook tự tạo)
```

### Chạy Phase 1 (khảo sát dữ liệu)

Mở `notebooks/01_data_prep.ipynb` và chạy từ trên xuống.

- **Không cần GPU** — bước này chỉ đọc ảnh, đếm pixel và tách component bằng CPU.
  Chạy ở runtime CPU cũng được, khỏi tốn suất T4.
- **Không cần cài MMDetection** — chỉ dùng `cv2`, `numpy`, `yaml`. Đừng chạy
  `00_colab_setup.ipynb` chỉ để làm việc này, tốn ~10 phút vô ích.

Hai ô chạy, cố ý tách rời:

| Ô | Việc | Thời gian |
|---|---|---|
| `[1.5]` | Chạy thử **3 ảnh mỗi split** vào thư mục `eda_thu` riêng | ~30 giây |
| `[1.6]` | Chạy **đầy đủ** | ~15–30 phút |

Chạy ô `[1.5]` trước để phát hiện sớm nếu cấu trúc zip khác kỳ vọng — nếu sai thì
biết sau 30 giây thay vì sau 20 phút. Ô `[1.6]` **tự chạy tiếp nếu bị ngắt**: kết
quả từng ảnh ghi ngay xuống đĩa, chạy lại sẽ bỏ qua ảnh đã xử lý.

Kết quả nằm ở `MyDrive/Flood_House_AI/runs/eda/`: `EDA_REPORT.md` và thư mục
`overlay/` chứa ảnh có vẽ box (đỏ = nhà ngập, xanh = không ngập, **tím = box to
bất thường, nghi bị gộp**).

### Chạy Phase 2 (mask → COCO)

Mở `notebooks/02_build_coco.ipynb` và chạy từ trên xuống. Cũng **không cần GPU**
và **không cần cài MMDetection**.

| Ô | Việc | Thời gian |
|---|---|---|
| `[2.5]` | Chạy thử **3 ảnh mỗi split** vào thư mục `_thu` riêng | ~30 giây |
| `[2.6]` | Chạy **đầy đủ** | ~15–30 phút |
| `[2.8]` | Nén `floodnet_coco.zip` (~1–2 GB) lên Drive | ~3–5 phút |

**Chạy ô `[1.6]` của notebook 01 trước ô `[2.6]`, trong cùng một phiên.** Phase 2
cần `audit.jsonl` còn nằm trong `/content` để làm phép đối chiếu quan trọng nhất
của GATE 2: nó tính **lại** số box từ kết quả Phase 1 bằng **đúng bộ lọc của Phase
2** rồi so với dataset vừa dựng. Khớp hoàn toàn nghĩa là hai phase dùng chung một
định nghĩa "nhà hợp lệ" — nếu lệch thì **đừng train**, gửi kết quả lại.

Kết quả ở `MyDrive/Flood_House_AI/runs/build/`: `BUILD_REPORT.md` (số ảnh/số box
mỗi split, kết quả đối chiếu, cảnh báo) và `overlay_gt/` — ảnh vẽ box **lấy từ file
COCO JSON** đè lên **ảnh JPEG đã ghi ra đĩa**, tức đúng thứ model sẽ đọc lúc train.

Dataset dựng ra (`/content/floodnet_coco`) rồi nén thành **một tệp zip** trên Drive:
4686 file ảnh/mask ghi thẳng lên Drive qua FUSE sẽ chậm hơn nhiều lần, mà Phase 3
chỉ cần giải nén một tệp. Mất ~20 phút dựng lại nếu quên ô `[2.8]`.

**Kết quả Phase 2 (đã chốt GATE 2 ngày 30/09/2026):**

| Split | Số ảnh | Box nhà ngập | Box nhà không ngập | Ảnh có nhà ngập |
|---|---|---|---|---|
| test | 448 | 604 | 651 | 47 |
| train | 1.445 | 1.841 | 1.938 | 149 |
| val | 450 | 643 | 624 | 49 |
| **Tổng** | **2.343** | **3.088** | **3.213** | **245** |

Phép đối chiếu với Phase 1 ra **+0 ở cả hai lớp** (`docs/NOTES.md` §2.7) — hai phase
dùng chung một định nghĩa "nhà hợp lệ". Dataset đóng gói sẵn ở
`MyDrive/Flood_House_AI/processed/floodnet_coco.zip` (1,86 GB), **từ Phase 3 không cần
tới `floodnet_raw.zip` 13 GB nữa**.

Hai con số đo được ở Phase 2 mà Phase 3–6 phải dùng tới: box có cạnh **trung vị 186px**
(ảnh chỉ rộng 1536px), và **46,9% số box bị cắt ở mép ảnh** — xem §2.7 để biết vì sao
con số thứ hai là thật chứ không phải lỗi đếm.

### Chạy Phase 3 (cấu hình model & sanity check)

**Cần GPU T4** (chỉ ở ô `[3.9]` trở đi). Môi trường **tự cài ngay trong notebook này**:
ô `[3.3b]` gọi `scripts/cai_moi_truong.py` — cùng script mà notebook 00 dùng — nên chỉ
cần **một tab Colab**, không phải mở notebook 00 ở tab thứ hai. Không cần
`floodnet_raw.zip` 13 GB nữa, chỉ cần `floodnet_coco.zip` 1,86 GB.

Mở `notebooks/03_train.ipynb` và chạy từ trên xuống:

| Ô | Việc | Thời gian |
|---|---|---|
| `[3.3b]` | Cài môi trường (đã cài rồi thì tự bỏ qua) | 3–5 phút |
| `[3.5]` | Giải nén dataset vào `/content` + đối chiếu đúng 2.343 ảnh / 6.301 box | 1–2 phút |
| `[3.6]` | Kiểm dữ liệu: tên lớp, **ảnh có thật trên đĩa**, số box mmdet thực nhận | ~1–2 phút |
| `[3.7]` | Đo dải anchor của RPN trên box thật | vài giây |
| `[3.8]` | Dựng `instances_overfit20.json` (20 ảnh, seed 42) | vài giây |
| `[3.9]` | **`train.py --dry-run`** — chạy thử vài vòng, đo thời gian 1 epoch thật (GATE 3) | ~5 phút |
| `[3.10]` | Train overfit 20 ảnh — **phép thử đường ống** (GATE 3) | 20–40 phút |
| `[3.11]` | Train thật (Phase 4) | 6–8 giờ |

**Dừng sau ô `[3.10]` và báo cáo lại.** Ô `[3.11]` là Phase 4, chỉ chạy sau khi chốt số
epoch từ phép đo ở `[3.9]` — 24 epoch trong config hiện tại là **mặc định tạm**.

Điều notebook này **không** làm: **không chạy test trên tập test**. Test chỉ được chạy
**một lần duy nhất** ở Phase 5; chọn ngưỡng đếm và chọn checkpoint đều lấy từ val. Tập
test ở đây chỉ được kiểm tra về mặt **cấu trúc** (đếm ảnh, đếm box), không dùng để chọn
tham số. Kết quả overfit 20 ảnh cũng **không phải** kết quả của đồ án — nó chỉ chứng minh
đường ống học được, xem đầu tệp `src/floodcount/data/overfit.py` để biết nó **không**
chứng minh điều gì.

---

## Cấu trúc repo

| Đường dẫn | Vai trò |
|---|---|
| `PLAN_DO_AN_DEM_NHA_NGAP.md` | Kế hoạch 8 phase, có GATE giữa các phase |
| `docs/NOTES.md` | **Ghi chép thực tế**: phiên bản thư viện, cấu trúc dataset, các bẫy đã kiểm chứng |
| `docs/RESULTS.md` | Bảng kết quả mọi thí nghiệm (sẽ tạo ở Phase 6) |
| `configs/data.yaml` | **Mọi tham số** của Phase 1–2: đường dẫn, ngưỡng quyết định, tham số EDA |
| `configs/mmdet/` | Config MMDetection của đồ án (`cascade_convnext_t_floodnet.py`), kế thừa config gốc; `overfit20.py` cho phép thử 20 ảnh |
| `src/floodcount/data/` | `audit.py` (EDA), `mask_to_coco.py`, `resize.py`, `visualize.py`, `kiem_tra.py` (chốt chặn trước train), `overfit.py`, `photometric.py` |
| `src/floodcount/models/` | `transforms.py` — transform tự viết đăng ký vào registry của mmdet |
| `src/floodcount/eval/` | `coco_eval.py` (mAP), `count_eval.py` (MAE/RMSE đếm), `error_analysis.py` |
| `src/floodcount/infer/` | `predict.py` (ảnh → box + số đếm), `tta_wbf.py` |
| `scripts/` | Lệnh CLI mỏng gọi vào `src/`; `cai_moi_truong.py` — cài + vá môi trường Colab, dùng chung cho notebook 00 và 03 |
| `notebooks/` | Notebook **mỏng** cho Colab — chỉ gọi script, không chứa logic |
| `tests/` | **453 phép kiểm** trong 8 bộ test, chạy trên máy CPU — không cần GPU, không cần dataset thật (dùng zip giả có cả bẫy ColorMasks). `test_train.py` kiểm được cả trên máy **chưa cài MMDetection**, vì `scripts/train.py` chỉ import mmdet/mmengine bên trong hàm (và khoá luôn thứ tự build `optim_wrapper` của vòng đo `--dry-run` — lỗi thật đã gặp, §3.8 NOTES); `test_cai_moi_truong.py` khoá cách dò wheel `mmcv` và ba miếng vá môi trường |
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

**Vì sao bỏ hẳn nhánh mask của model gốc?**
Bản gốc là Cascade **Mask** R-CNN. Nhãn FloodNet vốn là mask, nhưng đồ án đã chuyển
sang box từ Phase 2 và câu hỏi của đồ án là **đếm**, không phải tách hình dạng. Giữ
mask head chỉ tốn thêm VRAM và thời gian. Điểm cần lưu ý khi bảo vệ: phải dùng
`_delete_=True` ở `roi_head`, vì mmengine gộp dict theo chiều sâu — chỉ khai báo đè
vài khoá thì `mask_roi_extractor`/`mask_head` của bản gốc **vẫn còn nguyên** và model
lặng lẽ trở thành model có mask.

**Vì sao giữ lại ảnh không có căn nhà nào, khi mmdet mặc định bỏ chúng?**
Vì đầu ra cuối cùng của đồ án là **đếm trên mọi ảnh**, kể cả ảnh chỉ có nước và cây.
Bỏ các ảnh rỗng thì model chỉ học "ảnh kiểu gì cũng có nhà" và sẽ đếm thừa trên ảnh
không có nhà — đúng loại ảnh chiếm gần một nửa dataset. Vì vậy `filter_empty_gt=False`
là lựa chọn có chủ ý, và Phase 6 sẽ có thí nghiệm bật/tắt để chứng minh bằng số liệu.

**Vì sao tự viết transform tăng sáng thay vì dùng `PhotoMetricDistortion` có sẵn?**
Bản có sẵn của mmdet kết thúc bằng `if swap_flag: img = img[..., swap_value]` — tức
**đảo kênh màu ngẫu nhiên với xác suất 1/2**. Với bài toán này, màu nước là manh mối
chính để phân biệt nhà ngập với nhà không ngập, nên đảo kênh là phá đúng tín hiệu cần
học. Transform tự viết chỉ tăng/giảm sáng và tương phản, giữ nguyên màu.

**Vì sao viết `scripts/train.py` riêng mà không gọi `tools/train.py` của mmdet?**
Phần train thật chỉ là `Runner.from_cfg(cfg).train()` — một dòng. Thứ đáng viết là
phần **tiền kiểm** chạy trước đó một phút: mmdet có ít nhất ba kiểu hỏng **không báo
lỗi** (tên lớp không khớp JSON → bỏ im lặng cả một lớp; `keep_ratio` thiếu → ảnh méo;
`num_classes` sót ở một trong ba tầng cascade). Một suất train 24 epoch là 6–8 giờ,
phát hiện sai ở epoch 20 là mất trắng. Tiền kiểm in ra và **dừng** nếu có vấn đề.

**Vì sao phải chạy thử overfit 20 ảnh trước khi train thật?**
Để chứng minh đường ống chạy được đầu-cuối (dữ liệu đọc đúng, nhãn gắn đúng lớp, loss
giảm được) với cái giá vài chục phút thay vì vài giờ. Nói rõ để bảo vệ: nó **không**
chứng minh chất lượng phân loại ngập/không ngập — model chỉ cần nhận ra "đây là ảnh
nào" là đủ đạt loss gần 0. Vì vậy kết quả overfit **không bao giờ** được trích làm kết
quả của đồ án.

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

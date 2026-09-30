# NOTES — ghi chép thực tế của đồ án

File này ghi lại **những gì đã kiểm tra thật** (kiểm ngày nào, kiểm bằng cách
nào, kết quả ra sao) và **những gì còn là giả định**.

> Quy tắc của đồ án: **không đoán**. Cái gì chưa kiểm thì ghi rõ là chưa kiểm,
> không viết như thể đã biết.

---

## 1. Môi trường Colab — kiểm tra ngày 30/09/2026

### 1.1 Phiên bản thư viện (tra từ PyPI/GitHub, không dựa vào trí nhớ)

| Gói | Bản mới nhất | Ngày phát hành | Ghi chú | Ghim dùng |
|---|---|---|---|---|
| mmdet | 3.3.0 | 5/2024 | vẫn là bản mới nhất tính đến 30/09/2026 | `3.3.0` |
| mmcv | 2.2.0 | 24/04/2024 | OpenMMLab **ngừng phát hành** sau bản này | `2.2.0` + hậu tố bản dựng |
| mmengine | 0.10.7 | 04/03/2025 | bản ổn định mới nhất (0.11.0rc* là pre-release) | `0.10.7` |
| mmpretrain | 1.2.0 | 04/01/2024 | chứa backbone ConvNeXt | `1.2.0` |
| pycocotools | 2.0.11 | 15/12/2025 | có wheel `cp312-abi3` → cài được trên Python 3.13 | `2.0.11` |

Nguồn: `pypi.org/pypi/<tên gói>/json` (trường `info.version` và ngày upload),
`raw.githubusercontent.com/open-mmlab/...` cho phần source.

### 1.2 Bảy cái bẫy đã xác minh bằng cách đọc source (không phải suy đoán)

**(a) mmdet 3.3.0 từ chối mmcv 2.2.0 — và đây là lỗi *im lặng cho tới lúc import*.**
`mmdet/__init__.py` (tag v3.3.0) có:

```python
mmcv_maximum_version = '2.2.0'
assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version))
```

Dấu `<` là **nghiêm ngặt**, nên cài đúng bản 2.2.0 vẫn trượt assert. Phải nới
ngưỡng trong file này thành `2.3.0`. Giới hạn mmengine là `>=0.7.1, <1.0.0`
→ bản 0.10.7 hợp lệ.

**(b) Lỗi trùng tên `Adafactor` — mmengine 0.10.7 đã sửa, 0.10.4 thì chưa.**
Đọc `mmengine/optim/optimizer/builder.py` ở hai tag:

| | đăng ký Adafactor của torch | của transformers | kết quả |
|---|---|---|---|
| v0.10.4 | tên `Adafactor` (dòng 33) | tên `Adafactor` (dòng 174) | **trùng tên → KeyError** |
| v0.10.7 | tên `TorchAdafactor` (dòng 28–30) | tên `Adafactor` (dòng 178) | không đụng nhau |

Cả hai bản đều gọi hàm đăng ký **ngay lúc import module**
(`TRANSFORMERS_OPTIMIZERS = register_transformers_optimizers()`), nên lỗi nổ
ngay khi import nếu môi trường có `transformers` (Colab có sẵn). Vì đã ghim
mmengine 0.10.7 nên **không cần vá**; notebook vẫn giữ một hàm vá tự phát hiện,
chỉ hoạt động nếu gặp lại bản cũ.

**(c) `mmpretrain` là bắt buộc, không phải tùy chọn.**
`configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py`
(tag v3.3.0) có:

```python
custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)
model = dict(backbone=dict(type='mmpretrain.ConvNeXt', arch='tiny', ...))
```

`allow_failed_imports=False` nghĩa là thiếu mmpretrain thì mmdet dừng ngay, không
chạy tiếp. Điểm may: ràng buộc `mmcv>=2.0.0,<2.4.0` của mmpretrain nằm ở **extra
`mim`**, nên `pip install mmpretrain` thường **không** kéo mmcv về bản cũ.

**Ghi chú thêm:** `mmpretrain/__init__.py` cũng có assert riêng cho mmcv
(`>= 2.0.0, < 2.4.0`) — bản mmcv 2.2.0 của ta lọt qua nên **không cần vá gì thêm**.

**(d) Gói pip của mmdet KHÔNG chứa config ConvNeXt.**
`mmdet/configs/` trong gói đã cài chỉ có `_base_`, `cascade_rcnn`, `faster_rcnn`,
`mask_rcnn`, `retinanet`, ... — **không có thư mục `convnext`** (kiểm bằng
GitHub API). Nghĩa là config gốc của ConvNeXt chỉ nằm trong repo mmdetection.
→ Notebook setup **clone nông** repo này ở tag `v3.3.0` (~63 MB) để lấy đủ cây
config, và vì các config kế thừa nhau bằng đường dẫn tương đối (`_base_`) nên
phải có đủ cây, không tải lẻ một file được.

**(e) `pycocotools` không có wheel `cp313` thường, nhưng vẫn cài được binary.**
Bản 2.0.11 phát hành wheel `cp312-abi3-manylinux...` — **abi3 tương thích tiến**,
nên pip chấp nhận trên Python 3.13. Không phải biên dịch từ source.

**(f) Wheel mmcv: phải dò theo runtime, không hard-code.**
Tên wheel có hậu tố theo tổ hợp, ví dụ
`mmcv-2.2.0+a8073c7pt2.11.0cu128-cp313-cp313-linux_x86_64.whl`. Colab đổi torch
là tên đổi theo. Notebook dò từ chính runtime (Python tag + phiên bản torch +
tag CUDA) rồi tìm trong index cộng đồng `miropsota.github.io/torch_packages_builder`.
Index này có cả bản cho `cp314`/`cu130` (kiểm ngày 30/09/2026) nên còn dùng được
khi Colab nâng cấp.

**(g) mmpretrain 1.2.0 không import được với transformers 5.x — và nó kéo sập cả backbone ConvNeXt.**
`mmpretrain/models/multimodal/__init__.py` chỉ nạp BLIP / BLIP2 / LLaVA / OFA…
khi cờ `WITH_MULTIMODAL = True`, mà cờ này (`mmpretrain/utils/dependency.py`)
**bật khi máy có `transformers>=4.28.0`** — Colab có bản 5.x nên cờ bật. Trong
đó, `multimodal/blip/language_model.py` viết:

```python
try:
    from transformers.modeling_utils import (PreTrainedModel,
        apply_chunking_to_forward, find_pruneable_heads_and_indices,
        prune_linear_layer)
except:                       # bắt TẤT CẢ lỗi, không in ra gì
    PreTrainedModel = None    # ...rồi gán tất cả = None
...
class BertPreTrainedModel(PreTrainedModel):   # nhưng VẪN đem None ra làm lớp cha
```

transformers **5.17.0** (bản mới nhất trên PyPI ngày 30/09/2026; đã đối chiếu
source tại tag `v5.17.0`) đã **chuyển** `apply_chunking_to_forward` và
`prune_linear_layer` sang `transformers.pytorch_utils`, và **bỏ hẳn**
`find_pruneable_heads_and_indices`. Khối import thất bại → `PreTrainedModel = None`
→ `class X(None)` ném `TypeError: NoneType takes no arguments`, làm sập **cả gói**
`mmpretrain.models`. Hậu quả không nhỏ: config ConvNeXt của mmdet đặt
`allow_failed_imports=False` nên mất mmpretrain là chết ngay — mà mmdet 3.3.0
**không có** ConvNeXt riêng (đã liệt kê `mmdet/models/backbones/`: chỉ có
resnet, resnext, swin, cspnext, pvt, regnet… **không có convnext**).

Cách sửa: **tắt cờ `WITH_MULTIMODAL`** — đúng công tắc của OpenMMLab, không phải
vá chắp vá. Khi tắt, `register_multimodal_placeholder()` đăng ký bản thay thế cho
các lớp đó (ai lỡ dùng sẽ nhận thông báo "cần cài mmpretrain[multimodal]").
Đã kiểm tra `__all__.extend([...])` nằm **trong** nhánh `if WITH_MULTIMODAL:`
nên tắt cờ **không** gây lỗi `from .utils import *`.

⚠️ **Bẫy phụ khi vá:** phải **xoá `mmpretrain*` khỏi `sys.modules`** sau khi vá.
`mmpretrain.utils.dependency` đã được nạp với giá trị cờ CŨ; không xoá thì lần
import sau vẫn dùng lại giá trị cũ và việc vá coi như không có tác dụng.

*Đã đối chiếu:* Colab cấp `transformers` **5.16.1** (không phải 5.17.0 — bản mới
nhất trên PyPI lúc kiểm). Source đọc để tìm nguyên nhân là tag `v5.17.0`, nhưng
lỗi tái hiện y hệt trên 5.16.1, và cách vá vẫn đúng vì nó **không phụ thuộc tên
hàm nào cả** — chỉ tắt một cờ.

### 1.3 Kiểm chứng chạy được trên máy cá nhân (CPU, không cần GPU)

- **Logic dò wheel mmcv**: chạy thật với index, đọc được **1466** tên `.whl`.
  Bốn tổ hợp thử:
  - `cp313 + pt2.11.0cu128` → `mmcv-2.2.0+a8073c7pt2.11.0cu128-cp313-cp313-linux_x86_64.whl`
  - `cp313 + pt2.10.0cu126` → tìm thấy
  - `cp314 + pt2.13.0cu130` → tìm thấy
  - tổ hợp vô lý `pt1.0.0cu999` → không tìm thấy (đúng như mong đợi, nhánh báo lỗi hoạt động)
  - Không có tên nào bị cắt cụt do khớp vào `href` (bẫy đã biết: `href` mã hóa
    dấu `+` thành `%2B`).

### 1.4 Checkpoint pretrain — kiểm tra HTTP 200 và kích thước thật

| Dùng cho | URL | Dung lượng |
|---|---|---|
| Smoke test Phase 0 + E2 (Cascade Mask R-CNN ConvNeXt-T, COCO) | `download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/...8f07c40b.pth` | 343.842.335 B (344 MB) |
| Phương án nhẹ hơn / E7 (Mask R-CNN ConvNeXt-T, COCO) | `.../mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco/...050731f4.pth` | 192.446.607 B (192 MB) |
| E1 — khởi tạo ImageNet cho ConvNeXt-T | `download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth` | 114.410.172 B (114 MB) |
| Ảnh demo cho smoke test | `<repo mmdetection>/demo/demo.jpg` (có sẵn trong bản clone) | 259.865 B |

### 1.5 Colab thực tế cấp gì — đo ngày 30/09/2026

Chạy ô [0.4] và [0.7] trên Colab, runtime **T4 GPU**:

- **Python 3.13.15** · **PyTorch 2.11.0+cu128** · **CUDA 12.8** · GPU khả dụng: True
  → khớp **đúng** tổ hợp `cp313 + pt2.11.0 + cu128` mà notebook tự dò ra.
- Wheel mmcv cài được **đúng tên đã tìm thấy trước đó trên máy CPU**:
  `mmcv-2.2.0+a8073c7pt2.11.0cu128-cp313-cp313-linux_x86_64.whl`
- `mmcv.ops.nms` chạy **thật trên GPU**: giữ lại 2/3 box, đúng kỳ vọng → wheel có
  CUDA ops thật (không phải mmcv-lite) và **không thiếu kiến trúc `sm_75`** của T4.
- mmengine 0.10.7 → hàm vá `Adafactor` báo *"không cần"*, đúng như đã đọc ở §1.2b.
- `transformers` do Colab cấp: **5.16.1**.
- Sau khi vá `WITH_MULTIMODAL` (§1.2g), chạy lại ô [0.7] — **đã hết lỗi**:
  `from mmpretrain.models import ConvNeXt` → **OK**, và cờ báo `TẮT (đúng như đồ án cần)`.
- `TorchAdafactor=True, Adafactor=True` → không còn trùng tên optimizer (§1.2b).
- **Smoke test đạt**: Cascade Mask R-CNN + ConvNeXt-Tiny (COCO) chạy suy luận 1 ảnh
  trong **2,02 giây** trên T4, trả về 44 box, lớp cao nhất `car 0.999` / `bench 0.997`
  — đúng với ảnh demo (cảnh đường phố). Nghĩa là `roi_align` và `nms` của mmcv chạy
  thật trong mạng đầy đủ, wheel **không thiếu kiến trúc `sm_75`** của T4.
  (Nạp model 12 giây; clone repo mmdet chỉ 3 giây.)
- **Đường ghi ra Drive hoạt động thật**: kiểm `runs/phase0_smoke/vis/` thấy tệp
  `demo.jpg` nặng **179.472 byte**, giờ 10:16 — khớp lượt chạy GATE 0. Ảnh lỗi hoặc
  rỗng sẽ chỉ vài trăm byte hoặc 0 byte, nên đây là ảnh có nội dung thật.
  → Xác nhận **không chỉ đường dẫn được tạo** mà ghi tệp ra Drive thật sự chạy.
  Đây chính là đường mà Phase 4–6 dựa vào để lưu checkpoint và kết quả; nếu nó hỏng
  mà không biết thì sẽ train xong mới phát hiện mất trắng.

**Báo cáo GATE 0 đầy đủ — 30/09/2026 10:17 giờ Colab, Tesla T4 15,6 GB:**

```
mmcv          2.2.0+a8073c7pt2.11.0cu128   <- bản dựng cộng đồng, CÓ CUDA ops
mmengine      0.10.7      mmdet 3.3.0      mmpretrain 1.2.0 (đã tắt WITH_MULTIMODAL)
pycocotools   2.0.11      transformers 5.16.1   numpy 2.1.3   opencv-python 5.0.0.93
floodnet_raw.zip: 13,06 GB — có đúng trên Drive
/content: 113 GB, còn trống 65 GB
```

Đáng lưu ý cho các phase sau:

- **opencv-python 5.0.0.93** — major version mới (4.x → 5.x). Smoke test chỉ chạy suy
  luận nên chưa đụng nhiều tới `cv2`; pipeline dữ liệu ở Phase 1–2 mới dùng nhiều
  (`cv2.imread`, connected components) — nếu có lỗi API thì sẽ lộ ra ở đó.
- **numpy 2.1.3** — numpy 2.x từng làm vỡ nhiều thư viện cũ; hiện chưa thấy vấn đề.
- `/content` chỉ còn **65 GB** trống: dataset giải nén (~13 GB) + dữ liệu đã xử lý
  phải nằm gọn trong đó.

Lỗi gặp ở lượt chạy đầu tiên (runtime **CPU-only**, trước khi bật T4): runtime đó
**không có lệnh `nvidia-smi`**, gọi thẳng bằng `subprocess.run` ném
`FileNotFoundError` và **làm sập cả ô**, che mất phần in đĩa và RAM phía sau.
→ Ô [0.4] đã thêm hàm `lenh()` bắt lỗi này.

### 1.6 `/content` mất sạch khi runtime reset — gặp thật ngày 30/09/2026

Diễn biến đo được:

1. `git clone` repo về `/content/flood-house-counting` → **thành công**
   (23 object, 35,52 KiB, log đầy đủ).
2. Ngay sau đó, `!cd /content/flood-house-counting` → `No such file or directory`.
3. `!ls /content` → chỉ còn `sample_data`.

Nguyên nhân: runtime Colab đã **khởi động lại** giữa hai lệnh. `/content` là **đĩa
tạm**, bị xoá sạch mỗi lần runtime reset — kể cả khi người dùng đổi
`Runtime → Change runtime type` (ví dụ bật T4 GPU). Không phải lỗi của repo, không
phải lỗi của `.gitignore`; clone đã thành công thật rồi mới bị xoá.

Hệ quả bắt buộc cho **mọi phase sau**:

- Đầu **mỗi phiên** Colab phải `git clone` lại repo (~3 giây). Đây là việc bình
  thường phải làm, không phải sự cố.
- **Không bao giờ để kết quả chỉ nằm ở `/content`** — hết phiên là mất trắng.
- Checkpoint, log, kết quả đánh giá **phải ghi ra Drive** (`MyDrive/Flood_House_AI/runs/`),
  vì đó là chỗ duy nhất còn dữ liệu sau khi runtime reset. Đây chính là lý do kỹ
  thuật để trả lời khi bảo vệ.

### 1.7 Console Windows không in được tiếng Việt — gặp thật 30/09/2026

Lỗi bắt được khi viết test cho `audit.py`, không phải suy đoán:

```
> py scripts/audit.py --help
UnicodeEncodeError: 'charmap' codec can't encode character 'ả' ... cp1252.py
```

Nguyên nhân: console Windows mặc định dùng bảng mã **cp1252** (Tây Âu), không có
chữ `ả`. Python gặp chữ có dấu khi ghi ra stdout là ném lỗi ngay và thoát.

Hệ quả nếu không sửa: **mọi script in tiếng Việt đều chết trên máy cá nhân** —
kể cả `--help`. Trên Colab không thấy vì ở đó mặc định là UTF-8, nên lỗi này chỉ
lộ ra khi chạy trên máy, rất dễ bị bỏ qua rồi mới vỡ lúc cần dùng.

Cách sửa (đã áp dụng ở đầu `main()` của `audit.py`):

```python
for luong in (sys.stdout, sys.stderr):
    try:
        luong.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
```

Trên Colab vốn đã là UTF-8 nên bước này không đổi gì. Test mục 11 ép
`PYTHONIOENCODING=cp1252` rồi chạy `--help` để **giữ lỗi này không quay lại** —
sau này thêm script mới thì chép luôn đoạn trên vào `main()`.

### 1.8 Colab KHÔNG hiện stderr của tiến trình con — gặp thật 30/09/2026

Triệu chứng người dùng gặp khi chạy ô `[1.5]` lần đầu:

```
Chạy: /usr/bin/python3 scripts/audit.py --config configs/data.yaml ...
========================================================================
========================================================================
Mã thoát: 1 (0 là thành công)
```

Hai dòng `====` nằm sát nhau, **không có một dòng nào ở giữa**. Script thoát với
mã 1 mà không nói gì.

**Hai nguyên nhân chồng lên nhau, cả hai đều là lỗi thiết kế của mình:**

1. **`print` đầu tiên nằm quá muộn.** Trong `audit.py`, dòng in băng-rôn ở sau
   `open(args.config)`. File config mở không được → script thoát trước khi in
   được gì → stdout trống trơn.
2. **Trên Colab, tiến trình con thừa hưởng `fd 1`/`fd 2` thì không hiện gì cả.**
   (Kết luận ban đầu của mình — "chỉ stderr bị giấu" — **sai, đính chính ngay
   trong mục này**.) Bằng chứng: lần chạy thử cuối cùng thoát với **mã 0**, tức là
   script chạy trọn vẹn và có in đủ báo cáo, vậy mà ô output vẫn **trống trơn**.
   Vậy cả stdout lẫn stderr của tiến trình con đều không chảy vào ô output; chỉ
   những gì đi qua `sys.stdout` của kernel mới hiện.

   Lần duy nhất nhìn thấy output là lần dùng `subprocess.run(..., stdout=PIPE)`
   rồi tự `print` — chính là mấu chốt để sửa.

Đây là bẫy nguy hiểm vì "không có thông báo lỗi" trông giống hệt "script chạy
xong không có gì để in" — mất hẳn manh mối để chẩn đoán.

**Đã sửa, ba chỗ:**

- `audit.py`: in băng-rôn **trước** khi mở config, và kiểm file config có tồn tại
  không rồi in thông báo tiếng Việt rõ ràng (kèm gợi ý chạy lại ô `[1.3]`).
  Nguyên tắc rút ra: **dòng in đầu tiên phải nằm trước mọi thứ có thể thất bại.**
- Notebook: hai ô chạy **tự đọc ống dẫn rồi in qua `sys.stdout` của kernel**, chứ
  không để tiến trình con thừa hưởng fd 1. Vẫn giữ được log hiện dần theo thời
  gian thực. Kèm `-u` cho python con, vì không có nó thì chính tiến trình con lại
  đệm theo khối và màn hình im lặng vài phút rồi output mới ập ra một lúc.

  ```python
  kq = subprocess.Popen(lenh, cwd=cwd, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
  for dong in kq.stdout:
      print(dong, end="")
  kq.wait()
  ```

  **Ô chạy nào sau này thêm vào cũng phải làm theo cách này.** Test mục 11 khoá
  lại: cấm `subprocess.run(lenh_thu...)`, bắt buộc có `-u` và có hàm đọc ống.
- Test mục 10 và 11 kiểm cả hai điều trên, để không tái phát.

**Còn một lớp bẫy nữa ở notebook:** Colab chạy từng ô độc lập, nhảy thẳng xuống
ô `[1.5]` mà chưa chạy ô `[1.1]` thì lỗi `NameError: name 'EDA_OUT' is not defined`
— trông như lỗi code nhưng thật ra chỉ là chưa chạy ô cấu hình. Ba ô chạy nay đều
tự kiểm và báo rõ phải chạy ô `[1.1]` trước.

---

## 2. Dataset FloodNet

### 2.1 Đã kiểm chứng ngày 16/09/2026 — trên đúng file `floodnet_raw.zip` này

- Cấu trúc zip: `FloodNet-Supervised_v1.0/` (dùng bản này) và
  `ColorMasks-FloodNetv1.0/` (ảnh màu **chỉ để xem**, không dùng để train).
  ⚠️ Nếu trỏ nhầm vào thư mục gốc, công cụ dò theo tên sẽ tưởng `ColorMasks` là
  mask (vì tên có chữ "mask") → **train trên dữ liệu sai mà không báo lỗi**.
- Mask là ảnh xám `mode=L`, kích thước **4000×3000** (khác `ColorMasks` 1024×1024).

**Cấu trúc chính xác bên trong zip — đo lại ngày 30/09/2026 bằng `audit.py`:**

| Thư mục | Số file | Đo được |
|---|---|---|
| `FloodNet-Supervised_v1.0/train/train-org-img/` | 1445 | ảnh, 3 kênh, 4000×3000 |
| `FloodNet-Supervised_v1.0/train/train-label-img/` | 1445 | mask, 1 kênh, 4000×3000 |
| `FloodNet-Supervised_v1.0/val/val-org-img/` | 450 | ảnh |
| `FloodNet-Supervised_v1.0/val/val-label-img/` | 450 | mask |
| `FloodNet-Supervised_v1.0/test/test-org-img/` | 448 | ảnh |
| `FloodNet-Supervised_v1.0/test/test-label-img/` | 448 | mask |
| `ColorMasks-FloodNetv1.0/ColorMasks-{Train,Val,Test}Set/` | 1445 / 450 / 448 | ảnh MÀU 3 kênh, 1024×1024 |

Hai kết luận quan trọng rút ra từ bảng này:

1. **Tập test CÓ nhãn** (448 mask) — không phải tự chia lại val thành val/test. Giữ
   nguyên split gốc của FloodNet, đúng tinh thần "test chỉ chạy một lần" ở Phase 5.
2. **Ảnh và mask cùng kích thước** 4000×3000 — không phải resize mask trước khi lấy
   box. (Vẫn phải resize vì lý do khác: đưa về `target_long_side` cho vừa VRAM.)

**Quy ước đặt tên — chỗ đã làm hỏng lần chạy thật đầu tiên:**

```
ảnh : FloodNet-Supervised_v1.0/train/train-org-img/1234.jpg
mask: FloodNet-Supervised_v1.0/train/train-label-img/1234_lab.png
```

Hai bên **KHÔNG trùng tên**: mask có thêm hậu tố `_lab`. Ghép cặp bằng cách so tên
nguyên bản thì không file nào khớp, dù hai thư mục bằng nhau đúng 1445 file. Phải bỏ
hậu tố `_lab` trước khi so (`chuan_hoa_stem()` trong `audit.py`).

Bài học: bản test đầu tiên đặt ảnh và mask **trùng tên** nên 56 assertion đều pass
mà vẫn để lọt lỗi này. Dữ liệu giả phải bắt chước cả **quy ước đặt tên** của dữ liệu
thật, không chỉ cấu trúc thư mục.
- Bảng lớp (đo bằng tỉ lệ mảng liền kề chạm lớp `Water`, không phải đoán):

  | Giá trị | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Lớp | Background | Building-Flooded | Building-Non-Flooded | Road-Flooded | Road-Non-Flooded | Water | Tree | Vehicle | Pool | Grass |

- Phát hiện quan trọng cho báo cáo: chỉ **~22%** số nhà gán nhãn "ngập" thực sự
  chạm vùng nước; nhãn `Building-Flooded` của FloodNet rộng hơn định nghĩa
  thường hiểu. Hệ quả: **đừng kỳ vọng mAP@50 > 0,9**; khoảng hợp lý là 0,4–0,7.

### 2.2 Phase 1 PHẢI xác nhận lại (chưa kiểm cho repo này)

> **Trạng thái 30/09/2026:** công cụ đã viết xong và test xong trên dữ liệu giả,
> **nhưng chưa chạy trên zip thật**. Các ô dưới đây vẫn để trống cho tới khi chạy
> `notebooks/01_data_prep.ipynb` trên Colab và dán kết quả về. Khi đó điền số thật
> vào đây — đây chính là nội dung GATE 1.

Các số dưới đây chỉ là **kỳ vọng** để đối chiếu, chưa phải kết luận:

- [x] Số cặp (ảnh, mask) thực tế trong `FloodNet-Supervised_v1.0` và tên các split
      → **1445 train / 450 val / 448 test**, xem bảng ở §2.1. (đo 30/09/2026)
- [x] Tập **test có nhãn hay không** → **CÓ**, 448 mask. Giữ nguyên split gốc. (đo 30/09/2026)
- [x] Ảnh và mask có cùng kích thước không → **cùng 4000×3000**, không lệch. (đo 30/09/2026)
- [ ] Số ảnh mỗi split, số pixel mỗi lớp, số ảnh có/không có nhà ngập
- [ ] Phân bố diện tích nhà và **cạnh box nhỏ nhất sau khi resize** → quyết định
      có cần cắt tile hay không (ngưỡng đã chốt trong plan: percentile 5 ≥ ~32px thì không cần)

**Hai điểm lệch có chủ ý so với câu chữ của plan — để bảo vệ được:**

1. Plan viết "giải nén `floodnet_raw.zip` vào `/content`". Bản cài đặt **đọc thẳng
   từ trong zip**, không giải nén. Lý do: zip ~12 GB, giải nén ra gấp đôi và `/content`
   chỉ có ~100 GB dùng chung với dataset đã xử lý ở Phase 2; mà Phase 1 chỉ cần đọc
   từng ảnh một rồi bỏ. Kết quả đo không đổi.
2. Plan viết overlay ra `outputs/eda/`. Bản cài đặt ghi ra
   `MyDrive/Flood_House_AI/runs/eda/overlay/`. Lý do: `outputs/` nằm trong `.gitignore`
   và `/content` mất sạch khi runtime reset (xem §1.6) — để trên Drive thì ảnh còn
   nguyên mà mở xem bằng điện thoại/máy tính cũng được.

Một điểm **khác plan có chủ ý**: plan nói đo phân bố kích thước nhà "sau khi resize".
Bản cài đặt resize mask bằng `cv2.INTER_NEAREST` rồi mới tách component — đúng như
vậy, vì nội suy tuyến tính trên ảnh nhãn sẽ **sinh ra giá trị lớp không tồn tại**
(ví dụ giữa lớp 1 và lớp 2 nội suy ra 1.5 → làm tròn thành lớp 2, sai nhãn).

---

## 3. Việc tiếp theo

1. ~~Chạy `notebooks/00_colab_setup.ipynb` trên Colab (GPU T4) → chốt GATE 0.~~
   **Xong 30/09/2026** — xem §1.5 và §1.6.
2. Repo đã đẩy lên GitHub: **https://github.com/21xDHoang/flood-house-counting**
   (public, nhánh `main`). Colab clone repo này về `/content` ở đầu mỗi phiên.
3. **Phase 1 — code xong, chờ chạy thật để chốt GATE 1.** Đã đẩy lên GitHub:
   - `configs/data.yaml` — toàn bộ tham số (đường dẫn, ngưỡng quyết định, tham số EDA)
   - `src/floodcount/data/audit.py` — khảo sát mask thật, trả lời checklist §2.2
   - `scripts/audit.py` — CLI mỏng bọc quanh module trên
   - `tests/test_audit.py` — 46 assertion trên zip giả có cả bẫy ColorMasks
   - `notebooks/01_data_prep.ipynb` — notebook chạy trên Colab (không cần GPU)

   **Việc của người dùng:** mở `notebooks/01_data_prep.ipynb` trên Colab, chạy ô
   [1.5] (chạy thử 3 ảnh) trước, rồi ô [1.6] (chạy đầy đủ), rồi gửi lại phần
   `CẤU TRÚC ZIP`, nội dung `EDA_REPORT.md`, và nhận xét bằng mắt về ảnh overlay.

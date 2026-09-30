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
- Mask là ảnh xám `mode=L` (khác `ColorMasks` 1024×1024). Kích thước **KHÔNG đồng
  nhất**: lần chạy đầy đủ 30/09/2026 đếm được **1991 ảnh 4000×3000** và **352 ảnh
  4592×3072** (xem §2.4) — không phải "tất cả 4000×3000" như ghi nhận ban đầu. Bảng
  phân loại thư mục ở dưới chỉ giải mã **một** file mỗi thư mục, nên nó in ra một cỡ
  duy nhất cho cả thư mục; đó là kết luận quá rộng so với phép đo.

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
2. **Ảnh và mask cùng kích thước** — nên không phải resize mask cho khớp ảnh trước khi
   lấy box. (Vẫn phải resize vì lý do khác: đưa về `target_long_side` cho vừa VRAM.)
   ⚠️ Mức độ tin cậy của kết luận này **thấp hơn hai kết luận trên**: bảng phân loại
   chỉ giải mã 1 file mỗi thư mục, còn `audit.py` chỉ đọc **mask** của cả 2343 cặp.
   Đối chiếu được trên 30 ảnh overlay (không ảnh nào báo lệch) và trên §2.4 (352 ảnh
   cỡ khác). **Phase 2 phải khẳng định lại bằng một dòng assert cho TỪNG cặp** — nếu
   15% dữ liệu lệch cỡ thì box lệch chỗ và hỏng cả tập train mà không ai báo.

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

- ~~Phát hiện quan trọng cho báo cáo: chỉ **~22%** số nhà gán nhãn "ngập" thực sự chạm
  vùng nước~~ → **con số 22% KHÔNG tái lập được** (đo lại 30/09/2026, xem §2.4): trên
  mẫu 60 ảnh, tỉ lệ component chạm nước là 83,3% với `flooded_building` và 65,7% với
  `non_flooded_building`. Hướng thì đúng (nhà ngập chạm nước nhiều hơn), nhưng mẫu nhỏ
  và phép đo bị pha loãng bởi đốm nhiễu chưa lọc, nên **chưa đủ cơ sở để trích dẫn một
  con số nào**. Mục "kiểm chứng lớp 1" phải đo lại tử tế trước khi đưa vào báo cáo.
  Lập luận "nhãn `Building-Flooded` rộng hơn định nghĩa thường hiểu" thì **vẫn giữ**:
  hai lớp chạm nước 83% và 66% là gần nhau, tức vùng nước không phải thứ phân biệt
  chúng. Hệ quả vẫn là: **đừng kỳ vọng mAP@50 > 0,9**; khoảng hợp lý 0,4–0,7.

### 2.2 Phase 1 PHẢI xác nhận lại (chưa kiểm cho repo này)

> **Trạng thái 30/09/2026:** công cụ đã viết xong, test xong trên dữ liệu giả, và đã
> chạy **thử 9 ảnh** trên zip thật (kết quả ở §2.3). Các ô dưới đây vẫn để trống cho
> tới khi chạy **đầy đủ** `notebooks/01_data_prep.ipynb` trên Colab và dán kết quả về.
> Khi đó điền số thật vào đây — đây chính là nội dung GATE 1.

Các số dưới đây chỉ là **kỳ vọng** để đối chiếu, chưa phải kết luận:

- [x] Số cặp (ảnh, mask) thực tế trong `FloodNet-Supervised_v1.0` và tên các split
      → **1445 train / 450 val / 448 test**, xem bảng ở §2.1. (đo 30/09/2026)
- [x] Tập **test có nhãn hay không** → **CÓ**, 448 mask. Giữ nguyên split gốc. (đo 30/09/2026)
- [x] Ảnh và mask có cùng kích thước không → **cùng 4000×3000**, không lệch. (đo 30/09/2026)
- [x] Số ảnh mỗi split, số pixel mỗi lớp, số ảnh có/không có nhà ngập
      → **1445/450/448 cặp**; đủ **cả 10 giá trị mask**; **chỉ 245 ảnh (10,5%) có nhà
      ngập** và 880 ảnh (37,6%) có nhà không ngập — dữ liệu thưa hơn nhiều so với dự
      đoán, kéo theo hệ quả cho cả Phase 3 (lấy mẫu) lẫn Phase 5 (cách chấm điểm).
      Bảng đầy đủ ở §2.4. (đo 30/09/2026, chạy đầy đủ 2343 ảnh)
- [x] Phân bố diện tích nhà và **cạnh box nhỏ nhất sau khi resize** → đã đo, xem §2.4.
      Kết quả: p5 (đã lọc nhiễu) = **28.0px** < 32px → theo đúng quy tắc đã chốt trong
      plan thì **CẦN cắt tile**. Đây là kết quả SÁT NGƯỠNG (28 so với 32) và toàn bộ
      phần thiếu đến từ một lớp, nên phải bàn kỹ ở GATE 1 — xem lập luận ở §2.4.

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

### 2.3 Lần chạy thử 9 ảnh trên zip THẬT — hai lỗi phương pháp lộ ra (30/09/2026)

Chạy `--max-images 3` (3 ảnh mỗi split, tổng 9 ảnh). Mục đích ban đầu chỉ là kiểm cấu
trúc, nhưng chính lần chạy nhỏ này phát hiện hai lỗi mà 62 assertion lúc đó trên dữ liệu
giả không bắt được. **Chưa kết luận gì về dataset từ 9 ảnh** — số liệu dưới đây chỉ dùng
để soi lỗi công cụ.

**Lỗi 1 — nhiễu gán nhãn đang quyết định thay nhà thật.**

Báo cáo in ra: `flooded_building` có `p5 = 1.0px` trong khi `p50 = 118.0px`, và căn cứ
vào đó kết luận **"Cắt tile: CÓ"**. Nhưng p5 = 1px là **đốm vài pixel** trong mask, không
phải căn nhà nào. Nghĩa là một quyết định về độ phân giải của cả đồ án lại do rác gán
nhãn quyết định.

Cách sửa: thêm `decisions.min_side_px = 8` vào config. Component có cạnh nhỏ nhất < 8px
bị loại **trước khi tính mọi thống kê và mọi kết luận**. Bảng báo cáo giữ **cả hai cột**
(`p5 đã lọc` và `p5 thô`) để người đọc tự thấy mức nhiễu — giấu cột thô đi thì không ai
kiểm chứng được nữa.

Vì sao lọc theo **cạnh nhỏ nhất** chứ không theo diện tích: một mảng 1×500 pixel có diện
tích lớn hơn ngưỡng nhưng vẫn là vệt rác, không phải nhà. `preprocess.min_area` cũng đổi
thành `min_side_px² = 64` để hai ngưỡng không mâu thuẫn nhau.

**Lỗi 2 — lớp bắt buộc phải có mà không có pixel nào, báo cáo chỉ ghi một dòng `0`.**

Trong 9 ảnh này lớp 1 (`flooded_building`) **không có pixel nào**. Nếu cứ thế mà train
thì model học lớp "nhà ngập" từ hư không. Báo cáo cũ chỉ in một dòng `| flooded_building |
1 | 0 |` giữa bảng — rất dễ đọc lướt qua.

Cách sửa: `ket_luan()` kiểm việc này **trước mọi kết luận khác**, và báo cáo đặt khối
`> ## ⚠️ CẢNH BÁO` **ngay đầu file, trước cả bảng số liệu**.

**Câu hỏi còn để mở — phải chạy đầy đủ mới trả lời được:**

Bảng "Số pixel mỗi giá trị mask" của 9 ảnh đó chỉ có các giá trị `{2, 4, 5, 6, 7, 8, 9}`;
các giá trị `0` (nền), `1` (Building-Flooded), `3` (Road-Flooded) đều **không xuất hiện**.
Thiếu `0` là chuyện đáng ngờ — mask nào cũng phải có nền. Hai khả năng:

1. 9 ảnh này không đại diện: chúng là 3 cụm ID gần nhau (10168, 10169, 10173), nhiều khả
   năng cùng một chuyến bay, cùng một khu — nên cùng thiếu một số lớp.
2. Bảng lớp trong §2.1 sai với bản zip này, và giá trị lớp bị lệch.

Phân biệt được bằng cách chạy **đầy đủ** rồi đọc lại đúng bảng đó. Nếu chạy hết 2343 ảnh
mà `1` vẫn không có pixel nào thì gần như chắc chắn là khả năng 2, phải dừng lại tra bảng
lớp trước khi sang Phase 2.

Báo cáo có thêm **bảng số pixel theo từng split** (train/val/test) vì câu hỏi tiếp theo
luôn là "vắng ở mọi split hay chỉ một split" — trả lời sẵn trong báo cáo thì không phải
chạy lại 2343 ảnh chỉ để hỏi một câu. Tỉ lệ tính riêng trong từng split: ba split lệch
số ảnh (1445/450/448) nên lấy tổng chung làm mẫu số thì so với nhau là sai.

Trạng thái đường ống: đã đóng gói lại thành `min_side_px` trong config, cảnh báo lớp vắng
mặt, và **8 assertion mới** khoá hai hành vi này lại (`tests/test_audit.py` mục 12) —
trong đó có ca "nhà to lẫn một đốm nhiễu": lấy nhầm cột thì kết luận đảo ngược từ KHÔNG
thành CÓ.

### 2.4 Chạy ĐẦY ĐỦ 2343 ảnh — số liệu thật của Phase 1 (30/09/2026)

Chạy `notebooks/01_data_prep.ipynb` ô [1.6] trên Colab, mã thoát 0. Đây là **nội dung
GATE 1**.

**Cấu trúc & ghép cặp — khớp hoàn toàn với §2.1:** 1445 + 450 + 448 = 2343 cặp, không
thiếu file nào; bẫy `ColorMasks` bị chặn đúng (3 thư mục, 2343 file bị bỏ qua).

**Hai cỡ ảnh, không phải một:**

| Cỡ mask | Số ảnh | Tỉ lệ resize về cạnh dài 1536 |
|---|---|---|
| 4000×3000 | 1991 | 0,3840 |
| 4592×3072 | 352 | 0,3345 |

Hệ quả: độ phân giải thực trên mặt đất của 352 ảnh này **nhỏ hơn 13%** so với phần còn
lại, tức cùng một căn nhà cho ra box nhỏ hơn 13%. Không sai, nhưng là lý do chính đáng
để bật augmentation đổi tỉ lệ khi train (Phase 3).

**Số pixel mỗi giá trị mask (toàn bộ 2343 ảnh):**

| Giá trị | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Vai trò | nền | nhà ngập | nhà khô | đường ngập | đường khô | nước | cây | xe | bể bơi | cỏ |
| Tỉ lệ | 1,92% | 1,84% | 3,18% | 2,97% | 5,59% | 11,11% | 17,77% | 0,18% | 0,20% | 55,24% |

→ **Câu hỏi để mở ở §2.3 đã có lời giải: cả 10 giá trị đều có mặt.** Lớp 1
(`flooded_building`) tồn tại thật (1,84% số pixel, 3534 box), nên bảng lớp ở §2.1
**đúng**, và 9 ảnh mẫu trước đó chỉ là không đại diện. Không phải dừng lại tra bảng lớp.
Tỉ lệ theo từng split cũng cân (lớp 1: 1,79% train / 1,99% val / 1,87% test).

**⚠️ Phát hiện quan trọng nhất của lần chạy này — dữ liệu cực kỳ thưa ở mức ảnh:**

| Lớp | Ảnh CÓ ít nhất 1 nhà | Ảnh KHÔNG có nhà nào | Box mỗi ảnh CÓ nhà |
|---|---|---|---|
| flooded_building | **245 (10,5%)** | 2098 (89,5%) | 14,4 |
| non_flooded_building | 880 (37,6%) | 1463 (62,4%) | 4,5 |

**89,5% số ảnh không có một căn nhà ngập nào.** Con số "box/ảnh TB = 1,51" ở bảng dưới
là trung bình trên *toàn bộ* 2343 ảnh; trên những ảnh thật sự có nhà ngập thì là 14,4
box/ảnh. Hai mặt của cùng một dữ liệu, nhưng mặt thứ hai mới là mặt quyết định cách
train và cách chấm điểm:

- Chỉ ~150 ảnh train có nhà ngập (10,5% của 1445) — model sẽ **đói dương tính**. Phase 3
  phải cân nhắc `RepeatFactorTrainingDataset` hoặc lấy mẫu cân bằng, đừng để mặc định
  lấy đều (mỗi batch ~10% ảnh có lớp cần học).
- **Chấm điểm đếm nhà không được dùng "accuracy"**: đoán 0 cho mọi ảnh đã đúng 89,5%.
  Phase 5 phải dùng MAE/RMSE trên số nhà mỗi ảnh, và báo cáo riêng trên tập ảnh CÓ nhà.

**Thống kê box (sau resize cạnh dài 1536, đã bỏ đốm nhiễu < 8px):**

| Lớp | Số box | Bỏ do nhiễu | Box/ảnh TB | Max | p5 đã lọc | p50 đã lọc | p5 thô | DT p50 thô | DT p50 đã lọc | Box to bất thường |
|---|---|---|---|---|---|---|---|---|---|---|
| flooded_building | 3534 | 433 (10,9%) | 1,51 | 45 | 44,0 | 166,0 | 1,0 | 17.987 | 19.580 | 10 (0,3%) |
| non_flooded_building | 3985 | 744 (15,7%) | 1,70 | 54 | **28,0** | 148,0 | 1,0 | 12.285 | 20.618 | 747 (23,0%) |

Đọc bảng này:

- **1177 đốm nhiễu đã bị loại** (13,5% tổng số component). Không lọc thì cả hai lớp đều
  có p5 = 1,0px và mọi kết luận đều do rác quyết định — đúng lỗi 1 ở §2.3.
- Nhà thật **rất to so với ngưỡng**: p50 = 148–166px, gấp ~5 lần mốc 32px. Chỉ có đuôi
  nhỏ nhất mới đáng lo.
- Hai cột `DT p50` cho thấy mức thiệt hại của lỗi 3: với `non_flooded_building`, trung vị
  thô (12.285) **thấp hơn 40%** so với trung vị đã lọc (20.618) — mốc 3× vì thế bị hạ
  xuống chỉ còn ~60% giá trị đúng.

**Kết luận tự động của công cụ và đánh giá của tôi:**

1. **Cắt tile: theo quy tắc đã chốt thì CÓ** — nhưng đây là kết quả **sát ngưỡng** và
   cần bàn: p5 = 28,0px < 32px, mà **toàn bộ phần thiếu đến từ `non_flooded_building`**
   (`flooded_building` một mình là 44,0px, vượt ngưỡng thoải mái). Nghĩa là quyết định
   độ phân giải của cả đồ án đang do một lớp quyết định, và chỉ vì 28 < 32 đúng 4px.
2. **Tách nhà dính: số liệu nói CÓ (11,9%, sau khi sửa lỗi 3) — nhưng TÔI CHO RẰNG CHỈ
   SỐ NÀY KHÔNG ĐO CHUYỆN GỘP NHÀ.** Lý do: `flooded_building` chỉ 0,3% box bị gắn cờ
   còn `non_flooded_building` tới 23,0% — chênh **75 lần**. Nếu là nhà dính nhau thật
   thì hai lớp phải na ná nhau (nhà kề nhà thì kề như nhau, ngập hay không ngập không
   ảnh hưởng gì). Chênh lệch cỡ đó nghĩa là chỉ số đang đo **phân bố kích thước nhà**
   (nhà ống vs nhà xưởng), không đo chuyện gộp. Sửa lỗi 3 làm con số tổng giảm từ 17,5%
   xuống 11,9% nhưng **độ lệch giữa hai lớp thì vẫn nguyên** — càng chắc rằng nó nằm
   trong bản chất dữ liệu, không phải trong cách tính.
3. **`min_area` = 64** (bằng `min_side_px²`) — giữ nguyên như config.

**Lỗi thứ ba lộ ra ở lần chạy này — mốc "3× trung vị" tính sai.**

Tử số đã lọc nhiễu (chỉ đem box to đi so) nhưng **mẫu số/mốc vẫn lấy trung vị của tất cả
box**, kể cả 744 đốm của `non_flooded_building`. Đốm kéo trung vị xuống, hạ thấp mốc 3×,
và mọi căn nhà to thật đều vượt mốc. Mức thiệt hại đo được: trung vị thô của
`non_flooded_building` là 12.285 còn trung vị đã lọc là 20.618 (**lệch 40%**), và số box
bị gắn cờ tụt từ 1096 xuống 747 khi sửa. Đã sửa: mốc tính trên trung vị **đã lọc**; có
test riêng (`tests/test_audit.py` mục 14) dựng đúng ca "2 nhà to + 6 đốm" — lấy trung vị
thô thì kết luận đảo ngược thành "CÓ".

**Kiểm chứng "nhà ngập nằm cạnh nước" — yếu, chưa dùng được:**

| Lớp | Tổng component | Chạm nước | Tỉ lệ |
|---|---|---|---|
| flooded_building | 18 | 15 | 83,3% |
| non_flooded_building | 67 | 44 | 65,7% |

Hướng đúng (nhà ngập chạm nước nhiều hơn) nhưng **mẫu quá nhỏ** (27 ảnh, 18 component) và tỉ lệ
bị pha loãng vì tính trên mọi component kể cả đốm nhiễu chưa lọc — đốm nhiễu hiếm khi
chạm nước nên tỉ lệ thật phải cao hơn bảng này. Đây là lý do con số "~22%" ở §2.1 không
tái lập được. Muốn trích dẫn thì phải đo lại trên mẫu lớn hơn và chỉ trên box đã lọc.

### 2.5 GATE 1 — kiểm bằng mắt và quyết định đã chốt (30/09/2026)

Người dùng mở 30 ảnh trong `runs/eda/overlay/` soi bằng mắt và trả lời hai câu:

1. **Box đỏ/xanh có khớp nhà không?** → **Có.** Xác nhận chiều mask → component → box
   là đúng, không bị lệch toạ độ. Đây là phép kiểm bù cho việc `audit.py` chỉ đọc mask
   chứ không đọc lại ảnh.
2. **Box viền TÍM là một căn nhà to hay nhiều căn dính nhau?** → **Một căn nhà to.**
   Đây là bằng chứng **trực tiếp** phủ định giả thuyết "nhà bị gộp" — mạnh hơn chỉ số
   thống kê, vì chỉ số chỉ nói "box to gấp 3 lần trung vị" chứ không nói được bên trong
   box có mấy căn.

**Quyết định của GATE 1 — đã ghi vào `configs/data.yaml`, mục `decisions`:**

| Tham số | Chốt | Ngưỡng tự động nói | Lý do |
|---|---|---|---|
| `target_long_side` | **1536** | — | Trung vị nhà 148–166px, gấp ~5 lần mốc 32px; tăng/giảm đều không đáng so với VRAM trên T4 |
| `min_area` | **64** | — | Bằng `min_side_px²` để "nhà hợp lệ" chỉ có một định nghĩa |
| `cat_tile` | **false (KHÔNG)** | CÓ | Sát ngưỡng đúng 4px và chỉ do một lớp; 28px = 3,5 ô ở stride-8 vẫn phát hiện được; cắt tile đắt gấp 4 lần mà dữ liệu vốn đã thưa |
| `tach_nha_dinh` | **false (KHÔNG)** | CÓ | Người dùng đã soi 30 ảnh: box tím là một căn nhà to. Chỉ số lệch 75 lần giữa hai lớp cũng cho thấy nó đo phân bố kích thước, không đo chuyện gộp |

Hai chỗ lệch ngưỡng là **có chủ ý** và báo cáo EDA in rõ ("Khác với ngưỡng tự động — có
chủ ý, không phải lỗi"), để người đọc không tưởng có chỗ nào bị bỏ quên. Cả hai đều được
quyết **trước khi train**, không phải sau khi xem mAP — giữ đúng tinh thần "không chỉnh
tham số sau khi thấy kết quả" của plan.

**Đổi lại, Phase 6 phải trả nợ:** một thí nghiệm so **tiled / không-tiled** trên val. Nếu
phân tích lỗi cho thấy nhà nhỏ là nguồn lỗi chính thì đó chính là căn cứ để bật lại
`cat_tile: true` — và khi đó đồ án có số liệu để bảo vệ, thay vì chỉ có một quy tắc.

---

## 3. Việc tiếp theo

1. ~~Chạy `notebooks/00_colab_setup.ipynb` trên Colab (GPU T4) → chốt GATE 0.~~
   **Xong 30/09/2026** — xem §1.5 và §1.6.
2. Repo đã đẩy lên GitHub: **https://github.com/21xDHoang/flood-house-counting**
   (public, nhánh `main`). Colab clone repo này về `/content` ở đầu mỗi phiên.
3. **Phase 1 — XONG, GATE 1 ĐÃ CHỐT (30/09/2026).** Số liệu ở §2.4, quyết định và
   phép kiểm bằng mắt ở §2.5. Đã đẩy lên GitHub:
   - `configs/data.yaml` — toàn bộ tham số (đường dẫn, ngưỡng quyết định, tham số EDA)
     **+ khối "CHỐT CỦA GATE 1"** ghi 4 lý do cho mỗi quyết định
   - `src/floodcount/data/audit.py` — khảo sát mask thật, trả lời checklist §2.2
   - `scripts/audit.py` — CLI mỏng bọc quanh module trên
   - `tests/test_audit.py` — 103 assertion trên zip giả có cả bẫy ColorMasks
   - `notebooks/01_data_prep.ipynb` — notebook chạy trên Colab (không cần GPU)

   Chạy lại [1.6] bây giờ rất nhanh (~2 phút) vì nó đọc lại `audit.jsonl` cũ thay vì
   xử lý lại 2343 ảnh — dùng mỗi khi chỉ cần đổi phần báo cáo.

4. **Phase 2 (tiếp theo)** — chuyển mask → COCO cho hai lớp nhà, dùng đúng
   `target_long_side: 1536`, `min_area: 64`, không cắt tile, không watershed.

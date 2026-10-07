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

### 1.9 Cài đặt chuyển vào `scripts/cai_moi_truong.py` — một tab Colab là đủ (01/10/2026)

**Ràng buộc mới:** người dùng chỉ mở được **một tab Colab** tại một thời điểm. Colab
tính một notebook = một tab = **một runtime**, và bản miễn phí chỉ cho một GPU một
lúc — mở notebook 00 ở tab thứ hai là dựng runtime mới và **phiên GPU đang chạy ở tab
kia bị giành mất**. Cách cũ ("chạy ô `[0.6]` của notebook 00 trước, rồi quay lại
notebook 03") vì thế không thực hiện được.

**Đã sửa:** toàn bộ logic cài đặt + ba miếng vá chuyển từ ô `[0.6]` của notebook 00
vào `scripts/cai_moi_truong.py`. Notebook 00 ô `[0.6]` và notebook 03 ô `[3.3b]` cùng
gọi script này, nên **mỗi notebook tự cài được trong chính tab của nó**.

- Phiên bản thư viện giờ nằm ở **một chỗ duy nhất** trong script — trước đây khai ở
  notebook 00 ô `[0.2]`, notebook 03 không nhìn thấy được.
- Script tự dò tổ hợp Python/torch/CUDA của runtime rồi tìm wheel `mmcv` khớp, nhận
  `--drive-dir` để dùng cache wheel trên Drive, và ghi dấu `/content/.floodcount_env_ready`
  để lần chạy sau **bỏ qua phần cài** (chỉ chạy lại phần vá).
- Ô gọi script **bắt buộc** tự đọc ống dẫn rồi `print` (bẫy §1.8) — nếu không thì mất
  3–5 phút nhìn ô output trống.
- Sau khi script chạy xong, ô gọi **phải xoá `mmpretrain` khỏi `sys.modules`**: nếu ô
  `[3.4]` đã chạy từ trước (nó nổ vì thiếu gói), gói `mmpretrain` vẫn nằm trong cache
  với giá trị cờ `WITH_MULTIMODAL` **cũ**, và miếng vá coi như không có tác dụng.

`tests/test_cai_moi_truong.py` (59 phép kiểm) khoá lại: cách dò tên wheel — kể cả bẫy
`%2B` trong `href` của trang index và bẫy tiền tố (`pt2.1.1` khớp nhầm trong
`pt2.1.10`) — ba miếng vá đúng chỗ và **không vá nhầm file**, mã thoát của
`--xem-phien-ban`, và ràng buộc **không import `torch` ở cấp module** (script còn dùng
để in phiên bản khi chưa cài gì).

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

### 2.6 Phase 2 — mask → COCO: bốn quyết định và hai lỗi tự bắt được (30/09/2026)

**Bốn quyết định kỹ thuật, ghi lại để trả lời khi bảo vệ:**

| Quyết định | Lý do |
|---|---|
| **Giữ nguyên 10 lớp** trong mask đã resize (không chỉ 2 lớp nhà) | Mask PNG nén rất tốt (vùng màu phẳng) nên tốn thêm không đáng kể, mà đổi lại vẫn làm lại được phép kiểm chứng "nhà ngập có chạm nước không" ở Phase 6. Bỏ đi thì phải giải nén lại 13 GB |
| `area` = **số pixel thật** của component, không phải diện tích đa giác | Đây mới là con số đã dùng để lọc nhiễu, và đúng về hình học cả khi hình có lỗ. COCO chỉ dùng `area` để phân loại small/medium/large nên không ảnh hưởng lúc train |
| **Box sát mép ảnh vẫn giữ** | Nhà bị cắt ở rìa vẫn là nhà cần đếm; bỏ đi thì số nhà của đồ án ít hơn số nhà thật. Cái giá là box cụt một cạnh, nên số box sát mép được **đếm và in ra** trong báo cáo để sau này phân tích lỗi còn có số liệu mà bàn |
| `id` ảnh COCO đánh theo **tên file đã sắp xếp**, không theo thứ tự xử lý | Chạy lại giữa chừng (resume) hay đổi thứ tự xử lý đều cho ra file giống hệt nhau — Phase 5 phải tái lập được |

**Bộ lọc khác Phase 1 một chỗ, và đó là chỗ dễ tưởng nhầm là bug.** Cột `so_box_loc`
trong `EDA_REPORT.md` chỉ lọc theo **cạnh**; Phase 2 lọc theo **cả cạnh VÀ diện tích**
(`min(bw,bh) >= 8` và `area >= 64`). Hai bộ lọc chỉ khác nhau ở những hình thoi kiểu
đường chéo 16×16px (bbox 16×16 nhưng chỉ có 16 pixel thật). Vì vậy `doi_chieu_voi_phase1()`
**tính lại** từ `audit.jsonl` bằng đúng bộ lọc của Phase 2 rồi mới so — so thẳng với cột
`so_box_loc` là so sai và sẽ tưởng nhầm là Phase 2 có bug. Test khoá lại cả hai hình nhiễu
này: một vệt `100×1` (lọt ngưỡng diện tích, bị ngưỡng cạnh chặn) và một đường chéo `16×16`
(lọt ngưỡng cạnh, bị ngưỡng diện tích chặn).

**Hai lỗi tự bắt được, cùng một họ — "thất bại im lặng":**

1. **Đọc `.jsonl` bằng sai khoá.** Phase 1 ghi khoá `anh`, Phase 2 ghi khoá `file_name`.
   `doc_ket_qua_da_co()` cũ bắt `(json.JSONDecodeError, KeyError)` rồi `continue`, nên dùng
   nhầm khoá thì **mọi** bản ghi đều ném `KeyError`, hàm trả về rỗng, và chế độ resume tắt
   ngóm — chạy lại từ đầu mà không có lấy một dòng cảnh báo. Tệ hơn: dòng ráp COCO
   (`ban_ghi = list(doc_ket_qua_da_co(duong_jsonl).values())`) đã **thực sự** thiếu
   `khoa="file_name"`, nghĩa là Phase 2 sẽ ghi ra dataset **0 ảnh** rồi in "XONG", mã thoát 0.
   Sửa: tách `KeyError` ra khỏi `except`, ném lỗi kèm thông báo chỉ rõ hai khoá của hai phase;
   và thêm chốt chặn — không có bản ghi nào thì thoát mã 1, in lý do từng cặp bị bỏ.
2. **`min_area` chỉ lọc được một nửa số ca nhiễu.** Lọc chỉ theo diện tích thì vệt rác
   `1×500px` lọt qua (diện tích 500 > 64); lọc chỉ theo cạnh thì đường chéo `8×8px` lọt qua
   (bbox 8×8 nhưng chỉ 8 pixel thật). Phải áp dụng **cả hai** thì "một căn nhà hợp lệ" mới chỉ
   có một định nghĩa trong toàn pipeline — đây là lý do `min_area = min_side_px²` trong config.

**`notebooks/02_build_coco.ipynb` không cần GPU và không cần MMDetection.** Dataset dựng trên
`/content` rồi mới nén thành **một tệp zip** đẩy lên Drive: 4686 file ảnh/mask ghi thẳng lên
Drive qua FUSE chậm hơn nhiều lần, mà Phase 3 chỉ cần giải nén một tệp. Nén bằng `ZIP_STORED`
(không nén lại): JPEG và PNG vốn đã nén sẵn, thử nén thêm chỉ thu ~0% mà tốn vài phút CPU.

**Test:** `tests/test_mask_to_coco.py`, 106 assertion trên zip giả — khoá lại tiền tố split
trong tên file (cả ba split đều đặt tên `1.jpg`, thiếu tiền tố là ghi đè nhau mất ảnh), hai
loại nhiễu, hai nhà kề nhau ra đúng 1 box, counts CSV khớp COCO JSON, phép đối chiếu Phase 1,
resume không nhân đôi, ảnh/mask lệch kích thước, và `tao_categories` từ chối id không liên tục.

### 2.7 Phase 2 chạy thật trên Colab — GATE 2 ĐẠT (30/09/2026)

Chạy đủ 2.343 ảnh, mã thoát 0. Báo cáo ở `MyDrive/Flood_House_AI/runs/build/BUILD_REPORT.md`.

| Split | Số ảnh | Box nhà ngập | Box nhà không ngập | Tổng box | Ảnh có nhà ngập |
|---|---|---|---|---|---|
| test | 448 | 604 | 651 | 1.255 | 47 |
| train | 1.445 | 1.841 | 1.938 | 3.779 | 149 |
| val | 450 | 643 | 624 | 1.267 | 49 |
| **Tổng** | **2.343** | **3.088** | **3.213** | **6.301** | **245** |

**Phép kiểm quan trọng nhất — ĐẠT.** Đối chiếu với Phase 1 ra **+0 ở cả hai lớp**
(3.088 và 3.213). Nghĩa là hai phase dùng chung đúng một định nghĩa "nhà hợp lệ", và
mọi so sánh số liệu giữa hai phase từ đây về sau mới có nghĩa.

Ba con số khác khớp với Phase 1, không phải chỉnh gì:

- **2.343 ảnh**, đúng 1.445/450/448 — bằng số cặp ảnh/mask Phase 1 ghép được.
- **245 ảnh có nhà ngập** — Phase 1 đếm bằng bộ lọc **cạnh**, Phase 2 bằng bộ lọc
  **cạnh + diện tích**, mà vẫn ra đúng 245. Tức bộ lọc diện tích không làm mất căn
  nhà cuối cùng của ảnh nào.
- **`pycocotools` đọc được cả 3 file** (448/1.255, 1.445/3.779, 450/1.267 annotation)
  và kiểm tra cấu trúc COCO ĐẠT: mọi `image_id` tồn tại, mọi `bbox` nằm trong ảnh và
  có kích thước dương, `id` không trùng.

**41 box bị loại thêm so với Phase 1 — bằng chứng bộ lọc hai điều kiện có tác dụng.**
Phase 1 (chỉ lọc cạnh) đếm 3.101 + 3.241 = **6.342** box; Phase 2 (lọc cả cạnh và diện
tích) còn 3.088 + 3.213 = **6.301** box. Chênh 13 + 28 = 41 box (0,65%) — đúng là các
hình thoi/đường chéo có bbox đủ to nhưng số pixel thật quá ít, tức đúng loại nhiễu mà
bộ lọc diện tích sinh ra để chặn. **Nếu chênh bằng 0 thì mới đáng lo**, vì như vậy bộ
lọc diện tích chẳng chặn được gì và rất có thể đang bị vô hiệu.

#### Phát hiện quan trọng: nhà RẤT TO, và gần một nửa số box bị cắt ở mép ảnh

Người dùng soi 90 ảnh overlay xác nhận box đỏ/xanh **khớp nhà** và **không có box nào
ôm nhiều căn nhà**. Nhưng câu hỏi "box bị cắt ở rìa có nhiều không" thì mắt trả lời
"không nhiều", trong khi báo cáo ghi **2.953/6.301 box = 46,9% chạm mép**. Đã phân xử
bằng số liệu, và **46,9% là ĐÚNG**:

- Đo lại trên `build.jsonl`: 803 chạm mép trái, 969 mép trên, 793 mép phải, 831 mép
  dưới. **Bốn cạnh gần bằng nhau** — nếu `w2`/`h2` bị hoán vị (lỗi hay gặp nhất ở loại
  kiểm tra này) thì một cạnh sẽ vọt lên còn một cạnh về 0. Phân bố đều là dấu hiệu của
  hiện tượng hình học thật, không phải lỗi đếm.
- Mô hình hình học: nhà phân bố trên mặt đất, ảnh là một cửa sổ cắt ra, nên nhà bị cắt
  ở mép **được đếm nhiều hơn** tỉ lệ diện tích. Với cửa sổ dài `L` và bề rộng nhà `b`,
  xác suất một nhà chạm một mép là `2b/(L+b)`: ngang `2×203,8/(1536+203,8) = 0,234`;
  dọc `2×193,7/(1152+193,7) = 0,288`; gộp `1 − 0,766×0,712 = 0,455`.
  **Mô hình đoán 45,5%, thực đo 46,9%.**
- Vì sao mắt lại nói "không nhiều": box bị cắt ở rìa **trông không có gì sai** — nó chỉ
  là một box nằm sát khung. Mắt bắt cái bất thường, mà cái này thì bình thường.

**Kích thước box thật** (đo trên `build.jsonl`, cả 6.301 box):

| | trung vị | trung bình | p90 |
|---|---|---|---|
| chiều rộng `bw` | 186 | 203,8 | 348 |
| chiều cao `bh` | 180 | 193,7 | 336 |

Hai hệ quả phải nhớ:

1. **Phase 3 — dải anchor của RPN.** Nhà có cạnh trung vị **186px**, lớn hơn nhiều so
   với mức mà bộ anchor mặc định của COCO nhắm tới. Phải kiểm dải anchor phủ tới
   ~350–500px, đừng để mặc định rồi ngồi đoán vì sao recall thấp.
2. **Phase 6 — cắt tile có thêm một lập luận độc lập.** GATE 1 hoãn quyết định tiling
   vì `p5 = 28px` chỉ sát dưới ngưỡng 32px. Con số 46,9% này là lập luận thứ hai và
   mạnh hơn: **gần một nửa số box bị cụt một cạnh**, mà box cụt thì khó học hơn box
   nguyên. Một nửa số box là quá nhiều để đổ cho nhiễu.

Ảnh sau resize có **hai cỡ**: `1536×1152` (1.991 ảnh) và `1536×1028` (352 ảnh) — cùng
chiều ngang 1536, chiều dọc lệch 11%. Đây chính là lý do phải bật augmentation đổi tỉ
lệ ở Phase 3 (đã ghi ở §2.4).

#### Bài học vận hành: ngắt giữa chừng và tiến trình mồ côi

Lần chạy đầu bị ngắt ở ảnh 600/2.343 (`KeyboardInterrupt`). **Không mất gì**: mỗi ảnh
ghi xong là `flush()` ngay một dòng vào `build.jsonl`, và lúc khởi động lại bản ghi chỉ
được tin khi **ảnh còn thật trên đĩa** — nên chạy lại chỉ bỏ qua 640 ảnh đã xong (đọc
lại đúng 640 bản ghi) rồi làm tiếp từ 641.

Nhưng có một cái bẫy: khi ô notebook bị ngắt, **tiến trình con không chết theo**. Nó
vẫn chạy và ghi vào ống dẫn mà không ai đọc → ống đầy (64 KB) → nó kẹt luôn. Chạy lại ô
đó ngay lúc ấy là **hai tiến trình cùng ghi vào một thư mục**. Trước khi chạy lại phải
dọn: `pkill -f build_coco.py` rồi kiểm lại bằng `ps`.

#### Tệp dataset đã đóng gói

`MyDrive/Flood_House_AI/processed/floodnet_coco.zip` — **4.692 tệp, 1,86 GB** (2.343 ảnh
+ 2.343 mask + 3 JSON annotation + 3 CSV counts), nén `ZIP_STORED` (JPEG/PNG vốn đã nén
sẵn). Từ Phase 3 chỉ cần giải nén tệp này vào `/content`, **không cần tới
`floodnet_raw.zip` 13 GB** nữa.

### 2.8 Dựng lại Phase 1 + 2 trên máy cá nhân (CPU) — đối chiếu GATE 2 (01/10/2026)

Người dùng tải `floodnet_raw.zip` (13 GB) về máy và đưa đường dẫn. Hoá ra **cả hai phase
chạy được trên CPU, không cần Colab**: `audit.py` và `build_coco.py` đọc thẳng trong zip
chứ không giải nén. Chạy đủ 2.343 ảnh, mã thoát 0.

| Số | GATE 2 (Colab, 30/09) | Dựng lại trên máy (01/10) |
|---|---|---|
| Ảnh mỗi split | 448 / 1.445 / 450 | 448 / 1.445 / 450 |
| Box nhà ngập | 3.088 | 3.088 |
| Box nhà không ngập | 3.213 | 3.213 |
| Tổng box | 6.301 | 6.301 |
| Ảnh có nhà ngập | 245 | 245 |
| Box chạm mép | 2.953 | 2.953 |
| Phase 1 tính lại (chỉ lọc cạnh) | 3.101 + 3.241 = 6.342 | 6.342 |
| Đối chiếu Phase 1 | +0 / +0 | +0 / +0 |

**Khớp từng con số.** Đây là phép kiểm tái lập độc lập: cùng input và cùng code nhưng
khác máy, khác phiên chạy — kết quả không đổi. Hệ quả vận hành: từ giờ Phase 1/2 **không
cần chiếm GPU Colab** nữa, chạy trên máy rồi chỉ đưa gói kết quả lên Drive.

Gói dựng lại: `floodnet_coco.zip` — **4.692 mục, 1,86 GB**, giống hệt bản Colab
(2.343 ảnh + 2.343 mask + 3 JSON + 3 CSV, `ZIP_STORED`; không tên tệp nào chứa `\`, tức
không dính bẫy đường dẫn Windows). Đã kiểm **ruột gói** chứ không chỉ đếm tệp: mọi
`file_name` trong JSON đều có tệp ảnh trong zip, `image_id` không mồ côi, `bbox` dương,
category đúng thứ tự (`flooded_building` trước).

*Một chỗ dễ kiểm nhầm:* `file_name` trong JSON để **trần** (`train_10165.jpg`), mmdet ghép
với `data_prefix.img = images/<split>/`. Đem so thẳng `file_name` với danh sách tệp trong
zip thì **cả 2.343 ảnh đều báo "thiếu"** — lỗi của phép kiểm, không phải của dữ liệu.

---

## 3. Phase 3 — cấu hình model & phép thử trước khi train (30/09/2026)

Trạng thái: **code đã viết xong, test xanh trên máy CPU; CHƯA chạy gì trên Colab**
— chưa train, chưa chạy phép thử overfit. Mọi con số ở mục này là số **đọc được
từ source** hoặc **đo trên máy**, không phải kết quả train.

### 3.1 Đã viết những gì

| Tệp | Vai trò |
|---|---|
| `configs/mmdet/cascade_convnext_t_floodnet.py` | Config chính — kế thừa `mmdet::convnext/cascade-mask-rcnn_...` rồi **bỏ toàn bộ nhánh mask**, còn 2 lớp |
| `configs/mmdet/overfit20.py` | Config con cho phép thử học vẹt 20 ảnh |
| `src/floodcount/data/kiem_tra.py` | Chốt chặn dữ liệu: đối chiếu file JSON với config train **trước khi** train |
| `src/floodcount/data/overfit.py` | Dựng `instances_overfit20.json` (20 ảnh chọn theo seed) |
| `src/floodcount/data/photometric.py` | Tăng sáng/tương phản nhẹ, thuần numpy |
| `src/floodcount/models/transforms.py` | Lớp `TangSangNhe` — "keo" nối hàm trên vào pipeline mmdet |
| `scripts/kiem_tra_du_lieu.py`, `kiem_anchor.py`, `tao_overfit20.py`, `train.py` | CLI mỏng |
| `notebooks/03_train.ipynb` | Notebook chạy trên Colab |
| `tests/test_train.py`, `test_kiem_tra.py`, `test_anchor.py`, `test_overfit.py`, `test_photometric.py` | 5 bộ test mới, tất cả chạy được trên CPU |

### 3.2 Năm điều phải tra source mới biết (không suy đoán)

**(a) `custom_imports` do `Config.fromfile` chạy, KHÔNG phải `Runner`.**
`mmengine/config/config.py` (0.10.7) có tham số `import_custom_modules=True` và
thân hàm gọi `import_modules_from_strings(**cfg_dict['custom_imports'])`; `Runner`
không đụng tới khoá này. Hệ quả thực tế: `src/` phải nằm trong `sys.path`
**TRƯỚC dòng `Config.fromfile(...)`**, nếu không thì config đồ án (có khai
`floodcount.models.transforms`) chết ngay lúc nạp với `Failed to import`.
`scripts/train.py` vì thế chèn `sys.path` ở đầu tệp, trước mọi import khác.

**(b) Cú pháp `mmdet::<đường dẫn>` CHỈ dùng được bên trong `_base_`.**
`_file2dict` bắt đầu bằng `filename = osp.abspath(osp.expanduser(filename))` rồi
`check_file_exist(filename)`, còn nhánh xử lý `'::'` nằm **trong vòng lặp `_base_`**.
Nên `Config.fromfile('mmdet::convnext/...')` **không** chạy được, nhưng
`_base_ = ['mmdet::convnext/...']` trong config đồ án thì chạy. Đây chính là lý do
config đồ án **không cần clone repo mmdet**, trong khi ô smoke test của notebook 00
(cần mở trực tiếp một tệp cấp cao nhất) thì vẫn cần.

**(c) `resume=True` an toàn cả khi chưa có checkpoint nào.**
Config `mmdet::convnext/...` **không** đặt `load_from` — trọng số ImageNet đến từ
`init_cfg=dict(type='Pretrained', checkpoint='...convnext-tiny...in1k....pth',
prefix='backbone.')` gắn trên chính backbone. `Runner.load_or_resume` xử lý đúng
trường hợp này: `resume=True` + `load_from is None` → `find_latest_checkpoint()`;
không tìm thấy thì không nạp gì và **không sập**. Nhờ vậy, chạy lại notebook 03
sau khi Colab ngắt phiên sẽ tự tiếp tục từ checkpoint mới nhất mà không phải đổi
tham số nào.

**(d) `RandomChoiceResize` bắt buộc phải ghi `keep_ratio=True`.**
Đây là lớp của **mmcv** (không phải mmdet) và nó tự dựng đối tượng `Resize` qua
registry gốc — mà `Resize` của mmcv có `keep_ratio` mặc định **False**, ngược với
`Resize` của mmdet. Bỏ tham số này thì ảnh bị bóp méo tỉ lệ mà **không có lỗi nào
báo**, vì box vẫn được scale theo tỉ lệ của từng trục. `scripts/train.py` in ra
`keep_ratio` của mọi transform có "esize" trong tên và cảnh báo nếu khác `True`.

**(e) Bỏ ảnh rỗng là mặc định của mmdet — đồ án cố ý tắt.**
`filter_cfg=dict(filter_empty_gt=False, min_size=32)`. Khoảng một nửa FloodNet
không có căn nhà nào; bỏ các ảnh đó thì model chỉ học "ảnh nào cũng có nhà",
trong khi đầu ra của đồ án là **đếm trên mọi ảnh**, kể cả ảnh không có nhà.
`scripts/train.py` cảnh báo nếu giá trị này khác `False`.

### 3.3 Dải anchor của RPN — đo, không đoán

Config đồ án thừa hưởng RPN từ bản gốc nên mở file `.py` của đồ án sẽ **không
thấy** `scales`/`ratios`. Đọc từ config gốc: `scales=[8]`, `ratios=[0.5, 1, 2]`,
`strides=[4, 8, 16, 32, 64]`, **không** có `base_sizes`; `AnchorGenerator.__init__`
đặt `base_sizes = [min(stride) for stride in strides]` → 4/8/16/32/64. Nhân với
`scale = 8`, rồi với `1/√ratio` và `√ratio`:

| Mức | Lưới | Anchor dẹt nhất | Anchor cao nhất |
|---|---|---|---|
| 1 | stride 4 | 45,3 × 22,6 | 22,6 × 45,3 |
| 5 | stride 64 | 724,1 × 362,0 | 362,0 × 724,1 |

Tất cả **15 hình dạng**. Cạnh box thật của FloodNet: **trung vị 186px, p90
348×336** (§2.7). Dải anchor phủ từ 22,6px tới 724,1px — **thừa sức** chứa cả
những căn nhà to nhất, nên **không cần sửa anchor**. `scripts/kiem_anchor.py` đo
lại điều này trên box thật và in bảng phủ theo từng mức lưới; chạy nó trước khi
train để báo cáo có số liệu thay vì lập luận suông.

**Đã đo thật — 3.779 box train, trên Colab 01/10/2026, mã thoát 0:**

| Số đo | Giá trị |
|---|---|
| Cạnh trung vị của box | 164px (p95 cạnh lớn hơn: 493px) |
| IoU lớn nhất đạt được | nhỏ nhất 0,105 · p5 0,510 · **trung vị 0,688** · p95 0,886 |
| Box đạt IoU ≥ 0,3 | **99,3%** |
| Box đạt IoU ≥ 0,5 | **97,6%** |
| Box đạt IoU ≥ 0,7 | 47,6% |
| Theo lớp (đạt 0,5) | flooded 98,8% · non_flooded 96,4% |

Anchor 256×256 (stride 32) là anchor tốt nhất cho 1.150 box; ba anchor lớn nhất
(724×362, 512×512, 362×724) phủ 72 + 191 + 34 = **297 box to**. Tức đuôi nhà to —
`kiem_tra_du_lieu` báo cạnh nhỏ **lớn nhất 951px** (§3.5) — **vẫn có anchor khớp**.
Kết luận "không cần nới dải anchor" giờ có số liệu chống lưng, không còn là lập luận
từ p90.

Một chi tiết đáng ghi: bảng "IoU lớn nhất theo lưới anchor thật" **trùng khít** bảng
"giới hạn trên khi đặt anchor cùng tâm box" (0,688 · 99,3% · 97,6% · 47,6%) — lưới
anchor dày đủ để chuyện lệch tâm không làm mất IoU. Giới hạn nằm ở **hình dạng
anchor**, không ở mật độ lưới.

**2,4% box (≈91 box) không đạt IoU 0,5** — ghi lại làm đầu vào cho phân tích lỗi ở
Phase 6 (`docs/RESULTS.md`), đừng để nó thành câu hỏi mở khi đã train xong.

### 3.4 Checkpoint ghi ra Drive — và cách đổi lại

`default_hooks.checkpoint` đặt `max_keep_ckpts=2` + `save_last=True` +
`save_best='coco/bbox_mAP'`, và **`save_optimizer=False`**. Lý do: model ~54
triệu tham số nên checkpoint chỉ trọng số ≈ **216 MB**, thêm trạng thái AdamW
(`m` và `v`) là ≈ **432 MB** nữa; ghi thêm chừng đó mỗi epoch lên Drive qua FUSE
là quá đắt cho một lợi ích duy nhất là resume mượt hơn. `Runner.resume()` có
guard `if 'optimizer' in checkpoint`, nên resume từ checkpoint thiếu optimizer
**không sập** — chỉ là các moment của AdamW khởi động lại, ảnh hưởng vài chục
vòng lặp đầu.

Muốn giữ nhiều checkpoint hơn thì sửa `max_keep_ckpts` ở mục 8 của
`configs/mmdet/cascade_convnext_t_floodnet.py`; muốn resume mượt hơn nữa thì đổi
`save_optimizer=True` (chỉ nên làm nếu chạy dài ngày).

### 3.5 Test đã chạy trên máy (CPU, không cần GPU và không cần dataset thật)

| Bộ test | Số phép kiểm |
|---|---|
| `test_mask_to_coco.py` | 106 |
| `test_audit.py` | 103 |
| `test_cai_moi_truong.py` | 59 |
| `test_train.py` | **89** (70 trước §3.14, 63 trước §3.13 vòng 2, 58 trước §3.13 vòng 1, 49 trước §3.12) |
| `test_kiem_tra.py` | 41 |
| `test_anchor.py` | 39 |
| `test_overfit.py` | 37 |
| `test_photometric.py` | 21 |
| **Tổng** | **495** |

Cả 8 bộ **PASS** ngày 01/10/2026, và chạy lại **PASS** ngày 07/10/2026 năm lần:
sau khi thêm mục 5b của §3.12, sau khi gia hạn `[3.10b]` lên 120 epoch, sau
khi vá resume vòng 1 của §3.13 (mục 1c — 469 phép kiểm), sau khi vá trọn
danh sách cho phép vòng 2 (§3.13 — 476 phép kiểm), và sau khi vá lỗi LR sau
resume (§3.14 — 495 phép kiểm). `test_train.py` chạy được cả trên máy sạch
chưa cài MMDetection vì `scripts/train.py` chỉ import mmdet/mmengine **bên trong
hàm** — đây là ràng buộc thiết kế có chủ ý, và có một phép kiểm khoá đúng điều đó
(nếu ai đó chuyển các import lên đầu tệp, test sẽ đỏ ngay).

Ba bộ in tiếng Việt (`test_cai_moi_truong.py`, `test_kiem_tra.py`, `test_overfit.py`)
trước đây nổ `UnicodeEncodeError` trên console Windows — chúng thiếu đoạn
`sys.stdout.reconfigure(encoding="utf-8")` mà §1.7 đã chốt là bắt buộc. Đã chép vào
`main()` của cả ba.

### 3.6 Một lỗi tìm được TRƯỚC khi chạy Colab — và một test suýt bỏ lọt nó (01/10/2026)

Rà lại Phase 3 trước khi đẩy lên Colab, đối chiếu `configs/mmdet/overfit20.py` với
source mmdet 3.3.0 thì thấy: **`val_dataloader` của phép thử overfit trỏ nhầm thư
mục ảnh.**

**Cơ chế.** `overfit20.py` chỉ khai lại `ann_file`; mmengine gộp dict theo chiều
sâu, nên mọi khoá không khai lại đều **thừa hưởng từ config cha**. Với
`train_dataloader` thì thừa hưởng `data_prefix=images/train/` — tình cờ đúng, vì
20 ảnh overfit lấy từ split train. Nhưng `val_dataloader` thừa hưởng
`data_prefix=images/val/`, mà annotation thì trỏ vào chính 20 ảnh của train.

**Vì sao không có gì báo lỗi.** Đọc `CocoDataset.parse_data_info` (mmdet 3.3.0):

```python
img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
```

`file_name` do Phase 2 ghi ra là tên **phẳng có tiền tố split** (`train_10168.jpg`,
xem §2.6), không kèm thư mục — nên thư mục ảnh do `data_prefix['img']` quyết định
hoàn toàn. Và mmdet **không kiểm tra ảnh có tồn tại lúc dựng dataset**: đã đọc cả
`coco.py` lẫn `base_det_dataset.py`, không có `check_file_exist`/`osp.exists` nào
trên đường dẫn ảnh. Hệ quả: tiền kiểm vẫn in **ĐẠT HẾT**, train vẫn chạy hết
epoch 1, và chỉ tới lượt validate đầu tiên mới nổ `FileNotFoundError` — tức là
lỗi chỉ lộ ra sau khi đã trả tiền GPU, đúng loại lỗi mà cả notebook 03 sinh ra để
chặn.

**Đã sửa:** khai `data_prefix=dict(img='images/train/')` **tường minh** ở cả hai
dataloader của `overfit20.py`, kèm chú thích tại chỗ.

**Bài học đáng giá hơn cả lỗi: bản test đầu tiên suýt bỏ lọt nó.** Test mới đọc
config bằng `ast` (không cần mmengine) để đòi mọi dataset trỏ tới annotation
overfit phải tự khai `data_prefix`. Nhưng bản đầu của hàm đọc AST chỉ hiểu literal
`{...}`, mà config mmengine viết `dict(...)` — nên nó trả `None` cho **mọi**
dataloader, và vòng lặp `if ds is None: continue` biến tất cả thành "đạt". Test
xanh trong khi không kiểm gì cả.

Phát hiện được là nhờ **chạy test trên chính bản config còn lỗi trước khi sửa**
(phép thử âm): nó phải đỏ ở đúng hai mục `data_prefix`, mà lại đỏ ở ba mục khác
(config chính) và **im lặng** ở hai mục cần đỏ nhất. Sửa hai chỗ: hàm đọc AST hiểu
thêm `dict(...)`, và tách hẳn "tệp không khai biến" (`None`) khỏi "có khai nhưng
đọc không ra" (`_BieuThuc`) để không thể "đạt" bằng cách không đọc được gì.

Nhân tiện đối chiếu luôn cú pháp `{{_base_.ann_overfit}}` với source mmengine
0.10.7: hàm `_pre_substitute_base_vars` dùng đúng regex
`r'\{\{\s*_base_\.([\w\.]+)\s*\}\}'`, và việc thay thế chỉ xảy ra khi **toàn bộ**
chuỗi khớp — xác nhận chú thích trong `overfit20.py` là đúng, không phải suy đoán.

> **Sửa lại (01/10/2026, xem §3.11):** kết luận ngay trên là SAI. Phép đối chiếu
> mới chỉ kiểm regex và điều kiện khớp toàn chuỗi, chưa kiểm một chi tiết:
> placeholder mmengine sinh ra **đã bọc sẵn ngoặc kép**. Với giá trị chuỗi, cú
> pháp đó hỏng **im lặng** — và lần chạy `[3.10]` đã chết vì nó.

**Chốt chặn thêm, để lỗi này không tái phát ở config khác.** Sửa đúng một chỗ
`data_prefix` thì chỉ chặn được đúng ca đã gặp. Nên `scripts/train.py` có thêm
**mục [2b]** trong phần tiền kiểm: mở từng annotation, ghép
`data_prefix.img` + `data_root` với **từng** `file_name` rồi kiểm file có thật
trên đĩa, in ra `2.343/2.343 ảnh có thật trong ...`. Đây chính là phép kiểm mà
mmdet không làm, nên nó bắt được **mọi** config trỏ nhầm thư mục ảnh, không riêng
gì overfit20. Chạy hết ba split tốn vài giây trên đĩa cục bộ.

Kèm theo, `scripts/train.py` còn **từ chối** dataset không khai `data_prefix.img`
thay vì bỏ qua im lặng: không khai thì không có cách nào biết ảnh nằm ở đâu, và
đó cũng là thứ mmdet cần để dựng đường dẫn.

### 3.7 Phần còn lại của Phase 3

GATE 3 đang chạy trên Colab (01/10/2026). Đã báo cáo về: `[3.6]`
`kiem_tra_du_lieu` (ĐẠT HẾT), `[3.7]` `kiem_anchor` (ĐẠT — số đo ở §3.3),
`[3.8]` dựng tập overfit (20 ảnh / 125 box, seed 42, script tự đọc lại từ đĩa
xác nhận khớp). `[3.9]` `train.py --dry-run`: lần đầu hỏng (§3.8), lần hai xong
— số đo ở §3.9. `[3.10]` train overfit 20 ảnh: lần đầu **hỏng** — cú pháp
`{{_base_.ann_overfit}}` hoá thành đường dẫn rác mà không lỗi nào báo (§3.11);
đã sửa và khoá bằng test, chờ chạy lại. Train thật (Phase 4) chỉ bắt đầu sau khi
chốt số epoch từ phép đo `--dry-run` (ước lượng thô hiện có: 24 epoch ≈ 4,2 giờ
phần train).

### 3.8 Lỗi thật đầu tiên trên Colab: `optim_wrapper` chưa được build (01/10/2026)

Ô `[3.9]` (`train.py --dry-run`) chết ngay vòng lặp đầu tiên:

    AttributeError: 'ConfigDict' object has no attribute 'optim_context'

Tiền kiểm `[1]`–`[6]` ĐẠT HẾT và model dựng xong trong 3,2 giây — nên lỗi không
nằm ở config, dataset hay model. Đọc source mmengine 0.10.7 (tag `v0.10.7` trên
GitHub; máy cá nhân không cài mmengine) thì ra: `Runner.__init__` **chỉ gán**
`self.optim_wrapper = optim_wrapper`, còn việc build thật chỉ xảy ra bên trong
`Runner.train()`:

    self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)

Vòng lặp đo trong `chay_thu` tự viết, không đi qua `train()`, nên
`runner.optim_wrapper` vẫn là ConfigDict và `train_step` nổ ngay khi gọi
`optim_wrapper.optim_context(self)`.

**Sửa:** gọi đúng hàm mà `train()` gọi —
`runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)` — trước
vòng lặp. Hàm này đi qua `constructor: LearningRateDecayOptimizerConstructor` của
mmdet và bọc `AmpOptimWrapper`, nên phép đo vẫn trung thực với train thật.

**Khoá lại bằng test:** `tests/test_train.py` mục 1b (4 phép kiểm) dùng AST để
buộc `chay_thu` phải (a) gọi `build_optim_wrapper`, (b) gán kết quả lại vào
`runner.optim_wrapper`, (c) truyền keyword `optim_wrapper` cho mọi lời gọi
`train_step`. Đã kiểm âm: xoá đúng dòng build khỏi nguồn thì phép kiểm (a) và
(b) chuyển thành FAIL — test không phải loại "xanh cả khi bỏ lỗi ra".

Bài học giống §3.6: **chép đúng một dòng của vòng lặp thật là chưa đủ** — phải
chép cả những gì `Runner.train()` làm TRƯỚC vòng lặp. Bài học phụ: phép đo
`--dry-run` đáng giá đúng như thiết kế — nó chết trước khi tốn một epoch GPU nào.

### 3.9 Số đo `--dry-run` trên Colab T4 (01/10/2026)

Chạy 20 vòng thật; vòng 0 (2.199,8 ms) bị bỏ khi tính trung bình vì còn khởi động
worker và autotune cuDNN.

| Đại lượng | Số đo |
|---|---|
| Thời gian/vòng (TB 18 vòng) | **866,3 ms** (thấp nhất 695,2 — cao nhất 1.038,0) |
| Số vòng mỗi epoch | 723 (= 1.445 ảnh ÷ batch 2) |
| **1 epoch (phần train)** | **10,4 phút** |
| VRAM đỉnh (đã cấp phát) | 6,66 GB |
| VRAM đỉnh (đã giành chỗ) | **7,81 GB / 14,56 GB (54%)** |
| Loss 20 vòng đầu | không NaN; 3,82 → dao động 0,01–2,27 |

- **VRAM còn dư nhiều** (54%): không phải giảm `batch_size` hay siết `scales_train`
  vì lý do bộ nhớ.
- **Dao động loss giữa các vòng là bình thường ở đây**: mỗi vòng là một batch
  ngẫu nhiên khác nhau, và 812/1.445 ảnh train không có box (ô `[3.6]`) nên có
  batch chỉ đóng góp loss của RPN. Khi train thật, nhìn trung bình trượt window 50.
- Log `LearningRateDecayOptimizerConstructor` in đủ 8 nhóm tham số, thang LR
  `8,24e-6 → 1e-4` — xác nhận bản sửa §3.8 đi đúng đường build của `train()`.
- Script tự in ước lượng: 24 epoch ≈ **4,2 giờ** — chỉ phần train, chưa tính
  validate mỗi epoch (450 ảnh) và thời gian ghi checkpoint lên Drive.

**Hai cảnh báo cosmetic đã gặp (không ảnh hưởng số đo):**

1. `train.py` — `float()` trên tensor còn `requires_grad` → `UserWarning` một lần
   mỗi lần chạy. Đã sửa luôn trong lượt này: `.detach()` trước khi `float()`.
2. Ô `[3.4]` in `pycocotools : ?` — thư viện không có `__version__`; đọc bằng
   `importlib.metadata.version("pycocotools")` sẽ ra 2.0.11. Chưa sửa (dọn sau).

### 3.10 Colab ngắt giữa chừng thì resume khôi phục được gì (01/10/2026)

Đọc source mmengine 0.10.7 (tag `v0.10.7`) trước khi đốt 6 giờ GPU:

| Thứ | Có trong checkpoint? | Resume khôi phục? |
|---|---|---|
| Trọng số model | có | có |
| `epoch` / `iter` (từ `meta`) | có | có |
| Lịch LR (`param_schedulers`) | **có** — `save_param_scheduler` mặc định `True`, độc lập với `save_optimizer` | có |
| Trạng thái optimizer (AdamW moments) | **không** — config để `save_optimizer=False` | **bỏ qua im lặng** |

Ba chi tiết kiểm bằng cách đọc code:

- `Runner.resume()` chỉ nạp optimizer khi `'optimizer' in checkpoint`; **không có
  nhánh `else`** → checkpoint thiếu khoá đó thì không lỗi, không cảnh báo.
- `ParamSchedulerHook` gọi `scheduler.step()` **không tham số** — vị trí trong lịch
  nằm ở bộ đếm nội bộ `_global_step` của scheduler, KHÔNG đọc `runner.epoch`. Nếu
  `param_schedulers` không được lưu thì mốc giảm LR [16, 22] sẽ lệch. Ở config này
  nó được lưu (mặc định), nên lịch LR resume ĐÚNG.
- Checkpoint "best" (`best_coco_bbox_mAP_*`) **luôn** bị ép `save_optimizer=False`
  và `save_param_scheduler=False` — chỉ để suy luận, không resume được. Resume dùng
  checkpoint epoch mới nhất, qua tệp `last_checkpoint` (hook ghi vì `save_last=True`).

**Kết luận cho lần train thật:** Colab ngắt giữa chừng → chạy lại ô `[3.11]` là
tiếp đúng epoch và đúng lịch LR; chỉ mất trạng thái AdamW (vài trăm vòng đầu sau
resume "khởi động lại" quán tính — chấp nhận được, không đáng đánh đổi bằng việc
ghi thêm ~600 MB optimizer lên Drive mỗi epoch). Muốn train lại từ đầu: xoá
`runs/train/e1_cascade_convnext_t` trên Drive.

⚠️ **Cập nhật 07/10/2026:** phần resume dưới đây nổ trên thực tế ở lần chạy thật
đầu tiên — không phải vì logic mmengine sai (nó vẫn đúng), mà vì torch ≥ 2.6 đổi
mặc định `weights_only` của `torch.load`. Đã vá và khoá bằng test; xem **§3.13**.

### 3.11 Lỗi thật thứ hai trên Colab: `{{_base_.ann_overfit}}` hỏng IM LẶNG (01/10/2026)

Ô `[3.10]` (train overfit 20 ảnh) dừng ở tiền kiểm: "đường dẫn" annotation nó
nhận được không phải đường dẫn mà là một chuỗi rác dạng `"_ann_overfit_d1b839"`
(có cả ngoặc kép bên trong giá trị). mmengine không ném lỗi nào ở bất kỳ bước
nào — họ "thất bại im lặng" lần thứ ba của đồ án (§2.6, §3.6, và đây).

**Cơ chế** (đọc source mmengine 0.10.7, `_pre_substitute_base_vars`).
`overfit20.py` viết `ann_file='{{_base_.ann_overfit}}'` trong ngoặc kép. Hàm
thay thế dùng `re.sub(regexp, f'"{randstr}"', ...)` — placeholder **đã bọc sẵn
ngoặc kép**. Giá trị chuỗi vì thế thành `'"_ann_overfit_d1b839"'`, tức mang luôn
hai dấu ngoặc kép. Bước tra cứu sau đó so `v in base_var_dict` với `v` đang có
ngoặc kép → **trượt**, và mmengine **không kiểm tra tra cứu trượt**; giá trị rác
đi thẳng vào config rồi vào tiền kiểm.

Dạng **trần** (không ngoặc kép) thì chạy được theo source — nhưng như §3.6 đã
nêu, việc thay thế chỉ xảy ra khi **toàn bộ** chuỗi đúng bằng cú pháp đó, mà ở
đây cần một giá trị chuỗi nằm trong `dict(...)`.

**Điều §3.6 kết luận sai, và vì sao.** §3.6 đã đối chiếu source rồi kết luận cú
pháp này "đúng, không phải suy đoán" — nhưng phép đối chiếu đó **chỉ kiểm regex
và điều kiện khớp toàn chuỗi**, không kiểm placeholder có bọc ngoặc kép hay
không, và không hề chạy thử. Đọc source đúng vẫn có thể kết luận sai nếu bỏ sót
một dòng. Đây là lý do §3.5/§3.6 đặt nguyên tắc: kết luận từ source chỉ là giả
thuyết cho tới khi có lần chạy thật xác nhận.

**Sửa:** bỏ hẳn cơ chế lấy biến từ config cha, config con tự viết đường dẫn.

- `configs/mmdet/overfit20.py` — `train_dataloader`/`val_dataloader` ghi
  `ann_file` **TƯƠNG ĐỐI** (`annotations/instances_overfit20.json`; mmdet tự ghép
  với `data_root` thừa hưởng từ config cha), `val_evaluator` ghi **TUYỆT ĐỐI** —
  CocoMetric mở thẳng tệp bằng pycocotools, không biết `data_root` là gì. Hai
  dạng khác nhau là **cố ý**, đúng theo cách config cha đang làm.
- `configs/mmdet/cascade_convnext_t_floodnet.py` — xoá biến `ann_overfit` (không
  còn ai dùng).

**Khoá lại bằng test** (`tests/test_train.py` mục 5, đọc config bằng `ast`, không
cần mmengine) — ba phép kiểm mới:

1. **Cấm** cú pháp `{{_base_.` trong **mọi** tệp `.py` của `configs/` (bỏ qua
   dòng comment — chính chú thích giải thích vì sao cấm thì được phép nhắc tới
   nó). Cấm ở mức cả thư mục chứ không riêng `overfit20.py`, để config thí nghiệm
   sau này (E2–E7) không lặp lại.
2. Ba đường dẫn train / val / `val_evaluator` phải trỏ về **cùng một tệp**, và
   tệp đó phải khớp `data_root` của config cha (ghép bằng `posixpath` — Colab
   chạy Linux còn máy này Windows, `os.path.join` sẽ ra `\` và so sánh sai).
3. Hai giá trị đọc ra phải là **chuỗi thật** trước khi so sánh: hai `_BieuThuc`
   luôn bằng nhau (`__eq__` cố ý như vậy), so sánh thẳng là tự cho mình ĐẠT.

**Số phép kiểm:** 455 (§3.5). Cả 8 bộ PASS trên máy ngày 01/10/2026.

**Bài học:** cú pháp "tiện" của mmengine chỉ tiện khi **toàn bộ** chuỗi là cú
pháp đó. Đã dùng tới nó thì phải có một phép kiểm chạm vào giá trị thật sau khi
mmengine xử lý xong — không phép kiểm nào trong repo này làm được điều đó nếu
không chạy thật, nên cách rẻ nhất là **đừng dùng**.

### 3.12 Lần chạy lại [3.10] ngày 07/10/2026: bản sửa §3.11 SỐNG, nhưng GATE 3 CHƯA ĐẠT (mAP 0,618)

Bản sửa §3.11 được chạy lại trên Colab. **Tiền kiểm ĐẠT HẾT** — đúng thứ cần
kiểm chứng: `[2]` trỏ `instances_overfit20.json` cho cả train / val /
val_evaluator (không còn chuỗi rác), `[2b]` xác nhận 20/20 ảnh nằm trong
`images/train/`. Chạy sạch: 60/60 epoch, thoát 0, không NaN, checkpoint "best"
nằm ở chính epoch cuối.

**Máy:** Colab cấp **A100-SXM4-40GB** (không phải T4 như dự kiến — số thời gian
dưới đây KHÔNG dùng để lên kế hoạch cho T4). mmdet 3.3.0 / mmcv 2.2.0 /
mmengine 0.10.7 / torch 2.11.0+cu130. Tập overfit: 20 ảnh / 125 box. Cả lần
chạy (60 epoch = 600 vòng) mất **8 phút 52 giây** (11:24:44 → 11:33:36),
0,26 s/vòng lúc cuối, VRAM 6.509 MB.

**Số đo mAP trên chính 20 ảnh train** (val_dataloader của overfit20 = 20 ảnh đó):

| Epoch | mAP | mAP50 | ghi chú |
|---|---|---|---|
| 1–4 | 0,000 | 0,000 | warmup 50 vòng (5 epoch) — chưa kịp học |
| 5 | 0,035 | 0,119 | bắt đầu nhích |
| 20 | 0,127 | 0,327 | |
| 29 | 0,315 | 0,632 | |
| 39 | 0,518 | 0,862 | |
| 40 | 0,413 | 0,833 | dao động thường (epoch 30 cũng tụt vậy) — CHƯA liên quan LR |
| 48 | 0,588 | 0,885 | leo lại và vượt mức cũ |
| 52–54 | 0,602–0,609 | ~0,89 | chững lại TRƯỚC mốc giảm LR [55] |
| 55 | 0,613 | 0,887 | |
| 56–60 | 0,611 → 0,618 | 0,888 | 5 epoch cuối nhích tổng +0,005 |

Chốt epoch 60: mAP 0,618 / mAP50 0,888 / mAP75 0,754 (small 0,245 — medium
0,559 — large 0,690), AR 0,707.

**Phán quyết — GATE 3 CHƯA ĐẠT**, theo đúng tiêu chí do `overfit20.py` tự viết:
"thấy mAP > 0.9 là đủ kết luận" và "nếu chạy tới cuối mà mAP vẫn thấp thì là lỗi
thật, không phải 'cần thêm epoch'". 0,618 < 0,9, và đuôi đường cong đã bão hoà
(5 epoch cuối +0,005) — không thể là chuyện thiếu epoch.

**Nhưng phán quyết đó KHÔNG nói đường ống có lỗi** — chỗ dễ đọc sai nhất. Kiểu
hỏng "mAP ~0 vì nhãn / đường dẫn / eval sai" đã bị loại: loss giảm đều, RPN hội
tụ (`loss_rpn_cls` 0,0152 / `loss_rpn_bbox` 0,0169), đầu phân loại cả ba stage
đạt acc ~92–95%, mAP lớn dần qua suốt 60 epoch. Model CÓ học; nó học chậm và
dừng thấp. Câu hỏi mở: đường ống có khả năng **học vẹt** hay không, hay có lỗi
thật (nhãn / box / loss / eval) mà biểu hiện là trần thấp?

**Hai nghi phạm, cả hai đều thuộc recipe chứ không thuộc dữ liệu hay code:**

1. **Tăng cường quá mạnh cho một bài thuộc lòng.** Mỗi epoch, cùng một ảnh lại
   vào ở một tỉ lệ khác (5 mức), lật với xác suất 0,75 (3 hướng), đổi sáng.
   Model phải học bất biến theo tỉ lệ TỪ 20 ẢNH trước khi thuộc lòng được —
   ngược hẳn mục đích của phép thử.
2. **Lịch LR tự bóp.** Cả lần chạy chỉ 600 vòng. Ba pha LR đo được từ log:
   epoch 1–40 chạy `base_lr` 1e-4, epoch 41–55 chạy 1e-5, epoch 56–60 chạy 1e-6
   (`lr` của layer_0 lần lượt 8,2e-06 → 8,2e-07 → 8,2e-08). Đường cong chững từ
   epoch 52–54, tức TRƯỚC khi mốc [55] kịp giảm lần hai — nhưng đừng đọc ngược:
   chững ở 1e-5 chỉ nói 1e-5 không đủ đẩy tiếp, không chứng minh 1e-4 sẽ đủ.
   Đó chính là việc của phép thử dưới đây.

**Phép thử phân biệt — ô `[3.10b]`, config `overfit20_hocvet.py`.** Giữ NGUYÊN
dữ liệu, nhãn, kiến trúc, hàm loss và phép đo (`val_dataloader` không bị ghi đè
→ so được trực tiếp với [3.10]); chỉ hạ hai nghi phạm về điều kiện dễ nhất:
pipeline train = y hệt pipeline lúc ĐÁNH GIÁ (bỏ toàn bộ tăng cường), LR ×10
(1e-3) với MỘT mốc giảm ở epoch 30 trên 40 epoch, thêm `classwise=True` để có
AP theo từng lớp, `work_dir` riêng (chống `resume=True` chạy tiếp nhầm
checkpoint cũ). Cách đọc kết quả, chốt trước khi chạy để không tự huyễn hoặc:

- **mAP > 0,95** → đường ống ĐÚNG, thứ chặn [3.10] là recipe → **GATE 3 ĐẠT**.
  (Không suy ngược gì cho train thật: LR 1e-4 + lịch [16,22]/24 epoch của config
  chính là để train 1.445 ảnh, không phải để thuộc lòng 20 ảnh.)
- **Vẫn ~0,6x và bão hoà** → lúc đó mới là lỗi thật trong nhãn / box / loss /
  eval; log lần này in kèm AP từng lớp để khoanh vùng tiếp.

Một chi tiết của [3.10] cần liếc mắt khi chạy [3.10b]: log có 240 dòng cảnh báo
"bbox/polygon is out of bounds" (4 dòng/epoch) phát ra từ **bước VẼ ảnh minh
hoạ** của visualizer, không phải bước tính metric (metric tính trên toạ độ gốc,
không đi qua đường vẽ). Nghi do làm tròn toạ độ khi resize nhưng chưa kiểm
chứng — nên mở vài ảnh trong `runs/sanity/overfit20_hocvet/vis_data` xem box vẽ
có ĐÚNG CHỖ không; box vẽ lệch thật thì đó là manh mối, không còn là chuyện thẩm
mỹ.

**Khoá lại bằng test** (`tests/test_train.py` mục 5b, đọc config bằng `ast`,
không cần mmengine) — 9 phép kiểm giữ đúng những gì làm nên giá trị của phép
thử: pipeline train chỉ còn 4 bước không tăng cường; Resize của train TRÙNG
Resize của `test_pipeline` ở config chính; LR 1e-3; một mốc [30] trên 40 epoch;
`classwise=True`; `work_dir` khác hẳn của [3.10]; `val_dataloader` không bị ghi
đè; `_base_` đúng `./overfit20.py`; và tệp tồn tại. "Dọn dẹp" config rồi chạy
nhầm một lần khác thì phép chẩn đoán mất giá trị mà không có gì báo — đúng loại
lỗi im lặng mà §3.11 vừa dạy.

**Số phép kiểm:** 455 → 464 (§3.5). Cả 8 bộ PASS trên máy ngày 07/10/2026.

**Chạy [3.10b] cùng ngày — giai đoạn 1 (40 epoch): recipe bị buộc tội, mốc chưa
đạt.** Chạy sạch 40/40 epoch trên A100 (thoát 0, **6 phút 23 giây**, 11:52:05 →
11:58:28), tiền kiểm ĐẠT HẾT (đúng 20 ảnh / 125 box, `classwise` in AP từng
lớp), cùng lượng cảnh báo "out of bounds" như [3.10] (80 dòng, vẫn chỉ ở bước
vẽ). `base_lr` đo từ log: epoch 1–30 chạy **1e-3**, epoch 31–40 chạy **1e-4** —
lịch thiết kế sống đúng.

| Epoch | mAP | ghi chú |
|---|---|---|
| 25–30 | 0,35–0,47 | 30 epoch ở 1e-3 chỉ loanh quanh ~0,4 |
| 31 | 0,523 | vừa hạ xuống 1e-4 — nhảy ngay |
| 32–36 | 0,625 → 0,669 | |
| 37–40 | 0,722 / 0,736 / **0,744** / 0,742 | 10 epoch ở 1e-4 leo +0,37 |

Chốt epoch 40: mAP **0,742** / mAP50 0,905 / mAP75 0,898 (small 0,580 — medium
0,748 — large 0,801), AR 0,833. AP từng lớp: **flooded_building 0,812** (mAP50
0,961), **non_flooded_building 0,673** (mAP50 0,848). Loss epoch cuối:
`loss_rpn_cls` 0,0034 (so 0,0152 của [3.10]), acc ba stage 96,8 / 96,6 / 96,3
(so 94,5 / 91,9 / 93,1) — chặt hơn hẳn.

**Đọc giai đoạn 1** — nghi phạm recipe được xác nhận, "lỗi thật" bị loại thêm
một bậc, nhưng mốc 0,95 chưa đạt nên **GATE 3 vẫn chưa chốt**:

- So trực tiếp với [3.10] (cùng dữ liệu, cùng phép đo — chỉ recipe đổi):
  0,618-bão-hoà-sau-60-epoch → **0,742-còn-leo-sau-40-epoch**. Cùng 20 ảnh đó,
  chỉ đổi tăng cường + lịch LR mà mAP +0,124 và tốc độ leo cuối gần gấp 5. Kiểu
  hỏng "nhãn / box / eval sai" càng khó đứng vững.
- Bước ngoặt nằm ĐÚNG ở mốc [30]: 10 epoch ở 1e-4 leo +0,37 trong khi 30 epoch
  ở 1e-3 chỉ loanh quanh 0,4. Nửa "LR ×10" của phép chẩn đoán hoá ra **phản tác
  dụng** — LR chạy được là **1e-4, đúng base_lr của config chính** (tin tốt cho
  Phase 4). Nửa "tắt tăng cường" không tách được ở lần này (cố ý đổi cùng lúc).
- KHÔNG rơi vào nhánh "~0,6x và bão hoà → lỗi thật": đuôi đường cong vẫn leo
  (+0,02 trong 4 epoch cuối, so +0,005/5 epoch của [3.10]) — lần này **hết
  epoch**, không phải bão hoà. Cũng KHÔNG đạt nhánh "> 0,95". Số đo per-class để
  lại manh mối nếu phải đào tiếp: non_flooded thua flooded 0,14.

**Giai đoạn 2 — chốt trước khi chạy: gia hạn 40 → 120 epoch, không đổi gì khác.**
Chạy lại đúng ô [3.10b]; `resume=True` tự nạp checkpoint epoch 40 và chạy tiếp
41..120 ở nguyên `base_lr` 1e-4 (mốc [30] đã qua — dù lịch được nạp từ checkpoint
hay dựng lại từ config thì cũng vậy). Xác nhận trong log 2–3 epoch đầu:
`base_lr: 1.0000e-04` và mAP epoch 41 ~0,7x, KHÔNG phải ~0 (thấy ~0 là resume
hỏng, dừng báo ngay). Chốt đọc:

- **mAP > 0,95** → GATE 3 ĐẠT, bàn tiếp LR + số epoch cho train thật.
- **Bão hoà < 0,9** (10 epoch liền nhích < 0,01) → trần thật, không phải thiếu
  epoch; đào tiếp bằng AP từng lớp + ảnh `vis_data`.
- **0,9–0,95, hoặc hết 120 epoch mà còn leo** → gửi số liệu, quyết định cùng nhau.

Thấy > 0,95 là dừng được — checkpoint epoch vừa xong đã ghi ra Drive. Config đã
đổi `max_epochs` 40 → 120 và mục 5b của test khoá theo số mới (vẫn **464** phép
kiểm, cả 8 bộ PASS lại trên máy 07/10/2026).

**Hai lần chạy thử đầu của giai đoạn 2 (07/10/2026) nổ ngay khi resume** —
`_pickle.UnpicklingError` trên torch ≥ 2.6, chưa epoch nào chạy, checkpoint
`epoch_40.pth` còn nguyên (lần hai nổ sau khi vá vòng 1, ở global numpy). Đây là
lỗi thật thứ ba trên Colab; đã vá TRỌN — hai vòng, có bằng chứng dựng lại
checkpoint tại chỗ — và khoá bằng test ở **§3.13**. Chạy lại ô `[3.10b]` sau khi
pull code mới.

### 3.13 Lỗi thật thứ ba trên Colab: resume nổ `_pickle.UnpicklingError` (torch ≥ 2.6, 07/10/2026)

Lần ĐẦU TIÊN đồ án thật sự resume một checkpoint (ô `[3.10b]` giai đoạn 2, chạy
tiếp từ epoch 40) thì nổ ngay lúc nạp, trước khi epoch 41 kịp bắt đầu:

```
_pickle.UnpicklingError: Weights only load failed ... Unsupported global:
GLOBAL mmengine.logging.history_buffer.HistoryBuffer
```

Không mất gì: lỗi xảy ra ở bước ĐỌC, checkpoint `epoch_40.pth` còn nguyên. Log
báo thẳng `Mã thoát: 1` — lỗi ồn ào, không phải loại im lặng.

**Cơ chế.** Từ torch 2.6, `torch.load` mặc định `weights_only=True` và từ chối
nạp mọi lớp không nằm trong danh sách an toàn. Checkpoint của mmengine lưu cả
`message_hub` (lịch sử loss) — bên trong là các `HistoryBuffer` — còn mmengine
0.10.7 gọi `torch.load(filename, map_location=...)` trần, không truyền
`weights_only`. Hai thứ đó cộng lại thành: **mọi lần resume đều nổ**; còn nạp
`load_from` / trọng số ImageNet thì không (checkpoint đó thuần tensor — thấy rõ
trong log: bước nạp ImageNet chạy xong ngay trước đó).

**Vì sao §3.10 không bắt được.** §3.10 là bản đọc source mmengine để trả lời
"resume khôi phục được gì" — đúng về phía mmengine (nó có lưu `param_schedulers`),
nhưng không thấy được thay đổi mặc định nằm ở **torch**, ngoài tầm đọc đó. Và vì
mọi lần chạy trước ([3.10], `[3.10b]` giai đoạn 1) đều là `work_dir` mới, đường
resume chưa từng được THỰC THI. Bài học lặp lại lần thứ ba: đọc code không thay
được một lần chạy thật — và đây là lỗi đầu tiên thuộc loại "code của mình đúng
nhưng môi trường đổi mặc định".

**Vá vòng 1** (`scripts/train.py`, hàm `va_torch_load_resume()`, gọi trong
`main()` ngay trước `runner.train()`): cho phép ĐÚNG lớp `HistoryBuffer` bằng
`torch.serialization.add_safe_globals([HistoryBuffer])` — đúng như chính thông
báo lỗi gợi ý — thay vì hạ `weights_only=False` cho mọi checkpoint; phần còn
lại của checkpoint vẫn được nạp ở chế độ an toàn. Torch cũ không có
`add_safe_globals` (khi đó mặc định đã là `weights_only=False`) thì hàm tự bỏ
qua, không cần vá. Lúc chạy, log in một dòng `[vá] torch.load: ...` để xác nhận
vá đã sống.

#### Vòng 2 (cùng buổi 07/10/2026): cho phép lớp chứa vẫn CHƯA đủ

Đẩy vòng 1 lên, người dùng chạy lại ô `[3.10b]`. Dòng `[vá]` in ra (vá đã sống)
nhưng resume vẫn nổ — ở một global KHÁC, đọc thẳng từ checkpoint:

```
_pickle.UnpicklingError: ... Unsupported global:
GLOBAL numpy._core.multiarray._reconstruct
```

Vẫn không mất gì (nổ ở bước ĐỌC, `epoch_40.pth` còn nguyên, 0 epoch chạy).
**Bài học của vòng 1:** cho phép LỚP chứa chưa đủ — còn phải cho phép thứ nó
CHỨA. `HistoryBuffer` lưu dữ liệu bên trong bằng **hai mảng numpy**
(`_log_history`, `_count_history`); mảng numpy lại cần `_reconstruct` và — trên
numpy ≥ 2 — cả các **lớp mô tả dtype**.

**Truy tới cùng bằng cách dựng lại checkpoint tại chỗ (không cần GPU).** Tải
wheel mmengine 0.10.7 (`pip download --no-deps`, không đụng môi trường) rồi đọc
nguồn thật: `HistoryBuffer.__getstate__` (nhét 4 hàm thống kê `min/max/current/
mean` vào state), `MessageHub.state_dict`, `Runner.save_checkpoint`,
`ParamScheduler.state_dict`. Kết luận: ngoài `HistoryBuffer` + mảng numpy, phần
còn lại của checkpoint chỉ là dict/int/float/chuỗi/tensor — tức **chỉ còn đúng
hai cơ chế lạ** phải xử lý. Sau đó dựng LẠI một checkpoint đúng cấu trúc đó
ngay trên máy Windows (numpy 2.5.1, torch 2.14.0+cpu — cùng thế hệ ≥ 2.6), đặt
nguồn `history_buffer.py` THẬT vào đúng đường dẫn module để pickle ghi ra đúng
tên `mmengine.logging.history_buffer.HistoryBuffer`, rồi leo từng nấc lỗi y hệt
Colab — mỗi nấc chỉ qua được khi thêm đúng mảnh còn thiếu:

1. chỉ `HistoryBuffer` → nổ `numpy._core.multiarray._reconstruct` — **khớp y
   nguyên thông báo Colab thứ hai**;
2. thêm `_reconstruct`/`scalar`/`np.ndarray` → nổ
   `numpy.dtypes.Float64DType`: numpy 2 dựng lại dtype qua lớp mô tả riêng và
   torch kiểm tra ĐÚNG lớp (`type(inst)`), nên phải đăng ký cả bộ lớp trong
   `np.dtypes` (lọc bằng `issubclass(v, np.dtype)` — 33 lớp; `np.dtypes` chỉ có
   từ numpy 2, có guard `hasattr`);
3. thêm bộ dtype → nổ `getattr`: tên có dấu chấm (`HistoryBuffer.min`) nên
   pickle protocol 2 (mặc định của `torch.save`) viết chúng thành
   `getattr(HistoryBuffer, 'min')` — **4 hàm thống kê không cần đăng ký riêng,
   chỉ cần `getattr`**. Phụ chú: dạng tuple `(getattr, "__builtin__.getattr")`
   bị torch mới vô hiệu vì khi đọc nó tự đổi `__builtin__` → `builtins`; vẫn
   đăng ký cả hai kiểu khỏi phụ thuộc phiên bản torch.

**Danh sách cho phép chốt** (giữ đúng nguyên tắc "cho phép ĐÚNG thứ cần, không
hạ `weights_only=False`"):

```python
torch.serialization.add_safe_globals([
    HistoryBuffer,
    getattr, (getattr, "__builtin__.getattr"),
    _reconstruct, scalar, np.ndarray, np.dtype, *lop_dtype,
])
```

**Bằng chứng chạy trên máy (07/10/2026).** Script kiểm (đặt ngoài repo) rút
CHÍNH hàm `va_torch_load_resume` từ `scripts/train.py` (nguyên văn, không chép
tay) rồi: (a) đối chứng chưa vá — nổ đúng câu `Unsupported global: GLOBAL
mmengine.logging.history_buffer.HistoryBuffer` như Colab; (b) gọi hàm rồi nạp
lại — **sạch**: `_log_history [1. 0.9 0.8 0.7 0.6]` dtype `float64`, `mean()`
ra đúng số học (0,4), 4 hàm thống kê khôi phục, `param_schedulers` giữ
`milestones [30]`, tensor nguyên vẹn. Khác vòng 1 ở chỗ: danh sách lần này rút
từ SOURCE của đúng phiên bản mmengine đang chạy và dựng lại bằng đúng source
đó, tái hiện được cả hai thông báo lỗi Colab trước khi nạp sạch — không còn
mảnh nào là suy đoán. Nếu Colab vẫn còn nổ ở global thứ tư thì thông báo lỗi in
thẳng tên nó: thêm một mảnh vào danh sách là xong (mỗi mảnh đều có tên và lý
do — đúng thứ `weights_only=False` đánh mất).

**Khoá bằng test** (mục 1c, **12 phép kiểm AST** — 5 của vòng 1 + 7 của vòng 2,
đã kiểm âm: bỏ `_reconstruct`/`scalar`/`*lop_dtype` khỏi danh sách là 2 test đỏ
đúng chỗ, khôi phục thì xanh): hàm vá tồn tại; có guard `hasattr`; list chứa
`HistoryBuffer`; import `torch` / `HistoryBuffer` nằm TRONG hàm (giữ được tính
"test trên máy sạch"); `main()` gọi hàm vá TRƯỚC `runner.train()` (resume xảy ra
bên trong `train()`, gọi sau là đã quá muộn); list có `getattr` +
`_reconstruct`/`scalar`; có `np.ndarray`/`np.dtype`; có cặp tuple
`__builtin__.getattr`; có trải `*lop_dtype`; nạp được từ CẢ HAI đường
`numpy._core` và `numpy.core`; dự phòng nằm trong `try/except ImportError`; và
lọc dtype bằng `issubclass(..., np.dtype)` + guard `hasattr(np, 'dtypes')`.

**Số phép kiểm:** 464 → 469 (vòng 1) → **476** (vòng 2) (§3.5). Cả 8 bộ PASS
trên máy 07/10/2026.

### 3.14 Lỗi thật thứ tư trên Colab — lỗi IM LẶNG đầu tiên: resume train ở LR 1e-6 (07/10/2026)

Ô `[3.10b]` giai đoạn 2 chạy **trọn vẹn, thoát 0**: resume từ epoch 40, chạy tiếp
41..120, mAP 0,747 → **0,779** (mAP50 0,919; AP từng lớp ở epoch 120: flooded
0,838 / non_flooded 0,720; đỉnh 0,781 ở epoch 115). Nhìn bề ngoài là một lần
chạy tốt — bản vá §3.13 sống (dòng `[vá] torch.load: ...` in ra trước khi train),
mAP epoch 41 ~0,747 chứ không ~0, loss giảm đều.

Nhưng **cả 80 epoch train ở `base_lr: 1.0000e-06`**, `lr: 8.2354e-08`: đúng 80
dòng log, không một dòng nào khác. Lịch đã định 1e-4 — **sai 100 lần**, và chốt
ghi sẵn cho giai đoạn 2 (`base_lr: 1.0000e-04` trong 2–3 epoch đầu) đã bị bỏ qua
mà không ai để ý, vì con số in ra vẫn là một con số "trông hợp lý".

**Hệ quả: số của giai đoạn 2 KHÔNG dùng để chốt gì.** Nhánh "bão hoà < 0,9 →
trần thật" không được phép áp dụng — bão hoà ở LR sai 100 lần thì không nói gì
về trần thật. GATE 3 vẫn treo. Đây là lỗi đầu tiên trong đồ án mà **không có gì
báo**: không exception, không mã thoát khác 0, không dòng log bất thường.

**Chuỗi nhân quả** (4 bước, mỗi bước đối chiếu source mmengine 0.10.7):

1. `save_optimizer=False` (config chính lẫn config chẩn đoán) → checkpoint không
   chứa trạng thái optimizer → `Runner.resume()` bỏ qua bước nạp optimizer
   (`if 'optimizer' in checkpoint and resume_optimizer`, `runner.py:2005`), và
   các nhóm tham số **giữ nguyên LR vừa dựng từ config** (1e-3 ở config chẩn
   đoán, 1e-4 ở config chính).
2. `LinearLR(start_factor=0.001)` — `_ParamScheduler.__init__` kết thúc bằng
   `self.step()` (`param_scheduler.py:129`), và bước thứ 0 ấy nhân thẳng
   `start_factor` vào MỌI nhóm (`LinearParamScheduler._get_value`, dòng 791–795:
   `group['lr'] * self.start_factor`): **1e-3 → 1e-6**. Ở config chính
   (`lr=1e-4`, cùng `start_factor=0.001`): **1e-4 → 1e-7**.
3. `resume()` sau đó nạp `param_schedulers` **từ checkpoint**
   (`load_state_dict` = `self.__dict__.update`, dòng 152) — nạp về cả `end` lẫn
   `last_step`/`_global_step` của lần chạy CŨ. Kết quả: cả hai lịch **đứng yên
   vĩnh viễn** — `LinearLR` đã qua `end` (warmup xong từ epoch 5), còn
   `MultiStepLR` có `last_step` ngoài khoảng `[begin, end=40)` nên không bước nào
   còn được áp dụng nữa.
4. Không còn bước nào hoàn lại được hệ số 0,001 ở (2). LR đứng nguyên ở 1e-6 suốt
   80 epoch. **Và mốc giảm LR cũng không bao giờ nổ** — lịch bị đóng băng, nên
   triệu chứng nhìn thấy là một con số SAI nhưng KHÔNG ĐỔI, chứ không phải một
   đường LR méo mó dễ nhận ra.

**Vì sao trước đó không ai thấy.** Đây là lần resume ĐẦU TIÊN chạy được tới nơi
(§3.13 chặn hai lần trước đó ở bước đọc). §3.10 đã đọc source mmengine và kết
luận đúng rằng resume khôi phục được `param_schedulers` — chính vì thế mà không
ai ngờ rằng thứ được khôi phục ấy lại **đóng băng** lịch. Chú thích
`save_optimizer=False` trong config chính cũng đã đọc source và kết luận đúng
rằng resume "KHÔNG sập" — đúng, nó không sập, nó chỉ train sai 100 lần.

**Dựng lại tại chỗ, khớp từng chữ số.** Không cần GPU: tải wheel mmengine 0.10.7
về đọc source thật, dựng lại đúng đường resume của ô `[3.10b]` (AdamW nhiều nhóm
+ `OptimWrapper` + `LinearLR(end=50)` + `MultiStepLR(milestones=[30])`, resume
`save_optimizer=False`), và in ra `base_lr` — **ra đúng `1.0000e-06` /
`lr[0] = 8.2354e-08`, khớp từng chữ số với 80 dòng log Colab**. Đối chứng: nếu
checkpoint CÓ optimizer thì LR sau resume đúng ngay — tức thủ phạm đúng là bước
nạp optimizer bị bỏ qua.

**Cách vá** (`scripts/train.py`, hàm `dat_lai_lr_sau_resume` + hook
`dang_ky_va_lr_sau_resume`). Lịch của mmengine là hàm THUẦN của hai bộ đếm
(`_global_step`, `last_step`) nhân lên giá trị hiện có của nhóm (đã đọc
`_get_value` của cả hai lớp), nên dựng lại trạng thái đúng không cần mô phỏng
công thức nào — chỉ cần chạy lại chính mã của mmengine:

1. **Dựng lại lịch từ config HIỆN TẠI** (`runner.build_param_scheduler(runner.cfg.param_scheduler)`),
   không dùng lại lịch trong checkpoint — lịch cũ đã đóng băng `end`/`milestones`
   của lần chạy trước (gia hạn epoch mà không dựng lại thì mốc mới không bao giờ nổ).
2. **Trả LR của mọi nhóm về `initial_lr`** (kể cả nhóm giả `base_param_settings`
   mà mmengine dùng để in `base_lr` — con số duy nhất người đọc log nhìn thấy).
3. **Trả CẢ HAI bộ đếm của lịch về `-1`** rồi chạy lại `_so_buoc_lich(...)` bước.
   Chỗ này là chỗ suýt sai lần nữa: hàm dựng của mmengine gọi sẵn một bước
   `step()` (bước 0) NGAY LÚC DỰNG `moi` — tức trước khi ta trả LR về
   `initial_lr` — và bước ấy tiêu mất `last_step=0`. Chỉ trả LR mà không trả bộ
   đếm thì chuỗi replay thiếu bước 0 và bù lại ở cuối, lệch đúng
   `1/start_factor` = **1000 lần** — harness bắt được: LR ra **1e-1** thay vì
   1e-4. Vì thế `_so_buoc_lich` tính CẢ bước 0, và hàm vá trả bộ đếm về `-1`
   trước khi chạy lại.
4. Gán `runner.param_schedulers = moi` — từ đây lịch mới là lịch đang chạy, kể
   cả khi checkpoint tiếp theo được ghi.

Hook chạy ở mốc `before_train` (priority `VERY_HIGH`) vì đó là mốc SỚM NHẤT mà cả
ba việc đã xong: wrapper dựng rồi (1733), lịch dựng rồi (1737), và resume đã nạp
xong (1765) — `ParamSchedulerHook` không có `before_train` nên không giành mất
lượt đặt LR. Không phải resume thì hàm không đụng gì (`iter` và `epoch` đều 0).
Vẫn **KHÔNG cần lưu optimizer**: khác biệt còn lại chỉ là các moment của AdamW
khởi động lại — đúng thứ chú thích `save_optimizer=False` đã cân nhắc.

**Kiểm chứng bằng Runner THẬT, không phải mô phỏng** (harness đặt ngoài repo,
mmengine 0.10.7 + torch trên máy Windows, không cần GPU/mmdet). Dùng đúng
`Runner.from_cfg` + `runner.train()` + hook thật, chỉ thay model bằng một mô-đun
nhỏ và dataloader bằng 8 mẫu rỗng. Sáu ca, tất cả ĐẠT:

| Ca | Kết quả |
|---|---|
| Chạy liên tục 120 epoch (tham chiếu) | warmup kết thúc trong epoch 13 — harness dùng **4 vòng/epoch**; dữ liệu thật 10 vòng/epoch nên warmup hết ở cuối epoch 5 (§3.15). Mốc [30] nổ ở cuối epoch 30 (epoch 31 trở đi học ở 1e-4) |
| Resume **không vá** | `1.0000e-06` — tái hiện đúng con số Colab |
| Resume **có vá** | `1.0000e-04` ngay sau resume, và khớp lần chạy liên tục ở **MỌI** epoch 40..119 (không sai epoch nào) |
| Chạy MỚI có vá | LR y hệt lần chạy sạch ở cả 120 epoch — vá không đụng vào chạy mới |
| Mở rộng kèm MỐC MỚI `[30, 70]` | mốc 70 nổ đúng chỗ (1e-4 → 1e-5 ở cuối epoch 70) dù checkpoint cũ đóng băng `end=40` |
| Resume GIỮA warmup (sau 1 epoch) | khớp lần chạy liên tục (1,6410e-04) |

**Khoá bằng test** (mục 9, **18 phép kiểm AST** + 1 phép kiểm thư mục `2` ở mục
5b, đã kiểm âm: bỏ hai dòng trả bộ đếm → đúng 1 test đỏ; bỏ `+ 1` của
`_so_buoc_lich` → đỏ; đổi `priority="VERY_HIGH"` → đỏ; khôi phục thì xanh): hàm
`_so_buoc_lich` trả `(...) + 1` (tính cả bước 0); hàm vá import TRONG hàm (giữ
máy sạch chạy được bộ test); có đủ các nhánh báo rồi BỎ QUA thay vì sửa bừa
(wrapper kiểu khác, không có lịch LR, chạy mới, thiếu `initial_lr`, lịch khác
cấu trúc); dựng lại lịch từ config hiện tại; trả LR về `initial_lr`; trả CẢ HAI
bộ đếm về -1 trong vòng `for s in moi`; chạy lại đúng `_so_buoc_lich(...)` bước;
gán `runner.param_schedulers = moi`; hook có `before_train` + priority
`VERY_HIGH`; `main()` đăng ký TRƯỚC `runner.train()`.

**Giới hạn đã biết:** chỉ chạy cho `OptimWrapper` đơn (`OptimWrapperDict` — không
có trong đồ án — thì hàm tự BỎ QUA kèm dòng in lý do, chứ không sửa bừa); replay
chạy tuần tự từng lịch (đúng cho config của đồ án: cả hai lịch đều kiểu NHÂN lên
giá trị hiện có, phép nhân giao hoán; nếu sau này trộn lịch "đặt giá trị tuyệt
đối" với lịch "nhân" trên cùng một tham số thì phải xem lại).

**Chạy lại giai đoạn 2 — lần hai.** Không resume được nữa: `epoch_40.pth` đã bị
`max_keep_ckpts=2` xoá, và thư mục cũ đã ở epoch 120 (resume vào đó là no-op im
lặng). Nên lần chạy đúng là **một lần chạy MỚI trọn 120 epoch trong
`overfit20_hocvet2`** — đúng bằng lịch mà giai đoạn 1 + 2 đã định ghép lại. Bản
vá không tham gia lần chạy này (nó chỉ can thiệp khi resume), nên phép kiểm nằm
ở chính dòng log LR, ghi sẵn trong đầu config: TỪ EPOCH 31 = `1.0000e-04` giữ tới
hết. (Bản checklist đầu tiên còn ghi "cuối epoch 13 = `1.0000e-03`" — SAI, xem
đính chính ở **§3.15**: con số đó suy từ harness 4 vòng/epoch, dữ liệu thật 10
vòng/epoch nên warmup hết ở cuối epoch 5.) Chốt đọc không đổi.

**Số phép kiểm:** 476 → **495**. Cả 8 bộ PASS trên máy 07/10/2026.

### 3.15 GATE 3 ĐẠT — [3.10b] lần chạy đúng: 120 epoch, best mAP 0,952 (07/10/2026)

Lần chạy MỚI trong `runs/sanity/overfit20_hocvet2` (thay cho lần resume vô hiệu ở
§3.14) đã chạy trọn **120/120 epoch, thoát 0**, không NaN. Đây là lần chạy `[3.10b]`
đầu tiên có lịch LR đúng, và nó trả lời dứt điểm câu hỏi của GATE 3. Hai dòng đầu
log xác nhận KHÔNG resume (`Did not find last_checkpoint to be resumed` /
`Auto resumed from the latest checkpoint None`) — đúng là chạy mới từ epoch 1.

**Lịch LR trong log** (10 vòng/epoch: 20 ảnh, batch 2):
- epoch 1–4: `base_lr` leo dần `1,8449e-04 → 7,9612e-04` (warmup 50 vòng);
- dòng epoch 5 trở đi: **`1.0000e-03`** (giữ tới hết epoch 30, 26 dòng);
- TỪ EPOCH 31 tới 120: **`1.0000e-04`** (90 dòng) — mốc `[30]` nổ đúng.

> ⚠️ **Đính chính số liệu của chính §3.14/§4 (và đã sửa ở config/notebook/README):**
> bản checklist đầu tiên ghi *"warmup hết giữa epoch 13, cuối epoch 13 =
> `1.0000e-03`"*. Con số đó **SAI**: nó suy từ harness kiểm chứng bản vá — mà harness
> dùng 8 ảnh / batch 2 = **4 vòng/epoch**, còn dữ liệu thật là 20 ảnh / batch 2 =
> **10 vòng/epoch**, nên 50 vòng warmup hết ở CUỐI EPOCH 5. Bản vá không hề bị ảnh
> hưởng (nó tự đếm bằng `runner.iter`, không hard-code vòng/epoch — đó là lý do
> harness 4 vòng/epoch vẫn kiểm chứng đúng được); chỉ con số ghi trong tài liệu sai,
> và chỉ mốc epoch 31 mới là mốc dùng để chốt nên không ảnh hưởng kết luận.

**Đường cong mAP** (val trên chính 20 ảnh train):

| Epoch | 30 | 40 | 50 | 88 | 100 | 110 | 114 | 116 | 120 |
|---|---|---|---|---|---|---|---|---|---|
| mAP | 0,447 | **0,741** | 0,833 | 0,909 | **0,950** | **0,952** | 0,951 | **0,952** | 0,918 |

- **epoch 40 = 0,741 khớp gần như y hệt giai đoạn 1 (0,742)** — hai lần chạy độc
  lập cho cùng kết quả ở cùng epoch: tái lập tốt, và càng chắc rằng số của giai
  đoạn 1 không phải may mắn.
- Sau mốc `[30]`: 10 epoch leo +0,37 (0,475 → 0,833 ở epoch 50) — đúng kiểu "vừa hạ
  LR là leo" đã thấy ở giai đoạn 1, nhưng lần này còn 90 epoch để đi tiếp.
- **Đuôi 0,918–0,952 = đã chững dạng DAO ĐỘNG, không còn leo**: 10 epoch cuối nhích
  −0,024 (10 epoch trước đó +0,023). Nguyên nhân dao động: LR đứng yên 1e-4 không
  giảm + eval chỉ 20 ảnh (biên độ ±0,03 là bình thường). Đây KHÔNG phải "trần thật"
  — nhánh trần thật yêu cầu chững *dưới* 0,9.

**Eval cuối (epoch 120):** mAP **0,918** — **mAP50 = mAP75 = 1,000**; AP theo vật:
nhỏ **0,900** / vừa 0,930 / lớn 0,922 (vật nhỏ đã leo từ 0,657 ở epoch 50); AR 0,939;
classwise flooded 0,908 / non_flooded 0,927 — **mỗi lớp đều mAP50 = mAP75 = 1,0**.
Nghĩa là: tìm đúng hết nhà, đúng lớp, box khít tới IoU 0,75; phần mAP còn thiếu nằm
hết ở các ngưỡng IoU 0,85–0,95 (rung vài pixel ở mép box) — không ảnh hưởng bài toán
đếm. Checkpoint tốt nhất: `best_coco_bbox_mAP_epoch_110.pth` = **0,952** (trên Drive).

**Đối chiếu với lần chạy dính lỗi LR (§3.14)** — cùng xuất phát từ epoch 40:
bản đúng đi **0,741 → 0,918** (best 0,952); bản lỗi đi 0,747 → 0,779 suốt 80 epoch.
Lỗi im lặng "ăn" ~0,17 mAP và suýt dẫn tới kết luận sai "trần thật ~0,78".

**Chốt GATE 3: ĐẠT** (quyết định cùng người dùng, 07/10/2026). Căn cứ: chốt đăng ký
ghi *"mAP > 0,95 → GATE 3 ĐẠT; thấy > 0,95 là DỪNG ĐƯỢC ngay (checkpoint epoch vừa
xong đã ghi ra Drive)"* — tức tiêu chí tính theo lúc ĐẠT TRONG lúc chạy, và đường
cong đã đạt ≥ 0,95 ở 4 epoch (100, 110, 114, 116), best 0,952. Ghi rõ cả hai con số
ở đây (0,952 best và 0,918 cuối) để không ai trích dẫn một con mà bỏ con kia —
**và tuyệt đối không trích dẫn cả hai như kết quả của đồ án**: đây là điểm trên tập
TRAIN 20 ảnh, chỉ chứng minh đường ống học được.

**Hệ quả:** đường ống (dữ liệu → nhãn → kiến trúc → loss → eval) đã được chứng minh;
nguyên nhân `[3.10]` dừng ở 0,618 là RECIPE (tăng cường mạnh + LR bị bóp bởi
milestones [40, 55] trên 60 epoch), không phải lỗi dữ liệu. Trước `[3.11]` còn phải
chọn recipe cho train thật:
- **LR 1e-4 của config chính là mức chạy được** (1e-3 suốt 30 epoch chỉ loanh quanh
  0,45) → giữ nguyên `base_lr`;
- **không kết luận được gì về tăng cường** từ `[3.10b]`: phép thử cố ý đổi hai biến
  cùng lúc (tắt tăng cường + LR ×10) — đúng như thiết kế đã ghi ở đầu config; tách
  biến là việc của Phase 4, không phải của GATE;
- **số epoch: chốt 60** (07/10/2026, quyết định cùng người dùng — chi tiết ở §4
  mục 6): đo được từ `[3.9]` là ~10,4 phút/epoch trên T4 (866,3 ms/vòng) → ~10,4
  giờ T4 / ~3 giờ A100; quyết định theo ngân sách thời gian chứ không phải theo
  phép thử overfit này.

---

### 3.16 E1 — train thật (1.445 ảnh): 60/60 epoch, best mAP 0,627 / mAP@50 0,816 (07/10/2026)

Ô `[3.11]` chạy xong ngày 07/10/2026: **thoát 0, không NaN, đủ 60/60 epoch** — val
có mặt ở cả 60 epoch, không epoch nào thiếu. Recipe đúng như đã chốt ở §4 mục 6:
60 epoch, `base_lr` 1e-4, mốc LR `[40, 55]`, warmup 1.000 vòng, AMP + tích luỹ 4
(batch hiệu dụng 8); train 1.445 ảnh (723 vòng/epoch), val 450 ảnh.

**Chạy thành 3 đoạn, có 2 lần tự resume — cả hai đều sạch:**

| Đoạn | Bắt đầu → kết thúc | Làm được gì | Dừng vì |
|---|---|---|---|
| `20261007_141146` | 14:11:46 → 14:30:49 (19 phút) | epoch 1–5 (val 1–5); dừng GIỮA epoch 6 (vòng 300/723) | phiên chạy dừng |
| `20261007_143113` | 14:31:13 → 15:05:05 (34 phút) | **resume từ `epoch_5.pth`**; epoch 6–14 (val 6–14); dừng GIỮA epoch 15 (vòng 550/723) | phiên chạy dừng |
| `20261007_150527` | 15:05:27 → 17:44:10 (2 giờ 39 phút) | **resume từ `epoch_14.pth`**; epoch 15–60 (val 15–60) | chạy hết |

Bằng chứng nằm trong log chứ không phải suy đoán: `Auto resumed from the latest
checkpoint .../epoch_5.pth` + `resumed epoch: 5, iter: 3615`, và `.../epoch_14.pth`
+ `resumed epoch: 14, iter: 10122`. Đoạn 1 → đoạn 2 cách nhau **24 giây**, đoạn 2 →
đoạn 3 cách nhau **22 giây**. Đây là **lần đầu hai miếng vá resume chạy thật trên
Colab**: §3.13 (vá `torch.load` — không có nó thì resume nổ ngay ở bước đọc
checkpoint) và §3.14 (vá LR sau resume). `vis_data/config.py` của cả ba đoạn **giống
nhau từng byte** (md5 `a18ee1cc…`) — hai lần resume chạy đúng cùng một config,
không có chuyện đoạn sau lỡ đổi tham số.

Ba điều đọc được từ log về resume:

- **Mốc LR nổ đúng sau khi resume.** `base_lr` = 1e-4 tới hết epoch 40, **1e-5 từ
  epoch 41**, **1e-6 từ epoch 56** — log chỉ in đúng ba mức đó, không lệch epoch
  nào. Sổ đếm epoch của runner sống sót qua cả hai lần resume.
- **Hai lần resume này KHÔNG kiểm chứng được miếng vá §3.14**: cả hai rơi vào vùng
  LR phẳng 1e-4 (mốc `[40, 55]` còn xa), mà ở vùng phẳng thì LR đặt đúng hay sai
  trông y hệt nhau. Miếng vá đó chỉ được kiểm chứng bằng harness Runner thật
  (§3.14). Resume lần này sạch, nhưng đừng lấy nó làm bằng chứng cho §3.14.
- mmengine cảnh báo `Resumed iteration number is not divisible by
  _accumulative_counts` ở cả hai lần resume (điểm resume không chia hết cho 4 →
  cửa sổ tích luỹ đầu tiên gộp thiếu vi-batch). Chạy tiếp trọn vẹn, không thấy hệ
  quả nào; ghi lại vì đó là dấu vết khác của resume trong log.

**Chi phí thật (A100 40 GB, đo bằng mốc thời gian trong log):** 154 s train +
~30 s val + ~23 s ghi checkpoint = **~207 s/epoch (3,45 phút)** → 60 epoch ≈ 3,5
giờ; cộng cả hai lần chạy lại thì tổng thời gian tường là 3 giờ 32 phút. Khớp ước
tính ~3 giờ A100 ở §4 mục 6. So với số đo T4 ở §3.9 (866,3 ms/vòng): A100 chạy
213 ms/vòng, **nhanh ~4 lần**.
⚠️ Đừng lấy cột `time:` trong log mmengine để tính thời gian — lần này nó in
~0,073 trong khi đo thật là 213 ms/vòng (154 s / 723 vòng).

**Kết quả val (450 ảnh):**

| Chỉ số | Tốt nhất — epoch 26 | epoch 37 (đồng hạng) | epoch 60 (cuối) |
|---|---|---|---|
| mAP (0,50:0,95) | **0,627** | 0,627 | 0,593 |
| mAP@50 | **0,816** | 0,810 | 0,751 |
| mAP@75 | 0,707 | 0,706 | 0,659 |
| mAP vật nhỏ | 0,145 | 0,174 | 0,168 |
| mAP vật vừa | 0,462 | 0,463 | 0,397 |
| mAP vật lớn | 0,703 | 0,701 | 0,682 |

`save_best='coco/bbox_mAP'` giữ lại tệp **`best_coco_bbox_mAP_epoch_26.pth`**.
Epoch 37 đạt ĐÚNG 0,6270 như epoch 26 nhưng tệp best **không** bị ghi đè — mmengine
chỉ thay tệp best khi điểm cao hơn hẳn, bằng điểm thì giữ bản cũ (quan sát từ chính
lần chạy này, qua tên tệp còn lại). Nên bản đang có là epoch 26, và đó cũng là bản
có mAP@50 cao nhất cả lần chạy (0,816 so với 0,810). Bản epoch 37 không còn:
`max_keep_ckpts=2` chỉ giữ `epoch_59.pth` + `epoch_60.pth` + tệp best — đúng 3 tệp
trong zip, ~313–320 MB mỗi tệp.

**Đường cong đã CHỮNG từ lâu, không phải đang leo.** Trung bình theo từng chục epoch:

| Epoch | 15–20 | 21–30 | 31–40 | 41–50 | 51–60 |
|---|---|---|---|---|---|
| TB mAP | 0,593 | 0,590 | 0,593 | 0,591 | **0,579** |
| TB mAP@50 | 0,789 | 0,771 | 0,767 | 0,755 | 0,733 |

Từ epoch 15 trở đi (epoch sớm nhất còn log val — hai đoạn đầu dừng trước khi kịp
val) đường cong đứng yên trong khoảng 0,58–0,63, chênh giữa các chục epoch dưới
0,015. **Hai mốc giảm LR (41 và 56) không mua được gì**: chục epoch cuối là chục
thấp nhất bảng. Dao động thì lớn: 5 epoch tụt sâu (23: 0,480; 40: 0,502; 50: 0,446;
52: 0,465; 57: 0,555) rồi hồi ngay epoch sau, không ứng với mốc LR nào. Vì thế đừng
đọc 0,627 của epoch 26 là "hơn hẳn" 0,593 của epoch 60 — với eval 450 ảnh, chênh
0,03 nằm trong nhiễu.

**Đối chiếu mốc đã đăng ký ở §2.1** (*"đừng kỳ vọng mAP@50 > 0,9; khoảng hợp lý
0,4–0,7"*): E1 ra mAP@50 **0,733–0,816** — **trên** khoảng đã đăng ký, và vẫn dưới
0,9 đúng như dự đoán. Đây là số trên val 450 ảnh của chính FloodNet, không phải chỉ
số để so với bài toán khác.

**Chỗ yếu nhất: vật nhỏ.** mAP vật lớn 0,68–0,70 nhưng vật nhỏ chỉ 0,145–0,174 —
đúng như lo ngại ở §2.1: nhà nhỏ trong ảnh 1536 px là chỗ mô hình hụt hơi. Với đầu
ra cuối cùng là **đếm** nhà, đây là con số phải nhớ khi đọc kết quả đếm: nhà to thì
đếm được, nhà nhỏ dễ sót.

**E1 so với `[3.10]` (cùng recipe, 20 ảnh overfit, mAP 0,618):** nhích hơn, đúng
chiều mong đợi. Hai số không đo cùng một tập (0,618 là val trên chính 20 ảnh train
của `[3.10]`; 0,627 là val 450 ảnh của E1) nên chỉ nên đọc là "cùng cỡ" — và cùng
cỡ ở hai bài toán khác hẳn nhau thì càng cho thấy trần của recipe này nằm quanh
0,6–0,65 chứ không phải ở dữ liệu.

**Hồ sơ:** 3 tệp log + `scalars.json` từng đoạn + 3 checkpoint nằm trong zip người
dùng tải về (`e1_cascade_convnext_t-…zip`, 859 MB, **không commit**). Số liệu ở mục
này đọc trực tiếp từ trong zip, không giải nén 900 MB checkpoint ra đĩa.

---

## 4. Việc tiếp theo

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

4. **Phase 2 — XONG, GATE 2 ĐÃ CHỐT (30/09/2026).** Số liệu chạy thật, phép đối chiếu và
   phát hiện về kích thước nhà ở §2.7; quyết định thiết kế và hai lỗi tự bắt được ở §2.6.
   Đã thêm:
   - `src/floodcount/data/resize.py` — module resize dùng chung cho cả hai phase, để
     không có chuyện Phase 1 và Phase 2 resize lệch nhau (ảnh `INTER_AREA`,
     mask **luôn** `INTER_NEAREST`)
   - `src/floodcount/data/mask_to_coco.py` + `scripts/build_coco.py`
   - `tests/test_mask_to_coco.py` — 106 assertion
   - `notebooks/02_build_coco.ipynb`
   - `build_coco:` trong `configs/data.yaml` (chất lượng JPEG, có lưu mask không, eps polygon)

   **Phép kiểm quan trọng nhất của GATE 2 — ĐÃ CHẠY VÀ ĐẠT:** chạy ô `[1.6]` của notebook 01
   **trong cùng phiên** trước ô `[2.6]`, để Phase 2 đối chiếu được với `audit.jsonl`. Kết quả
   `+0` ở cả hai lớp. Người dùng cũng đã soi 90 ảnh `overlay_gt/` và xác nhận box khớp nhà,
   không có box nào ôm nhiều căn.

   Dataset đã dựng và đóng gói: **`MyDrive/Flood_House_AI/processed/floodnet_coco.zip`**
   (1,86 GB). `audit.jsonl` và `build.jsonl` cũng đã sao lưu lên `MyDrive/Flood_House_AI/state/`
   nên phiên Colab sau không phải dựng lại gì.

5. **Phase 3 — CODE XONG, CHƯA CHẠY TRÊN COLAB (30/09/2026; rà soát lại 01/10/2026).** Config,
   script, notebook và test đã viết xong; chi tiết và các điều tra từ source ở **§3**. Ba điểm
   đáng chú ý:
   - **Dải anchor không cần sửa.** Dự đoán ban đầu là "anchor mặc định của COCO nhắm tới box
     nhỏ hơn nhiều, phải nới cho nhà to". Đọc kỹ config gốc thì **không phải vậy**: anchor
     lớn nhất đã là 724×362px, thừa sức chứa cả nhà to nhất (p90 = 348×336px). Không đổi gì
     — nhưng `scripts/kiem_anchor.py` vẫn đo lại trên box thật để báo cáo có số liệu (§3.3).
   - **Bỏ ảnh không có nhà là mặc định của mmdet**, đồ án phải chủ động tắt
     (`filter_empty_gt=False`), vì đầu ra cuối cùng là **đếm trên mọi ảnh** (§3.2e).
   - **Rà soát trước khi chạy Colab tìm được một lỗi thật** — `val_dataloader` của
     `overfit20.py` thừa hưởng nhầm `data_prefix=images/val/`. Đã sửa, khoá bằng test, và
     thêm **mục [2b]** vào tiền kiểm để bắt được cả họ lỗi này ở config khác; chi tiết ở
     **§3.6**.
   - **Lỗi thật đầu tiên trên Colab (ô `[3.9]`)**: vòng đo `--dry-run` quên build
     `optim_wrapper` — mmengine chỉ build nó trong `Runner.train()`. Đã sửa, khoá bằng
     4 phép kiểm AST (đã kiểm âm). Số phép kiểm: **455** (§3.5). Chi tiết ở **§3.8**.
   - **Lỗi thật thứ hai trên Colab (ô `[3.10]`)**: cú pháp `{{_base_.ann_overfit}}`
     trong config con hoá thành đường dẫn rác, **không lỗi nào báo**. Đã bỏ hẳn cú pháp
     đó khỏi repo (config con tự viết đường dẫn) và cấm nó bằng test ở mức cả thư mục
     `configs/`. Chi tiết ở **§3.11**.
   - **Lần chạy lại `[3.10]` ngày 07/10/2026**: bản sửa §3.11 SỐNG (tiền kiểm đạt hết,
     chạy đủ 60/60 epoch, thoát 0) nhưng mAP dừng ở **0,618** so với mốc 0,9 mà chính
     `overfit20.py` đặt ra, đuôi đường cong bão hoà → **GATE 3 chưa đạt**. Nghi phạm
     thuộc recipe (tăng cường + lịch LR bóp trong 600 vòng), không phải lỗi đã chứng
     minh. Số đo đầy đủ và phép thử phân biệt ở **§3.12**.
   - **`[3.10b]` giai đoạn 1 ngày 07/10/2026 (40 epoch)**: mAP **0,742** (mAP50 0,905)
     — vượt hẳn 0,618 của [3.10] và đuôi đường cong **vẫn leo** (hết epoch, không phải
     bão hoà); bước ngoặt nằm đúng ở mốc LR [30] (1e-3 → 1e-4). Recipe là thứ chặn,
     nhưng mốc 0,95 chưa đạt → GATE 3 vẫn chưa chốt. Số đo + cách đọc ở **§3.12**.
   - **Lỗi thật thứ ba trên Colab (ô `[3.10b]` giai đoạn 2)**: lần resume thật đầu
     tiên của đồ án nổ `_pickle.UnpicklingError` — torch ≥ 2.6 mặc định
     `weights_only=True`, checkpoint mmengine chứa `HistoryBuffer` không nằm trong
     danh sách an toàn. Chưa mất gì (lỗi ở bước đọc). Đã vá bằng
     `add_safe_globals([HistoryBuffer])` trong `scripts/train.py`, khoá bằng 5 phép
     kiểm AST (đã kiểm âm). Chi tiết ở **§3.13** — lỗi này cũng sẽ chặn resume của
     `[3.11]` nếu không vá.
   - **`[3.10b]` giai đoạn 2 ngày 07/10/2026: chạy ĐƯỢC nhưng VÔ HIỆU.** Resume
     từ epoch 40 chạy trọn 41..120 (thoát 0, mAP 0,747 → 0,779) — nhưng cả 80
     epoch train ở `base_lr: 1.0000e-06`, **sai 100 lần** so với 1e-4 mà lịch đã
     định. Không dùng số này để chốt gì (bão hoà ở LR sai không nói gì về trần
     thật). **Lỗi thật thứ tư trên Colab — lần đầu tiên IM LẶNG**: không
     exception, không mã thoát lạ, log in đúng con số sai ấy ở cả 80 dòng. Đã vá
     bằng `dat_lai_lr_sau_resume` + hook `before_train`
     (`scripts/train.py`), kiểm chứng bằng Runner THẬT (6 ca, khớp lần chạy liên
     tục từng epoch), khoá bằng 18 phép kiểm AST (đã kiểm âm). Chi tiết ở
     **§3.14**.
   - **`[3.10b]` lần chạy đúng ngày 07/10/2026 (một lần chạy MỚI trọn 120 epoch,
     thư mục `overfit20_hocvet2`): GATE 3 ĐẠT.** mAP đạt ≥ 0,95 ở 4 epoch (100,
     110, 114, 116), best **0,952** (`best_coco_bbox_mAP_epoch_110.pth`); số cuối
     epoch 120 là 0,918 (dao động của eval 20 ảnh — không phải trần); mAP50 =
     mAP75 = **1,000** ở cả hai lớp; AP vật nhỏ 0,900. Đường ống đã được chứng
     minh — thứ chặn `[3.10]` là recipe, không phải lỗi dữ liệu. Số đo đầy đủ ở
     **§3.15**.
6. **Recipe của `[3.11]` (Phase 4 — train thật trên 2.343 ảnh) ĐÃ CHỐT ngày
   07/10/2026, VÀ ĐÃ CHẠY XONG cùng ngày.** Kết quả: 60/60 epoch, thoát 0,
   **best mAP 0,627 / mAP@50 0,816** (epoch 26, tệp `best_coco_bbox_mAP_epoch_26.pth`),
   mAP cuối 0,593; 2 lần tự resume giữa chừng đều sạch; 3 giờ 32 phút trên A100.
   Số đầy đủ + cách đọc ở **§3.16**. Ba mục recipe dưới đây giữ nguyên làm hồ sơ
   (đã chốt TRƯỚC khi chạy, căn cứ §3.15):
   - **số epoch: 60** (quyết định cùng người dùng; dự kiến chạy trên A100). Căn
     cứ: `[3.9]` đo 10,4 phút/epoch trên T4 → 60 epoch ≈ 10,4 giờ T4 / ~3 giờ
     A100 (ước tính — `[3.9]` in số thật trên máy đang chạy); checkpoint tốt nhất
     chọn theo VAL (`save_best`) nên epoch dư chỉ tốn thời gian, không làm hỏng
     số báo cáo; 60 epoch ở batch hiệu dụng 8 ≈ 10.900 bước cập nhật — đủ cho
     lịch LR anneal hết một chu kỳ. Con số 24 trước đây là mặc định TẠM, chưa
     từng được chốt.
   - **lịch LR**: giữ `base_lr` 1e-4 (bằng chứng: 1e-3 chỉ loanh quanh 0,45 sau
     30 epoch); mốc giảm LR [40, 55] giữ đúng tỉ lệ cũ (2/3 và ~11/12, tức 66,7%
     và 91,7%) và trùng luôn lịch mà `overfit20.py` đã chạy ở GATE 3. Ô `[3.1]`
     đã để `SO_EPOCH = 60` và config chính cũng vậy; `scripts/train.py` tự giãn
     mốc theo cùng tỉ lệ nếu sau này đổi số epoch.
   - **tăng cường**: baseline E1 giữ nguyên recipe hiện tại (GATE 3 không tách
     được biến này — `[3.10b]` cố ý đổi hai thứ cùng lúc); tách biến là việc của
     các thí nghiệm E2–E7. Muốn đổi số epoch cho một thí nghiệm thì phải ghi lại
     lý do: E2–E7 so với E1 nên phải chạy cùng recipe.

   Ô `[3.11]` **tự resume** nếu `work_dir` trên Drive đã có checkpoint (cả hai lỗi
   resume đã vá: §3.13 và §3.14) — điều này đã được kiểm chứng THẬT trong chính lần
   chạy E1: hai lần dừng giữa chừng, hai lần chạy lại ăn ngay từ checkpoint, xem
   §3.16. Ô `[3.3b]` tự cài môi trường (gọi `scripts/cai_moi_truong.py`, §1.9) nên
   **không phải mở notebook 00 ở tab thứ hai**. Phải **push lên GitHub trước** —
   Colab clone code từ GitHub, chưa push là Colab chạy đúng bản cũ.
   Không đụng tới `floodnet_raw.zip` 13 GB nữa — chỉ cần `floodnet_coco.zip` 1,86 GB.

7. **Dùng mô hình: công cụ thử trên ảnh thật (viết xong 07–08/10/2026, chờ push để
   Colab chạy).** Sau khi E1 xong, việc tiếp theo là NHÌN mô hình đếm trên ảnh —
   mAP không nói gì về chuyện đếm. Đã thêm:
   - `src/floodcount/infer/dem.py` — logic thuần Python, test được trên máy CPU:
     đếm theo lớp + ngưỡng điểm, đối chiếu nhãn thật từ COCO, chọn ảnh demo (một
     nửa có nhà ngập), tìm checkpoint best trong `work_dir`.
   - `scripts/du_doan.py` — CLI hai chế độ: `--anh-dir` (ảnh của người dùng, không
     có nhãn nên chỉ in số mô hình đếm) và demo trên một split của dataset (**có**
     đối chiếu nhãn thật — cách duy nhất để biết con số đếm có đáng tin). Vẽ overlay
     (box đỏ = ngập) và ghi `du_doan.json`.
   - `scripts/web_du_doan.py` — trang web Gradio: kéo ảnh vào → ảnh đã vẽ box + số
     đếm, kèm thanh trượt ngưỡng điểm; `--share` cho link công khai mở được trên
     điện thoại. Lõi suy luận dùng lại nguyên `du_doan.py`.
   - `notebooks/03_train.ipynb` — ô `[3.12]` (thả ảnh vào
     `MyDrive/Flood_House_AI/anh_cua_toi/` rồi chạy, overlay hiện ngay trong ô) và
     ô `[3.13]` (mở trang web, link sống theo phiên Colab).
   - `tests/test_du_doan.py` — 68 phép kiểm, chạy trên máy không cài
     mmdet/gradio/torch.

   Việc còn lại: push → chạy `[3.12]` và/hoặc `[3.13]` trên Colab → xem số đếm trên
   ảnh thật. Nhắc lại giới hạn đã ghi ở §2.1 và trong ô `[3.12]`: mô hình học ảnh
   chụp **từ trên cao**; ảnh chụp ngang tầm mắt sẽ kém, và đó là giới hạn dữ liệu
   chứ không phải lỗi code.

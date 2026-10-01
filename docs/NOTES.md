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
| `test_train.py` | 47 |
| `test_kiem_tra.py` | 41 |
| `test_anchor.py` | 39 |
| `test_overfit.py` | 37 |
| `test_photometric.py` | 21 |
| **Tổng** | **453** |

Cả 8 bộ **PASS** ngày 01/10/2026. `test_train.py` chạy được cả trên máy sạch
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
xác nhận khớp). Còn lại: `[3.9]` `train.py --dry-run` — chạy lần đầu hỏng, xem
§3.8 — rồi `[3.10]` train overfit 20 ảnh, **dừng và báo cáo lại**. Train thật
(Phase 4) chỉ bắt đầu sau khi chốt số epoch từ phép đo `--dry-run`.

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
     4 phép kiểm AST (đã kiểm âm). Số phép kiểm: **453** (§3.5). Chi tiết ở **§3.8**.
6. **Bước kế tiếp ngay: chạy `notebooks/03_train.ipynb` trên Colab T4 (Phase 3) — MỘT tab.**
   Ô `[3.3b]` tự cài môi trường (gọi `scripts/cai_moi_truong.py`, §1.9) nên **không phải mở
   notebook 00 ở tab thứ hai**. Thứ tự các ô và thời gian dự kiến ghi trong README. Dừng ở
   phép thử overfit 20 ảnh và báo cáo lại (số epoch, loss, thời gian một epoch đo được ở ô
   `--dry-run`) trước khi sang Phase 4.
   Không đụng tới `floodnet_raw.zip` 13 GB nữa — chỉ cần `floodnet_coco.zip` 1,86 GB.

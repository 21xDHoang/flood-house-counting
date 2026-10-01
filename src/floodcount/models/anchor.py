# -*- coding: utf-8 -*-
"""Đo xem dải anchor của RPN có phủ được kích thước nhà của FloodNet không.

VÌ SAO CẦN ĐO
Phase 2 đo được box nhà có cạnh TRUNG VỊ 186px và p90 = 348px, trên ảnh chỉ
rộng 1536px. RPN mặc định của MMDetection sinh anchor theo công thức cố định,
và nếu dải anchor không phủ tới cỡ đó thì những căn nhà to nhất sẽ không bao giờ
khớp được với anchor nào — model vẫn chạy, loss vẫn giảm, nhưng recall thấp một
cách khó hiểu. Đây là thứ phải kiểm bằng SỐ chứ không được đoán.

SỐ CỦA BỘ ANCHOR ĐANG DÙNG (đọc từ config gốc mmdet 3.3.0, không suy đoán)
`configs/_base_/models/cascade-mask-rcnn_r50_fpn.py` khai báo RPN::

    anchor_generator=dict(type='AnchorGenerator',
                          scales=[8], ratios=[0.5, 1.0, 2.0],
                          strides=[4, 8, 16, 32, 64])

Không có `base_sizes`, mà `AnchorGenerator.__init__` viết::

    self.base_sizes = [min(stride) for stride in self.strides] \\
        if base_sizes is None else base_sizes

nên `base_sizes = [4, 8, 16, 32, 64]` và cỡ anchor mỗi tầng là
`base_size * scale` = **32, 64, 128, 256, 512**. Với mỗi cỡ lại có 3 tỉ lệ, và
theo `gen_single_level_base_anchors` thì ``w = base_size / sqrt(ratio)``,
``h = base_size * sqrt(ratio)`` — nên tỉ lệ 0.5 cho anchor RỘNG (w = 512 x 1,414
= 724 khi cỡ 512) và tỉ lệ 2.0 cho anchor CAO. Tổng cộng 5 tầng x 3 tỉ lệ
= 15 hình dạng anchor cho mỗi vị trí.

HAI PHÉP ĐO, KHÁC NHAU Ở CHỖ CÓ TÍNH VỊ TRÍ HAY KHÔNG
1. `iou_cung_tam` — đặt anchor cùng tâm với box rồi tính IoU. Đây là GIỚI HẠN
   TRÊN: anchor thật nằm trên lưới của feature map nên tâm lệch tối đa nửa
   stride, không bao giờ trùng tâm box một cách hoàn hảo. Con số này dùng để
   trả lời "hình dạng anchor có hợp với hình dạng nhà không".
2. `do_phu_thuc` — với mỗi box, tìm ĐÚNG anchor gần tâm box nhất trên lưới của
   từng tầng rồi tính IoU. Đây là con số SÁT với thực tế train, vì nó tính cả
   phần lệch tâm. Con số này dùng để trả lời "RPN có thể đạt IoU bao nhiêu".

Hai con số này chặn trên và chặn dưới cho nhau: `do_phu_thuc <= iou_cung_tam`,
và test có kiểm đúng bất đẳng thức đó. Cách đo thứ hai vẫn là một cận trên của
IoU thật lúc train, vì lưới anchor trải khắp ảnh còn nhiều hơn những gì ta xét
(anchor tràn ra ngoài mép ảnh), nhưng phần chênh đó nhỏ và không đổi kết luận.

Module này CHỈ dùng numpy — chạy được trên máy CPU không có MMDetection, và có
test riêng ở `tests/test_anchor.py`. Nó KHÔNG import mmdet; tham số truyền vào
là các con số đọc từ config.
"""

import math

import numpy as np

# Giá trị của config gốc mmdet 3.3.0, khai báo ở đây để script và test dùng
# chung một nguồn. Nếu config đồ án đổi RPN thì phải sửa cả hai nơi —
# `scripts/kiem_anchor.py` đọc lại các giá trị này từ chính file config đồ án và
# báo lỗi nếu lệch, nên không thể lệch âm thầm.
STRIDES_MAC_DINH = (4, 8, 16, 32, 64)
SCALES_MAC_DINH = (8,)
RATIOS_MAC_DINH = (0.5, 1.0, 2.0)


def canh_nho(stride):
    """Cạnh nhỏ của một stride — dùng chung cho mọi chỗ cần `min(stride)`.

    `AnchorGenerator` cho phép stride là số nguyên (4) hoặc cặp (4, 4), và quy
    tắc của nó là lấy cạnh nhỏ. Viết `min(stride)` trực tiếp sẽ nổ TypeError khi
    stride là số nguyên, nên phải rẽ nhánh ở một chỗ duy nhất.
    """
    if isinstance(stride, (tuple, list)):
        return int(min(stride))
    return int(stride)


def kich_thuoc_anchor(strides=STRIDES_MAC_DINH,
                      scales=SCALES_MAC_DINH,
                      ratios=RATIOS_MAC_DINH,
                      base_sizes=None):
    """Sinh danh sách (w, h) của mọi hình dạng anchor.

    Bám đúng công thức của `AnchorGenerator.gen_single_level_base_anchors`:
    ``w = base_size * (1/sqrt(ratio)) * scale``, ``h = base_size * sqrt(ratio)
    * scale``, với ``base_size = min(stride)`` khi không truyền `base_sizes`.

    Returns:
        list[tuple[float, float]]: mỗi phần tử là (w, h) tính bằng pixel.
    """
    if base_sizes is None:
        base_sizes = [canh_nho(s) for s in strides]
    ra = []
    for base_size in base_sizes:
        for ratio in ratios:
            can = math.sqrt(ratio)
            for scale in scales:
                ra.append((base_size / can * scale, base_size * can * scale))
    return ra


def iou_cung_tam(box_wh, anchor_whs):
    """IoU của một box với nhiều anchor khi ĐẶT CÙNG TÂM.

    Công thức: giao = min(w1,w2) * min(h1,h2); hợp = w1*h1 + w2*h2 - giao.
    Không cần toạ độ vì hai hình chữ nhật cùng tâm.

    Args:
        box_wh (tuple): (w, h) của box.
        anchor_whs (list): danh sách (w, h) của anchor.

    Returns:
        numpy.ndarray: IoU với từng anchor, cùng thứ tự với `anchor_whs`.
    """
    a = np.asarray(anchor_whs, dtype=np.float64)
    bw, bh = float(box_wh[0]), float(box_wh[1])
    giao = np.minimum(bw, a[:, 0]) * np.minimum(bh, a[:, 1])
    hop = bw * bh + a[:, 0] * a[:, 1] - giao
    # Chia cho 0 không xảy ra vì kích thước box/anchor đều > 0; vẫn chặn để nếu
    # ai truyền vào box rỗng thì ra 0 chứ không ra nan rồi lan ra cả bảng.
    return np.divide(giao, hop, out=np.zeros_like(giao), where=hop > 0)


def do_phu_thuc(boxes_xywh, strides=STRIDES_MAC_DINH,
                scales=SCALES_MAC_DINH, ratios=RATIOS_MAC_DINH,
                base_sizes=None, chunk=2048):
    """IoU lớn nhất đạt được với lưới anchor THẬT, cho từng box.

    Với mỗi box và mỗi tầng, anchor gần tâm box nhất nằm ở
    ``round(tâm / stride) * stride``, nên tâm lệch tối đa nửa stride mỗi trục.
    Lấy IoU lớn nhất qua tất cả các tầng và tỉ lệ.

    Args:
        boxes_xywh (numpy.ndarray): mảng (N, 4) theo định dạng COCO
            ``[x, y, w, h]``, toạ độ trong hệ của ảnh lúc train.
        chunk (int): Số box xử lý mỗi lượt, để bộ nhớ không phụ thuộc số box.

    Returns:
        tuple: ``(iou_max, chi_so_anchor)`` — mảng (N,) IoU lớn nhất và mảng
        (N,) chỉ số của hình dạng anchor đạt IoU đó (theo thứ tự của
        `kich_thuoc_anchor`).
    """
    boxes = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
    hinh_dang = kich_thuoc_anchor(strides, scales, ratios, base_sizes)
    a = np.asarray(hinh_dang, dtype=np.float64)          # (M, 2) -> (w, h)
    a_w = a[None, :, 0]                                  # (1, M)
    a_h = a[None, :, 1]                                  # (1, M)
    leo_thang = [canh_nho(s) for s in strides]

    n = len(boxes)
    iou_max = np.zeros(n, dtype=np.float64)
    chi_so = np.zeros(n, dtype=np.int64)

    for bat_dau in range(0, n, chunk):
        phan = boxes[bat_dau:bat_dau + chunk]
        cx = phan[:, 0] + phan[:, 2] / 2.0               # (n_phan,)
        cy = phan[:, 1] + phan[:, 3] / 2.0
        bw = phan[:, 2:3]                                # (n_phan, 1)
        bh = phan[:, 3:4]
        x1 = phan[:, 0:1]
        y1 = phan[:, 1:2]
        x2 = x1 + bw
        y2 = y1 + bh

        for s in leo_thang:
            # Tâm anchor gần nhất trên lưới của tầng này: lưới là 0, s, 2s...
            # nên làm tròn về bội số gần nhất của s, mỗi trục lệch tối đa s/2.
            #
            # Chú ý hình dạng mảng: box là (n_phan, 1) còn anchor là (1, M), nên
            # phép trừ cho ra (n_phan, M) — mỗi dòng là một box, mỗi cột là một
            # hình dạng anchor. Thêm một trục nữa (thành (n, 1, M)) sẽ làm
            # argmax(axis=1) luôn trả về 0 và cả bảng kết quả sai im lặng.
            ax = (np.round(cx / s) * s)[:, None]         # (n_phan, 1)
            ay = (np.round(cy / s) * s)[:, None]

            giao_w = np.clip(np.minimum(x2, ax + a_w / 2.0)
                             - np.maximum(x1, ax - a_w / 2.0), 0.0, None)
            giao_h = np.clip(np.minimum(y2, ay + a_h / 2.0)
                             - np.maximum(y1, ay - a_h / 2.0), 0.0, None)
            giao = giao_w * giao_h                       # (n_phan, M)
            hop = bw * bh + a_w * a_h - giao
            iou_tang = np.divide(giao, hop, out=np.zeros_like(giao),
                                 where=hop > 0)

            chi_so_tot = np.argmax(iou_tang, axis=1)     # (n_phan,)
            iou_tot = iou_tang[np.arange(len(phan)), chi_so_tot]

            # Giữ lại nếu tầng này tốt hơn những tầng đã xét trước đó.
            do_hon = iou_tot > iou_max[bat_dau:bat_dau + chunk]
            iou_max[bat_dau:bat_dau + chunk] = np.where(
                do_hon, iou_tot, iou_max[bat_dau:bat_dau + chunk])
            chi_so[bat_dau:bat_dau + chunk] = np.where(
                do_hon, chi_so_tot, chi_so[bat_dau:bat_dau + chunk])

    return iou_max, chi_so


def phan_tich(boxes_xywh, strides=STRIDES_MAC_DINH, scales=SCALES_MAC_DINH,
              ratios=RATIOS_MAC_DINH, base_sizes=None,
              cac_nguong=(0.3, 0.5, 0.7), nhan=None):
    """Bảng phân tích độ phủ anchor cho một tập box.

    Args:
        boxes_xywh (numpy.ndarray): mảng (N, 4) COCO `[x, y, w, h]`.
        cac_nguong (tuple): Các mốc IoU cần đếm tỉ lệ đạt được.
        nhan (numpy.ndarray | None): mảng (N,) nhãn lớp, nếu muốn tách bảng
            theo từng lớp. Tên lớp do người gọi truyền kèm ở `ten_lop`.

    Returns:
        dict: gồm `so_box`, `hinh_dang`, `iou_cung_tam`, `iou_thuc`,
        `ti_le_dat` (theo `cac_nguong`), `dem_theo_nguong`, `best_anchor`
        (đếm số box thắng ở mỗi hình dạng anchor), và `theo_lop` nếu có `nhan`.
    """
    boxes = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
    hinh_dang = kich_thuoc_anchor(strides, scales, ratios, base_sizes)

    # IoU cùng tâm: vector hoá theo lô để không dựng mảng (N, M) cho cả dataset.
    iou_ct = np.empty(len(boxes), dtype=np.float64)
    for i, (_, _, w, h) in enumerate(boxes):
        iou_ct[i] = float(np.max(iou_cung_tam((w, h), hinh_dang)))

    iou_thuc, chi_so = do_phu_thuc(boxes, strides, scales, ratios, base_sizes)

    def _thong_ke(mang):
        ra = {
            "so_box": int(len(mang)),
            "nho_nhat": float(np.min(mang)) if len(mang) else float("nan"),
            "p5": float(np.percentile(mang, 5)) if len(mang) else float("nan"),
            "trung_vi": float(np.median(mang)) if len(mang) else float("nan"),
            "p95": float(np.percentile(mang, 95)) if len(mang) else float("nan"),
            "lon_nhat": float(np.max(mang)) if len(mang) else float("nan"),
        }
        for nguong in cac_nguong:
            ra[f"ti_le_iou>={nguong}"] = (
                float(np.mean(mang >= nguong)) if len(mang) else float("nan"))
        return ra

    ra = {
        "so_box": int(len(boxes)),
        "hinh_dang": hinh_dang,
        "iou_cung_tam": _thong_ke(iou_ct),
        "iou_thuc": _thong_ke(iou_thuc),
        "dem_best_anchor": np.bincount(chi_so, minlength=len(hinh_dang)).tolist()
        if len(boxes) else [0] * len(hinh_dang),
        "canh_box": {
            "trung_vi": float(np.median(np.minimum(boxes[:, 2], boxes[:, 3])))
            if len(boxes) else float("nan"),
            "p95_canh_lon": float(
                np.percentile(np.maximum(boxes[:, 2], boxes[:, 3]), 95))
            if len(boxes) else float("nan"),
        },
    }

    if nhan is not None:
        nhan = np.asarray(nhan).reshape(-1)
        if len(nhan) != len(boxes):
            raise ValueError(
                f"số nhãn ({len(nhan)}) phải bằng số box ({len(boxes)})")
        ra["theo_lop"] = {
            int(lop): {
                "so_box": int(np.sum(nhan == lop)),
                "iou_thuc": _thong_ke(iou_thuc[nhan == lop]),
            }
            for lop in np.unique(nhan)
        }
    return ra


def bao_cao(ket_qua, ten_lop=None):
    """Đổi dict của `phan_tich` thành văn bản để in trong notebook."""
    dong = []
    them = dong.append
    ct = ket_qua["iou_cung_tam"]
    th = ket_qua["iou_thuc"]

    them(f"  Tổng số box phân tích : {ket_qua['so_box']}")
    them(f"  Cạnh TRUNG VỊ của box : {ket_qua['canh_box']['trung_vi']:.0f}px"
         f"   |  p95 cạnh lớn hơn: {ket_qua['canh_box']['p95_canh_lon']:.0f}px")
    them("")
    them("  Hình dạng anchor đang có (w x h, đơn vị pixel):")
    for i, (w, h) in enumerate(ket_qua["hinh_dang"]):
        dem = ket_qua["dem_best_anchor"][i]
        them(f"    [{i:2d}] {w:7.1f} x {h:7.1f}    "
             f"là anchor tốt nhất cho {dem:5d} box")
    them("")
    them("  IoU lớn nhất đạt được (theo lưới anchor thật):")
    them(f"    nhỏ nhất {th['nho_nhat']:.3f} | p5 {th['p5']:.3f} | "
         f"trung vị {th['trung_vi']:.3f} | p95 {th['p95']:.3f} | "
         f"lớn nhất {th['lon_nhat']:.3f}")
    for khoa in sorted(k for k in th if k.startswith("ti_le_iou>=")):
        them(f"    {khoa:>18s}: {th[khoa] * 100:5.1f}% số box")
    them("")
    them("  (Giới hạn trên, khi đặt anchor cùng tâm box — để so sánh "
         "hình dạng:)")
    them(f"    trung vị {ct['trung_vi']:.3f}")
    for khoa in sorted(k for k in ct if k.startswith("ti_le_iou>=")):
        them(f"    {khoa:>18s}: {ct[khoa] * 100:5.1f}% số box")

    if "theo_lop" in ket_qua:
        them("")
        them("  Tách theo lớp:")
        for lop, so in sorted(ket_qua["theo_lop"].items()):
            ten = (ten_lop or {}).get(lop, f"lớp {lop}")
            them(f"    {ten:24s} {so['so_box']:5d} box | "
                 f"trung vị IoU {so['iou_thuc']['trung_vi']:.3f} | "
                 f"đạt 0.5: {so['iou_thuc'].get('ti_le_iou>=0.5', float('nan')) * 100:5.1f}%"
                 )
    return "\n".join(dong)

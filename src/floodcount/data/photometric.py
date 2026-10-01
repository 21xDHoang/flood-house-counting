# -*- coding: utf-8 -*-
"""Tăng/giảm sáng và tương phản nhẹ — bản thay thế `PhotoMetricDistortion`.

VÌ SAO KHÔNG DÙNG `PhotoMetricDistortion` CÓ SẴN CỦA MMDET
Đã đọc source mmdet 3.3.0 (`mmdet/datasets/transforms/transforms.py`). Hàm
`PhotoMetricDistortion.transform()` kết thúc bằng::

    if swap_flag:
        img = img[..., swap_value]

tức nó ĐẢO KÊNH MÀU ngẫu nhiên với xác suất 1/2 (`swap_value` là một hoán vị
ngẫu nhiên của ba kênh B, G, R). Với bài toán này thì đó là phá đúng tín hiệu
cần học: màu nước là manh mối chính để phân biệt nhà NGẬP với nhà KHÔNG ngập
(nước đục màu bùn, mái tôn sáng, đường bê tông xám). Đảo kênh biến nước xanh
thành nước đỏ, và vì việc đó chỉ xảy ra ở một nửa số ảnh nên model không thể
học được quy luật nào ổn định từ màu nữa. Ảnh UAV càng khó bù, vì ngoài màu ra
gần như không còn manh mối rẻ tiền nào khác.

`PhotoMetricDistortion` còn bật cùng lúc bốn phép (sáng, tương phản, bão hoà,
hue) với bốn xác suất riêng — quá nhiều thay đổi cho một dataset mà tín hiệu
nằm ở MÀU. Ở đây chỉ giữ hai phép không đụng tới màu: độ sáng và tương phản.

CÁCH LÀM
Cường độ mới: ``y = (x - mean) * alpha + mean + delta``, rồi cắt về [0, 255].

  * ``delta`` (độ sáng) cộng đều vào mọi điểm ảnh — mô phỏng nắng gắt / trời râm.
  * ``alpha`` (tương phản) kéo giãn quanh GIÁ TRỊ TRUNG BÌNH của ảnh, không
    phải quanh 0. Kéo quanh 0 sẽ kèm theo làm ảnh sáng lên hoặc tối đi, tức là
    trộn hai phép vào nhau và không còn giải thích được phép nào làm gì.

KHÔNG dùng ``cv2.convertScaleAbs`` cho việc này: hàm đó lấy TRỊ TUYỆT ĐỐI của
kết quả trước khi cắt về 8 bit, nên một pixel bị đẩy xuống dưới 0 sẽ thành một
pixel SÁNG (giá trị dương) thay vì bị cắt về 0 — đúng kiểu lỗi im lặng: ảnh vẫn
ra bình thường, chỉ có vài vùng bị đảo sáng tối.

Module này CỐ Ý chỉ dùng numpy, không import mmdet/mmcv/mmengine: nhờ vậy nó
chạy và được kiểm thử trên máy CPU không có MMDetection. Phần nối vào pipeline
của mmdet nằm ở `src/floodcount/models/transforms.py`.
"""

import numpy as np


def chon_tham_so(delta_sang, tuong_phan, rng=None):
    """Bốc ngẫu nhiên một cặp (delta, alpha) trong khoảng cho phép.

    Args:
        delta_sang (float): Biên độ độ sáng, bốc đều trong
            ``[-delta_sang, +delta_sang]``.
        tuong_phan (tuple): Cặp ``(nhỏ nhất, lớn nhất)`` cho alpha.
        rng: Đối tượng có phương thức ``uniform`` — truyền
            ``numpy.random.default_rng(seed)`` khi cần tái lập. Mặc định dùng
            ``numpy.random`` (giống các transform khác của mmdet).

    Returns:
        tuple: ``(delta, alpha)``.
    """
    if rng is None:
        rng = np.random
    delta = float(rng.uniform(-delta_sang, delta_sang))
    alpha = float(rng.uniform(tuong_phan[0], tuong_phan[1]))
    return delta, alpha


def tang_sang_nhe(anh, delta, alpha):
    """Trả về ảnh đã chỉnh sáng/tương phản. KHÔNG sửa mảng đầu vào.

    Args:
        anh (numpy.ndarray): Ảnh uint8, 2 hoặc 3 chiều (BGR hoặc xám).
        delta (float): Cộng vào mọi điểm ảnh.
        alpha (float): Hệ số kéo giãn quanh trung bình ảnh.

    Returns:
        numpy.ndarray: Ảnh uint8 cùng kích thước, dtype và số kênh.
    """
    if anh.dtype != np.uint8:
        raise TypeError(f"ảnh phải là uint8, nhận được {anh.dtype}")

    # Tính bằng float32 để phép nhân không bị tràn số, rồi mới cắt về [0, 255].
    # `np.clip` trước `astype` là bắt buộc: astype(uint8) một mình sẽ lấy phần
    # dư (modulo 256) — pixel 260 thành 4, tức đốm trắng thành đốm đen.
    ra = anh.astype(np.float32)
    trung_binh = float(ra.mean())
    ra = (ra - trung_binh) * alpha + trung_binh + delta
    return np.clip(ra, 0.0, 255.0).astype(np.uint8)

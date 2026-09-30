# -*- coding: utf-8 -*-
"""Tiền xử lý kích thước ảnh/mask — dùng chung cho Phase 1 và Phase 2.

VÌ SAO TÁCH RIÊNG THÀNH MODULE: công thức resize phải GIỐNG HỆT nhau ở hai nơi.
Phase 1 (`audit.py`) đo kích thước nhà SAU RESIZE để quyết định có cắt tile
không; Phase 2 (`mask_to_coco.py`) tạo ra chính những ảnh đó. Hai công thức lệch
nhau dù chỉ một pixel cũng làm số box của Phase 2 không khớp con số Phase 1 đã
đo — và lệch kiểu đó rất khó lần ra, vì nhìn vào thì cả hai bên đều "đúng".
Để chung một hàm thì không thể lệch được nữa.

HAI QUY TẮC KHÔNG ĐƯỢC ĐỔI:

  * ẢNH  -> INTER_AREA khi thu nhỏ, INTER_LINEAR khi phóng to.
            Ảnh UAV gốc 4000x3000 thu về 1536 là THU NHỎ 2,6 lần. Với downsample
            thì INTER_AREA là phép lấy trung bình vùng — đúng về mặt mẫu. Dùng
            INTER_NEAREST cho ảnh sẽ vứt đi 6 trong 7 pixel, ảnh bị răng cưa nặng,
            và mọi chi tiết nhỏ (lan can, mép mái) đều có thể biến mất.

  * MASK -> LUÔN INTER_NEAREST, kể cả khi phóng to.
            Mask là ẢNH NHÃN: mỗi pixel là một số nguyên 0..9, không phải một độ
            sáng. Nội suy tuyến tính giữa lớp 2 và lớp 3 sẽ sinh ra giá trị 2.4,
            rồi `mask == 2` không khớp pixel nào nữa — mất nhà MÀ KHÔNG HỀ BÁO
            LỖI, vì mảng vẫn đúng kiểu dữ liệu và vẫn đúng kích thước.
"""

import cv2
import numpy as np


def kich_thuoc_sau_resize(h, w, target_long_side):
    """Kích thước (h, w) sau khi đưa CẠNH DÀI về `target_long_side`, giữ tỉ lệ.

    Giữ đúng công thức mà Phase 1 đã dùng để đo (NOTES muc 2.4). Đừng "cải tiến"
    bằng cách làm tròn kiểu khác: hai cỡ ảnh gốc 4000x3000 và 4592x3072 cho ra
    hai tỉ lệ resize khác nhau (0,3840 và 0,3345), và chính con số đó được dùng
    để giải thích vì sao phải bật augmentation đổi tỉ lệ ở Phase 3.

    Làm tròn bằng `round()` của Python (làm tròn chẵn) chứ không phải int() trần:
    int() cắt cụt, làm ảnh thấp đi 1 pixel một cách có hệ thống.
    """
    if max(h, w) <= 0:
        raise ValueError(f"kích thước ảnh vô lý: {h}x{w}")
    ty_le = target_long_side / max(h, w)
    return int(round(h * ty_le)), int(round(w * ty_le))


def resize_anh(anh, kich_thuoc):
    """Resize ẢNH MÀU (hoặc ảnh xám) về `kich_thuoc` = (h, w).

    Chọn phép nội suy theo CHIỀU: thu nhỏ thì INTER_AREA, phóng to thì
    INTER_LINEAR. Chọn cứng một phép cho cả hai chiều là sai một trong hai.
    """
    h2, w2 = kich_thuoc
    h, w = anh.shape[:2]
    if (h, w) == (h2, w2):
        return anh
    noi_suy = cv2.INTER_AREA if (h2 * w2) < (h * w) else cv2.INTER_LINEAR
    return cv2.resize(anh, (w2, h2), interpolation=noi_suy)


def resize_mask(mask, kich_thuoc):
    """Resize MASK NHÃN về `kich_thuoc` = (h, w). Luôn INTER_NEAREST."""
    h2, w2 = kich_thuoc
    h, w = mask.shape[:2]
    if (h, w) == (h2, w2):
        return mask
    return cv2.resize(mask, (w2, h2), interpolation=cv2.INTER_NEAREST)


def ghi_anh(duong_dan, anh, tham_so=None):
    """Ghi ảnh ra đĩa, trả về True/False. Dùng `imencode` thay cho `cv2.imwrite`.

    `cv2.imwrite` KHÔNG ghi được ra đường dẫn có ký tự ngoài ASCII trên Windows —
    nó trả về False trong im lặng, không ném lỗi. Đường dẫn đồ án này có dấu
    ("nhân diện nhà ngập"), nên nếu sau này có ai chạy Phase 2 trên máy Windows
    thì `imwrite` sẽ mất ảnh mà không ai biết vì sao. `imencode` + ghi nhị phân
    thì không phụ thuộc vào bảng mã của hệ điều hành.
    """
    ext = "." + duong_dan.rsplit(".", 1)[-1].lower()
    ok, buf = cv2.imencode(ext, anh, tham_so or [])
    if not ok:
        return False
    with open(duong_dan, "wb") as f:
        f.write(buf.tobytes())
    return True

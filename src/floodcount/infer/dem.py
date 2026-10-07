# -*- coding: utf-8 -*-
"""Logic thuần Python cho bước suy luận: đếm, đối chiếu nhãn, chọn ảnh.

Tách khỏi `scripts/du_doan.py` vì hai lý do — cùng lý do đã tách
`floodcount.data.kiem_tra` khỏi `scripts/train.py`:

  1. Máy Windows (CPU) không cài mmdet vẫn test THẬT được các hàm dưới đây,
     bằng dữ liệu giả dựng ngay trong test — thay vì chờ tới lúc chạy Colab mới
     biết sai.
  2. Đếm là ĐẦU RA CUỐI CÙNG của đồ án: mọi thứ phía trước (dữ liệu, train,
     chọn checkpoint) đều để phục vụ con số này. Sai ở đây thì không có bảng
     mAP nào phát hiện hộ.

Quy ước xuyên suốt: `labels` của mmdet là chỉ số lớp **0-based** (0 =
`flooded_building`), còn `category_id` trong file COCO JSON do Phase 2 dựng là
**1-based** (theo giá trị mask: 1 = flooded, 2 = non-flooded). Hai hệ khác nhau
nên mọi chỗ đổi qua lại đều đi qua đúng một hàm, không tự trừ 1 rải rác.
"""

import os
import pathlib
import random
import re

# Đuôi tệp mà OpenCV đọc được — dùng cho chế độ "ảnh của tôi". Cố ý KHÔNG nhận
# GIF: cv2.imread trả None im lặng với ảnh động, người dùng chỉ thấy "0 ảnh"
# mà không hiểu vì sao.
DUOI_ANH = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ten_lop_theo_id(coco):
    """`{category_id: tên lớp}` đọc từ chính file JSON, không đoán.

    Đọc từ JSON chứ không lấy từ `data.yaml`: nếu hai nguồn lệch nhau thì đó là
    lỗi cần lộ ra (script gọi sẽ đối chiếu), không phải lỗi để hàm này che đi.
    """
    ra = {}
    for c in coco.get("categories") or []:
        ra[int(c["id"])] = c["name"]
    return ra


def chi_muc_gt(coco):
    """`{file_name: {tên lớp: số box}}` — số nhãn THẬT của từng ảnh.

    Ảnh không có box nào KHÔNG xuất hiện trong từ điển; chỗ dùng tra bằng
    `.get(ten_anh, {})` để ra 0, không phải xử lý hai trường hợp.
    """
    ten_theo_id = ten_lop_theo_id(coco)
    ten_anh = {int(im["id"]): im["file_name"] for im in coco.get("images") or []}

    ra = {}
    for ann in coco.get("annotations") or []:
        ten_anh_cua_ann = ten_anh.get(int(ann["image_id"]))
        if ten_anh_cua_ann is None:
            # Annotation trỏ tới image_id không tồn tại — file hỏng. Bỏ qua ở
            # đây thì số đếm nhãn thật THẤP giả, và bảng đối chiếu sẽ đổ oan
            # cho mô hình. Ném ra để lộ chứ không im lặng.
            raise ValueError(
                f"Annotation có image_id={ann['image_id']} nhưng không có ảnh "
                f"nào mang id đó trong mục `images` — file COCO không nhất quán.")
        lop = ten_theo_id.get(int(ann["category_id"]))
        if lop is None:
            raise ValueError(
                f"Annotation có category_id={ann['category_id']} không nằm "
                f"trong `categories` {sorted(ten_theo_id)} — file COCO không "
                f"nhất quán.")
        dem = ra.setdefault(ten_anh_cua_ann, {})
        dem[lop] = dem.get(lop, 0) + 1
    return ra


def dem_theo_lop(labels, scores, ten_lop, nguong_diem):
    """Đếm box có điểm `>= nguong_diem` theo từng lớp. Trả `{tên lớp: số box}`.

    Ngưỡng so bằng `>=` (lấy luôn box đúng bằng ngưỡng): đây là ngưỡng người
    dùng tự đặt để xem, không phải ngưỡng chấm điểm, nên "đúng bằng" thuộc về
    phía được giữ.

    Nhãn ngoài dải `ten_lop` là lỗi lập trình (số lớp model khác số lớp đang
    đếm) — ném ra kèm chỉ số, không bỏ qua im lặng.
    """
    ra = {ten: 0 for ten in ten_lop}
    for nhan, diem in zip(labels, scores):
        if float(diem) < nguong_diem:
            continue
        nhan = int(nhan)
        if not 0 <= nhan < len(ten_lop):
            raise ValueError(
                f"Nhãn {nhan} nằm ngoài danh sách {len(ten_lop)} lớp "
                f"{list(ten_lop)} — model và danh sách lớp không khớp.")
        ra[ten_lop[nhan]] += 1
    return ra


def chon_anh_demo(coco, so_anh, ten_lop_ngap, seed=42):
    """Chọn ảnh cho phần demo: **một nửa có nhà ngập, một nửa không**.

    Chọn thuần "N ảnh đầu" thì gặp đúng vấn đề của dữ liệu FloodNet: chỉ ~10,5%
    ảnh có nhà ngập, nên 6 ảnh đầu rất dễ ra toàn ảnh không nhà — nhìn overlay
    không biết mô hình có tìm được nhà ngập hay không. Chia đôi để mắt thấy cả
    hai mặt, còn thiếu bên nào thì lấy bù bên kia.

    Có `seed` để hai lần chạy ra CÙNG danh sách ảnh — muốn so sánh hai
    checkpoint thì phải xem đúng cùng ảnh.
    """
    gt = chi_muc_gt(coco)
    co_ngap, khong_ngap = [], []
    for im in coco.get("images") or []:
        ten = im["file_name"]
        if (gt.get(ten, {}).get(ten_lop_ngap) or 0) > 0:
            co_ngap.append(ten)
        else:
            khong_ngap.append(ten)

    rnd = random.Random(seed)
    rnd.shuffle(co_ngap)
    rnd.shuffle(khong_ngap)

    n_ngap = min(len(co_ngap), (so_anh + 1) // 2)
    ra = co_ngap[:n_ngap]
    con_thieu = so_anh - len(ra)
    ra += khong_ngap[:con_thieu]
    if len(ra) < so_anh:  # bên "không ngập" không đủ thì bù từ bên ngập
        ra += co_ngap[n_ngap:n_ngap + (so_anh - len(ra))]
    return ra


def liet_ke_anh(thu_muc):
    """Đường dẫn các ảnh trong `thu_muc`, sắp theo tên (để chạy lại ra cùng thứ tự)."""
    ra = []
    for ten in sorted(os.listdir(thu_muc)):
        duong = pathlib.Path(thu_muc) / ten
        if duong.is_file() and duong.suffix.lower() in DUOI_ANH:
            ra.append(str(duong))
    return ra


def tim_checkpoint(work_dir):
    """Checkpoint tốt nhất trong `work_dir`, ưu tiên `best_coco_bbox_mAP_epoch_N.pth`.

    `save_best` của mmengine đặt tên tệp best kèm số epoch. Có nhiều tệp best
    (resume nhiều vòng) thì lấy epoch LỚN NHẤT — đó là bản mới nhất trong chuỗi
    cải thiện, không phải bản cũ còn sót.
    """
    thu_muc = pathlib.Path(work_dir)
    if not thu_muc.is_dir():
        return None

    ung_vien = []
    for p in thu_muc.glob("best_coco_bbox_mAP_epoch_*.pth"):
        m = re.search(r"epoch_(\d+)\.pth$", p.name)
        if m:
            ung_vien.append((int(m.group(1)), p))
    if ung_vien:
        ung_vien.sort()
        return str(ung_vien[-1][1])

    # Không có best (lần chạy dở, hoặc config không bật save_best): đọc
    # `last_checkpoint`. Tệp này mmengine ghi đường dẫn TUYỆT ĐỐI của lúc train
    # (trên Colab) — tệp đó có thể không còn nếu chạy ở máy khác, nên khi không
    # thấy thì ghép TÊN TỆP với work_dir hiện tại (work_dir vẫn là nơi chứa nó).
    tep_last = thu_muc / "last_checkpoint"
    if tep_last.is_file():
        noi_dung = tep_last.read_text(encoding="utf-8").strip()
        if os.path.isfile(noi_dung):
            return noi_dung
        ten = noi_dung.replace("\\", "/").rsplit("/", 1)[-1]
        duong = thu_muc / ten
        if duong.is_file():
            return str(duong)
    return None

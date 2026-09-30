"""Khảo sát dữ liệu FloodNet — Phase 1.

Trả lời bằng SỐ LIỆU hai câu hỏi phải chốt trước khi sang Phase 2:

  1. CÓ PHẢI CẮT TILE KHÔNG?
     Ảnh UAV gốc 4000x3000, resize về 1536px thì nhà có thể teo nhỏ tới mức model
     không học được. Nếu percentile 5 của cạnh box nhỏ nhất SAU RESIZE >= 32px
     thì không cần cắt tile (ngưỡng đã định trước trong plan, không chọn sau khi
     nhìn kết quả).

  2. CÓ PHẢI TÁCH NHÀ DÍNH NHAU KHÔNG?
     FloodNet không có nhãn instance — hai nhà sát nhau sẽ thành MỘT component.
     Nếu nhiều component có diện tích > 3x trung vị thì nghi đã bị gộp, phải tách
     bằng distance transform + watershed.

CẢ HAI KẾT LUẬN ĐỀU TÍNH TRÊN TẬP ĐÃ LỌC NHIỄU (component có cạnh nhỏ nhất >=
`min_side_px`). Mask FloodNet có vô số đốm vài pixel do lỗi gán nhãn; để lẫn thì
percentile 5 tụt xuống 1px và ta kết luận "phải cắt tile" vì RÁC, trong khi nhà
thật có cạnh nhỏ nhất trung vị hàng trăm pixel. Gặp thật ngày 30/09/2026.

Nguyên tắc: KHÔNG đoán. Cấu trúc zip, giá trị mask, kích thước ảnh đều đọc từ dữ
liệu thật rồi mới kết luận. Chỗ nào chưa chắc thì in ra cho người dùng kiểm.

HAI QUYẾT ĐỊNH KỸ THUẬT ĐÁNG CHÚ Ý (để trả lời khi bảo vệ):

  * ĐỌC THẲNG TỪ TRONG ZIP, không giải nén. Zip nặng 13 GB; giải nén tốn ~10 phút
    và 13 GB đĩa, mà /content mất sạch mỗi lần runtime reset (NOTES muc 1.6) nên
    phiên sau phải giải nén lại từ đầu. Đọc từng file từ zip nhanh tương đương.

  * TÁCH COMPONENT TRÊN MASK ĐÃ RESIZE, không phải mask gốc. Phase 2 cũng resize
    trước rồi mới tách, nên đo ở đây là đo ĐÚNG thứ Phase 2 sẽ tạo ra — vừa sát
    thực tế hơn vừa nhanh hơn ~6 lần (1,8M pixel so với 12M mỗi ảnh).

Chạy:
    python scripts/audit.py --config configs/data.yaml
    python scripts/audit.py --config configs/data.yaml --max-images 20   # chạy thử
"""

import argparse
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

import cv2
import numpy as np
import yaml

# Chỉ coi là ảnh khi có đuôi quen thuộc — tránh đọc nhầm file lạ trong zip
DUOI_ANH = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ===========================================================================
# Đọc dữ liệu từ zip
# ===========================================================================

def doc_anh(zf, ten, giai_ma_mau=True):
    """Đọc một file ảnh nằm trong zip, trả về mảng numpy (hoặc None nếu hỏng).

    `giai_ma_mau=True`  -> ảnh màu 3 kênh (để vẽ overlay)
    `giai_ma_mau=False` -> giữ nguyên số kênh gốc, dùng khi cần BIẾT file có mấy
                           kênh (phân biệt mask xám với ảnh màu)
    """
    du_lieu = np.frombuffer(zf.read(ten), np.uint8)
    co = cv2.IMREAD_COLOR if giai_ma_mau else cv2.IMREAD_UNCHANGED
    return cv2.imdecode(du_lieu, co)


def liet_ke_cay(ds_ten):
    """Liệt kê cấu trúc zip: mỗi thư mục có bao nhiêu file. Trả về dict đã sắp xếp."""
    dem = Counter()
    for ten in ds_ten:
        if ten.endswith("/"):
            continue
        dem[str(PurePosixPath(ten).parent)] += 1
    return dict(sorted(dem.items()))


def phan_loai_thu_muc(zf, cay, ds_ten):
    """Với mỗi thư mục, đọc thử 1 file để biết bên trong là ẢNH MÀU hay MASK XÁM.

    Cố ý phân loại bằng cách GIẢI MÃ THẬT chứ không nhìn tên thư mục.

    Lý do rất quan trọng: trong zip có thư mục tên `ColorMasks-FloodNetv1.0` —
    tên có chữ "mask" nhưng bên trong là ẢNH MÀU 1024x1024 chỉ để xem. Nếu dò
    theo tên, script sẽ chọn nhầm nó làm mask rồi train trên dữ liệu sai MÀ KHÔNG
    HỀ BÁO LỖI. Đó là loại lỗi im lặng, phải chặn bằng cách kiểm tra thật.
    """
    # Gom file theo thư mục một lần, thay vì quét lại danh sách tên cho từng thư mục
    file_theo_thu_muc = defaultdict(list)
    for ten in ds_ten:
        if ten.endswith("/") or not ten.lower().endswith(DUOI_ANH):
            continue
        file_theo_thu_muc[str(PurePosixPath(ten).parent)].append(ten)

    thong_tin = {}
    for thu_muc in cay:
        ds = sorted(file_theo_thu_muc.get(thu_muc, []))
        if not ds:
            continue

        anh = doc_anh(zf, ds[0], giai_ma_mau=False)
        if anh is None:
            thong_tin[thu_muc] = {"loai": "khac", "kenh": 0, "kich_thuoc": None,
                                  "vi_du": ds[0], "so_file": len(ds)}
            continue

        kenh = 1 if anh.ndim == 2 else anh.shape[2]
        loai = "mask" if kenh == 1 else ("anh" if kenh >= 3 else "khac")
        thong_tin[thu_muc] = {
            "loai": loai,
            "kenh": kenh,
            "kich_thuoc": (int(anh.shape[0]), int(anh.shape[1])),
            "vi_du": ds[0],
            "so_file": len(ds),
        }
    return thong_tin


def chuan_hoa_stem(ten_file):
    """Bỏ hậu tố `_lab` khỏi tên file, để ghép được ảnh với mask.

    FloodNet đặt tên KHÁC NHAU cho hai bên: ảnh là `1234.jpg`, mask là
    `1234_lab.png`. So tên nguyên bản thì không file nào khớp, dù hai thư mục có
    đúng bằng nhau số file. Đây là lỗi gặp thật ngày 30/09/2026 trên zip thật —
    dữ liệu giả ban đầu đặt hai bên trùng tên nên test không bắt được.

    Bỏ hậu tố ở CẢ HAI phía cho an toàn: ảnh không có hậu tố này nên vô hại, mà
    lỡ về sau ảnh cũng đổi tên theo quy ước khác thì vẫn ghép đúng.
    """
    return re.sub(r"_lab$", "", PurePosixPath(ten_file).stem, flags=re.IGNORECASE)


def ghep_cap_anh_mask(thong_tin, ds_ten):
    """Ghép mỗi thư mục ẢNH với thư mục MASK cùng cấp, kiểm tra tên file khớp nhau.

    Trả về (ds_cap, canh_bao):
      ds_cap   = [(tên_split, [(tên_ảnh, tên_mask), ...]), ...]
      canh_bao = danh sách câu cảnh báo cho người dùng đọc
    """
    canh_bao = []

    # Gom theo thư mục cha: ".../train/train-org-img" và ".../train/train-label-img"
    # có cùng cha là ".../train" -> cùng một split.
    #
    # Giá trị là DANH SÁCH chứ không phải một thư mục: ba thư mục ColorMasks-*Set
    # là anh em ruột cùng cha "ColorMasks-FloodNetv1.0", dùng dict đơn thì hai
    # thư mục sau đè mất thư mục đầu và ta báo thiếu mất hai chỗ.
    theo_cha = defaultdict(lambda: {"anh": [], "mask": []})
    for thu_muc, tt in thong_tin.items():
        if tt["loai"] in ("mask", "anh"):
            theo_cha[str(PurePosixPath(thu_muc).parent)][tt["loai"]].append(thu_muc)

    # Gom file theo thư mục một lần
    file_theo_thu_muc = defaultdict(list)
    for ten in ds_ten:
        if ten.endswith("/") or not ten.lower().endswith(DUOI_ANH):
            continue
        file_theo_thu_muc[str(PurePosixPath(ten).parent)].append(ten)

    def theo_stem(thu_muc):
        """{tên_file_đã_chuẩn_hoá: đường_dẫn_đầy_đủ}"""
        return {chuan_hoa_stem(t): t for t in file_theo_thu_muc.get(thu_muc, [])}

    ds_cap = []
    for cha, hai in sorted(theo_cha.items()):
        ds_anh, ds_mask = hai["anh"], hai["mask"]
        if not ds_anh or not ds_mask:
            continue
        if len(ds_anh) > 1 or len(ds_mask) > 1:
            canh_bao.append(
                f"[!] '{cha}': có {len(ds_anh)} thư mục ảnh và {len(ds_mask)} thư mục "
                f"mask cùng cấp -> không biết ghép cái nào với cái nào, bỏ qua. "
                f"ảnh={ds_anh}, mask={ds_mask}")
            continue

        file_anh, file_mask = theo_stem(ds_anh[0]), theo_stem(ds_mask[0])
        chung = sorted(set(file_anh) & set(file_mask))
        if not chung:
            canh_bao.append(
                f"[!] '{cha}': có cả thư mục ảnh và mask nhưng KHÔNG ghép được cặp "
                f"nào (đã thử bỏ hậu tố '_lab' ở tên).\n"
                f"    ảnh  ({len(file_anh)}): {sorted(file_anh)[:3]}\n"
                f"    mask ({len(file_mask)}): {sorted(file_mask)[:3]}")
            continue
        if len(file_anh) != len(file_mask):
            canh_bao.append(
                f"[!] '{cha}': số file lệch nhau — ảnh {len(file_anh)}, "
                f"mask {len(file_mask)}, ghép được {len(chung)} cặp.")

        ds_cap.append((cha, [(file_anh[k], file_mask[k]) for k in chung]))

    # Thư mục có ẢNH MÀU nhưng không có mask cùng cấp -> nghi là bẫy ColorMasks.
    # Gộp thành MỘT cảnh báo: liệt kê từng thư mục ra thì ba dòng giống hệt nhau,
    # người đọc lại tưởng có ba vấn đề khác nhau.
    mo_coi = [t for _, hai in sorted(theo_cha.items()) if not hai["mask"]
              for t in hai["anh"]]
    if mo_coi:
        tong = sum(thong_tin[t]["so_file"] for t in mo_coi)
        canh_bao.append(
            f"[!] {len(mo_coi)} thư mục có ẢNH MÀU nhưng KHÔNG có thư mục mask cùng "
            f"cấp -> bỏ qua ({tong} file): " + ", ".join(mo_coi) + ".\n"
            f"    Nếu tên thư mục có chữ 'mask' thì đây đúng là bẫy ColorMasks "
            f"(ảnh màu để xem, không phải nhãn).")

    return ds_cap, canh_bao


# ===========================================================================
# Ghi/đọc file trung gian — chạy dở mà Colab ngắt thì không mất hết
# ===========================================================================

def doc_ket_qua_da_co(duong_dan):
    """Đọc lại .jsonl của lần chạy trước -> {tên_ảnh: bản_ghi} để bỏ qua ảnh đã làm."""
    da_co = {}
    if not os.path.exists(duong_dan):
        return da_co
    with open(duong_dan, encoding="utf-8") as f:
        for dong in f:
            dong = dong.strip()
            if not dong:
                continue
            try:
                r = json.loads(dong)
                da_co[r["anh"]] = r
            except (json.JSONDecodeError, KeyError):
                # Dòng cuối bị cắt cụt do crash giữa chừng -> bỏ qua dòng đó
                continue
    return da_co


# ===========================================================================
# Xử lý một ảnh
# ===========================================================================

def xu_ly_mot_anh(zf, ten_anh, ten_mask, classes, water_value,
                  target_long_side, kiem_tra_nuoc=False):
    """Đọc 1 cặp ảnh/mask, trả về bản ghi thống kê (dict) hoặc None nếu lỗi.

    Box và diện tích trả về đều ở TOẠ ĐỘ SAU RESIZE — đúng thứ Phase 2 sẽ tạo ra.
    """
    mask = doc_anh(zf, ten_mask, giai_ma_mau=False)
    if mask is None:
        return None
    if mask.ndim != 2:
        # Mask nhiều kênh nghĩa là đã chọn nhầm thư mục — không đoán, báo lỗi luôn
        return {"anh": ten_anh,
                "loi": f"mask có {mask.shape[2]} kênh, không phải ảnh xám"}

    h, w = mask.shape[:2]

    # --- Thống kê giá trị mask (để xác minh bảng lớp) ---
    gia_tri, so_pixel = np.unique(mask, return_counts=True)
    dem_gia_tri = {int(g): int(c) for g, c in zip(gia_tri, so_pixel)}

    # --- Resize mask về kích thước dự kiến. Nhãn là số nguyên nên BẮT BUỘC
    #     INTER_NEAREST: nội suy tuyến tính sẽ tạo ra giá trị lớp không tồn tại. ---
    ty_le = target_long_side / max(h, w)
    h2, w2 = int(round(h * ty_le)), int(round(w * ty_le))
    mask_nho = cv2.resize(mask, (w2, h2), interpolation=cv2.INTER_NEAREST)

    ban_ghi = {
        "anh": ten_anh,
        "kich_thuoc_goc": [h, w],
        "kich_thuoc_resize": [h2, w2],
        "ty_le": round(ty_le, 6),
        "dem_gia_tri": dem_gia_tri,
        "lop": {},
    }

    # --- Tách component cho từng lớp nhà ---
    nhan_moi_lop = {}
    for gia_tri_lop, ten_lop in classes.items():
        nhi_phan = (mask_nho == gia_tri_lop).astype(np.uint8)
        so_nhan, nhan, stats, _ = cv2.connectedComponentsWithStats(
            nhi_phan, connectivity=8)

        # stats[i] = [x, y, w, h, area]; nhãn 0 là nền nên bỏ
        ds_box = [[int(v) for v in stats[i]] for i in range(1, so_nhan)]

        nhan_moi_lop[gia_tri_lop] = nhan
        ban_ghi["lop"][ten_lop] = {
            "gia_tri_mask": gia_tri_lop,
            "so_component": len(ds_box),
            "boxes": ds_box,
        }

    # --- Kiểm chứng "nhà ngập phải nằm cạnh nước" (chỉ chạy trên mẫu cho nhanh) ---
    # Cách làm: giãn vùng nước ra 1 pixel, rồi đếm xem có bao nhiêu COMPONENT của
    # mỗi lớp chạm vào vùng đã giãn. Đây là phép kiểm chứng LẠI kết luận ở NOTES
    # muc 2.1 ("chỉ ~22% nhà gán nhãn ngập thực sự chạm nước") bằng code, thay vì
    # tin vào trí nhớ.
    if kiem_tra_nuoc:
        nuoc = (mask_nho == water_value).astype(np.uint8)
        if nuoc.any():
            nuoc_gian = cv2.dilate(nuoc, np.ones((3, 3), np.uint8)).astype(bool)
            for gia_tri_lop, ten_lop in classes.items():
                nhan = nhan_moi_lop[gia_tri_lop]
                cham = nuoc_gian & (mask_nho == gia_tri_lop)
                # np.unique trả về cả nhãn 0 (nền) nếu nền có chạm — phải loại ra
                nhan_cham = np.unique(nhan[cham]) if cham.any() else np.array([], int)
                ban_ghi["lop"][ten_lop]["so_component_cham_nuoc"] = int(
                    (nhan_cham != 0).sum())

    return ban_ghi


# ===========================================================================
# Vẽ ảnh minh hoạ
# ===========================================================================

def ten_split(ten_anh):
    """Tên split của một ảnh, suy từ đường dẫn: `.../<split>/<thư_mục_ảnh>/<tên>.jpg`.

    Lấy thư mục CHA của thư mục chứa ảnh (tức `parts[-3]`), không lấy tên thư mục
    chứa ảnh — tên đó là `train-org-img` chứ không phải `train`.
    """
    cha = PurePosixPath(ten_anh).parent
    return cha.parent.name or cha.name or "?"


def ten_file_overlay(ten_trong_zip):
    """Đổi đường dẫn trong zip thành tên file phẳng, GIỮ LẠI tên split.

    Cố ý không dùng thẳng tên file gốc: cả ba split đều đặt tên ảnh là 1.jpg,
    2.jpg... nên nếu chỉ lấy tên file thì ảnh của train/val/test sẽ ghi ĐÈ lên
    nhau, mất ảnh mà không hề báo lỗi. Lấy 3 phần cuối của đường dẫn
    (split/thư_mục/tên_file) là đủ phân biệt.
    """
    phan = [x for x in PurePosixPath(ten_trong_zip).parts if x not in (".", "/")]
    return "_".join(phan[-3:]).rsplit(".", 1)[0] + ".jpg"


def ve_overlay(zf, ten_anh, ban_ghi, thu_muc_ra, nguong_to_bat_thuong,
               trung_vi_dien_tich, min_side_px):
    """Vẽ box lên ảnh rồi lưu JPEG.

    Box có diện tích > nguong_to_bat_thuong x trung vị được vẽ MÀU KHÁC để người
    dùng nhìn bằng mắt xem có phải nhiều nhà bị gộp làm một không.

    `trung_vi_dien_tich` là trung vị của TOÀN BỘ dữ liệu (đã lọc nhiễu), truyền từ
    tong_hop() vào — KHÔNG lấy trung vị riêng của ảnh đang vẽ. Lấy trung vị riêng
    thì một ảnh chỉ có 2 căn nhà sẽ tô tím căn to hơn, dù nó chẳng to gì so với
    đồ án, và màu tím mất hết ý nghĩa.

    Trả về (thành_công, cảnh_báo).
    """
    anh = doc_anh(zf, ten_anh, giai_ma_mau=True)
    if anh is None:
        return False, f"đọc không được ảnh {ten_anh}"

    canh_bao = ""
    h_goc, w_goc = ban_ghi["kich_thuoc_goc"]
    if (anh.shape[0], anh.shape[1]) != (h_goc, w_goc):
        # Ảnh và mask lệch kích thước là mục cần kiểm trong checklist Phase 1.
        # Không tự ý sửa: báo ra để người dùng biết mà xử lý ở Phase 2.
        canh_bao = (f"ảnh {anh.shape[1]}x{anh.shape[0]} lệch với mask "
                    f"{w_goc}x{h_goc}")

    # Resize ảnh về đúng hệ toạ độ của mask đã resize, để box vẽ khớp vị trí
    h2, w2 = ban_ghi["kich_thuoc_resize"]
    anh = cv2.resize(anh, (w2, h2), interpolation=cv2.INTER_AREA)

    trung_vi = trung_vi_dien_tich or 0.0

    mau = {"flooded_building": (0, 0, 255),        # đỏ = nhà ngập
           "non_flooded_building": (0, 200, 0)}    # xanh lá = nhà không ngập

    for ten_lop, lop in ban_ghi["lop"].items():
        # Không vẽ đốm nhiễu: box 2px chỉ làm rối mắt, mà mắt người dùng đang cần
        # soi chuyện "nhiều nhà bị gộp" chứ không phải soi rác gán nhãn — rác đã
        # có cột "Bỏ do nhiễu" trong báo cáo lo.
        for x, y, bw, bh, area in lop["boxes"]:
            if min(bw, bh) < min_side_px:
                continue
            to_bat_thuong = trung_vi > 0 and area > nguong_to_bat_thuong * trung_vi
            mau_vien = (255, 0, 255) if to_bat_thuong else mau.get(ten_lop, (255, 255, 255))
            cv2.rectangle(anh, (x, y), (x + bw, y + bh), mau_vien,
                          3 if to_bat_thuong else 2)

    dong = []
    for t, l in ban_ghi["lop"].items():
        ve = sum(1 for b in l["boxes"] if min(b[2], b[3]) >= min_side_px)
        bo = len(l["boxes"]) - ve
        dong.append(f"{t}: {ve} box" + (f" (bỏ {bo} đốm < {min_side_px}px)" if bo else ""))
    for i, d in enumerate(dong):
        vi_tri = (10, 30 + i * 28)
        cv2.putText(anh, d, vi_tri, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(anh, d, vi_tri, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    ten_ra = os.path.join(thu_muc_ra, ten_file_overlay(ten_anh))
    if not cv2.imwrite(ten_ra, anh, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        return False, f"ghi không được ảnh {ten_ra}"
    return True, canh_bao


# ===========================================================================
# Tổng hợp & báo cáo
# ===========================================================================

def tong_hop(ds_ban_ghi, classes, decisions, preprocess):
    """Gộp mọi bản ghi thành các con số để ra quyết định. Trả về dict."""
    canh = defaultdict(list)          # tên lớp -> cạnh nhỏ nhất của từng box
    dien_tich = defaultdict(list)     # tên lớp -> diện tích từng box
    box_moi_anh = defaultdict(list)
    nuoc_tong = defaultdict(int)      # chỉ tính trên các ảnh có kiểm tra nước
    nuoc_cham = defaultdict(int)
    dem_gia_tri = Counter()
    dem_gia_tri_split = defaultdict(Counter)   # split -> giá trị mask -> số pixel
    so_anh_co_lop = Counter()         # ảnh có ít nhất 1 component (kể cả đốm nhiễu)
    so_anh_co_box_loc = Counter()     # ảnh có ít nhất 1 box ĐÃ LỌC NHIỄU
    kich_thuoc_goc = Counter()
    so_anh_tong = 0
    so_anh_kiem_tra_nuoc = 0          # số ảnh thực sự chạy phép kiểm chứng nước
    min_side_px = decisions["min_side_px"]

    for r in ds_ban_ghi:
        if "loi" in r:
            continue
        so_anh_tong += 1
        kich_thuoc_goc[tuple(r["kich_thuoc_goc"])] += 1
        # Đếm pixel theo TỪNG SPLIT, không chỉ tổng. Nếu một lớp vắng mặt thì câu hỏi
        # đầu tiên luôn là "vắng ở mọi split hay chỉ một split" — trả lời được ngay
        # trong báo cáo thì đỡ phải chạy lại 2343 ảnh chỉ để hỏi một câu.
        split = ten_split(r["anh"])
        for g, c in r["dem_gia_tri"].items():
            dem_gia_tri[int(g)] += c
            dem_gia_tri_split[split][int(g)] += c

        if any("so_component_cham_nuoc" in lop for lop in r["lop"].values()):
            so_anh_kiem_tra_nuoc += 1

        for ten_lop, lop in r["lop"].items():
            boxes = lop["boxes"]
            box_moi_anh[ten_lop].append(len(boxes))
            if boxes:
                so_anh_co_lop[ten_lop] += 1
            for x, y, bw, bh, area in boxes:
                canh[ten_lop].append(min(bw, bh))
                dien_tich[ten_lop].append(area)
            # Đếm theo box ĐÃ LỌC: một ảnh chỉ có đúng một đốm 2px không phải là
            # "ảnh có nhà ngập", mà đó mới là câu người đọc bảng này muốn hỏi.
            if any(min(b[2], b[3]) >= min_side_px for b in boxes):
                so_anh_co_box_loc[ten_lop] += 1
            if "so_component_cham_nuoc" in lop:
                nuoc_tong[ten_lop] += len(boxes)
                nuoc_cham[ten_lop] += lop["so_component_cham_nuoc"]

    ket_qua = {
        "so_anh_tong": so_anh_tong,
        "target_long_side": preprocess["target_long_side"],
        "kich_thuoc_goc": {f"{h}x{w}": n for (h, w), n in kich_thuoc_goc.most_common()},
        "dem_gia_tri_mask": dict(sorted(dem_gia_tri.items())),
        "dem_gia_tri_theo_split": {s: dict(sorted(c.items()))
                                   for s, c in sorted(dem_gia_tri_split.items())},
        "so_anh_co_lop": dict(so_anh_co_lop),
        "so_anh_co_box_loc": dict(so_anh_co_box_loc),
        "so_anh_kiem_tra_nuoc": so_anh_kiem_tra_nuoc,
        "kiem_tra_nuoc": {
            t: {"tong_component": nuoc_tong[t],
                "cham_nuoc": nuoc_cham[t],
                "ty_le": round(nuoc_cham[t] / nuoc_tong[t], 4) if nuoc_tong[t] else None}
            for t in nuoc_tong},
        "lop": {},
    }

    dt_loc_tat_ca = []                # gom lại để tính mốc "to bất thường" cho overlay
    for gia_tri_lop, ten_lop in classes.items():
        c = np.array(canh[ten_lop]) if canh[ten_lop] else np.array([])
        dt = np.array(dien_tich[ten_lop]) if dien_tich[ten_lop] else np.array([])
        if len(c) == 0:
            ket_qua["lop"][ten_lop] = {"gia_tri_mask": gia_tri_lop, "so_box": 0}
            continue

        # --- Lọc nhiễu trước khi kết luận về tile ---
        # Mask FloodNet có vô số đốm vài pixel (lỗi gán nhãn). Nếu để lẫn, percentile
        # 5 của cạnh nhỏ nhất bị kéo xuống 1px và script kết luận "phải cắt tile" —
        # trong khi nhà thật có cạnh nhỏ nhất trung vị hàng trăm pixel. Tức là quyết
        # định về độ phân giải lại do rác quyết định. Đã gặp thật ngày 30/09/2026.
        #
        # Lọc theo CẠNH nhỏ nhất chứ không theo diện tích: một mảng 1x500 pixel có
        # diện tích lớn hơn ngưỡng nhưng vẫn là vệt rác, không phải nhà.
        loc = c >= min_side_px
        c_loc, dt_loc = c[loc], dt[loc]
        if len(dt_loc):
            dt_loc_tat_ca.append(dt_loc)

        # Mốc "to bất thường" (3x trung vị) cũng phải tính trên tập ĐÃ LỌC. Tính
        # trên tập thô thì hàng trăm đốm vài pixel kéo trung vị xuống, hạ thấp mốc,
        # và nhà to thật bị gắn cờ "nghi bị gộp". Đã xảy ra thật ở lần chạy đầy đủ
        # 30/09/2026: non_flooded_building bị gắn cờ 1096/3985 box (27.5%) còn
        # flooded_building chỉ 14 box — nếu là gộp nhà thật thì hai lớp phải na ná
        # nhau, chênh 78 lần nghĩa là cái mốc đang đo phân bố kích thước, không đo
        # chuyện gộp nhà.
        trung_vi_dt_loc = float(np.median(dt_loc)) if len(dt_loc) else None

        ket_qua["lop"][ten_lop] = {
            "gia_tri_mask": gia_tri_lop,
            "so_box": int(len(c)),
            "so_box_loc": int(len(c_loc)),
            "so_box_nhieu": int(len(c) - len(c_loc)),
            "box_moi_anh_trung_binh": round(float(np.mean(box_moi_anh[ten_lop])), 2),
            "box_moi_anh_lon_nhat": int(np.max(box_moi_anh[ten_lop])),
            # Số liệu thô, giữ lại để đối chiếu chứ KHÔNG dùng ra quyết định
            "canh_nho_nhat_p5": float(np.percentile(c, 5)),
            "canh_nho_nhat_p50": float(np.percentile(c, 50)),
            "canh_nho_nhat_p95": float(np.percentile(c, 95)),
            # Số liệu đã lọc nhiễu — đây mới là căn cứ quyết định tiling
            "canh_nho_nhat_p5_loc": float(np.percentile(c_loc, 5)) if len(c_loc) else None,
            "canh_nho_nhat_p50_loc": float(np.percentile(c_loc, 50)) if len(c_loc) else None,
            "dien_tich_p1": float(np.percentile(dt, 1)),
            "dien_tich_p5": float(np.percentile(dt, 5)),
            "dien_tich_p50": float(np.median(dt)),          # thô, để đối chiếu
            "dien_tich_p50_loc": trung_vi_dt_loc,           # mốc dùng ra quyết định
            "dien_tich_p95": float(np.percentile(dt, 95)),
            "so_box_to_bat_thuong": int(
                (dt_loc > decisions["area_outlier_ratio"] * trung_vi_dt_loc).sum())
            if trung_vi_dt_loc else 0,
        }

    # Một mốc duy nhất cho cả hai lớp, để ảnh overlay không tuỳ tiện theo ảnh.
    ket_qua["trung_vi_dien_tich_loc"] = (
        float(np.median(np.concatenate(dt_loc_tat_ca))) if dt_loc_tat_ca else None)

    ket_qua["ket_luan"] = ket_luan(ket_qua, decisions, preprocess)
    return ket_qua


def ket_luan(kq, decisions, preprocess):
    """So số liệu với ngưỡng đã định TRƯỚC trong config để ra quyết định."""
    kl = {}
    min_side = decisions["min_side_px"]

    # --- 0. Lớp được cấu hình nhưng KHÔNG có pixel nào trong toàn bộ dữ liệu ---
    # Kiểm việc này TRƯỚC mọi thứ khác. Nếu lớp "nhà ngập" không tồn tại ở đâu thì
    # mọi kết luận phía dưới đều vô nghĩa và đồ án không train được — phải la lên
    # thật to, chứ để lọt vào báo cáo dưới dạng một dòng "so_box = 0" thì rất dễ
    # đọc lướt qua. (Không có box nào <=> không có pixel nào: pixel nào cũng tạo
    # thành một component.)
    kl["lop_khong_co_pixel"] = [
        f"{ten_lop} (giá trị mask {v['gia_tri_mask']})"
        for ten_lop, v in kq["lop"].items() if not v.get("so_box")]

    # --- 1. Tiling? ---
    # Dùng số liệu ĐÃ LỌC NHIỄU. Đốm vài pixel kéo percentile 5 xuống 1px và làm
    # script kết luận phải cắt tile, trong khi nhà thật to gấp trăm lần.
    p5_theo_lop = {t: v["canh_nho_nhat_p5_loc"] for t, v in kq["lop"].items()
                   if v.get("canh_nho_nhat_p5_loc") is not None}
    if p5_theo_lop:
        p5 = min(p5_theo_lop.values())
        kl["canh_box_p5_nho_nhat"] = p5
        kl["can_tiling"] = bool(p5 < decisions["min_box_side_p5_ok"])
        kl["ly_do_tiling"] = (
            f"percentile 5 cạnh box nhỏ nhất sau resize, đã bỏ đốm < {min_side}px, "
            f"= {p5:.1f}px; ngưỡng = {decisions['min_box_side_p5_ok']}px")

    # --- 2. Tách nhà dính? ---
    # Cũng tính trên tập đã lọc: đốm rác không thể là "nhà bị gộp".
    tong_to = sum(v.get("so_box_to_bat_thuong", 0) for v in kq["lop"].values())
    tong_box = sum(v.get("so_box_loc", 0) for v in kq["lop"].values())
    ty_le_to = tong_to / tong_box if tong_box else 0.0
    kl["so_box_to_bat_thuong"] = tong_to
    kl["so_box_dung_de_ket_luan"] = tong_box
    kl["ty_le_box_to_bat_thuong"] = round(ty_le_to, 4)
    # Mốc 5%: dưới mức đó coi như đuôi phân bố bình thường của diện tích nhà,
    # chưa đáng đánh đổi bằng việc thêm cả một bước watershed vào pipeline.
    kl["can_tach_nha_dinh"] = bool(ty_le_to > 0.05)
    # Nói rõ "đã lọc" trong câu kết luận: mẫu số ở đây nhỏ hơn cột "Số box" trong
    # bảng, người đọc đối chiếu hai chỗ mà không thấy giải thích sẽ tưởng script sai.
    kl["ly_do_tach"] = (f"{tong_to}/{tong_box} box đã lọc nhiễu ({ty_le_to:.1%}) có "
                        f"diện tích > {decisions['area_outlier_ratio']}x trung vị "
                        f"diện tích đã lọc")
    # Con số này KHÔNG chứng minh có nhà bị gộp — nó chỉ nói "có nhiều box to gấp 3
    # lần trung vị". Phân bố diện tích nhà vốn dĩ rất rộng (nhà ống vs nhà xưởng),
    # nên phải nhìn ảnh overlay mới phân biệt được "một căn nhà to" với "hai căn bị
    # dính". Ghi thẳng vào kết luận để người đọc không tự suy diễn quá mức.
    kl["ly_do_tach_can_kiem_bang_mat"] = (
        "chỉ số này đo phân bố kích thước nhà, KHÔNG tự chứng minh có gộp nhà; "
        "phải xem box viền TÍM trong overlay/ mới kết luận được")

    # --- 3. min_area gợi ý ---
    # Đây là LỰA CHỌN THIẾT KẾ, không phải số đo: một nhà nhỏ hơn min_side x min_side
    # pixel ở cạnh dài target_long_side thì mắt người cũng không nhận ra, giữ lại
    # chỉ thêm rác cho cả lúc train lẫn lúc đếm. Lấy đúng bình phương của min_side.
    kl["min_area_goi_y"] = min_side ** 2
    kl["ly_do_min_area"] = (
        f"cạnh {min_side}px là mức nhỏ nhất còn nhận ra một căn nhà ở cạnh dài "
        f"{preprocess['target_long_side']}px -> diện tích tối thiểu {min_side ** 2}px")
    kl["min_area_dang_dung"] = preprocess["min_area"]

    # --- 4. Quyết định người dùng đã chốt ở GATE 1 (ghi trong config) ---
    # Ngưỡng ở trên là MÁY tự tính; đây mới là thứ đồ án LÀM. Hai bên có thể khác
    # nhau (ngưỡng là quy tắc chung, quyết định có thêm bằng chứng nhìn bằng mắt và
    # ngân sách thời gian train) — nên in cả hai. Báo cáo chỉ in "Cắt tile: CÓ" mà
    # code lại không cắt tile thì người đọc sẽ tưởng có chỗ nào đó bị bỏ quên.
    for khoa, ten in (("cat_tile", "cắt tile"), ("tach_nha_dinh", "tách nhà dính")):
        if khoa in decisions:
            kl[f"chot_{khoa}"] = bool(decisions[khoa])

    return kl


def ghi_bao_cao(kq, duong_dan):
    """Ghi báo cáo markdown (để dán thẳng vào docs/NOTES.md) và trả về nội dung."""
    d = ["# Báo cáo EDA — Phase 1\n",
         f"Tổng số ảnh đã xử lý: **{kq['so_anh_tong']}**\n"]

    # Cảnh báo đặt NGAY ĐẦU báo cáo, trước cả số liệu. Nếu lớp cần train không
    # tồn tại trong dữ liệu thì không có con số nào phía dưới đáng đọc nữa.
    if kq["ket_luan"].get("lop_khong_co_pixel"):
        d += ["> ## ⚠️ CẢNH BÁO: có lớp không xuất hiện trong dữ liệu\n>",
              "> Các lớp sau được khai báo trong config nhưng **không có pixel nào** "
              "trong toàn bộ ảnh đã xử lý:\n>"]
        for t in kq["ket_luan"]["lop_khong_co_pixel"]:
            d.append(f"> - **{t}**")
        d += [">",
              "> Nếu đây là lớp bắt buộc phải có (ví dụ `flooded_building`) thì "
              "**dừng lại, đừng train**: kiểm tra lại bảng lớp, hoặc chạy trên nhiều "
              "ảnh hơn — mẫu nhỏ có thể chưa gặp lớp đó.\n"]

    d += ["## Kích thước ảnh gốc\n"]
    for k, n in kq["kich_thuoc_goc"].items():
        # Kèm luôn tỉ lệ resize: hai cỡ ảnh trong cùng một dataset nghĩa là hai tỉ lệ
        # khác nhau, tức độ phân giải thực trên mặt đất của hai nhóm ảnh lệch nhau.
        # Không sai, nhưng phải biết mà bật augmentation đổi tỉ lệ lúc train.
        canh_dai = max(int(x) for x in k.split("x"))
        d.append(f"- `{k}`: {n} ảnh (resize về cạnh dài {kq['target_long_side']}px "
                 f"-> tỉ lệ {kq['target_long_side'] / canh_dai:.4f})")

    d += ["", "## Số pixel mỗi giá trị mask\n", "| Giá trị | Số pixel | Tỉ lệ |",
          "|---|---|---|"]
    tong_px = sum(kq["dem_gia_tri_mask"].values()) or 1
    for g, c in kq["dem_gia_tri_mask"].items():
        d.append(f"| {g} | {c:,} | {c / tong_px:.4%} |")

    # Bảng theo split: trả lời ngay "lớp vắng ở MỌI split hay chỉ một split". Chỉ in
    # khi có từ hai split trở lên — một split thì bảng này lặp lại y hệt bảng trên.
    theo_split = kq.get("dem_gia_tri_theo_split") or {}
    if len(theo_split) > 1:
        # Cột xếp theo thứ tự quen thuộc train/val/test, không xếp alphabet (alphabet
        # cho ra test/train/val — đọc ngược, dễ nhầm cột).
        quen_thuoc = ["train", "val", "test"]
        ds_split = ([s for s in quen_thuoc if s in theo_split]
                    + sorted(s for s in theo_split if s not in quen_thuoc))
        d += ["", "## Số pixel mỗi giá trị mask theo split\n",
              "Tỉ lệ tính riêng trong từng split (ba split lệch số ảnh nên phải chuẩn "
              "hoá mới so được với nhau). Ô `0.0000%` nghĩa là **split đó không có "
              "pixel nào của lớp này**.\n",
              "| Giá trị | " + " | ".join(ds_split) + " |",
              "|---" * (len(ds_split) + 1) + "|"]
        for g in kq["dem_gia_tri_mask"]:
            o = [f"{theo_split[s].get(g, 0) / (sum(theo_split[s].values()) or 1):.4%}"
                 for s in ds_split]
            d.append(f"| {g} | " + " | ".join(o) + " |")

    # Bảng "ảnh có / không có lớp": câu hỏi đầu tiên khi chia train/val và khi đọc
    # mAP. Đếm theo box ĐÃ LỌC, vì ảnh chỉ có một đốm 2px không phải ảnh có nhà.
    loc_theo_lop = kq.get("so_anh_co_box_loc") or {}
    if loc_theo_lop and kq["so_anh_tong"]:
        d += ["", "## Số ảnh có / không có từng lớp\n",
              f"Đếm trên **{kq['so_anh_tong']}** ảnh, chỉ tính box đã bỏ đốm nhiễu "
              "(ảnh chỉ có một đốm vài pixel không tính là ảnh có nhà).\n",
              "| Lớp | Ảnh CÓ ít nhất 1 box | Ảnh KHÔNG có box nào |",
              "|---|---|---|"]
        for ten_lop, v in kq["lop"].items():
            co = loc_theo_lop.get(ten_lop, 0)
            khong = kq["so_anh_tong"] - co
            d.append(f"| {ten_lop} | {co:,} ({co / kq['so_anh_tong']:.1%}) "
                     f"| {khong:,} ({khong / kq['so_anh_tong']:.1%}) |")

    d += ["", "## Thống kê theo lớp\n",
          "Cột **p5 đã lọc** là con số dùng để quyết định tiling: đã bỏ các đốm nhỏ "
          "hơn `min_side_px`. Cột **p5 thô** giữ lại để thấy rõ mức nhiễu gán nhãn "
          "trong dữ liệu.\n",
          "Cột **DT p50 đã lọc** là mốc nhân 3 để gắn cờ \"box to bất thường\"; cột "
          "**DT p50 thô** in kèm để thấy đốm nhiễu kéo trung vị lệch bao nhiêu.\n",
          "| Lớp | Giá trị mask | Số box | Bỏ do nhiễu | Box/ảnh TB | Box/ảnh max "
          "| p5 đã lọc | p50 đã lọc | p5 thô | DT p50 thô | DT p50 đã lọc "
          "| Box to bất thường |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ten_lop, v in kq["lop"].items():
        if not v.get("so_box"):
            d.append(f"| {ten_lop} | {v.get('gia_tri_mask', '?')} | 0 "
                     f"| | | | | | | | | |")
            continue
        p5_loc = f"{v['canh_nho_nhat_p5_loc']:.1f}" if v.get("canh_nho_nhat_p5_loc") is not None else "—"
        p50_loc = f"{v['canh_nho_nhat_p50_loc']:.1f}" if v.get("canh_nho_nhat_p50_loc") is not None else "—"
        dt_loc = f"{v['dien_tich_p50_loc']:.0f}" if v.get("dien_tich_p50_loc") else "—"
        d.append(f"| {ten_lop} | {v['gia_tri_mask']} | {v['so_box']:,} "
                 f"| {v['so_box_nhieu']:,} "
                 f"| {v['box_moi_anh_trung_binh']} | {v['box_moi_anh_lon_nhat']} "
                 f"| {p5_loc} | {p50_loc} "
                 f"| {v['canh_nho_nhat_p5']:.1f} | {v['dien_tich_p50']:.0f} "
                 f"| {dt_loc} | {v['so_box_to_bat_thuong']} |")

    if kq["kiem_tra_nuoc"]:
        d += ["", "## Kiểm chứng lớp 1 = nhà NGẬP (component có chạm vùng nước)\n",
              f"Chỉ chạy trên **{kq.get('so_anh_kiem_tra_nuoc', 0)} ảnh** có vùng nước "
              f"trong số ít ảnh được lấy mẫu (`eda.num_water_check`) — mẫu nhỏ thì "
              "tỉ lệ chỉ để xem HƯỚNG, đừng trích như một con số chắc chắn.\n",
              "Tỉ lệ tính trên **mọi** component, kể cả đốm nhiễu chưa lọc; đốm nhiễu "
              "hiếm khi chạm nước nên tỉ lệ thật của nhà sẽ **cao hơn** bảng này.\n",
              "| Lớp | Tổng component | Chạm nước | Tỉ lệ |", "|---|---|---|---|"]
        for t, v in kq["kiem_tra_nuoc"].items():
            ty = f"{v['ty_le']:.1%}" if v["ty_le"] is not None else "—"
            d.append(f"| {t} | {v['tong_component']} | {v['cham_nuoc']} | {ty} |")

    d += ["", "## Kết luận & đề xuất tham số\n"]
    kl = kq["ket_luan"]
    if "can_tiling" in kl:
        d.append(f"- **Cắt tile: {'CÓ' if kl['can_tiling'] else 'KHÔNG'}** — {kl['ly_do_tiling']}")
    if "can_tach_nha_dinh" in kl:
        d.append(f"- **Tách nhà dính: {'CÓ' if kl['can_tach_nha_dinh'] else 'KHÔNG'}** "
                 f"— {kl['ly_do_tach']}")
        if kl.get("ly_do_tach_can_kiem_bang_mat"):
            d.append(f"  - ⚠️ {kl['ly_do_tach_can_kiem_bang_mat']}")
    if "min_area_goi_y" in kl:
        d.append(f"- **min_area đề xuất: {kl['min_area_goi_y']}** — {kl['ly_do_min_area']} "
                 f"(config đang để {kl['min_area_dang_dung']})")

    # Quyết định đã chốt ở GATE 1. Chỉ nói "khác ngưỡng" khi thật sự khác — nói câu
    # đó vô điều kiện thì nó thành nhiễu, và lần sau không ai đọc nữa.
    for khoa, nhan, nguong in (("cat_tile", "Cắt tile", kl.get("can_tiling")),
                               ("tach_nha_dinh", "Tách nhà dính",
                                kl.get("can_tach_nha_dinh"))):
        if f"chot_{khoa}" not in kl:
            continue
        chot = kl[f"chot_{khoa}"]
        d.append(f"- **Chốt của đồ án: {'CÓ' if chot else 'KHÔNG'} {nhan.lower()}** "
                 f"(lý do ghi trong `configs/data.yaml`)")
        if nguong is not None and chot != nguong:
            d.append(f"  - ⚠️ Khác với ngưỡng tự động ở trên "
                     f"({'CÓ' if nguong else 'KHÔNG'}) — có chủ ý, không phải lỗi.")
    d.append("")

    noi_dung = "\n".join(d)
    with open(duong_dan, "w", encoding="utf-8") as f:
        f.write(noi_dung)
    return noi_dung


# ===========================================================================
# main
# ===========================================================================

def main():
    # Console Windows mặc định là cp1252, không mã hoá được chữ có dấu: in một dòng
    # tiếng Việt là ném UnicodeEncodeError ngay, kể cả `--help`. Ép hai luồng ra
    # UTF-8 để script chạy được cả trên máy lẫn trên Colab. Trên Colab vốn đã là
    # UTF-8 nên bước này không đổi gì.
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass  # luồng đã bị thay bằng thứ khác (bị chuyển hướng, bị bọc) — bỏ qua

    ap = argparse.ArgumentParser(description="Khảo sát dữ liệu FloodNet (Phase 1)")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--zip", dest="zip_path", default=None, help="Ghi đè đường dẫn zip")
    ap.add_argument("--output-dir", default=None, help="Ghi đè nơi ghi kết quả")
    ap.add_argument("--work-dir", default=None,
                    help="Ghi đè thư mục trung gian. Dùng thư mục riêng khi chạy "
                         "thử, để bản ghi dở không lẫn vào kết quả chạy thật.")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Chỉ xử lý N ảnh đầu mỗi split (chạy thử)")
    ap.add_argument("--fresh", action="store_true",
                    help="Bỏ qua kết quả lần chạy trước, làm lại từ đầu")
    args = ap.parse_args()

    # In băng-rôn TRƯỚC khi mở file config. Nếu để sau, một lỗi sớm (thiếu file
    # config chẳng hạn) làm script thoát mà stdout TRỐNG TRƠN — nhìn vào chỉ thấy
    # "mã thoát 1" mà không biết vì sao. Đúng lỗi đã gặp ngày 30/09/2026.
    print("=" * 72)
    print("KHẢO SÁT DỮ LIỆU FLOODNET — PHASE 1")
    print("=" * 72)

    if not os.path.exists(args.config):
        print(f"[!] Không thấy file cấu hình:\n    {args.config}\n"
              f"    Thường là repo clone về chưa có thư mục configs/. "
              f"Chạy lại ô [1.3] trong notebook để lấy code mới nhất.")
        raise SystemExit(1)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    zip_path = args.zip_path or cfg["paths"]["zip"]
    output_dir = args.output_dir or cfg["paths"]["output_dir"]
    work_dir = args.work_dir or cfg["paths"]["work_dir"]
    classes = {int(k): v for k, v in cfg["preprocess"]["classes"].items()}
    water_value = cfg["preprocess"]["water_value"]
    target_long_side = cfg["preprocess"]["target_long_side"]
    decisions = cfg["decisions"]
    eda = cfg["eda"]
    max_images = args.max_images if args.max_images is not None else eda["max_images"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    duong_jsonl = os.path.join(work_dir, "audit.jsonl")

    if not os.path.exists(zip_path):
        # In ra stdout rồi mới thoát: SystemExit(msg) đẩy thông báo sang stderr,
        # mà người dùng cần copy được nguyên văn để gửi lại khi nhờ trợ giúp.
        print(f"[!] Không thấy file zip:\n    {zip_path}\n"
              f"    Kiểm tra Drive đã mount chưa, và đường dẫn trong {args.config}.")
        raise SystemExit(1)

    print(f"zip     : {zip_path}")
    print(f"kết quả : {output_dir}")
    print(f"resize  : cạnh dài {target_long_side}px")
    print(f"lớp     : {classes}")

    with zipfile.ZipFile(zip_path) as zf:
        ds_ten = zf.namelist()

        # ---------------------------------------------------------------
        # Bước 1: cấu trúc thật của zip
        # ---------------------------------------------------------------
        print("\n--- BƯỚC 1: CẤU TRÚC ZIP ---")
        cay = liet_ke_cay(ds_ten)
        for thu_muc, so_file in cay.items():
            print(f"  {so_file:>6} file  {thu_muc}/")

        thong_tin = phan_loai_thu_muc(zf, cay, ds_ten)
        print("\n  Phân loại từng thư mục (giải mã thật, không nhìn tên):")
        for thu_muc, tt in sorted(thong_tin.items()):
            mo_ta = f"{tt['kenh']} kênh"
            if tt["kich_thuoc"]:
                mo_ta += f", {tt['kich_thuoc'][1]}x{tt['kich_thuoc'][0]}"
            print(f"    [{tt['loai']:>4}] {mo_ta:<22} {thu_muc}/")

        ds_cap, canh_bao = ghep_cap_anh_mask(thong_tin, ds_ten)
        if canh_bao:
            print("\n  CẢNH BÁO:")
            for c in canh_bao:
                print(f"    {c}")

        if not ds_cap:
            print("\n[!] Không ghép được cặp ảnh/mask nào.\n"
                  "    Đọc phần CẤU TRÚC ZIP ở trên rồi báo lại để chỉnh script.")
            raise SystemExit(1)

        print("\n  Ghép được các split:")
        for cha, ds in ds_cap:
            print(f"    {cha}/  ->  {len(ds)} cặp ảnh/mask")

        # ---------------------------------------------------------------
        # Bước 2: xử lý từng ảnh (có resume)
        # ---------------------------------------------------------------
        print("\n--- BƯỚC 2: TÁCH COMPONENT & THỐNG KÊ ---")
        da_co = {} if args.fresh else doc_ket_qua_da_co(duong_jsonl)
        if da_co:
            print(f"  Đọc lại {len(da_co)} ảnh đã xử lý lần trước "
                  f"(dùng --fresh nếu muốn làm lại từ đầu)")

        tong_cap = sum(min(len(ds), max_images) if max_images else len(ds)
                       for _, ds in ds_cap)
        print(f"  Sẽ xử lý {tong_cap} cặp ảnh/mask")

        # Mẫu kiểm chứng nước — seed cố định để chạy lại ra đúng kết quả cũ
        rng = random.Random(eda["seed"])
        tat_ca_ten = [t for _, ds in ds_cap for t, _ in ds]
        mau_nuoc = set(rng.sample(tat_ca_ten, min(eda["num_water_check"], len(tat_ca_ten))))

        with open(duong_jsonl, "a", encoding="utf-8") as f_jsonl:
            dem = 0
            for _, ds in ds_cap:
                for ten_anh, ten_mask in (ds[:max_images] if max_images else ds):
                    if ten_anh in da_co:
                        continue
                    r = xu_ly_mot_anh(zf, ten_anh, ten_mask, classes, water_value,
                                      target_long_side, kiem_tra_nuoc=(ten_anh in mau_nuoc))
                    if r is None:
                        print(f"    [!] đọc lỗi, bỏ qua: {ten_anh}")
                        continue
                    dem += 1
                    f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f_jsonl.flush()   # ghi ngay: Colab ngắt giữa chừng vẫn giữ được
                    if dem % 50 == 0:
                        print(f"    ... đã xử lý {dem}/{tong_cap}")

        ds_ban_ghi = list(doc_ket_qua_da_co(duong_jsonl).values())
        print(f"  Xong: {len(ds_ban_ghi)} bản ghi")

        # ---------------------------------------------------------------
        # Bước 3: tổng hợp & kết luận
        # ---------------------------------------------------------------
        print("\n--- BƯỚC 3: TỔNG HỢP & KẾT LUẬN ---")
        kq = tong_hop(ds_ban_ghi, classes, decisions, cfg["preprocess"])
        print(ghi_bao_cao(kq, os.path.join(output_dir, "EDA_REPORT.md")))

        # ---------------------------------------------------------------
        # Bước 4: ảnh overlay
        # ---------------------------------------------------------------
        print("\n--- BƯỚC 4: ẢNH OVERLAY ---")
        thu_muc_overlay = os.path.join(output_dir, "overlay")
        os.makedirs(thu_muc_overlay, exist_ok=True)

        co_box = [r for r in ds_ban_ghi
                  if sum(l["so_component"] for l in r.get("lop", {}).values()) > 0]
        rng2 = random.Random(eda["seed"])
        so_ve, canh_bao_kt = 0, []
        for r in rng2.sample(co_box, min(eda["num_overlays"], len(co_box))):
            ok, cb = ve_overlay(zf, r["anh"], r, thu_muc_overlay,
                                decisions["area_outlier_ratio"],
                                kq.get("trung_vi_dien_tich_loc"),
                                decisions["min_side_px"])
            so_ve += int(ok)
            if cb and cb not in canh_bao_kt:
                canh_bao_kt.append(cb)

        print(f"  Đã vẽ {so_ve} ảnh vào {thu_muc_overlay}/")
        print(f"  Box viền TÍM = diện tích > {decisions['area_outlier_ratio']}x "
              f"trung vị (nghi nhiều nhà bị gộp)")
        for cb in canh_bao_kt:
            print(f"  [!] {cb}")

    print("\n" + "=" * 72)
    print("XONG. Gửi lại cho Claude:")
    print("  1. Toàn bộ phần BƯỚC 1 (cấu trúc zip + cảnh báo)")
    print(f"  2. Nội dung {os.path.join(output_dir, 'EDA_REPORT.md')}")
    print(f"  3. Mở {thu_muc_overlay}/ xem bằng mắt, mô tả box có khớp nhà không")
    print("=" * 72)


if __name__ == "__main__":
    main()

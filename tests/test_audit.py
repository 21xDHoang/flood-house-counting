# -*- coding: utf-8 -*-
"""Test cho src/floodcount/data/audit.py — chạy trên máy CPU, không cần GPU, không cần dataset thật.

Ý tưởng: dựng một file zip GIẢ có cấu trúc y hệt FloodNet thật, trong đó có cả
thư mục bẫy `ColorMasks-FloodNetv1.0` (tên có chữ "mask" nhưng bên trong là ẢNH
MÀU). Mask giả được vẽ bằng hình chữ nhật đã biết trước toạ độ, nên mọi con số
script tính ra đều kiểm được bằng tay.

Điều quan trọng nhất test này phải chứng minh: script KHÔNG chọn nhầm thư mục
ColorMasks làm mask. Nếu chọn nhầm, đồ án sẽ train trên dữ liệu sai mà không hề
có thông báo lỗi — đó là loại lỗi tốn kém nhất.

Chạy:
    py tests/test_audit.py
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

GOC_REPO = Path(__file__).resolve().parents[1]

# Kích thước ảnh giả. Dùng 400x300 (tỉ lệ 4:3 y như 4000x3000 thật) và đặt
# target_long_side=200 để phép resize có ý nghĩa mà vẫn chạy nhanh.
H, W = 300, 400
TARGET = 200
TY_LE = TARGET / W          # = 0.5


# ===========================================================================
# Dựng dữ liệu giả
# ===========================================================================

def tao_mask(loai_anh):
    """Vẽ mask giả bằng hình chữ nhật có toạ độ biết trước (giá trị 0 = nền).

    Ảnh 1 — kiểm tra chạm nước và đếm component cơ bản:
      - lớp 1 (nhà ngập)      : chữ nhật (20, 20, 60, 40)  -> CHẠM nước
      - lớp 2 (nhà không ngập): chữ nhật (200, 20, 80, 60) -> KHÔNG chạm nước
      - lớp 5 (nước)          : chữ nhật (10, 60, 40, 10), kề cạnh dưới của lớp 1

    Ảnh 2 — kiểm tra hai nhà DÍNH nhau và nhiễu nhỏ:
      - lớp 2: hai chữ nhật (10, 10, 40, 20) và (50, 10, 40, 20) kề nhau tại x=50
               -> phải ra ĐÚNG 1 component (FloodNet không có nhãn instance)
      - lớp 1: một đốm 2x2 pixel ở (300, 200) -> nhiễu, min_area phải lọc được
      - không có lớp nước
    """
    mask = np.zeros((H, W), np.uint8)
    if loai_anh == 1:
        mask[20:60, 20:80] = 1          # lớp 1: y 20..60, x 20..80
        mask[20:80, 200:280] = 2        # lớp 2: y 20..80, x 200..280
        mask[60:70, 10:50] = 5          # lớp 5: nước, kề đáy lớp 1 tại y=60
    else:
        mask[10:30, 10:50] = 2          # lớp 2, nửa trái
        mask[10:30, 50:90] = 2          # lớp 2, nửa phải — kề nhau tại x=50
        mask[200:202, 300:302] = 1      # nhiễu 2x2
    return mask


def ma_hoa_png(mang):
    ok, buf = cv2.imencode(".png", mang)
    assert ok, "khong ma hoa duoc PNG"
    return buf.tobytes()


def ma_hoa_jpg(mang):
    ok, buf = cv2.imencode(".jpg", mang, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok, "khong ma hoa duoc JPEG"
    return buf.tobytes()


def tao_zip_gia(duong_dan_zip):
    """Dựng zip giả có cấu trúc giống FloodNet thật, kèm thư mục bẫy ColorMasks."""
    anh_mau = np.full((H, W, 3), 180, np.uint8)   # ảnh xám nhạt, đủ để giải mã

    with zipfile.ZipFile(duong_dan_zip, "w", zipfile.ZIP_DEFLATED) as z:
        # --- Cấu trúc và CÁCH ĐẶT TÊN y hệt zip thật (đo được 30/09/2026) ---
        #   ảnh : FloodNet-Supervised_v1.0/train/train-org-img/1234.jpg
        #   mask: FloodNet-Supervised_v1.0/train/train-label-img/1234_lab.png
        # Hậu tố "_lab" là chi tiết đã làm hỏng lần chạy thật đầu tiên: bản test
        # trước đây đặt hai bên trùng tên nên không bắt được lỗi này.
        for split, so_anh in [("train", 2), ("val", 1), ("test", 2)]:
            for i in range(1, so_anh + 1):
                loai = i if i <= 2 else 1
                z.writestr(f"FloodNet-Supervised_v1.0/{split}/{split}-org-img/{i}.jpg",
                           ma_hoa_jpg(anh_mau))
                z.writestr(f"FloodNet-Supervised_v1.0/{split}/{split}-label-img/"
                           f"{i}_lab.png", ma_hoa_png(tao_mask(loai)))

        # --- BẪY: tên có chữ "mask" nhưng là ẢNH MÀU 3 kênh, không phải nhãn.
        # Thật ra có BA thư mục anh em ruột cùng cha, không phải một. ---
        anh_mau_lon = np.full((1024, 1024, 3), 120, np.uint8)
        for ten_split in ("TrainSet", "ValSet", "TestSet"):
            for i in range(1, 3):
                z.writestr(f"ColorMasks-FloodNetv1.0/ColorMasks-{ten_split}/{i}.png",
                           ma_hoa_png(anh_mau_lon))

    return duong_dan_zip


def tao_zip_sach(duong_dan_zip):
    """Zip giả CHỈ có lớp 1, toàn nhà to, kèm đúng một đốm nhiễu nhỏ.

    Bộ dữ liệu giả chính có đủ hai lớp nên không dựng lại được hai tình huống đã
    gặp thật ngày 30/09/2026 trên dữ liệu FloodNet:

      - Nhà to nhưng lẫn một đốm nhiễu -> percentile 5 THÔ tụt xuống vài pixel.
        Kết luận tiling phải đổi theo nếu script biết lọc nhiễu trước.
      - Một lớp khai báo trong config nhưng vắng mặt hoàn toàn trong dữ liệu.

    Đây là zip riêng, nhỏ, chạy nhanh, để hai tình huống đó có test khoá lại.
    """
    anh_mau = np.full((H, W, 3), 180, np.uint8)
    mask = np.zeros((H, W), np.uint8)
    mask[20:100, 20:100] = 1        # nhà to: 80x80 -> 40x40 sau resize, cạnh 40px
    mask[200:202, 300:302] = 1      # đốm nhiễu 2x2 -> 1x1 sau resize, cạnh 1px

    with zipfile.ZipFile(duong_dan_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("FloodNet-Supervised_v1.0/train/train-org-img/1.jpg",
                   ma_hoa_jpg(anh_mau))
        z.writestr("FloodNet-Supervised_v1.0/train/train-label-img/1_lab.png",
                   ma_hoa_png(mask))
    return duong_dan_zip


def tao_zip_median(duong_dan_zip):
    """Zip giả: nhà TO lẫn NHIỀU đốm nhiễu — khoá mốc "3x trung vị".

    Lần chạy đầy đủ 30/09/2026 cho ra non_flooded_building bị gắn cờ "to bất
    thường" 1096/3985 box còn flooded_building chỉ 14 — chênh 78 lần. Nếu là nhà
    bị gộp thật thì hai lớp phải na ná nhau, nên con số lệch như vậy nghĩa là cái
    MỐC so sánh hỏng: trung vị tính trên tất cả box (kể cả đốm vài pixel), mà đốm
    kéo trung vị xuống, hạ thấp mốc 3x, và mọi căn nhà to thật đều vượt mốc.

    Dựng đúng tình huống đó: lớp 1 có 2 nhà to + 6 đốm, lớp 2 có 1 nhà to.
      - Trung vị THÔ = 1px (sáu giá trị nhỏ nhất đều là đốm) -> mốc 3px -> cả 3
        nhà to bị gắn cờ -> "Tách nhà dính: CÓ".
      - Trung vị ĐÃ LỌC = 1600px -> mốc 4800 -> không nhà nào bị gắn cờ -> "KHÔNG".
    """

    def tao_mask_median():
        mask = np.zeros((H, W), np.uint8)
        mask[10:90, 10:90] = 1          # nhà to 1 -> 40x40 sau resize, diện tích 1600
        mask[10:90, 200:280] = 1        # nhà to 2
        for i in range(6):              # 6 đốm 2x2 -> 1x1 sau resize, cách nhau 10px
            mask[200:202, 10 + i * 20:12 + i * 20] = 1
        mask[120:200, 300:380] = 2      # lớp 2: một nhà to, để không có lớp vắng mặt
        return mask

    anh_mau = np.full((H, W, 3), 180, np.uint8)
    with zipfile.ZipFile(duong_dan_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("FloodNet-Supervised_v1.0/train/train-org-img/1.jpg",
                   ma_hoa_jpg(anh_mau))
        z.writestr("FloodNet-Supervised_v1.0/train/train-label-img/1_lab.png",
                   ma_hoa_png(tao_mask_median()))
    return duong_dan_zip


def tao_config_gia(duong_dan_yaml, zip_path, output_dir, work_dir):
    """Config tối thiểu, trỏ vào dữ liệu giả. Dùng y hệt khoá của configs/data.yaml."""
    noi_dung = f"""
paths:
  zip: {zip_path.replace(chr(92), '/')}
  output_dir: {output_dir.replace(chr(92), '/')}
  work_dir: {work_dir.replace(chr(92), '/')}
preprocess:
  target_long_side: {TARGET}
  min_area: 64
  classes:
    1: flooded_building
    2: non_flooded_building
  water_value: 5
decisions:
  min_side_px: 8
  min_box_side_p5_ok: 32
  area_outlier_ratio: 3.0
eda:
  num_overlays: 4
  seed: 42
  max_images: null
  num_water_check: 60
"""
    Path(duong_dan_yaml).write_text(noi_dung, encoding="utf-8")
    return duong_dan_yaml


# ===========================================================================
# Chạy & kiểm tra
# ===========================================================================

def chay_audit(config, work_dir, them_arg=None):
    """Chạy scripts/audit.py qua subprocess (test luôn cả đường CLI).

    Ép PYTHONIOENCODING=utf-8: console Windows ở đây là cp1252, in tiếng Việt sẽ
    ném UnicodeEncodeError — lỗi của console chứ không phải lỗi của script.
    """
    lenh = [sys.executable, str(GOC_REPO / "scripts" / "audit.py"), "--config", str(config)]
    if them_arg:
        lenh += them_arg
    moi_truong = dict(os.environ, PYTHONIOENCODING="utf-8")
    kq = subprocess.run(lenh, capture_output=True, cwd=str(GOC_REPO), env=moi_truong)
    return kq.returncode, kq.stdout.decode("utf-8", "replace"), kq.stderr.decode("utf-8", "replace")


def doc_jsonl(work_dir):
    duong_dan = os.path.join(work_dir, "audit.jsonl")
    assert os.path.exists(duong_dan), f"khong thay {duong_dan}"
    ban_ghi = {}
    with open(duong_dan, encoding="utf-8") as f:
        for dong in f:
            dong = dong.strip()
            if dong:
                r = json.loads(dong)
                ban_ghi[r["anh"]] = r
    return ban_ghi


def doc_dong_lop(bao_cao, ten_lop):
    """Tách dòng của một lớp trong bảng 'Thống kê theo lớp' thành list ô đã strip.

    Trả về None nếu không có dòng nào. Thứ tự ô theo đúng header của bảng:
      1=tên lớp, 2=giá trị mask, 3=số box, 4=bỏ do nhiễu, 5=box/ảnh TB,
      6=box/ảnh max, 7=p5 đã lọc, 8=p50 đã lọc, 9=p5 thô, 10=DT p50 thô,
      11=DT p50 đã lọc, 12=box to bất thường.
    Kiểm bằng ô số chứ không so cả dòng: so cả dòng thì mỗi lần đổi định dạng
    hiển thị là test đỏ, mà định dạng hiển thị không phải thứ cần bảo vệ.
    """
    # Cắt lấy phần SAU tiêu đề bảng. Bảng "Số ảnh có / không có từng lớp" cũng có
    # dòng bắt đầu bằng "| flooded_building |" và đứng TRƯỚC bảng này, nên tìm
    # trong cả báo cáo sẽ vớ phải dòng 4 ô của bảng đó.
    phan = bao_cao.split("## Thống kê theo lớp")[-1]
    for d in phan.splitlines():
        if d.startswith(f"| {ten_lop} |"):
            return [o.strip() for o in d.split("|")]
    return None


def co_khoa(cfg, duong_dan_khoa):
    """True nếu config CÓ khoá này — kể cả khi giá trị là `null`.

    Không dùng `.get()` rồi so với None: `eda.max_images: null` là giá trị hợp lệ
    (nghĩa là "chạy hết"), mà `.get()` trả None cho cả trường hợp thiếu khoá lẫn
    trường hợp khoá có giá trị null — hai chuyện hoàn toàn khác nhau.
    """
    nut = cfg
    for phan in duong_dan_khoa.split("."):
        if not isinstance(nut, dict) or phan not in nut:
            return False
        nut = nut[phan]
    return True


def main():
    tmp = tempfile.mkdtemp(prefix="audit_test_")
    loi = []

    def kiem(dieu_kien, mo_ta):
        trang_thai = "OK  " if dieu_kien else "FAIL"
        print(f"  [{trang_thai}] {mo_ta}")
        if not dieu_kien:
            loi.append(mo_ta)

    try:
        zip_path = os.path.join(tmp, "floodnet_gia.zip")
        tao_zip_gia(zip_path)
        output_dir = os.path.join(tmp, "eda")
        work_dir = os.path.join(tmp, "work")
        config = tao_config_gia(os.path.join(tmp, "data.yaml"), zip_path, output_dir, work_dir)

        print("=== 1. Chay audit tren du lieu gia ===")
        ma_tra_ve, stdout, stderr = chay_audit(config, work_dir)
        if ma_tra_ve != 0:
            print(stdout[-3000:])
            print(stderr[-3000:])
            raise SystemExit(f"*** audit.py thoat voi ma {ma_tra_ve} ***")
        print("  audit.py chay xong, ma tra ve 0")

        print("\n=== 2. Bay ColorMasks phai bi chan ===")
        # Dòng phân loại có dạng:  "    [ anh] 3 kênh, 1024x1024   ColorMasks.../"
        # (dòng ở phần CÂY chỉ là "3 file  ColorMasks.../" nên phải lọc theo "kênh")
        dong_phan_loai = [d for d in stdout.splitlines()
                          if "ColorMasks-FloodNetv1.0/" in d and "kênh" in d]
        # Ba thư mục ColorMasks-*Set là anh em ruột, CẢ BA đều phải được phân loại.
        # Bản cũ gom theo thư mục cha bằng dict đơn nên hai thư mục sau đè mất thư
        # mục đầu — test cũ chỉ có một thư mục nên không lộ ra.
        kiem(len(dong_phan_loai) == 3,
             f"phan loai du ca 3 thu muc ColorMasks, thuc te {len(dong_phan_loai)}")
        for d in dong_phan_loai:
            kiem("[ anh]" in d,
                 f"ColorMasks bi phan loai la ANH MAU, khong phai mask -> {d.strip()}")
            kiem("3 kênh" in d, f"ColorMasks nhan dien dung 3 kenh -> {d.strip()}")
        kiem(all("ColorMasks-" + s in stdout for s in ("TrainSet", "ValSet", "TestSet")),
             "ca ba thu muc TrainSet/ValSet/TestSet deu xuat hien trong phan loai")
        kiem(not any("ColorMasks" in d and "[ mask]" in d for d in stdout.splitlines()),
             "KHONG co dong nao phan loai ColorMasks la mask")
        kiem("bẫy ColorMasks" in stdout, "co canh bao tu khoa 'bay ColorMasks'")
        kiem("FloodNet-Supervised_v1.0/train/" in stdout, "tim thay split train")
        kiem("FloodNet-Supervised_v1.0/val/" in stdout, "tim thay split val")
        kiem("FloodNet-Supervised_v1.0/test/" in stdout, "tim thay split test")

        print("\n=== 3. So component phai khop hinh ve tay ===")
        ban_ghi = doc_jsonl(work_dir)
        # Con số này là phép thử thật của việc ghép cặp: ảnh "1.jpg" phải tìm được
        # mask "1_lab.png" (bỏ hậu tố '_lab'). Quên quy tắc đó thì ban_ghi rỗng
        # trơn, và mọi mục bên dưới đều đổ theo.
        kiem(len(ban_ghi) == 5,
             f"ghep duoc 5 cap anh/mask qua hau to '_lab' "
             f"(train 2 + val 1 + test 2), thuc te {len(ban_ghi)}")

        # Ảnh 1: lớp 1 có 1 component, lớp 2 có 1 component
        r1 = ban_ghi["FloodNet-Supervised_v1.0/train/train-org-img/1.jpg"]
        kiem(r1["lop"]["flooded_building"]["so_component"] == 1,
             "anh 1: lop 1 co dung 1 component")
        kiem(r1["lop"]["non_flooded_building"]["so_component"] == 1,
             "anh 1: lop 2 co dung 1 component")

        # Toạ độ sau resize: scale 0.5 -> lớp 1 thành (10,10,30,20), diện tích 600
        box1 = r1["lop"]["flooded_building"]["boxes"][0]
        kiem(box1 == [10, 10, 30, 20, 600],
             f"anh 1: box lop 1 sau resize = [10,10,30,20,600], thuc te {box1}")
        kiem(r1["kich_thuoc_resize"] == [150, 200],
             f"anh 1: kich thuoc sau resize = 150x200, thuc te {r1['kich_thuoc_resize']}")

        # Ảnh 2: hai chữ nhật kề nhau -> PHẢI gộp thành 1 component
        r2 = ban_ghi["FloodNet-Supervised_v1.0/train/train-org-img/2.jpg"]
        kiem(r2["lop"]["non_flooded_building"]["so_component"] == 1,
             "anh 2: hai hinh ke nhau GOM thanh 1 component (FloodNet khong co nhan instance)")
        # Bao trùm cả hai hình: rộng 80 -> 40 sau resize, cao 20 -> 10, diện tích 400
        box2 = r2["lop"]["non_flooded_building"]["boxes"][0]
        kiem(box2 == [5, 5, 40, 10, 400],
             f"anh 2: box gop = [5,5,40,10,400], thuc te {box2}")

        # Ảnh 2: đốm nhiễu 2x2 -> 1 component riêng, diện tích tí hon
        kiem(r2["lop"]["flooded_building"]["so_component"] == 1,
             "anh 2: dom nhieu 2x2 thanh 1 component rieng")
        kiem(r2["lop"]["flooded_building"]["boxes"][0][4] <= 4,
             "anh 2: dien tich dom nhieu <= 4 px -> min_area loc duoc")

        print("\n=== 4. Kiem chung 'nha ngap nam canh nuoc' ===")
        bao_cao = Path(output_dir, "EDA_REPORT.md").read_text(encoding="utf-8")
        kiem("Kiểm chứng lớp 1 = nhà NGẬP" in bao_cao, "bao cao co muc kiem chung nuoc")
        # 3 ảnh có nước (train/1, val/1, test/1); mỗi ảnh có 1 component mỗi lớp
        kiem("| flooded_building | 3 | 3 | 100.0% |" in bao_cao,
             "lop 1: 3/3 component cham nuoc (100%)")
        kiem("| non_flooded_building | 3 | 0 | 0.0% |" in bao_cao,
             "lop 2: 0/3 component cham nuoc (0%)")
        # Phép kiểm nước chỉ chạy trên ảnh có nước (3/5 ảnh ở đây) — mẫu nhỏ thì
        # tỉ lệ chỉ nói lên HƯỚNG, nên báo cáo phải nói rõ mẫu gồm mấy ảnh. Bản
        # đầu không in con số này, đọc bảng xong tưởng 3 component là toàn bộ dữ liệu.
        kiem("Chỉ chạy trên **3 ảnh** có vùng nước" in bao_cao,
             "bao cao ghi ro phep kiem nuoc chi chay tren 3 anh co nuoc")

        print("\n=== 5. Ket luan tiling dua tren so lieu DA LOC NHIEU ===")
        # Ảnh 2 có một đốm nhiễu 2x2 ở lớp 1. Không lọc thì percentile 5 thô của
        # lớp 1 bị kéo xuống ~2px và kết luận "phải cắt tile" là do RÁC quyết định,
        # không phải do nhà — đúng cái bẫy đã gặp thật ngày 30/09/2026 (p5 = 1.0px
        # trong khi p50 = 118px). Bảng phải có đủ hai cột để đối chiếu được.
        o1 = doc_dong_lop(bao_cao, "flooded_building")
        o2 = doc_dong_lop(bao_cao, "non_flooded_building")
        # Lớp 1 có 5 box trong 5 ảnh: 3 chữ nhật to (train/1, val/1, test/1) và 2 đốm
        # nhiễu (train/2, test/2). Chỉ hai đốm bị lọc.
        kiem(o1 is not None and o1[3] == "5" and o1[4] == "2",
             f"lop 1: 5 box, dung 2 dom nhieu bi loc, thuc te {o1[3:5] if o1 else None}")
        kiem(o2 is not None and o2[4] == "0",
             f"lop 2: khong box nao bi loc, thuc te {o2[4] if o2 else None}")
        kiem(o1 is not None and float(o1[9]) < float(o1[7]),
             f"p5 tho ({o1[9] if o1 else '?'}) nho hon han p5 da loc "
             f"({o1[7] if o1 else '?'}) -> hai cot phai khac nhau")
        kiem("Cắt tile: CÓ" in bao_cao,
             "canh p5 DA LOC nho nhat = 10px < 32px -> ket luan PHAI cat tile")
        kiem("min_area đề xuất: 64" in bao_cao,
             "min_area de xuat = min_side^2 = 8^2 = 64")

        # Bảng "ảnh có / không có lớp": lớp 1 chỉ có box HỢP LỆ ở 3 ảnh loại 1
        # (train/1, val/1, test/1); hai ảnh loại 2 chỉ có một đốm 2x2 nên KHÔNG
        # tính là "ảnh có nhà ngập". Đếm theo box thô thì ra 5/5 và bảng này thôi
        # nói lên điều gì — mà đây đúng là con số dùng để chia train/val.
        kiem("## Số ảnh có / không có từng lớp" in bao_cao,
             "bao cao co bang so anh co/khong co tung lop")
        kiem("| flooded_building | 3 (60.0%) | 2 (40.0%) |" in bao_cao,
             "lop 1: 3/5 anh co nha that, 2 anh chi co dom nhieu")
        kiem("| non_flooded_building | 5 (100.0%) | 0 (0.0%) |" in bao_cao,
             "lop 2: ca 5 anh deu co nha that")

        # Bảng theo split: khi một lớp vắng mặt thì câu hỏi đầu tiên luôn là "vắng ở
        # mọi split hay chỉ một split". Trả lời sẵn trong báo cáo để khỏi phải chạy
        # lại toàn bộ dữ liệu chỉ để hỏi một câu.
        kiem("## Số pixel mỗi giá trị mask theo split" in bao_cao,
             "bao cao co bang so pixel theo tung split")
        kiem("| Giá trị | train | val | test |" in bao_cao,
             "cot xep theo thu tu train/val/test, khong xep alphabet")
        # Mẫu số phải là tổng số pixel CỦA TỪNG SPLIT, không phải tổng chung: train
        # và test mỗi bên 2 ảnh, val chỉ 1 ảnh, nên cùng 400 pixel nước mà tỉ lệ của
        # val phải gấp đôi. Dùng tổng chung thì ba ô sẽ bằng nhau và test này đỏ.
        # Cắt lấy phần SAU tiêu đề: bảng tổng cũng có dòng "| 5 |" và nó đứng trước,
        # tìm trong cả báo cáo thì vớ phải dòng của bảng tổng (3 cột, có dấu phẩy).
        phan_split = bao_cao.split("## Số pixel mỗi giá trị mask theo split")[-1]
        dong_nuoc = next((d for d in phan_split.splitlines() if d.startswith("| 5 |")), None)
        kiem(dong_nuoc is not None, "bang theo split co dong cua gia tri 5 (nuoc)")
        if dong_nuoc:
            ty = [float(x.strip().rstrip("%")) for x in dong_nuoc.split("|")[2:5]]
            kiem(ty[0] == ty[2] > 0,
                 f"nuoc: train ({ty[0]}%) = test ({ty[2]}%) — hai split cung so anh")
            kiem(abs(ty[1] - 2 * ty[0]) < 0.001,
                 f"nuoc: val ({ty[1]}%) = 2x train ({ty[0]}%) — val chi co 1 anh, "
                 f"mau so chia theo tung split chu khong dung tong chung")

        print("\n=== 6. Anh overlay duoc tao, KHONG bi ghi de ===")
        thu_muc_overlay = Path(output_dir, "overlay")
        ds_jpg = sorted(p.name for p in thu_muc_overlay.glob("*.jpg")) \
            if thu_muc_overlay.exists() else []
        # Các split đều đặt tên ảnh là 1.jpg / 2.jpg, nên tên file overlay PHẢI
        # giữ tên split, nếu không 4 ảnh sẽ ghi đè nhau còn 2.
        kiem(len(ds_jpg) == 4, f"ve du 4 anh overlay (khong trung ten), thuc te {len(ds_jpg)}: {ds_jpg}")
        kiem(len(set(ds_jpg)) == len(ds_jpg), "khong co ten file overlay nao trung nhau")
        for split in ("train", "val", "test"):
            kiem(any(split in t for t in ds_jpg),
                 f"ten file overlay co giu ten split '{split}'")
        if ds_jpg:
            # Ảnh phải có nội dung thật, không phải file rỗng
            co_noi_dung = [p for p in thu_muc_overlay.glob("*.jpg") if p.stat().st_size > 1000]
            kiem(len(co_noi_dung) == len(ds_jpg), "moi anh overlay deu > 1KB (co noi dung)")

        print("\n=== 7. Chay lai phai resume, khong lam lai tu dau ===")
        ma_tra_ve2, stdout2, _ = chay_audit(config, work_dir)
        kiem(ma_tra_ve2 == 0, "chay lai lan 2 van thanh cong")
        kiem("Đọc lại 5 ảnh đã xử lý" in stdout2,
             "lan 2 doc lai 5 anh cu thay vi xu ly lai")
        kiem(len(doc_jsonl(work_dir)) == 5, "so ban ghi khong bi nhan doi")

        print("\n=== 8. --fresh phai lam lai tu dau ===")
        ma_tra_ve3, stdout3, _ = chay_audit(config, work_dir, ["--fresh"])
        kiem(ma_tra_ve3 == 0, "--fresh chay thanh cong")
        kiem("Đọc lại" not in stdout3, "--fresh khong doc lai ket qua cu")
        kiem(len(doc_jsonl(work_dir)) == 5,
             "sau --fresh van dung 5 ban ghi (khong bi ghi trung)")

        print("\n=== 9. --max-images gioi han so anh ===")
        work_dir2 = os.path.join(tmp, "work2")
        config2 = tao_config_gia(os.path.join(tmp, "data2.yaml"), zip_path,
                                 os.path.join(tmp, "eda2"), work_dir2)
        ma_tra_ve4, stdout4, _ = chay_audit(config2, work_dir2, ["--max-images", "1"])
        kiem(ma_tra_ve4 == 0, "--max-images chay thanh cong")
        kiem(len(doc_jsonl(work_dir2)) == 3,
             f"--max-images 1 -> 3 anh (1 moi split), thuc te {len(doc_jsonl(work_dir2))}")

        print("\n=== 10. Zip khong ton tai phai bao loi ro rang ===")
        config3 = tao_config_gia(os.path.join(tmp, "data3.yaml"),
                                 os.path.join(tmp, "khong-ton-tai.zip"),
                                 os.path.join(tmp, "eda3"), os.path.join(tmp, "work3"))
        ma_tra_ve5, stdout5, _ = chay_audit(config3, os.path.join(tmp, "work3"))
        kiem(ma_tra_ve5 != 0, "zip khong ton tai -> thoat voi ma khac 0")
        kiem("Không thấy file zip" in stdout5, "co thong bao ro rang ve file zip thieu")

        # Lỗi gặp thật ngày 30/09/2026, tốn một vòng hỏi đáp mới tìm ra: thiếu file
        # config làm script thoát với stdout TRỐNG TRƠN, vì dòng print đầu tiên nằm
        # sau lệnh mở file. Trên Colab còn tệ hơn — traceback đi vào stderr của tiến
        # trình con, mà Colab không hiện stderr của tiến trình con. Nhìn vào chỉ thấy
        # "Mã thoát: 1" và không có manh mối nào.
        config4 = os.path.join(tmp, "khong-co-file-nay.yaml")
        ma_tra_ve6, stdout6, _ = chay_audit(config4, os.path.join(tmp, "work4"))
        kiem(ma_tra_ve6 != 0, "thieu config -> thoat voi ma khac 0")
        kiem(stdout6.strip() != "",
             "thieu config -> VAN IN RA stdout (khong duoc im lang)")
        kiem("Không thấy file cấu hình" in stdout6,
             "thieu config -> noi ro thieu file nao")
        kiem("configs" in stdout6,
             "thieu config -> goi y chay lai o [1.3] de clone code moi")

        print("\n=== 11. Notebook 01 phai khop voi CLI cua audit.py ===")
        # Đây là loại lỗi chỉ lộ ra khi đã ngồi chờ trên Colab: notebook truyền cờ
        # mà script không có, hoặc ngược lại. Kiểm ngay trên máy cho rẻ.
        nb = json.loads((GOC_REPO / "notebooks" / "01_data_prep.ipynb")
                        .read_text(encoding="utf-8"))
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                compile("".join(c["source"]), f"nb-cell-{i}", "exec")
        kiem(True, "moi o code cua notebook 01 compile duoc")

        # Trên Colab, fd 1 của tiến trình con KHÔNG chảy vào ô output: script chạy
        # xong, mã thoát 0, mà ô output trống trơn. Dính ba lần ngày 30/09/2026 mới
        # nhận ra quy luật, nên phải khoá lại bằng test.
        # Bỏ dòng comment: chính comment giải thích cũng nhắc lại các chuỗi này.
        ma_nguon_tam = "\n".join(
            d for d in ("\n".join("".join(c["source"]) for c in nb["cells"]
                                  if c["cell_type"] == "code")).splitlines()
            if not d.lstrip().startswith("#"))
        kiem("def chay_hien_dan(" in ma_nguon_tam,
             "co ham chay_hien_dan tu doc ong roi in qua stdout cua kernel")
        so_lan_goi = ma_nguon_tam.count("chay_hien_dan(lenh_")
        kiem(so_lan_goi == 2,
             f"ca HAI o chay deu goi chay_hien_dan (thuc te {so_lan_goi})")
        so_lan_u = ma_nguon_tam.count('"-u"')
        kiem(so_lan_u == 2,
             f"ca hai lenh chay deu truyen -u cho python con (thuc te {so_lan_u})")
        # Chỉ soi hai lệnh chạy dài. `chay()` của ô [1.3] vẫn dùng subprocess.run
        # nhưng có capture_output rồi tự print, nên nó hiện ra bình thường.
        kiem(not re.search(r"subprocess\.run\(\s*lenh_(thu|day_du)", ma_nguon_tam),
             "khong chay lenh dai bang subprocess.run thua huong fd 1")

        # Bỏ các dòng gọi git: `--oneline` là cờ của git chứ không phải của audit.py
        ma_nguon = "\n".join("".join(c["source"]) for c in nb["cells"]
                             if c["cell_type"] == "code")
        dong_python = [d for d in ma_nguon.splitlines() if "git" not in d]
        co_dung = sorted(set(re.findall(r'"(--[a-z][a-z-]+)"', "\n".join(dong_python))))
        kiem(len(co_dung) >= 3, f"notebook co dung it nhat 3 co CLI, thuc te {co_dung}")
        # Ép console giả về cp1252 — đúng như console Windows thật — để chắc rằng
        # script tự lo được chuyện mã hoá chữ có dấu, không phải nhờ Colab UTF-8.
        # Không dùng text=True: ta tự giải mã utf-8 để kiểm luôn phần mã hoá.
        kq_help = subprocess.run(
            [sys.executable, str(GOC_REPO / "scripts" / "audit.py"), "--help"],
            capture_output=True, cwd=str(GOC_REPO),
            env=dict(os.environ, PYTHONIOENCODING="cp1252"))
        kiem(kq_help.returncode == 0,
             "audit.py --help khong crash du console la cp1252 (Windows)")
        tro_giup = kq_help.stdout.decode("utf-8", errors="replace")
        kiem("Khảo sát" in tro_giup, "chu co dau ra dung utf-8, khong bi thay bang '?'")
        for co in co_dung:
            kiem(co in tro_giup, f"notebook dung co {co}, va co nay CO THAT trong audit.py")

        print("\n=== 12. Loc nhieu phai DOI DUOC ket luan + canh bao lop vang mat ===")
        # Hai chuyện này chỉ lộ ra khi chạy trên dữ liệu THẬT (30/09/2026), vì bộ
        # dữ liệu giả chính vừa đủ hai lớp vừa không có ca "nhà to lẫn đốm rác".
        # Không dựng test cho chúng thì lần sau sửa code lại trôi mất.
        zip_sach = tao_zip_sach(os.path.join(tmp, "sach.zip"))
        work_sach = os.path.join(tmp, "work_sach")
        output_sach = os.path.join(tmp, "eda_sach")
        config_sach = tao_config_gia(os.path.join(tmp, "sach.yaml"), zip_sach,
                                     output_sach, work_sach)
        ma_sach, stdout_sach, stderr_sach = chay_audit(config_sach, work_sach)
        if ma_sach != 0:
            print(stdout_sach[-3000:])
            print(stderr_sach[-3000:])
            raise SystemExit(f"*** audit.py tren zip 'sach' thoat voi ma {ma_sach} ***")

        bao_cao_sach = Path(output_sach, "EDA_REPORT.md").read_text(encoding="utf-8")
        dong_sach = doc_dong_lop(bao_cao_sach, "flooded_building")
        kiem(dong_sach is not None and dong_sach[3] == "2" and dong_sach[4] == "1",
             f"lop 1: 2 box, 1 dom nhieu bi loc, "
             f"thuc te {dong_sach[3:5] if dong_sach else None}")
        kiem(dong_sach is not None and float(dong_sach[7]) >= 32,
             f"p5 DA LOC = {dong_sach[7] if dong_sach else '?'}px (nha that canh 40px)")
        kiem(dong_sach is not None and float(dong_sach[9]) < 32,
             f"p5 THO = {dong_sach[9] if dong_sach else '?'}px "
             f"bi dom nhieu keo xuong duoi nguong")
        # Đây mới là điều đáng test: dùng nhầm cột thì kết luận đảo ngược hẳn.
        kiem("Cắt tile: KHÔNG" in bao_cao_sach,
             "ket luan dung p5 DA LOC -> KHONG can cat tile "
             "(lay p5 tho thi da ket luan sai thanh CO)")

        # Lớp vắng mặt: mẫu 9 ảnh thật ngày 30/09/2026 không có pixel nào của lớp 1.
        # Nếu chỉ in ra một dòng "so_box = 0" trong bảng thì rất dễ đọc lướt qua.
        kiem("CẢNH BÁO: có lớp không xuất hiện trong dữ liệu" in bao_cao_sach,
             "bao cao co canh bao lop khai bao ma khong co pixel nao")
        kiem("non_flooded_building (giá trị mask 2)" in bao_cao_sach,
             "canh bao neu DUNG TEN lop vang mat va gia tri mask cua no")
        kiem(bao_cao_sach.index("CẢNH BÁO") < bao_cao_sach.index("## Thống kê theo lớp"),
             "canh bao nam TRUOC bang so lieu, khong phai doc luot qua")
        kiem("| non_flooded_building | 2 | 0 |" in bao_cao_sach,
             "lop vang mat van co dong rieng trong bang, ghi ro 0 box")
        kiem("Số pixel mỗi giá trị mask theo split" not in bao_cao_sach,
             "chi mot split thi KHONG in bang theo split (lap lai bang tong)")

        print("\n=== 13. configs/data.yaml THAT phai co du moi khoa audit.py doc ===")
        # File config thật chỉ được chạy trên Colab, mà vòng sửa-lỗi ở đó rất đắt
        # (một lần chạy là một lần chờ). Thiếu một khoá thì lỗi chỉ lộ ra ở đó, dưới
        # dạng KeyError — nên kiểm ngay trên máy, gần như miễn phí.
        import yaml
        cfg_that = yaml.safe_load((GOC_REPO / "configs" / "data.yaml").read_text(encoding="utf-8"))
        for duong_dan_khoa in [
                "paths.zip", "paths.output_dir", "paths.work_dir",
                "preprocess.target_long_side", "preprocess.min_area",
                "preprocess.classes", "preprocess.water_value",
                "decisions.min_side_px", "decisions.min_box_side_p5_ok",
                "decisions.area_outlier_ratio",
                "eda.num_overlays", "eda.seed", "eda.max_images", "eda.num_water_check"]:
            kiem(co_khoa(cfg_that, duong_dan_khoa),
                 f"configs/data.yaml co khoa {duong_dan_khoa}")
        # min_area phải khớp min_side^2, nếu không thì "nhà hợp lệ" có hai định nghĩa
        # khác nhau trong cùng một pipeline: Phase 1 lọc một kiểu, Phase 2 lọc kiểu khác.
        kiem(cfg_that["preprocess"]["min_area"] == cfg_that["decisions"]["min_side_px"] ** 2,
             "min_area == min_side_px^2 (hai nguong khong mau thuan)")

        print("\n=== 14. Moc '3x trung vi' phai tinh tren trung vi DA LOC ===")
        # Bẫy thứ ba của lần chạy đầy đủ 30/09/2026. Bản đầu đã lọc nhiễu cho tử số
        # (box đem đi so) nhưng vẫn lấy trung vị của TẤT CẢ box làm mốc, nên đốm vài
        # pixel kéo mốc xuống và nhà to thật bị gắn cờ "nghi bị gộp" — đúng thứ làm
        # cho non_flooded_building có 1096 box bị gắn cờ so với 14 của lớp kia.
        zip_med = tao_zip_median(os.path.join(tmp, "median.zip"))
        work_med = os.path.join(tmp, "work_median")
        output_med = os.path.join(tmp, "eda_median")
        config_med = tao_config_gia(os.path.join(tmp, "median.yaml"), zip_med,
                                    output_med, work_med)
        ma_med, stdout_med, stderr_med = chay_audit(config_med, work_med)
        if ma_med != 0:
            print(stdout_med[-3000:])
            print(stderr_med[-3000:])
            raise SystemExit(f"*** audit.py tren zip 'median' thoat voi ma {ma_med} ***")

        bao_cao_med = Path(output_med, "EDA_REPORT.md").read_text(encoding="utf-8")
        dong_med = doc_dong_lop(bao_cao_med, "flooded_building")
        kiem(dong_med is not None and dong_med[3] == "8" and dong_med[4] == "6",
             f"lop 1: 8 box, dung 6 dom nhieu bi loc, "
             f"thuc te {dong_med[3:5] if dong_med else None}")
        # Đây mới là điều đáng test: lấy trung vị thô thì cả 2 nhà to của lớp 1 bị
        # gắn cờ, kết luận đảo thành "CÓ" (2+1 box bị cờ trên 3 box = 100% > 5%).
        kiem(dong_med is not None and dong_med[12] == "0",
             f"moc 3x tinh tren trung vi DA LOC -> khong nha to nao bi gan co, "
             f"thuc te {dong_med[12] if dong_med else '?'} box bi gan co")
        kiem("Tách nhà dính: KHÔNG" in bao_cao_med,
             "lay trung vi tho thi ket luan dao nguoc thanh CO")
        kiem("Cắt tile: KHÔNG" in bao_cao_med,
             "nha 40px sau resize >= 32px -> KHONG can cat tile")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if loi:
        print(f"*** {len(loi)} MUC KHONG DAT ***")
        for m in loi:
            print("   - " + m.encode("ascii", "backslashreplace").decode())
        raise SystemExit(1)
    print("*** TAT CA PASS ***")


if __name__ == "__main__":
    main()

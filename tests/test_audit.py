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
        # --- Cấu trúc thật: mỗi split có thư mục image/ và label/ ---
        for split, so_anh in [("train", 2), ("val", 1), ("test", 2)]:
            for i in range(1, so_anh + 1):
                loai = i if i <= 2 else 1
                z.writestr(f"FloodNet-Supervised_v1.0/{split}/image/{i}.jpg",
                           ma_hoa_jpg(anh_mau))
                z.writestr(f"FloodNet-Supervised_v1.0/{split}/label/{i}.png",
                           ma_hoa_png(tao_mask(loai)))

        # --- BẪY: tên có chữ "mask" nhưng là ẢNH MÀU 3 kênh, không phải nhãn ---
        anh_mau_lon = np.full((1024, 1024, 3), 120, np.uint8)
        for i in range(1, 4):
            z.writestr(f"ColorMasks-FloodNetv1.0/{i}.png", ma_hoa_png(anh_mau_lon))

    return duong_dan_zip


def tao_config_gia(duong_dan_yaml, zip_path, output_dir, work_dir):
    """Config tối thiểu, trỏ vào dữ liệu giả."""
    noi_dung = f"""
paths:
  zip: {zip_path.replace(chr(92), '/')}
  output_dir: {output_dir.replace(chr(92), '/')}
  work_dir: {work_dir.replace(chr(92), '/')}
preprocess:
  target_long_side: {TARGET}
  min_area: 200
  classes:
    1: flooded_building
    2: non_flooded_building
  water_value: 5
decisions:
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
        kiem(len(dong_phan_loai) == 1,
             f"co dung 1 dong phan loai cho ColorMasks, thuc te {len(dong_phan_loai)}")
        if dong_phan_loai:
            kiem("[ anh]" in dong_phan_loai[0],
                 f"ColorMasks bi phan loai la ANH MAU, khong phai mask -> {dong_phan_loai[0].strip()}")
            kiem("3 kênh" in dong_phan_loai[0], "ColorMasks nhan dien dung 3 kenh")
        kiem(not any("ColorMasks" in d and "[ mask]" in d for d in stdout.splitlines()),
             "KHONG co dong nao phan loai ColorMasks la mask")
        kiem("bẫy ColorMasks" in stdout, "co canh bao tu khoa 'bay ColorMasks'")
        kiem("FloodNet-Supervised_v1.0/train/" in stdout, "tim thay split train")
        kiem("FloodNet-Supervised_v1.0/val/" in stdout, "tim thay split val")
        kiem("FloodNet-Supervised_v1.0/test/" in stdout, "tim thay split test")

        print("\n=== 3. So component phai khop hinh ve tay ===")
        ban_ghi = doc_jsonl(work_dir)
        kiem(len(ban_ghi) == 5, f"co 5 anh (train 2 + val 1 + test 2), thuc te {len(ban_ghi)}")

        # Ảnh 1: lớp 1 có 1 component, lớp 2 có 1 component
        r1 = ban_ghi["FloodNet-Supervised_v1.0/train/image/1.jpg"]
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
        r2 = ban_ghi["FloodNet-Supervised_v1.0/train/image/2.jpg"]
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

        print("\n=== 5. Ket luan tiling dua tren so lieu ===")
        kiem("Cắt tile: CÓ" in bao_cao,
             "canh p5 = 1px < 32px -> ket luan PHAI cat tile")
        kiem("min_area đề xuất" in bao_cao, "bao cao co de xuat min_area")

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

        # Colab chỉ nối fd 1 của tiến trình con vào ô output; fd 2 (stderr) đi vào
        # log máy chủ Jupyter và người dùng không thấy. Thiếu stderr=STDOUT thì mọi
        # traceback biến mất, chỉ còn "Mã thoát: 1" — đúng cái bẫy đã sập ngày 30/09.
        # Bỏ dòng comment: chính comment giải thích cũng nhắc lại chuỗi này.
        ma_nguon_tam = "\n".join(
            d for d in ("\n".join("".join(c["source"]) for c in nb["cells"]
                                  if c["cell_type"] == "code")).splitlines()
            if not d.lstrip().startswith("#"))
        so_lan = ma_nguon_tam.count("stderr=subprocess.STDOUT")
        kiem(so_lan == 2,
             f"ca HAI o chay deu truyen stderr=subprocess.STDOUT (thuc te {so_lan})")

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

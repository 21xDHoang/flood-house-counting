# -*- coding: utf-8 -*-
"""Test cho src/floodcount/data/mask_to_coco.py (Phase 2) — CPU, không cần GPU/dataset thật.

Ý tưởng: dựng zip GIẢ có cấu trúc y hệt FloodNet thật, mask vẽ bằng hình chữ nhật
biết trước toạ độ, nên mọi con số script tính ra đều kiểm được bằng tay.

Ba điều test này phải chứng minh, vì chúng là loại lỗi IM LẶNG (sai mà không báo):

  1. HAI LOẠI NHIỄU, HAI ĐIỀU KIỆN LỌC. Một vệt rác 1x200px lọt qua ngưỡng diện
     tích (100 >= 64) nhưng phải bị ngưỡng cạnh chặn; một đường chéo 16x16px lọt
     qua ngưỡng cạnh (16 >= 8) nhưng phải bị ngưỡng diện tích chặn. Chỉ áp dụng
     một trong hai điều kiện thì một trong hai hình đó thành "nhà" và đi thẳng
     vào tập train.
  2. TIỀN TỐ SPLIT TRONG TÊN FILE. Cả ba split đều đặt tên ảnh là `1.jpg`, nên
     nếu không có tiền tố thì ảnh của val ghi đè ảnh của train mà KHÔNG hề báo lỗi
     — dataset thiếu ảnh, số liệu vẫn đẹp.
  3. KHOÁ KHI ĐỌC LẠI .jsonl. Phase 1 ghi khoá `anh`, Phase 2 ghi khoá
     `file_name`. Dùng nhầm khoá thì mọi bản ghi đều bị bỏ qua và dataset ra 0
     ảnh. Đã suýt xảy ra thật ở dòng ráp COCO ngày 30/09/2026.

Chạy:
    py tests/test_mask_to_coco.py
"""

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
sys.path.insert(0, str(GOC_REPO / "src"))

# Kích thước ảnh giả. 400x300 (tỉ lệ 4:3 y như ảnh thật) và target_long_side=200
# để phép resize có ý nghĩa mà vẫn chạy nhanh. Tỉ lệ = 0,5 nên mọi toạ độ sau
# resize đều là toạ độ gốc chia đôi — nhẩm được bằng tay.
H, W = 300, 400
TARGET = 200
TY_LE = TARGET / W          # = 0.5

MIN_SIDE = 8
MIN_AREA = 64


# ===========================================================================
# Dựng dữ liệu giả
# ===========================================================================

def tao_mask_nhieu():
    """Mask lớp 1 có ĐỦ HAI loại nhiễu, mỗi loại lọt qua đúng một điều kiện.

    Toạ độ ghi theo ẢNH GỐC 400x300; sau resize x0,5 thì chia đôi.

      - nhà thật   : chữ nhật (20,20,60,40) -> 30x20, DT 600  -> GIỮ
      - vệt rác    : mask[100:101, 20:220]  -> 100x1, DT 100
                     DT 100 >= 64 nên LỌT ngưỡng diện tích, nhưng cao 1px < 8
                     nên bị ngưỡng cạnh chặn.
      - đường chéo : 16 pixel rời ở (200+2i, 200+2i) -> sau resize thành 16 pixel
                     liền nhau, bbox 16x16, DT 16.
                     Cạnh 16 >= 8 nên LỌT ngưỡng cạnh, nhưng DT 16 < 64 nên bị
                     ngưỡng diện tích chặn.
    """
    mask = np.zeros((H, W), np.uint8)
    mask[20:60, 20:80] = 1
    mask[100:101, 20:220] = 1
    for i in range(16):
        mask[200 + 2 * i, 200 + 2 * i] = 1
    return mask


def tao_mask_dinh():
    """Hai chữ nhật KỀ NHAU của lớp 2 -> phải ra ĐÚNG 1 component.

    FloodNet không có nhãn instance (mỗi pixel chỉ có một giá trị lớp), nên hai
    căn nhà sát nhau là một khối liền. Đây là hành vi đã chốt ở GATE 1 (không
    tách bằng watershed) và Phase 2 phải giữ đúng.
    """
    mask = np.zeros((H, W), np.uint8)
    mask[10:30, 10:50] = 2
    mask[10:30, 50:90] = 2
    return mask


def tao_mask_rong():
    """Chỉ có một đốm nhiễu 2x2 -> 0 box. Ảnh này vẫn phải có mặt trong COCO."""
    mask = np.zeros((H, W), np.uint8)
    mask[200:202, 300:302] = 1
    return mask


def ma_hoa_png(mang):
    ok, buf = cv2.imencode(".png", mang)
    assert ok, "khong ma hoa duoc PNG"
    return buf.tobytes()


def ma_hoa_jpg(mang):
    ok, buf = cv2.imencode(".jpg", mang, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok, "khong ma hoa duoc JPEG"
    return buf.tobytes()


def tao_zip_chinh(duong_dan_zip):
    """Zip giả có cấu trúc + cách đặt tên y hệt FloodNet thật, kèm bẫy ColorMasks.

    CẢ BA SPLIT ĐỀU ĐẶT TÊN FILE LÀ `1.jpg`, `2.jpg`... đúng như thật. Đó chính
    là cái bẫy ở mục 3: thiếu tiền tố split trong tên file ra thì ba split ghi đè
    lên nhau.
    """
    anh_mau = np.full((H, W, 3), 180, np.uint8)   # ảnh xám nhạt, đủ để giải mã
    theo_split = [("train", [tao_mask_nhieu(), tao_mask_dinh(), tao_mask_rong()]),
                  ("val",   [tao_mask_nhieu()]),
                  ("test",  [tao_mask_dinh()])]

    with zipfile.ZipFile(duong_dan_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for split, ds_mask in theo_split:
            for i, mask in enumerate(ds_mask, start=1):
                z.writestr(f"FloodNet-Supervised_v1.0/{split}/{split}-org-img/{i}.jpg",
                           ma_hoa_jpg(anh_mau))
                # Hậu tố "_lab": ảnh là `1.jpg`, mask là `1_lab.png`.
                z.writestr(f"FloodNet-Supervised_v1.0/{split}/{split}-label-img/"
                           f"{i}_lab.png", ma_hoa_png(mask))

        # BẪY: tên có chữ "mask" nhưng bên trong là ẢNH MÀU 3 kênh, không phải nhãn.
        anh_mau_lon = np.full((1024, 1024, 3), 120, np.uint8)
        for ten_split in ("TrainSet", "ValSet", "TestSet"):
            for i in range(1, 3):
                z.writestr(f"ColorMasks-FloodNetv1.0/ColorMasks-{ten_split}/{i}.png",
                           ma_hoa_png(anh_mau_lon))
    return duong_dan_zip


def tao_zip_lech(duong_dan_zip, them_cap_tot=True):
    """Zip có một cặp ẢNH/MASK LỆCH KÍCH THƯỚC.

    Phase 1 chỉ kiểm chuyện lệch kích thước trên 30 ảnh overlay (độ tin cậy thấp,
    ghi ở NOTES muc 2.1). Phase 2 kiểm TỪNG CẶP. Lệch nhau thì box tách từ mask
    nằm sai chỗ trên ảnh — loại nhãn sai tệ nhất vì nó vẫn "hợp lệ" về mọi mặt.

    `them_cap_tot=True`  -> 1 cặp tốt + 1 cặp lệch: cặp lệch bị bỏ, cặp tốt vẫn đi.
    `them_cap_tot=False` -> chỉ có cặp lệch: phải THOÁT với mã 1, không được lặng
                            lẽ ghi ra dataset 0 ảnh rồi báo thành công.
    """
    anh_mau = np.full((H, W, 3), 180, np.uint8)
    mask_ngan = np.zeros((H // 2, W // 2), np.uint8)
    mask_ngan[10:30, 10:50] = 1

    with zipfile.ZipFile(duong_dan_zip, "w", zipfile.ZIP_DEFLATED) as z:
        if them_cap_tot:
            z.writestr("FloodNet-Supervised_v1.0/train/train-org-img/1.jpg",
                       ma_hoa_jpg(anh_mau))
            z.writestr("FloodNet-Supervised_v1.0/train/train-label-img/1_lab.png",
                       ma_hoa_png(tao_mask_dinh()))
        z.writestr("FloodNet-Supervised_v1.0/train/train-org-img/2.jpg",
                   ma_hoa_jpg(anh_mau))
        z.writestr("FloodNet-Supervised_v1.0/train/train-label-img/2_lab.png",
                   ma_hoa_png(mask_ngan))
    return duong_dan_zip


def tao_config_gia(duong_dan_yaml, zip_path, eda_dir, report_dir, work_dir,
                   processed_dir, luu_mask=True, num_overlays=4, classes=None):
    """Config tối thiểu, trỏ vào dữ liệu giả. Dùng y hệt khoá của configs/data.yaml.

    Một file config phục vụ CẢ HAI script: audit.py đọc `paths.output_dir`, còn
    build_coco.py đọc `paths.build_output_dir`. Nhờ vậy phép đối chiếu ở mục 7
    chạy được trên cùng một work_dir mà không phải vá config.
    """
    def _p(s):
        return str(s).replace(chr(92), "/")

    khoi_lop = classes if classes is not None else ("    1: flooded_building\n"
                                                    "    2: non_flooded_building")
    noi_dung = f"""
paths:
  zip: {_p(zip_path)}
  output_dir: {_p(eda_dir)}
  work_dir: {_p(work_dir)}
  processed_dir: {_p(processed_dir)}
  build_output_dir: {_p(report_dir)}
preprocess:
  target_long_side: {TARGET}
  min_area: {MIN_AREA}
  classes:
{khoi_lop}
  water_value: 5
decisions:
  min_side_px: {MIN_SIDE}
  min_box_side_p5_ok: 32
  area_outlier_ratio: 3.0
  cat_tile: false
  tach_nha_dinh: false
eda:
  num_overlays: 4
  seed: 42
  max_images: null
  num_water_check: 60
build_coco:
  jpeg_quality: 95
  luu_mask: {'true' if luu_mask else 'false'}
  eps_polygon: 1.0
  num_overlays: {num_overlays}
  seed: 42
  max_images: null
"""
    Path(duong_dan_yaml).write_text(noi_dung, encoding="utf-8")
    return duong_dan_yaml


# ===========================================================================
# Chạy & đọc kết quả
# ===========================================================================

def _chay(script, config, them_arg=None):
    """Chạy một script trong repo qua subprocess (test luôn cả đường CLI).

    Ép PYTHONIOENCODING=utf-8: console Windows ở đây là cp1252, in tiếng Việt sẽ
    ném UnicodeEncodeError — lỗi của console chứ không phải lỗi của script.
    """
    lenh = [sys.executable, str(GOC_REPO / "scripts" / script), "--config", str(config)]
    if them_arg:
        lenh += them_arg
    moi_truong = dict(os.environ, PYTHONIOENCODING="utf-8")
    kq = subprocess.run(lenh, capture_output=True, cwd=str(GOC_REPO), env=moi_truong)
    return (kq.returncode, kq.stdout.decode("utf-8", "replace"),
            kq.stderr.decode("utf-8", "replace"))


def chay_build(config, them_arg=None):
    return _chay("build_coco.py", config, them_arg)


def chay_audit(config, them_arg=None):
    return _chay("audit.py", config, them_arg)


def doc_coco(processed_dir, split):
    duong_dan = os.path.join(processed_dir, "annotations", f"instances_{split}.json")
    assert os.path.exists(duong_dan), f"khong thay {duong_dan}"
    with open(duong_dan, encoding="utf-8") as f:
        return json.load(f)


def doc_counts(processed_dir, split):
    """Đọc counts_{split}.csv -> list các dòng đã tách ô (kể cả dòng tiêu đề)."""
    duong_dan = os.path.join(processed_dir, f"counts_{split}.csv")
    assert os.path.exists(duong_dan), f"khong thay {duong_dan}"
    with open(duong_dan, encoding="utf-8") as f:
        return [d.strip().split(",") for d in f if d.strip()]


def id_theo_ten(coco):
    """{tên file: image_id} — để tra annotation của một ảnh cụ thể."""
    return {im["file_name"]: im["id"] for im in coco["images"]}


def ann_cua(coco, ten_file):
    return [a for a in coco["annotations"] if a["image_id"] == id_theo_ten(coco)[ten_file]]


# ===========================================================================
# main
# ===========================================================================

def main():
    tmp = tempfile.mkdtemp(prefix="build_coco_test_")
    loi = []

    def kiem(dieu_kien, mo_ta):
        trang_thai = "OK  " if dieu_kien else "FAIL"
        print(f"  [{trang_thai}] {mo_ta}")
        if not dieu_kien:
            loi.append(mo_ta)

    try:
        zip_path = os.path.join(tmp, "floodnet_gia.zip")
        tao_zip_chinh(zip_path)
        P = os.path.join(tmp, "coco")               # processed_dir
        R = os.path.join(tmp, "build")              # report_dir / build_output_dir
        W_ = os.path.join(tmp, "work")              # work_dir, dùng chung với audit
        E = os.path.join(tmp, "eda")                # output_dir của Phase 1
        config = tao_config_gia(os.path.join(tmp, "data.yaml"), zip_path, E, R, W_, P)

        print("=== 1. Dung dataset tu zip gia ===")
        # Chạy Phase 1 TRƯỚC, cùng work_dir: mục 7 cần audit.jsonl nằm cạnh build.jsonl.
        ma_a, stdout_a, stderr_a = chay_audit(config)
        if ma_a != 0:
            print(stdout_a[-3000:])
            print(stderr_a[-3000:])
            raise SystemExit(f"*** audit.py thoat voi ma {ma_a} ***")

        ma_tra_ve, stdout, stderr = chay_build(config)
        if ma_tra_ve != 0:
            print(stdout[-3000:])
            print(stderr[-3000:])
            raise SystemExit(f"*** build_coco.py thoat voi ma {ma_tra_ve} ***")
        kiem(True, "build_coco.py chay xong, ma tra ve 0")

        print("\n=== 2. Cay thu muc sinh ra ===")
        for duong_dan_can in ["annotations/instances_train.json",
                              "annotations/instances_val.json",
                              "annotations/instances_test.json",
                              "counts_train.csv", "counts_val.csv", "counts_test.csv"]:
            kiem(os.path.exists(os.path.join(P, duong_dan_can)),
                 f"co {duong_dan_can}")
        kiem(os.path.isdir(os.path.join(P, "images", "train")), "co images/train/")
        kiem(os.path.isdir(os.path.join(P, "masks", "train")), "co masks/train/")
        kiem(os.path.exists(os.path.join(R, "BUILD_REPORT.md")), "co BUILD_REPORT.md")
        kiem(os.path.isdir(os.path.join(R, "overlay_gt")), "co overlay_gt/")

        print("\n=== 3. Ten file PHAI co tien to split ===")
        # Cả ba split trong zip đều có file tên `1.jpg`. Thiếu tiền tố thì
        # images/val/1.jpg đè images/train/1.jpg và mất ảnh mà không báo gì.
        ds_jpg = []
        for s in ("train", "val", "test"):
            ds_jpg += os.listdir(os.path.join(P, "images", s))
        kiem(len(ds_jpg) == 5, f"tong 5 anh JPEG (train 3 + val 1 + test 1), thuc te {len(ds_jpg)}")
        kiem(not any(t == "1.jpg" for t in ds_jpg),
             "KHONG co file tran '1.jpg' — ba split khong ghi de len nhau")
        kiem(sorted(os.listdir(os.path.join(P, "images", "train")))
             == ["train_1.jpg", "train_2.jpg", "train_3.jpg"],
             "images/train/ dung 3 file, ten co tien to 'train_'")
        kiem(sorted(os.listdir(os.path.join(P, "masks", "train")))
             == ["train_1.png", "train_2.png", "train_3.png"],
             "masks/train/ ten khop voi images/train/ (chi khac duoi .png)")
        # Ảnh và mask phải CÙNG kích thước sau resize, nếu không box nằm sai chỗ.
        for ten in ("train_1.jpg", "val_1.jpg"):
            s = ten.split("_")[0]
            anh = cv2.imread(os.path.join(P, "images", s, ten))
            mask = cv2.imread(os.path.join(P, "masks", s, ten[:-4] + ".png"),
                              cv2.IMREAD_UNCHANGED)
            kiem(anh is not None and anh.shape[:2] == (int(H * TY_LE), TARGET),
                 f"{ten} resize dung {TARGET}x{int(H * TY_LE)}, thuc te "
                 f"{None if anh is None else anh.shape[:2]}")
            kiem(mask is not None and mask.ndim == 2 and mask.shape[:2] == anh.shape[:2],
                 f"mask cua {ten} cung kich thuoc, van la anh xam 1 kenh")

        print("\n=== 4. Noi dung COCO ===")
        c_train = doc_coco(P, "train")
        c_val = doc_coco(P, "val")
        c_test = doc_coco(P, "test")
        kiem(sorted(c["id"] for c in c_train["categories"]) == [1, 2],
             "category id = [1, 2] (lien tuc tu 1 — mmdet bat buoc)")
        kiem([c["name"] for c in c_train["categories"]]
             == ["flooded_building", "non_flooded_building"],
             "ten category dung thu tu theo gia tri mask")
        kiem(len(c_train["images"]) == 3, f"train: 3 anh, thuc te {len(c_train['images'])}")
        kiem(len(c_val["images"]) == 1, f"val: 1 anh, thuc te {len(c_val['images'])}")
        kiem(len(c_test["images"]) == 1, f"test: 1 anh, thuc te {len(c_test['images'])}")
        kiem(sorted(im["file_name"] for im in c_train["images"])
             == ["train_1.jpg", "train_2.jpg", "train_3.jpg"],
             "file_name trong COCO khop ten file da ghi")
        kiem(all(im["width"] == TARGET and im["height"] == int(H * TY_LE)
                 for im in c_train["images"]),
             f"moi anh COCO ghi dung {TARGET}x{int(H * TY_LE)}")
        kiem(all(im.get("ten_goc") for im in c_train["images"]),
             "moi anh co khoa 'ten_goc' de tra nguoc ve FloodNet")

        # --- Ảnh 1: nhà thật + HAI loại nhiễu ---
        a1 = ann_cua(c_train, "train_1.jpg")
        kiem(len(a1) == 1,
             f"train_1.jpg: DUNG 1 box — ca vet 100x1 va duong cheo 16x16 deu bi loc, "
             f"thuc te {len(a1)} box")
        if a1:
            kiem(a1[0]["bbox"] == [10, 10, 30, 20],
                 f"bbox = (10,10,30,20) tuc toa do goc (20,20,60,40) chia doi, "
                 f"thuc te {a1[0]['bbox']}")
            kiem(a1[0]["area"] == 600,
                 f"area = 600 pixel that (30x20), thuc te {a1[0]['area']}")
            kiem(a1[0]["category_id"] == 1, "box cua train_1 la lop 1 (nha ngap)")
            kiem(a1[0]["iscrowd"] == 0, "iscrowd = 0")
            seg = a1[0]["segmentation"]
            kiem(isinstance(seg, list) and len(seg) == 1 and len(seg[0]) >= 6,
                 f"segmentation co >= 3 diem (COCO yeu cau toi thieu 6 so)")
            if seg and seg[0]:
                xs, ys = seg[0][0::2], seg[0][1::2]
                kiem(all(0 <= x <= TARGET for x in xs)
                     and all(0 <= y <= int(H * TY_LE) for y in ys),
                     "moi dinh polygon nam trong anh")

        # --- Ảnh 2: hai nhà kề nhau -> 1 box ---
        a2 = ann_cua(c_train, "train_2.jpg")
        kiem(len(a2) == 1,
             f"train_2.jpg: hai hinh chu nhat KE NHAU -> dung 1 box "
             f"(FloodNet khong co nhan instance), thuc te {len(a2)}")
        if a2:
            kiem(a2[0]["bbox"] == [5, 5, 40, 10],
                 f"bbox = (5,5,40,10) — bao ca hai nua, thuc te {a2[0]['bbox']}")
            kiem(a2[0]["area"] == 400,
                 f"area = 400 = dien tich GOP cua hai nua, thuc te {a2[0]['area']}")
            kiem(a2[0]["category_id"] == 2, "box cua train_2 la lop 2 (khong ngap)")

        # --- Ảnh 3: chỉ có nhiễu ---
        kiem(ann_cua(c_train, "train_3.jpg") == [],
             "train_3.jpg (chi co dom nhieu 2x2): 0 box")
        kiem(len(c_train["annotations"]) == 2,
             f"train: tong 2 annotation, thuc te {len(c_train['annotations'])}")
        kiem(len(c_val["annotations"]) == 1
             and c_val["annotations"][0]["category_id"] == 1,
             "val: 1 annotation lop 1")
        kiem(len(c_test["annotations"]) == 1
             and c_test["annotations"][0]["category_id"] == 2,
             "test: 1 annotation lop 2")

        print("\n=== 5. Dem box sat mep anh ===")
        # Box (10,10,30,20) va (5,5,40,10) deu KHONG cham mep. Kiem rieng vi con so
        # nay duoc in trong bao cao de sau nay phan tich loi (xem docstring dau file).
        kiem("| train | 3 | 1 | 1 | 2 | 1 | 0 |" in open(
            os.path.join(R, "BUILD_REPORT.md"), encoding="utf-8").read(),
            "bao cao: train 3 anh, 1+1 box, 1 anh co nha ngap, 0 box sat mep")

        print("\n=== 6. counts CSV phai khop COCO JSON ===")
        # CSV là GT để Phase 5 chấm điểm đếm. Lệch với JSON nghĩa là điểm số sau
        # này đo sai thứ.
        tieu_de = doc_counts(P, "train")[0]
        kiem(tieu_de == ["image_id", "file_name", "ten_goc", "flooded_building",
                         "non_flooded_building", "tong"],
             f"tieu de CSV dung thu tu cot, thuc te {tieu_de}")
        theo_file = {d[1]: d for d in doc_counts(P, "train")[1:]}
        kiem(len(theo_file) == 3, "CSV train co dung 3 dong du lieu")
        kiem(theo_file.get("train_1.jpg", [])[3:] == ["1", "0", "1"],
             f"train_1: 1 nha ngap, 0 khong ngap, tong 1")
        kiem(theo_file.get("train_2.jpg", [])[3:] == ["0", "1", "1"],
             f"train_2: 0 nha ngap, 1 khong ngap")
        kiem(theo_file.get("train_3.jpg", [])[3:] == ["0", "0", "0"],
             "train_3 khong co nha van CO dong rieng, ghi 0 (khong bi bo sot)")
        for s in ("train", "val", "test"):
            c, d = doc_coco(P, s), doc_counts(P, s)
            tong_csv = sum(int(r[-1]) for r in d[1:])
            kiem(tong_csv == len(c["annotations"]),
                 f"{s}: CSV va JSON khop so box ({tong_csv} = {len(c['annotations'])})")

        print("\n=== 7. Doi chieu Phase 1 — phep kiem quan trong nhat cua GATE 2 ===")
        # Chứng minh Phase 2 dùng ĐÚNG định nghĩa "nhà hợp lệ" mà Phase 1 đã dùng
        # để quyết định tiling. Lệch nghĩa là một trong hai hiểu "nhà" khác nhau,
        # và mọi so sánh số liệu giữa hai phase từ đó về sau đều vô nghĩa.
        bao_cao = open(os.path.join(R, "BUILD_REPORT.md"), encoding="utf-8").read()
        kiem("KHỚP HOÀN TOÀN" in bao_cao,
             "doi chieu Phase 1: KHOP HOAN TOAN (tinh lai tu audit.jsonl bang dung "
             "bo loc cua Phase 2)")
        kiem("BỎ QUA" not in bao_cao,
             "co audit.jsonl trong cung phien -> KHONG duoc bo qua phep doi chieu")
        kiem("KHỚP HOÀN TOÀN" in stdout,
             "ket qua doi chieu cung hien tren man hinh, khong chi trong file")
        kiem("Đối chiếu Phase 1: LỆCH" not in stdout, "khong co dong LECH nao")

        print("\n=== 8. Chay lai (resume) khong duoc nhan doi ===")
        # Colab ngắt giữa chừng là chuyện thường; chạy lại phải bỏ qua ảnh đã làm.
        # Và khoá đọc .jsonl phải là `file_name` — dùng nhầm khoá `anh` của Phase 1
        # thì mọi bản ghi bị bỏ qua và dataset ra 0 ảnh MÀ KHÔNG BÁO GÌ.
        ma_lai, stdout_lai, stderr_lai = chay_build(config)
        kiem(ma_lai == 0, f"chay lai lan 2 van ma 0, thuc te {ma_lai}")
        kiem("Đọc lại 5 ảnh đã xử lý lần trước" in stdout_lai,
             "lan 2 bao dung 'Doc lai 5 anh da xu ly lan truoc' (resume song)")
        with open(os.path.join(W_, "build.jsonl"), encoding="utf-8") as f:
            so_dong = sum(1 for d in f if d.strip())
        kiem(so_dong == 5, f"build.jsonl van dung 5 dong, khong nhan doi (thuc te {so_dong})")
        kiem(len(doc_coco(P, "train")["annotations"]) == 2,
             "chay lai khong lam tang so annotation")

        # Khoá sai phải NÉM LỖI, không được lặng lẽ trả về rỗng.
        from floodcount.data.audit import doc_ket_qua_da_co
        try:
            doc_ket_qua_da_co(os.path.join(W_, "build.jsonl"))   # khoá mặc định "anh"
            kiem(False, "doc build.jsonl bang khoa mac dinh 'anh' PHAI nem KeyError")
        except KeyError as e:
            kiem("file_name" in str(e),
                 "thong bao loi chi ro khoa dung can dung ('file_name')")
        kiem(len(doc_ket_qua_da_co(os.path.join(W_, "build.jsonl"),
                                   khoa="file_name")) == 5,
             "doc lai bang DUNG khoa 'file_name' thi du 5 ban ghi")

        print("\n=== 9. Chay thu --max-images va luu_mask=false ===")
        P2 = os.path.join(tmp, "coco_thu")
        R2 = os.path.join(tmp, "build_thu")
        W2 = os.path.join(tmp, "work_thu")
        config2 = tao_config_gia(os.path.join(tmp, "thu.yaml"), zip_path,
                                 os.path.join(tmp, "eda_thu"), R2, W2, P2,
                                 luu_mask=False, num_overlays=2)
        ma_thu, stdout_thu, stderr_thu = chay_build(config2, ["--max-images", "1", "--fresh"])
        if ma_thu != 0:
            print(stdout_thu[-3000:])
            print(stderr_thu[-3000:])
            raise SystemExit(f"*** build_coco.py --max-images thoat voi ma {ma_thu} ***")
        kiem(len(doc_coco(P2, "train")["images"]) == 1, "--max-images 1 -> train con 1 anh")
        kiem(len(doc_coco(P2, "val")["images"]) == 1, "--max-images 1 -> val con 1 anh")
        kiem(not os.path.isdir(os.path.join(P2, "masks")),
             "luu_mask=false -> KHONG tao thu muc masks/")
        # --max-images là chạy thử nên KHÔNG được đối chiếu Phase 1: audit.jsonl
        # đầy đủ mà build.jsonl mới có 3 ảnh thì con số lệch là đương nhiên, báo
        # "LỆCH" ở đây sẽ khiến người dùng tưởng có bug thật.
        bct = open(os.path.join(R2, "BUILD_REPORT.md"), encoding="utf-8").read()
        kiem("BỎ QUA" in bct,
             "chay thu (--max-images) -> BO QUA doi chieu Phase 1, khong bao LECH gia")

        print("\n=== 10. tao_categories tu choi id khong lien tuc ===")
        # mmdet dùng category id làm chỉ số lớp trong head, nên id phải liên tục
        # từ 1. Config đổi thành bộ giá trị khác thì phải chết NGAY ở đây với
        # thông báo đọc được, chứ không phải chết ở tận Phase 3.
        from floodcount.data.mask_to_coco import tao_categories
        kiem([c["id"] for c in tao_categories({1: "a", 2: "b"})] == [1, 2],
             "id lien tuc 1..2 -> dung")
        kiem([c["id"] for c in tao_categories({1: "a", 2: "b", 3: "c"})] == [1, 2, 3],
             "id lien tuc 1..3 -> dung")
        for bo_loi in ({1: "a", 3: "c"}, {0: "a", 1: "b"}, {2: "a", 3: "b"}):
            try:
                tao_categories(bo_loi)
                kiem(False, f"bo gia tri {sorted(bo_loi)} PHAI bi tu choi")
            except ValueError as e:
                kiem("liên tục" in str(e) or "lien tuc" in str(e),
                     f"bo gia tri {sorted(bo_loi)} bi tu choi, loi chi ro vi sao")

        print("\n=== 11. Anh/mask lech kich thuoc ===")
        zip_lech = tao_zip_lech(os.path.join(tmp, "lech.zip"), them_cap_tot=True)
        P3, R3, W3 = (os.path.join(tmp, "coco_lech"), os.path.join(tmp, "build_lech"),
                      os.path.join(tmp, "work_lech"))
        config3 = tao_config_gia(os.path.join(tmp, "lech.yaml"), zip_lech,
                                 os.path.join(tmp, "eda_lech"), R3, W3, P3)
        ma_lech, stdout_lech, stderr_lech = chay_build(config3)
        kiem(ma_lech == 0, f"con cap tot nen van chay xong, ma 0 (thuc te {ma_lech})")
        kiem(len(doc_coco(P3, "train")["images"]) == 1,
             "cap lech bi bo, cap tot van vao dataset")
        bc3 = open(os.path.join(R3, "BUILD_REPORT.md"), encoding="utf-8").read()
        kiem("⚠️ Có cặp ảnh/mask bị bỏ" in bc3,
             "bao cao co khoi canh bao cap bi bo")
        kiem("lệch mask" in bc3, "canh bao noi ro LY DO: lech kich thuoc mask")

        # Chỉ có cặp lệch -> phải THOÁT với mã 1. Ghi ra dataset 0 ảnh rồi báo
        # "XONG" là kiểu thất bại tệ nhất: nhìn mã thoát thì tưởng đã xong.
        zip_toan_loi = tao_zip_lech(os.path.join(tmp, "lech_all.zip"), them_cap_tot=False)
        config4 = tao_config_gia(os.path.join(tmp, "lech_all.yaml"), zip_toan_loi,
                                 os.path.join(tmp, "eda_lech2"),
                                 os.path.join(tmp, "build_lech2"),
                                 os.path.join(tmp, "work_lech2"),
                                 os.path.join(tmp, "coco_lech2"))
        ma_all, stdout_all, stderr_all = chay_build(config4)
        kiem(ma_all == 1, f"toan bo cap loi -> thoat ma 1 (thuc te {ma_all})")
        kiem("KHÔNG dựng được ảnh nào" in stdout_all,
             "thoat voi thong bao ro 'KHONG dung duoc anh nao', khong im lang")

        print("\n=== 12. configs/data.yaml THAT phai co du moi khoa Phase 2 doc ===")
        # File config thật chỉ chạy trên Colab, mà vòng sửa-lỗi ở đó rất đắt. Thiếu
        # một khoá thì lỗi chỉ lộ ra ở đó dưới dạng KeyError — kiểm ngay cho rẻ.
        import yaml
        cfg_that = yaml.safe_load((GOC_REPO / "configs" / "data.yaml")
                                  .read_text(encoding="utf-8"))
        for duong_dan_khoa in [
                "paths.zip", "paths.work_dir", "paths.processed_dir",
                "paths.build_output_dir",
                "preprocess.target_long_side", "preprocess.min_area",
                "preprocess.classes", "decisions.min_side_px",
                "build_coco.jpeg_quality", "build_coco.luu_mask",
                "build_coco.eps_polygon", "build_coco.num_overlays",
                "build_coco.seed", "build_coco.max_images"]:
            kiem(co_khoa(cfg_that, duong_dan_khoa),
                 f"configs/data.yaml co khoa {duong_dan_khoa}")
        kiem(cfg_that["preprocess"]["min_area"] == cfg_that["decisions"]["min_side_px"] ** 2,
             "min_area == min_side_px^2 (hai nguong loc khong mau thuan)")
        # processed_dir phải nằm trên /content, KHÔNG phải Drive: ghi 2343 ảnh qua
        # FUSE chậm hơn nhiều lần, mà dữ liệu này dựng lại được từ zip.
        kiem(cfg_that["paths"]["processed_dir"].startswith("/content/"),
             "processed_dir nam tren /content (khong ghi hang nghin file len Drive)")

        print("\n=== 13. Notebook 02 phai khop voi CLI cua build_coco.py ===")
        # Loại lỗi chỉ lộ ra khi đã ngồi chờ trên Colab: notebook truyền cờ mà
        # script không có, hoặc ngược lại. Kiểm ngay trên máy cho rẻ.
        nb = json.loads((GOC_REPO / "notebooks" / "02_build_coco.ipynb")
                        .read_text(encoding="utf-8"))
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                compile("".join(c["source"]), f"nb-cell-{i}", "exec")
        kiem(True, "moi o code cua notebook 02 compile duoc")

        # Trên Colab, fd 1 của tiến trình con KHÔNG chảy vào ô output. Dính ba lần
        # ngày 30/09/2026 mới nhận ra quy luật, nên phải khoá lại bằng test.
        ma_nguon_nb = "\n".join("".join(c["source"]) for c in nb["cells"])
        kiem("chay_hien_dan" in ma_nguon_nb,
             "notebook 02 dung ham chay_hien_dan (doc duoc output tren Colab)")
        kiem("stdout=subprocess.PIPE" in ma_nguon_nb,
             "chay_hien_dan tu doc ong dan thay vi de Colab tu lo")
        kiem('"-u"' in ma_nguon_nb,
             "co truyen -u cho python con (khong co thi output bi dem theo khoi)")
        kiem(ma_nguon_nb.count("chay_hien_dan(") >= 3,
             "chay_hien_dan duoc dinh nghia VA goi o cac o chay that")

        # Bỏ các dòng gọi git: `--oneline` là cờ của git chứ không phải của build_coco.py
        dong_python = [d for d in ma_nguon_nb.splitlines() if "git" not in d]
        co_dung = sorted(set(re.findall(r'"(--[a-z][a-z-]+)"', "\n".join(dong_python))))
        kiem(len(co_dung) >= 3, f"notebook co dung it nhat 3 co CLI, thuc te {co_dung}")
        kq_help = subprocess.run(
            [sys.executable, str(GOC_REPO / "scripts" / "build_coco.py"), "--help"],
            capture_output=True, cwd=str(GOC_REPO),
            env=dict(os.environ, PYTHONIOENCODING="cp1252"))
        kiem(kq_help.returncode == 0,
             "build_coco.py --help khong crash du console la cp1252 (Windows)")
        tro_giup = kq_help.stdout.decode("utf-8", errors="replace")
        kiem("COCO" in tro_giup, "chu co dau ra dung utf-8, khong bi thay bang '?'")
        for co in co_dung:
            kiem(co in tro_giup, f"notebook dung co {co}, va co nay CO THAT trong CLI")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if loi:
        print(f"*** {len(loi)} MUC KHONG DAT ***")
        for m in loi:
            print("   - " + m.encode("ascii", "backslashreplace").decode())
        raise SystemExit(1)
    print("*** TAT CA PASS ***")


def co_khoa(cfg, duong_dan_khoa):
    """True nếu config CÓ khoá này — kể cả khi giá trị là `null`.

    Không dùng `.get()` rồi so với None: `build_coco.max_images: null` là giá trị
    hợp lệ (nghĩa là "chạy hết"), mà `.get()` trả None cho cả trường hợp thiếu
    khoá lẫn trường hợp khoá có giá trị null — hai chuyện hoàn toàn khác nhau.
    """
    nut = cfg
    for phan in duong_dan_khoa.split("."):
        if not isinstance(nut, dict) or phan not in nut:
            return False
        nut = nut[phan]
    return True


if __name__ == "__main__":
    main()

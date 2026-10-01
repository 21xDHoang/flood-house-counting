# -*- coding: utf-8 -*-
"""Test cho src/floodcount/data/overfit.py — CPU, không cần mmdet, không cần dataset thật.

Hai điều test này phải khoá:

  1. CHỌN LẠI ĐƯỢC. GATE 3 yêu cầu "cùng 20 ảnh đó" cho mọi lần chạy. Chọn theo
     thứ tự file trên đĩa thì Windows và Linux cho hai tập khác nhau — mà Colab
     là Linux còn máy này là Windows, nên đây là lỗi sẽ xảy ra thật.

  2. `categories` PHẢI ĐƯỢC CHÉP NGUYÊN VĂN, kể cả thứ tự. Nhãn mà mmdet gán phụ
     thuộc thứ tự category trong file JSON. Đổi thứ tự ở đây là đảo nhãn
     ngập/không ngập của cả tập overfit — và phép thử "học vẹt" VẪN ĐẠT, vì nó
     chỉ cần nhớ chứ không cần đúng lớp. Tức là lỗi này đi lọt qua đúng cái chốt
     kiểm sinh ra để bắt lỗi.

Chạy:
    py tests/test_overfit.py
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

GOC_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC_REPO / "src"))

from floodcount.data.overfit import (bao_cao, cat_coco, chon_anh,  # noqa: E402
                                     phan_nhom, tao_overfit20)

NGAP, KHONG = 1, 2          # id category, dung nhu trong dataset that


def tao_coco_gia(so_co_ngap=20, so_chi_khong=12, so_rong=8,
                 thu_tu_category=(NGAP, KHONG)):
    """COCO giả có ba nhóm ảnh rõ ràng. Box được đặt lệch nhau để kiểm toạ độ."""
    ds_anh, ds_ann = [], []
    ten = {"flooded_building": "flooded_building", "non_flooded_building":
           "non_flooded_building"}
    im_id = 0
    for nhom, so_anh in (("co_ngap", so_co_ngap), ("chi_khong", so_chi_khong),
                         ("rong", so_rong)):
        for k in range(so_anh):
            im_id += 1
            ds_anh.append({"id": im_id, "file_name": f"train_{im_id}.jpg",
                           "width": 1536, "height": 1152})
            if nhom == "co_ngap":
                # Hai box: toạ độ phụ thuộc im_id nên so sai toạ độ là lộ ngay
                ds_ann.append({"id": len(ds_ann) + 1, "image_id": im_id,
                               "category_id": NGAP, "bbox": [10 + im_id, 20, 100, 80],
                               "area": 8000, "segmentation": [], "iscrowd": 0})
                ds_ann.append({"id": len(ds_ann) + 1, "image_id": im_id,
                               "category_id": KHONG, "bbox": [300, 40 + im_id, 60, 60],
                               "area": 3600, "segmentation": [], "iscrowd": 0})
            elif nhom == "chi_khong":
                ds_ann.append({"id": len(ds_ann) + 1, "image_id": im_id,
                               "category_id": KHONG, "bbox": [50, 50, 40, 40],
                               "area": 1600, "segmentation": [], "iscrowd": 0})
    id_theo_ten = {NGAP: ten["flooded_building"], KHONG: ten["non_flooded_building"]}
    return {
        "info": {"description": "gia"},
        "licenses": [],
        "images": ds_anh,
        "annotations": ds_ann,
        "categories": [{"id": c, "name": id_theo_ten[c], "supercategory": "building"}
                       for c in thu_tu_category],
    }


def main():
    tmp = tempfile.mkdtemp(prefix="overfit_test_")
    loi = []

    def kiem(dieu_kien, mo_ta):
        trang_thai = "OK  " if dieu_kien else "FAIL"
        print(f"  [{trang_thai}] {mo_ta}")
        if not dieu_kien:
            loi.append(mo_ta)

    try:
        coco = tao_coco_gia()

        print("=== 1. Chia nhom anh theo noi dung ===")
        nhom = phan_nhom(coco)
        kiem(len(nhom["co_ngap"]) == 20, f"20 anh co nha ngap, thuc te {len(nhom['co_ngap'])}")
        kiem(len(nhom["chi_khong"]) == 12,
             f"12 anh chi co nha khong ngap, thuc te {len(nhom['chi_khong'])}")
        kiem(len(nhom["rong"]) == 8,
             f"8 anh khong co box nao, thuc te {len(nhom['rong'])}")
        kiem(sum(len(v) for v in nhom.values()) == len(coco["images"]),
             "ba nhom phu kin so anh, khong sot anh nao")
        # Sap xep theo file_name thi Windows va Linux moi ra cung thu tu
        kiem(nhom["co_ngap"] == sorted(nhom["co_ngap"], key=lambda i: i["file_name"]),
             "moi nhom da sap xep theo file_name (Windows va Linux cung ket qua)")

        print("\n=== 2. Chon 12/6/2 va TAI LAP DUOC ===")
        chon1, cb1 = chon_anh(nhom, 12, 6, 2, seed=42)
        chon2, _ = chon_anh(nhom, 12, 6, 2, seed=42)
        kiem(len(chon1) == 20, f"chon du 20 anh, thuc te {len(chon1)}")
        kiem(not cb1, f"du anh o ca ba nhom nen khong co canh bao, thuc te {cb1}")
        kiem([i["file_name"] for i in chon1] == [i["file_name"] for i in chon2],
             "cung seed -> DUNG 20 anh do (GATE 3 yeu cau chay lai phai giong)")

        chon3, _ = chon_anh(nhom, 12, 6, 2, seed=7)
        kiem([i["file_name"] for i in chon1] != [i["file_name"] for i in chon3],
             "seed khac -> tap khac (seed co tac dung, khong bi bo qua)")
        kiem([i["file_name"] for i in chon1]
             == sorted(i["file_name"] for i in chon1),
             "ket qua tra ve da sap xep theo file_name")

        # Dung luong ba nhom phai dung ti le: neu chon nham sang nhom khac thi
        # phep thu khong con cham tới nhánh "ảnh rỗng" — nhánh dễ hỏng nhất.
        dem_nhom = Counter()
        for im in chon1:
            for ten_nhom, ds in nhom.items():
                if any(x["id"] == im["id"] for x in ds):
                    dem_nhom[ten_nhom] += 1
        kiem(dem_nhom == {"co_ngap": 12, "chi_khong": 6, "rong": 2},
             f"dung ti le 12/6/2, thuc te {dict(dem_nhom)}")

        print("\n=== 3. cat_coco: id danh lai, du lieu giu nguyen ===")
        coco_moi, tt = tao_overfit20(coco, 12, 6, 2, seed=42)
        kiem(len(coco_moi["images"]) == 20, "20 anh")
        kiem([im["id"] for im in coco_moi["images"]] == list(range(1, 21)),
             "image id danh lai 1..20 lien tuc")
        kiem([a["id"] for a in coco_moi["annotations"]]
             == list(range(1, len(coco_moi["annotations"]) + 1)),
             "annotation id danh lai 1..M lien tuc")
        kiem({a["image_id"] for a in coco_moi["annotations"]}
             <= set(range(1, 21)),
             "moi annotation tro tới mot trong 20 anh, khong tro ra ngoai")

        # 12 anh "co nha ngap" x (1 box ngap + 1 box khong ngap) = 12 + 12
        #  6 anh "chi nha khong ngap" x 1 box khong ngap        =      6
        #                                                  tong  = 30 box
        # Anh co nha ngap VAN mang theo box nha khong ngap — dung nhu FloodNet
        # that (mot khu vuc chup co ca nha ngap lan nha khong ngap). Bo sot dieu
        # nay thi tuong dem sai.
        kiem(tt["so_box"] == 30, f"30 box (12x2 + 6x1), thuc te {tt['so_box']}")
        kiem(tt["dem_theo_lop"] == {"flooded_building": 12,
                                    "non_flooded_building": 18},
             f"dem theo lop: 12 ngap / 18 khong ngap, thuc te {tt['dem_theo_lop']}")
        kiem(sum(tt["dem_theo_lop"].values()) == tt["so_box"],
             "tong hai lop bang so box, khong sot box nao")

        print("\n=== 4. categories CHÉP NGUYÊN VĂN, kể cả thứ tự ===")
        kiem(coco_moi["categories"] == coco["categories"],
             "categories giong het ban goc (ca thu tu lan noi dung)")
        # Thu tu dao: day la truong hop that su nguy hiem, va loi chi lo ra khi
        # mmdet in bang ket qua chu khong lo ra o bat ky cho nao khac.
        coco_dao = tao_coco_gia(3, 2, 1, thu_tu_category=(KHONG, NGAP))
        coco_moi_dao, _ = tao_overfit20(coco_dao, 2, 1, 1, seed=42)
        kiem([c["id"] for c in coco_moi_dao["categories"]] == [KHONG, NGAP],
             f"thu tu category dao (2,1) duoc giu nguyen, thuc te "
             f"{[c['id'] for c in coco_moi_dao['categories']]}")

        print("\n=== 5. Du lieu cua anh duoc chon khong bi doi ===")
        goc = {a["id"]: a for a in coco["annotations"]}
        ten_theo_id_moi = {im["id"]: im["file_name"] for im in coco_moi["images"]}
        ten_theo_id_cu = {im["id"]: im["file_name"] for im in coco["images"]}
        sai = 0
        for a in coco_moi["annotations"]:
            # Tim annotation goc tuong ung: cung file_name + cung bbox + cung lop
            ten = ten_theo_id_moi[a["image_id"]]
            im_cu = next(im for im in coco["images"] if im["file_name"] == ten)
            co = [b for b in coco["annotations"]
                  if b["image_id"] == im_cu["id"]
                  and b["bbox"] == a["bbox"] and b["category_id"] == a["category_id"]]
            if not co:
                sai += 1
            elif co[0]["area"] != a["area"]:
                sai += 1
        kiem(sai == 0, f"moi box giu nguyen toa do/dien tich/lop, so box sai = {sai}")
        kiem(set(ten_theo_id_moi.values()) <= set(ten_theo_id_cu.values()),
             "chi dung lai file_name co trong tap nguon (khong bia ten moi)")

        print("\n=== 6. Thieu anh o mot nhom -> CANH BAO, khong im lang ===")
        coco_it = tao_coco_gia(so_co_ngap=20, so_chi_khong=2, so_rong=0)
        _, tt_it = tao_overfit20(coco_it, 12, 6, 2, seed=42)
        kiem(len(tt_it["canh_bao"]) == 2,
             f"2 nhom bi thieu -> 2 canh bao, thuc te {len(tt_it['canh_bao'])}")
        # Lay `min(can, co)` theo TUNG NHOM: nhom co_ngap co 20 anh nhung chi CAN
        # 12 nen van lay 12 (lay het 20 se pha ti le 12/6/2 — ma ti le do moi la
        # thu khien phep thu cham duoc vao ca ba nhanh). Hai nhom thieu thi lay
        # het nhung gi co: 2 + 0. Tong 12 + 2 + 0 = 14.
        kiem(tt_it["so_anh"] == 14,
             f"nhom thieu lay het nhung gi co, nhom du van theo ti le -> 14 anh, "
             f"thuc te {tt_it['so_anh']}")
        kiem(any("rong" in c for c in tt_it["canh_bao"]),
             "canh bao goi dung ten nhom thieu")
        van_ban = bao_cao(tt_it)
        kiem("[!]" in van_ban, "bao cao in canh bao ra chu khong chi de trong dict")
        kiem("20 ảnh đã chọn" in van_ban, "bao cao liet ke 20 anh da chon")

        print("\n=== 7. CLI: ghi file roi DOC LAI tu dia ===")
        processed = os.path.join(tmp, "floodnet_coco")
        os.makedirs(os.path.join(processed, "annotations"), exist_ok=True)
        with open(os.path.join(processed, "annotations", "instances_train.json"),
                  "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        cfg = os.path.join(tmp, "data.yaml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write(f"paths:\n  processed_dir: {processed.replace(chr(92), '/')}\n"
                    f"eda:\n  seed: 42\n")

        def chay_cli():
            lenh = [sys.executable, str(GOC_REPO / "scripts" / "tao_overfit20.py"),
                    "--config", str(cfg)]
            return subprocess.run(lenh, capture_output=True, cwd=str(GOC_REPO),
                                  env=dict(os.environ, PYTHONIOENCODING="utf-8"))

        kq = chay_cli()
        stdout = kq.stdout.decode("utf-8", "replace")
        if kq.returncode != 0:
            print(stdout[-2000:])
            print(kq.stderr.decode("utf-8", "replace")[-2000:])
        kiem(kq.returncode == 0, f"CLI chay thanh cong, ma thoat {kq.returncode}")
        kiem("Tập overfit: 20 ảnh / 30 box" in stdout,
             "in dung 20 anh / 30 box")

        duong_ra = os.path.join(processed, "annotations", "instances_overfit20.json")
        kiem(os.path.exists(duong_ra), "da ghi instances_overfit20.json")
        with open(duong_ra, encoding="utf-8") as f:
            lan1 = f.read()
        kiem("KHỚP" in stdout, "doc lai tu dia va bao KHOP")

        # Chay lan 2: file phai Y NGUYEN tung byte. Day moi la dieu GATE 3 can —
        # neu khong thi "chay lai 20 anh do" chi la y dinh chu khong phai su that.
        chay_cli()
        with open(duong_ra, encoding="utf-8") as f:
            lan2 = f.read()
        kiem(lan1 == lan2, "chay lan 2 -> file y nguyen tung byte (tai lap duoc)")

        with open(duong_ra, encoding="utf-8") as f:
            lai = json.load(f)
        kiem(len(lai["images"]) == 20 and len(lai["annotations"]) == 30,
             f"doc lai: 20 anh / 30 box, thuc te "
             f"{len(lai['images'])}/{len(lai['annotations'])}")
        kiem(lai["categories"] == coco["categories"],
             "categories trong file tren dia van dung thu tu goc")

        print("\n=== 8. Thieu annotation nguon -> bao ro, khong traceback ===")
        cfg_xau = os.path.join(tmp, "khong-co.yaml")
        with open(cfg_xau, "w", encoding="utf-8") as f:
            f.write("paths:\n  processed_dir: /khong/co/that\neda:\n  seed: 1\n")
        kq2 = subprocess.run(
            [sys.executable, str(GOC_REPO / "scripts" / "tao_overfit20.py"),
             "--config", str(cfg_xau)],
            capture_output=True, cwd=str(GOC_REPO),
            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        out2 = kq2.stdout.decode("utf-8", "replace")
        kiem(kq2.returncode != 0, f"thieu annotation nguon -> ma khac 0, thuc te {kq2.returncode}")
        kiem("Không thấy annotation nguồn" in out2, "bao ro thieu file nao")
        kiem("Traceback" not in out2, "khong do traceback ra stdout")

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

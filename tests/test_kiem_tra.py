# -*- coding: utf-8 -*-
"""Test cho src/floodcount/data/kiem_tra.py — CPU, không cần mmdet, không cần dataset thật.

Điều quan trọng nhất test này phải khoá: BỐN KIỂU LỖI IM LẶNG mà bước kiểm tra
sinh ra để bắt. Cả bốn đều thuộc loại "train vẫn chạy, số vẫn ra, chỉ là sai":

  1. `metainfo.classes` khai một tên mà file JSON không có  -> mmdet bỏ hết box
     của lớp đó, không một dòng cảnh báo.
  2. Thứ tự category trong JSON khác thứ tự `metainfo.classes` -> nhãn ngập và
     không ngập bị in NGƯỢC TÊN ở bảng AP và bảng đếm cuối.
  3. Box nằm ngoài ảnh / area <= 0 / cạnh < 1 -> `_parse_ann_info` bỏ im lặng.
  4. Ảnh có trong JSON nhưng không có trên đĩa.

Với mỗi lỗi, test đòi script phải thoát với MÃ KHÁC 0 và in ra câu giải thích
đúng chỗ — chứ không chỉ đòi "có phát hiện". Một bước kiểm báo lỗi mà không nói
lỗi gì thì khi chạy trên Colab vẫn phải đi hỏi lại, mất đúng cái lợi ích nó mang.

Chạy:
    py tests/test_kiem_tra.py
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

GOC_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC_REPO / "src"))

from floodcount.data.kiem_tra import (doc_classes_tu_config_mmdet,  # noqa: E402
                                      ly_do_mmdet_bo, thong_ke,
                                      thu_tu_nhan_mmdet)

# Kích thước ảnh giả. Nhỏ để test chạy nhanh; chỉ cần đủ lớn hơn các box dựng
# trong bài, vì box nằm ngoài ảnh chính là một trong những thứ phải bắt được.
W, H = 200, 150

CONFIG_MMDET_DUNG = "metainfo = dict(classes=('flooded_building', 'non_flooded_building'))\n"


# ===========================================================================
# Dựng dữ liệu giả
# ===========================================================================

def anh_rong(im_id, ten_file):
    return {"id": im_id, "file_name": ten_file, "width": W, "height": H}


def tao_coco(ds_anh, ds_ann, thu_tu_category=("flooded_building", "non_flooded_building")):
    """Ráp một COCO dict. `thu_tu_category` là thứ tự category GHI VÀO FILE.

    Thứ tự này là thứ quyết định nhãn 0/1 mà mmdet gán, nên test phải điều khiển
    được nó — đó là cách duy nhất dựng lại được lỗi "in ngược tên".
    """
    cat_id = {"flooded_building": 1, "non_flooded_building": 2}
    return {
        "info": {"description": "gia"},
        "licenses": [],
        "images": ds_anh,
        "annotations": ds_ann,
        "categories": [{"id": cat_id[t], "name": t, "supercategory": "building"}
                       for t in thu_tu_category],
    }


def ann(ann_id, im_id, cat, bbox, area=None):
    return {"id": ann_id, "image_id": im_id, "category_id": cat,
            "bbox": list(bbox), "area": area if area is not None else bbox[2] * bbox[3],
            "segmentation": [], "iscrowd": 0}


def ghi_dataset(processed, theo_split):
    """Ghi dataset giả: ảnh rỗng (chỉ cần TỒN TẠI) + file JSON cho từng split."""
    os.makedirs(os.path.join(processed, "annotations"), exist_ok=True)
    for split, coco in theo_split.items():
        thu_muc = os.path.join(processed, "images", split)
        os.makedirs(thu_muc, exist_ok=True)
        for im in coco["images"]:
            # Nội dung file không quan trọng: bước kiểm này chỉ hỏi file có tồn
            # tại hay không (việc đọc ảnh là của mmdet ở mức 2).
            with open(os.path.join(thu_muc, im["file_name"]), "wb") as f:
                f.write(b"\xff\xd8\xff\xe0 gia")
        with open(os.path.join(processed, "annotations", f"instances_{split}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)


def dataset_mau():
    """Dataset ĐẠT: 3 ảnh train (1 ngập, 1 chỉ không ngập, 1 rỗng) + 1 val + 1 test."""
    train = tao_coco(
        [anh_rong(1, "train_1.jpg"), anh_rong(2, "train_2.jpg"), anh_rong(3, "train_3.jpg")],
        [ann(1, 1, 1, (10, 10, 50, 50)),      # nhà ngập
         ann(2, 1, 2, (100, 100, 40, 40)),    # nhà không ngập
         ann(3, 2, 2, (20, 20, 30, 30))])     # ảnh 3: KHÔNG có annotation nào
    val = tao_coco([anh_rong(1, "val_1.jpg")], [ann(1, 1, 1, (5, 5, 60, 60))])
    test = tao_coco([anh_rong(1, "test_1.jpg")], [ann(1, 1, 2, (5, 5, 60, 60))])
    return {"train": train, "val": val, "test": test}


def config_yaml(processed):
    """Config giả, dùng ĐÚNG các khoá mà scripts/kiem_tra_du_lieu.py đọc.

    `preprocess.classes` là bắt buộc: đó là nguồn tên lớp mà Phase 1/Phase 2 đã
    dùng, thiếu nó thì script dừng ngay (và test số 11 kiểm đúng chuyện đó).
    """
    return (f"paths:\n  processed_dir: {processed.replace(chr(92), '/')}\n"
            f"preprocess:\n  classes:\n    1: flooded_building\n"
            f"    2: non_flooded_building\n"
            f"eda:\n  seed: 42\n")


def chay_cli(config, config_mmdet):
    """Chạy scripts/kiem_tra_du_lieu.py ở mức 1 (--mmdet khong)."""
    lenh = [sys.executable, str(GOC_REPO / "scripts" / "kiem_tra_du_lieu.py"),
            "--config", str(config), "--mmdet-config", str(config_mmdet),
            "--mmdet", "khong"]
    kq = subprocess.run(lenh, capture_output=True, cwd=str(GOC_REPO),
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return (kq.returncode,
            kq.stdout.decode("utf-8", "replace"),
            kq.stderr.decode("utf-8", "replace"))


# ===========================================================================
# Test
# ===========================================================================

def main():
    tmp = tempfile.mkdtemp(prefix="kiem_tra_test_")
    loi = []

    def kiem(dieu_kien, mo_ta):
        trang_thai = "OK  " if dieu_kien else "FAIL"
        print(f"  [{trang_thai}] {mo_ta}")
        if not dieu_kien:
            loi.append(mo_ta)

    def dung_lai(ten, theo_split):
        """Ghi mot dataset gia vao thu muc rieng, tra ve (config, config_mmdet)."""
        goc = os.path.join(tmp, ten)
        processed = os.path.join(goc, "floodnet_coco")
        ghi_dataset(processed, theo_split)
        cfg = os.path.join(goc, "data.yaml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write(config_yaml(processed))
        return cfg

    def ghi_config_mmdet(ten, noi_dung):
        duong = os.path.join(tmp, ten)
        with open(duong, "w", encoding="utf-8") as f:
            f.write(noi_dung)
        return duong

    try:
        cfg_mmdet_dung = ghi_config_mmdet("mmdet_dung.py", CONFIG_MMDET_DUNG)

        print("=== 1. Dataset dat -> thoat ma 0 ===")
        cfg = dung_lai("tot", dataset_mau())
        ma, stdout, stderr = chay_cli(cfg, cfg_mmdet_dung)
        if ma != 0:
            print(stdout[-2000:])
            print(stderr[-2000:])
        kiem(ma == 0, f"dataset sach -> ma thoat 0, thuc te {ma}")
        kiem("ĐẠT HẾT" in stdout, "co dong ket luan 'ĐẠT HẾT'")
        kiem("[train] 3 ảnh | 3 box" in stdout,
             "dem dung 3 anh / 3 box o train")
        kiem("Tổng 5 ảnh / 5 box" in stdout,
             f"tong ket 5 anh / 5 box (train 3 + val 1 + test 1)")
        kiem("flooded_building" in stdout and "non_flooded_building" in stdout,
             "in so box tach theo tung lop")

        print("\n=== 2. Ten lop khai trong metainfo ma JSON khong co ===")
        # Day la loi nguy hiem nhat: khong phai crash, ma la MAT DU LIEU.
        cfg_mmdet_thieu = ghi_config_mmdet(
            "mmdet_thieu.py",
            "metainfo = dict(classes=('flooded_building', 'nha_khong_ton_tai'))\n")
        ma, stdout, _ = chay_cli(dung_lai("thieu_ten", dataset_mau()), cfg_mmdet_thieu)
        kiem(ma != 0, f"phat hien ten lop khong co trong JSON -> ma khac 0, thuc te {ma}")
        kiem("BỎ IM LẶNG" in stdout,
             "noi ro hau qua: mmdet BO IM LANG cac box cua lop do")
        kiem("nha_khong_ton_tai" in stdout, "goi ten lop khai sai")
        kiem("2,000 box" not in stdout and "2,000" not in stdout,
             "dem dung so box se mat (khong phong dai)")

        print("\n=== 3. Thu tu lop lech -> in nguoc ten ===")
        # JSON ghi category theo thu tu (non_flooded, flooded) nhung metainfo
        # khai (flooded, non_flooded). mmdet gan nhan 0 cho category DUNG TRUOC
        # TRONG JSON, tuc la nhan 0 = non_flooded — trong khi bang ket qua se in
        # ten metainfo[0] = 'flooded_building'. Nguoc hoan toan.
        train_nguoc = tao_coco(
            [anh_rong(1, "train_1.jpg")],
            [ann(1, 1, 1, (10, 10, 50, 50)), ann(2, 1, 2, (100, 100, 40, 40))],
            thu_tu_category=("non_flooded_building", "flooded_building"))
        ma, stdout, _ = chay_cli(dung_lai("thu_tu", {"train": train_nguoc}),
                                 cfg_mmdet_dung)
        kiem(ma != 0, f"thu tu lop lech -> ma khac 0, thuc te {ma}")
        kiem("LỆCH" in stdout and "THỨ TỰ" in stdout,
             "bao dung loai loi: THU TU / TEN LOP LECH")
        kiem("nhãn 0" in stdout and "nhãn 1" in stdout,
             "chi ro tung nhan bi lech, khong chi noi chung chung")

        print("\n=== 4. Box nam ngoai anh -> mmdet bo im lang ===")
        train_ngoai = tao_coco(
            [anh_rong(1, "train_1.jpg")],
            [ann(1, 1, 1, (10, 10, 50, 50)),
             ann(2, 1, 2, (2000, 2000, 50, 50))])     # hoan toan ngoai anh 200x150
        ma, stdout, _ = chay_cli(dung_lai("ngoai_anh", {"train": train_ngoai}),
                                 cfg_mmdet_dung)
        kiem(ma != 0, f"box ngoai anh -> ma khac 0, thuc te {ma}")
        kiem("BỎ IM LẶNG" in stdout and "1 box" in stdout,
             "dem dung 1 box se bi bo")
        kiem("không giao với ảnh" in stdout, "noi ro ly do bi bo")

        print("\n=== 5. Anh co trong JSON nhung khong co tren dia ===")
        cfg = dung_lai("thieu_anh", dataset_mau())
        os.remove(os.path.join(tmp, "thieu_anh", "floodnet_coco", "images",
                               "train", "train_2.jpg"))
        ma, stdout, _ = chay_cli(cfg, cfg_mmdet_dung)
        kiem(ma != 0, f"thieu file anh -> ma khac 0, thuc te {ma}")
        kiem("KHÔNG có trên đĩa" in stdout, "bao ro la thieu file anh")
        kiem("train_2.jpg" in stdout, "goi ten file anh bi thieu")

        print("\n=== 6. Annotation id trung nhau ===")
        train_trung = tao_coco(
            [anh_rong(1, "train_1.jpg")],
            [ann(7, 1, 1, (10, 10, 50, 50)), ann(7, 1, 2, (100, 100, 40, 40))])
        ma, stdout, _ = chay_cli(dung_lai("trung_id", {"train": train_trung}),
                                 cfg_mmdet_dung)
        kiem(ma != 0, f"annotation id trung -> ma khac 0, thuc te {ma}")
        kiem("annotation id trùng" in stdout, "goi dung loi id trung")

        print("\n=== 7. Thu tu nhan mmdet: theo JSON, KHONG theo metainfo ===")
        # Phep thu don vi nay khoa dung cai co che da lam sai tai lieu mot lan:
        # `metainfo` quyet dinh TAP nao duoc giu, con THU TU thi theo file JSON.
        coco = tao_coco([], [],
                        thu_tu_category=("non_flooded_building", "flooded_building"))
        thu_tu, bi_thieu = thu_tu_nhan_mmdet(
            coco, ("flooded_building", "non_flooded_building"))
        kiem(thu_tu == ["non_flooded_building", "flooded_building"],
             f"thu tu theo JSON chu khong theo metainfo, thuc te {thu_tu}")
        kiem(bi_thieu == [], "khong ten nao bi thieu")

        # metainfo khai THUA mot ten o CUOI: nhan 0/1 van dung, chi la thua.
        thu_tu2, bi_thieu2 = thu_tu_nhan_mmdet(
            coco, ("non_flooded_building", "flooded_building", "lop_thua"))
        kiem(thu_tu2 == ["non_flooded_building", "flooded_building"],
             f"ten khai thua khong lot vao thu tu nhan, thuc te {thu_tu2}")
        kiem(bi_thieu2 == ["lop_thua"], f"nhung phai bao la thua, thuc te {bi_thieu2}")

        print("\n=== 8. ly_do_mmdet_bo: ba dieu kien cua _parse_ann_info ===")
        im = {"width": W, "height": H}
        kiem(ly_do_mmdet_bo(ann(1, 1, 1, (10, 10, 50, 50)), im) is None,
             "box nam gon trong anh -> giu")
        # Box THO RA NGOAI nhung van giao voi anh PHAI DUOC GIU: mmdet chi doi
        # hoi dien tich giao > 0, no khong cat box. Kiem sai cho nay thi script
        # bao dong gia tren dung nhung anh co nha sat mep — ma 46,9% box FloodNet
        # cham mep, tuc la gan mot nua du lieu se bi bao sai.
        kiem(ly_do_mmdet_bo(ann(1, 1, 1, (150, 100, 100, 100)), im) is None,
             "box tran ra ngoai nhung con giao anh -> VAN GIU (mmdet khong cat)")
        kiem(ly_do_mmdet_bo(ann(1, 1, 1, (250, 10, 50, 50)), im) is not None,
             "box hoan toan ngoai anh -> bo")
        kiem(ly_do_mmdet_bo(ann(1, 1, 1, (10, 10, 0, 50)), im) is not None,
             "canh w = 0 -> bo (mmdet doi w >= 1)")
        kiem(ly_do_mmdet_bo(ann(1, 1, 1, (10, 10, 50, 50), area=0), im) is not None,
             "area = 0 -> bo")
        kiem(ly_do_mmdet_bo(ann(1, 1, 1, (10, 10, 0.5, 0.5)), im) is not None,
             "canh < 1px -> bo")

        print("\n=== 9. thong_ke ===")
        tk = thong_ke(dataset_mau()["train"], {1: "flooded_building",
                                              2: "non_flooded_building"})
        kiem(tk["so_anh"] == 3 and tk["so_box"] == 3,
             f"3 anh / 3 box, thuc te {tk['so_anh']}/{tk['so_box']}")
        kiem(tk["so_anh_rong"] == 1,
             f"1 anh khong co box nao (train_3), thuc te {tk['so_anh_rong']}")
        kiem(tk["dem_theo_lop"] == {"flooded_building": 1, "non_flooded_building": 2},
             f"dem theo lop, thuc te {tk['dem_theo_lop']}")
        kiem(tk["canh_nho_nhat"] == 30.0 and tk["canh_lon_nhat"] == 50.0,
             f"canh nho cua box: min 30, max 50, thuc te "
             f"{tk['canh_nho_nhat']}/{tk['canh_lon_nhat']}")
        kiem(tk["cac_co_anh"] == [(200, 150)], f"co anh, thuc te {tk['cac_co_anh']}")

        print("\n=== 10. Doc metainfo.classes tu config ===")
        ten, nguon = doc_classes_tu_config_mmdet(str(cfg_mmdet_dung))
        kiem(ten == ("flooded_building", "non_flooded_building"),
             f"doc dung ten lop tu config, thuc te {ten}")
        # Tren may CPU khong co mmengine thi phai roi ve regex; tren Colab thi
        # phai dung mmengine. Test chap nhan ca hai nhung doi nguon phai noi ro.
        kiem(nguon in ("mmengine", "regex"),
             f"noi ro dang doc bang gi ({nguon}) — khong doc am tham")
        ten_xau, _ = doc_classes_tu_config_mmdet(os.path.join(tmp, "khong-co.py"))
        kiem(ten_xau is None, "file khong ton tai -> tra None chu khong crash")

        print("\n=== 11. Config thieu `preprocess.classes` -> bao ro, khong traceback ===")
        # Loi nay gap that khi viet test: thieu khoa thi `cfg["preprocess"]` nem
        # KeyError, traceback chi vao dong code chu khong chi vao file cau hinh.
        # Tren Colab nhin traceback do khong biet phai sua gi.
        cfg_thieu = os.path.join(tmp, "thieu_khoa.yaml")
        with open(cfg_thieu, "w", encoding="utf-8") as f:
            f.write("paths:\n  processed_dir: /khong/co/that\n")
        ma, stdout, _ = chay_cli(cfg_thieu, cfg_mmdet_dung)
        kiem(ma != 0, f"thieu khoa -> ma khac 0, thuc te {ma}")
        kiem("thiếu `preprocess.classes`" in stdout,
             "noi ro THIEU KHOA NAO, khong do traceback ra man hinh")
        kiem("Traceback" not in stdout,
             "khong de traceback lot ra stdout (Colab khong hien stderr cua tien trinh con)")

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

# -*- coding: utf-8 -*-
"""Dựng tập 20 ảnh con để chạy phép thử "học vẹt" của GATE 3.

VÌ SAO PHẢI CÓ PHÉP THỬ NÀY

Trước khi trả tiền cho 24 epoch trên T4, phải chứng minh đường ống chạy được
đầu-cuối: dữ liệu đọc ra đúng, nhãn gắn đúng lớp, loss giảm, model có thể NHỚ
được (không phải chỉ đoán bừa). Cách rẻ nhất là lấy 20 ảnh rồi bắt model học
thuộc lòng chúng. Nếu sau 60 epoch mà loss không về gần 0 trên chính 20 ảnh đó
thì có gì đó hỏng ở đường ống — và lúc đó ta mới mất vài phút, không phải vài giờ.

ĐIỀU PHÉP THỬ NÀY **KHÔNG** CHỨNG MINH
Học vẹt được 20 ảnh KHÔNG nói gì về chất lượng phân loại ngập/không ngập. Một
model chỉ cần nhìn "có phải ảnh này không" là đủ để đạt loss ~0, không cần học
gì về nhà ngập. Đừng trích con số của phép thử này vào báo cáo như một kết quả.

CHỌN 20 ẢNH NHƯ THẾ NÀO (và vì sao không lấy 20 ảnh đầu danh sách)
Lấy 20 ảnh đầu theo tên file là lấy 20 ảnh liên tiếp cùng một khu vực chụp —
chúng gần giống nhau, nên model chỉ cần nhớ một kiểu ảnh là xong, và ta không
kiểm được gì về khả năng phân biệt. Ở đây chia 20 ảnh thành ba nhóm theo ĐÚNG
thứ mà model phải phân biệt:

  * 12 ảnh CÓ nhà ngập      — nhóm chính, có cả hai lớp cùng lúc.
  *  6 ảnh chỉ có nhà KHÔNG ngập — để phép thử bắt buộc phải phân biệt hai lớp.
  *  2 ảnh KHÔNG có nhà nào  — vì dữ liệu thật có hơn một nửa số ảnh như vậy, và
    `filter_empty_gt=False` giữ chúng lại. Nếu 20 ảnh toàn ảnh có nhà thì phép
    thử không chạm tới nhánh "ảnh rỗng" — đúng nhánh dễ hỏng nhất.
    Con số đọc từ Phase 2 (docs/NOTES.md §2.4): 2.343 ảnh, trong đó 245 ảnh có
    nhà ngập và 880 ảnh có nhà không ngập. Hai tập này có thể chồng nhau nên số
    ảnh KHÔNG có nhà nào tối thiểu là 2.343 − (245 + 880) = 1.218 ảnh (52,0%).
    Đây là cận dưới suy ra được từ số đã đo, không phải một con số đo mới.

Mỗi nhóm lấy bằng `random.Random(seed).sample` trên danh sách ĐÃ SẮP XẾP theo
`file_name`, nên chạy lại luôn ra đúng 20 ảnh đó, không phụ thuộc thứ tự file
trên đĩa (thứ tự này khác nhau giữa Windows và Linux).

File JSON sinh ra KHÔNG chép ảnh: nó dùng lại `images/train/` và chỉ trỏ tới.
"""

import json
import os
import random
from collections import Counter


def phan_nhom(coco, ten_ngap="flooded_building"):
    """Chia ảnh của một split thành ba nhóm: có nhà ngập / chỉ nhà không ngập / rỗng.

    Returns:
        dict: `{"co_ngap": [...], "chi_khong": [...], "rong": [...]}` — mỗi phần
        tử là dict image của COCO, đã sắp xếp theo `file_name`.
    """
    ten_theo_id = {c["id"]: c["name"] for c in coco["categories"]}
    dem_ngap = Counter()
    ds_anh_co_box = set()
    for a in coco["annotations"]:
        ds_anh_co_box.add(a["image_id"])
        if ten_theo_id.get(a["category_id"]) == ten_ngap:
            dem_ngap[a["image_id"]] += 1

    nhom = {"co_ngap": [], "chi_khong": [], "rong": []}
    for im in sorted(coco["images"], key=lambda i: i["file_name"]):
        if dem_ngap.get(im["id"], 0) > 0:
            nhom["co_ngap"].append(im)
        elif im["id"] in ds_anh_co_box:
            nhom["chi_khong"].append(im)
        else:
            nhom["rong"].append(im)
    return nhom


def chon_anh(nhom, so_co_ngap=12, so_chi_khong=6, so_rong=2, seed=42):
    """Lấy ngẫu nhiên nhưng TÁI LẬP ĐƯỢC từng nhóm. Trả về (danh sách, cảnh báo)."""
    yeu_cau = {"co_ngap": so_co_ngap, "chi_khong": so_chi_khong, "rong": so_rong}
    rng = random.Random(seed)
    chon, canh_bao = [], []
    for ten_nhom, can in yeu_cau.items():
        ds = nhom[ten_nhom]
        if len(ds) < can:
            # Không im lặng lấy thiếu: 20 ảnh mà thiếu nhóm rỗng thì phép thử
            # không còn chạm tới nhánh dễ hỏng nhất, và người đọc kết quả sẽ
            # tưởng nó đã kiểm rồi.
            canh_bao.append(
                f"nhóm '{ten_nhom}' chỉ có {len(ds)} ảnh, cần {can} — lấy hết "
                f"{len(ds)} ảnh. Phép thử vẫn chạy nhưng KHÔNG còn kiểm được "
                f"nhánh này.")
        chon += rng.sample(ds, min(can, len(ds)))
    chon.sort(key=lambda i: i["file_name"])
    return chon, canh_bao


def cat_coco(coco, ds_anh_chon):
    """Cắt COCO xuống đúng `ds_anh_chon`, đánh lại id 1..N.

    `categories` được CHÉP NGUYÊN VĂN — kể cả thứ tự. Đây không phải chi tiết
    nhỏ: nhãn mà mmdet gán cho từng lớp phụ thuộc thứ tự category trong file
    JSON, nên đổi thứ tự ở đây là đảo nhãn ngập/không ngập của cả tập overfit,
    và phép thử "học vẹt" sẽ vẫn đạt (nó chỉ cần nhớ, không cần đúng lớp) —
    đúng kiểu lỗi đi qua được chốt kiểm.
    """
    id_cu = [im["id"] for im in ds_anh_chon]
    anh_moi = {cu: moi for moi, cu in enumerate(id_cu, start=1)}

    images = []
    for im in ds_anh_chon:
        moi = dict(im)
        moi["id"] = anh_moi[im["id"]]
        images.append(moi)

    annotations = []
    for a in coco["annotations"]:
        if a["image_id"] not in anh_moi:
            continue
        moi = dict(a)
        moi["id"] = len(annotations) + 1
        moi["image_id"] = anh_moi[a["image_id"]]
        annotations.append(moi)

    return {
        "info": {"description": f"Tập con overfit của {coco.get('info', {}).get('description', '')}",
                 "version": "1.0"},
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def tao_overfit20(coco, so_co_ngap=12, so_chi_khong=6, so_rong=2, seed=42):
    """Trọn gói: chia nhóm -> chọn -> cắt. Trả về (coco_moi, thong_tin)."""
    nhom = phan_nhom(coco)
    chon, canh_bao = chon_anh(nhom, so_co_ngap, so_chi_khong, so_rong, seed)
    coco_moi = cat_coco(coco, chon)

    ten_theo_id = {c["id"]: c["name"] for c in coco["categories"]}
    dem = Counter(ten_theo_id[a["category_id"]] for a in coco_moi["annotations"])
    thong_tin = {
        "so_anh": len(coco_moi["images"]),
        "so_box": len(coco_moi["annotations"]),
        "dem_theo_lop": dict(sorted(dem.items())),
        "co_san": {k: len(v) for k, v in nhom.items()},
        "canh_bao": canh_bao,
        "ds_anh": [im["file_name"] for im in coco_moi["images"]],
    }
    return coco_moi, thong_tin


def bao_cao(thong_tin):
    """Văn bản mô tả tập con, để in ra và để dán lại khi cần đối chiếu."""
    d = [f"Tập overfit: {thong_tin['so_anh']} ảnh / {thong_tin['so_box']} box"]
    for ten, n in thong_tin["dem_theo_lop"].items():
        d.append(f"  {ten:<24s} {n:>4} box")
    d.append("  Nguồn ảnh trong train: " + ", ".join(
        f"{k}={v}" for k, v in sorted(thong_tin["co_san"].items())))
    for c in thong_tin["canh_bao"]:
        d.append(f"  [!] {c}")
    d.append("  20 ảnh đã chọn: " + ", ".join(thong_tin["ds_anh"]))
    return "\n".join(d)


# ===========================================================================
# main
# ===========================================================================

def main():
    import argparse
    import sys
    from collections import Counter as _Counter

    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    import yaml

    ap = argparse.ArgumentParser(
        description="Dựng tập con overfit cho phép thử GATE 3")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--processed-dir", default=None,
                    help="Ghi đè thư mục dataset (mặc định lấy từ config)")
    ap.add_argument("--split-nguon", default="train")
    ap.add_argument("--ten-file-ra", default="instances_overfit20.json")
    ap.add_argument("--so-anh", type=int, default=20,
                    help="Tổng số ảnh (mặc định 20)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Mặc định lấy `eda.seed` trong config")
    args = ap.parse_args()

    print("=" * 72)
    print("DỰNG TẬP CON OVERFIT (GATE 3 — phép thử học vẹt)")
    print("=" * 72)

    if not os.path.exists(args.config):
        print(f"[!] Không thấy file cấu hình:\n    {args.config}")
        raise SystemExit(1)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    processed_dir = args.processed_dir or cfg["paths"]["processed_dir"]
    seed = args.seed if args.seed is not None else cfg["eda"]["seed"]

    # Chia 20 ảnh theo tỉ lệ 12/6/2 đã ghi trong docstring. Suy ra từ `--so-anh`
    # để đổi tổng số vẫn giữ đúng tỉ lệ (20 -> 12/6/2, 10 -> 6/3/1).
    tong = args.so_anh
    so_co_ngap = max(1, round(tong * 0.6))
    so_rong = max(1, round(tong * 0.1))
    so_chi_khong = max(1, tong - so_co_ngap - so_rong)

    duong_nguon = os.path.join(processed_dir, "annotations",
                               f"instances_{args.split_nguon}.json")
    if not os.path.exists(duong_nguon):
        print(f"[!] Không thấy annotation nguồn:\n    {duong_nguon}\n"
              f"    Chạy Phase 2 (notebook 02) trước.")
        raise SystemExit(1)
    with open(duong_nguon, encoding="utf-8") as f:
        coco = json.load(f)

    print(f"nguồn      : {duong_nguon}")
    print(f"chia nhóm  : {so_co_ngap} có nhà ngập / {so_chi_khong} chỉ nhà không ngập"
          f" / {so_rong} không có nhà")
    print(f"seed       : {seed}  (chạy lại cho ra ĐÚNG 20 ảnh này)")

    coco_moi, thong_tin = tao_overfit20(
        coco, so_co_ngap, so_chi_khong, so_rong, seed)
    print()
    print(bao_cao(thong_tin))

    if thong_tin["so_anh"] != tong:
        print(f"\n[!] Chỉ chọn được {thong_tin['so_anh']}/{tong} ảnh — xem cảnh báo trên.")
    if not coco_moi["annotations"]:
        print("\n[!] Tập con không có box nào — vô nghĩa. Dừng.")
        raise SystemExit(1)

    duong_ra = os.path.join(processed_dir, "annotations", args.ten_file_ra)
    with open(duong_ra, "w", encoding="utf-8") as f:
        json.dump(coco_moi, f, ensure_ascii=False)

    # Đọc LẠI từ đĩa rồi mới kết luận — nếu khâu ghi hỏng thì phải lộ ra ở đây,
    # chứ không phải ở epoch 30 trên Colab.
    with open(duong_ra, encoding="utf-8") as f:
        lai = json.load(f)
    dem_lai = _Counter(a["category_id"] for a in lai["annotations"])
    khop = (len(lai["images"]) == len(coco_moi["images"])
            and len(lai["annotations"]) == len(coco_moi["annotations"])
            and dict(dem_lai) == dict(_Counter(
                a["category_id"] for a in coco_moi["annotations"])))
    print(f"\nĐọc lại từ đĩa: {len(lai['images'])} ảnh / {len(lai['annotations'])} box "
          f"-> {'KHỚP' if khop else 'LỆCH, kiểm lại đĩa!'}")
    if not khop:
        raise SystemExit(1)

    print("\n" + "=" * 72)
    print(f"Đã ghi: {duong_ra}")
    print("Ảnh KHÔNG bị chép lại — file này dùng chung images/"
          f"{args.split_nguon}/ với tập train.")
    print("=" * 72)


if __name__ == "__main__":
    main()

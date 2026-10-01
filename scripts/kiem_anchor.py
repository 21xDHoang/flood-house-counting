#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Đo dải anchor của RPN trên box thật của FloodNet (Phase 3, trước khi train).

VÌ SAO PHẢI LÀ MỘT SCRIPT RIÊNG

RPN sinh anchor theo công thức cố định trong config mmdet, không liên quan gì
tới dữ liệu. Nếu dải anchor không phủ tới cỡ nhà của FloodNet thì những căn nhà
to nhất không bao giờ khớp được với anchor nào — model vẫn chạy, loss vẫn giảm,
chỉ có recall thấp một cách khó hiểu, và phải đến lúc đọc bảng AP mới thấy. Đo
trước mất vài giây; đoán sai mất một suất train trên Colab.

Tham số anchor được đọc TỪ CHÍNH CONFIG ĐỒ ÁN (đã phân giải `_base_` bằng
mmengine), không chép tay. Config đồ án thừa hưởng RPN từ config mmdet nên nhìn
vào file .py của đồ án sẽ không thấy `scales`/`ratios` — chép tay từ tài liệu
là chép sai. Khi mmengine không có (máy CPU chưa cài), script rơi về bộ giá trị
mặc định trong `floodcount.models.anchor` và NÓI RÕ là đang dùng đường lui.

Toàn bộ phần tính toán nằm ở `src/floodcount/models/anchor.py` (numpy thuần, có
test riêng). Ở đây chỉ đọc dữ liệu, gọi hàm, in báo cáo.

Chạy:
    python scripts/kiem_anchor.py --config configs/mmdet/cascade_convnext_t_floodnet.py \\
        --processed-dir /content/floodnet_coco --split train
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from floodcount.models.anchor import (RATIOS_MAC_DINH, SCALES_MAC_DINH,  # noqa: E402
                                      STRIDES_MAC_DINH, bao_cao, phan_tich)


def doc_rpn_tu_config(duong_config):
    """Đọc (strides, scales, ratios) của `rpn_head.anchor_generator`.

    Returns:
        tuple: (strides, scales, ratios, nguon). `nguon` là "mmengine" khi đọc
        được từ config, hoặc câu giải thích vì sao phải dùng giá trị mặc định.
    """
    try:
        from mmengine.config import Config
    except ImportError:
        return (STRIDES_MAC_DINH, SCALES_MAC_DINH, RATIOS_MAC_DINH,
                "mmengine không có trên máy này -> dùng giá trị mặc định trong "
                "floodcount/models/anchor.py")

    if not os.path.exists(duong_config):
        return (STRIDES_MAC_DINH, SCALES_MAC_DINH, RATIOS_MAC_DINH,
                f"không thấy {duong_config} -> dùng giá trị mặc định")

    # `Config.fromfile` phân giải `_base_` và cả tiền tố `mmdet::`, nên đây là
    # giá trị THẬT SAU KHI KẾ THỪA. Đọc file .py bằng mắt sẽ không thấy các khoá
    # này vì chúng nằm ở config gốc của mmdet.
    cfg = Config.fromfile(duong_config)
    ag = cfg.model.rpn_head.anchor_generator
    return (tuple(ag.get("strides", STRIDES_MAC_DINH)),
            tuple(ag.get("scales", SCALES_MAC_DINH)),
            tuple(ag.get("ratios", RATIOS_MAC_DINH)),
            "mmengine (đã phân giải _base_)")


def doc_box(duong_json, ten_lop_ngap="flooded_building"):
    """Đọc box từ COCO JSON -> (mảng (N,4) xywh, mảng nhãn, dict tên theo id)."""
    with open(duong_json, encoding="utf-8") as f:
        coco = json.load(f)
    ten_theo_id = {c["id"]: c["name"] for c in coco["categories"]}
    ds_box, ds_nhan = [], []
    for a in coco["annotations"]:
        ds_box.append(a["bbox"])
        ds_nhan.append(a["category_id"])
    boxes = np.array(ds_box, dtype=np.float64).reshape(-1, 4)
    nhan = np.array(ds_nhan, dtype=np.int64).reshape(-1)
    return boxes, nhan, ten_theo_id


def main():
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(
        description="Đo độ phủ anchor của RPN trên box FloodNet")
    ap.add_argument("--config",
                    default="configs/mmdet/cascade_convnext_t_floodnet.py")
    ap.add_argument("--processed-dir", default=None,
                    help="Thư mục dataset; bỏ trống thì đọc từ config đồ án")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--nguong", default="0.3,0.5,0.7",
                    help="Các mốc IoU cần đếm tỉ lệ đạt được")
    args = ap.parse_args()

    print("=" * 72)
    print("ĐO ĐỘ PHỦ ANCHOR CỦA RPN — PHASE 3")
    print("=" * 72)

    strides, scales, ratios, nguon = doc_rpn_tu_config(args.config)
    print(f"config RPN : {args.config}")
    print(f"nguồn      : {nguon}")
    print(f"strides    : {strides}")
    print(f"scales     : {scales}")
    print(f"ratios     : {ratios}")

    # Đối chiếu với hằng số trong anchor.py. Lệch nghĩa là RPN đã bị đổi mà tài
    # liệu chưa cập nhật — không phải lỗi, nhưng phải nhìn thấy chứ không để im.
    if (tuple(strides) != tuple(STRIDES_MAC_DINH)
            or tuple(scales) != tuple(SCALES_MAC_DINH)
            or tuple(ratios) != tuple(RATIOS_MAC_DINH)):
        print("\n[!] RPN trong config KHÁC giá trị ghi trong "
              "floodcount/models/anchor.py:")
        print(f"    anchor.py ghi: strides={STRIDES_MAC_DINH}, "
              f"scales={SCALES_MAC_DINH}, ratios={RATIOS_MAC_DINH}")
        print("    Bảng dưới đây tính bằng giá trị TỪ CONFIG (đúng hơn), nhưng "
              "phải sửa anchor.py cho khớp.")
    else:
        print("           (khớp với hằng số trong floodcount/models/anchor.py)")

    processed_dir = args.processed_dir
    if processed_dir is None:
        try:
            from mmengine.config import Config
            processed_dir = Config.fromfile(args.config).data_root
        except ImportError:
            processed_dir = None
    if not processed_dir:
        print("\n[!] Không biết thư mục dataset — truyền --processed-dir.")
        raise SystemExit(1)

    duong_ann = os.path.join(processed_dir, "annotations",
                             f"instances_{args.split}.json")
    if not os.path.exists(duong_ann):
        print(f"\n[!] Không thấy annotation:\n    {duong_ann}\n"
              f"    Chạy Phase 2 (notebook 02) và giải nén dataset trước.")
        raise SystemExit(1)

    boxes, nhan, ten_theo_id = doc_box(duong_ann)
    print(f"\ndữ liệu    : {duong_ann}  ({len(boxes):,} box)")
    if not len(boxes):
        print("[!] Không có box nào để đo.")
        raise SystemExit(1)

    cac_nguong = tuple(float(x) for x in args.nguong.split(","))
    ket_qua = phan_tich(boxes, strides, scales, ratios,
                        cac_nguong=cac_nguong, nhan=nhan)
    print()
    print(bao_cao(ket_qua, ten_lop=ten_theo_id))

    # Kết luận thành CÂU, để không phải tự đọc bảng rồi tự đoán.
    ti_le_05 = ket_qua["iou_thuc"].get("ti_le_iou>=0.5", float("nan"))
    print()
    print("-" * 72)
    if ti_le_05 >= 0.98:
        print(f"KẾT LUẬN: {ti_le_05 * 100:.1f}% box đạt IoU >= 0.5 với lưới anchor "
              f"thật -> dải anchor PHỦ TỐT. Không phải sửa RPN.")
    elif ti_le_05 >= 0.90:
        print(f"KẾT LUẬN: {ti_le_05 * 100:.1f}% box đạt IoU >= 0.5 — có "
              f"{(1 - ti_le_05) * 100:.1f}% box khó khớp. Chấp nhận được, nhưng "
              f"ghi lại con số này để Phase 6 phân tích lỗi có căn cứ.")
    else:
        print(f"KẾT LUẬN: chỉ {ti_le_05 * 100:.1f}% box đạt IoU >= 0.5 — dải anchor "
              f"KHÔNG phủ đủ. Phải sửa `scales`/`ratios` của RPN trong config đồ "
              f"án TRƯỚC khi train, nếu không recall sẽ thấp mà không rõ vì sao.")
    print("-" * 72)


if __name__ == "__main__":
    main()

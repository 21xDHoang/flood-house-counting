# -*- coding: utf-8 -*-
"""Suy luận: đưa ảnh vào, NHÌN thấy mô hình nhận diện và đếm nhà ngập.

Hai chế độ:

  1. `--anh-dir <thư mục>` — chạy trên ẢNH CỦA NGƯỜI DÙNG (ảnh điện thoại, ảnh
     chụp màn hình, bất kỳ). Không có nhãn thật nên chỉ in số mô hình đếm.
     Trên Colab, thư mục hay dùng là `MyDrive/Flood_House_AI/anh_cua_toi/`.

  2. Không có `--anh-dir` — demo trên split của dataset (mặc định `val`), CÓ
     ĐỐI CHIẾU với nhãn thật đọc từ file COCO: mỗi ảnh in "mô hình đếm X nhà
     ngập / nhãn thật Y". Đây là cách duy nhất để biết con số đếm được có đáng
     tin không, vì mAP không nói gì về chuyện đếm.

Cả hai chế độ vẽ overlay (box đỏ = nhà ngập, xanh = nhà không ngập, kèm điểm)
và ghi `du_doan.json` để lần sau khỏi chạy lại.

Vì sao script này import `train`:
    Checkpoint mmengine chứa `HistoryBuffer`, mà torch >= 2.6 từ chối nạp mọi
    lớp không nằm trong danh sách an toàn — lỗi thật đã gặp trên Colab
    (docs/NOTES.md §3.13). `scripts/train.py` có sẵn bản vá
    `va_torch_load_resume`; suy luận nạp checkpoint y hệt đường resume nên
    phải dùng ĐÚNG bản vá đó. Chép lại vào đây là cách chắc chắn nhất để một
    ngày nào đó vá một chỗ mà quên chỗ kia.

Chạy trên Colab (sau khi đã train xong, có checkpoint trong work_dir):
    python scripts/du_doan.py --anh-dir /content/drive/MyDrive/Flood_House_AI/anh_cua_toi
    python scripts/du_doan.py --so-anh 6            # demo val, có đối chiếu nhãn
"""

import argparse
import json
import os
import pathlib
import sys
import time

GOC_REPO = pathlib.Path(__file__).resolve().parents[1]

# PHẢI đứng trước `Config.fromfile` — cùng lý do đã ghi ở scripts/train.py:
# `custom_imports` của config cần `src/` có trên sys.path mới import được
# `floodcount.models.transforms`.
sys.path.insert(0, str(GOC_REPO / "src"))
# `scripts/` để import `train` (bản vá torch.load, xem docstring đầu tệp).
sys.path.insert(0, str(GOC_REPO / "scripts"))

from floodcount.data.kiem_tra import (doc_classes_tu_data_yaml,  # noqa: E402
                                      doc_json)
from floodcount.infer.dem import (chi_muc_gt, chon_anh_demo,  # noqa: E402
                                  dem_theo_lop, liet_ke_anh, tim_checkpoint)
import train as _train  # noqa: E402

# Tên ngắn để vẽ lên ảnh. cv2.putText KHÔNG vẽ được chữ có dấu, nên mọi chữ
# trên overlay phải là ASCII — tên đầy đủ vẫn in ra console bình thường.
TEN_NGAN = {"flooded_building": "NGAP", "non_flooded_building": "KHONG NGAP"}
# BGR (OpenCV), không phải RGB. Đỏ = ngập, xanh dương = không ngập; hai màu này
# khác nhau cả ở người mù màu đỏ-lục.
MAU_BGR = {"flooded_building": (0, 0, 255),
           "non_flooded_building": (255, 128, 0)}


def in_utf8():
    """Ép stdout/stderr sang utf-8 (console Windows mặc định là cp1252)."""
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def pipeline_suy_luan(cfg, dau_vao_bang_mang=False):
    """Pipeline chạy ảnh MỚI, lấy từ chính `test_pipeline` của config.

    Bỏ `LoadAnnotations`: nó chỉ có nghĩa khi ảnh đi kèm nhãn. Ảnh mới không có
    `instances`, mmdet 3.3.0 đọc `results.get('instances', [])` nên không nổ,
    nhưng giữ lại thì mỗi ảnh còn dựng thêm mấy mảng gt rỗng vô nghĩa — và ý
    định "đây là suy luận, không phải chấm điểm" không còn hiện ra ở đâu cả.

    `dau_vao_bang_mang=True` đổi bước đầu thành `LoadImageFromNDArray` — dùng
    cho web (ảnh từ trình duyệt là mảng numpy, không có đường dẫn). Đây đúng
    là cách mmdet 3.3.0 tự làm trong `mmdet/apis/inference.py` (dòng 157).
    """
    from mmcv.transforms import Compose

    buoc = [dict(t) for t in cfg.test_dataloader.dataset.pipeline
            if t.get("type") != "LoadAnnotations"]
    if dau_vao_bang_mang:
        buoc[0]["type"] = "mmdet.LoadImageFromNDArray"
    return Compose(buoc)


def nap_model(cfg, duong_checkpoint, thiet_bi):
    """Dựng model + nạp checkpoint.

    PHẢI gọi SAU `_train.va_torch_load_resume()` — torch >= 2.6 từ chối
    `HistoryBuffer` trong checkpoint mmengine nếu chưa cho phép trước (§3.13).
    """
    from mmdet.apis import init_detector
    return init_detector(cfg, duong_checkpoint, device=thiet_bi)


def chay_mot_anh(model, pipeline, duong_anh, nguong_diem, ten_lop):
    """Chạy 1 ảnh (theo đường dẫn), trả dict kết quả — cùng cấu trúc cho cả hai
    chế độ, để web dùng lại được nguyên vẹn.

    Đúng theo `inference_detector` của mmdet 3.3.0: pipeline tự dựng batch, rồi
    `test_step` tự gọi `data_preprocessor` bên trong — KHÔNG gọi tay lần nữa
    (gọi hai lần là chuẩn hoá ảnh hai lần, điểm số lệch hết).
    """
    return _chay_mot_anh_mau(model, pipeline, dict(img_path=str(duong_anh)),
                             nguong_diem, ten_lop)


def chay_mot_mang(model, pipeline, mang_bgr, nguong_diem, ten_lop):
    """Như `chay_mot_anh` nhưng ảnh là mảng numpy BGR (đường của web)."""
    return _chay_mot_anh_mau(model, pipeline, dict(img=mang_bgr),
                             nguong_diem, ten_lop)


def _chay_mot_anh_mau(model, pipeline, dau_vao, nguong_diem, ten_lop):
    import torch

    dau_vao["img_id"] = 0
    mau = pipeline(dau_vao)
    batch = {"inputs": [mau["inputs"]], "data_samples": [mau["data_samples"]]}

    t0 = time.perf_counter()
    with torch.no_grad():
        ket = model.test_step(batch)[0]
    ms = (time.perf_counter() - t0) * 1000

    pi = ket.pred_instances
    giu = pi.scores >= nguong_diem
    labels = pi.labels[giu].tolist()
    scores = pi.scores[giu].tolist()
    return {
        "dem": dem_theo_lop(labels, scores, ten_lop, nguong_diem),
        "boxes": [[round(float(v), 1) for v in b] for b in pi.bboxes[giu].tolist()],
        "labels": labels,
        "scores": [round(float(s), 3) for s in scores],
        "ms": round(ms, 1),
    }


def ve_anh(anh_bgr, ket_qua, ten_lop, dong_tieu_de):
    """Vẽ box + dải tiêu đề lên ảnh, TRẢ VỀ mảng mới (không ghi đĩa).

    Tách khỏi phần ghi tệp để web dùng lại: web cần mảng trả về trình duyệt,
    không cần tệp.
    """
    import cv2

    anh = anh_bgr.copy()
    h, w = anh.shape[:2]
    day = max(2, round(max(h, w) / 700))
    co_chu = max(0.7, max(h, w) / 1600)

    for box, nhan, diem in zip(ket_qua["boxes"], ket_qua["labels"],
                               ket_qua["scores"]):
        x1, y1, x2, y2 = (int(v) for v in box)
        ten = ten_lop[int(nhan)]
        mau = MAU_BGR.get(ten, (0, 255, 0))
        cv2.rectangle(anh, (x1, y1), (x2, y2), mau, day)
        chu = f"{TEN_NGAN.get(ten, ten)} {diem:.2f}"
        (rong_chu, cao_chu), _ = cv2.getTextSize(
            chu, cv2.FONT_HERSHEY_SIMPLEX, co_chu, day)
        y_chu = y1 - 6 if y1 - 6 - cao_chu > 0 else y2 + cao_chu + 6
        cv2.rectangle(anh, (x1, y_chu - cao_chu - 4),
                      (x1 + rong_chu + 6, y_chu + 4), mau, -1)
        cv2.putText(anh, chu, (x1 + 3, y_chu), cv2.FONT_HERSHEY_SIMPLEX,
                    co_chu, (255, 255, 255), day, cv2.LINE_AA)

    if dong_tieu_de:
        cv2.rectangle(anh, (0, 0), (w, int(34 * co_chu) + 12), (0, 0, 0), -1)
        cv2.putText(anh, dong_tieu_de, (10, int(26 * co_chu)),
                    cv2.FONT_HERSHEY_SIMPLEX, co_chu, (255, 255, 255),
                    day, cv2.LINE_AA)
    return anh


def luu_overlay(duong_anh, ket_qua, ten_lop, duong_ra, dong_tieu_de):
    """Đọc ảnh từ đĩa, vẽ, ghi ra `duong_ra` (JPEG chất lượng 92)."""
    import cv2

    anh = cv2.imread(str(duong_anh))
    if anh is None:
        raise ValueError(f"OpenCV không đọc được ảnh: {duong_anh}")
    ra = os.path.dirname(duong_ra)
    if ra:
        os.makedirs(ra, exist_ok=True)
    cv2.imwrite(duong_ra, ve_anh(anh, ket_qua, ten_lop, dong_tieu_de),
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def _dong_tieu_de(ten_anh, ket_qua, nhan_that, ten_lop):
    """Dòng chữ ASCII vẽ trên overlay (cv2 không vẽ được dấu tiếng Việt)."""
    n_ngap = ket_qua["dem"].get("flooded_building", 0)
    n_khong = ket_qua["dem"].get("non_flooded_building", 0)
    chu = f"{ten_anh} | MH dem: {n_ngap} ngap / {n_khong} khong ngap"
    if nhan_that is not None:
        g_ngap = nhan_that.get("flooded_building", 0)
        g_khong = nhan_that.get("non_flooded_building", 0)
        chu += f" | nhan that: {g_ngap} / {g_khong}"
    return chu


def in_bang(ket_qua_tung_anh, ten_lop, co_nhan_that):
    """In bảng đối chiếu. Cột nhãn thật chỉ hiện khi có (chế độ demo split)."""
    rong_ten = max([len(k["ten"]) for k in ket_qua_tung_anh] + [4])
    dau = f"{'Ảnh':<{rong_ten}} | {'Mô hình đếm':<22}"
    if co_nhan_that:
        dau += f" | {'Nhãn thật':<22}"
    print(dau + " | ms")
    print("-" * len(dau.encode("utf-8").decode("utf-8")))

    tong_mh = {"flooded_building": 0, "non_flooded_building": 0}
    tong_gt = {"flooded_building": 0, "non_flooded_building": 0}
    for k in ket_qua_tung_anh:
        mh = (f"{k['dem'].get('flooded_building', 0)} ngập / "
              f"{k['dem'].get('non_flooded_building', 0)} không")
        dong = f"{k['ten']:<{rong_ten}} | {mh:<22}"
        if co_nhan_that:
            nt = k.get("nhan_that") or {}
            dong += (f" | {nt.get('flooded_building', 0)} ngập / "
                     f"{nt.get('non_flooded_building', 0)} không".ljust(22))
            for ten in tong_gt:
                tong_gt[ten] += (k.get("nhan_that") or {}).get(ten, 0)
        for ten in tong_mh:
            tong_mh[ten] += k["dem"].get(ten, 0)
        print(dong + f" | {k['ms']:.0f}")

    print("-" * 60)
    print(f"TỔNG mô hình đếm : {tong_mh['flooded_building']} nhà ngập / "
          f"{tong_mh['non_flooded_building']} nhà không ngập")
    if co_nhan_that:
        print(f"TỔNG nhãn thật  : {tong_gt['flooded_building']} nhà ngập / "
              f"{tong_gt['non_flooded_building']} nhà không ngập")
    tb = sum(k["ms"] for k in ket_qua_tung_anh) / len(ket_qua_tung_anh)
    print(f"Thời gian        : {tb:.0f} ms/ảnh (trung bình, gồm cả đọc ảnh)")


def main():
    in_utf8()

    ap = argparse.ArgumentParser(
        description="Suy luận: đếm nhà ngập trên ảnh mới (ảnh của bạn) hoặc "
                    "demo trên split của dataset kèm đối chiếu nhãn thật")
    ap.add_argument("--config",
                    default="configs/mmdet/cascade_convnext_t_floodnet.py")
    ap.add_argument("--data-yaml", default="configs/data.yaml",
                    help="Nguồn sự thật cho tên lớp")
    ap.add_argument("--checkpoint", default=None,
                    help="Đường dẫn .pth; mặc định tự tìm bản best trong work_dir")
    ap.add_argument("--anh-dir", default=None,
                    help="Thư mục ẢNH CỦA BẠN (bỏ trống = demo trên split)")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"),
                    help="Split dùng khi không có --anh-dir (mặc định val)")
    ap.add_argument("--so-anh", type=int, default=0,
                    help="Số ảnh chạy (0 = tất cả). Chế độ demo mặc định lấy 6")
    ap.add_argument("--nguong-diem", type=float, default=0.3,
                    help="Ngưỡng điểm để giữ một box (mặc định 0.3, mmdet hay dùng)")
    ap.add_argument("--ra-dir", default=None,
                    help="Nơi ghi overlay + JSON (mặc định <work_dir>/du_doan/<giờ>)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed chọn ảnh demo — chạy lại ra đúng danh sách ảnh cũ")
    ap.add_argument("--thiet-bi", default="cuda:0")
    args = ap.parse_args()

    print("=" * 72)
    print("SUY LUẬN — CASCADE R-CNN + CONVNEXT-TINY TRÊN FLOODNET")
    print("=" * 72)
    print(f"config: {args.config}")

    try:
        from mmengine.config import Config
        from mmengine.version import __version__ as mmengine_ver
        import mmdet
        import mmcv
        import torch
    except ImportError as e:
        print(f"\n[!] Thiếu thư viện: {e}")
        print("    Trên Colab: chạy ô [3.3b] (cài môi trường) trước.")
        print("    Máy này (CPU) không cài mmdet — script này chỉ chạy được trên "
              "Colab.")
        raise SystemExit(1)

    print(f"mmdet {mmdet.__version__} | mmcv {mmcv.__version__} | "
          f"mmengine {mmengine_ver} | torch {torch.__version__}")

    cfg = Config.fromfile(args.config)
    ten_lop = list((cfg.get("metainfo") or {}).get("classes", []))
    if not ten_lop:
        print("[!] Config không khai `metainfo.classes` — không biết tên lớp.")
        raise SystemExit(1)
    try:
        ten_lop_yaml = doc_classes_tu_data_yaml(args.data_yaml)
        if ten_lop_yaml != ten_lop:
            print(f"\n[!] Tên lớp trong config {ten_lop} KHÁC trong "
                  f"{args.data_yaml} {ten_lop_yaml} — bảng kết quả sẽ in sai tên.")
    except ValueError as e:
        print(f"\n[!] {e}")

    # ---- checkpoint ----
    duong_ckpt = args.checkpoint or tim_checkpoint(cfg.work_dir)
    if duong_ckpt is None:
        print(f"\n[!] Không tìm thấy checkpoint nào trong work_dir:\n"
              f"    {cfg.work_dir}\n"
              f"    Chạy ô [3.11] (train thật) trên Colab trước — checkpoint "
              f"best nằm ở đó.\n"
              f"    Hoặc trỏ thẳng: --checkpoint <đường dẫn .pth>")
        raise SystemExit(1)
    print(f"checkpoint: {duong_ckpt}")

    # ---- danh sách ảnh ----
    nhan_that = {}      # file_name -> {tên lớp: số box} (chỉ chế độ demo)
    if args.anh_dir:
        if not os.path.isdir(args.anh_dir):
            print(f"\n[!] Không thấy thư mục ảnh: {args.anh_dir}")
            raise SystemExit(1)
        ds_anh = liet_ke_anh(args.anh_dir)
        if not ds_anh:
            print(f"\n[!] Thư mục không có ảnh nào đọc được: {args.anh_dir}\n"
                  f"    Trên Colab: mở Drive và tải ảnh vào\n"
                  f"        MyDrive/Flood_House_AI/anh_cua_toi/\n"
                  f"    rồi chạy lại. Nhận các đuôi: .jpg .jpeg .png .bmp "
                  f".webp .tif .tiff")
            raise SystemExit(1)
        if args.so_anh > 0:
            ds_anh = ds_anh[:args.so_anh]
        che_do = f"ảnh của bạn ({args.anh_dir})"
    else:
        ds_cfg = cfg[f"{args.split}_dataloader"].dataset
        goc = ds_cfg["data_root"]
        duong_ann = os.path.join(goc, ds_cfg["ann_file"])
        thu_muc_anh = os.path.join(goc, ds_cfg["data_prefix"]["img"])
        coco = doc_json(duong_ann)
        so_anh = args.so_anh if args.so_anh > 0 else 6
        ten_anh = chon_anh_demo(coco, so_anh, "flooded_building",
                                seed=args.seed)
        ds_anh = [os.path.join(thu_muc_anh, t) for t in ten_anh]
        nhan_that = chi_muc_gt(coco)
        che_do = f"demo split {args.split} ({so_anh} ảnh, có đối chiếu nhãn thật)"

    print(f"chế độ    : {che_do}")
    print(f"ngưỡng điểm: {args.nguong_diem}")
    thieu = [d for d in ds_anh if not os.path.isfile(d)]
    if thieu:
        print(f"\n[!] {len(thieu)} ảnh không tồn tại, ví dụ: {thieu[0]}\n"
              f"    Dataset đã giải nén chưa? Trên Colab chạy ô [3.5] trước.")
        raise SystemExit(1)

    ra_dir = args.ra_dir or os.path.join(cfg.work_dir, "du_doan",
                                         time.strftime("%Y%m%d_%H%M%S"))

    # ---- chạy ----
    if _train.va_torch_load_resume():
        print("  [vá] torch.load: đã cho phép HistoryBuffer + mảng numpy + lớp "
              "dtype + getattr (docs/NOTES.md §3.13).")
    model = nap_model(cfg, duong_ckpt, args.thiet_bi)
    pipeline = pipeline_suy_luan(cfg)

    print(f"\nChạy {len(ds_anh)} ảnh...\n")
    ket_qua_tung_anh = []
    loi = []
    for i, duong in enumerate(ds_anh, 1):
        ten = os.path.basename(duong)
        try:
            kq = chay_mot_anh(model, pipeline, duong, args.nguong_diem, ten_lop)
        except Exception as e:  # 1 ảnh hỏng không được giết cả lượt chạy
            loi.append(f"{ten}: {type(e).__name__}: {e}")
            print(f"  [{i}/{len(ds_anh)}] {ten}: LỖI — {e}")
            continue
        gt = nhan_that.get(ten) if nhan_that else None
        kq.update({"ten": ten, "nhan_that": gt})
        ket_qua_tung_anh.append(kq)

        ra_anh = os.path.join(ra_dir, "overlay", ten + ".jpg")
        luu_overlay(duong, kq, ten_lop, ra_anh,
                    _dong_tieu_de(ten, kq, gt, ten_lop))
        print(f"  [{i}/{len(ds_anh)}] {ten}: "
              f"{kq['dem'].get('flooded_building', 0)} ngập / "
              f"{kq['dem'].get('non_flooded_building', 0)} không ngập "
              f"({kq['ms']:.0f} ms)")

    if not ket_qua_tung_anh:
        print(f"\n*** CẢ {len(ds_anh)} ẢNH ĐỀU LỖI ***")
        for l in loi:
            print("   - " + l)
        raise SystemExit(1)

    print()
    in_bang(ket_qua_tung_anh, ten_lop, bool(nhan_that))

    os.makedirs(ra_dir, exist_ok=True)
    duong_json = os.path.join(ra_dir, "du_doan.json")
    with open(duong_json, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config,
            "checkpoint": duong_ckpt,
            "nguong_diem": args.nguong_diem,
            "che_do": che_do,
            "ten_lop": ten_lop,
            "anh": ket_qua_tung_anh,
        }, f, ensure_ascii=False, indent=1)
    if loi:
        print(f"\n[!] {len(loi)} ảnh lỗi (bỏ qua, không tính vào bảng):")
        for l in loi:
            print("   - " + l)
    print(f"\nOverlay + JSON đã ghi vào:\n    {ra_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Huấn luyện Cascade R-CNN + ConvNeXt-Tiny trên FloodNet (Phase 3/4).

VÌ SAO KHÔNG GỌI THẲNG `tools/train.py` CỦA MMDET
------------------------------------------------
Phần "train" thật chỉ là `Runner.from_cfg(cfg).train()` — đúng một dòng, và
cũng là gần như toàn bộ thân của `tools/train.py` sau khi parse tham số. Thứ
đáng viết ra ở đây là phần TIỀN KIỂM chạy trước đó khoảng một phút.

Lý do rất cụ thể: mmdet có ít nhất ba kiểu hỏng KHÔNG báo lỗi, chỉ làm kết quả
sai âm thầm (liệt kê ở đầu `src/floodcount/data/kiem_tra.py`). Một suất train
24 epoch trên Colab là 6–8 giờ; phát hiện sai ở epoch 20 nghĩa là mất trắng
suất đó, mà nguyên nhân thì nằm ở dữ liệu chứ không ở code. Tiền kiểm ở đây
đối chiếu tên lớp, đếm lại số box mmdet THỰC SỰ nhận được, và in ra các tham
số then chốt — tất cả trước khi động tới GPU.

VÌ SAO PHẢI ĐỌC CONFIG BẰNG `Config.fromfile`
---------------------------------------------
Config đồ án kế thừa config ConvNeXt của mmdet, và config đó kế thừa tiếp ba
tầng `_base_` nữa. Nhìn vào tệp .py của đồ án sẽ KHÔNG thấy `scales`/`ratios`
của RPN, không thấy `pad_mask`, không thấy `load_from`... Phải qua
`Config.fromfile` mới ra giá trị THẬT SAU KHI KẾ THỪA — và đó mới là thứ runner
dùng. Đây cũng là lý do script này import `mmengine` chứ không tự parse.

Chạy:
    # GATE 3 — đo thời gian 1 epoch + VRAM đỉnh, KHÔNG ghi checkpoint
    python scripts/train.py --config configs/mmdet/cascade_convnext_t_floodnet.py --dry-run

    # Sanity check học vẹt 20 ảnh (config con, ghi vào work_dir riêng)
    python scripts/train.py --config configs/mmdet/overfit20.py

    # Train thật; tự resume nếu work_dir đã có checkpoint
    python scripts/train.py --config configs/mmdet/cascade_convnext_t_floodnet.py
"""

import argparse
import copy
import os
import pathlib
import shutil
import sys
import tempfile
import time

GOC_REPO = pathlib.Path(__file__).resolve().parents[1]

# PHẢI đứng TRƯỚC `Config.fromfile`. mmengine's `Config.fromfile` tự chạy
# `custom_imports` của config (đã đọc source 0.10.7: tham số
# `import_custom_modules=True`, thân hàm gọi `import_modules_from_strings`),
# mà config đồ án khai `floodcount.models.transforms` — chỉ import được khi
# `src/` đã có trong sys.path. Đặt sau thì thông báo lỗi là một ImportError kèm
# danh sách sys.path dài cả chục dòng, nhìn vào không biết phải sửa gì.
sys.path.insert(0, str(GOC_REPO / "src"))

from floodcount.data.kiem_tra import (doc_classes_tu_data_yaml,  # noqa: E402
                                      doc_json, kiem_bang_mmdet,
                                      kiem_box_bi_bo, kiem_cau_truc,
                                      kiem_khop_ten_lop)

GB = 1024 ** 3


def in_utf8():
    """Ép stdout/stderr sang utf-8.

    Console Windows mặc định là cp1252, in tiếng Việt có dấu sẽ nổ
    UnicodeEncodeError giữa bảng báo cáo. Colab thì không cần, nhưng gọi ở đây
    để cùng một script chạy được cả hai nơi.
    """
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


# ===========================================================================
# Đọc config
# ===========================================================================

def doc_yaml(duong_dan):
    """Đọc configs/data.yaml. YAML là nguồn sự thật cho đường dẫn và tên lớp."""
    import yaml
    with open(duong_dan, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cac_dataset(cfg):
    """{vai trò: dict `dataset`} cho train/val/test — để soi đường dẫn ảnh."""
    ra = {}
    for vai in ("train", "val", "test"):
        ds = (cfg.get(f"{vai}_dataloader") or {}).get("dataset") or {}
        if ds:
            ra[vai] = ds
    return ra


def thu_muc_anh(ds, goc_mac_dinh):
    """Thư mục chứa ảnh của một dataset, ghép sẵn với `data_root` của nó.

    `data_prefix` của mmdet nhận cả dict (thường dùng) lẫn một chuỗi trần, nên
    phải hiểu cả hai — bỏ sót một kiểu thì phép kiểm bên dưới lặng lẽ không chạy.
    """
    prefix = ds.get("data_prefix")
    if isinstance(prefix, dict):
        prefix = prefix.get("img")
    if not prefix:
        return None
    goc = str(ds.get("data_root") or goc_mac_dinh or "")
    return prefix if os.path.isabs(str(prefix)) else os.path.join(goc, str(prefix))


def kiem_anh_co_that(ds, duong_ann, goc_mac_dinh):
    """(danh sách ảnh thiếu, tổng số ảnh) của một dataset.

    Vì sao phải tự kiểm: `CocoDataset.parse_data_info` của mmdet 3.3.0 chỉ ghép
    `osp.join(self.data_prefix['img'], img_info['file_name'])` rồi trả về — đã
    đọc source, KHÔNG có `check_file_exist` nào trên đường dẫn ảnh. Sai thư mục
    ảnh vì thế không lộ ra lúc dựng dataset, cũng không lộ ra lúc bắt đầu train:
    nó nổ ở lượt validate đầu tiên, tức là sau khi đã train xong một epoch. Đây
    đúng là lỗi đã gặp ở `configs/mmdet/overfit20.py` — xem docs/NOTES.md §3.6.
    """
    anh = thu_muc_anh(ds, goc_mac_dinh) or ""
    coco = doc_json(duong_ann)
    ds_anh = coco.get("images", [])
    thieu = [im.get("file_name") for im in ds_anh
             if not os.path.exists(os.path.join(anh, str(im.get("file_name"))))]
    return thieu, len(ds_anh)


def cac_ann_can_co(cfg):
    """Mọi tệp annotation mà config NÀY thực sự dùng -> {vai trò: đường dẫn}.

    Lấy từ chính config chứ không đoán ba tên cố định: `configs/mmdet/overfit20.py`
    trỏ cả train lẫn val vào `instances_overfit20.json`, nên nếu cứng nhắc đòi
    `instances_train.json` thì chạy sanity check sẽ bị chặn oan.

    `val_evaluator`/`test_evaluator` được liệt kê riêng vì CocoMetric tự mở
    annotation của nó — cùng một tệp với dataloader ở config chính, nhưng ở
    config con thì có thể khác.
    """
    ra = {}
    data_root = cfg.get("data_root", "")
    for vai in ("train", "val", "test"):
        dl = cfg.get(f"{vai}_dataloader") or {}
        ds = dl.get("dataset") or {}
        ann = ds.get("ann_file")
        if ann:
            goc = ds.get("data_root", data_root)
            ra[vai] = ann if os.path.isabs(ann) else os.path.join(goc, ann)
    for vai in ("val", "test"):
        ev = cfg.get(f"{vai}_evaluator") or {}
        if ev.get("ann_file"):
            ra[f"{vai}_evaluator"] = ev["ann_file"]
    return ra


# ===========================================================================
# Tiền kiểm
# ===========================================================================

def kiem_tien_de(cfg, duong_config, processed_dir, kiem_cham=False):
    """Đối chiếu config với dữ liệu thật. Trả về danh sách lỗi (rỗng = chạy được).

    Đây là chỗ duy nhất trong đồ án chạm vào cả ba nguồn cùng lúc: config train,
    file COCO do Phase 2 sinh ra, và chính đối tượng dataset mà mmdet dựng lên.
    """
    loi = []

    # --- 1. data_root phải trỏ đúng nơi Phase 2 đã dựng dataset -------------
    print("\n[1] Đường dẫn dataset")
    data_root = cfg.get("data_root")
    print(f"    data_root (config train) : {data_root}")
    print(f"    processed_dir (data.yaml): {processed_dir}")
    if not processed_dir:
        print("    [!] configs/data.yaml không có `paths.processed_dir` — bỏ qua "
              "phép đối chiếu này.")
    elif os.path.normpath(str(data_root)) != os.path.normpath(str(processed_dir)):
        loi.append(
            f"`data_root` trong config train ({data_root}) KHÁC "
            f"`paths.processed_dir` trong configs/data.yaml ({processed_dir}).\n"
            f"      Hai đường dẫn này phải trỏ cùng một chỗ: Phase 2 ghi dataset "
            f"theo data.yaml, còn runner đọc theo config train. Lệch nhau thì "
            f"runner đọc một thư mục không tồn tại (hoặc tệ hơn: một bản dataset "
            f"cũ còn sót lại) mà không có gì báo là đang đọc nhầm chỗ.")
    else:
        print("    -> khớp.")

    if data_root and not os.path.isdir(str(data_root)):
        loi.append(f"thư mục `data_root` không tồn tại trên máy này: {data_root}\n"
                   f"      Trên Colab: chạy notebook 03 phần giải nén dataset "
                   f"trước, hoặc kiểm tra Drive đã mount chưa.")

    # --- 2. Các tệp annotation phải có mặt ---------------------------------
    print("\n[2] Tệp annotation")
    ann_theo_vai = cac_ann_can_co(cfg)
    for vai, duong in ann_theo_vai.items():
        co = os.path.exists(duong)
        print(f"    [{'OK  ' if co else 'THIẾU'}] {vai:<12} {duong}")
        if not co:
            loi.append(f"không thấy annotation cho vai trò `{vai}`: {duong}")

    # --- 2b. Ảnh ghi trong annotation phải có thật trên đĩa ----------------
    # mmdet KHÔNG kiểm việc này (xem docstring `kiem_anh_co_that`), nên phải tự
    # kiểm. Đây là bản tổng quát của lỗi đã gặp ở `configs/mmdet/overfit20.py`:
    # `val_dataloader` thừa hưởng nhầm `data_prefix=images/val/` trong khi ảnh
    # nằm ở `images/train/` (docs/NOTES.md §3.6). Chạy hết 2.343 + 450 + 448
    # đường dẫn chỉ tốn vài giây trên đĩa cục bộ của Colab.
    print("\n[2b] Ảnh của từng split: file ghi trong annotation có thật không")
    for vai, ds in cac_dataset(cfg).items():
        ann = ds.get("ann_file")
        if not ann:
            continue
        duong_ann = (str(ann) if os.path.isabs(str(ann))
                     else os.path.join(str(ds.get("data_root") or data_root or ""),
                                       str(ann)))
        if not os.path.exists(duong_ann):
            continue                    # đã báo ở mục [2]
        anh = thu_muc_anh(ds, data_root)
        if not anh:
            print(f"    [!] {vai:<6} dataset không khai `data_prefix.img` — "
                  f"KHÔNG kiểm được ảnh.")
            loi.append(f"[{vai}] dataset không khai `data_prefix.img`, nên không "
                       f"có cách nào biết ảnh nằm ở đâu. mmdet cũng cần khoá này "
                       f"để dựng đường dẫn ảnh.")
            continue
        try:
            thieu, tong = kiem_anh_co_that(ds, duong_ann, data_root)
        except (OSError, ValueError) as e:
            loi.append(f"[{vai}] không kiểm được ảnh: {e}")
            continue
        print(f"    [{'OK  ' if not thieu else 'THIẾU'}] {vai:<6} "
              f"{tong - len(thieu):,}/{tong:,} ảnh có thật trong {anh}")
        if thieu:
            vi_du = "; ".join(str(t) for t in thieu[:3])
            loi.append(
                f"[{vai}] {len(thieu):,}/{tong:,} ảnh ghi trong "
                f"{os.path.basename(duong_ann)} KHÔNG có thật trong {anh}.\n"
                f"      Ví dụ: {vi_du}\n"
                f"      mmdet không kiểm tra việc này lúc dựng dataset, nên lỗi "
                f"chỉ nổ ở lượt validate đầu tiên — sau khi đã train xong một "
                f"epoch.\n"
                f"      Kiểm lại `data_prefix.img` của dataset này: nó phải trỏ "
                f"tới thư mục chứa ĐÚNG tên file ghi trong annotation (Phase 2 "
                f"ghi tên phẳng có tiền tố split, ví dụ `train_10168.jpg`).")

    # --- 3. Tên lớp và thứ tự category ------------------------------------
    # Đây là phép kiểm quan trọng nhất. Hai kiểu hỏng đều im lặng: (a) tên trong
    # `metainfo` không có trong JSON -> mmdet bỏ hết box của lớp đó; (b) thứ tự
    # lệch -> bảng kết quả in ngược tên ngập/không ngập.
    print("\n[3] Tên lớp: `metainfo.classes` vs `categories` trong file JSON")
    ten_lop = tuple(cfg.get("metainfo", {}).get("classes", ()))
    print(f"    metainfo.classes = {ten_lop}")
    if not ten_lop:
        loi.append("config không có `metainfo.classes` — CocoDataset không dựng "
                   "được, và cũng không có gì để đối chiếu tên lớp.")
    else:
        # Cùng một tệp JSON có thể xuất hiện ở hai vai trò (overfit20 dùng chung
        # cho train và val) — chỉ kiểm một lần cho đỡ rối.
        da_kiem = set()
        for vai, duong in ann_theo_vai.items():
            if duong in da_kiem or not os.path.exists(duong):
                continue
            da_kiem.add(duong)
            try:
                coco = doc_json(duong)
            except (OSError, ValueError) as e:
                loi.append(f"không đọc được {duong}: {e}")
                continue
            print(f"    --- {vai}: {os.path.basename(duong)} "
                  f"({len(coco['images']):,} ảnh / "
                  f"{len(coco['annotations']):,} box)")
            for e in kiem_khop_ten_lop(coco, ten_lop):
                loi.append(f"[{vai}] {e}")
            for e in kiem_cau_truc(coco):
                loi.append(f"[{vai}] {e}")

            # Box mà `_parse_ann_info` sẽ bỏ. Bình thường phải bằng 0; khác 0
            # nghĩa là dữ liệu Phase 2 có vấn đề, không phải config.
            ds_bo = kiem_box_bi_bo(coco)
            if ds_bo:
                vi_du = "; ".join(f"ann {i} ({t}): {l}" for i, t, l in ds_bo[:3])
                loi.append(
                    f"[{vai}] {len(ds_bo)} box sẽ bị mmdet BỎ IM LẶNG "
                    f"(không có cảnh báo nào lúc chạy). Ví dụ: {vi_du}\n"
                    f"      Đây là lỗi ở khâu dựng COCO (Phase 2), sửa ở đó chứ "
                    f"không sửa config.")
            else:
                print("        không có box nào bị mmdet bỏ.")

    # --- 4. Số lớp ở CẢ BA tầng head ---------------------------------------
    # Cascade có ba bbox_head riêng biệt. Quên đổi `num_classes` ở một trong ba
    # là lỗi rất hay gặp: không có exception nào, chỉ có mAP thấp khó hiểu.
    print("\n[4] `num_classes` ở ba tầng Cascade")
    cac_head = cfg.get("model", {}).get("roi_head", {}).get("bbox_head")
    if isinstance(cac_head, dict):
        cac_head = [cac_head]
    if not cac_head:
        loi.append("config không có `model.roi_head.bbox_head` — kiểm lại khoá "
                   "`_delete_` ở roi_head, có thể nó đã xoá nhầm cả bbox_head.")
    else:
        for i, head in enumerate(cac_head):
            nc = head.get("num_classes")
            khop = nc == len(ten_lop)
            print(f"    [{'OK  ' if khop else 'LỆCH'}] tầng {i}: "
                  f"num_classes = {nc} (cần {len(ten_lop)})")
            if not khop:
                loi.append(
                    f"bbox_head tầng {i} có `num_classes = {nc}` nhưng config "
                    f"khai {len(ten_lop)} lớp. Phải sửa CẢ BA tầng cho bằng "
                    f"nhau — mmdet không báo lỗi, chỉ ra mAP thấp.")

    # --- 5. Đếm lại số instance bằng chính mmdet ---------------------------
    # Suy luận tay qua bốn điều kiện lồng nhau của `_parse_ann_info` là chỗ dễ
    # sai nhất. Hàm dưới không suy luận: nó dựng đúng đối tượng CocoDataset mà
    # runner sẽ dùng rồi đếm.
    print("\n[5] Dựng dataset bằng chính config này rồi đếm lại số box")
    try:
        for vai in ("train", "val"):
            if vai not in ann_theo_vai:
                continue
            kq = kiem_bang_mmdet(duong_config, vai) if kiem_cham else \
                kiem_bang_mmdet(duong_config, vai, so_anh_kiem=20)
            print(f"    --- {vai}: dataset {kq['so_anh_dataset']:,} ảnh "
                  f"(JSON {kq['so_anh_json']:,}) | "
                  f"instance {kq['so_instance']:,} (JSON {kq['so_ann_json']:,})")
            if kq["so_anh_dataset"] != kq["so_anh_json"]:
                loi.append(
                    f"[{vai}] mmdet dựng được {kq['so_anh_dataset']:,} ảnh nhưng "
                    f"JSON có {kq['so_anh_json']:,} ảnh. Chênh lệch này thường là "
                    f"`filter_cfg.filter_empty_gt` — đồ án cố ý để False, xem "
                    f"chú thích trong config.")
            if kq["so_instance"] != kq["so_ann_json"]:
                thieu = kq["so_ann_json"] - kq["so_instance"]
                vi_du = "; ".join(f"{t} (-{n})" for t, n in kq["ds_bi_bo"][:3])
                loi.append(
                    f"[{vai}] mmdet nhận {kq['so_instance']:,} box nhưng JSON có "
                    f"{kq['so_ann_json']:,} -> THIẾU {thieu:,} box, model sẽ học "
                    f"thiếu. Ví dụ ảnh bị hụt: {vi_du or '(không rõ)'}")
            if kq["ds_lech"]:
                loi.append(f"[{vai}] {len(kq['ds_lech'])} ảnh có box trong JSON "
                           f"nhưng mmdet nhận 0 box: {kq['ds_lech'][:5]}")
    except Exception as e:                     # noqa: BLE001
        # Bắt rộng là có chủ ý: đây là bước dựng dataset thật, lỗi có thể đến từ
        # mmdet/mmcv/thiếu ảnh. In ra rồi đi tiếp để các phép kiểm sau vẫn chạy,
        # thay vì để một traceback che mất toàn bộ bảng tiền kiểm.
        loi.append(f"không dựng được dataset bằng mmdet: {type(e).__name__}: {e}")

    return loi


def in_thong_so(cfg, so_epoch, resume, work_dir):
    """In ra những tham số mà nếu sai thì rất khó lần ra từ log train."""
    print("\n[6] Tham số huấn luyện (giá trị SAU KHI kế thừa `_base_`)")

    # `load_from` — trọng số khởi tạo. Config gốc của mmdet KHÔNG đặt khoá này;
    # backbone ConvNeXt nhận trọng số ImageNet qua `init_cfg` của chính nó. Nói
    # ra vì hai chuyện này rất dễ nhầm khi đọc log.
    load_from = cfg.get("load_from")
    print(f"    load_from        : {load_from}")
    init_cfg = cfg.get("model", {}).get("backbone", {}).get("init_cfg")
    if init_cfg:
        print(f"    backbone.init_cfg: {init_cfg.get('type')} <- "
              f"{init_cfg.get('checkpoint', '')[:80]}...")
    print(f"    resume           : {resume}  (work_dir: {work_dir})")

    dl = cfg.get("train_dataloader") or {}
    ds = dl.get("dataset") or {}
    filt = (ds.get("filter_cfg") or {}).get("filter_empty_gt")
    print(f"    filter_empty_gt  : {filt}")
    if filt is not False:
        print("        [!] Đồ án cố ý để False: khoảng một nửa FloodNet không có "
              "nhà nào, bỏ các ảnh đó đi thì model học 'ảnh nào cũng có nhà' — "
              "sai hẳn với bài toán ĐẾM.")

    # `keep_ratio` của các phép resize trong pipeline train. Bỏ sót là ảnh bị
    # bóp méo mà không có cảnh báo nào (xem chú thích dài trong config).
    for i, tf in enumerate(ds.get("pipeline", [])):
        if "esize" in tf.get("type", ""):
            print(f"    pipeline[{i}] {tf['type']:<20} "
                  f"keep_ratio={tf.get('keep_ratio')}")
            if tf.get("keep_ratio") is not True:
                print("        [!] keep_ratio KHÔNG phải True -> ảnh bị bóp về "
                      "đúng khung, méo tỉ lệ, mà không có gì báo lỗi.")

    ow = cfg.get("optim_wrapper") or {}
    acc = ow.get("accumulative_counts", 1)
    bs = dl.get("batch_size", 1)
    print(f"    batch_size       : {bs}")
    print(f"    accumulative     : {acc}  -> batch hiệu dụng {bs * acc}")
    print(f"    optimizer        : {ow.get('optimizer')}")
    print(f"    amp              : {ow.get('type')}")

    print(f"    max_epochs       : {so_epoch}")
    for sch in cfg.get("param_scheduler", []):
        if sch.get("type") == "MultiStepLR":
            print(f"    giảm LR tại epoch: {sch.get('milestones')} "
                  f"(gamma {sch.get('gamma')})")
        elif sch.get("type") == "LinearLR":
            print(f"    warmup           : {sch.get('end')} vòng lặp "
                  f"(by_epoch={sch.get('by_epoch')})")

    ck = (cfg.get("default_hooks") or {}).get("checkpoint") or {}
    print(f"    checkpoint       : giữ {ck.get('max_keep_ckpts')} bản, "
          f"save_best={ck.get('save_best')!r}, "
          f"save_optimizer={ck.get('save_optimizer')}")


def dat_lai_so_epoch(cfg, so_epoch_moi):
    """Ghi đè số epoch VÀ chỉnh lịch giảm LR theo cùng tỉ lệ.

    Đổi mỗi `max_epochs` mà để nguyên `milestones` là cái bẫy: chạy 8 epoch với
    milestones [16, 22] thì LR không bao giờ giảm, model dừng ở giữa lịch và mAP
    thấp hơn hẳn mức đáng ra phải đạt — nhìn bề ngoài y hệt "cần train thêm".

    CHỈ ĐƯỢC GỌI MỘT LẦN. Hàm sửa thẳng `milestones` tại chỗ và lấy
    `train_cfg.max_epochs` hiện tại làm mốc tỉ lệ; gọi lần thứ hai thì tỉ lệ bị
    nhân chồng lên chính nó và ra một lịch sai mà không có gì báo.

    Returns:
        list: các cảnh báo cần in ra.
    """
    canh_bao = []
    goc = int(cfg.train_cfg.max_epochs)

    for sch in cfg.get("param_scheduler", []):
        if sch.get("type") == "MultiStepLR":
            # Giữ đúng TỈ LỆ của lịch gốc, rồi kẹp vào [1, so_epoch - 1]. Cận
            # trên là so_epoch - 1 vì mốc giảm ở đúng epoch cuối thì vô nghĩa:
            # LR giảm xong là hết lịch.
            # Chạy 1 epoch thì mọi mốc đều vô nghĩa (chỉ số epoch chạy là 0), nên
            # trả về rỗng và để phần cảnh báo bên dưới nói ra.
            if so_epoch_moi < 2:
                moi = []
            else:
                moi = sorted({max(1, min(so_epoch_moi - 1,
                                         round(m / goc * so_epoch_moi)))
                              for m in sch.get("milestones", [])})
            if sch.get("milestones") and not moi:
                canh_bao.append(
                    f"không đặt được mốc giảm LR nào cho {so_epoch_moi} epoch "
                    f"(lịch gốc {sch['milestones']}/{goc} quá thưa) — LR sẽ giữ "
                    f"nguyên suốt lần chạy.")
            print(f"    mốc giảm LR: {sch.get('milestones')} (trên {goc} epoch) "
                  f"-> {moi} (trên {so_epoch_moi} epoch)")
            sch["milestones"] = moi
            sch["end"] = so_epoch_moi

    cfg.train_cfg.max_epochs = so_epoch_moi

    # Warmup tính bằng VÒNG LẶP, không bằng epoch. Nếu tổng số vòng của cả lần
    # chạy nhỏ hơn `end` thì model chưa ra khỏi warmup đã hết lịch, LR vẫn ở
    # mức 1/1000 và kết quả thấp một cách khó hiểu.
    for sch in cfg.get("param_scheduler", []):
        if sch.get("type") == "LinearLR" and sch.get("by_epoch") is False:
            canh_bao.append(
                f"warmup đang đặt {sch.get('end')} VÒNG LẶP. Hãy kiểm con số "
                f"'vòng lặp mỗi epoch' ở phần đo bên dưới: nếu "
                f"{so_epoch_moi} epoch có ít hơn {sch.get('end')} vòng thì LR "
                f"chưa kịp tăng đã hết lịch.")
    return canh_bao


# ===========================================================================
# Chạy thử — đo thời gian và VRAM
# ===========================================================================

def chay_thu(cfg, so_vong, so_vong_khoi_dong=2):
    """Chạy thật vài vòng lặp để đo, KHÔNG ghi gì lên Drive.

    Gọi đúng hàm mà vòng lặp train gọi. Đã đọc source mmengine 0.10.7
    (`EpochBasedTrainLoop.run_iter`):

        outputs = self.runner.model.train_step(data_batch,
                                               optim_wrapper=self.runner.optim_wrapper)

    nên gọi thẳng `runner.model.train_step(...)` là mô phỏng trung thực: có
    forward, có backward, có bước cập nhật của optimizer (kể cả AMP và tích luỹ
    gradient). Không phải ước lượng bằng cách đo riêng phần forward.

    `Runner.train()` còn gọi `_maybe_compile('train_step')`, nhưng hàm đó thoát
    ngay khi config không có khoá `compile` (đã đọc source) — config đồ án không
    có, nên bỏ qua được.
    """
    import torch
    from mmengine.runner import Runner

    cfg = copy.deepcopy(cfg)
    cfg.resume = False
    # Trỏ work_dir vào thư mục tạm. Ghi log/checkpoint lên Drive qua FUSE vừa làm
    # phép đo thời gian sai lệch, vừa để lại rác trong thư mục kết quả thật.
    tmp = tempfile.mkdtemp(prefix="floodcount_dryrun_")
    cfg.work_dir = tmp

    try:
        print("\n[7] Chạy thử để đo thời gian và VRAM")
        print(f"    work_dir tạm: {tmp}  (xoá sau khi đo xong)")
        print("    Đang dựng model... lần đầu sẽ tải trọng số ImageNet của "
              "ConvNeXt-Tiny (~110 MB) nên bước này chậm.")
        t0 = time.perf_counter()
        runner = Runner.from_cfg(cfg)
        print(f"    dựng xong sau {time.perf_counter() - t0:.1f}s")

        runner.model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        dl = runner.train_dataloader
        so_vong_that = min(so_vong, len(dl))
        print(f"    {len(dl):,} vòng lặp/epoch — chạy thử {so_vong_that} vòng "
              f"(bỏ {so_vong_khoi_dong} vòng đầu khi tính trung bình vì vòng đầu "
              f"còn khởi động worker và autotune của cuDNN)\n")
        print(f"    {'vòng':>5}  {'ms':>9}  {'loss':>12}")
        print(f"    {'-' * 5}  {'-' * 9}  {'-' * 12}")

        ds_ms, ds_loss, loi_loss = [], [], []
        for i, data in enumerate(dl):
            if i >= so_vong_that:
                break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            log_vars = runner.model.train_step(
                data, optim_wrapper=runner.optim_wrapper)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000.0

            loss = float(log_vars.get("loss", float("nan")))
            if i >= so_vong_khoi_dong:
                ds_ms.append(ms)
                ds_loss.append(loss)
            # Loss NaN là kiểu hỏng im lặng đáng sợ nhất: train vẫn chạy hết
            # epoch, checkpoint vẫn ghi, chỉ có trọng số là hỏng.
            if loss != loss:
                loi_loss.append(i)
            print(f"    {i:>5}  {ms:>9.1f}  {loss:>12.4f}")

        if not ds_ms:
            print("\n    [!] Chạy thử quá ít vòng để tính trung bình — tăng "
                  "--so-vong.")
            return

        ms_tb = sum(ds_ms) / len(ds_ms)
        epoch_s = ms_tb / 1000.0 * len(dl)

        print(f"\n    {'-' * 40}")
        print(f"    thời gian/vòng (TB {len(ds_ms)} vòng) : {ms_tb:,.1f} ms")
        print(f"    số vòng lặp mỗi epoch               : {len(dl):,}")
        print(f"    => 1 epoch                         : {epoch_s / 60:,.1f} phút "
              f"({epoch_s:,.0f} giây)")

        if loi_loss:
            print(f"\n    [!!!] LOSS = NaN ở vòng {loi_loss[:5]} — DỪNG, đừng "
                  f"train. NaN ngay từ những vòng đầu gần như luôn là lỗi dữ "
                  f"liệu (toạ độ box vô lý, area = 0) hoặc LR quá lớn, không "
                  f"phải 'cần thêm epoch'.")

        if torch.cuda.is_available():
            dinh = torch.cuda.max_memory_allocated()
            dat_truoc = torch.cuda.max_memory_reserved()
            tong = torch.cuda.get_device_properties(0).total_memory
            print(f"    VRAM đỉnh (đã cấp phát)             : {dinh / GB:,.2f} GB")
            print(f"    VRAM đỉnh (đã giành chỗ)            : "
                  f"{dat_truoc / GB:,.2f} GB")
            print(f"    tổng VRAM của GPU                   : {tong / GB:,.2f} GB")
            if dat_truoc > 0.92 * tong:
                print("    [!] Đã dùng trên 92% VRAM. Còn dư ít thì rất dễ OOM "
                      "khi gặp batch toàn ảnh 1536px — nên giảm batch_size hoặc "
                      "siết `scales_train` lại.")
        else:
            print("    [!!!] KHÔNG có GPU. Con số thời gian trên là của CPU nên "
                  "vô nghĩa với Colab — chỉ dùng để biết code chạy được.")

        print(f"\n    Vòng lặp train thật sẽ chạy {cfg.train_cfg.max_epochs} epoch "
              f"-> khoảng {epoch_s * cfg.train_cfg.max_epochs / 3600:,.1f} giờ "
              f"(chưa tính thời gian validate mỗi epoch).")
    finally:
        # Bộ đếm VRAM là trạng thái toàn cục của CUDA — không dọn thì lần đo sau
        # sẽ thấy đỉnh của cả lần đo trước.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"    (đã xoá {tmp})")


# ===========================================================================
# main
# ===========================================================================

def main():
    in_utf8()

    ap = argparse.ArgumentParser(
        description="Tiền kiểm rồi huấn luyện Cascade R-CNN + ConvNeXt-T "
                    "trên FloodNet")
    ap.add_argument("--config",
                    default="configs/mmdet/cascade_convnext_t_floodnet.py")
    ap.add_argument("--data-yaml", default="configs/data.yaml",
                    help="Nguồn sự thật cho đường dẫn dataset và tên lớp")
    ap.add_argument("--work-dir", default=None,
                    help="Ghi đè work_dir (mặc định: lấy từ config)")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Ghi đè số epoch; lịch giảm LR được chỉnh theo cùng tỉ lệ")
    ap.add_argument("--no-resume", action="store_true",
                    help="Bỏ qua checkpoint cũ trong work_dir, train lại từ đầu")
    ap.add_argument("--dry-run", action="store_true",
                    help="Chỉ đo thời gian/vòng và VRAM đỉnh rồi thoát")
    ap.add_argument("--so-vong", type=int, default=20,
                    help="Số vòng lặp chạy thử ở chế độ --dry-run (mặc định 20)")
    ap.add_argument("--kiem-cham", action="store_true",
                    help="Đối chiếu TỪNG ảnh khi đếm box (chậm, dùng trước lần "
                         "train thật)")
    args = ap.parse_args()

    print("=" * 72)
    print("TRAIN — CASCADE R-CNN + CONVNEXT-TINY TRÊN FLOODNET")
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
        print("    Trên Colab: chạy ô cài đặt của notebook 00 trước.")
        print("    Máy này (CPU) không cài mmdet — script này chỉ chạy được trên "
              "Colab.")
        raise SystemExit(1)

    print(f"mmdet {mmdet.__version__} | mmcv {mmcv.__version__} | "
          f"mmengine {mmengine_ver} | torch {torch.__version__}")
    print(f"CUDA khả dụng: {torch.cuda.is_available()}"
          + (f" | {torch.cuda.get_device_name(0)}"
             if torch.cuda.is_available() else ""))

    # `Config.fromfile` cũng là chỗ mmengine chạy `custom_imports` của config
    # (tham số `import_custom_modules=True`). Nghĩa là tới đây backbone ConvNeXt
    # (mmpretrain) và transform `TangSangNhe` của đồ án đã được đăng ký. Kiểm lại
    # cho chắc, vì nếu thiếu thì thông báo lỗi lúc dựng model sẽ là "chưa đăng ký
    # TangSangNhe" — đúng nhưng trỏ vào chỗ khác với nguyên nhân thật.
    cfg = Config.fromfile(args.config)
    for ten in (cfg.get("custom_imports") or {}).get("imports", []):
        print(f"custom_imports: {ten} -> "
              f"{'đã nạp' if ten in sys.modules else 'CHƯA NẠP (?!)'}")

    if args.work_dir:
        print(f"work_dir (ghi đè) : {args.work_dir}")
        cfg.work_dir = args.work_dir
    else:
        print(f"work_dir          : {cfg.work_dir}")

    # Số epoch SAU KHI kế thừa, đọc trước khi bị ghi đè — cần làm mốc để chỉnh
    # lịch giảm LR theo đúng tỉ lệ.
    so_epoch = int(cfg.train_cfg.max_epochs)
    if args.max_epochs and args.max_epochs != so_epoch:
        print(f"\nGhi đè số epoch: {so_epoch} -> {args.max_epochs}")
        for cb in dat_lai_so_epoch(cfg, args.max_epochs):
            print(f"    [!] {cb}")
        so_epoch = args.max_epochs

    resume = not args.no_resume
    cfg.resume = resume

    data_yaml = doc_yaml(args.data_yaml)
    processed_dir = (data_yaml.get("paths") or {}).get("processed_dir")
    # Đọc tên lớp từ data.yaml để đối chiếu chéo với `metainfo` — nguồn sự thật
    # của tên lớp là file đó, cùng file mà Phase 1 và Phase 2 đã dùng.
    try:
        ten_lop_yaml = doc_classes_tu_data_yaml(args.data_yaml)
        ten_lop_cfg = list((cfg.get("metainfo") or {}).get("classes", []))
        if ten_lop_cfg and ten_lop_cfg != ten_lop_yaml:
            print(f"\n[!] Tên lớp trong config train {ten_lop_cfg} KHÁC tên lớp "
                  f"trong {args.data_yaml} {ten_lop_yaml} — phải thống nhất, nếu "
                  f"không thì bảng kết quả cuối cùng in sai tên.")
    except ValueError as e:
        print(f"\n[!] {e}")

    loi = kiem_tien_de(cfg, args.config, processed_dir, kiem_cham=args.kiem_cham)
    in_thong_so(cfg, so_epoch, resume, cfg.work_dir)

    if loi:
        print("\n" + "=" * 72)
        print(f"*** {len(loi)} VẤN ĐỀ — CHƯA NÊN TRAIN ***")
        print("=" * 72)
        for i, e in enumerate(loi, 1):
            print(f"\n{i}. {e}")
        raise SystemExit(1)

    print("\n" + "=" * 72)
    print("TIỀN KIỂM ĐẠT HẾT.")
    print("=" * 72)

    if args.dry_run:
        chay_thu(cfg, args.so_vong)
        return

    from mmengine.runner import Runner
    print(f"\nBắt đầu train. Log và checkpoint ghi vào:\n    {cfg.work_dir}")
    if resume:
        print("Nếu work_dir đã có checkpoint, runner sẽ TỰ ĐỘNG resume từ bản mới "
              "nhất.\nMuốn train lại từ đầu: thêm --no-resume.")
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()

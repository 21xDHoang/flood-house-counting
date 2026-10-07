# -*- coding: utf-8 -*-
"""Test cho scripts/train.py — CPU, KHÔNG cần mmdet/mmengine/torch.

Script train chỉ import mmengine/mmdet BÊN TRONG hàm, không phải ở đầu tệp. Đó
là ràng buộc thiết kế có chủ ý: nhờ nó mà phần logic thuần Python ở đây test
được ngay trên máy Windows không cài MMDetection, thay vì phải chờ tới lúc chạy
trên Colab mới biết sai.

Sáu thứ test này phải khoá:

  1. `cac_ann_can_co` trả về ĐÚNG các tệp annotation mà config dùng. Sai chỗ này
     thì tiền kiểm đi kiểm nhầm tệp — tệ nhất là báo "ĐẠT HẾT" trong khi tệp
     thật của tập train chưa từng được mở.

  2. `dat_lai_so_epoch` chỉnh `milestones` theo đúng TỈ LỆ. Để nguyên milestones
     khi giảm số epoch là lỗi im lặng: LR không bao giờ giảm, mAP thấp hơn hẳn
     mức đáng ra phải đạt, và nhìn bề ngoài y hệt "cần train thêm epoch".

  3. Tên lớp ở `configs/data.yaml` và ở `metainfo.classes` của config mmdet phải
     TRÙNG NHAU. Lệch tên thì mmdet bỏ im lặng mọi box của lớp đó; lệch thứ tự
     thì bảng kết quả in ngược nhãn ngập/không ngập. Cả hai đều không báo lỗi.

  4. Mỗi dataloader phải khai TƯỜNG MINH `data_prefix` trỏ đúng thư mục ảnh của
     nó. Bỏ qua là mmengine gộp dict theo chiều sâu và để nó thừa hưởng từ
     config cha — mmdet không kiểm tra ảnh có tồn tại lúc dựng dataset, nên lỗi
     chỉ nổ ở bước validate, tức là sau khi đã train xong một epoch.

  5. `chay_thu` phải build optim_wrapper y như `Runner.train()` làm. mmengine
     chỉ build nó bên trong `train()`; vòng đo tự viết mà quên bước này thì
     `runner.optim_wrapper` vẫn là ConfigDict và `train_step` nổ ngay vòng đầu —
     lỗi thật đã gặp trên Colab ở GATE 3, sau khi đã dựng xong model.

  6. Config con (`overfit20.py`) phải tự viết đường dẫn annotation của nó, và cả
     ba đường dẫn train/val/val_evaluator phải trỏ về CÙNG một tệp khớp
     `data_root` của config cha. Cú pháp `{{_base_.xxx}}` bị CẤM trong configs/:
     viết trong ngoặc kép thì mmengine thay bằng một placeholder đã bọc sẵn ngoặc
     kép, giá trị chuỗi thành ra chứa cả ngoặc, phép tra cứu trượt và đường dẫn
     hoá thành tên rác — im lặng hoàn toàn (lỗi thật ở lần chạy [3.10],
     docs/NOTES.md §3.11).

Chạy:
    py tests/test_train.py
"""

import ast
import importlib.util
import json
import os
import pathlib
import posixpath
import re
import sys
import tempfile

GOC_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC_REPO / "src"))

from floodcount.data.kiem_tra import doc_classes_tu_config_mmdet  # noqa: E402


class CfgGia(dict):
    """dict truy cập được bằng thuộc tính — đủ giống `mmengine.Config`.

    `dat_lai_so_epoch` đọc `cfg.train_cfg.max_epochs`, gán lại
    `cfg.train_cfg.max_epochs`, và gọi `cfg.get("param_scheduler", [])`. Đó là
    toàn bộ giao diện nó cần, nên không phải kéo mmengine về chỉ để test.

    `__getattr__` chỉ được gọi khi tra thuộc tính thường thất bại, nên `cfg.get`
    vẫn là `dict.get` thật.
    """

    def __getattr__(self, ten):
        try:
            return self[ten]
        except KeyError:
            raise AttributeError(ten)

    def __setattr__(self, ten, gia_tri):
        self[ten] = gia_tri


def nap_train():
    """Nạp scripts/train.py như một module, không chạy `main()`.

    `scripts/` không phải package nên phải nạp bằng đường dẫn.
    """
    duong = GOC_REPO / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_duoi_test", duong)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    tmp = tempfile.mkdtemp(prefix="train_test_")
    loi = []

    def kiem(dieu_kien, mo_ta):
        trang_thai = "OK  " if dieu_kien else "FAIL"
        print(f"  [{trang_thai}] {mo_ta}")
        if not dieu_kien:
            loi.append(mo_ta)

    try:
        train = nap_train()

        print("=== 1. scripts/train.py KHONG keo mmdet/mmengine/torch vao ===")
        # Rang buoc thiet ke: neu khong giu duoc thi ca bo test nay khong chay
        # noi o may sach, va phan logic ben duoi chi con kiem duoc tren Colab.
        nang = [m for m in sys.modules
                if m.split(".")[0] in ("mmdet", "mmcv", "mmengine", "torch")]
        kiem(not nang, f"import train.py khong keo thu vien nang, thuc te {nang}")

        print("\n=== 1b. chay_thu phai build optim_wrapper truoc khi do ===")
        # Loi that da gap tren Colab (GATE 3, o [3.9]): mmengine chi build
        # optim_wrapper BEN TRONG `Runner.train()` —
        #     self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        # (`Runner.__init__` chi gan). Vong lap do tu viet khong di qua train(),
        # nen phai tu goi dung ham do; thieu la `train_step` nhan ConfigDict va
        # no ngay o vong dau bang
        #     AttributeError: 'ConfigDict' object has no attribute 'optim_context'
        # Test nay khoa lai: xoa dong build di la FAIL, khong the tai xuat.
        cay = ast.parse((GOC_REPO / "scripts" / "train.py")
                        .read_text(encoding="utf-8"))
        than = next((n for n in ast.walk(cay)
                     if isinstance(n, ast.FunctionDef) and n.name == "chay_thu"),
                    None)
        kiem(than is not None, "tim thay ham chay_thu trong scripts/train.py")
        if than is not None:
            goi_build = [n for n in ast.walk(than)
                         if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute)
                         and n.func.attr == "build_optim_wrapper"]
            kiem(bool(goi_build),
                 "chay_thu goi runner.build_optim_wrapper(...)")
            gan_lai = [n for n in ast.walk(than)
                       if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Attribute)
                               and t.attr == "optim_wrapper"
                               for t in n.targets)]
            kiem(bool(gan_lai),
                 "... va gan ket qua lai vao runner.optim_wrapper")
            # train_step phai nhan optim_wrapper qua keyword — bo di la quay
            # ve dung loi cu.
            ts = [n for n in ast.walk(than)
                  if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "train_step"]
            kiem(bool(ts) and all(any(k.arg == "optim_wrapper"
                                      for k in n.keywords) for n in ts),
                 "moi loi goi train_step deu truyen keyword optim_wrapper")

        print("\n=== 2. cac_ann_can_co: tim dung tep annotation ===")
        # Duong dan tuong doi -> giai theo data_root cua CHINH dataset do
        cfg = CfgGia(
            data_root="/content/floodnet_coco",
            train_dataloader=CfgGia(dataset=CfgGia(
                ann_file="annotations/instances_train.json")),
            val_dataloader=CfgGia(dataset=CfgGia(
                ann_file="annotations/instances_val.json")),
            test_dataloader=CfgGia(dataset=CfgGia(
                ann_file="annotations/instances_test.json")),
            val_evaluator=CfgGia(
                ann_file="/content/floodnet_coco/annotations/instances_val.json"),
            test_evaluator=CfgGia(
                ann_file="/content/floodnet_coco/annotations/instances_test.json"))
        ra = train.cac_ann_can_co(cfg)
        kiem(set(ra) == {"train", "val", "test", "val_evaluator", "test_evaluator"},
             f"du 5 vai tro, thuc te {sorted(ra)}")
        kiem(ra["train"] == os.path.join("/content/floodnet_coco",
                                         "annotations/instances_train.json")
             or ra["train"] == "/content/floodnet_coco/annotations/instances_train.json",
             f"ann_file tuong doi duoc giai theo data_root, thuc te {ra['train']}")
        # KHONG dung `os.path.isabs` o day: duong dan cua Colab la kieu POSIX
        # ("/content/...") va ham do tra ve False tren Windows. Thu can khoa la
        # gia tri DI NGUYEN, khong bi ghep them data_root — kiem dung tinh chat
        # do thi chay dung tren ca hai he.
        kiem(ra["val_evaluator"]
             == "/content/floodnet_coco/annotations/instances_val.json",
             f"ann_file tuyet doi duoc giu nguyen (khong ghep data_root), "
             f"thuc te {ra['val_evaluator']}")
        # pytest khong co o day, nhung Windows tra ve '\' con Linux tra ve '/', nen
        # phai chuan hoa truoc khi so.
        kiem(os.path.normpath(ra["val"]) == os.path.normpath(ra["val_evaluator"]),
             "val_dataloader va val_evaluator tro cung mot tep")

        # data_root rieng cua dataset phai duoc uu tien hon data_root cua config.
        cfg2 = CfgGia(
            data_root="/content/khong-dung",
            train_dataloader=CfgGia(dataset=CfgGia(
                ann_file="a/train.json", data_root="/content/dung-cai-nay")))
        ra2 = train.cac_ann_can_co(cfg2)
        kiem(os.path.normpath(ra2["train"])
             == os.path.normpath("/content/dung-cai-nay/a/train.json"),
             f"data_root cua dataset duoc uu tien, thuc te {ra2['train']}")

        # Kieu configs/mmdet/overfit20.py: train va val cung mot tep.
        cfg3 = CfgGia(
            data_root="/content/floodnet_coco",
            train_dataloader=CfgGia(dataset=CfgGia(
                ann_file="/content/floodnet_coco/annotations/instances_overfit20.json")),
            val_dataloader=CfgGia(dataset=CfgGia(
                ann_file="/content/floodnet_coco/annotations/instances_overfit20.json")),
            val_evaluator=CfgGia(
                ann_file="/content/floodnet_coco/annotations/instances_overfit20.json"))
        ra3 = train.cac_ann_can_co(cfg3)
        kiem(set(ra3.values()) == {"/content/floodnet_coco/annotations/"
                                  "instances_overfit20.json"},
             f"config overfit20: moi vai tro tro cung mot tep, thuc te {ra3}")

        print("\n=== 3. dat_lai_so_epoch: chinh milestones theo TI LE ===")
        # Gia tri that cua config do an: 24 epoch, moc [16, 22].
        def chay(so_epoch_moi, goc=24, moc=(16, 22), warmup=1000):
            cfg_gia = CfgGia(
                train_cfg=CfgGia(max_epochs=goc),
                param_scheduler=[
                    {"type": "LinearLR", "start_factor": 0.001,
                     "by_epoch": False, "begin": 0, "end": warmup},
                    {"type": "MultiStepLR", "begin": 0, "end": goc,
                     "by_epoch": True, "milestones": list(moc), "gamma": 0.1}])
            canh_bao = train.dat_lai_so_epoch(cfg_gia, so_epoch_moi)
            return cfg_gia, canh_bao

        # 16/24*12 = 8 ; 22/24*12 = 11
        cfg12, _ = chay(12)
        kiem(cfg12.train_cfg.max_epochs == 12, "max_epochs duoc ghi de")
        kiem(cfg12.param_scheduler[1]["milestones"] == [8, 11],
             f"12 epoch -> moc [8, 11] (16/24*12 va 22/24*12), "
             f"thuc te {cfg12.param_scheduler[1]['milestones']}")
        kiem(cfg12.param_scheduler[1]["end"] == 12,
             f"`end` cua MultiStepLR phai theo so epoch moi, "
             f"thuc te {cfg12.param_scheduler[1]['end']}")

        # 16/24*8 = 5.33 -> 5 ; 22/24*8 = 7.33 -> 7
        cfg8, _ = chay(8)
        kiem(cfg8.param_scheduler[1]["milestones"] == [5, 7],
             f"8 epoch -> moc [5, 7], thuc te {cfg8.param_scheduler[1]['milestones']}")

        # 22/24*6 = 5.5 -> round(5.5) = 6 (lam tron ve so chan cua Python), roi
        # bi kep xuong so_epoch-1 = 5: moc giam o dung epoch cuoi la vo nghia.
        cfg6, _ = chay(6)
        kiem(cfg6.param_scheduler[1]["milestones"] == [4, 5],
             f"6 epoch -> moc [4, 5] (moc cuoi bi kep xuong 5), "
             f"thuc te {cfg6.param_scheduler[1]['milestones']}")
        kiem(all(m < 6 for m in cfg6.param_scheduler[1]["milestones"]),
             "khong moc nao nam ngoai pham vi epoch")

        # 22/24*2 = 1.83 -> 2, kep xuong 1 ; 16/24*2 = 1.33 -> 1. Ca hai thanh 1.
        cfg2, _ = chay(2)
        kiem(cfg2.param_scheduler[1]["milestones"] == [1],
             f"2 epoch -> moc [1] (hai moc gop lai), "
             f"thuc te {cfg2.param_scheduler[1]['milestones']}")

        # Chay 1 epoch: moi moc deu vo nghia -> phai CANH BAO, khong im lang.
        cfg1, cb1 = chay(1)
        kiem(cfg1.param_scheduler[1]["milestones"] == [],
             f"1 epoch -> khong con moc nao, "
             f"thuc te {cfg1.param_scheduler[1]['milestones']}")
        kiem(any("giữ nguyên suốt lần chạy" in c for c in cb1),
             f"1 epoch -> canh bao LR khong bao gio giam, thuc te {cb1}")

        # Giu nguyen so epoch thi moc phai y nguyen.
        cfg24, _ = chay(24)
        kiem(cfg24.param_scheduler[1]["milestones"] == [16, 22],
             f"24 epoch (khong doi) -> moc giu nguyen [16, 22], "
             f"thuc te {cfg24.param_scheduler[1]['milestones']}")

        # Warmup tinh bang VONG LAP: luon phai canh bao de nguoi doc tu doi chieu
        # voi so vong moi epoch in o phan do --dry-run.
        _, cb_warm = chay(12)
        kiem(any("VÒNG LẶP" in c for c in cb_warm),
             "luon canh bao ve warmup tinh bang vong lap")
        kiem(any("1000" in c for c in cb_warm),
             f"canh bao neu dung con so warmup that (1000), thuc te {cb_warm}")

        print("\n=== 4. Ten lop: data.yaml vs config mmdet ===")
        # Khong co mmengine thi `doc_classes_tu_config_mmdet` roi ve regex — van
        # du de bat loi go sai ten trong config. Tren Colab no doc bang mmengine
        # (gia tri sau khi ke thua), con o day doc chuoi viet tren giay.
        ten_yaml = [t for _, t in sorted(
            _doc_yaml(os.path.join(str(GOC_REPO), "configs", "data.yaml"))
            ["preprocess"]["classes"].items())]
        ten_cfg, nguon = doc_classes_tu_config_mmdet(
            os.path.join(str(GOC_REPO), "configs", "mmdet",
                         "cascade_convnext_t_floodnet.py"))
        print(f"      data.yaml    : {ten_yaml}")
        print(f"      config mmdet : {list(ten_cfg or [])}  (nguon: {nguon})")
        kiem(ten_cfg is not None, f"doc duoc metainfo.classes tu config ({nguon})")
        kiem(list(ten_cfg or []) == ten_yaml,
             "ten lop o config mmdet TRUNG voi data.yaml (ca thu tu lan ten)")

        # Ten lop phai dung thu tu ma Phase 2 ghi ra file COCO: category id tang
        # dan. Dao thu tu o day la dao nhan ngap/khong ngap cua ca do an.
        kiem(ten_yaml == ["flooded_building", "non_flooded_building"],
             f"thu tu ten lop dung (id 1 = ngap, id 2 = khong ngap), "
             f"thuc te {ten_yaml}")

        print("\n=== 5. Config overfit20 tro dung tep do tao_overfit20 sinh ra ===")
        # Hai script phai thong nhat ten tep, neu khong thi GATE 3 chay vao mot
        # tep khong ton tai (hoac tep cu con sot lai cua lan chay truoc).
        duong_overfit = os.path.join(str(GOC_REPO), "configs", "mmdet", "overfit20.py")
        duong_chinh = os.path.join(str(GOC_REPO), "configs", "mmdet",
                                   "cascade_convnext_t_floodnet.py")
        ma_cfg = open(duong_overfit, encoding="utf-8").read()
        kiem("instances_overfit20.json" in ma_cfg,
             "config overfit20 tro toi instances_overfit20.json")
        ma_tao = open(os.path.join(str(GOC_REPO), "src", "floodcount", "data",
                                   "overfit.py"), encoding="utf-8").read()
        kiem("instances_overfit20.json" in ma_tao,
             "tao_overfit20.py ghi ra dung cai ten do")

        # Cu phap {{_base_.xxx}} BI CAM trong moi config. Lan chay [3.10] ngay
        # 01/10/2026 tren Colab: config viet ann_file='{{_base_.ann_overfit}}',
        # mmengine 0.10.7 thay phan trong ngoac bang mot placeholder da BOC SAN
        # ngoac kep (`_pre_substitute_base_vars`: re.sub(regexp, f'"{randstr}"',
        # ...)), nen gia tri chuoi thanh '"_ann_overfit_d1b839"' — tra
        # `v in base_var_dict` truot (khoa khong co ngoac), the la duong dan rac
        # di thang vao tien kiem, KHONG loi nao bao. Dang TRAN (khong ngoac) thi
        # chay duoc theo source, nhung khi do config khong con la chuoi cho chinh
        # test nay doc bang ast. Chi tiet: docs/NOTES.md §3.11.
        vi_pham = []
        for goc, _, ten_tep in os.walk(os.path.join(str(GOC_REPO), "configs")):
            for ten in ten_tep:
                if ten.endswith(".py"):
                    duong = os.path.join(goc, ten)
                    with open(duong, encoding="utf-8") as f:
                        # Bo dong comment TRUOC khi tim: chinh chu thich dau
                        # overfit20.py nhac lai cu phap nay de giai thich vi sao
                        # no bi cam — do la tai lieu, khong phai code.
                        if any("{{_base_." in d
                               for d in f.read().splitlines()
                               if not d.lstrip().startswith("#")):
                            vi_pham.append(os.path.relpath(duong, str(GOC_REPO)))
        kiem(not vi_pham,
             "khong config nao dung cu phap {{_base_.}} (hong im lang voi gia "
             f"tri chuoi — NOTES §3.11), thuc te {vi_pham}")

        # Ba duong dan annotation phai tro ve CUNG mot tep, va tep do phai khop
        # `data_root` cua config cha. Dataloader ghi TUONG DOI (mmdet ghep voi
        # data_root thua huong tu config cha), con val_evaluator ghi TUYET DOI
        # (CocoMetric mo thang tep bang pycocotools, khong biet data_root).
        data_root_cha = _doc_bien(duong_chinh, "data_root")
        ds_train = _doc_dataset(duong_overfit, "train_dataloader")
        ds_val = _doc_dataset(duong_overfit, "val_dataloader")
        ev_val = _doc_bien(duong_overfit, "val_evaluator")
        ann_train = ds_train.get("ann_file") if isinstance(ds_train, dict) else None
        ann_val = ds_val.get("ann_file") if isinstance(ds_val, dict) else None
        ann_ev = ev_val.get("ann_file") if isinstance(ev_val, dict) else None
        # `isinstance(..., str)` TRUOC khi so sanh: hai `_BieuThuc` luon bang
        # nhau (__eq__ co y nhu vay), so sanh thang la tu cho minh DAT.
        doc_duoc = all(isinstance(x, str)
                       for x in (data_root_cha, ann_train, ann_val, ann_ev))
        kiem(doc_duoc,
             f"doc duoc data_root cua config cha va ba ann_file cua config "
             f"overfit, thuc te {data_root_cha!r}, {ann_train!r}, "
             f"{ann_val!r}, {ann_ev!r}")
        if doc_duoc:
            # posixpath chu khong os.path: Colab chay Linux, con may nay Windows
            # — os.path.join se tra ve '\' va so sanh sai.
            dich = posixpath.join(data_root_cha, ann_train)
            kiem(posixpath.join(data_root_cha, ann_val) == dich
                 and ann_ev == dich,
                 f"train/val/val_evaluator tro CUNG mot tep {dich!r} (tinh tu "
                 f"data_root cua config cha), thuc te {ann_train!r}, "
                 f"{ann_val!r}, {ann_ev!r}")

        print("\n=== 6. data_prefix: anh cua tap overfit nam o images/train ===")
        # Loi that, tim ra bang cach doc source mmdet 3.3.0:
        # `CocoDataset.parse_data_info` dung duong dan anh bang
        #     osp.join(self.data_prefix['img'], img_info['file_name'])
        # ma `file_name` do Phase 2 ghi ra la ten PHANG co tien to split
        # ("train_10168.jpg"), khong kem thu muc. Nghia la thu muc anh hoan toan
        # do `data_prefix['img']` quyet dinh, va mmdet KHONG kiem tra anh co ton
        # tai luc dung dataset (khong co `check_file_exist` nao tren duong dan anh).
        #
        # configs/mmdet/overfit20.py chi khai lai `ann_file`; mmengine gop dict
        # theo chieu sau nen `data_prefix` THUA HUONG tu config cha. Voi train thi
        # tinh co dung (images/train/), con voi val thi thua huong 'images/val/'
        # — trong khi 20 anh overfit lay tu split train. Hau qua: tien kiem van
        # bao DAT HET, train chay qua epoch 1 roi moi chet o buoc validate.
        #
        # Luat khoa lai: dataset nao tro tới annotation overfit thi phai khai
        # TUONG MINH data_prefix, khong duoc de thua huong.
        for vai in ("train", "val", "test"):
            ds = _doc_dataset(duong_overfit, f"{vai}_dataloader")
            if ds is None:
                continue        # config này không khai biến đó (test_dataloader)
            # Không đọc được thì KHÔNG được coi là đạt — bỏ qua im lặng ở đúng
            # chỗ này là cách bản test đầu tiên "đạt" mà không kiểm gì cả.
            kiem(isinstance(ds, dict) and isinstance(ds.get("ann_file"), str)
                 and "overfit" in ds["ann_file"],
                 f"doc duoc overfit20 {vai}_dataloader tu config, thuc te {ds!r}")
            if not isinstance(ds, dict):
                continue
            ann = ds.get("ann_file")
            if not isinstance(ann, str) or "overfit" not in ann:
                continue
            kiem(ds.get("data_prefix") == {"img": "images/train/"},
                 f"overfit20 {vai}_dataloader khai tuong minh "
                 f"data_prefix={{'img': 'images/train/'}} (khong thua huong "
                 f"images/val/ cua config cha), thuc te {ds.get('data_prefix')!r}")

        # Cung phep kiem cho config chinh: moi split phai tro dung thu muc anh cua
        # no. Neu ai do chep kieu khai bao cua overfit20 vao day thi test do ngay.
        for vai in ("train", "val", "test"):
            ds = _doc_dataset(duong_chinh, f"{vai}_dataloader")
            kiem(isinstance(ds, dict),
                 f"doc duoc config chinh {vai}_dataloader, thuc te {ds!r}")
            if not isinstance(ds, dict):
                continue
            kiem(ds.get("data_prefix") == {"img": f"images/{vai}/"},
                 f"config chinh: {vai}_dataloader tro images/{vai}/, "
                 f"thuc te {ds.get('data_prefix')!r}")

        print("\n=== 7. data_root (config mmdet) khop processed_dir (data.yaml) ===")
        # Hai tep nay phai tro cung mot cho: Phase 2 ghi dataset theo data.yaml,
        # con runner doc theo config train. Lech nhau thi runner doc mot thu muc
        # khong ton tai — hoac te hon: mot ban dataset CU con sot lai tu lan chay
        # truoc — ma khong co gi bao la dang doc nham cho.
        #
        # Doc bang regex chu khong bang mmengine: tren may CPU khong co mmengine,
        # ma day la phep kiem phai chay duoc ngay luc commit, khong doi tới Colab.
        # Bo dong comment truoc khi tim, neu khong se bat nham vi du trong chu
        # thich (chinh comment ngay tren `data_root` co nhac ten no).
        yaml_cfg = _doc_yaml(os.path.join(str(GOC_REPO), "configs", "data.yaml"))
        processed_dir = (yaml_cfg.get("paths") or {}).get("processed_dir")
        ma_mmdet = "\n".join(
            d for d in open(duong_chinh, encoding="utf-8").read().splitlines()
            if not d.lstrip().startswith("#"))
        # Chi bat phep GAN chuoi: `data_root = '/content/...'`. Cac dict dataset
        # dung `data_root=data_root` (khong co dau ngoac kep) nen khong khop —
        # dung nhu mong muon, vi do la tham chieu chu khong phai gia tri.
        m = re.search(r"^data_root\s*=\s*['\"]([^'\"]+)['\"]", ma_mmdet, re.M)
        data_root = m.group(1) if m else None
        print(f"      data.yaml    : paths.processed_dir = {processed_dir}")
        print(f"      config mmdet : data_root            = {data_root}")
        kiem(data_root is not None,
             "doc duoc `data_root = '<chuoi>'` tu config mmdet bang regex")
        # Chuan hoa truoc khi so: Windows tra ve '\\' con Linux tra ve '/'.
        kiem(processed_dir is not None and data_root is not None
             and os.path.normpath(data_root) == os.path.normpath(processed_dir),
             f"data_root TRUNG paths.processed_dir, "
             f"thuc te {data_root!r} vs {processed_dir!r}")

        print("\n=== 8. kiem_anh_co_that: bat anh nam sai thu muc ===")
        # mmdet khong kiem anh co ton tai (xem docstring kiem_anh_co_that), nen
        # day la phong tuyen duy nhat bat duoc loi `data_prefix` tro nham thu muc
        # — dung loi da gap o overfit20 (docs/NOTES.md §3.6). Dung dataset gia
        # nho de phep kiem chay duoc ngay tren may CPU.
        goc = os.path.join(tmp, "dataset_gia")
        os.makedirs(os.path.join(goc, "images", "train"))
        os.makedirs(os.path.join(goc, "annotations"))
        open(os.path.join(goc, "images", "train", "train_1.jpg"), "wb").close()
        duong_ann_gia = os.path.join(goc, "annotations", "instances_train.json")
        with open(duong_ann_gia, "w", encoding="utf-8") as f:
            json.dump({"images": [{"id": 1, "file_name": "train_1.jpg"},
                                  {"id": 2, "file_name": "train_2.jpg"}],
                       "annotations": [], "categories": []}, f)

        ds_dung = {"data_prefix": {"img": "images/train/"}}
        thieu, tong = train.kiem_anh_co_that(ds_dung, duong_ann_gia, goc)
        kiem(thieu == ["train_2.jpg"] and tong == 2,
             f"chi bao thieu dung anh khong co that, thuc te {thieu} / tong {tong}")

        # Dung ca loi da gap: anh o images/train/ nhung data_prefix tro images/val/.
        ds_sai = {"data_prefix": {"img": "images/val/"}}
        thieu_sai, tong_sai = train.kiem_anh_co_that(ds_sai, duong_ann_gia, goc)
        kiem(thieu_sai == ["train_1.jpg", "train_2.jpg"] and tong_sai == 2,
             f"data_prefix tro nham thu muc -> bao thieu CA HAI anh, "
             f"thuc te {thieu_sai}")

        # data_root cua chinh dataset phai duoc uu tien hon data_root cua config.
        ds_goc_rieng = {"data_prefix": {"img": "images/train/"},
                        "data_root": goc}
        thieu2, _ = train.kiem_anh_co_that(ds_goc_rieng, duong_ann_gia, "/khong/co")
        kiem(thieu2 == ["train_2.jpg"],
             f"data_root cua dataset duoc uu tien, thuc te {thieu2}")

        # `data_prefix` kieu chuoi tran va kieu tuyet doi — mmdet nhan ca hai.
        kiem(train.thu_muc_anh({"data_prefix": "images/train/"}, goc)
             == os.path.join(goc, "images/train/"),
             "data_prefix kieu chuoi tran van ghep duoc voi data_root")
        # Duong dan tuyet doi phai duoc giu NGUYEN, khong ghep them data_root.
        # Dung `os.path.abspath` chu KHONG viet cung "/tuyet/doi/": tren Windows
        # `os.path.isabs("/tuyet/doi/")` tra ve False (goc POSIX khong tinh la
        # tuyet doi o day), nen viet cung la test do o Windows trong khi Colab
        # — noi duy nhat dung den — lai dung. Cung cai bay da ghi o muc 2.
        duong_tuyet = os.path.abspath(os.path.join(tmp, "tuyet", "doi"))
        kiem(train.thu_muc_anh({"data_prefix": {"img": duong_tuyet}}, "/khong/co")
             == duong_tuyet,
             f"data_prefix tuyet doi duoc giu nguyen, "
             f"thuc te {train.thu_muc_anh({'data_prefix': {'img': duong_tuyet}}, '/khong/co')!r}")
        kiem(train.thu_muc_anh({}, goc) is None,
             "dataset khong khai data_prefix -> None (de phan goi bao loi, "
             "khong im lang bo qua)")

    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if loi:
        print(f"*** {len(loi)} MUC KHONG DAT ***")
        for m in loi:
            print("   - " + m.encode("ascii", "backslashreplace").decode())
        raise SystemExit(1)
    print("*** TAT CA PASS ***")


def _doc_yaml(duong_dan):
    import yaml
    with open(duong_dan, encoding="utf-8") as f:
        return yaml.safe_load(f)


class _BieuThuc:
    """Đánh dấu một giá trị trong config KHÔNG viết thẳng ra được (biến, gọi hàm).

    Cần phân biệt với `None`: `None` là "tệp config ghi đúng chữ None", còn cái
    này là "ở đây có một biểu thức, test đọc chữ không biết giá trị". Gộp hai
    thứ làm một thì test sẽ báo ĐẠT cho những chỗ nó thật ra không đọc được.
    """

    __slots__ = ("mo_ta",)

    def __init__(self, mo_ta):
        self.mo_ta = mo_ta

    def __repr__(self):
        return f"<biểu thức: {self.mo_ta}>"

    def __eq__(self, khac):
        return isinstance(khac, _BieuThuc)

    def __hash__(self):
        return hash("<biểu thức>")


def _gia_tri(node):
    """Đổi nút AST thành giá trị Python, nếu là literal đơn giản.

    Hiểu chuỗi/số/None, `{...}`, `[...]`, `(...)`, và `dict(...)` — vì config
    của mmengine viết `dict(...)` chứ không viết `{...}`. Bản đầu của hàm này chỉ
    hiểu `{...}`, nên nó trả `None` cho MỌI dataloader và phép kiểm bên dưới
    "đạt" một cách vô nghĩa vì không đọc được gì. Mọi thứ khác (tên biến, toán
    tử) trả về `_BieuThuc` để phép kiểm biết là mình KHÔNG đọc được chỗ đó.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {_gia_tri(k): _gia_tri(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_gia_tri(x) for x in node.elts]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "dict"):
        ra = {}
        for tu_khoa in node.keywords:
            # `**gi_do` (tu_khoa.arg is None) không đọc được tên khoá.
            ten = tu_khoa.arg if tu_khoa.arg is not None else _BieuThuc("**")
            ra[ten] = _gia_tri(tu_khoa.value)
        return ra
    try:
        return _BieuThuc(ast.unparse(node))
    except Exception:                            # noqa: BLE001
        return _BieuThuc("?")


def _doc_bien(duong_config, ten_bien):
    """Đọc literal gán cho một biến cấp module trong tệp config, KHÔNG cần mmengine.

    Trả về giá trị literal (chuỗi/số/dict...), `_BieuThuc` nếu chỗ đó là biểu
    thức đọc chữ không ra, và `None` nếu tệp không gán biến ấy. Dùng để đọc
    `data_root` của config cha và `val_evaluator` của config con — hai thứ nằm
    ngoài `dataset=` mà `_doc_dataset` với tới.
    """
    with open(duong_config, encoding="utf-8") as f:
        cay = ast.parse(f.read())
    for nut in cay.body:
        if isinstance(nut, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == ten_bien
                for t in nut.targets):
            return _gia_tri(nut.value)
    return None


def _doc_dataset(duong_config, ten_bien):
    """Đọc `ten_bien = dict(..., dataset=dict(...))` từ tệp config, KHÔNG cần mmengine.

    Trả về phần `dataset` VIẾT THẲNG trong tệp đó (chưa gộp `_base_`), hoặc
    `None` nếu tệp KHÔNG khai biến ấy. Tệp có khai nhưng đọc không ra `dataset`
    thì trả về `_BieuThuc` — cố ý tách khỏi `None`, để phép kiểm không thể "đạt"
    chỉ vì nó không đọc được gì (bản test đầu tiên đã mắc đúng lỗi đó).

    Đọc phần viết thẳng chứ không phần đã gộp là có chủ ý: đúng cái cần khoá ở
    đây là "config CON có tự khai hay không", vì thứ bị thừa hưởng im lặng từ
    config cha chính là thứ test này sinh ra để bắt.
    """
    with open(duong_config, encoding="utf-8") as f:
        cay = ast.parse(f.read())
    for nut in cay.body:
        if not isinstance(nut, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == ten_bien
                   for t in nut.targets):
            continue
        gia = _gia_tri(nut.value)
        if isinstance(gia, dict) and isinstance(gia.get("dataset"), dict):
            return gia["dataset"]
        return _BieuThuc(f"{ten_bien} = {ast.unparse(nut.value)[:60]}")
    return None


if __name__ == "__main__":
    main()

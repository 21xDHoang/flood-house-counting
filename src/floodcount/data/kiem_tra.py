# -*- coding: utf-8 -*-
"""Kiểm tra dataset COCO trước khi train — chốt chặn cuối của Phase 3.

VÌ SAO CẦN MỘT BƯỚC KIỂM RIÊNG

Phase 2 đã kiểm cấu trúc file COCO (id không trùng, box nằm trong ảnh, area
dương...). Nhưng những lỗi nguy hiểm nhất KHÔNG nằm trong cấu trúc file — chúng
nằm ở chỗ GIAO NHAU giữa file JSON và config train, và chúng không hề báo lỗi:

  1. Tên lớp trong `metainfo.classes` không có trong JSON. `CocoDataset` gọi
     `self.coco.get_cat_ids(cat_names=self.metainfo['classes'])`; tên nào không
     khớp thì bị loại khỏi `cat_ids`, rồi `_parse_ann_info` gặp
     `if ann['category_id'] not in self.cat_ids: continue` và **bỏ im lặng toàn
     bộ box của lớp đó**. Model học trên nửa dữ liệu, loss vẫn giảm đều.

  2. Tên khớp nhưng THỨ TỰ khác. pycocotools `getCatIds` trả về id theo đúng thứ
     tự category trong file JSON (đã lọc theo tên), rồi
     `cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}`. Tức là
     nhãn 0/1 gán theo thứ tự JSON, còn `metainfo['classes'][i]` lại là tên mà
     mmdet dùng để IN RA. Hai thứ lệch nhau thì bảng AP theo lớp và bảng đếm cuối
     cùng in TÊN NGƯỢC NHAU — sai đúng ở kết quả quan trọng nhất của đồ án.

  3. Box nằm ngoài ảnh / area <= 0 / cạnh < 1 bị `_parse_ann_info` bỏ im lặng.

Cả ba đều thuộc loại "chạy vẫn xong, số vẫn ra, chỉ là sai". Bước kiểm này biến
chúng thành lỗi ồn ào ngay trên máy CPU, trước khi tốn một suất train trên Colab.

HAI MỨC KIỂM

  * Mức 1 — không cần thư viện gì ngoài numpy/pyyaml. Chạy được trên máy CPU,
    có test riêng ở `tests/test_kiem_tra.py`. Soi JSON bằng tay.
  * Mức 2 — cần mmdet/mmengine/mmcv (tức là trên Colab). Dựng THẬT dataset bằng
    chính config của đồ án rồi đếm số instance mà mmdet nhìn thấy, đối chiếu với
    số annotation trong JSON. Đây là phép kiểm duy nhất chứng minh được rằng
    không có box nào bị bỏ im lặng, vì nó đi qua đúng đoạn code sẽ chạy lúc train.

Kiểm bằng cách "đọc lại code mmdet rồi suy luận" là không đủ: chỉ cần một điều
kiện nữa trong `_parse_ann_info` mà mình đọc sót là kết luận sai. Mức 2 không
suy luận, nó chạy thật.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np


# ===========================================================================
# Đọc file & kiểm cấu trúc
# ===========================================================================

def doc_json(duong_dan):
    """Đọc file COCO JSON. Lỗi đọc file thì ném ra kèm đường dẫn, không im lặng."""
    if not os.path.exists(duong_dan):
        raise FileNotFoundError(
            f"Không thấy file annotation:\n    {duong_dan}\n"
            f"    Kiểm tra dataset đã giải nén đúng chỗ chưa.")
    with open(duong_dan, encoding="utf-8") as f:
        return json.load(f)


def thu_tu_nhan_mmdet(coco, ten_lop_metainfo):
    """Thứ tự tên lớp mà mmdet sẽ gán cho nhãn 0, 1, 2...

    Mô phỏng đúng hai bước của mmdet:

        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

    và `getCatIds` của pycocotools là::

        cats = [cat for cat in dataset['categories'] if cat['name'] in catNms]
        return [cat['id'] for cat in cats]

    Tức là: LỌC theo tên (thứ tự do `metainfo` quyết định tập nào được giữ) nhưng
    GIỮ NGUYÊN THỨ TỰ của file JSON. Hai chuyện đó khác nhau, và đó chính là
    nguồn của lỗi số 2 ở đầu file.

    Returns:
        tuple: (thu_tu_ten, ten_bi_thieu) — `thu_tu_ten` là list tên lớp theo
        đúng thứ tự nhãn 0..N-1, `ten_bi_thieu` là những tên khai trong
        `metainfo` mà JSON không có.
    """
    ds_ten = [c["name"] for c in coco["categories"]]
    giu_lai = [t for t in ds_ten if t in set(ten_lop_metainfo)]
    ten_bi_thieu = [t for t in ten_lop_metainfo if t not in set(ds_ten)]
    return giu_lai, ten_bi_thieu


def kiem_khop_ten_lop(coco, ten_lop_metainfo):
    """Lỗi về quan hệ giữa `metainfo.classes` và `categories` của file JSON."""
    loi = []
    thu_tu, bi_thieu = thu_tu_nhan_mmdet(coco, ten_lop_metainfo)

    if bi_thieu:
        # Đếm luôn số box sẽ mất, để con số thiệt hại cụ thể chứ không trừu tượng.
        id_bi_thieu = {c["id"] for c in coco["categories"] if c["name"] in set(bi_thieu)}
        so_box = sum(1 for a in coco["annotations"] if a["category_id"] in id_bi_thieu)
        loi.append(
            f"`metainfo.classes` khai tên {bi_thieu!r} nhưng file JSON không có "
            f"category nào tên như vậy. mmdet sẽ BỎ IM LẶNG toàn bộ "
            f"{so_box:,} box thuộc (các) category đó.")

    # Điều kiện ĐÚNG không phải "hai danh sách bằng nhau" mà là: với MỌI nhãn mà
    # mmdet thực sự dùng (0 .. len(cat_ids)-1), `metainfo[i]` phải là tên của
    # đúng category đó. Chỉ so hai danh sách với nhau sẽ báo lỗi oan khi
    # `metainfo` khai thừa một tên ở CUỐI — tên thừa đó không bao giờ được dùng
    # để in ra gì cả.
    lech = [(i, ten_lop_metainfo[i], thu_tu[i])
            for i in range(len(thu_tu)) if thu_tu[i] != ten_lop_metainfo[i]]
    if lech:
        chi_tiet = "\n".join(
            f"        nhãn {i}: category trong JSON là {ten_json!r} nhưng "
            f"`metainfo` gọi nó là {ten_metainfo!r} -> mọi kết quả của lớp này "
            f"sẽ in ra dưới tên {ten_metainfo!r}"
            for i, ten_metainfo, ten_json in lech)
        loi.append(
            f"THỨ TỰ / TÊN LỚP LỆCH — đây là lỗi in ngược tên ở bảng kết quả, "
            f"không phải lỗi cấu trúc:\n{chi_tiet}\n"
            f"      Nhãn mmdet gán theo thứ tự category TRONG FILE JSON (đã lọc "
            f"theo tên), không theo thứ tự `metainfo.classes`:\n"
            f"        {dict(enumerate(thu_tu))}\n"
            f"      Sửa: đặt `metainfo.classes` ĐÚNG BẰNG thứ tự category trong "
            f"file JSON (hoặc sửa thứ tự category trong JSON cho khớp).")

    # Category trong JSON mà config không dùng — không phải lỗi, nhưng phải nói
    # ra: dữ liệu có thứ mình cố ý không dùng là chuyện nên biết, không nên đoán.
    thua = [c["name"] for c in coco["categories"] if c["name"] not in set(ten_lop_metainfo)]
    if thua:
        id_thua = {c["id"] for c in coco["categories"] if c["name"] in set(thua)}
        so_box = sum(1 for a in coco["annotations"] if a["category_id"] in id_thua)
        print(f"      (ghi chú) JSON có category không dùng tới: {thua} — "
              f"{so_box:,} box sẽ nằm ngoài mọi phép đếm.")
    return loi


def kiem_cau_truc(coco):
    """Lỗi cấu trúc: id trùng, annotation trỏ tới ảnh không tồn tại."""
    loi = []
    ds_anh = {im["id"]: im for im in coco["images"]}
    if len(ds_anh) != len(coco["images"]):
        loi.append(f"có image id trùng: {len(coco['images'])} dòng nhưng chỉ "
                   f"{len(ds_anh)} id duy nhất")

    ds_ann_id = set()
    for a in coco["annotations"]:
        if a["id"] in ds_ann_id:
            loi.append(f"annotation id trùng: {a['id']}")
        ds_ann_id.add(a["id"])
        if a["image_id"] not in ds_anh:
            loi.append(f"annotation {a['id']} trỏ tới image_id không tồn tại "
                       f"{a['image_id']}")
    return loi


# ===========================================================================
# Box mà mmdet sẽ BỎ IM LẶNG
# ===========================================================================

def ly_do_mmdet_bo(ann, im):
    """Mô phỏng điều kiện bỏ box trong `CocoDataset._parse_ann_info`.

    Chép lại ĐÚNG ba điều kiện của mmdet 3.3.0::

        x1, y1, w, h = ann['bbox']
        inter_w = max(0, min(x1 + w, img_w) - max(x1, 0))
        inter_h = max(0, min(y1 + h, img_h) - max(y1, 0))
        if inter_w * inter_h == 0:            continue
        if ann['area'] <= 0 or w < 1 or h < 1: continue
        if ann['category_id'] not in self.cat_ids: continue

    Chép lại thay vì gọi mmdet vì hàm này phải chạy được trên máy CPU không cài
    mmdet. Điều kiện thứ ba (category) do `kiem_khop_ten_lop` lo.

    Returns:
        str | None: lý do bị bỏ, hoặc None nếu box được giữ.
    """
    x1, y1, w, h = ann["bbox"]
    inter_w = max(0.0, min(x1 + w, im["width"]) - max(x1, 0.0))
    inter_h = max(0.0, min(y1 + h, im["height"]) - max(y1, 0.0))
    if inter_w * inter_h == 0:
        return "không giao với ảnh (nằm ngoài hoặc dẹt về 0)"
    if ann["area"] <= 0 or w < 1 or h < 1:
        return "area <= 0 hoặc có cạnh < 1"
    return None


def kiem_box_bi_bo(coco):
    """Danh sách (ann_id, file_name, lý do) của box mmdet sẽ bỏ im lặng."""
    ds_anh = {im["id"]: im for im in coco["images"]}
    ds_bo = []
    for a in coco["annotations"]:
        im = ds_anh.get(a["image_id"])
        if im is None:
            continue                       # đã báo ở kiem_cau_truc
        ly_do = ly_do_mmdet_bo(a, im)
        if ly_do:
            ds_bo.append((a["id"], im["file_name"], ly_do))
    return ds_bo


def kiem_anh_ton_tai(coco, thu_muc_anh):
    """Danh sách file_name có trong JSON nhưng KHÔNG có trên đĩa."""
    thieu = []
    for im in coco["images"]:
        if not os.path.exists(os.path.join(thu_muc_anh, im["file_name"])):
            thieu.append(im["file_name"])
    return thieu


# ===========================================================================
# Thống kê
# ===========================================================================

def thong_ke(coco, ten_lop_theo_id):
    """Thống kê số ảnh / số box / ảnh rỗng / phân bố cỡ box."""
    dem = Counter()
    so_box_moi_anh = {}
    for a in coco["annotations"]:
        ten = ten_lop_theo_id.get(a["category_id"], f"id lạ {a['category_id']}")
        dem[ten] += 1
        so_box_moi_anh[a["image_id"]] = so_box_moi_anh.get(a["image_id"], 0) + 1

    so_anh_rong = sum(1 for im in coco["images"] if so_box_moi_anh.get(im["id"], 0) == 0)

    canh = np.array([[min(a["bbox"][2], a["bbox"][3])] for a in coco["annotations"]],
                    dtype=np.float64) if coco["annotations"] else np.zeros((0, 1))
    canh = canh.reshape(-1)
    kich_thuoc = {(im["width"], im["height"]) for im in coco["images"]}

    return {
        "so_anh": len(coco["images"]),
        "so_box": len(coco["annotations"]),
        "dem_theo_lop": dict(sorted(dem.items())),
        "so_anh_rong": so_anh_rong,
        "ti_le_anh_rong": so_anh_rong / max(1, len(coco["images"])),
        "canh_nho_nhat": float(canh.min()) if len(canh) else float("nan"),
        "canh_trung_vi": float(np.median(canh)) if len(canh) else float("nan"),
        "canh_lon_nhat": float(canh.max()) if len(canh) else float("nan"),
        "cac_co_anh": sorted(kich_thuoc),
    }


def bao_cao(split, tk, ten_lop_metainfo=""):
    """Đổi dict của `thong_ke` thành văn bản để in."""
    d = [f"  [{split}] {tk['so_anh']:,} ảnh | {tk['so_box']:,} box | "
         f"ảnh không có box nào: {tk['so_anh_rong']:,} "
         f"({tk['ti_le_anh_rong'] * 100:.1f}%)"]
    for ten, n in tk["dem_theo_lop"].items():
        d.append(f"      {ten:<24s} {n:>6,} box")
    d.append(f"      cỡ ảnh: {tk['cac_co_anh']}")
    if tk["so_box"]:
        d.append(f"      cạnh nhỏ của box: min {tk['canh_nho_nhat']:.0f}px | "
                 f"trung vị {tk['canh_trung_vi']:.0f}px | "
                 f"max {tk['canh_lon_nhat']:.0f}px")
    return "\n".join(d)


# ===========================================================================
# Đọc tên lớp từ config
# ===========================================================================

def doc_classes_tu_data_yaml(duong_yaml):
    """Đọc `preprocess.classes` -> list tên lớp theo thứ tự giá trị mask.

    Nguồn sự thật của TÊN lớp là `configs/data.yaml` — cùng file mà Phase 1 và
    Phase 2 đã dùng để dựng COCO. Dùng nó thay vì tự đoán tên từ file JSON.

    Raises:
        ValueError: thiếu khoá, kèm tên khoá và tên file. Để nguyên `KeyError`
        của dict là một traceback trỏ vào dòng code chứ không trỏ vào file cấu
        hình — trên Colab nhìn vào đó không biết phải sửa gì.
    """
    import yaml
    with open(duong_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "preprocess" not in cfg \
            or "classes" not in (cfg["preprocess"] or {}):
        raise ValueError(
            f"{duong_yaml} thiếu `preprocess.classes` — đây là nguồn tên lớp "
            f"mà Phase 1 và Phase 2 đã dùng để dựng COCO, không có nó thì không "
            f"đối chiếu được tên lớp nữa.")
    return [ten for _, ten in sorted(cfg["preprocess"]["classes"].items())]


def doc_classes_tu_config_mmdet(duong_config):
    """Đọc `metainfo.classes` từ config train.

    Ưu tiên mmengine: `Config.fromfile` phân giải `_base_` nên đọc được giá trị
    THẬT SAU KHI KẾ THỪA, không phải giá trị viết trên giấy. Đây là điều quan
    trọng — config đồ án thừa hưởng từ config mmdet, nên nhìn vào file để đoán
    là đoán sai.

    Returns:
        tuple: (danh_sach_ten, nguon) với `nguon` là "mmengine" hoặc "regex".
        Không đọc được thì trả (None, lý do).
    """
    try:
        from mmengine.config import Config
    except ImportError:
        # Máy CPU không có mmengine. Đọc thô bằng regex — CHỈ để cảnh báo sớm
        # trên máy, không phải nguồn sự thật; trên Colab luôn dùng mmengine.
        try:
            ma_nguon = open(duong_config, encoding="utf-8").read()
        except OSError as e:
            return None, f"không đọc được file: {e}"
        # Bỏ dòng comment để không bắt nhầm ví dụ nằm trong chú thích.
        ma_nguon = "\n".join(d for d in ma_nguon.splitlines()
                             if not d.lstrip().startswith("#"))
        m = re.search(r"classes\s*=\s*\(([^)]*)\)", ma_nguon)
        if not m:
            return None, "không tìm thấy `metainfo.classes` bằng regex"
        ten = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
        return tuple(ten), "regex"

    cfg = Config.fromfile(duong_config)
    metainfo = cfg.get("metainfo")
    if not metainfo or "classes" not in metainfo:
        return None, "config không có `metainfo.classes`"
    return tuple(metainfo["classes"]), "mmengine"


# ===========================================================================
# Mức 2 — dựng dataset THẬT bằng mmdet rồi đếm
# ===========================================================================

def kiem_bang_mmdet(duong_config, split, so_anh_kiem=0):
    """Dựng dataset bằng chính config đồ án, so số instance với số trong JSON.

    VÌ SAO PHẢI CHẠY THẬT
    `_parse_ann_info` có bốn điều kiện bỏ box. Ba điều kiện đầu đã được mô phỏng
    ở `ly_do_mmdet_bo`; điều kiện thứ tư (category) phụ thuộc `cat_ids` — mà
    `cat_ids` lại phụ thuộc `metainfo` và thứ tự category trong JSON. Suy luận
    tay qua bốn điều kiện lồng nhau là chỗ dễ sai nhất. Hàm này không suy luận:
    nó gọi đúng `CocoDataset` mà runner sẽ dùng, rồi đếm.

    Args:
        duong_config (str): Đường dẫn config train (file .py của mmengine).
        split (str): "train", "val", "test".
        so_anh_kiem (int): Số ảnh lấy ra đối chiếu chi tiết từng ảnh.
            0 = kiểm hết (chậm hơn nhưng đây là bước chạy một lần trước train).

    Returns:
        dict: `{"so_anh_dataset", "so_anh_json", "so_instance", "so_ann_json",
               "ds_lech", "ds_bi_bo"}`.
    """
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import DATASETS

    cfg = Config.fromfile(duong_config)

    # `custom_imports` ĐÃ được chạy ở dòng `Config.fromfile` ngay trên: mmengine
    # 0.10.7 có tham số `import_custom_modules=True` và thân hàm gọi
    # `import_modules_from_strings(**cfg_dict['custom_imports'])` (đã đọc source,
    # `mmengine/config/config.py`). `Runner` thì KHÔNG đụng tới khoá này.
    #
    # Gọi lại ở đây là thừa nhưng vô hại — `import_module` một gói đã nằm trong
    # `sys.modules` chỉ là phép tra từ điển. Giữ lại để hàm này vẫn đúng nếu về
    # sau config được nạp bằng đường khác (lazy_import, hoặc config dựng tay
    # không qua `fromfile`): thiếu nó thì pipeline có `TangSangNhe` (transform tự
    # viết của đồ án) không build được, và thông báo lỗi là "chưa đăng ký
    # TangSangNhe" — đúng nhưng trỏ vào chỗ khác với nguyên nhân thật.
    custom = cfg.get("custom_imports")
    if custom:
        import_modules_from_strings(**custom)

    split_cfg = cfg[f"{split}_dataloader"]["dataset"]
    ann_file = split_cfg["ann_file"]
    data_root = split_cfg.get("data_root", cfg.get("data_root", ""))
    data_prefix = split_cfg.get("data_prefix", {})
    duong_ann = ann_file if os.path.isabs(ann_file) else os.path.join(data_root, ann_file)

    coco = doc_json(duong_ann)

    # Dựng đúng đối tượng mà runner dùng. `test_mode=False` cho cả val/test vì ta
    # muốn thấy pipeline TRAIN (có augmentation) có làm mất box nào không.
    ds_cfg = {k: v for k, v in split_cfg.items() if k != "batch_sampler"}
    ds_cfg["test_mode"] = False
    if "data_prefix" not in ds_cfg:
        ds_cfg["data_prefix"] = data_prefix
    dataset = DATASETS.build(ds_cfg)

    dem_json = Counter(a["image_id"] for a in coco["annotations"])
    id_theo_ten = {im["file_name"]: im["id"] for im in coco["images"]}

    ds_lech, ds_bi_bo = [], []
    so_instance = 0
    so_anh_kiem = so_anh_kiem or len(dataset)
    for i in range(len(dataset)):
        data_info = dataset.get_data_info(i)
        so_mmdet = len(data_info["instances"])
        so_instance += so_mmdet
        if i >= so_anh_kiem:
            continue
        ten_file = os.path.basename(data_info["img_path"])
        img_id = data_info.get("img_id", id_theo_ten.get(ten_file))
        so_json = dem_json.get(img_id, 0)
        if so_mmdet != so_json:
            ds_bi_bo.append((ten_file, so_json - so_mmdet))
        if so_mmdet == 0 and so_json > 0:
            ds_lech.append(ten_file)

    return {
        "so_anh_dataset": len(dataset),
        "so_anh_json": len(coco["images"]),
        "so_instance": so_instance,
        "so_ann_json": len(coco["annotations"]),
        "ds_lech": ds_lech,
        "ds_bi_bo": ds_bi_bo,
    }


# ===========================================================================
# main
# ===========================================================================

def main():
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(
        description="Kiểm tra dataset COCO trước khi train (Phase 3)")
    ap.add_argument("--config", default="configs/data.yaml",
                    help="Config chứa đường dẫn dataset và tên lớp")
    ap.add_argument("--mmdet-config", default="configs/mmdet/cascade_convnext_t_floodnet.py",
                    help="Config train — nguồn của `metainfo.classes`")
    ap.add_argument("--processed-dir", default=None,
                    help="Ghi đè thư mục dataset (mặc định lấy từ config)")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--mmdet", choices=["auto", "co", "khong"], default="auto",
                    help="Mức 2 (dựng dataset thật bằng mmdet): auto/co/khong")
    ap.add_argument("--so-anh-doi-chieu", type=int, default=0,
                    help="Số ảnh đối chiếu chi tiết từng ảnh (0 = tất cả)")
    args = ap.parse_args()

    print("=" * 72)
    print("KIỂM TRA DATASET TRƯỚC KHI TRAIN")
    print("=" * 72)

    if not os.path.exists(args.config):
        print(f"[!] Không thấy file cấu hình:\n    {args.config}")
        raise SystemExit(1)

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    processed_dir = args.processed_dir or cfg["paths"]["processed_dir"]
    try:
        ten_lop_data = doc_classes_tu_data_yaml(args.config)
    except ValueError as e:
        print(f"[!] {e}")
        raise SystemExit(1)
    print(f"dataset     : {processed_dir}")
    print(f"tên lớp     : {ten_lop_data}  (từ {args.config})")

    # Tên lớp mà mmdet SẼ dùng — ưu tiên đọc từ config train đã phân giải _base_.
    ten_lop_metainfo, nguon = (None, "không đọc được")
    if os.path.exists(args.mmdet_config):
        ten_lop_metainfo, nguon = doc_classes_tu_config_mmdet(args.mmdet_config)
    if ten_lop_metainfo is None:
        print(f"[!] Không đọc được `metainfo.classes` từ {args.mmdet_config} "
              f"({nguon}) — dùng tạm tên lớp trong {args.config}.")
        print("    Trên Colab sẽ đọc được bằng mmengine; đây chỉ là đường lui "
              "cho máy CPU chưa cài mmengine.")
        ten_lop_metainfo = tuple(ten_lop_data)
    else:
        print(f"metainfo    : {tuple(ten_lop_metainfo)}  "
              f"(đọc bằng {nguon} từ {args.mmdet_config})")

    ds_loi = []
    ds_split = [s.strip() for s in args.splits.split(",") if s.strip()]
    tg_theo_split = {}
    for split in ds_split:
        print("\n" + "-" * 72)
        print(f"SPLIT {split.upper()}")
        print("-" * 72)
        duong_ann = os.path.join(processed_dir, "annotations",
                                 f"instances_{split}.json")
        try:
            coco = doc_json(duong_ann)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[!] {e}")
            ds_loi.append(f"{split}: {e}")
            continue

        # --- Lỗi chặn train: phải sửa mới chạy tiếp ---
        for t in kiem_cau_truc(coco):
            ds_loi.append(f"{split}: {t}")
            print(f"  [LỖI] {t}")
        for t in kiem_khop_ten_lop(coco, ten_lop_metainfo):
            ds_loi.append(f"{split}: {t}")
            print(f"  [LỖI] {t}")

        ds_bo = kiem_box_bi_bo(coco)
        if ds_bo:
            ds_loi.append(f"{split}: {len(ds_bo)} box mmdet sẽ bỏ im lặng")
            print(f"  [LỖI] {len(ds_bo)} box sẽ bị mmdet BỎ IM LẶNG "
                  f"(không có cảnh báo nào khi train):")
            for ann_id, ten_file, ly_do in ds_bo[:10]:
                print(f"        ann {ann_id} ({ten_file}): {ly_do}")
            if len(ds_bo) > 10:
                print(f"        ... và {len(ds_bo) - 10} box nữa")

        # --- Cảnh báo, không chặn ---
        thieu_anh = kiem_anh_ton_tai(coco, os.path.join(processed_dir, "images", split))
        if thieu_anh:
            ds_loi.append(f"{split}: {len(thieu_anh)} file ảnh không có trên đĩa")
            print(f"  [LỖI] {len(thieu_anh)} ảnh có trong JSON nhưng KHÔNG có trên đĩa:")
            for t in thieu_anh[:10]:
                print(f"        {t}")

        ten_lop_theo_id = {c["id"]: c["name"] for c in coco["categories"]}
        tk = thong_ke(coco, ten_lop_theo_id)
        tg_theo_split[split] = tk
        print()
        print(bao_cao(split, tk))
        if tk["so_anh_rong"]:
            print(f"      ghi chú: {tk['so_anh_rong']:,} ảnh không có box nào — "
                  f"GIỮ LẠI có chủ ý (filter_cfg.filter_empty_gt=False), vì bỏ "
                  f"chúng đi thì model chỉ thấy toàn ảnh có nhà và đếm thừa.")

    # --- Mức 2 ---
    dung_mmdet = args.mmdet == "co"
    if args.mmdet == "auto":
        try:
            import mmdet  # noqa: F401
            dung_mmdet = True
        except ImportError:
            dung_mmdet = False
    print("\n" + "=" * 72)
    if not dung_mmdet:
        print("MỨC 2 (đếm instance bằng chính mmdet): BỎ QUA — máy này không có mmdet.")
        print("  Mức 2 là phép kiểm chứng minh KHÔNG box nào bị bỏ im lặng. Trên Colab")
        print("  luôn chạy được; ở đây thiếu thì bước kiểm này chưa hoàn tất.")
    elif not os.path.exists(args.mmdet_config):
        print(f"[!] Không thấy config train: {args.mmdet_config} — bỏ qua mức 2.")
    else:
        print("MỨC 2 — dựng dataset thật bằng config đồ án rồi đếm instance")
        print("=" * 72)
        for split in ds_split:
            try:
                kq = kiem_bang_mmdet(args.mmdet_config, split, args.so_anh_doi_chieu)
            except Exception as e:                              # noqa: BLE001
                ds_loi.append(f"{split}: mức 2 lỗi — {e}")
                print(f"  [{split}] [LỖI] dựng dataset thất bại: {e}")
                continue
            print(f"  [{split}] dataset {kq['so_anh_dataset']:,} ảnh "
                  f"(JSON {kq['so_anh_json']:,}) | "
                  f"instance {kq['so_instance']:,} (JSON {kq['so_ann_json']:,})")
            if kq["so_instance"] != kq["so_ann_json"]:
                chenh = kq["so_ann_json"] - kq["so_instance"]
                ds_loi.append(f"{split}: mmdet thấy ít hơn JSON {chenh:,} instance")
                print(f"      [LỖI] mmdet thấy ÍT HƠN {chenh:,} box so với JSON — "
                      f"khoanh vùng:")
                for ten_file, so_thieu in kq["ds_bi_bo"][:10]:
                    print(f"        {ten_file}: thiếu {so_thieu} box")
            else:
                print(f"      ✅ KHỚP HOÀN TOÀN — không box nào bị bỏ im lặng.")
            if kq["so_anh_dataset"] != kq["so_anh_json"]:
                print(f"      [LỖI] số ảnh lệch: mmdet giữ "
                      f"{kq['so_anh_dataset']:,} / JSON {kq['so_anh_json']:,} "
                      f"(thường là do `filter_cfg` lọc ảnh)")

    print("\n" + "=" * 72)
    if ds_loi:
        print(f"*** {len(ds_loi)} VẤN ĐỀ — CHƯA NÊN TRAIN ***")
        for t in ds_loi:
            print("  - " + t)
        raise SystemExit(1)
    print("*** ĐẠT HẾT — dataset dùng được để train. ***")
    if tg_theo_split:
        tong = sum(t["so_box"] for t in tg_theo_split.values())
        print(f"Tổng {sum(t['so_anh'] for t in tg_theo_split.values()):,} ảnh / "
              f"{tong:,} box.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Chuyển mask FloodNet -> COCO — Phase 2.

Biến 13 GB zip thành một dataset gọn mà model đọc được:

    floodnet_coco/
      images/{train,val,test}/{split}_{id}.jpg    ảnh đã resize, JPEG q95
      masks/{train,val,test}/{split}_{id}.png     mask 10 lớp đã resize (tuỳ chọn)
      annotations/instances_{split}.json          COCO, 2 category
      counts_{split}.csv                          số nhà mỗi lớp mỗi ảnh

BA QUYẾT ĐỊNH KỸ THUẬT ĐÁNG CHÚ Ý (để trả lời khi bảo vệ):

  * GIỮ NGUYÊN 10 LỚP trong mask đã resize, không chỉ giữ 2 lớp nhà. Mask PNG
    nén rất tốt (các vùng màu phẳng) nên tốn thêm không đáng kể, mà đổi lại giữ
    được khả năng làm lại phép kiểm chứng "nhà ngập có chạm nước không" ở Phase 6
    và mọi phân tích lỗi cần đến lớp nước. Bỏ đi thì phải giải nén lại 13 GB.

  * DIỆN TÍCH `area` = SỐ PIXEL THẬT của component, không phải diện tích đa giác.
    Đây mới là con số đã dùng để lọc nhiễu, và là con số đúng về mặt hình học khi
    hình có lỗ hoặc rỗng ruột. COCO chỉ dùng `area` để phân loại small/medium/
    large nên không ảnh hưởng gì tới lúc train.

  * BOX SÁT MÉP ẢNH VẪN GIỮ. Nhà bị cắt ở rìa ảnh vẫn là nhà cần đếm; bỏ đi thì
    số nhà của đồ án ít hơn số nhà thật trên ảnh. Cái giá là box bị cụt một cạnh,
    và box cụt thì khó học hơn — nên số lượng box sát mép được ĐẾM và in ra
    trong báo cáo, để nếu sau này phân tích lỗi thấy nhà sát mép là nguồn lỗi
    chính thì đã có sẵn số liệu mà bàn.

KHÁC BIỆT CÓ CHỦ Ý so với Phase 1: bộ lọc ở đây là CẢ HAI điều kiện
(`min(bw,bh) >= min_side_px` VÀ `area >= min_area`), trong khi cột `so_box_loc`
của báo cáo EDA chỉ lọc theo cạnh. Hai bộ lọc chỉ khác nhau ở những hình thoi
kiểu đường chéo 8x8px (8 pixel thật nhưng bbox 8x8). Vì vậy bước đối chiếu ở đây
tính LẠI từ `audit.jsonl` bằng đúng bộ lọc của Phase 2 rồi mới so — so thẳng với
cột `so_box_loc` là so sai và sẽ tưởng nhầm là Phase 2 có bug.

Chạy:
    python scripts/build_coco.py --config configs/data.yaml
    python scripts/build_coco.py --config configs/data.yaml --max-images 3   # chạy thử
"""

import argparse
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

import cv2
import numpy as np
import yaml

from floodcount.data.audit import (chuan_hoa_stem, doc_anh, doc_ket_qua_da_co,
                                   ghep_cap_anh_mask, liet_ke_cay,
                                   phan_loai_thu_muc)
from floodcount.data.resize import (ghi_anh, kich_thuoc_sau_resize, resize_anh,
                                    resize_mask)


# ===========================================================================
# COCO: category & tên file
# ===========================================================================

def tao_categories(classes):
    """{1: 'flooded_building', 2: 'non_flooded_building'} -> list category COCO.

    `id` lấy ĐÚNG BẰNG giá trị mask, KHÔNG đánh số lại. mmdet dùng `id` làm chỉ
    số lớp trong head, nên id bắt buộc liên tục từ 1..N. FloodNet có 1 và 2 nên
    tình cờ đã đúng — nhưng không dựa vào may mắn: kiểm ngay ở đây và báo lỗi
    thật rõ nếu config đổi thành bộ giá trị khác, thay vì để mmdet chết ở tận
    Phase 3 với thông báo khó hiểu.
    """
    ds = sorted(classes.items())
    gia_tri = [k for k, _ in ds]
    if gia_tri != list(range(1, len(ds) + 1)):
        raise ValueError(
            f"Giá trị mask của các lớp phải liên tục từ 1..{len(ds)} thì COCO/mmdet "
            f"mới dùng được, config đang có {gia_tri}. "
            f"Sửa `preprocess.classes` trong configs/data.yaml.")
    return [{"id": k, "name": ten, "supercategory": "building"} for k, ten in ds]


def ten_file_moi(split, ten_trong_zip):
    """Tên file phẳng trong dataset đã xử lý: `train_1.jpg`.

    PHẢI có tiền tố split. Cả ba split đều đặt tên ảnh là `1.jpg`, `2.jpg`... nên
    nếu chỉ lấy tên file thì ảnh của train/val/test ghi ĐÈ lên nhau và mất ảnh mà
    không hề báo lỗi. Ở đây mỗi split có thư mục riêng nên `train_1` là đủ phân
    biệt (bên `audit.py` phải dài hơn vì nó ghi cả ba split vào MỘT thư mục).
    """
    return f"{split}_{chuan_hoa_stem(ten_trong_zip)}.jpg"


# ===========================================================================
# Tách component -> box
# ===========================================================================

def tach_box_va_polygon(mask_nho, gia_tri_lop, min_side_px, min_area,
                        eps_polygon=1.0):
    """Tách component của MỘT lớp trong mask đã resize -> box đã lọc nhiễu.

    Trả về (ds_box, so_component_tho, so_bi_loc) với
    ds_box = [{"bbox": [x, y, w, h], "area": int, "segmentation": [poly]}, ...]

    LỌC BẰNG CẢ HAI ĐIỀU KIỆN, không phải một:
      * cạnh nhỏ nhất >= min_side_px
      * diện tích      >= min_area
    Chỉ lọc diện tích thì một vệt rác 1x500px lọt qua (diện tích 500 > 64 nhưng
    bbox chỉ cao 1px — không phải nhà). Chỉ lọc cạnh thì một đường chéo 8x8 pixel
    lọt qua (bbox 8x8 nhưng chỉ có 8 pixel thật, cũng không phải nhà). Config đặt
    `min_area = min_side_px^2` chính là để hai ngưỡng khớp nhau, và phải áp dụng
    cả hai thì "một căn nhà hợp lệ" mới chỉ có MỘT định nghĩa trong toàn pipeline.
    """
    nhi_phan = (mask_nho == gia_tri_lop).astype(np.uint8)
    so_nhan, nhan, stats, _ = cv2.connectedComponentsWithStats(
        nhi_phan, connectivity=8)

    ds_box = []
    for i in range(1, so_nhan):          # nhãn 0 là nền
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if min(bw, bh) < min_side_px or area < min_area:
            continue

        # Polygon tìm trên ROI của CHÍNH component (bbox vài chục pixel) chứ không
        # quét lại cả ảnh 1536x1152 cho từng box. Cả dataset có ~7500 box; quét
        # toàn ảnh mỗi box thì chậm gấp hàng trăm lần mà kết quả y hệt.
        #
        # `== i` (không phải `> 0`) là bắt buộc: ROI có thể chứa pixel của
        # component KHÁC cùng lớp nằm lọt trong bbox, lấy `> 0` là gộp nhầm hai
        # căn nhà vào một polygon.
        roi = (nhan[y:y + bh, x:x + bw] == i).astype(np.uint8)
        duong_vien, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        if duong_vien:
            c = max(duong_vien, key=cv2.contourArea)
            # approxPolyDP giảm số điểm, sai số <= eps_polygon pixel. Không
            # simplify thì một căn nhà có thể thành polygon hàng nghìn điểm, làm
            # file JSON phình ra vô ích.
            c = cv2.approxPolyDP(c, eps_polygon, True)
            diem = (c.reshape(-1, 2) + np.array([x, y])).flatten().tolist()
            # COCO yêu cầu polygon tối thiểu 3 điểm (6 số). Ít hơn thì thà để rỗng
            # còn hơn ghi một polygon không hợp lệ vào file.
            if len(diem) >= 6:
                segmentation = [diem]

        ds_box.append({
            "bbox": [x, y, bw, bh],
            # Số PIXEL thật của component — đúng bằng con số đã dùng để lọc nhiễu,
            # và đúng về hình học cả khi hình có lỗ. Không dùng diện tích đa giác.
            "area": area,
            "segmentation": segmentation,
        })

    so_tho = so_nhan - 1
    return ds_box, so_tho, so_tho - len(ds_box)


# ===========================================================================
# Xử lý một cặp ảnh/mask
# ===========================================================================

def xu_ly_mot_cap(zf, ten_anh, ten_mask, ten_file_ra, classes, target_long_side,
                  min_side_px, min_area, eps_polygon, jpeg_quality, luu_mask,
                  thu_muc_anh, thu_muc_mask):
    """Đọc 1 cặp ảnh/mask, ghi ảnh (và mask) đã resize, trả về bản ghi để ráp COCO.

    Bản ghi CỐ Ý không chứa `id` của COCO: id được đánh lúc ráp file cuối, theo
    thứ tự tên file đã sắp xếp. Nếu đánh id ngay ở đây thì chạy lại giữa chừng
    (resume) sẽ sinh id trùng hoặc nhảy cóc, mà COCO thì id phải duy nhất.

    Trả về dict bản ghi, hoặc {"loi": "..."} nếu cặp này không dùng được.
    """
    mask = doc_anh(zf, ten_mask, giai_ma_mau=False)
    if mask is None:
        return {"loi": f"đọc không được mask {ten_mask}"}
    if mask.ndim != 2:
        # Mask nhiều kênh nghĩa là đã chọn nhầm thư mục — không đoán, báo lỗi luôn
        return {"loi": f"mask có {mask.shape[2]} kênh, không phải ảnh xám"}

    anh = doc_anh(zf, ten_anh, giai_ma_mau=True)
    if anh is None:
        return {"loi": f"đọc không được ảnh {ten_anh}"}

    # Ảnh và mask PHẢI cùng kích thước. Phase 1 mới chỉ kiểm chuyện này trên 30
    # ảnh overlay (độ tin cậy thấp, ghi ở NOTES muc 2.1), nên Phase 2 kiểm TỪNG
    # CẶP. Lệch nhau thì box tách từ mask sẽ nằm sai chỗ trên ảnh — thà mất một
    # cặp và biết rõ còn hơn train trên nhãn lệch mà không ai phát hiện.
    if anh.shape[:2] != mask.shape[:2]:
        return {"loi": (f"ảnh {anh.shape[1]}x{anh.shape[0]} lệch mask "
                        f"{mask.shape[1]}x{mask.shape[0]}")}

    h, w = mask.shape[:2]
    h2, w2 = kich_thuoc_sau_resize(h, w, target_long_side)
    mask_nho = resize_mask(mask, (h2, w2))
    anh_nho = resize_anh(anh, (h2, w2))

    duong_anh = os.path.join(thu_muc_anh, ten_file_ra)
    if not ghi_anh(duong_anh, anh_nho, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        return {"loi": f"ghi không được {duong_anh}"}

    if luu_mask:
        duong_mask = os.path.join(thu_muc_mask,
                                  ten_file_ra.rsplit(".", 1)[0] + ".png")
        if not ghi_anh(duong_mask, mask_nho):
            return {"loi": f"ghi không được {duong_mask}"}

    boxes, dem_lop, so_sat_mep = [], {}, 0
    for gia_tri_lop, ten_lop in classes.items():
        ds_box, _, _ = tach_box_va_polygon(
            mask_nho, gia_tri_lop, min_side_px, min_area, eps_polygon)
        dem_lop[ten_lop] = len(ds_box)
        for b in ds_box:
            x, y, bw, bh = b["bbox"]
            # Box chạm mép ảnh: giữ lại nhưng đếm riêng (xem docstring đầu file)
            if x == 0 or y == 0 or x + bw >= w2 or y + bh >= h2:
                so_sat_mep += 1
            boxes.append({"category_id": gia_tri_lop, **b})

    return {
        "file_name": ten_file_ra,
        "width": w2,
        "height": h2,
        "ten_goc": chuan_hoa_stem(ten_anh),
        "kich_thuoc_goc": [h, w],
        "boxes": boxes,
        "dem_lop": dem_lop,
        "so_box_sat_mep": so_sat_mep,
    }


# ===========================================================================
# Ráp file COCO & kiểm tra
# ===========================================================================

def rap_coco(ds_ban_ghi, classes, split):
    """Ráp bản ghi của MỘT split thành dict COCO hoàn chỉnh.

    `id` của ảnh đánh theo thứ tự TÊN FILE ĐÃ SẮP XẾP, không theo thứ tự xử lý.
    Nhờ vậy chạy lại giữa chừng (resume) hay đổi thứ tự xử lý đều cho ra file
    giống hệt nhau — quan trọng vì kết quả Phase 5 phải tái lập được.
    """
    ds = sorted(ds_ban_ghi, key=lambda r: r["file_name"])
    coco = {
        "info": {"description": f"FloodNet {split} — 2 lớp nhà, mask -> COCO",
                 "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": tao_categories(classes),
    }
    ann_id = 1
    for img_id, r in enumerate(ds, start=1):
        coco["images"].append({
            "id": img_id,
            "file_name": r["file_name"],
            "width": r["width"],
            "height": r["height"],
            # Hai khoá dưới đây KHÔNG thuộc chuẩn COCO, chỉ để tra ngược về ảnh
            # gốc trong FloodNet. mmdet bỏ qua khoá lạ nên vô hại.
            "ten_goc": r["ten_goc"],
            "kich_thuoc_goc": r["kich_thuoc_goc"],
        })
        for b in r["boxes"]:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": b["category_id"],
                "bbox": b["bbox"],
                "area": b["area"],
                "segmentation": b["segmentation"],
                "iscrowd": 0,
            })
            ann_id += 1
    return coco


def kiem_tra_coco(coco, classes):
    """Kiểm tra cấu trúc COCO bằng tay. Trả về danh sách câu lỗi (rỗng = đạt).

    Cố ý KHÔNG dùng pycocotools ở đây: hàm này phải chạy được cả trên máy không
    cài pycocotools, và phải bắt được lỗi TRƯỚC khi pycocotools kịp khó chịu.
    pycocotools chỉ kiểm cấu trúc; nó không biết box có nằm trong ảnh hay không.
    """
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

        im = ds_anh.get(a["image_id"])
        if im is None:
            loi.append(f"annotation {a['id']} trỏ tới image_id không tồn tại "
                       f"{a['image_id']}")
            continue
        if a["category_id"] not in classes:
            loi.append(f"annotation {a['id']} có category_id lạ {a['category_id']}")

        x, y, bw, bh = a["bbox"]
        if bw <= 0 or bh <= 0:
            loi.append(f"annotation {a['id']} bbox không dương: {a['bbox']}")
        elif x < 0 or y < 0 or x + bw > im["width"] or y + bh > im["height"]:
            loi.append(f"annotation {a['id']} bbox {a['bbox']} vượt ra ngoài ảnh "
                       f"{im['width']}x{im['height']}")
        if a["area"] <= 0:
            loi.append(f"annotation {a['id']} area không dương: {a['area']}")
    return loi


def doc_lai_va_dem(duong_json, classes):
    """Đọc lại file COCO TỪ ĐĨA và đếm số box mỗi lớp mỗi ảnh.

    Đọc lại từ đĩa chứ không dùng dict còn trong bộ nhớ: như vậy counts CSV và
    mọi phép kiểm đều soi ĐÚNG thứ đã ghi ra file. Nếu khâu ghi JSON có lỗi
    (khoá sai, số bị làm tròn) thì lỗi lộ ra ngay ở đây, thay vì lộ ra ở Phase 5
    khi đã train xong mới phát hiện GT sai.
    """
    with open(duong_json, encoding="utf-8") as f:
        coco = json.load(f)

    ten_lop = {k: v for k, v in classes.items()}
    dem = defaultdict(lambda: Counter())
    for a in coco["annotations"]:
        dem[a["image_id"]][ten_lop[a["category_id"]]] += 1
    return coco, dem


def ghi_counts_csv(coco, dem, duong_csv, classes):
    """Ghi `counts_{split}.csv` từ COCO đã đọc lại. Đây là GT để Phase 5 chấm điểm."""
    ds_lop = [ten for _, ten in sorted(classes.items())]
    with open(duong_csv, "w", encoding="utf-8", newline="") as f:
        f.write("image_id,file_name,ten_goc," + ",".join(ds_lop) + ",tong\n")
        for im in sorted(coco["images"], key=lambda i: i["id"]):
            d = dem.get(im["id"], Counter())
            so = [d.get(t, 0) for t in ds_lop]
            f.write(f"{im['id']},{im['file_name']},{im['ten_goc']},"
                    f"{','.join(str(x) for x in so)},{sum(so)}\n")


# ===========================================================================
# Đối chiếu với Phase 1
# ===========================================================================

def doi_chieu_voi_phase1(duong_jsonl, classes, min_side_px, min_area, dem_phase2):
    """So số box của Phase 2 với con số tính LẠI từ `audit.jsonl` của Phase 1.

    Đây là phép kiểm quan trọng nhất của GATE 2: nó chứng minh Phase 2 dùng đúng
    định nghĩa "nhà hợp lệ" mà Phase 1 đã dùng để ra quyết định tiling. Hai bên
    lệch nghĩa là một trong hai đang hiểu "nhà" theo nghĩa khác, và mọi so sánh
    số liệu giữa hai phase từ đó về sau đều vô nghĩa.

    Chú ý: tính lại với ĐÚNG bộ lọc của Phase 2 (cả cạnh VÀ diện tích), không so
    thẳng với cột `so_box_loc` của báo cáo EDA — cột đó chỉ lọc theo cạnh.

    Trả về dict, hoặc None nếu không có `audit.jsonl` (ví dụ /content đã reset).
    """
    if not os.path.exists(duong_jsonl):
        return None

    dem_p1 = Counter()
    so_anh = 0
    with open(duong_jsonl, encoding="utf-8") as f:
        for dong in f:
            dong = dong.strip()
            if not dong:
                continue
            try:
                r = json.loads(dong)
            except json.JSONDecodeError:
                continue          # dòng cuối bị cắt cụt do crash giữa chừng
            if "loi" in r:
                continue
            so_anh += 1
            for ten_lop, lop in r["lop"].items():
                for _x, _y, bw, bh, area in lop["boxes"]:
                    if min(bw, bh) >= min_side_px and area >= min_area:
                        dem_p1[ten_lop] += 1

    lech = {t: dem_phase2.get(t, 0) - dem_p1.get(t, 0)
            for t in set(dem_p1) | set(dem_phase2)}
    return {"so_anh": so_anh, "dem_phase1": dict(dem_p1), "lech": lech,
            "khop": all(v == 0 for v in lech.values())}


# ===========================================================================
# Ảnh overlay GT
# ===========================================================================

def ve_overlay_gt(coco, thu_muc_anh, thu_muc_ra, ten_lop_theo_id, so_anh, seed):
    """Vẽ box LẤY TỪ COCO JSON lên ảnh ĐÃ RESIZE nằm trong dataset.

    Cố ý vẽ từ JSON và đọc lại ảnh JPEG từ đĩa, chứ không vẽ từ mask gốc: như vậy
    phép kiểm này soi ĐÚNG thứ model sẽ đọc lúc train (file JSON + ảnh JPEG).
    JSON sai toạ độ, hay ảnh ghi nhầm chỗ, đều lộ ra ngay ở đây — còn vẽ từ mask
    thì lại kiểm hoá ra chính cái mask đã đúng từ đầu.

    Trả về số ảnh vẽ được.
    """
    mau = {"flooded_building": (0, 0, 255),        # đỏ = nhà ngập
           "non_flooded_building": (0, 200, 0)}    # xanh lá = nhà không ngập

    # Gom annotation theo ảnh, chỉ vẽ ảnh CÓ box (ảnh rỗng không nói lên điều gì)
    theo_anh = defaultdict(list)
    for a in coco["annotations"]:
        theo_anh[a["image_id"]].append(a)
    ds_co_box = [im for im in coco["images"] if theo_anh.get(im["id"])]
    if not ds_co_box:
        return 0

    rng = random.Random(seed)
    so_ve = 0
    for im in rng.sample(ds_co_box, min(so_anh, len(ds_co_box))):
        duong = os.path.join(thu_muc_anh, im["file_name"])
        anh = cv2.imread(duong)
        if anh is None:
            print(f"    [!] không đọc lại được ảnh vừa ghi: {duong}")
            continue
        for a in theo_anh[im["id"]]:
            x, y, bw, bh = (int(v) for v in a["bbox"])
            ten = ten_lop_theo_id[a["category_id"]]
            cv2.rectangle(anh, (x, y), (x + bw, y + bh), mau.get(ten, (255, 255, 255)), 2)
        # Ghi số box của từng lớp lên ảnh, để người xem đối chiếu được với CSV
        dem = Counter(ten_lop_theo_id[a["category_id"]] for a in theo_anh[im["id"]])
        for i, (t, n) in enumerate(sorted(dem.items())):
            dong = f"{t}: {n} box"
            vi_tri = (10, 30 + i * 28)
            cv2.putText(anh, dong, vi_tri, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(anh, dong, vi_tri, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)

        if ghi_anh(os.path.join(thu_muc_ra, f"gt_{im['file_name']}"), anh,
                   [cv2.IMWRITE_JPEG_QUALITY, 88]):
            so_ve += 1
    return so_ve


# ===========================================================================
# Báo cáo
# ===========================================================================

def ghi_bao_cao(duong_dan, thong_ke, doi_chieu):
    """Ghi BUILD_REPORT.md và trả về nội dung."""
    d = ["# Báo cáo Phase 2 — mask → COCO\n",
         f"Cạnh dài sau resize: **{thong_ke['target_long_side']}px**; "
         f"bộ lọc: cạnh nhỏ nhất >= {thong_ke['min_side_px']}px "
         f"VÀ diện tích >= {thong_ke['min_area']}px².\n"]

    if thong_ke["ds_loi"]:
        d += ["> ## ⚠️ Có cặp ảnh/mask bị bỏ\n>",
              f"> **{len(thong_ke['ds_loi'])}** cặp không dùng được:\n>"]
        for t in thong_ke["ds_loi"][:20]:
            d.append(f"> - {t}")
        if len(thong_ke["ds_loi"]) > 20:
            d.append(f"> - ... và {len(thong_ke['ds_loi']) - 20} cặp nữa")
        d.append("")

    d += ["## Số ảnh và số box mỗi split\n",
          "| Split | Số ảnh | Box nhà ngập | Box nhà không ngập | Tổng box "
          "| Ảnh có nhà ngập | Box sát mép ảnh |",
          "|---|---|---|---|---|---|---|"]
    for split, tk in thong_ke["theo_split"].items():
        d.append(f"| {split} | {tk['so_anh']:,} | {tk['dem'].get('flooded_building', 0):,} "
                 f"| {tk['dem'].get('non_flooded_building', 0):,} | {tk['tong_box']:,} "
                 f"| {tk['so_anh_co_nha_ngap']:,} | {tk['so_sat_mep']:,} |")
    t = thong_ke["tong"]
    d.append(f"| **Tổng** | **{t['so_anh']:,}** | **{t['dem'].get('flooded_building', 0):,}** "
             f"| **{t['dem'].get('non_flooded_building', 0):,}** | **{t['tong_box']:,}** "
             f"| **{t['so_anh_co_nha_ngap']:,}** | **{t['so_sat_mep']:,}** |")

    # --- Đối chiếu Phase 1: phép kiểm quan trọng nhất của GATE 2 ---
    d += ["", "## Đối chiếu với Phase 1\n"]
    if doi_chieu is None:
        d += ["**BỎ QUA** — không thấy `audit.jsonl` của Phase 1 "
              "(thường là `/content` đã bị reset). Muốn có phép kiểm này thì chạy "
              "lại ô [1.6] của notebook 01 trong cùng phiên, rồi chạy lại Phase 2.\n"]
    else:
        d += [f"Tính lại từ `audit.jsonl` ({doi_chieu['so_anh']:,} ảnh) bằng **đúng bộ lọc "
              "của Phase 2**, rồi so với số box vừa dựng:\n",
              "| Lớp | Phase 1 tính lại | Phase 2 | Lệch |", "|---|---|---|---|"]
        for ten in sorted(set(doi_chieu["dem_phase1"]) | set(doi_chieu["lech"])):
            d.append(f"| {ten} | {doi_chieu['dem_phase1'].get(ten, 0):,} "
                     f"| {doi_chieu['dem_phase1'].get(ten, 0) + doi_chieu['lech'][ten]:,} "
                     f"| {doi_chieu['lech'][ten]:+,} |")
        if doi_chieu["khop"]:
            d.append("\n✅ **KHỚP HOÀN TOÀN** — Phase 2 dùng đúng định nghĩa \"nhà hợp lệ\" "
                     "mà Phase 1 đã dùng để quyết định tiling. Số liệu hai phase so sánh "
                     "được với nhau.\n")
        else:
            d += ["\n❌ **LỆCH** — hai phase đang hiểu \"nhà\" khác nhau. **Đừng train.** "
                  "Kiểm lại `min_side_px`/`min_area` trong config, và xem có phải "
                  "`audit.jsonl` được tạo với `--max-images` (chạy thử) không.\n"]

    d += ["", "## Kiểm tra cấu trúc COCO\n"]
    if thong_ke["loi_coco"]:
        d += [f"❌ **{len(thong_ke['loi_coco'])} lỗi:**\n"]
        for t in thong_ke["loi_coco"][:20]:
            d.append(f"- {t}")
        if len(thong_ke["loi_coco"]) > 20:
            d.append(f"- ... và {len(thong_ke['loi_coco']) - 20} lỗi nữa")
    else:
        d.append("✅ Không có lỗi cấu trúc: mọi `image_id` đều tồn tại, mọi `bbox` "
                 "nằm trong ảnh và có kích thước dương, `id` không trùng.\n")

    if thong_ke.get("pycocotools"):
        d.append(f"✅ `pycocotools` đọc được cả 3 file: {thong_ke['pycocotools']}\n")

    noi_dung = "\n".join(d)
    with open(duong_dan, "w", encoding="utf-8") as f:
        f.write(noi_dung)
    return noi_dung


# ===========================================================================
# main
# ===========================================================================

def main():
    # Console Windows mặc định là cp1252, không mã hoá được chữ có dấu: in một
    # dòng tiếng Việt là ném UnicodeEncodeError ngay. Ép hai luồng ra UTF-8.
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="Chuyển mask FloodNet -> COCO (Phase 2)")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--zip", dest="zip_path", default=None, help="Ghi đè đường dẫn zip")
    ap.add_argument("--processed-dir", default=None, help="Ghi đè nơi dựng dataset")
    ap.add_argument("--report-dir", default=None, help="Ghi đè nơi ghi báo cáo")
    ap.add_argument("--work-dir", default=None,
                    help="Ghi đè thư mục trung gian (dùng thư mục riêng khi chạy thử)")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Chỉ xử lý N ảnh đầu mỗi split (chạy thử)")
    ap.add_argument("--fresh", action="store_true",
                    help="Bỏ qua kết quả lần chạy trước, làm lại từ đầu")
    args = ap.parse_args()

    # In băng-rôn TRƯỚC khi mở file config: nếu để sau, một lỗi sớm làm script
    # thoát mà stdout trống trơn, nhìn vào chỉ thấy "mã thoát 1" mà không biết vì sao.
    print("=" * 72)
    print("CHUYỂN MASK -> COCO — PHASE 2")
    print("=" * 72)

    if not os.path.exists(args.config):
        print(f"[!] Không thấy file cấu hình:\n    {args.config}\n"
              f"    Chạy lại ô [2.3] trong notebook để lấy code mới nhất.")
        raise SystemExit(1)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    zip_path = args.zip_path or cfg["paths"]["zip"]
    processed_dir = args.processed_dir or cfg["paths"]["processed_dir"]
    report_dir = args.report_dir or cfg["paths"]["build_output_dir"]
    work_dir = args.work_dir or cfg["paths"]["work_dir"]
    classes = {int(k): v for k, v in cfg["preprocess"]["classes"].items()}
    target_long_side = cfg["preprocess"]["target_long_side"]
    min_area = cfg["preprocess"]["min_area"]
    min_side_px = cfg["decisions"]["min_side_px"]
    c2 = cfg["build_coco"]
    max_images = args.max_images if args.max_images is not None else c2["max_images"]

    for d in (processed_dir, report_dir, work_dir):
        os.makedirs(d, exist_ok=True)
    duong_jsonl = os.path.join(work_dir, "build.jsonl")

    if not os.path.exists(zip_path):
        print(f"[!] Không thấy file zip:\n    {zip_path}\n"
              f"    Kiểm tra Drive đã mount chưa, và đường dẫn trong {args.config}.")
        raise SystemExit(1)

    print(f"zip        : {zip_path}")
    print(f"dataset ra : {processed_dir}")
    print(f"báo cáo    : {report_dir}")
    print(f"resize     : cạnh dài {target_long_side}px")
    print(f"lọc nhiễu  : cạnh >= {min_side_px}px VÀ diện tích >= {min_area}px²")
    print(f"lớp        : {classes}")

    with zipfile.ZipFile(zip_path) as zf:
        # ---------------------------------------------------------------
        # Bước 1: cấu trúc thật của zip
        # ---------------------------------------------------------------
        print("\n--- BƯỚC 1: CẤU TRÚC ZIP ---")
        ds_ten = zf.namelist()
        cay = liet_ke_cay(ds_ten)
        thong_tin = phan_loai_thu_muc(zf, cay, ds_ten)
        ds_cap, canh_bao = ghep_cap_anh_mask(thong_tin, ds_ten)
        if canh_bao:
            print("\n  CẢNH BÁO:")
            for c in canh_bao:
                print(f"    {c}")
        if not ds_cap:
            print("\n[!] Không ghép được cặp ảnh/mask nào.\n"
                  "    Chạy notebook 01 trước để xem cấu trúc zip.")
            raise SystemExit(1)

        # ---------------------------------------------------------------
        # Bước 2: dựng dataset (có resume)
        # ---------------------------------------------------------------
        print("\n--- BƯỚC 2: DỰNG ẢNH + MASK + BOX ---")
        # Mỗi split một tên thư mục riêng. Tên split lấy từ thư mục CHA của cặp
        # ảnh/mask (vd ".../train/train-org-img" -> "train"), đúng bằng khoá mà
        # notebook 01 đã dùng trong báo cáo EDA.
        theo_split = [(PurePosixPath(cha).name, ds) for cha, ds in ds_cap]
        ds_split = [s for s, _ in theo_split]
        print(f"  Split: {ds_split}")

        for s in ds_split:
            os.makedirs(os.path.join(processed_dir, "images", s), exist_ok=True)
            if c2["luu_mask"]:
                os.makedirs(os.path.join(processed_dir, "masks", s), exist_ok=True)

        # Khoá là `file_name` (KHÔNG phải `anh` như Phase 1) — xem docstring
        # doc_ket_qua_da_co() để hiểu vì sao dùng nhầm khoá lại tắt resume im lặng.
        da_co = {} if args.fresh else doc_ket_qua_da_co(duong_jsonl, khoa="file_name")
        if da_co:
            # Ảnh có thể đã mất trong khi bản ghi còn (hoặc ngược lại) nếu lần
            # chạy trước bị ngắt giữa chừng. Chỉ tin bản ghi khi ẢNH CÒN THẬT.
            da_co = {k: v for k, v in da_co.items()
                     if os.path.exists(os.path.join(processed_dir, "images",
                                                    v["split"], v["file_name"]))}
            print(f"  Đọc lại {len(da_co)} ảnh đã xử lý lần trước "
                  f"(dùng --fresh nếu muốn làm lại từ đầu)")

        tong_cap = sum(min(len(ds), max_images) if max_images else len(ds)
                       for _, ds in theo_split)
        print(f"  Sẽ xử lý {tong_cap} cặp ảnh/mask")

        ds_loi = []
        with open(duong_jsonl, "a", encoding="utf-8") as f_jsonl:
            dem = 0
            for split, ds in theo_split:
                thu_muc_anh = os.path.join(processed_dir, "images", split)
                thu_muc_mask = os.path.join(processed_dir, "masks", split)
                for ten_anh, ten_mask in (ds[:max_images] if max_images else ds):
                    ten_file_ra = ten_file_moi(split, ten_anh)
                    if ten_file_ra in da_co:
                        continue
                    r = xu_ly_mot_cap(
                        zf, ten_anh, ten_mask, ten_file_ra, classes, target_long_side,
                        min_side_px, min_area, c2["eps_polygon"], c2["jpeg_quality"],
                        c2["luu_mask"], thu_muc_anh, thu_muc_mask)
                    if "loi" in r:
                        ds_loi.append(f"{split}/{ten_file_ra}: {r['loi']}")
                        continue
                    r["split"] = split
                    dem += 1
                    f_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f_jsonl.flush()   # ghi ngay: Colab ngắt giữa chừng vẫn giữ được
                    if dem % 100 == 0:
                        print(f"    ... đã xử lý {dem}/{tong_cap}")
        if ds_loi:
            print(f"  [!] {len(ds_loi)} cặp bị bỏ (xem chi tiết trong báo cáo)")

        # ---------------------------------------------------------------
        # Bước 3: ráp COCO JSON + counts CSV
        # ---------------------------------------------------------------
        print("\n--- BƯỚC 3: RÁP COCO JSON + COUNTS CSV ---")
        # Khoá `file_name` — xem docstring doc_ket_qua_da_co(). Thiếu tham số này
        # là Phase 2 ghi ra dataset 0 ảnh mà vẫn báo thành công.
        ban_ghi = list(doc_ket_qua_da_co(duong_jsonl, khoa="file_name").values())
        print(f"  {len(ban_ghi)} bản ghi")
        if not ban_ghi:
            # Không có bản ghi nào thì mọi bước sau đều "thành công" một cách vô
            # nghĩa: 3 file COCO rỗng, CSV chỉ có dòng tiêu đề, 0 ảnh overlay. Đó
            # là kiểu thất bại tệ nhất — nhìn mã thoát thì tưởng đã xong.
            print(f"\n[!] KHÔNG dựng được ảnh nào — cả {len(ds_loi)} cặp đều lỗi:")
            for t in ds_loi[:10]:
                print(f"    {t}")
            if len(ds_loi) > 10:
                print(f"    ... và {len(ds_loi) - 10} cặp nữa")
            raise SystemExit(1)

        thu_muc_ann = os.path.join(processed_dir, "annotations")
        os.makedirs(thu_muc_ann, exist_ok=True)

        theo_split_tk, tong_tk = {}, Counter()
        for split in ds_split:
            cua_split = [r for r in ban_ghi if r.get("split") == split]
            coco = rap_coco(cua_split, classes, split)
            duong_json = os.path.join(thu_muc_ann, f"instances_{split}.json")
            with open(duong_json, "w", encoding="utf-8") as f:
                json.dump(coco, f, ensure_ascii=False)

            # Đọc LẠI từ đĩa rồi mới đếm — xem docstring doc_lai_va_dem()
            coco_doc, dem = doc_lai_va_dem(duong_json, classes)
            ghi_counts_csv(coco_doc, dem, os.path.join(processed_dir,
                                                       f"counts_{split}.csv"), classes)

            dem_lop = Counter()
            so_co_nha_ngap = 0
            for im in coco_doc["images"]:
                d_im = dem.get(im["id"], Counter())
                for t, n in d_im.items():
                    dem_lop[t] += n
                if d_im.get("flooded_building", 0) > 0:
                    so_co_nha_ngap += 1

            theo_split_tk[split] = {
                "so_anh": len(coco_doc["images"]),
                "dem": dict(dem_lop),
                "tong_box": sum(dem_lop.values()),
                "so_anh_co_nha_ngap": so_co_nha_ngap,
                "so_sat_mep": sum(r.get("so_box_sat_mep", 0) for r in cua_split),
            }
            tong_tk.update(dem_lop)
            print(f"  {split:<6}: {len(coco_doc['images']):>5} ảnh, "
                  f"{sum(dem_lop.values()):>5} box "
                  f"({dict(sorted(dem_lop.items()))})")

        tong = {
            "so_anh": sum(v["so_anh"] for v in theo_split_tk.values()),
            "dem": dict(tong_tk),
            "tong_box": sum(tong_tk.values()),
            "so_anh_co_nha_ngap": sum(v["so_anh_co_nha_ngap"]
                                      for v in theo_split_tk.values()),
            "so_sat_mep": sum(v["so_sat_mep"] for v in theo_split_tk.values()),
        }

        # ---------------------------------------------------------------
        # Bước 4: kiểm tra COCO + đối chiếu Phase 1
        # ---------------------------------------------------------------
        print("\n--- BƯỚC 4: KIỂM TRA ---")
        ds_loi_coco = []
        for split in ds_split:
            duong_json = os.path.join(thu_muc_ann, f"instances_{split}.json")
            with open(duong_json, encoding="utf-8") as f:
                ds_loi_coco += [f"{split}: {t}"
                                for t in kiem_tra_coco(json.load(f), classes)]
        print(f"  Kiểm cấu trúc COCO: {'ĐẠT' if not ds_loi_coco else f'{len(ds_loi_coco)} LỖI'}")
        for t in ds_loi_coco[:10]:
            print(f"    {t}")

        # pycocotools: kiểm bằng thư viện thật mà mmdet sẽ dùng lúc train
        pycocotools_ok = ""
        try:
            from pycocotools.coco import COCO
            for split in ds_split:
                duong_json = os.path.join(thu_muc_ann, f"instances_{split}.json")
                c = COCO(duong_json)
                pycocotools_ok += (f"{split}: {len(c.imgs)} ảnh/"
                                   f"{len(c.anns)} ann; ")
            print(f"  pycocotools: ĐỌC ĐƯỢC cả 3 file")
        except ImportError:
            pycocotools_ok = ""
            print("  pycocotools: không có (bỏ qua — Colab có sẵn, cài mmdet là có)")
        except Exception as e:                                  # noqa: BLE001
            ds_loi_coco.append(f"pycocotools đọc lỗi: {e}")
            pycocotools_ok = ""
            print(f"  pycocotools: LỖI — {e}")

        doi_chieu = None
        if max_images is None:
            doi_chieu = doi_chieu_voi_phase1(
                os.path.join(work_dir, "audit.jsonl"), classes, min_side_px,
                min_area, tong["dem"])
        if doi_chieu is None:
            print("  Đối chiếu Phase 1: BỎ QUA (không thấy audit.jsonl trong phiên này)")
        elif doi_chieu["khop"]:
            print(f"  Đối chiếu Phase 1: KHỚP HOÀN TOÀN — "
                  f"{dict(sorted(tong['dem'].items()))}")
        else:
            print(f"  Đối chiếu Phase 1: LỆCH {doi_chieu['lech']} — ĐỪNG TRAIN, "
                  f"gửi lại kết quả này")

        # ---------------------------------------------------------------
        # Bước 5: ảnh overlay GT
        # ---------------------------------------------------------------
        print("\n--- BƯỚC 5: ẢNH OVERLAY GT ---")
        thu_muc_overlay = os.path.join(report_dir, "overlay_gt")
        os.makedirs(thu_muc_overlay, exist_ok=True)
        ten_lop_theo_id = {k: v for k, v in classes.items()}
        tong_ve = 0
        for split in ds_split:
            duong_json = os.path.join(thu_muc_ann, f"instances_{split}.json")
            with open(duong_json, encoding="utf-8") as f:
                coco_doc = json.load(f)
            tong_ve += ve_overlay_gt(coco_doc, os.path.join(processed_dir, "images", split),
                                     thu_muc_overlay, ten_lop_theo_id,
                                     c2["num_overlays"], c2["seed"])
        print(f"  Đã vẽ {tong_ve} ảnh vào {thu_muc_overlay}/")

        # ---------------------------------------------------------------
        # Báo cáo
        # ---------------------------------------------------------------
        thong_ke = {
            "target_long_side": target_long_side,
            "min_side_px": min_side_px,
            "min_area": min_area,
            "theo_split": theo_split_tk,
            "tong": tong,
            "ds_loi": ds_loi,
            "loi_coco": ds_loi_coco,
            "pycocotools": pycocotools_ok,
        }
        duong_bao_cao = os.path.join(report_dir, "BUILD_REPORT.md")
        print("\n" + "=" * 72)
        print(ghi_bao_cao(duong_bao_cao, thong_ke, doi_chieu))

    print("\n" + "=" * 72)
    print("XONG. Gửi lại cho Claude:")
    print(f"  1. Nội dung {duong_bao_cao}")
    print(f"  2. Mở {thu_muc_overlay}/ xem bằng mắt, mô tả box có khớp nhà không")
    print("=" * 72)


if __name__ == "__main__":
    main()

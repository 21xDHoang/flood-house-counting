# -*- coding: utf-8 -*-
"""Test cho bước suy luận — CPU, KHÔNG cần mmdet/mmengine/torch.

Hai tầng:

  A. `floodcount.infer.dem` — logic THUẦN Python, test bằng dữ liệu giả dựng
     ngay trong tệp này. Đây là tầng đáng test thật nhất: nó quyết định CON SỐ
     cuối cùng của đồ án (đếm nhà ngập), mà sai ở đây thì không có bảng mAP nào
     phát hiện hộ.

  B. `scripts/du_doan.py` — soi bằng AST, cùng cách đã dùng cho `train.py`:
     script chỉ được import mmdet BÊN TRONG hàm (giữ cho tầng A test được trên
     máy sạch), và thứ tự các bước trong `main()` phải đúng — bản vá
     `torch.load` (§3.13) phải chạy TRƯỚC khi nạp checkpoint, và pipeline suy
     luận phải BỎ `LoadAnnotations` (giữ lại thì `LoadAnnotations` đòi
     `ann_info` không tồn tại trên ảnh mới và nổ ngay ảnh đầu).

Bốn thứ dễ sai im lặng mà tầng A khoá lại:

  1. `chi_muc_gt` phải NÉM RA khi file COCO không nhất quán (annotation trỏ
     tới ảnh không tồn tại / lớp không tồn tại). Bỏ qua im lặng thì số nhãn
     thật thấp giả, và bảng đối chiếu "mô hình đếm 5, nhãn thật 3" hoá ra sai
     theo hướng ngược — đổ oan cho mô hình.

  2. `dem_theo_lop` so ngưỡng bằng `>=` (box đúng bằng ngưỡng được giữ) và
     NÉM RA khi nhãn nằm ngoài danh sách lớp. Nhãn lạ mà bị bỏ qua im lặng thì
     tổng số đếm thiếu mà không ai biết.

  3. `chon_anh_demo` chia đôi ảnh có/không có nhà ngập. Lấy "N ảnh đầu" thì
     với FloodNet (~10,5% ảnh có nhà ngập) rất dễ ra toàn ảnh không nhà —
     người xem overlay không biết mô hình có tìm được nhà ngập hay không.

  4. `tim_checkpoint` phải chọn tệp best có epoch LỚN NHẤT khi có nhiều tệp
     (resume nhiều vòng để lại nhiều `best_*`), và đỡ được `last_checkpoint`
     ghi đường dẫn tuyệt đối của máy khác (trên Colab) — tệp không còn thì
     ghép tên tệp với work_dir hiện tại.

Chạy:
    py tests/test_du_doan.py
"""

import ast
import importlib.util
import json
import pathlib
import sys
import tempfile

GOC_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC_REPO / "src"))

from floodcount.infer.dem import (chi_muc_gt, chon_anh_demo,  # noqa: E402
                                  dem_theo_lop, liet_ke_anh,
                                  ten_lop_theo_id, tim_checkpoint)

TEN_LOP = ["flooded_building", "non_flooded_building"]


def _coco_gia():
    """COCO 5 ảnh: a có 2 ngập + 1 không, b có 1 ngập, c/d có 1 không, e rỗng.

    Cố ý để 3 ảnh không có nhà ngập và 2 ảnh có — lệch về phía giống dữ liệu
    thật (10,5% ảnh có nhà ngập), đủ để thử nhánh "chia đôi" của `chon_anh_demo`.
    """
    return {
        "images": [{"id": i, "file_name": f"{t}.jpg"}
                   for i, t in enumerate("abcde", 1)],
        "categories": [
            {"id": 1, "name": "flooded_building"},
            {"id": 2, "name": "non_flooded_building"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 1},
            {"image_id": 1, "category_id": 1},
            {"image_id": 1, "category_id": 2},
            {"image_id": 2, "category_id": 1},
            {"image_id": 3, "category_id": 2},
            {"image_id": 4, "category_id": 2},
        ],
    }


def nap_du_doan():
    """Nạp scripts/du_doan.py như một module, không chạy `main()`."""
    duong = GOC_REPO / "scripts" / "du_doan.py"
    spec = importlib.util.spec_from_file_location("du_doan_duoi_test", duong)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nap_web_du_doan():
    """Nạp scripts/web_du_doan.py như một module, không chạy `main()`.

    Nạp được ở máy KHÔNG cài gradio chính là phép kiểm: gradio phải được
    import BÊN TRONG hàm, nếu không thì tệp này nổ ngay khi import.
    """
    duong = GOC_REPO / "scripts" / "web_du_doan.py"
    spec = importlib.util.spec_from_file_location("web_du_doan_duoi_test", duong)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _duong_anh_cua_ham(cay, ten_ham, ten_tu_khoa):
    """Các dòng gọi hàm có tên chứa `ten_tu_khoa` bên trong `ten_ham`.

    Dùng để khoá THỨ TỰ: vá torch.load phải đứng trước bước nạp checkpoint.
    """
    than = next((n for n in ast.walk(cay)
                 if isinstance(n, ast.FunctionDef) and n.name == ten_ham), None)
    if than is None:
        return None
    ra = []
    for nut in ast.walk(than):
        if isinstance(nut, ast.Call):
            ten = ast.unparse(nut.func)
            if ten_tu_khoa in ten:
                ra.append(nut.lineno)
    return ra


def main():
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    tmp = tempfile.mkdtemp(prefix="du_doan_test_")
    loi = []

    def kiem(dieu_kien, mo_ta):
        trang_thai = "OK  " if dieu_kien else "FAIL"
        print(f"  [{trang_thai}] {mo_ta}")
        if not dieu_kien:
            loi.append(mo_ta)

    try:
        coco = _coco_gia()

        print("=== A1. ten_lop_theo_id doc dung tu JSON ===")
        kiem(ten_lop_theo_id(coco) == {1: "flooded_building",
                                       2: "non_flooded_building"},
             "map category_id -> ten lop dung")
        kiem(ten_lop_theo_id({}) == {}, "JSON khong co categories -> {}")

        print("\n=== A2. chi_muc_gt dem dung nhan that ===")
        gt = chi_muc_gt(coco)
        kiem(gt.get("a.jpg") == {"flooded_building": 2,
                                 "non_flooded_building": 1},
             "a.jpg: 2 ngap + 1 khong ngap")
        kiem(gt.get("b.jpg") == {"flooded_building": 1}, "b.jpg: 1 ngap")
        kiem(gt.get("c.jpg") == {"non_flooded_building": 1},
             "c.jpg: 1 khong ngap")
        kiem("e.jpg" not in gt, "anh khong co box KHONG nam trong chi muc")

        coco_hong = _coco_gia()
        coco_hong["annotations"] = [{"image_id": 99, "category_id": 1}]
        try:
            chi_muc_gt(coco_hong)
            kiem(False, "annotation tro anh khong ton tai -> phai nem ra")
        except ValueError as e:
            kiem("99" in str(e), "annotation tro anh khong ton tai -> nem ra")

        coco_hong = _coco_gia()
        coco_hong["annotations"] = [{"image_id": 1, "category_id": 7}]
        try:
            chi_muc_gt(coco_hong)
            kiem(False, "category_id la -> phai nem ra")
        except ValueError as e:
            kiem("7" in str(e), "category_id la -> nem ra")

        print("\n=== A3. dem_theo_lop: nguong >= va nhan la ===")
        kiem(dem_theo_lop([0, 0, 1], [0.9, 0.5, 0.8], TEN_LOP, 0.5)
             == {"flooded_building": 2, "non_flooded_building": 1},
             "diem dung bang nguong (0,5) duoc GIU")
        kiem(dem_theo_lop([0, 1], [0.49, 0.3], TEN_LOP, 0.5)
             == {"flooded_building": 0, "non_flooded_building": 0},
             "duoi nguong -> khong dem")
        kiem(dem_theo_lop([], [], TEN_LOP, 0.5)
             == {"flooded_building": 0, "non_flooded_building": 0},
             "khong co box -> 0 ca hai lop")
        try:
            dem_theo_lop([2], [0.9], TEN_LOP, 0.5)
            kiem(False, "nhan ngoai danh sach lop -> phai nem ra")
        except ValueError:
            kiem(True, "nhan ngoai danh sach lop -> nem ra")
        kiem(dem_theo_lop([1.0], [0.7], TEN_LOP, 0.5)
             == {"flooded_building": 0, "non_flooded_building": 1},
             "nhan dang float (1.0) van dem dung lop 2")

        print("\n=== A4. chon_anh_demo: chia doi co/khong ngap ===")
        chon = chon_anh_demo(coco, 4, "flooded_building", seed=42)
        kiem(len(chon) == 4, "tra dung so anh xin (4)")
        gt4 = chi_muc_gt(coco)
        so_ngap = sum(1 for t in chon
                      if gt4.get(t, {}).get("flooded_building"))
        kiem(so_ngap == 2, "4 anh -> dung 2 anh CO nha ngap")
        kiem(chon_anh_demo(coco, 4, "flooded_building", seed=42) == chon,
             "cung seed -> cung danh sach (so sanh 2 checkpoint duoc)")
        kiem(len(set(chon)) == len(chon), "khong trung anh")
        kiem(len(chon_anh_demo(coco, 99, "flooded_building", seed=1)) == 5,
             "xin nhieu hon tong so anh -> tra het, khong no")
        kiem(len(chon_anh_demo(coco, 5, "flooded_building", seed=7)) == 5,
             "xin 5 anh (chi co 5) -> tra du 5")

        print("\n=== A5. liet_ke_anh: dung duoi, dung thu tu ===")
        thu_muc = pathlib.Path(tmp) / "anh"
        thu_muc.mkdir()
        for ten in ("b.JPG", "a.jpg", "c.png", "ghi_chu.txt", "d.webp"):
            (thu_muc / ten).write_bytes(b"x")
        (thu_muc / "thu_muc_con").mkdir()
        (thu_muc / "thu_muc_con" / "e.jpg").write_bytes(b"x")
        ds = liet_ke_anh(thu_muc)
        kiem([pathlib.Path(d).name for d in ds]
             == ["a.jpg", "b.JPG", "c.png", "d.webp"],
             "nhan .jpg/.JPG/.png/.webp, bo .txt va thu muc con")
        kiem(liet_ke_anh(thu_muc)[0].startswith(str(thu_muc)),
             "tra duong dan day du")
        kiem(liet_ke_anh(thu_muc / "thu_muc_con")
             == [str(thu_muc / "thu_muc_con" / "e.jpg")],
             "thu muc con: chi thay anh cua chinh no")

        print("\n=== A6. tim_checkpoint: best moi nhat, roi last_checkpoint ===")
        kiem(tim_checkpoint(pathlib.Path(tmp) / "khong_ton_tai") is None,
             "work_dir khong ton tai -> None")
        wd = pathlib.Path(tmp) / "wd"
        wd.mkdir()
        kiem(tim_checkpoint(wd) is None, "work_dir rong -> None")
        for ten in ("best_coco_bbox_mAP_epoch_9.pth",
                    "best_coco_bbox_mAP_epoch_26.pth",
                    "best_coco_bbox_mAP_epoch_14.pth",
                    "epoch_60.pth"):
            (wd / ten).write_bytes(b"x")
        kiem(pathlib.Path(tim_checkpoint(wd)).name
             == "best_coco_bbox_mAP_epoch_26.pth",
             "nhieu best -> chon epoch LON NHAT (26)")

        wd2 = pathlib.Path(tmp) / "wd2"
        wd2.mkdir()
        (wd2 / "epoch_60.pth").write_bytes(b"x")
        (wd2 / "last_checkpoint").write_text(
            "/content/drive/MyDrive/Flood_House_AI/runs/train/e1/epoch_60.pth",
            encoding="utf-8")
        kiem(pathlib.Path(tim_checkpoint(wd2)).name == "epoch_60.pth",
             "last_checkpoint tro duong dan Colab khong con -> ghep ten tep "
             "voi work_dir hien tai")
        kiem(pathlib.Path(tim_checkpoint(wd2)).parent == wd2,
             "duong dan tra ve nam TRONG work_dir")

        wd3 = pathlib.Path(tmp) / "wd3"
        wd3.mkdir()
        (wd3 / "last_checkpoint").write_text("/khong/con/gi.pth",
                                             encoding="utf-8")
        kiem(tim_checkpoint(wd3) is None,
             "last_checkpoint tro mo -> None (khong tra ve duong dan chet)")

        print("\n=== B1. scripts/du_doan.py KHONG keo thu vien nang vao ===")
        du_doan = nap_du_doan()
        nang = [m for m in sys.modules
                if m.split(".")[0] in ("mmdet", "mmcv", "mmengine", "torch")]
        kiem(not nang,
             f"import du_doan.py khong keo mmdet/mmcv/mmengine/torch, "
             f"thuc te {nang}")

        print("\n=== B2. thu tu trong main(): va torch.load TRUOC nap checkpoint ===")
        cay = ast.parse((GOC_REPO / "scripts" / "du_doan.py")
                        .read_text(encoding="utf-8"))
        # `init_detector` nam trong `nap_model` (de web dung chung), con `main`
        # phai goi ban va TRUOC `nap_model`.
        dong_init = _duong_anh_cua_ham(cay, "nap_model", "init_detector")
        kiem(dong_init is not None and len(dong_init) >= 1,
             "nap_model() goi init_detector()")
        dong_va = _duong_anh_cua_ham(cay, "main", "va_torch_load_resume")
        dong_nap = _duong_anh_cua_ham(cay, "main", "nap_model")
        kiem(dong_va is not None and len(dong_va) >= 1,
             "main() goi va_torch_load_resume()")
        kiem(dong_nap is not None and len(dong_nap) >= 1,
             "main() goi nap_model()")
        if dong_va and dong_nap:
            kiem(max(dong_va) < min(dong_nap),
                 "va_torch_load_resume dung TRUOC nap_model (neu nguoc lai: "
                 "_pickle.UnpicklingError tren torch >= 2.6, §3.13)")
        kiem(any("_train" in ast.unparse(n.func) or "train" in ast.unparse(n.func)
                 for n in ast.walk(cay)
                 if isinstance(n, ast.Call)
                 and "va_torch_load_resume" in ast.unparse(n.func)),
             "ban va duoc LAY TU train.py (khong chep lai)")

        print("\n=== B3. pipeline suy luan BO LoadAnnotations ===")
        ten_ham = [n.name for n in ast.walk(cay)
                   if isinstance(n, ast.FunctionDef)]
        kiem("pipeline_suy_luan" in ten_ham,
             "co ham pipeline_suy_luan (de test khoa duoc hanh vi)")
        if "pipeline_suy_luan" in ten_ham:
            than = next(n for n in ast.walk(cay)
                        if isinstance(n, ast.FunctionDef)
                        and n.name == "pipeline_suy_luan")
            # Phép kiểm chính: LoadAnnotations phải bị LOẠI bằng `!=` — tức nó
            # đứng ở vế bị so sánh, không nằm trong danh sách giữ lại. Giữ nó
            # lại thì LoadAnnotations đòi `ann_info` mà ảnh mới không có.
            loi_la = [n for n in ast.walk(than)
                      if isinstance(n, ast.Compare)
                      and any(isinstance(o, ast.NotEq) for o in n.ops)
                      and any(isinstance(c, ast.Constant)
                              and c.value == "LoadAnnotations"
                              for c in [n.left, *n.comparators])]
            kiem(bool(loi_la),
                 "LoadAnnotations bi loai bang so sanh != (khong nam trong "
                 "danh sach giu lai)")
            kiem(any("Compose" in ast.unparse(n.func)
                     for n in ast.walk(than) if isinstance(n, ast.Call)),
                 "pipeline_suy_luan tra ve Compose (khop kieu mmengine doi)")

        print("\n=== B4. cac co phai co mat ===")
        co = set()
        for nut in ast.walk(cay):
            if isinstance(nut, ast.Call):
                for a in nut.args:
                    if isinstance(a, ast.Constant) and \
                            isinstance(a.value, str) and \
                            a.value.startswith("--"):
                        co.add(a.value)
        for ten_co in ("--config", "--data-yaml", "--checkpoint", "--anh-dir",
                       "--split", "--so-anh", "--nguong-diem", "--ra-dir"):
            kiem(ten_co in co, f"co {ten_co}")

        print("\n=== B5. dem di qua module da test (khong tu dem trong script) ===")
        goi_dem = [n for n in ast.walk(cay)
                   if isinstance(n, ast.Call)
                   and "dem_theo_lop" in ast.unparse(n.func)]
        kiem(bool(goi_dem),
             "script goi dem_theo_lop cua floodcount.infer.dem")

        print("\n=== B6. che do 'anh cua toi' chan thu muc rong bang loi ro rang ===")
        nguon = (GOC_REPO / "scripts" / "du_doan.py").read_text(encoding="utf-8")
        kiem("anh_cua_toi" in nguon,
             "co nhac thu muc anh_cua_toi (huong dan nguoi dung tha anh vao)")

        print("\n=== C1. scripts/web_du_doan.py KHONG keo thu vien nang vao ===")
        web = nap_web_du_doan()
        nang = [m for m in sys.modules
                if m.split(".")[0] in ("mmdet", "mmcv", "mmengine", "torch",
                                       "gradio")]
        kiem(not nang,
             f"import web_du_doan.py khong keo mmdet/mmcv/mmengine/torch/gradio, "
             f"thuc te {nang}")

        print("\n=== C2. web DUNG LAI loi suy luan cua du_doan.py ===")
        cay_web = ast.parse((GOC_REPO / "scripts" / "web_du_doan.py")
                            .read_text(encoding="utf-8"))
        goi_trong_web = {ast.unparse(n.func) for n in ast.walk(cay_web)
                         if isinstance(n, ast.Call)}
        for ten_goi in ("du_doan.chay_mot_mang", "du_doan.nap_model",
                        "du_doan.ve_anh", "du_doan.pipeline_suy_luan"):
            kiem(ten_goi in goi_trong_web, f"web goi {ten_goi} (khong chep lai)")

        print("\n=== C3. web: pipeline nhan mang numpy + launch khong chan o Colab ===")
        kiem(any(isinstance(n, ast.keyword) and n.arg == "dau_vao_bang_mang"
                 and isinstance(n.value, ast.Constant) and n.value.value is True
                 for n in ast.walk(cay_web)),
             "pipeline_suy_luan(dau_vao_bang_mang=True) — anh tu trinh duyet la "
             "mang numpy, khong co duong dan")
        launch = [n for n in ast.walk(cay_web)
                  if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "launch"]
        kiem(bool(launch), "co goi .launch()")
        if launch:
            tu_khoa = {k.arg: k.value for k in launch[0].keywords}
            kiem("prevent_thread_lock" in tu_khoa
                 and isinstance(tu_khoa["prevent_thread_lock"], ast.Constant)
                 and tu_khoa["prevent_thread_lock"].value is True,
                 "launch(prevent_thread_lock=True) — o Colab chay xong ngay, "
                 "trang web van song o luong nen")

        print("\n=== D1. notebook 03_train.ipynb: JSON hop le + CRLF nguyen ven ===")
        # Notebook bi sua bang cach chen thang vao JSON (giu diff nho). Chen hong
        # thi phai lo ra o day, chu khong phai luc mo Colab moi biet.
        duong_nb = GOC_REPO / "notebooks" / "03_train.ipynb"
        tho = duong_nb.read_bytes().decode("utf-8")
        kiem(tho.count("\n") == tho.count("\r\n"),
             "moi dong ket thuc bang CRLF (khong lan LF)")
        nb = json.loads(tho)
        kiem(nb["nbformat"] == 4, "json.loads duoc, nbformat = 4")
        nguon_o = ["".join(c["source"]) for c in nb["cells"]]

        print("\n=== D2. notebook co dung 1 o [3.12] va 1 o [3.13] ===")
        o_312 = [s for s in nguon_o if s.startswith("# [3.12]")]
        o_313 = [s for s in nguon_o if s.startswith("# [3.13]")]
        kiem(len(o_312) == 1, "co dung 1 o [3.12]")
        kiem(len(o_313) == 1, "co dung 1 o [3.13]")

        print("\n=== D3. o [3.12] tro dung thu muc anh + ghi ket qua ra Drive ===")
        if o_312:
            s312 = o_312[0]
            kiem('{DRIVE_DIR}/anh_cua_toi' in s312,
                 "thu muc anh la {DRIVE_DIR}/anh_cua_toi")
            kiem("--anh-dir" in s312 and "--ra-dir" in s312,
                 "goi du_doan.py voi --anh-dir va --ra-dir")
            kiem("overlay" in s312 and "IPython.display" in s312,
                 "hien anh da ve box ngay trong o bang IPython.display")

        print("\n=== D4. o [3.13] mo web --share, doc lai code tu dia ===")
        if o_313:
            s313 = o_313[0]
            kiem('"--share"' in s313,
                 "goi web.main voi --share (link cong khai)")
            kiem("spec_from_file_location" in s313,
                 "nap web_du_doan.py bang importlib — doc lai tu dia moi lan "
                 "chay, nen sau git pull la chay dung code moi")
            kiem("REPO / CFG_CHINH" in s313 and "REPO / DATA_YAML" in s313,
                 "truyen duong dan TUYET DOI cho config (cwd cua kernel la "
                 "/content, khong phai REPO_DIR)")

        print("\n=== D5. moi o code trong notebook deu compile duoc ===")
        so_o_code, o_hong = 0, []
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            so_o_code += 1
            try:
                compile("".join(c["source"]), f"o thu {i}", "exec")
            except SyntaxError as e:
                o_hong.append(f"o {i}: {e}")
        kiem(so_o_code >= 15, f"notebook co {so_o_code} o code")
        kiem(not o_hong, f"khong o code nao loi cu phap {o_hong}")

        print("\n=== D6. o [3.3] kiem ca 2 script moi (code phai duoc push) ===")
        s33 = next((s for s in nguon_o if s.startswith("# [3.3]")), "")
        kiem('"scripts/du_doan.py"' in s33 and '"scripts/web_du_doan.py"' in s33,
             "CAN_CO cua o [3.3] co du 2 script moi — thieu la Colab bao ngay")

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


if __name__ == "__main__":
    main()

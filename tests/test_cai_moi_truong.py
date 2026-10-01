# -*- coding: utf-8 -*-
"""Test cho scripts/cai_moi_truong.py — CPU, không cần torch/mmcv/mmdet.

Vì sao phải test phần này: đây là **cửa vào duy nhất** của cả đồ án trên Colab.
Script hỏng thì không cài được gì, mà lỗi của nó lại không lộ ra trên máy CPU —
máy CPU không có torch để mà chạy thử phần cài. Nên test bám vào đúng những
chỗ đã từng sai hoặc sẽ sai im lặng:

  1. **Bẫy `%2B` trong tên wheel.** Regex phải bắt phần chữ hiển thị của thẻ
     `<a>`; bắt trong `href` thì dấu `+` đã thành `%2B` và tên bị cắt cụt mất
     đúng hậu tố quyết định bản dựng (`+a8073c7pt2.11.0cu128`). Sai chỗ này thì
     pip vẫn chạy, chỉ là chọn nhầm wheel.

  2. **Bẫy khớp tiền tố.** `pt2.1.1` là tiền tố của `pt2.1.10` (torch 2.1.10) và
     `cp31` là tiền tố của `cp313`. Khớp trần là chọn nhầm wheel của tổ hợp khác.

  3. **Ba miếng vá chạy trên file giả**: mmdet (nới ngưỡng mmcv), mmengine (vá
     đúng MỘT trong bốn hàm cùng chứa dòng y hệt nhau), mmpretrain (tắt cờ đa
     phương thức). Vá nhầm hàm hay vá hụt đều không báo lỗi gì.

  4. **Không được import torch ở cấp module.** Có torch thì test này không chạy
     nổi trên máy sạch, mà máy sạch mới là nơi cần chạy test.

Chạy:
    py tests/test_cai_moi_truong.py
"""

import ast
import importlib.util
import pathlib
import subprocess
import sys
import tempfile

GOC_REPO = pathlib.Path(__file__).resolve().parents[1]
DUONG_SCRIPT = GOC_REPO / "scripts" / "cai_moi_truong.py"

# Nạp module theo ĐƯỜNG DẪN chứ không `import`: `scripts/` không phải gói Python
# và tên file có dấu tiếng Việt.
_spec = importlib.util.spec_from_file_location("cai_moi_truong", DUONG_SCRIPT)
cmt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmt)


# --- Dữ liệu giả dùng chung -------------------------------------------------
# Tên wheel y như thật: mmcv-<bản>+<hash>pt<torch>cu<cuda>-<py>-<abi>-<nền tảng>.whl
W_CP313 = "mmcv-2.2.0+a8073c7pt2.11.0cu128-cp313-cp313-linux_x86_64.whl"
W_CP312 = "mmcv-2.2.0+a8073c7pt2.11.0cu128-cp312-cp312-linux_x86_64.whl"
W_WIN = "mmcv-2.2.0+a8073c7pt2.11.0cu128-cp313-cp313-win_amd64.whl"

HTML_INDEX = f"""
<a href="mmcv-2.2.0%2Ba8073c7pt2.11.0cu128-cp313-cp313-linux_x86_64.whl">{W_CP313}</a>
<a href="mmcv-2.2.0%2Ba8073c7pt2.11.0cu128-cp312-cp312-linux_x86_64.whl">{W_CP312}</a>
<a href="mmcv-2.2.0%2Ba8073c7pt2.11.0cu128-cp313-cp313-win_amd64.whl">{W_WIN}</a>
"""

BUILDER_MMENGINE = '''\
def register_torch_optimizers() -> List[str]:
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if issubclass(_optim, Optimizer):
            OPTIMIZERS.register_module(module=_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


def ham_khac() -> None:
    """Ham nay co dong Y HET ham tren, de bat loi va nham ham."""
    OPTIMIZERS.register_module(module=_optim)


def register_transformers_optimizers() -> List[str]:
    transformer_optimizers = []
    for module_name in dir(transformers.optimization):
        _optim = getattr(transformers.optimization, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, Optimizer):
            OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)
            transformer_optimizers.append('Adafactor')
    return transformer_optimizers
'''

DEPENDENCY_MMPRETRAIN = """\
import importlib
digit_version = None

WITH_MULTIMODAL = all(
    satisfy_requirement(item)
    for item in ['pycocotools', 'transformers>=4.28.0'])


def check_metainfo():
    pass
"""


def main():
    # Console Windows mặc định là cp1252: in chữ có dấu là nổ UnicodeEncodeError
    # giữa chừng. Đổi sang UTF-8 ngay từ đầu (Colab vốn đã là UTF-8).
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    loi = []

    def kiem(dieu_kien, mo_ta):
        trang_thai = "OK  " if dieu_kien else "FAIL"
        print(f"  [{trang_thai}] {mo_ta}")
        if not dieu_kien:
            loi.append(mo_ta)

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="cai_moi_truong_"))

    def ghi(ten, noi_dung):
        f = tmp / ten
        f.write_text(noi_dung, encoding="utf-8")
        return f

    # ======================================================================
    print("=== 1. Ten wheel lay tu trang index (bay %2B) ===")
    ten = cmt.ten_wheel_tu_html(HTML_INDEX)
    kiem(len(ten) == 3, f"lay du 3 ten wheel, thuc te {len(ten)}")
    # Nếu regex bắt nhầm trong href, tên sẽ cụt ở "mmcv-2.2.0" vì dấu % chặn
    # giữa. Có "+a8073c7" nghĩa là đã bắt đúng phần chữ hiển thị.
    kiem(all("+a8073c7" in t for t in ten),
         "ten wheel giu nguyen hau to '+a8073c7...' (khong bi cat o %2B)")
    kiem(all(t.endswith(".whl") for t in ten), "moi ten deu ket thuc bang .whl")
    kiem(ten == sorted(ten), "danh sach da sap xep (de so sanh, on dinh)")

    # ======================================================================
    print("\n=== 2. Chon wheel theo to hop Python / torch / CUDA ===")
    kiem(cmt.chon_wheel(ten, "cp313", "2.11.0", "cu128") == W_CP313,
         "chon dung wheel cp313 + pt2.11.0 + cu128")
    kiem(cmt.chon_wheel(ten, "cp312", "2.11.0", "cu128") == W_CP312,
         "doi Python -> chon wheel khac")
    kiem(cmt.chon_wheel(ten, "cp313", "2.11.0", "cu121") is None,
         "khong co wheel cho cu121 -> tra ve None (khong doan bua)")
    kiem(cmt.chon_wheel(ten, "cp313", "2.10.0", "cu128") is None,
         "khong co wheel cho torch 2.10.0 -> tra ve None")
    # win_amd64 co dung cp313 + pt2.11.0cu128 nhung sai nen tang:
    kiem(cmt.chon_wheel([W_WIN], "cp313", "2.11.0", "cu128") is None,
         "wheel win_amd64 khong bi chon (loc theo nen tang linux)")

    print("\n--- Bay khop TIEN TO (khong doi gach noi la dinh) ---")
    w_torch_2110 = "mmcv-2.2.0+a8073c7pt2.1.10cu121-cp311-cp311-linux_x86_64.whl"
    kiem(cmt.chon_wheel([w_torch_2110], "cp311", "2.1.1", "cu121") is None,
         "torch 2.1.1 KHONG khop nham vao wheel cua torch 2.1.10")
    w_torch_211 = "mmcv-2.2.0+a8073c7pt2.1.1cu121-cp311-cp311-linux_x86_64.whl"
    kiem(cmt.chon_wheel([w_torch_211], "cp311", "2.1.1", "cu121") == w_torch_211,
         "torch 2.1.1 van chon dung wheel cua chinh no")
    w_py_3131 = "mmcv-2.2.0+a8073c7pt2.11.0cu128-cp3131-cp3131-linux_x86_64.whl"
    kiem(cmt.chon_wheel([w_py_3131], "cp31", "2.11.0", "cu128") is None,
         "py tag 'cp31' KHONG khop nham vao 'cp313'")

    print("\n--- Danh sach in ra khi khong tim duoc wheel ---")
    # Danh sach nay in ra cho nguoi doc gui lai khi Colab doi torch/CUDA, ma Colab
    # la Linux -> phai loc ca nen tang, va phai loc dung doi Python.
    co = cmt.cac_ban_cho_python(ten, "cp313")
    kiem(co == [W_CP313], f"chi con wheel cp313 dung nen tang linux, thuc te {co}")
    kiem(W_WIN not in co, "wheel win_amd64 khong lot vao danh sach (Colab la Linux)")
    kiem(W_CP312 not in co, "wheel cua doi Python khac khong lot vao")
    kiem(cmt.cac_ban_cho_python(ten, "cp39") == [],
         "doi Python khong co ban nao -> danh sach rong")

    # ======================================================================
    print("\n=== 3. Tach phien ban tu ten wheel ===")
    kiem(cmt.phien_ban_tu_ten_wheel(W_CP313, "cp313")
         == "2.2.0+a8073c7pt2.11.0cu128",
         "giu nguyen hau to '+a8073c7pt2.11.0cu128' (pip can chuoi nay)")
    kiem(cmt.phien_ban_tu_ten_wheel(W_CP312, "cp312")
         == "2.2.0+a8073c7pt2.11.0cu128",
         "tach dung voi py tag khac")

    # ======================================================================
    print("\n=== 4. Va mmdet: noi nguong mmcv 2.2.0 -> 2.3.0 ===")
    f = ghi("mmdet_init.py", "mmcv_maximum_version = '2.2.0'\n__version__ = '3.3.0'\n")
    kq = cmt.va_mmdet(f)
    kiem(kq.startswith("đã vá"), f"lan dau: {kq}")
    noi_dung = f.read_text(encoding="utf-8")
    kiem("mmcv_maximum_version = '2.3.0'" in noi_dung, "nguong da noi len 2.3.0")
    kiem("'2.2.0'" not in noi_dung, "khong con dau vet nguong cu")
    kiem("__version__ = '3.3.0'" in noi_dung, "khong dung den phan khac cua file")
    kiem(cmt.va_mmdet(f) == "đã vá trước đó", "chay lan hai: khong va lai")

    f = ghi("mmdet_la.py", "# khong co dong nao giong\n")
    kiem(cmt.va_mmdet(f).startswith("KHÔNG tìm thấy dòng cần vá"),
         "file khac phien ban -> bao KHONG tim thay, khong sua bua")
    kiem(f.read_text(encoding="utf-8") == "# khong co dong nao giong\n",
         "file khong bi sua khi khong khop mau")
    kiem(cmt.va_mmdet(tmp / "khong_ton_tai.py").startswith("KHÔNG thấy file"),
         "file khong ton tai -> bao ro, khong nem loi ra giua script")

    # ======================================================================
    print("\n=== 5. Va mmengine: va dung MOT trong bon ham cung ten dong ===")
    f = ghi("builder.py", BUILDER_MMENGINE)
    kq = cmt.va_mmengine_neu_can(f)
    kiem(kq.startswith("đã vá 2 chỗ"), f"va dung 2 cho (khong phai 3): {kq}")
    s = f.read_text(encoding="utf-8")

    # Cho 1: chi dong trong register_torch_optimizers duoc va.
    dong_torch = s.split("def ham_khac")[0]
    dong_khac = s.split("def ham_khac")[1].split("def register_transformers")[0]
    kiem("if module_name in OPTIMIZERS:" in dong_torch
         and dong_torch.count("if module_name in OPTIMIZERS:") == 1,
         "dong trong register_torch_optimizers duoc boc dung 1 lan")
    kiem("if module_name in OPTIMIZERS:" not in dong_khac,
         "ham KHAC co dong y het -> KHONG bi va (bay theo doi ten ham)")

    # Cho 2: Adafactor duoc boc, va dong append bi thut vao trong khoi if.
    # Kiem bang muc thut chu KHONG bang chuoi co dinh: file that cua mmengine
    # thut 12 space (trong `for` + `if`), con file gia o day cung vay — nhung
    # viet chuoi cung la tu buoc minh vao mot muc thut cu the.
    cac_dong = s.splitlines()

    def thut(dong):
        return len(dong) - len(dong.lstrip())

    i_if = next(i for i, l in enumerate(cac_dong)
                if "'Adafactor' not in OPTIMIZERS" in l)
    i_dk = next(i for i, l in enumerate(cac_dong)
                if "register_module(name='Adafactor'" in l)
    i_ap = next(i for i, l in enumerate(cac_dong)
                if "transformer_optimizers.append('Adafactor')" in l)
    kiem(i_dk == i_if + 1 and i_ap == i_if + 2,
         "hai dong nam NGAY SAU lenh if (khong bi bo quen dong append)")
    kiem(thut(cac_dong[i_dk]) == thut(cac_dong[i_if]) + 4,
         "dong register duoc thut vao trong khoi if")
    kiem(thut(cac_dong[i_ap]) == thut(cac_dong[i_dk]),
         "dong append cung muc thut voi dong register (nam trong khoi if)")
    kiem(cmt.va_mmengine_neu_can(f) == "đã vá trước đó", "chay lan hai: khong va lai")

    f = ghi("builder_moi.py",
            "# mmengine 0.10.7: da doi ten thanh TorchAdafactor\n"
            "OPTIMIZERS.register_module(name='TorchAdafactor', module=_optim)\n")
    kiem(cmt.va_mmengine_neu_can(f).startswith("không cần"),
         "ban da sua o upstream -> khong dung tới file")

    f = ghi("builder_la.py", "def khac():\n    pass\n")
    kiem(cmt.va_mmengine_neu_can(f).startswith("KHÔNG vá được"),
         "khong khop mau -> bao ro de nguoi doc xem lai")

    # ======================================================================
    print("\n=== 6. Va mmpretrain: tat co da phuong thuc ===")
    f = ghi("dependency.py", DEPENDENCY_MMPRETRAIN)
    kq = cmt.va_mmpretrain_da_phuong_thuc(f)
    kiem(kq.startswith("đã vá"), f"lan dau: {kq}")
    s = f.read_text(encoding="utf-8")
    kiem("WITH_MULTIMODAL = False" in s, "co da bi tat han")
    kiem("satisfy_requirement(item)" not in s, "khoi cu da bi go han")
    kiem("def check_metainfo():" in s, "phan con lai cua file con nguyen")
    kiem(cmt.va_mmpretrain_da_phuong_thuc(f) == "đã vá trước đó",
         "chay lan hai: khong va lai")

    f = ghi("dependency_la.py", "WITH_MULTIMODAL = True\n")
    kiem(cmt.va_mmpretrain_da_phuong_thuc(f).startswith("KHÔNG tìm thấy đoạn cần vá"),
         "file khac phien ban -> bao KHONG tim thay, khong sua bua")
    kiem(cmt.va_mmpretrain_da_phuong_thuc(tmp / "khong_co.py")
         .startswith("KHÔNG thấy file"),
         "file khong ton tai -> bao ro")

    # ======================================================================
    print("\n=== 7. va_ca_ba: gom 3 mieng va, bao loi neu bat ky mieng nao hong ===")
    ok_mmdet = ghi("ok_mmdet.py", "mmcv_maximum_version = '2.2.0'\n")
    ok_mmengine = ghi("ok_mmengine.py", BUILDER_MMENGINE)
    ok_mmpretrain = ghi("ok_mmpretrain.py", DEPENDENCY_MMPRETRAIN)
    ma, dong = cmt.va_ca_ba(ok_mmdet, ok_mmengine, ok_mmpretrain)
    kiem(ma == 0, f"ca ba va duoc -> ma thoat 0, thuc te {ma}")
    kiem(len(dong) == 3 and all(d.startswith("vá ") for d in dong),
         "tra ve du 3 dong bao cao, moi dong bat dau bang 'vá '")
    kiem(all("KHÔNG" not in d for d in dong), "khong dong nao bao loi")

    ma, dong = cmt.va_ca_ba(tmp / "thieu.py", ok_mmengine, ok_mmpretrain)
    kiem(ma == 1, f"mot mieng va hong -> ma thoat 1, thuc te {ma}")
    kiem(sum("KHÔNG" in d for d in dong) == 1, "bao dung 1 dong loi")
    kiem(len(dong) == 3, "VAN in du 3 dong: nguoi doc thay cai nao xong, cai nao chua")

    # ======================================================================
    print("\n=== 8. Script chay duoc that, va khong keo torch vao ===")
    kq = subprocess.run([sys.executable, str(DUONG_SCRIPT), "--xem-phien-ban"],
                        capture_output=True, text=True, encoding="utf-8",
                        cwd=str(GOC_REPO))
    kiem(kq.returncode == 0, f"--xem-phien-ban chay duoc, ma thoat {kq.returncode}")
    kiem("0.10.7" in kq.stdout and "1.2.0" in kq.stdout,
         "in ra dung cac phien ban dang ghim")
    kiem("mmcv" in kq.stdout and "(dò theo runtime)" in kq.stdout,
         "noi ro mmcv khong ghim cung ma do theo runtime")

    kq_hong = subprocess.run([sys.executable, str(DUONG_SCRIPT), "--khong-co-co-nay"],
                             capture_output=True, text=True, encoding="utf-8",
                             cwd=str(GOC_REPO))
    kiem(kq_hong.returncode != 0, "co sai -> ma thoat khac 0 (khong im lang bo qua)")

    cay = ast.parse(DUONG_SCRIPT.read_text(encoding="utf-8"))
    import_cap_module = set()
    for nut in cay.body:
        if isinstance(nut, ast.Import):
            import_cap_module.update(a.name.split(".")[0] for a in nut.names)
        elif isinstance(nut, ast.ImportFrom) and nut.module:
            import_cap_module.add(nut.module.split(".")[0])
    kiem("torch" not in import_cap_module,
         "khong import torch o cap module (may CPU phai chay duoc script)")
    trong_ham = [n for n in ast.walk(cay)
                 if isinstance(n, ast.Import)
                 and any(a.name.split(".")[0] == "torch" for a in n.names)]
    kiem(bool(trong_ham), "van co 'import torch' ben trong ham cai dat (dung y do)")

    print("\n=== 9. Phien ban ghim phai khop tai lieu (docs/NOTES.md muc 1) ===")
    # Doi cac so nay thi phai sua ca NOTES + bang o notebook 00 — test dung day
    # de khong the doi lung.
    for ten_hang, gia_tri in [("MMDET_VERSION", "3.3.0"),
                              ("MMENGINE_VERSION", "0.10.7"),
                              ("MMPRETRAIN_VERSION", "1.2.0"),
                              ("PYCOCOTOOLS_VERSION", "2.0.11")]:
        kiem(getattr(cmt, ten_hang) == gia_tri,
             f"{ten_hang} = {gia_tri} nhu tai lieu")
    kiem(cmt.INSTALL_MARKER == "/content/.floodcount_env_ready",
         "duong dan tep danh dau nhu cu (notebook 00 va 03 deu dua vao)")

    print()
    if loi:
        print(f"*** {len(loi)} MUC KHONG DAT ***")
        for m in loi:
            print("   - " + m.encode("ascii", "backslashreplace").decode())
        raise SystemExit(1)
    print("*** TAT CA PASS ***")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cài môi trường MMDetection cho Colab — notebook 00 và notebook 03 dùng CHUNG script này.

VÌ SAO PHẢI LÀ SCRIPT RIÊNG, KHÔNG ĐỂ TRONG NOTEBOOK

Trên Colab, một notebook = một tab = một runtime, mà bản miễn phí chỉ cho một
GPU một lúc. Phần cài đặt vốn nằm trong ô [0.6] của notebook 00, nên muốn train
thì phải mở thêm tab thứ hai chỉ để cài — mở tab đó là dựng runtime mới, và
phiên GPU đang chạy ở tab kia bị giành mất. Đưa logic vào repo thì notebook 03
clone repo ở ô [3.3] rồi gọi thẳng script này: mở MỘT tab, chạy từ trên xuống.

Cài 4 gói rồi vá 3 chỗ (chi tiết và nguồn: docs/NOTES.md mục 1.2):

  1. `mmcv` bản CÓ CUDA OPS (không phải `mmcv-lite`) — dò wheel khớp tổ hợp
     Python / torch / CUDA của runtime, KHÔNG hard-code tên file: Colab nâng
     cấp torch là tên đổi theo. Lấy đúng wheel thì xong trong ~1 phút; lấy sai
     thì pip quay ra tải bản source và tự biên dịch — 30–60 phút và thường hỏng.
  2. `mmengine`, `mmdet`, `mmpretrain`, `pycocotools` từ PyPI.
  3. Vá `mmdet/__init__.py`: mmdet 3.3.0 assert `mmcv < 2.2.0` (dấu `<` NGHIÊM
     NGẶT) nên cài đúng bản 2.2.0 vẫn trượt. mmcv 2.2.0 chỉ là bản sửa lỗi so
     với 2.1.0, API không đổi -> nới lên 2.3.0 là an toàn.
  4. Vá `mmengine`: bản 0.10.4 đăng ký trùng tên `Adafactor` -> `KeyError` khi
     import. Bản 0.10.7 đã sửa ở upstream; hàm vá ở đây chỉ để nếu pip kéo về
     bản cũ thì tự xử.
  5. Vá `mmpretrain`: tắt nhánh "đa phương thức" (BLIP, BLIP2, LLaVA, OFA...)
     viết cho transformers 4.x — Colab có 5.x nên nhánh đó nổ `TypeError` và
     làm sập cả gói, kéo theo backbone ConvNeXt mà đồ án bắt buộc phải có.

KHÔNG hạ PyTorch và KHÔNG cần restart runtime.

Chạy (trên Colab, sau khi repo đồ án đã được clone):
    python scripts/cai_moi_truong.py --drive-dir /content/drive/MyDrive/Flood_House_AI

Chạy lại nhiều lần vô hại: cài xong rồi thì bỏ qua phần cài (nhờ tệp đánh dấu
`/content/.floodcount_env_ready`), phần vá luôn chạy nhưng tự biết đã vá chưa.
"""

import argparse
import glob
import importlib.util
import os
import pathlib
import re
import subprocess
import sys
import urllib.request

# ---------------------------------------------------------------------------
# Hằng số — ĐÂY LÀ CHỖ DUY NHẤT chứa phiên bản thư viện của đồ án
# ---------------------------------------------------------------------------
# Trước đây các giá trị này nằm rải ở ô [0.2] notebook 00, ô [3.1] notebook 03
# và trong chính ô cài đặt. Gộp về một chỗ vì chúng phải khớp nhau: pip cài
# đúng bản nào thì phần vá phải nhắm đúng bản đó.

# Nguồn wheel mmcv — KHÔNG phải PyPI. OpenMMLab ngừng phát hành mmcv từ 7/2024,
# wheel chính thức chỉ có tới Python 3.12, mà Colab chạy Python 3.13.
MMCV_INDEX = "https://miropsota.github.io/torch_packages_builder"

# Phiên bản cài từ PyPI (tra ngày 30/09/2026, xem docs/NOTES.md mục 1).
MMDET_VERSION = "3.3.0"        # bản cuối của MMDetection (5/2024)
MMENGINE_VERSION = "0.10.7"    # đã sửa lỗi trùng tên Adafactor (0.10.4 còn lỗi)
MMPRETRAIN_VERSION = "1.2.0"   # BẮT BUỘC: backbone ConvNeXt nằm trong gói này
PYCOCOTOOLS_VERSION = "2.0.11"

# Tệp đánh dấu đã cài xong -> chạy lại trong cùng phiên sẽ không cài lại.
# `/content` mất sạch mỗi lần Colab reset runtime nên tệp này cũng mất theo,
# đúng như mong muốn: phiên mới thì phải cài lại.
INSTALL_MARKER = "/content/.floodcount_env_ready"

DRIVE_DIR_MAC_DINH = "/content/drive/MyDrive/Flood_House_AI"

# Tên miền cho wheel Linux (Colab). Có trong bộ lọc để một cái tên dành cho
# Windows/macOS lỡ nằm cùng trang index cũng không bị chọn nhầm.
NEN_TANG = "linux_x86_64"


# ---------------------------------------------------------------------------
# Hàm thuần — test được trên máy CPU, không cần torch/mmcv
# ---------------------------------------------------------------------------
def ten_wheel_tu_html(html):
    """Danh sách tên wheel mmcv có trên trang index, đã sắp xếp và bỏ trùng.

    Phải bắt phần CHỮ HIỂN THỊ của thẻ <a>, không bắt trong `href`: trong href
    dấu `+` bị mã hoá thành `%2B` nên regex sẽ khớp ra tên bị cắt cụt (mất đúng
    phần hậu tố `+a8073c7pt2.11.0cu128` — thứ quyết định wheel có đúng bản dựng
    hay không). Đã dính lỗi này một lần rồi nên ghi lại đây.
    """
    return sorted(set(re.findall(r"mmcv-[0-9][-A-Za-z0-9._+]*\.whl", html)))


def chon_wheel(danh_sach_ten, py_tag, torch_ver, cuda_tag):
    """Chọn wheel khớp tổ hợp runtime. Trả về tên wheel, hoặc None nếu không có.

    Tên wheel trên index có dạng:
        mmcv-2.2.0+a8073c7pt2.11.0cu128-cp313-cp313-linux_x86_64.whl
    Phải khớp cả ba: Python (cp313), torch (pt2.11.0) và CUDA (cu128). Lấy sai
    một trong ba là pip quay ra tự biên dịch từ source.

    Gạch nối sau mỗi mảnh là cố ý: so khớp trần `pt2.1.1` sẽ khớp luôn vào
    `pt2.1.10cu121` (torch 2.1.10), còn `cp31` khớp vào `cp313`. Tên wheel luôn
    có `-` ngay sau phần đó, nên đòi đúng gạch nối là hết nhầm tiền tố.
    """
    khop = [n for n in danh_sach_ten
            if f"-{py_tag}-" in n and f"pt{torch_ver}{cuda_tag}-" in n
            and NEN_TANG in n]
    return khop[0] if khop else None


def cac_ban_cho_python(danh_sach_ten, py_tag):
    """Các wheel dành cho một đời Python — để in ra khi không tìm được bản khớp.

    Lọc cả nền tảng: danh sách này in ra cho người đọc gửi lại khi Colab đổi
    tổ hợp torch/CUDA, mà Colab là Linux — in thêm wheel Windows/macOS chỉ làm
    rối mắt. Gạch nối hai đầu `py_tag` để `cp31` không kéo theo `cp313`.
    """
    return [n for n in danh_sach_ten
            if f"-{py_tag}-" in n and NEN_TANG in n]


def phien_ban_tu_ten_wheel(ten_wheel, py_tag):
    """Tách chuỗi phiên bản từ tên wheel.

    pip cần đúng chuỗi này, kể cả hậu tố `+a8073c7pt2.11.0cu128`: đó là cách
    nhà dựng wheel phân biệt các bản dựng khác torch, thiếu nó là pip chọn nhầm.
    """
    return ten_wheel[len("mmcv-"):].split(f"-{py_tag}")[0]


def _duong_goi(ten_goi, *phan_tiep):
    """Đường dẫn file bên trong gói đã cài; None nếu máy chưa có gói đó."""
    spec = importlib.util.find_spec(ten_goi)
    if spec is None or spec.origin is None:
        return None
    return pathlib.Path(spec.origin).parent.joinpath(*phan_tiep)


def va_mmdet(duong_init=None):
    """Nới ngưỡng mmcv tối đa mà mmdet chấp nhận: 2.2.0 -> 2.3.0.

    mmdet 3.3.0 assert `mmcv_version < digit_version('2.2.0')` — dấu nhỏ hơn
    NGHIÊM NGẶT, nên cài đúng bản 2.2.0 vẫn trượt assert. mmcv 2.2.0 chỉ là
    bản sửa lỗi so với 2.1.0, API không đổi, nên nới lên 2.3.0 là an toàn.

    Dò bằng `find_spec` thay vì `import`: import mmdet lúc này sẽ nổ ngay vì
    chính cái assert đang cần vá.
    """
    if duong_init is None:
        duong_init = _duong_goi("mmdet", "__init__.py")
        if duong_init is None:
            return "KHÔNG thấy gói mmdet (cài đặt chưa xong?) — chạy lại script này"
    f = pathlib.Path(duong_init)
    if not f.exists():
        return f"KHÔNG thấy file cần vá: {f}"
    s = f.read_text(encoding="utf-8")
    if "mmcv_maximum_version = '2.3.0'" in s:
        return "đã vá trước đó"
    s2 = s.replace("mmcv_maximum_version = '2.2.0'",
                   "mmcv_maximum_version = '2.3.0'")
    if s2 == s:
        return f"KHÔNG tìm thấy dòng cần vá — mở file vá tay: {f}"
    f.write_text(s2, encoding="utf-8")
    return f"đã vá ({f})"


def va_mmengine_neu_can(duong_builder=None):
    """Vá lỗi trùng tên `Adafactor` — CHỈ khi mmengine còn bản lỗi.

    Bản 0.10.4 đăng ký `Adafactor` từ HAI nguồn dưới cùng một tên: torch (vòng
    lặp `dir(torch.optim)`) và transformers. Năm 2024 torch chưa có lớp này nên
    không đụng nhau; torch 2.5+ đã thêm `torch.optim.Adafactor` và Colab cài sẵn
    `transformers` -> KeyError ngay khi import.
    Bản 0.10.7 đã sửa ở upstream (đăng ký bản của torch thành 'TorchAdafactor'),
    nên hàm này thường không làm gì. Giữ lại phòng khi pip kéo về bản cũ.

    Lưu ý khi vá: dòng `OPTIMIZERS.register_module(module=_optim)` xuất hiện ở
    BỐN hàm khác nhau trong file, phải theo dõi đang ở hàm nào mới vá đúng chỗ.
    """
    if duong_builder is None:
        duong_builder = _duong_goi("mmengine", "optim", "optimizer", "builder.py")
        if duong_builder is None:
            return "KHÔNG thấy gói mmengine (cài đặt chưa xong?) — chạy lại script này"
    f = pathlib.Path(duong_builder)
    if not f.exists():
        return f"KHÔNG thấy file cần vá: {f}"
    s = f.read_text(encoding="utf-8")
    if "name='TorchAdafactor'" in s:
        return "không cần (mmengine đã sửa ở upstream)"
    if "if module_name in OPTIMIZERS:" in s:
        return "đã vá trước đó"

    dong = s.splitlines(keepends=True)
    ra, ham, i, so_cho = [], None, 0, 0
    while i < len(dong):
        l = dong[i]
        eol = "\r\n" if l.endswith("\r\n") else "\n"
        than = l.strip()
        thut = l[:len(l) - len(l.lstrip())]
        if than.startswith("def "):
            ham = than[4:].split("(")[0]

        # (1) Vòng lặp đăng ký optimizer của torch: bỏ qua tên đã có.
        if (ham == "register_torch_optimizers"
                and than == "OPTIMIZERS.register_module(module=_optim)"):
            ra.append(thut + "if module_name in OPTIMIZERS:" + eol)
            ra.append(thut + "    continue" + eol)
            ra.append(l)
            i += 1
            so_cho += 1
            continue

        # (2) transformers.Adafactor trùng tên với torch.optim.Adafactor.
        if (ham == "register_transformers_optimizers"
                and than == "OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)"):
            ra.append(thut + "if 'Adafactor' not in OPTIMIZERS:" + eol)
            ra.append(thut + "    " + than + eol)
            if (i + 1 < len(dong) and dong[i + 1].strip()
                    == "transformer_optimizers.append('Adafactor')"):
                ra.append(thut + "    transformer_optimizers.append('Adafactor')" + eol)
                i += 2
            else:
                i += 1
            so_cho += 1
            continue

        ra.append(l)
        i += 1

    if not so_cho:
        return "KHÔNG vá được (không khớp mẫu) — báo lại để xem xét"
    f.write_text("".join(ra), encoding="utf-8")
    return f"đã vá {so_cho} chỗ ({f})"


def va_mmpretrain_da_phuong_thuc(duong_dependency=None):
    """Tắt nhánh "đa phương thức" của mmpretrain — nếu không thì import nổ.

    Chuyện gì xảy ra: mmpretrain chỉ nạp BLIP / BLIP2 / LLaVA / OFA... khi cờ
    `WITH_MULTIMODAL = True`, mà cờ này bật khi máy có `transformers>=4.28.0`.
    Colab có transformers 5.x -> cờ bật -> nạp cả vườn model viết cho
    transformers 4.x -> nổ ngay:

        TypeError: NoneType takes no arguments

    Lỗi trên phát sinh trong `blip/language_model.py`: khối
    `try: from transformers.modeling_utils import ...` thất bại (transformers 5
    đã chuyển `apply_chunking_to_forward`, `prune_linear_layer` sang
    `pytorch_utils` và bỏ hẳn `find_pruneable_heads_and_indices`) nên nó gán
    `PreTrainedModel = None`, rồi VẪN đem `None` ra làm lớp cha. Cái `except:`
    trần của nó che mất nguyên nhân thật.

    Hậu quả không nhỏ: sập cả gói `mmpretrain.models`, kéo theo backbone
    ConvNeXt mà đồ án bắt buộc phải có (mmdet 3.3.0 không có ConvNeXt riêng).

    Đồ án chỉ dùng backbone ConvNeXt, không dùng model đa phương thức nào, nên
    cách sửa là tắt hẳn nhánh đó bằng ĐÚNG công tắc của OpenMMLab. Khi tắt,
    mmpretrain tự đăng ký bản thay thế cho các lớp đó — ai lỡ dùng sẽ nhận
    thông báo "cần cài mmpretrain[multimodal]" thay vì lỗi khó hiểu.
    """
    if duong_dependency is None:
        duong_dependency = _duong_goi("mmpretrain", "utils", "dependency.py")
        if duong_dependency is None:
            return "KHÔNG thấy gói mmpretrain (cài đặt chưa xong?) — chạy lại script này"

    f = pathlib.Path(duong_dependency)
    if not f.exists():
        return f"KHÔNG thấy file cần vá: {f}"
    s = f.read_text(encoding="utf-8")
    if "WITH_MULTIMODAL = False" in s:
        return "đã vá trước đó"

    cu = ("WITH_MULTIMODAL = all(\n"
          "    satisfy_requirement(item)\n"
          "    for item in ['pycocotools', 'transformers>=4.28.0'])")
    moi = ("# [floodcount] Tắt nhánh đa phương thức: nó viết cho transformers 4.x,\n"
           "# Colab dùng 5.x nên import nổ và làm sập cả gói mmpretrain.\n"
           "# Đồ án chỉ cần backbone ConvNeXt. Chi tiết: docs/NOTES.md muc 1.6.\n"
           "WITH_MULTIMODAL = False")
    s2 = s.replace(cu, moi)
    if s2 == s:
        return f"KHÔNG tìm thấy đoạn cần vá — mở file vá tay: {f}"
    f.write_text(s2, encoding="utf-8")
    return f"đã vá ({f})"


def va_ca_ba(duong_mmdet=None, duong_mmengine=None, duong_mmpretrain=None):
    """Chạy cả ba miếng vá, trả về (mã thoát, các dòng báo cáo).

    Ba đối số chỉ dùng cho test: truyền đường dẫn file giả để chạy được trên máy
    chưa cài mmdet/mmengine/mmpretrain. Để trống thì vá đúng gói đã cài.
    """
    dong = [f"vá mmdet      : {va_mmdet(duong_mmdet)}",
            f"vá mmengine   : {va_mmengine_neu_can(duong_mmengine)}",
            f"vá mmpretrain : {va_mmpretrain_da_phuong_thuc(duong_mmpretrain)}"]
    # Không vá được thì báo lỗi, nhưng vẫn in đủ ba dòng: người đọc cần thấy
    # cái nào đã xong, cái nào chưa, chứ không phải chỉ thấy lỗi đầu tiên.
    loi = any("KHÔNG" in d for d in dong)
    return (1 if loi else 0), dong


# ---------------------------------------------------------------------------
# Phần cài đặt — cần torch thật, chỉ chạy được trên Colab
# ---------------------------------------------------------------------------
def chay(cmd):
    """Chạy lệnh shell, in ra lệnh trước khi chạy. Trả về mã thoát."""
    print("\n$ " + cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode


def cai_dat(drive_dir, marker=INSTALL_MARKER, cai_lai=False):
    """Cài 4 gói. Trả về mã thoát (0 là xong, kể cả khi bỏ qua vì đã cài)."""
    if cai_lai and os.path.exists(marker):
        os.remove(marker)
        print(f"[--cai-lai] đã xoá tệp đánh dấu {marker}")

    if os.path.exists(marker):
        print("[bỏ qua] Môi trường đã được cài ở lần chạy trước trong phiên này.")
        print(f"         Muốn cài lại từ đầu: xoá {marker} rồi chạy lại.")
        return 0

    import torch

    # --- Dò tổ hợp Python / torch / CUDA của runtime hiện tại ---------------
    PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"   # vd "cp313"
    TV = torch.__version__.split("+")[0]                             # vd "2.11.0"
    if torch.version.cuda is None:
        # Chặn sớm: bản torch CPU-only thì không có tổ hợp "cuXXX" nào để dò
        # wheel, và cũng không train được. Báo rõ cách sửa thay vì để pip lỗi.
        print(f"[!] Runtime này là bản CPU-only (torch {torch.__version__}) nên "
              "không có CUDA.\n"
              "=> Vào Runtime > Change runtime type > chọn T4 GPU, "
              "rồi chạy lại.")
        return 1
    CU = "cu" + torch.version.cuda.replace(".", "")                  # vd "cu128"
    print(f"Runtime: Python {sys.version.split()[0]} | torch {torch.__version__}")
    print(f"Cần wheel mmcv cho: {PY_TAG} + pt{TV} + {CU}")

    # --- Chọn chỗ cache wheel (ưu tiên Drive để phiên sau dùng lại) ---------
    wheel_cache = os.path.join(drive_dir, "wheels")
    if os.path.isdir(drive_dir):
        os.makedirs(wheel_cache, exist_ok=True)
    else:
        wheel_cache = "/content/wheels"
        os.makedirs(wheel_cache, exist_ok=True)
        print("[lưu ý] Chưa thấy thư mục Drive -> cache wheel vào /content/wheels")

    # Gạch nối trong mẫu glob là cố ý, cùng lý do như ở `chon_wheel`: không có
    # nó thì `pt2.1.1` khớp nhầm sang wheel của torch 2.1.10.
    da_cache = glob.glob(f"{wheel_cache}/mmcv-*pt{TV}{CU}-{PY_TAG}-*.whl")
    if da_cache:
        # Đã có wheel đúng tổ hợp trong cache -> cài thẳng, không phụ thuộc index.
        print(f"Dùng wheel đã cache:\n    {da_cache[0]}")
        if chay(f'pip install "{da_cache[0]}"') != 0:
            return 1
    else:
        # --- Tìm wheel khớp trong index cộng đồng ---------------------------
        with urllib.request.urlopen(f"{MMCV_INDEX}/mmcv/", timeout=60) as r:
            html = r.read().decode("utf-8", "replace")
        names = ten_wheel_tu_html(html)
        wheel = chon_wheel(names, PY_TAG, TV, CU)

        if wheel is None:
            print(f"\n[!] KHÔNG có wheel mmcv khớp {PY_TAG} + pt{TV}{CU}.")
            print(f"    Các bản đang có cho {PY_TAG} (linux):")
            for n in cac_ban_cho_python(names, PY_TAG):
                print("     ", n)
            print("\n=> Nhiều khả năng Colab vừa nâng cấp torch/CUDA.\n"
                  "   Gửi danh sách trên để cập nhật lại đồ án.")
            return 1

        mmcv_local = phien_ban_tu_ten_wheel(wheel, PY_TAG)
        print(f"Chọn wheel : {wheel}")
        print(f"Phiên bản  : mmcv=={mmcv_local}")

        # Phải là mmcv bản CÓ CUDA OPS (không phải mmcv-lite): mmdet dùng nms và
        # roi_align của mmcv trong cả lúc train lẫn lúc suy luận.
        if chay(f"pip install mmcv=={mmcv_local} --extra-index-url {MMCV_INDEX}") != 0:
            return 1

        # Lưu wheel lên Drive để phiên sau cài từ cache — index cộng đồng có thể
        # đổi hoặc ngừng hoạt động, còn đồ án thì phải chạy lại được sau nhiều tháng.
        if chay(f'pip download mmcv=={mmcv_local} --no-deps '
                f'--extra-index-url {MMCV_INDEX} -d "{wheel_cache}"') != 0:
            print("[lưu ý] Không lưu được wheel vào cache — không sao, chỉ là "
                  "phiên sau sẽ phải tải lại.")

    # --- Ba gói còn lại từ PyPI ---------------------------------------------
    # mmdet KHÔNG khai báo mmcv/mmengine trong metadata (chỉ ở extra "mim") nên
    # pip sẽ không kéo mmcv về bản cũ — đã kiểm tra trên PyPI ngày 30/09/2026.
    if chay(f"pip install mmengine=={MMENGINE_VERSION} mmdet=={MMDET_VERSION} "
            f"mmpretrain=={MMPRETRAIN_VERSION} pycocotools=={PYCOCOTOOLS_VERSION}") != 0:
        return 1

    pathlib.Path(marker).write_text("ok", encoding="utf-8")
    print("\n" + "!" * 72)
    print("CÀI ĐẶT XONG. Không cần restart runtime.")
    print("!" * 72)
    return 0


def in_phien_ban():
    """In các phiên bản script này sẽ cài — để đối chiếu khi cần gỡ lỗi."""
    print("Nguồn wheel mmcv    :", MMCV_INDEX)
    print("Tệp đánh dấu        :", INSTALL_MARKER)
    for ten, ban in (("mmcv", "(dò theo runtime)"),
                     ("mmengine", MMENGINE_VERSION),
                     ("mmdet", MMDET_VERSION),
                     ("mmpretrain", MMPRETRAIN_VERSION),
                     ("pycocotools", PYCOCOTOOLS_VERSION)):
        print(f"{ten:<20}: {ban}")


def in_utf8():
    """Console Windows mặc định cp1252 -> in chữ Việt là nổ UnicodeEncodeError."""
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def main():
    in_utf8()
    ap = argparse.ArgumentParser(
        description="Cài môi trường MMDetection cho Colab (mmcv + mmengine + "
                    "mmdet + mmpretrain) rồi vá 3 chỗ.")
    ap.add_argument("--drive-dir", default=DRIVE_DIR_MAC_DINH,
                    help="thư mục Drive để cache wheel mmcv (mặc định: %(default)s)")
    ap.add_argument("--marker", default=INSTALL_MARKER,
                    help="tệp đánh dấu đã cài xong (mặc định: %(default)s)")
    ap.add_argument("--cai-lai", action="store_true",
                    help="xoá tệp đánh dấu rồi cài lại từ đầu")
    ap.add_argument("--xem-phien-ban", action="store_true",
                    help="chỉ in phiên bản sẽ cài rồi thoát, không cài gì")
    args = ap.parse_args()

    if args.xem_phien_ban:
        in_phien_ban()
        return 0

    print("=" * 72)
    print("CÀI MÔI TRƯỜNG MMDetection — scripts/cai_moi_truong.py")
    print("=" * 72)
    print("Nguồn wheel mmcv    :", MMCV_INDEX)
    print("Phiên bản PyPI      : mmengine", MMENGINE_VERSION,
          "| mmdet", MMDET_VERSION,
          "| mmpretrain", MMPRETRAIN_VERSION,
          "| pycocotools", PYCOCOTOOLS_VERSION)

    if cai_dat(args.drive_dir, args.marker, args.cai_lai) != 0:
        return 1

    # Phần vá đặt NGOÀI nhánh "đã cài chưa": một phiên cài xong nhưng chưa vá
    # (hoặc vừa pip cài đè) vẫn phải được vá. Các hàm vá tự kiểm tra nên chạy
    # lại nhiều lần vô hại.
    ma, dong = va_ca_ba()
    print()
    for d in dong:
        print(d)
    return ma


if __name__ == "__main__":
    sys.exit(main())

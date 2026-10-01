# -*- coding: utf-8 -*-
"""Test cho src/floodcount/data/photometric.py — CPU, không cần mmdet.

Hai điều test này phải khoá, vì cả hai đều là lỗi IM LẶNG (ảnh vẫn ra bình
thường, chỉ có nội dung sai):

  1. KHÔNG được đảo sáng tối. `astype(np.uint8)` một mình lấy modulo 256: pixel
     350 thành 94 (đốm trắng thành đốm gần đen). Bắt buộc phải `np.clip` trước.
     Đây cũng chính là lỗi của `cv2.convertScaleAbs` — hàm đó lấy trị tuyệt đối
     nên pixel bị đẩy xuống dưới 0 thành pixel SÁNG.

  2. `tang_sang_nhe` KHÔNG được sửa mảng đầu vào. Sửa tại chỗ thì ảnh đã tăng
     sáng ở epoch trước vẫn còn tăng sáng ở epoch sau (cùng một mảng được
     transform dùng lại), và sai số cộng dồn qua các epoch.

Chạy:
    py tests/test_photometric.py
"""

import pathlib
import sys

import numpy as np

GOC_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC_REPO / "src"))

from floodcount.data.photometric import chon_tham_so, tang_sang_nhe  # noqa: E402
import floodcount.data.photometric as pm                              # noqa: E402


def main():
    loi = []

    def kiem(dieu_kien, mo_ta):
        trang_thai = "OK  " if dieu_kien else "FAIL"
        print(f"  [{trang_thai}] {mo_ta}")
        if not dieu_kien:
            loi.append(mo_ta)

    print("=== 1. Khong doi anh khi delta = 0 va alpha = 1 ===")
    rng = np.random.default_rng(0)
    anh = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)
    ra = tang_sang_nhe(anh, 0.0, 1.0)
    kiem(np.array_equal(ra, anh), "delta=0, alpha=1 -> anh y nguyen")
    kiem(ra.dtype == np.uint8, f"dtype giu nguyen uint8, thuc te {ra.dtype}")
    kiem(ra.shape == anh.shape, f"kich thuoc giu nguyen, thuc te {ra.shape}")

    print("\n=== 2. KHONG sua mang dau vao ===")
    goc = anh.copy()
    tang_sang_nhe(anh, 30.0, 1.1)
    kiem(np.array_equal(anh, goc), "mang dau vao khong bi sua tai cho")

    print("\n=== 3. Cat dung ve [0, 255], khong lay modulo 256 ===")
    # 250 + 100 = 350. Neu astype(uint8) truoc khi clip thi 350 % 256 = 94 —
    # dom trang thanh dom gan den, anh van "ra binh thuong".
    sang = np.full((4, 4, 3), 250, np.uint8)
    ra_sang = tang_sang_nhe(sang, 100.0, 1.0)
    kiem(bool(np.all(ra_sang == 255)),
         f"250 + 100 -> 255 (khong phai 94), thuc te {sorted(set(ra_sang.flatten()))}")

    # 3 - 100 = -97. cv2.convertScaleAbs se cho |-97| = 97 (TOI thanh SANG).
    toi = np.full((4, 4, 3), 3, np.uint8)
    ra_toi = tang_sang_nhe(toi, -100.0, 1.0)
    kiem(bool(np.all(ra_toi == 0)),
         f"3 - 100 -> 0 (khong phai 97), thuc te {sorted(set(ra_toi.flatten()))}")
    kiem(bool(np.all(ra_toi <= toi)),
         "lam toi thi moi diem anh phai <= gia tri cu (khong co diem nao SANG len)")

    print("\n=== 4. delta doi trung binh, alpha giu trung binh ===")
    nen = np.full((50, 50, 3), 120, np.uint8)
    # anh phang: trung binh = 120. Cong 20 -> 140.
    kiem(abs(float(tang_sang_nhe(nen, 20.0, 1.0).mean()) - 140.0) < 0.51,
         f"anh phang 120 + delta 20 -> trung binh ~140, "
         f"thuc te {tang_sang_nhe(nen, 20.0, 1.0).mean():.2f}")
    # Doi tuong phan keo gian quanh TRUNG BINH (khong phai quanh 0): trung binh
    # phai gan nhu giu nguyen, chi do phan bo rong ra.
    anh_tt = rng.integers(60, 180, size=(80, 80, 3), dtype=np.uint8)
    ra_tt = tang_sang_nhe(anh_tt, 0.0, 1.3)
    kiem(abs(float(ra_tt.mean()) - float(anh_tt.mean())) < 0.6,
         f"alpha 1.3 giu trung binh (keo quanh trung binh, khong quanh 0): "
         f"{anh_tt.mean():.2f} -> {ra_tt.mean():.2f}")
    kiem(float(ra_tt.std()) > float(anh_tt.std()),
         f"alpha 1.3 lam do lech chuan TANG: {anh_tt.std():.2f} -> {ra_tt.std():.2f}")
    ra_tt_nho = tang_sang_nhe(anh_tt, 0.0, 0.7)
    kiem(float(ra_tt_nho.std()) < float(anh_tt.std()),
         f"alpha 0.7 lam do lech chuan GIAM: {anh_tt.std():.2f} -> {ra_tt_nho.std():.2f}")

    print("\n=== 5. Anh xam (2 chieu) va anh 1 diem anh ===")
    xam = rng.integers(0, 256, size=(30, 30), dtype=np.uint8)
    ra_xam = tang_sang_nhe(xam, 10.0, 1.0)
    kiem(ra_xam.shape == (30, 30) and ra_xam.dtype == np.uint8,
         f"anh xam 2 chieu van ra 2 chieu, thuc te {ra_xam.shape}")
    kiem(abs(float(ra_xam.mean()) - float(xam.mean()) - 10.0) < 1.5,
         "anh xam cung duoc cong delta nhu anh mau")

    print("\n=== 6. dtype sai phai bao loi, khong tu doan ===")
    for dtype in (np.float32, np.int16, np.int32):
        try:
            tang_sang_nhe(np.zeros((4, 4, 3), dtype), 0.0, 1.0)
            kiem(False, f"anh {dtype} phai nem TypeError")
        except TypeError:
            kiem(True, f"anh {dtype} -> TypeError (khong tu ep kieu)")

    print("\n=== 7. chon_tham_so: trong khoang, tai lap duoc ===")
    for _ in range(200):
        d, a = chon_tham_so(24.0, (0.85, 1.15))
        if not (-24.0 <= d <= 24.0 and 0.85 <= a <= 1.15):
            kiem(False, f"tham so ra ngoai khoang: delta={d}, alpha={a}")
            break
    else:
        kiem(True, "200 lan boc deu nam trong khoang cho phep")

    d1, a1 = chon_tham_so(24.0, (0.85, 1.15), rng=np.random.default_rng(42))
    d2, a2 = chon_tham_so(24.0, (0.85, 1.15), rng=np.random.default_rng(42))
    kiem((d1, a1) == (d2, a2),
         f"cung seed -> cung ket qua ({d1:.6f}, {a1:.6f}) — tai lap duoc")
    d3, _ = chon_tham_so(24.0, (0.85, 1.15), rng=np.random.default_rng(7))
    kiem(d1 != d3, "seed khac -> ket qua khac")

    print("\n=== 8. Module nay KHONG duoc keo mmdet vao ===")
    # Day la rang buoc thiet ke: photometric.py phai chay duoc tren may CPU khong
    # cai MMDetection de con test duoc. Lo import mmdet vao day thi ca bo test
    # nay lan tests/test_kiem_tra.py deu khong chay noi o may sach.
    nang = [m for m in sys.modules
            if m.split(".")[0] in ("mmdet", "mmcv", "mmengine", "torch")]
    kiem(not nang, f"import photometric khong keo theo thu vien nang, thuc te {nang}")
    kiem(pm.__file__.endswith("photometric.py"), "import dung module can test")

    print()
    if loi:
        print(f"*** {len(loi)} MUC KHONG DAT ***")
        for m in loi:
            print("   - " + m.encode("ascii", "backslashreplace").decode())
        raise SystemExit(1)
    print("*** TAT CA PASS ***")


if __name__ == "__main__":
    main()

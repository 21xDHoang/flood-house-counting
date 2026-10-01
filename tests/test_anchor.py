# -*- coding: utf-8 -*-
"""Test cho src/floodcount/models/anchor.py — CPU, không cần mmdet, không cần dataset.

Điều quan trọng nhất test này phải khoá: `do_phu_thuc` trả về CHỈ SỐ ANCHOR TỐT
NHẤT ĐÚNG. Bản đầu của hàm này có lỗi broadcasting (`[:, :, None]` thừa một trục)
làm mảng IoU thành (n, 1, M) và `argmax(axis=1)` luôn trả 0 — tức là mọi box đều
được báo "anchor tốt nhất là anchor số 0". Bảng kết quả vẫn in ra đầy đủ, không
một dòng lỗi nào. Đó là loại lỗi mà chỉ test mới bắt được.

Cách khoá: dựng những box có ĐÁP ÁN BIẾT TRƯỚC (box trùng khít một hình dạng
anchor, đặt đúng trên một mốc lưới) rồi đòi đúng chỉ số đó. Nếu argmax hỏng, chỉ
số ra 0 và test đỏ ngay.

Chạy:
    py tests/test_anchor.py
"""

import math
import pathlib
import sys

import numpy as np

GOC_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC_REPO / "src"))

from floodcount.models.anchor import (RATIOS_MAC_DINH, SCALES_MAC_DINH,  # noqa: E402
                                      STRIDES_MAC_DINH, canh_nho,
                                      do_phu_thuc, iou_cung_tam,
                                      kich_thuoc_anchor, phan_tich)

# Chỉ số của hình dạng anchor trong danh sách sinh bởi `kich_thuoc_anchor`.
# Thứ tự sinh là: với từng base_size -> từng ratio -> từng scale. Không có
# `base_sizes` nên base_size = min(stride) = [4, 8, 16, 32, 64], và cỡ anchor
# = base_size * 8 = [32, 64, 128, 256, 512]. Ba tỉ lệ (0.5, 1.0, 2.0) cho
# (rộng, vuông, cao), nên mỗi cỡ chiếm 3 chỉ số liên tiếp.
IDX_256_VUONG = 10      # base_size 32 -> cỡ 256, ratio 1.0
IDX_512_VUONG = 13      # base_size 64 -> cỡ 512, ratio 1.0


def main():
    loi = []

    def kiem(dieu_kien, mo_ta):
        trang_thai = "OK  " if dieu_kien else "FAIL"
        print(f"  [{trang_thai}] {mo_ta}")
        if not dieu_kien:
            loi.append(mo_ta)

    def gan(a, b, eps=1e-9):
        return abs(float(a) - float(b)) < eps

    print("=== 1. Sinh hinh dang anchor dung cong thuc AnchorGenerator ===")
    hinh = kich_thuoc_anchor(STRIDES_MAC_DINH, SCALES_MAC_DINH, RATIOS_MAC_DINH)
    kiem(len(hinh) == 15, f"5 stride x 3 ratio = 15 hinh dang, thuc te {len(hinh)}")

    # 32 = min(stride 4) * scale 8, va day la anchor VUONG (ratio 1.0) dau tien
    kiem(hinh[1] == (32.0, 32.0), f"hinh dang [1] = 32x32, thuc te {hinh[1]}")
    kiem(hinh[13] == (512.0, 512.0), f"hinh dang [13] = 512x512, thuc te {hinh[13]}")
    # ratio < 1 cho anchor RONG: w = base/sqrt(ratio) > base
    kiem(hinh[0][0] > hinh[0][1], "ratio 0.5 -> anchor rong (w > h)")
    kiem(hinh[2][1] > hinh[2][0], "ratio 2.0 -> anchor cao (h > w)")
    # Canh lon nhat: 512 / sqrt(0.5) = 724.08 — con so nay la ly do phai do:
    # nha FloodNet co p90 canh lon 348px nen 724px phu qua thoai mai.
    canh_lon_nhat = max(max(w, h) for w, h in hinh)
    kiem(gan(canh_lon_nhat, 512.0 * math.sqrt(2.0), 1e-6),
         f"canh lon nhat = 512*sqrt(2) = {512 * math.sqrt(2):.2f}px, "
         f"thuc te {canh_lon_nhat:.2f}px")

    print("\n=== 2. base_sizes tuy chinh va stride dang cap (4, 4) ===")
    # AnchorGenerator cho phep stride la so nguyen hoac cap; quy tac la lay canh nho.
    kiem(canh_nho(4) == 4, "canh_nho(4) = 4 (stride la so nguyen)")
    kiem(canh_nho((4, 8)) == 4, "canh_nho((4, 8)) = 4 (lay canh nho)")
    # min() truc tiep tren so nguyen se no TypeError — day la ly do co ham nay.
    try:
        min(4)                                     # noqa: F821
        kiem(False, "min(4) phai no TypeError (neu khong thi ham canh_nho la thua)")
    except TypeError:
        kiem(True, "min(4) no TypeError -> bat buoc phai co ham canh_nho")

    hinh_cap = kich_thuoc_anchor([(4, 8), (16, 16)], [8], [1.0])
    kiem(hinh_cap == [(32.0, 32.0), (128.0, 128.0)],
         f"stride dang cap (4,8) va (16,16) -> 32x32 va 128x128, thuc te {hinh_cap}")

    hinh_bs = kich_thuoc_anchor([4], [1], [1.0], base_sizes=[100])
    kiem(hinh_bs == [(100.0, 100.0)],
         f"base_sizes tuy chinh duoc uu tien, thuc te {hinh_bs}")

    print("\n=== 3. iou_cung_tam — doi chieu bang so tinh tay ===")
    kiem(gan(iou_cung_tam((100, 100), [(100, 100)])[0], 1.0),
         "box 100x100 voi anchor 100x100 -> IoU = 1.0")
    # 100x100 long trong 200x200: giao 10000, hop 10000 + 40000 - 10000 = 40000
    kiem(gan(iou_cung_tam((100, 100), [(200, 200)])[0], 0.25),
         "box 100x100 trong anchor 200x200 -> IoU = 0.25 (tinh tay: 10000/40000)")
    kiem(gan(iou_cung_tam((200, 200), [(100, 100)])[0], 0.25),
         "dao lai hai ben cho cung ket qua (IoU doi xung)")
    kiem(gan(iou_cung_tam((0, 0), [(100, 100)])[0], 0.0),
         "box rong -> IoU = 0, khong ra nan")
    kiem(len(iou_cung_tam((50, 50), hinh)) == 15,
         "tra ve dung 15 gia tri, cung thu tu voi kich_thuoc_anchor")

    print("\n=== 4. do_phu_thuc: CHI SO ANCHOR TOT NHAT phai dung ===")
    # Box trung khit anchor 256x256, dat DUNG tren moc luoi cua moi tang
    # (256 chia het cho 4, 8, 16, 32, 64 nen khong tang nao bi lech tam).
    box = np.array([[256 - 128, 256 - 128, 256, 256]], dtype=np.float64)
    iou, chi_so = do_phu_thuc(box)
    kiem(iou.shape == (1,) and chi_so.shape == (1,),
         f"tra ve mang (N,) chu khong phai (N,1,M), thuc te {iou.shape} {chi_so.shape}")
    kiem(gan(iou[0], 1.0, 1e-6),
         f"box trung khit mot anchor -> IoU = 1.0, thuc te {iou[0]:.6f}")
    kiem(chi_so[0] == IDX_256_VUONG,
         f"chi so anchor tot nhat = {IDX_256_VUONG} (256x256), "
         f"thuc te {chi_so[0]}")

    # Day chinh la phep thu bat loi argmax-luon-tra-0: hai box khac co phai cho
    # ra HAI chi so khac nhau.
    box_to = np.array([[512 - 256, 512 - 256, 512, 512]], dtype=np.float64)
    _, chi_so_to = do_phu_thuc(box_to)
    kiem(chi_so_to[0] == IDX_512_VUONG,
         f"box 512x512 -> chi so {IDX_512_VUONG}, thuc te {chi_so_to[0]}")
    kiem(chi_so[0] != chi_so_to[0],
         f"hai box khac co ra HAI chi so khac nhau (neu bang nhau la loi argmax)")

    # Hai hinh dang lech nhau chi o CHO NAO la canh dai. Trong danh sach sinh ra,
    # hinh dang [12] = (724.08, 362.04) la anchor RONG (w > h) va [14] =
    # (362.04, 724.08) la anchor CAO (h > w). Nham hai cai nay la nham hung thu —
    # ca hai deu co cung dien tich nen IoU voi nhau chi 1/3.
    # ratio 0.5 -> w = base/sqrt(0.5) = base*sqrt(2) = 724.08 (canh DAI la chieu
    # NGANG); ratio 2.0 -> nguoc lai. De y sqrt(0.5) = 1/sqrt(2) nen chia cho
    # sqrt(0.5) chinh la nhan voi sqrt(2) — doi cho hai bieu thuc nay la dao
    # ngay y nghia rong/cao.
    w_rong = 512.0 * math.sqrt(2.0)      # 724.08, canh ngang
    h_rong = 512.0 / math.sqrt(2.0)      # 362.04, canh doc
    # Tam (512, 512) chia het cho ca 4, 8, 16, 32, 64 nen khong tang nao bi lech —
    # nho vay ket qua phai dung bang 1.0 chu khong phai "xap xi 1.0".
    box_rong = np.array([[512 - w_rong / 2, 512 - h_rong / 2, w_rong, h_rong]],
                        dtype=np.float64)
    iou_rong, chi_so_rong = do_phu_thuc(box_rong)
    kiem(gan(iou_rong[0], 1.0, 1e-6),
         f"box RONG dung bang hinh dang [12] -> IoU = 1.0, thuc te {iou_rong[0]:.6f}")
    kiem(chi_so_rong[0] == 12,
         f"  -> chi so 12 (khong phai 14), thuc te {chi_so_rong[0]}")

    box_cao = np.array([[512 - h_rong / 2, 512 - w_rong / 2, h_rong, w_rong]],
                       dtype=np.float64)
    _, chi_so_cao = do_phu_thuc(box_cao)
    kiem(chi_so_cao[0] == 14,
         f"box CAO dung bang hinh dang [14] -> chi so 14, thuc te {chi_so_cao[0]}")

    print("\n=== 5. Lech tam lam IoU giam, va do_phu_thuc <= iou_cung_tam ===")
    # Lech 2px moi truc. Phai chon so KHONG chia het cho 4: lech 32px nhu ban dau
    # thi 288 chia het cho 4/8/16/32 nen tang stride-32 van khop tam hoan hao va
    # IoU van la 1.0 — do la ket qua DUNG, khong phai loi.
    # Voi lech 2px, moi tang deu lech 2px nen giao = (256-2) x (256-2), va
    # IoU = 254^2 / (2*256^2 - 254^2) — tinh duoc bang tay.
    box_lech = np.array([[256 - 128 + 2, 256 - 128 + 2, 256, 256]], dtype=np.float64)
    iou_lech, chi_so_lech = do_phu_thuc(box_lech)
    mong_doi = 254.0 ** 2 / (2 * 256.0 ** 2 - 254.0 ** 2)
    kiem(gan(iou_lech[0], mong_doi, 1e-9),
         f"lech 2px -> IoU = {mong_doi:.6f} (tinh tay 254^2/(2*256^2-254^2)), "
         f"thuc te {iou_lech[0]:.6f}")
    kiem(chi_so_lech[0] == IDX_256_VUONG,
         f"lech tam khong doi hinh dang thang, van la chi so "
         f"{IDX_256_VUONG}, thuc te {chi_so_lech[0]}")

    # Bat dang thuc bao ham: do_phu_thuc la truong hop CO vi tri, iou_cung_tam la
    # truong hop KHONG vi tri (dat trung tam) — nen cai sau phai >= cai truoc.
    rng = np.random.default_rng(0)
    ds_box = np.concatenate([
        rng.uniform(0, 1400, size=(200, 2)),
        rng.uniform(20, 400, size=(200, 2)),
    ], axis=1)
    iou_thuc, _ = do_phu_thuc(ds_box)
    iou_ct = np.array([float(np.max(iou_cung_tam((w, h), hinh)))
                       for _, _, w, h in ds_box])
    kiem(bool(np.all(iou_thuc <= iou_ct + 1e-9)),
         "moi box: do_phu_thuc <= iou_cung_tam (co vi tri thi khong the tot hon)")
    kiem(bool(np.all((iou_thuc >= 0.0) & (iou_thuc <= 1.0))),
         "moi IoU nam trong [0, 1]")
    # Con so nay la tham chieu cho lan sau: doi `scales`/`ratios` cua RPN thi no
    # phai doi theo. Bo sinh so ngau nhien da co seed nen gia tri la xac dinh.
    trung_vi = float(np.median(iou_thuc))
    kiem(trung_vi >= 0.5,
         f"box co canh 20..400px: trung vi IoU thuc = {trung_vi:.3f} (>= 0.5)")

    print("\n=== 6. phan_tich: bang so lieu ===")
    nhan = np.array([1] * 100 + [2] * 100)
    kq = phan_tich(ds_box, nhan=nhan)
    kiem(kq["so_box"] == 200, f"so_box = 200, thuc te {kq['so_box']}")
    kiem(sum(kq["dem_best_anchor"]) == 200,
         f"tong dem_best_anchor = so box (moi box co dung mot anchor thang), "
         f"thuc te {sum(kq['dem_best_anchor'])}")
    kiem(len(kq["dem_best_anchor"]) == 15,
         f"dem theo 15 hinh dang anchor, thuc te {len(kq['dem_best_anchor'])}")
    kiem(set(kq["theo_lop"]) == {1, 2}, "tach duoc theo hai lop")
    kiem(kq["theo_lop"][1]["so_box"] == 100 and kq["theo_lop"][2]["so_box"] == 100,
         "moi lop 100 box")
    for nguong in (0.3, 0.5, 0.7):
        ti = kq["iou_thuc"][f"ti_le_iou>={nguong}"]
        kiem(0.0 <= ti <= 1.0, f"ti le dat IoU >= {nguong} nam trong [0,1]")

    # Loi ro rang khi so nhan lech so box, khong duoc im lang cat bot
    try:
        phan_tich(ds_box, nhan=nhan[:10])
        kiem(False, "so nhan lech so box phai nem ValueError")
    except ValueError:
        kiem(True, "so nhan lech so box -> ValueError (khong im lang)")

    # Mang rong khong duoc lam no ham
    kq_rong = phan_tich(np.zeros((0, 4)))
    kiem(kq_rong["so_box"] == 0 and not np.isnan(kq_rong["iou_thuc"]["trung_vi"])
         or np.isnan(kq_rong["iou_thuc"]["trung_vi"]),
         "mang box rong khong lam ham no")

    print()
    if loi:
        print(f"*** {len(loi)} MUC KHONG DAT ***")
        for m in loi:
            print("   - " + m.encode("ascii", "backslashreplace").decode())
        raise SystemExit(1)
    print("*** TAT CA PASS ***")


if __name__ == "__main__":
    main()

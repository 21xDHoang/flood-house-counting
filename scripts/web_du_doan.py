# -*- coding: utf-8 -*-
"""Trang web nhỏ để THỬ mô hình: tải ảnh lên, xem box và số nhà đếm được.

Cố ý tối giản — một ảnh vào, một ảnh ra kèm số đếm, cộng thêm thanh trượt
ngưỡng điểm. Không đăng nhập, không lưu gì, không hàng đợi phức tạp.

Chạy trên Colab (cần GPU + checkpoint đã train):
    python scripts/web_du_doan.py --share
Rồi mở link công khai mà Gradio in ra (dạng https://...gradio.live) — mở được
luôn trên điện thoại, cùng mạng hay khác mạng đều được. Link sống chừng nào
phiên Colab còn chạy; ngắt Colab là link chết.

Vì sao Gradio chứ không phải web tự viết: nó là thư viện chuẩn để bọc một hàm
Python thành giao diện, Colab cài sẵn, và `--share` lo luôn phần đưa link ra
internet mà không phải mở cổng hay dựng server. Ba thứ đó đúng bằng thứ đồ án
cần, không hơn.

LÕI SUY LUẬN nằm ở `scripts/du_doan.py` (đã test) — tệp này chỉ lo giao diện.
"""

import argparse
import pathlib
import sys

GOC_REPO = pathlib.Path(__file__).resolve().parents[1]
# Cùng thứ tự nạp như `scripts/du_doan.py`: `src/` cho custom_imports của
# config, `scripts/` để import được `du_doan` và `train`.
sys.path.insert(0, str(GOC_REPO / "src"))
sys.path.insert(0, str(GOC_REPO / "scripts"))

from floodcount.data.kiem_tra import doc_classes_tu_data_yaml  # noqa: E402
from floodcount.infer.dem import tim_checkpoint  # noqa: E402
import du_doan  # noqa: E402
import train as _train  # noqa: E402

MO_TA = (
    "Tải ảnh lên (hoặc chụp từ điện thoại) rồi bấm **Submit**.\n\n"
    "- Box **đỏ** = nhà ngập, box **xanh dương** = nhà không ngập; số cạnh box "
    "là độ tin cậy.\n"
    "- Thanh trượt *Ngưỡng điểm*: kéo lên để bớt box rác, kéo xuống để bắt "
    "được nhiều nhà hơn.\n"
    "- Ảnh càng giống ảnh máy bay (nhìn từ trên xuống) thì kết quả càng đáng "
    "tin — mô hình học trên ảnh chụp từ trên cao."
)


def dung_giao_dien(model, pipeline, ten_lop, nguong_mac_dinh):
    """Bọc lõi suy luận thành giao diện Gradio. Trả về đối tượng Interface."""
    import gradio as gr

    def suy_luan(anh_rgb, nguong_diem):
        if anh_rgb is None:
            return None, "Chưa có ảnh — tải ảnh lên rồi bấm Submit."
        # Gradio đưa ảnh dạng RGB; OpenCV và pipeline của mmdet dùng BGR.
        mang_bgr = anh_rgb[:, :, ::-1].copy()
        kq = du_doan.chay_mot_mang(model, pipeline, mang_bgr, nguong_diem,
                                   ten_lop)
        so_ngap = kq["dem"].get("flooded_building", 0)
        so_khong = kq["dem"].get("non_flooded_building", 0)
        dong_tieu_de = (f"MH dem: {so_ngap} ngap / {so_khong} khong ngap"
                        f" | nguong {nguong_diem:.2f}")
        anh_ve = du_doan.ve_anh(mang_bgr, kq, ten_lop, dong_tieu_de)
        bao_cao = (f"Nhà NGẬP: {so_ngap}\n"
                   f"Nhà KHÔNG ngập: {so_khong}\n"
                   f"Thời gian: {kq['ms']:.0f} ms\n"
                   f"(ngưỡng điểm {nguong_diem:.2f}, "
                   f"{len(kq['boxes'])} box được giữ)")
        # Trả về RGB cho Gradio hiển thị.
        return anh_ve[:, :, ::-1], bao_cao

    return gr.Interface(
        fn=suy_luan,
        inputs=[
            gr.Image(type="numpy", label="Ảnh đầu vào"),
            # Tham số đặt theo TÊN, không theo vị trí: thứ tự tham số của Slider
            # khác nhau giữa các đời Gradio, còn tên thì không.
            gr.Slider(minimum=0.05, maximum=0.9,
                      value=float(nguong_mac_dinh), step=0.05,
                      label="Ngưỡng điểm"),
        ],
        outputs=[
            gr.Image(type="numpy", label="Mô hình nhận diện"),
            gr.Textbox(label="Đếm được", lines=5),
        ],
        title="Nhận diện nhà ngập — Cascade R-CNN + ConvNeXt-Tiny (FloodNet)",
        description=MO_TA,
        # "never" = không hiện nút gắn cờ (nút này ghi tệp vào ổ đĩa Colab, chẳng
        # để làm gì ở đây). Tên tham số là `flagging_mode` — bản Gradio mới đã bỏ
        # tên cũ `allow_flagging`, nên dùng tên cũ có thể nổ ngay trên Colab.
        flagging_mode="never",
    )


def main(argv=None):
    """Chạy trang web. `argv` để notebook gọi được `main(["--share"])` mà không
    phải giả lập `sys.argv` (trong Colab, `sys.argv` là tham số của kernel)."""
    du_doan.in_utf8()

    ap = argparse.ArgumentParser(
        description="Trang web thử mô hình đếm nhà ngập (Gradio)")
    ap.add_argument("--config",
                    default="configs/mmdet/cascade_convnext_t_floodnet.py")
    ap.add_argument("--data-yaml", default="configs/data.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="Mặc định tự tìm bản best trong work_dir")
    ap.add_argument("--nguong-diem", type=float, default=0.3,
                    help="Giá trị khởi đầu của thanh trượt (mặc định 0.3)")
    ap.add_argument("--cong", type=int, default=None,
                    help="Cổng cục bộ (mặc định: Gradio tự chọn cổng còn trống)")
    ap.add_argument("--khong-share", action="store_true",
                    help="Không tạo link công khai (chỉ mở trong Colab)")
    ap.add_argument("--thiet-bi", default="cuda:0")
    args = ap.parse_args(argv)

    print("=" * 72)
    print("WEB THỬ MÔ HÌNH — CASCADE R-CNN + CONVNEXT-TINY TRÊN FLOODNET")
    print("=" * 72)

    try:
        from mmengine.config import Config
        import mmdet
        import torch
    except ImportError as e:
        print(f"\n[!] Thiếu thư viện: {e}")
        print("    Trên Colab: chạy ô [3.3b] (cài môi trường) trước.")
        raise SystemExit(1)
    try:
        import gradio as gr
    except ImportError:
        print("\n[!] Thiếu gradio. Cài bằng:\n    pip install -q gradio\n"
              "    (Colab thường có sẵn)")
        raise SystemExit(1)

    print(f"mmdet {mmdet.__version__} | gradio {gr.__version__} | "
          f"torch {torch.__version__}")

    cfg = Config.fromfile(args.config)
    ten_lop = list((cfg.get("metainfo") or {}).get("classes", []))
    if not ten_lop:
        print("[!] Config không khai `metainfo.classes`.")
        raise SystemExit(1)
    try:
        ten_lop_yaml = doc_classes_tu_data_yaml(args.data_yaml)
        if ten_lop_yaml != ten_lop:
            print(f"[!] Tên lớp trong config {ten_lop} KHÁC trong "
                  f"{args.data_yaml} {ten_lop_yaml}.")
    except ValueError as e:
        print(f"[!] {e}")

    duong_ckpt = args.checkpoint or tim_checkpoint(cfg.work_dir)
    if duong_ckpt is None:
        print(f"\n[!] Không tìm thấy checkpoint trong work_dir:\n"
              f"    {cfg.work_dir}\n"
              f"    Chạy ô [3.11] (train thật) trên Colab trước.")
        raise SystemExit(1)
    print(f"checkpoint: {duong_ckpt}")

    if _train.va_torch_load_resume():
        print("  [vá] torch.load: đã cho phép HistoryBuffer + mảng numpy + lớp "
              "dtype + getattr (docs/NOTES.md §3.13).")
    model = du_doan.nap_model(cfg, duong_ckpt, args.thiet_bi)
    # Ảnh từ trình duyệt là mảng numpy, không có đường dẫn — bước đầu của
    # pipeline phải là LoadImageFromNDArray.
    pipeline = du_doan.pipeline_suy_luan(cfg, dau_vao_bang_mang=True)

    giao_dien = dung_giao_dien(model, pipeline, ten_lop, args.nguong_diem)
    print("\nĐang mở trang web... (lần đầu mất vài giây)")
    giao_dien.queue().launch(
        share=not args.khong_share,
        server_port=args.cong,
        # Trả quyền điều khiển lại cho notebook ngay, để ô Colab chạy xong mà
        # trang web vẫn sống trong luồng nền.
        prevent_thread_lock=True,
    )
    print("\n" + "=" * 72)
    print("Trang web đang chạy. Mở link ở trên:")
    print("  - link https://...gradio.live là link CÔNG KHAI, mở được trên "
          "điện thoại")
    print("  - link localhost chỉ mở được từ máy đang chạy Colab")
    print("Link sống chừng nào phiên Colab còn chạy. Ngắt phiên là link chết — "
          "mở lại bằng cách chạy lại ô này.")


if __name__ == "__main__":
    main()

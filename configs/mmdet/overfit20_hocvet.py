# ===========================================================================
# CONFIG CHẨN ĐOÁN — HỌC VẸT 20 ẢNH VỚI "ĐIỀU KIỆN DỄ NHẤT CÓ THỂ" (GATE 3)
# ===========================================================================
# Vì sao có tệp này: lần chạy [3.10] ngày 07/10/2026 (config overfit20.py) đã
# chạy hết 60 epoch trên Colab A100, tiền kiểm ĐẠT HẾT (bản sửa §3.11 đã sống),
# loss không NaN — nhưng mAP dừng ở 0,618 (mAP50 0,888), dưới xa mốc 0,9 mà
# chính overfit20.py đặt ra ("thấy mAP > 0.9 là đủ kết luận"), và 5 epoch cuối
# chỉ nhích +0,005 nên KHÔNG phải "cần thêm epoch". Số đo đầy đủ: docs/NOTES.md
# §3.12.
#
# Câu hỏi còn treo sau [3.10]: đường ống CÓ khả năng học vẹt hay không, hay có
# lỗi thật (nhãn, box, hàm loss, eval)? [3.10] chưa trả lời được vì hai thứ có
# thể che mất khả năng học vẹt vẫn đang bật:
#   1. Tăng cường quá mạnh cho một bài "thuộc lòng": mỗi epoch, cùng một ảnh lại
#      vào ở một tỉ lệ khác (5 mức), lật 3 hướng với xác suất 0,75, đổi sáng —
#      model phải học bất biến theo tỉ lệ từ 20 ảnh TRƯỚC KHI thuộc lòng được.
#   2. Lịch LR tự bóp: milestones [40, 55] trên 60 epoch → epoch 41-55 chạy ở
#      LR 1e-5 (15 epoch) và 56-60 ở 1e-6 (5 epoch), trong khi cả lần chạy chỉ
#      có 600 vòng lặp.
#
# Tệp này giữ NGUYÊN dữ liệu, nhãn, kiến trúc, hàm loss và cách đánh giá — chỉ
# đổi hai thứ trên thành điều kiện dễ nhất có thể:
#   - pipeline train = y hệt pipeline lúc ĐÁNH GIÁ (1 tỉ lệ 1536x1152, không
#     lật, không đổi sáng) → train và eval nhìn ảnh giống hệt nhau.
#   - LR ×10 (1e-3) + MỘT mốc giảm LR ở epoch 30 trên 40 epoch → không bị bóp
#     giữa chừng như [3.10].
#
# Vì thế phép thử này có giá trị PHÂN BIỆT (đây là toàn bộ mục đích của nó):
#   - mAP > 0,95  -> đường ống đúng. Thứ chặn [3.10] là recipe (LR/tăng cường),
#     không phải lỗi dữ liệu -> GATE 3 ĐẠT, bàn tiếp LR + số epoch cho train thật.
#   - vẫn ~0,6x và bão hoà -> lúc đó mới kết luận "lỗi thật" nằm trong
#     nhãn/box/loss/eval; log lần này in thêm AP theo TỪNG LỚP để đào tiếp.
#
# ĐỔI HAI THỨ CÙNG LÚC LÀ CỐ Ý: câu hỏi của phép thử là "đường ống có học vẹt
# ĐƯỢC không", không phải "yếu tố nào đang chặn". Tách riêng từng biến là việc
# của Phase 4 (chọn recipe cho train thật), không phải việc của GATE.
#
# ---------------------------------------------------------------------------
# GIAI ĐOẠN 1 (07/10/2026, 40 epoch) ĐÃ CHẠY — vì sao có GIAI ĐOẠN 2
# ---------------------------------------------------------------------------
# Chạy sạch 40/40 epoch trên A100 (thoát 0, ~6 phút rưỡi), tiền kiểm ĐẠT HẾT.
# mAP cuối 0,742 (mAP50 0,905); AP từng lớp: flooded 0,812 / non_flooded 0,673.
# Hai điều đo được từ log:
#   - Bước ngoặt nằm ĐÚNG ở mốc [30]: 30 epoch ở 1e-3 chỉ loanh quanh ~0,4;
#     vừa hạ xuống 1e-4 thì 10 epoch leo +0,37 (0,369 -> 0,742). Hoá ra nửa
#     "LR ×10" của phép chẩn đoán phản tác dụng — LR chạy được là 1e-4, đúng
#     base_lr của config chính. Nửa "tắt tăng cường" không tách được ở lần này
#     (cố ý), nhưng đối chiếu với [3.10] là đủ thấy recipe — không phải nhãn /
#     box / loss / eval — mới là thứ đang chặn.
#   - Đuôi đường cong VẪN LEO (+0,02 trong 4 epoch cuối, so với +0,005/5 epoch
#     của [3.10]) — lần này HẾT EPOCH, không phải BÃO HOÀ. Vì vậy chưa được
#     phép kết luận "trần thật", mà cũng chưa đủ mốc để kết luận ĐẠT.
#
# GIAI ĐOẠN 2 chỉ đổi MỘT thứ: gia hạn 40 -> 120 epoch (max_epochs dưới đây).
# Chạy lại đúng ô [3.10b]: `resume=True` tự nạp checkpoint epoch 40 và chạy
# tiếp 41..120, giữ nguyên base_lr 1e-4 (mốc [30] đã qua — dù lịch được nạp
# từ checkpoint hay dựng lại từ config thì cũng vậy). Xác nhận trong log 2-3
# epoch đầu: `base_lr: 1.0000e-04` và mAP ~0,7x, KHÔNG phải ~0 (thấy ~0 là
# resume hỏng, dừng báo ngay).
#
# Chốt đọc giai đoạn 2 (giữ nguyên tinh thần chốt trước, không dịch mốc):
#   - mAP > 0,95  -> GATE 3 ĐẠT, bàn tiếp LR + số epoch cho train thật.
#   - bão hoà < 0,9 (10 epoch liền nhích < 0,01) -> trần thật; đào tiếp bằng
#     AP từng lớp (non_flooded đang thua flooded 0,14) + ảnh vis_data.
#   - 0,9-0,95, hoặc hết 120 epoch mà còn leo -> gửi số liệu, quyết định cùng.
# Thấy > 0,95 là DỪNG ĐƯỢC: checkpoint epoch vừa xong đã ghi ra Drive rồi.
#
# Thời lượng: giai đoạn 1 (40 epoch) mất ~6-7 phút trên A100; giai đoạn 2
# (thêm 80 epoch) ~13 phút trên A100, ~50-60 phút trên T4.
#
# ⚠️ Vẫn là điểm trên tập TRAIN. Tuyệt đối không trích dẫn mAP của tệp này vào
# báo cáo như một kết quả của đồ án (xem đầu tệp overfit20.py).
# ===========================================================================

# Kế thừa nguyên phần dữ liệu (ann_file/data_prefix/work_dir/visualization),
# chỉ ghi đè những gì liệt kê dưới đây. mmengine gộp dict theo chiều sâu nên
# `train_dataloader = dict(dataset=dict(pipeline=[...]))` GIỮ NGUYÊN ann_file và
# data_prefix mà overfit20.py đã khai tường minh (chúng được test_train.py mục 5
# khoá lại); danh sách `pipeline` thì bị THAY THẾ hoàn toàn, không gộp phần tử.
_base_ = ['./overfit20.py']

# ---------------------------------------------------------------------------
# 1. Pipeline train: bỏ TOÀN BỘ tăng cường
# ---------------------------------------------------------------------------
# Ba thứ bị bỏ so với pipeline của config chính, và vì sao:
#   - RandomChoiceResize (5 tỉ lệ) -> Resize 1 tỉ lệ ĐÚNG BẰNG lúc đánh giá.
#     Giữ nguyên `keep_ratio=True` — thiếu nó là ảnh bị bóp méo mà không có gì
#     báo (xem chú thích dài trong config chính).
#   - RandomFlip: lật ngang/dọc/chéo. Ảnh hàng không nhìn từ trên xuống nên lật
#     là hợp lý cho train thật, nhưng ở phép thử "thuộc lòng" thì nó bắt model
#     học thêm 4 biến thể của CÙNG 20 ảnh — đúng thứ cần gỡ ra để câu hỏi sạch.
#   - TangSangNhe: đổi sáng/tương phản. Cùng lý do.
# Thứ tự giữ như pipeline train của config chính (LoadAnnotations trước Resize;
# mmdet's Resize tự scale cả box).
train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1536, 1152), keep_ratio=True),
            dict(type='PackDetInputs')
        ]))

# val_dataloader KHÔNG ghi đè: giữ nguyên của overfit20.py (20 ảnh train +
# test_pipeline). Nhờ vậy con số mAP của lần này so được TRỰC TIẾP với [3.10] —
# chỉ recipe train đổi, phép đo không đổi.

# ---------------------------------------------------------------------------
# 2. Lịch train: LR chỉ giảm MỘT lần ở epoch 30 (40 -> 120 epoch ở giai đoạn 2)
# ---------------------------------------------------------------------------
# Giai đoạn 1 (đã chạy, 40 epoch): 300 vòng đầu ở 1e-3, 100 vòng cuối ở 1e-4.
# Giai đoạn 2 (chạy tiếp bằng resume): 800 vòng ở 1e-4. Warmup 50 vòng của
# overfit20.py (5 epoch đầu) giữ nguyên — LR 1e-3 ngay từ vòng đầu dễ làm hỏng
# backbone vừa nạp trọng số ImageNet.
max_epochs = 120
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

# ---------------------------------------------------------------------------
# 3. LR ×10 — chỉ đổi ĐÚNG khoá `lr`, giữ nguyên AdamW/betas/weight_decay và cả
# LearningRateDecayOptimizerConstructor (giảm LR theo tầng backbone) của config
# chính. `accumulative_counts=1` thừa hưởng từ overfit20.py (không tích luỹ).
# Nếu lần chạy này nổ NaN (log sẽ in rõ) thì hạ xuống 5e-4 — nhưng với warmup 50
# vòng và AMP có cơ chế bỏ bước khi tràn số, rủi ro thấp.
# ---------------------------------------------------------------------------
optim_wrapper = dict(optimizer=dict(lr=0.001))

# ---------------------------------------------------------------------------
# 4. Thêm AP theo TỪNG LỚP vào log
# ---------------------------------------------------------------------------
# [3.10] chỉ in mAP gộp. Nếu lớp flooded_building đạt 0,9 còn non_flooded_building
# quanh 0,3 thì vấn đề là ở nhãn/ranh giới hai lớp; còn cả hai cùng ~0,6 thì là
# chuyện học nói chung. Một dòng log, khỏi phải chạy lại để biết.
val_evaluator = dict(classwise=True)

# ---------------------------------------------------------------------------
# 5. Thư mục riêng — KHÔNG được lẫn với lần chạy [3.10]
# ---------------------------------------------------------------------------
# Lẫn work_dir thì `resume=True` sẽ tự chạy tiếp từ checkpoint CŨ và cả phép
# chẩn đoán thành vô nghĩa (nó sẽ chạy tiếp lịch cũ, không phải lịch 40 epoch).
work_dir = '/content/drive/MyDrive/Flood_House_AI/runs/sanity/overfit20_hocvet'

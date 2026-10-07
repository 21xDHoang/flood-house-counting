# ===========================================================================
# CONFIG SANITY CHECK — OVERFIT 20 ẢNH (GATE 3)
# ===========================================================================
# Mục đích: trả lời câu hỏi "model và pipeline có học được không?" trước khi
# đốt 6-8 giờ GPU cho lần train thật.
#
# Cách làm: lấy 20 ảnh của tập train (do `scripts/tao_overfit20.py` chọn và ghi
# ra `annotations/instances_overfit20.json`), rồi train và ĐÁNH GIÁ TRÊN CHÍNH
# 20 ẢNH ĐÓ. Đây là bài kiểm tra cố ý "học vẹt": nếu kiến trúc, hàm loss, nhãn
# và pipeline đều đúng thì model phải thuộc lòng được 20 ảnh này và mAP tiến gần
# 1.0. Nếu mAP quanh quẩn ở mức thấp sau vài trăm vòng lặp thì gần như chắc chắn
# có lỗi ở đâu đó (nhãn sai, box lệch, learning rate hỏng), KHÔNG phải do dữ
# liệu ít — và lúc đó phải dừng, đừng mang đi train 24 epoch.
#
# NGƯỢC LẠI, cũng phải nói rõ để không ngộ nhận: overfit 20 ảnh ĐẠT **không**
# chứng minh model sẽ phân loại tốt nhà ngập / không ngập. Nó chỉ chứng minh
# đường ống chạy đúng. Độ chính xác thật đo ở Phase 5 trên tập val.
#
# Đây là bước 7 của Phase 3 trong plan, và plan ghi rõ bước này là BẮT BUỘC.
# ===========================================================================

_base_ = ['./cascade_convnext_t_floodnet.py']

# Tệp annotation do scripts/tao_overfit20.py sinh ra, nằm cùng thư mục
# `annotations/` với 3 tệp của Phase 2. Ảnh dùng lại nguyên `images/train/`,
# KHÔNG copy 20 ảnh sang chỗ khác — copy ra là lại phải quản lý thêm một bản sao
# có thể lệch với bản gốc.
#
# Vì sao đường dẫn viết THẲNG ở đây, không lấy từ biến của config cha:
# mmengine exec tệp config con trong một namespace chỉ có ĐÚNG MỘT tên là
# `_base_`, nên `data_root` ở đây là biến không tồn tại (NameError). Cách duy
# nhất để với tới biến của cha là cú pháp hai-ngoặc-nhọn — đã thử và HỎNG, và
# hỏng IM LẶNG: viết `'{{_base_.ann_overfit}}'` trong ngoặc kép thì mmengine
# thay phần trong ngoặc bằng một placeholder đã bọc sẵn ngoặc kép, giá trị
# chuỗi thành ra chứa cả ngoặc, phép tra cứu trượt, và đường dẫn hoá thành tên
# rác `"_ann_overfit_xxxxxx"` mà không có lỗi nào báo. Đã kiểm bằng source
# mmengine 0.10.7 và bằng lần chạy [3.10] ngày 01/10/2026 — chi tiết ở
# docs/NOTES.md §3.11.
#
# Hai dạng đường dẫn dưới đây cố ý KHÁC nhau, theo đúng cách config cha làm:
#   * dataloader — TƯƠNG ĐỐI: mmdet ghép với `data_root` thừa hưởng từ config
#     cha (`/content/floodnet_coco`), nên gốc đường dẫn không chép lại ở đây.
#   * val_evaluator — TUYỆT ĐỐI: CocoMetric mở thẳng tệp bằng pycocotools,
#     không biết `data_root` là gì (config cha cũng vì thế mà dùng `ann_val`
#     tuyệt đối).
# tests/test_train.py mục 5 khoá ba đường dẫn phải trỏ về CÙNG một tệp, và tệp
# đó phải khớp `data_root` của config cha.

# ---------------------------------------------------------------------------
# Dữ liệu: cả train và val đều trỏ vào 20 ảnh đó
# ---------------------------------------------------------------------------
# ⚠️ `data_prefix` PHẢI khai TƯỜNG MINH ở đây, cả hai dataloader. Không khai là
# mmengine gộp dict theo chiều sâu rồi để nó THỪA HƯỞNG từ config cha: train thì
# tình cờ đúng (`images/train/`), nhưng val thừa hưởng `images/val/` — trong khi
# 20 ảnh overfit lấy từ split train.
#
# Vì sao lỗi này khó thấy: `CocoDataset.parse_data_info` của mmdet 3.3.0 dựng
# đường dẫn bằng `osp.join(self.data_prefix['img'], img_info['file_name'])`, mà
# `file_name` do Phase 2 ghi ra là tên PHẲNG có tiền tố split (`train_10168.jpg`)
# — nên chỉ có `data_prefix` quyết định thư mục. mmdet KHÔNG kiểm tra ảnh có tồn
# tại lúc dựng dataset (đã đọc source, không có `check_file_exist` nào trên
# đường dẫn ảnh), nên tiền kiểm vẫn ĐẠT HẾT, train vẫn chạy, và chỉ tới bước
# validate ở cuối epoch 1 mới nổ FileNotFoundError.
train_dataloader = dict(
    # batch 2 (bằng config chính) nhưng KHÔNG tích luỹ gradient (xem
    # optim_wrapper bên dưới): 20 ảnh / batch 2 = 10 vòng lặp mỗi epoch, cần
    # nhiều bước cập nhật thì mới hội tụ kịp trong thời gian chấp nhận được.
    dataset=dict(
        # TƯƠNG ĐỐI — mmdet ghép với `data_root` thừa hưởng từ config cha.
        ann_file='annotations/instances_overfit20.json',
        data_prefix=dict(img='images/train/')))

val_dataloader = dict(
    dataset=dict(
        # TƯƠNG ĐỐI, cùng tệp với train — phép thử này ĐÁNH GIÁ TRÊN CHÍNH 20
        # ảnh đã train (xem chú thích đầu tệp).
        ann_file='annotations/instances_overfit20.json',
        data_prefix=dict(img='images/train/')))

# Đánh giá trên chính tập vừa train — đây là điểm mấu chốt của phép kiểm này.
# Lưu ý: chỉ dùng để chẩn đoán, TUYỆT ĐỐI không trích dẫn con số mAP này vào
# báo cáo như một kết quả của đồ án (nó là điểm trên tập train).
val_evaluator = dict(
    # TUYỆT ĐỐI — CocoMetric mở thẳng tệp bằng pycocotools, không biết
    # `data_root` là gì. Đổi `data_root` ở config cha thì phải đổi cả dòng này
    # (tests/test_train.py mục 5 sẽ đỏ nếu quên).
    ann_file='/content/floodnet_coco/annotations/instances_overfit20.json')

# ---------------------------------------------------------------------------
# Lịch train
# ---------------------------------------------------------------------------
# 60 epoch = 600 vòng lặp. Đủ để thuộc 20 ảnh trong điều kiện bình thường; nếu
# chạy tới cuối mà mAP vẫn thấp thì là lỗi thật, không phải "cần thêm epoch".
# Có thể dừng sớm: notebook 03 in log từng epoch, thấy mAP > 0.9 là đủ kết luận.
max_epochs = 60
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    # PHẢI khai báo lại: config chính warmup 1000 vòng lặp, mà cả lần chạy này
    # chỉ có 600 vòng — giữ nguyên thì model chưa bao giờ ra khỏi warmup, LR
    # vẫn ở mức 1/1000 và kết quả sẽ rất thấp. Nhìn bề ngoài y hệt "dữ liệu có
    # vấn đề", nên đây là cái bẫy phải nhớ.
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40, 55],
        gamma=0.1)
]

# ---------------------------------------------------------------------------
# Optimizer: bỏ tích luỹ gradient
# ---------------------------------------------------------------------------
optim_wrapper = dict(
    # accumulative_counts 4 -> 1 so với config chính. Config chính dùng batch
    # hiệu dụng 8 để VRAM chứa nổi ảnh 1536px khi chạy 1.445 ảnh. Ở đây chỉ 20
    # ảnh nên vấn đề duy nhất là SỐ BƯỚC CẬP NHẬT: tích luỹ 4 làm số bước giảm
    # 4 lần (còn 150) thì khó kịp overfit.
    accumulative_counts=1)

# ---------------------------------------------------------------------------
# Ghi kết quả
# ---------------------------------------------------------------------------
# Thư mục riêng, không lẫn với lần train thật (e1_...) — nếu lẫn thì rất dễ
# nhầm checkpoint của lần chạy 20 ảnh sang làm kết quả đồ án.
work_dir = '/content/drive/MyDrive/Flood_House_AI/runs/sanity/overfit20'

default_hooks = dict(
    checkpoint=dict(
        # Giữ 1 bản gần nhất + 1 bản tốt nhất: lần chạy này chỉ để xem đường
        # ống có chạy không, không cần giữ nhiều bản như lần train thật.
        max_keep_ckpts=1),
    # BẬT vẽ ảnh ở lần chạy này (config chính để tắt). Đây là bằng chứng bằng
    # mắt cho GATE 3: mở ảnh trong work_dir/vis_data/... để thấy model vẽ box
    # đúng chỗ, thay vì chỉ tin vào con số mAP. interval=20 để vẽ ở epoch
    # 20/40/60 chứ không phải mỗi epoch (mỗi lần vẽ là 20 ảnh ghi lên Drive).
    visualization=dict(type='DetVisualizationHook', draw=True, interval=20))

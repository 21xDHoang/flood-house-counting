# ===========================================================================
# CONFIG CHÍNH CỦA ĐỒ ÁN — Cascade R-CNN (chỉ box) + ConvNeXt-Tiny + FPN
# ===========================================================================
# Kế thừa config CHÍNH THỨC của mmdet 3.3.0:
#   configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py
#
# Cú pháp `mmdet::<đường dẫn trong gói>` là của mmengine (hàm
# `_get_external_cfg_base_path`), nó trỏ tới
#   <site-packages>/mmdet/.mim/configs/<đường dẫn>
# ĐÃ KIỂM CHỨNG BẰNG CÁCH MỞ THẬT FILE WHEEL `mmdet-3.3.0-py3-none-any.whl` tải từ
# PyPI: trong wheel CÓ `mmdet/.mim/configs/` (1.564 mục), gồm đúng config ConvNeXt
# này và cả 3 config `_base_` mà nó kế thừa. Vì vậy KHÔNG cần clone repo mmdet,
# và tuyệt đối không copy config gốc về repo — bản gốc vẫn là thứ đối chiếu được
# với OpenMMLab, còn mọi thay đổi của đồ án nằm gọn trong tệp này.
#
# ---------------------------------------------------------------------------
# KHÁC GÌ SO VỚI BẢN GỐC — đọc kỹ trước khi bảo vệ
# ---------------------------------------------------------------------------
# 1. BỎ TOÀN BỘ NHÁNH MASK. Bản gốc là Cascade *Mask* R-CNN (có mask_roi_extractor
#    + mask_head, `pad_mask=True`). Đồ án chỉ cần BOX để ĐẾM nhà; nhãn FloodNet
#    tuy là mask nhưng đã chuyển thành box ở Phase 2. Giữ mask head chỉ tốn thêm
#    VRAM và thời gian mà không phục vụ câu hỏi nào của đồ án.
#    => phải `_delete_=True` ở roi_head: nếu chỉ khai báo đè vài khoá thì
#       mask_roi_extractor/mask_head của bản gốc VẪN CÒN (mmengine gộp dict theo
#       chiều sâu) và model lặng lẽ trở thành model có mask.
# 2. `num_classes` 80 -> 2 ở CẢ BA bbox head. Bản gốc để 80 (COCO). Quên đổi một
#    trong ba là lỗi rất hay gặp và nó không báo lỗi ngay — chỉ hiện ra dưới dạng
#    mAP thấp khó hiểu. `scripts/train.py` in lại cả ba giá trị này ở phần tiền
#    kiểm (mục [4]) và DỪNG nếu có tầng nào lệch, thay vì để train xong mới thấy.
# 3. BỎ tăng cường kiểu DETR của bản gốc (RandomCrop + 11 mức tỉ lệ quanh
#    (1333, 800) + crop tuyệt đối). Ảnh đồ án đã được resize sẵn về cạnh dài 1536
#    ở Phase 2, nên pipeline ở đây chỉ đa tỉ lệ NHẸ quanh mức đó (xem
#    `scales_train`), không mosaic/mixup như plan yêu cầu.
# 4. Đổi cấu hình dataset, số epoch, learning rate, và bật/tắt vài hook — giải
#    thích tại từng chỗ bên dưới.
#
# Quy tắc của đồ án: mọi tham số nằm trong file config, không hard-code trong code.
# ===========================================================================

_base_ = [
    'mmdet::convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py'  # noqa: E501
]

# --- Đăng ký module ngoài registry của mmdet ---------------------------------
# PHẢI viết lại CẢ danh sách: mmengine thay thế list chứ không gộp, nên nếu chỉ
# ghi module của mình thì 'mmpretrain.models' biến mất -> backbone ConvNeXt
# không đăng ký được -> chết ngay khi dựng model.
custom_imports = dict(
    imports=['mmpretrain.models', 'floodcount.models.transforms'],
    allow_failed_imports=False)

# ===========================================================================
# 1. ĐƯỜNG DẪN
# ===========================================================================
# `data_root` PHẢI khớp `paths.processed_dir` trong configs/data.yaml — nơi
# Phase 2 dựng dataset. Hai chốt giữ điều đó: `tests/test_train.py` mục 6 đối
# chiếu hai tệp này ngay trên máy (không cần Colab), và `scripts/train.py` kiểm
# lại lúc chạy kèm phép thử thư mục có thật.
data_root = '/content/floodnet_coco'

# Checkpoint ghi THẲNG ra Drive: /content mất sạch khi Colab ngắt phiên. Đổi
# đường dẫn này qua `--work-dir` khi chạy thí nghiệm khác (E2, E3...).
work_dir = '/content/drive/MyDrive/Flood_House_AI/runs/train/e1_cascade_convnext_t'

# --- Đường dẫn ĐẦY ĐỦ của các tệp annotation, khai báo một chỗ -------------
# Vì sao phải đặt tên thay vì viết thẳng: configs/mmdet/overfit20.py kế thừa
# config này và phải trỏ `val_evaluator` sang tệp instances_overfit20.json.
# mmengine KHÔNG đưa biến của config cha vào phạm vi của config con — đã đọc
# source (`_file2dict`): tệp con được exec trong namespace chỉ có ĐÚNG MỘT tên
# là `_base_`. Muốn dùng lại biến của cha thì phải qua cú pháp `_base_` đặt
# trong hai cặp ngoặc nhọn, mà cú pháp đó chỉ thay được khi **cả chuỗi** đúng
# bằng cú pháp đó — không ghép thêm chữ được (ghép vào là ra một chuỗi rác và
# không có lỗi nào báo). Vì vậy tính sẵn đường dẫn đầy đủ ở đây, config con chỉ
# việc lấy.
ann_val = data_root + '/annotations/instances_val.json'
ann_test = data_root + '/annotations/instances_test.json'
ann_overfit = data_root + '/annotations/instances_overfit20.json'

# ===========================================================================
# 2. HAI LỚP CỦA ĐỒ ÁN
# ===========================================================================
# THỨ TỰ VÀ TÊN ĐỀU QUAN TRỌNG — đã đọc source mmdet 3.3.0 để chắc chắn:
#
#   mmdet/datasets/coco.py:  self.cat_ids = self.coco.get_cat_ids(
#                                cat_names=self.metainfo['classes'])
#                            self.cat2label = {cat_id: i
#                                              for i, cat_id in enumerate(self.cat_ids)}
#
#   get_cat_ids gọi thẳng pycocotools, mà getCatIds trả về
#   `[cat['id'] for cat in dataset['categories']]` — tức giữ nguyên THỨ TỰ
#   CATEGORY TRONG FILE JSON, chỉ lọc theo tên. File COCO của Phase 2 ghi
#   category theo id tăng dần (1 = flooded_building, 2 = non_flooded_building).
#
# Nên có hai kiểu hỏng, cả hai đều KHÔNG báo lỗi:
#   (a) Tên ở đây không khớp tên trong JSON (gõ sai, hoặc JSON đổi tên) →
#       get_cat_ids không tìm thấy tên đó → category bị loại khỏi cat_ids →
#       `parse_data_info` gặp `if ann['category_id'] not in self.cat_ids:
#       continue` và BỎ IM LẶNG mọi box của lớp đó. Model vẫn chạy, chỉ học
#       một lớp.
#   (b) Thứ tự tên ở đây lệch với thứ tự category trong JSON → nhãn 0/1 gán
#       ngược so với tên: bảng AP theo lớp và bảng đếm nhà cuối cùng (Phase 5)
#       in ra SAI TÊN, dù model học đúng.
# Vì vậy `scripts/train.py` đối chiếu `metainfo.classes` với chính file JSON
# lúc bắt đầu chạy, đồng thời đếm lại số box mmdet thực sự nhận được và so với
# số annotation trong JSON — dừng ngay nếu lệch, thay vì train xong mới phát hiện.
metainfo = dict(classes=('flooded_building', 'non_flooded_building'))
num_classes = 2

# ===========================================================================
# 3. MODEL
# ===========================================================================
# Backbone ConvNeXt-Tiny pretrain ImageNet + FPN + RPN + Cascade ROI head 3 tầng
# lấy nguyên từ bản gốc (không đổi), chỉ thay phần đầu ROI.
model = dict(
    # Bản gốc `pad_mask=True` để đệm cả mask theo pad_size_divisor. Không còn
    # mask nào trong pipeline nên tắt đi cho đúng thực tế.
    data_preprocessor=dict(pad_mask=False),

    roi_head=dict(
        _delete_=True,          # xoá sạch roi_head của bản gốc (có mask head)
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        # RoIAlign 7x7 trên 4 tầng FPN (stride 4, 8, 16, 32) — như bản gốc.
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # Ba tầng head giống hệt nhau TRỪ `target_stds`: tầng sau càng siết chặt
        # độ lệch cho phép của box, vì box vào tầng sau đã chính xác hơn. Đây là
        # cơ chế lõi của Cascade, không được "cho giống nhau cho đẹp".
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                # SyncBN: bản gốc dùng cho 8 GPU. Trên 1 GPU nó chạy như BN
                # thường, giữ nguyên để không lệch khỏi công thức đã công bố.
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                # GIoU thay SmoothL1: bản gốc chọn GIoU (tên config có 'giou'),
                # đo độ trùng khớp của cả box chứ không đo từng toạ độ rời rạc.
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]),

    # -----------------------------------------------------------------------
    # train_cfg — viết lại toàn bộ để bỏ mask_size của bản gốc
    # -----------------------------------------------------------------------
    train_cfg=dict(
        _delete_=True,
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                # match_low_quality=True + min_pos_iou=0.3: nhà to của FloodNet
                # (trung vị cạnh 186px, p90 348px — đo ở Phase 2) hiếm khi đạt
                # IoU 0.7 với anchor nào, nên nếu tắt cờ này thì những box to
                # nhất gần như không bao giờ được gán nhãn dương ở tầng RPN.
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        # max_per_img 2000 -> 1000: ảnh chỉ rộng 1536px, số nhà trên một ảnh
        # nhiều nhất chỉ vài chục, nên 2000 đề xuất là thừa; 1000 vẫn dư sức
        # nhưng giảm được khối lượng NMS ở mỗi vòng lặp.
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),

    # -----------------------------------------------------------------------
    # test_cfg — viết lại toàn bộ để bỏ mask_thr_binary
    # -----------------------------------------------------------------------
    test_cfg=dict(
        _delete_=True,
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            # score_thr thấp (0.05) là CỐ Ý: đây chưa phải ngưỡng đếm cuối
            # cùng. Ngưỡng đếm được chọn trên tập val ở Phase 5, KHÔNG chọn ở
            # đây và tuyệt đối không chọn trên test.
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            # 100 (mặc định COCO) -> 300. Nhà trong một ảnh FloodNet nhiều nhất
            # là vài chục box, nhưng vì score_thr để 0.05 nên số box thô lớn hơn
            # nhiều; 300 là mức dư an toàn mà vẫn chặn được trường hợp model
            # "bung" hàng nghìn box rác. Plan cũng chốt mức ~300.
            max_per_img=300),
    ))

# ===========================================================================
# 4. PIPELINE
# ===========================================================================
# Cạnh dài của ảnh SAU Phase 2 đã là 1536. Đa tỉ lệ ở đây chỉ đi XUỐNG, không
# vượt 1536: ảnh gốc không còn chi tiết nào để phóng to, mà VRAM thì tốn thêm.
# Mức nhỏ nhất 1024 = 2/3 mức gốc, dạy model chịu được nhà nhỏ hơn một chút.
#
# Phần tử là (cạnh dài, cạnh ngắn); mmdet dùng max()/min() của tuple nên thứ tự
# trong ngoặc không quan trọng, nhưng ghi theo thứ tự này cho dễ đọc.
scales_train = [(1536, 1152), (1408, 1056), (1280, 960), (1152, 864), (1024, 768)]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    # CHỈ box, KHÔNG with_mask: nhãn mask không được nạp vào lúc train.
    dict(type='LoadAnnotations', with_bbox=True),
    # `keep_ratio=True` BẮT BUỘC phải ghi, không được bỏ:
    # RandomChoiceResize là lớp của mmcv (không phải mmdet) và nó tự dựng một
    # đối tượng `Resize` qua registry gốc của mmengine. `Resize` của mmcv có
    # `keep_ratio` mặc định **False**, ngược với `Resize` của mmdet (mặc định
    # True). Bỏ tham số này thì ảnh bị bóp về đúng (1536, 1152) — méo tỉ lệ —
    # mà box vẫn được scale theo scale_factor từng trục nên KHÔNG có gì báo
    # lỗi: model học trên ảnh nhà bị kéo giãn.
    dict(type='RandomChoiceResize', scales=scales_train, keep_ratio=True),
    # Lật ngang + dọc + chéo, mỗi hướng 0.25. Vì sao không xoay 90°: mmdet
    # 3.3.0 KHÔNG có transform xoay 90° (đã đọc source: RandomFlip chỉ nhận
    # 'horizontal' | 'vertical' | 'diagonal', và 'diagonal' là lật cả hai trục
    # tức xoay 180°, không phải 90°). Xoay 90° trong plan ghi rõ là "nếu
    # Albumentations có sẵn" — chưa xác nhận được Colab có gói này, mà thêm
    # một phụ thuộc mới cho một phép tăng cường chưa đo được lợi ích thì không
    # đáng. Ba hướng ở trên phủ 4/8 phép biến đổi nhị diện; xem docs/NOTES.md.
    dict(type='RandomFlip', prob=0.75,
         direction=['horizontal', 'vertical', 'diagonal']),
    # Transform TỰ VIẾT (src/floodcount/models/transforms.py). Lý do không dùng
    # PhotoMetricDistortion của mmdet: bản đó có nhánh ĐẢO KÊNH MÀU ngẫu nhiên
    # với xác suất 1/2 (đã đọc source: `if swap_flag: img = img[..., swap_value]`).
    # Với đồ án này màu nước là tín hiệu phân biệt nhà ngập với nhà không ngập,
    # đảo kênh là phá đúng tín hiệu cần học. Ở đây chỉ tăng/giảm sáng và tương
    # phản nhẹ, giữ nguyên màu.
    dict(type='TangSangNhe', delta_sang=24.0, tuong_phan=(0.85, 1.15), prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    # Resize tường minh về đúng mức đã chọn dù ảnh Phase 2 vốn đã ở mức đó:
    # val/test phải chạy ĐÚNG một tỉ lệ cố định, còn hơn dựa vào việc dữ liệu
    # tình cờ đã đúng kích thước.
    dict(type='Resize', scale=(1536, 1152), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# ===========================================================================
# 5. DATASET
# ===========================================================================
dataset_type = 'CocoDataset'
backend_args = None     # đọc file cục bộ trên /content, không qua backend lưu trữ

# Lọc ảnh không có box nào: mmdet mặc định `filter_empty_gt=True` (bỏ ảnh rỗng).
# Đồ án ĐỂ False, có chủ ý — đây là điểm khác bản gốc quan trọng:
#   Khoảng một NỬA dataset FloodNet không có căn nhà nào (chỉ nước/đường/cây).
#   Nếu bỏ các ảnh đó, model chỉ học "trong ảnh kiểu gì cũng có nhà", mà đầu ra
#   cuối cùng của đồ án lại là ĐẾM số nhà trên MỌI ảnh — kể cả ảnh không có nhà.
#   Giữ lại dạy model biết cảnh nào thì KHÔNG có nhà, tránh đếm thừa.
#   Số ảnh bị ảnh hưởng được đo và in ra ở scripts/kiem_tra_du_lieu.py.
#   Phase 6 sẽ có thí nghiệm bật/tắt để chứng minh bằng số liệu.
filter_cfg = dict(filter_empty_gt=False, min_size=32)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # Gom ảnh cùng tỉ lệ vào một batch: sau RandomChoiceResize mỗi ảnh có một
    # tỉ lệ khác nhau, gom lại thì bộ nhớ mỗi batch ổn định, tránh việc batch
    # toàn ảnh 1536px làm tràn VRAM.
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=metainfo,
        filter_cfg=filter_cfg,
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# metric CHỈ 'bbox': bản gốc (coco_instance) để ['bbox', 'segm'], và nếu giữ
# 'segm' thì CocoMetric sẽ đi tìm mask — mà dataset và model đều không có mask.
val_evaluator = dict(
    type='CocoMetric',
    ann_file=ann_val,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=ann_test,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

# ===========================================================================
# 6. LỊCH TRAIN
# ===========================================================================
# 24 epoch là MẶC ĐỊNH TẠM — GATE 3 yêu cầu đo thời gian 1 epoch trên Colab rồi
# người dùng chốt lại con số. Mốc giảm LR đặt theo tỉ lệ của bản gốc COCO
# (27/36 = 75%, 33/36 = 92%) chứ không copy nguyên [27, 33] của bản gốc.
max_epochs = 24

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    # Warmup 1000 vòng lặp: AdamW với LR lớn ngay từ vòng đầu dễ làm hỏng
    # backbone vừa nạp trọng số ImageNet. 1445 ảnh / batch 2 ≈ 723 vòng mỗi
    # epoch nên 1000 vòng ≈ 1,4 epoch — hợp lý.
    # LƯU Ý khi viết config khác: nếu tổng số vòng lặp của cả lần chạy nhỏ hơn
    # 1000 (ví dụ chạy overfit 20 ảnh) thì warmup sẽ chưa xong mà train đã hết,
    # LR gần như không tăng và model học được rất ít — nhìn bề ngoài y hệt lỗi
    # dữ liệu. configs/mmdet/overfit20.py vì thế phải khai báo lại mục này.
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# ===========================================================================
# 7. OPTIMIZER
# ===========================================================================
optim_wrapper = dict(
    # AMP: T4 có tensor core, dùng float16 cho phần tính toán và float32 cho
    # trọng số. Bắt buộc phải có nếu muốn batch/kích thước ảnh này vừa VRAM.
    type='AmpOptimWrapper',
    # Tích luỹ gradient 4 bước -> batch hiệu dụng 2 x 4 = 8. T4 15,6 GB không
    # chứa nổi batch 8 ở độ phân giải 1536px với Cascade 3 tầng.
    accumulative_counts=4,
    # Giảm LR theo từng tầng của backbone (tầng càng sâu LR càng nhỏ) — công
    # thức layer-wise của ConvNeXt, giữ nguyên từ bản gốc.
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg=dict(decay_rate=0.7, decay_type='layer_wise', num_layers=6),
    optimizer=dict(
        type='AdamW',
        # Bản gốc 2e-4 cho batch hiệu dụng 16 (8 GPU x 2). Đồ án batch hiệu
        # dụng 8 -> giảm nửa còn 1e-4, đúng tinh thần "LR tỉ lệ với batch".
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

# ===========================================================================
# 8. HOOK — chỗ tốn tiền và tốn dung lượng Drive nhất
# ===========================================================================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        # Giữ 2 checkpoint gần nhất + 1 best theo mAP. Không để 3 như plan vì
        # mỗi tệp ~216 MB (xem con số bên dưới) và Drive của đồ án đang chứa cả
        # dataset; xem docs/NOTES.md §3.4 để biết cách đổi lại.
        max_keep_ckpts=2,
        save_last=True,
        save_best='coco/bbox_mAP',
        rule='greater',
        # KHÔNG lưu trạng thái optimizer: model có ~54 triệu tham số, nên
        #   - chỉ trọng số      : 54e6 x 4 byte                ≈ 216 MB
        #   - + trạng thái AdamW: 54e6 x 4 x 2 (m và v)        ≈ 432 MB nữa
        # Ghi thêm 432 MB mỗi epoch lên Drive qua FUSE là quá đắt, trong khi lợi
        # ích duy nhất là resume mượt hơn. Đã đọc source mmengine 0.10.7 để
        # chắc: `Runner.resume()` có guard `if 'optimizer' in checkpoint`, nên
        # resume từ checkpoint không có optimizer KHÔNG sập — chỉ là các moment
        # của AdamW được khởi động lại, ảnh hưởng vài chục vòng lặp đầu.
        save_optimizer=False),
    logger=dict(type='LoggerHook', interval=50),
    # Nói rõ `draw=False` dù mặc định đã là vậy: mmdet 3.3.0 trong
    # `_base_/default_runtime.py` đặt `visualization=dict(type='DetVisualizationHook')`
    # KHÔNG kèm tham số nào, và DetVisualizationHook.__init__ có
    # `draw: bool = False, interval: int = 50` (đã đọc source) — nên mặc định
    # đã không vẽ gì. Ghi ra đây để người đọc config không phải đi tra, và để
    # nếu về sau có ai bật `draw=True` thì thấy ngay cảnh báo: hook này ghi 1
    # ảnh cho MỖI ảnh val (450 ảnh) vào work_dir, tức 450 tệp nhỏ ghi lên Drive
    # qua FUSE sau mỗi lần val. Ảnh để soi kết quả do Phase 5 vẽ có chủ đích.
    visualization=dict(type='DetVisualizationHook', draw=False))

# ===========================================================================
# 9. TÁI LẬP
# ===========================================================================
# Cố định seed theo quy tắc 8 của plan. `deterministic=False` là lựa chọn có ý
# thức: bật chế độ deterministic của cuDNN làm chậm đáng kể mà đồ án không cần
# chính xác đến từng bit, chỉ cần cùng seed thì kết quả xấp xỉ nhau.
randomness = dict(seed=42, deterministic=False)

# mmengine tự lưu bản config đã hợp nhất vào work_dir mỗi lần chạy, kèm cả các
# giá trị `--cfg-options` truyền từ dòng lệnh — đúng yêu cầu "lưu config vào thư
# mục kết quả của mỗi lần train để tái lập" trong plan.

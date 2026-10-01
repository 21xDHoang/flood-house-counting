# -*- coding: utf-8 -*-
"""Thành phần model phía đồ án: transform tự viết, công cụ đo anchor.

CỐ Ý KHÔNG import gì ở đây. Hai module con có yêu cầu trái ngược nhau:

  * `anchor.py`  — chỉ numpy, chạy được trên máy CPU không có MMDetection.
  * `transforms.py` — BẮT BUỘC có mmdet/mmcv/mmengine mới import được, vì nó
    đăng ký một transform vào registry của mmdet.

Nếu tệp này import `transforms`, thì mọi lần `import floodcount.models.anchor`
trên máy CPU sẽ kéo theo mmdet và sập. Vì vậy ai cần gì thì import đúng module
đó, ví dụ `from floodcount.models.anchor import ...`.
"""

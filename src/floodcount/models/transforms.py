# -*- coding: utf-8 -*-
"""Nối `tang_sang_nhe` (numpy thuần) vào pipeline của MMDetection.

Tệp này CHỈ import được khi đã cài mmdet/mmcv/mmengine (tức là trên Colab).
Toàn bộ phần tính toán nằm ở `floodcount/data/photometric.py`, chỗ đó chạy được
trên máy CPU không có MMDetection và có test riêng. Ở đây chỉ còn phần "keo":
đọc/ghi khoá `img` của `results` và bốc tham số ngẫu nhiên theo cách mmdet
hiểu được.

VÌ SAO PHẢI LÀ MỘT TRANSFORM CỦA MMDET MÀ KHÔNG GỌI HÀM TRỰC TIẾP
Pipeline của mmdet là một danh sách các dict trong file config, mỗi dict được
mmengine dựng thành đối tượng theo `type`. Config không gọi được hàm Python
thường, nên muốn dùng trong `configs/mmdet/cascade_convnext_t_floodnet.py` thì
phải là một lớp đã đăng ký vào registry.

VÌ SAO `custom_imports` TRONG CONFIG PHẢI GHI CẢ HAI MODULE
`custom_imports` của mmengine THAY THẾ danh sách chứ không gộp. Config gốc của
mmdet đã ghi `imports=['mmpretrain.models']` để đăng ký backbone ConvNeXt; nếu
config của đồ án chỉ ghi module của mình thì mmpretrain biến mất khỏi danh sách,
backbone không đăng ký được và model dựng lên là sập. Vì vậy trong config phải
ghi đủ::

    custom_imports = dict(
        imports=['mmpretrain.models', 'floodcount.models.transforms'],
        allow_failed_imports=False)

ĐĂNG KÝ VÀO REGISTRY NÀO
`mmdet.registry.TRANSFORMS` là registry CON của registry gốc trong mmengine.
Config của mmdet có `default_scope = 'mmdet'`, nên khi mmengine dựng pipeline nó
tra được lớp đăng ký ở registry con này. Đăng ký nhầm vào registry gốc cũng
chạy, nhưng để ở registry của mmdet mới đúng chỗ.
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmdet.registry import TRANSFORMS

from floodcount.data.photometric import chon_tham_so, tang_sang_nhe


@TRANSFORMS.register_module()
class TangSangNhe(BaseTransform):
    """Đổi độ sáng / tương phản nhẹ, GIỮ NGUYÊN MÀU.

    Thay cho `PhotoMetricDistortion` — lý do đầy đủ ở đầu
    `src/floodcount/data/photometric.py` (tóm tắt: bản của mmdet đảo kênh màu
    ngẫu nhiên với xác suất 1/2, mà ở đây màu nước chính là tín hiệu phân biệt
    nhà ngập với nhà không ngập).

    Required Keys:
        - img

    Modified Keys:
        - img

    Args:
        delta_sang (float): Biên độ độ sáng, bốc đều trong
            ``[-delta_sang, delta_sang]``. Mặc định 24 (khoảng 9% thang 0-255) —
            đủ để model không phụ thuộc vào một mức sáng cố định, nhưng không
            đến mức làm nước bùn giống mái tôn.
        tuong_phan (tuple): Khoảng của hệ số tương phản.
        prob (float): Xác suất áp dụng. Không áp dụng thì ảnh giữ nguyên, nên
            vẫn còn 50% số ảnh là ảnh gốc — model vẫn thấy được phân bố thật.
    """

    def __init__(self, delta_sang=24.0, tuong_phan=(0.85, 1.15), prob=0.5):
        if delta_sang < 0:
            raise ValueError(f"delta_sang phải >= 0, nhận {delta_sang}")
        if len(tuong_phan) != 2 or tuong_phan[0] <= 0:
            raise ValueError(
                f"tuong_phan phải là (nhỏ, lớn) với nhỏ > 0, nhận {tuong_phan}")
        if tuong_phan[0] > tuong_phan[1]:
            raise ValueError(f"tuong_phan phải tăng dần, nhận {tuong_phan}")
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob phải trong [0, 1], nhận {prob}")

        self.delta_sang = delta_sang
        self.tuong_phan = tuple(tuong_phan)
        self.prob = prob

    @cache_randomness
    def _boc_tham_so(self):
        """Bốc (có_áp_dụng, delta, alpha) — chỉ bốc MỘT LẦN cho mỗi lần gọi.

        `cache_randomness` là cơ chế của mmcv/mmengine: nó ghi lại giá trị đã
        bốc trong một lần `transform()` và dùng lại, để phần bốc ngẫu nhiên vẫn
        tái lập được theo seed của runner. Không có nó thì mỗi lần gọi lại bốc
        mới và kết quả không tái lập được dù đã cố định seed.
        """
        co_ap_dung = bool(np.random.rand() < self.prob)
        delta, alpha = chon_tham_so(self.delta_sang, self.tuong_phan)
        return co_ap_dung, delta, alpha

    def transform(self, results):
        """Áp dụng lên `results['img']`, trả về chính `results`."""
        co_ap_dung, delta, alpha = self._boc_tham_so()
        if not co_ap_dung:
            return results

        results["img"] = tang_sang_nhe(results["img"], delta, alpha)
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"delta_sang={self.delta_sang}, "
                f"tuong_phan={self.tuong_phan}, prob={self.prob})")

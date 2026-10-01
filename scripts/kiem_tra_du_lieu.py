#!/usr/bin/env python
"""CLI mỏng gọi vào src/floodcount/data/kiem_tra.py.

    python scripts/kiem_tra_du_lieu.py --config configs/data.yaml
    python scripts/kiem_tra_du_lieu.py --config configs/data.yaml --mmdet khong
"""

import pathlib
import sys

# Thêm src/ vào đường dẫn tìm module, tính từ vị trí file này -> chạy được từ
# bất kỳ thư mục nào, không phụ thuộc thư mục làm việc hiện tại.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from floodcount.data.kiem_tra import main  # noqa: E402

if __name__ == "__main__":
    main()

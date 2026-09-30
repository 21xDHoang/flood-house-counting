#!/usr/bin/env python
"""CLI mỏng gọi vào src/floodcount/data/audit.py.

Mục đích: gõ `python scripts/audit.py` từ thư mục gốc repo là chạy được, không
phải nhớ đặt biến môi trường PYTHONPATH.

    python scripts/audit.py --config configs/data.yaml
    python scripts/audit.py --config configs/data.yaml --max-images 20   # chạy thử
"""

import pathlib
import sys

# Thêm src/ vào đường dẫn tìm module, tính từ vị trí file này -> chạy được từ
# bất kỳ thư mục nào, không phụ thuộc thư mục làm việc hiện tại.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from floodcount.data.audit import main  # noqa: E402

if __name__ == "__main__":
    main()

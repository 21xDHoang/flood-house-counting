#!/usr/bin/env python
"""CLI mỏng gọi vào src/floodcount/data/overfit.py.

    python scripts/tao_overfit20.py --config configs/data.yaml
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from floodcount.data.overfit import main  # noqa: E402

if __name__ == "__main__":
    main()

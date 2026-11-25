"""Thêm vào sys.path các thư mục mà code trong tool/ import.

Script trong thư mục con của tool/ nạp tệp này bằng hai dòng:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # thư mục tool/
    import _paths  # noqa: F401,E402
Đường dẫn tính từ vị trí tệp này, nên vẫn đúng trong repo dựng cho act và trong gói release.
Tệp này cũng đổi stdout sang UTF-8 khi cần, để --help và thông báo tiếng Việt in được qua pipe trên Windows.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in ("experiments/ts02/selection", "experiments/spike", "experiments/graph", "experiments/ts02/thesis",
          "tool", "tool/ci", "tool/ciselect"):
    p = str(ROOT / d)
    if p not in sys.path:
        sys.path.insert(0, p)
# stdout chuyển hướng trên Windows dùng cp1252, không in được tiếng Việt
if (getattr(sys.stdout, "encoding", None) or "").lower() != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

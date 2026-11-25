"""Đọc unified diff thành danh sách thay đổi theo file (plan D3 T1).

Dùng chung cho patch D1-7 (D2) và diff giữa hai ref git (công cụ ciselect), để hai bên tính đặc trưng giống nhau.
"""
import re
import subprocess
from dataclasses import dataclass, field

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


@dataclass
class Change:
    path: str
    added: int = 0
    removed: int = 0
    old_lines: list = field(default_factory=list)  # vị trí trên file cũ của từng dòng đổi; dòng thêm neo vào dòng cũ ngay trước
    minus: list = field(default_factory=list)  # nội dung dòng xoá / dòng thêm (cho B1h)
    plus: list = field(default_factory=list)


def parse(text):
    out, cur, line_no = [], None, 0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            cur = Change(line.split(" b/", 1)[1])
            out.append(cur)
            line_no = None
        elif cur is None:
            continue
        elif m := HUNK.match(line):
            a, b = int(m.group(1)), m.group(2)
            line_no = a + 1 if b == "0" else a  # "-a,0": chèn sau dòng a
        elif line_no is None:
            continue  # header: index, ---, +++, Binary files ...
        elif line.startswith("-"):
            cur.removed += 1
            cur.minus.append(line[1:])
            cur.old_lines.append(line_no)
            line_no += 1
        elif line.startswith("+"):
            cur.added += 1
            cur.plus.append(line[1:])
            cur.old_lines.append(line_no - 1)
        elif line.startswith(" "):
            line_no += 1
    return out


def changes(repo, base, head):
    r = subprocess.run(["git", "-C", str(repo), "diff", "--no-renames", "--no-color", "-U0", base, head],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return parse(r.stdout)

"""Bản ghi thay đổi dùng chung cho đặc trưng D2 và công cụ ciselect (plan D3 T3, C6, #42)."""
from allowlist import PROTO
from scope import is_doc
from comments import comment_only


def change_record(changes, sc, allow, service):
    """service: do mô phỏng quy định ở D2 (#40), hoặc sc.service ở công cụ."""
    files, kinds, n_files, n_lines = allow["files"], set(), 0, 0
    for c in changes:
        if is_doc(c.path):
            continue
        gen = files.get(c.path, {}).get("generated", False)
        kinds.add("proto" if c.path == PROTO or gen else "config" if c.path == "compose.yaml" else "code")
        if gen:
            continue  # #42: không đếm stub sinh tự động
        n_files += 1
        n_lines += c.added + c.removed
    return {"service": service, "in_scope": sc.reason == "ok" and sc.service == service,  # #40, review D2-7 #3
            "kinds": tuple(sorted(kinds)),
            "n_files": n_files, "n_lines": n_lines,
            "comment_only": comment_only([c for c in changes if not is_doc(c.path)])}  # cho B1h (#32)

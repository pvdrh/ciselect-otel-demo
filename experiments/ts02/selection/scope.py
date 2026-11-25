"""Luật phạm vi của ciselect (#40, #42): diff nào được tự quyết, quy về service nào. Ngoài phạm vi → B0."""
from dataclasses import dataclass

from allowlist import PROTO


@dataclass
class Scope:
    in_scope: bool
    service: str | None
    reason: str
    n_files: int = 0
    n_lines: int = 0  # không gồm stub sinh tự động (#42)


def is_doc(path):
    return path.endswith(".md") or path.startswith(".chloggen/") or path == ".lychee.toml"


def prefix_owner(path, allow):
    """Repo khác OTel Demo: danh sách cho phép theo thư mục của service, có thể lọc theo đuôi (plan 2026-11-21, mục 3.2)."""
    if "suffixes" in allow and not path.endswith(tuple(allow["suffixes"])):
        return None
    hits = [p for p in allow.get("prefixes", {}) if path.startswith(p)]
    return allow["prefixes"][max(hits, key=len)] if hits else None


def scope(changes, allow, compose_owner, max_lines=None):
    changes = [c for c in changes if not is_doc(c.path)]
    if not changes:
        return Scope(True, None, "docs_only")
    files = allow["files"]
    svcs, stub_svcs, proto, n_files, n_lines = set(), set(), False, 0, 0
    for c in changes:
        if c.path == PROTO:
            proto = True
        elif c.path == "compose.yaml":
            owners = {compose_owner[i - 1] if 0 < i <= len(compose_owner) else (None, False) for i in c.old_lines}
            if not all(env for _, env in owners):
                return Scope(False, None, "file_not_in_scope:compose.yaml")
            svcs |= {s for s, _ in owners}
        elif c.path in files:
            svcs.add(files[c.path]["service"])
            if files[c.path]["generated"]:
                stub_svcs.add(files[c.path]["service"])
                continue
        elif owner := prefix_owner(c.path, allow):
            svcs.add(owner)
        else:
            return Scope(False, None, f"file_not_in_scope:{c.path}")
        n_files += 1
        n_lines += c.added + c.removed
    if len(svcs) > 1:
        return Scope(False, None, "multi_service")
    if proto and (not svcs or svcs != stub_svcs):
        return Scope(False, None, "proto_without_stub")
    if max_lines is not None and n_lines > max_lines:
        return Scope(False, None, "too_large")
    if not svcs:  # #133: ví dụ chỉ đổi quyền tệp compose.yaml, không dòng nào → không quy được service
        return Scope(False, None, "no_service")
    return Scope(True, svcs.pop(), "ok", n_files, n_lines)

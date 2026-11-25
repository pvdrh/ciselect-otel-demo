"""Plan D3 T2 (DoD 7): áp luật phạm vi (#40, #42) lên N commit không phải merge của upstream tính đến một tag.

Mỗi commit so với commit cha; chủ dòng compose dựng từ compose.yaml của commit cha.

Chạy: python scope_upstream.py UPSTREAM_GIT MUTANTS_DIR|ALLOWLIST.json OUT_DIR [TAG] [N] [PLAN]
      Đối số thứ hai đuôi .json: đọc danh sách cho phép từ tệp (plan 2026-11-21 Task 2.4), không dựng từ mutant.
      UPSTREAM_GIT: clone (bare, blobless được) của github.com/open-telemetry/opentelemetry-demo
"""
import csv
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

from allowlist import build
from compose_owner import owner_by_line
from diff import changes
from scope import scope

LIMITS = [2, 10, 50, None]  # #42


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def is_bot(author):
    """Tài khoản tự động: 'bot' hoặc 'renovate' trong tên (BoA có 'Mend Renovate', thêm 2026-11-22 sau đọc tay bước 3.4)."""
    a = author.lower()
    return "bot" in a or "renovate" in a


def main(repo, mutants_dir, out, tag="3.1.0", n=300, plan="plans/2026-10-02-d3-cong-cu.md T2 (DoD 7)"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    allow = json.loads(Path(mutants_dir).read_text(encoding="utf-8")) if str(mutants_dir).endswith(".json") else build(mutants_dir)
    rows = []
    for line in git(repo, "log", "--no-merges", f"-n{n}", "--format=%H%x09%an", tag).splitlines():
        h, author = line.split("\t", 1)
        parent = f"{h}^"
        compose = subprocess.run(["git", "-C", str(repo), "show", f"{parent}:compose.yaml"], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace").stdout  # rỗng nếu chưa có file
        ch = changes(repo, parent, h)
        s = scope(ch, allow, owner_by_line(compose))
        row = {"commit": h[:10], "bot": is_bot(author), "decided": s.reason == "ok", "service": s.service or "",
               "reason": s.reason.split(":")[0], "detail": s.reason.partition(":")[2], "n_lines": s.n_lines}
        for lim in LIMITS[:-1]:
            row[f"decided_max{lim}"] = scope(ch, allow, owner_by_line(compose), lim).reason == "ok"
        rows.append(row)
    with open(out / "table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def share(sub, key):
        return f"{sum(r[key] for r in sub)}/{len(sub)}"
    human = [r for r in rows if not r["bot"]]
    summary = {"n": len(rows), "bot": sum(r["bot"] for r in rows),
               "reason": dict(Counter(r["reason"] for r in rows).most_common()),
               "reason_human": dict(Counter(r["reason"] for r in human).most_common()),
               # "decided": công cụ tự quyết với một service (reason ok); docs_only đếm riêng ở "reason"
               "decided_by_max_lines": {str(lim): {"all": share(rows, k), "human": share(human, k)}
                                        for lim, k in zip(LIMITS, [f"decided_max{l}" for l in LIMITS[:-1]] + ["decided"])},
               "out_of_scope_top_dir": dict(Counter("/".join(r["detail"].split("/")[:2]) for r in rows
                                                    if r["reason"] == "file_not_in_scope").most_common(15)),
               "n_lines_decided": sorted(r["n_lines"] for r in rows if r["decided"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "params.json").write_text(json.dumps({
        "plan": plan, "decisions": "#40, #42",
        "upstream": "https://github.com/open-telemetry/opentelemetry-demo", "tag": tag,
        "tag_sha": git(repo, "rev-parse", f"{tag}^{{commit}}").strip(), "n": n, "merges": "excluded",
        "allowlist_from": str(mutants_dir), "max_lines": LIMITS, "bot_rule": "'bot' hoặc 'renovate' trong tên tác giả"},
        indent=1, ensure_ascii=False), encoding="utf-8")
    gitv = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()
    (out / "env.txt").write_text(f"Python {platform.python_version()}\n{gitv}\n{platform.platform()}\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "n_lines_decided"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4], *(sys.argv[4:5] or ["3.1.0"]), *[int(x) for x in sys.argv[5:6]], *sys.argv[6:7])

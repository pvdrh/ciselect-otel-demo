"""#51 (thăm dò, thêm sau khi đã thấy D2): cận trên oracle theo nhãn D1 và chi phí #39, trong không gian quyết định build S.

- O-test: chạy e2e khi và chỉ khi mutant có test fail; tập test = đúng các test fail.
- O-mut: như O-test nhưng chỉ chạy một test fail rẻ nhất (collector tính 0 theo #39).
- U(r): tiết kiệm lớn nhất với recall theo mutant ≥ r = bắt k = ⌈r·F⌉ mutant fail có chi phí bắt nhỏ nhất, bỏ e2e cho phần còn lại.

Chạy: python oracle.py RUN_DIR BUILD_NOCACHE_DIR DEMO OUT_DIR SUMMARY_DIR [SUMMARY_DIR ...]
"""
import csv
import json
import math
import os
import platform
import sys
from pathlib import Path

from data import COLLECTOR, load
from evaluate import Decision, metrics
from run import _by_id, _write

PLAN = "plans/2026-10-07-d2-rf-oracle.md"


def _fails(d, ids):
    ids, out = set(ids), {}
    for r in d.labels:
        if r["label"] and r["mutant_id"] in ids:
            out.setdefault(r["mutant_id"], set()).add(r["test_id"])
    return out


def _dur(d, t):
    return 0.0 if t == COLLECTOR else d.dur[t]


def _cheapest(d, tests):
    return min(tests, key=lambda t: (_dur(d, t), t))


def o_test(d, ids):
    ms, f = _by_id(d), _fails(d, ids)
    return {i: Decision({ms[i]["service"]}, i in f, set(f.get(i, ()))) for i in ids}


def o_mut(d, ids):
    ms, f = _by_id(d), _fails(d, ids)
    return {i: Decision({ms[i]["service"]}, i in f, {_cheapest(d, f[i])} if i in f else set()) for i in ids}


def upper(d, ids, r):
    dec, f = o_mut(d, ids), _fails(d, ids)
    k = max(0, math.ceil(r * len(f) - 1e-9))  # cùng cách làm tròn với pick_threshold
    for i in sorted(f, key=lambda i: (_dur(d, _cheapest(d, f[i])), i))[k:]:
        dec[i] = Decision(dec[i].build, False, set())
    return metrics(dec, d)["savings"]


def gap(d, ids, rows, split):
    out = []
    for r in rows:
        rec = float(r["recall_mutant_mean"])
        if math.isnan(rec):
            continue
        u = upper(d, ids, rec)
        out.append({"split": split, "method": r["method"], "op": r["op"], "savings": float(r["savings_mean"]),
                    "recall_mutant": rec, "upper": u, "gap": u - float(r["savings_mean"])})
    return out


def main(run_dir, build_dir, demo, out, *summaries):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(run_dir, build_dir, demo)
    ids = [m["mutant_id"] for m in d.mutants]
    subsets = {"all": ids, "T": [m["mutant_id"] for m in d.mutants if m["record"]["in_scope"]]}  # #40
    _write(out / "oracle.csv", [{"oracle": name, "subset": s, **metrics(fn(d, sub), d)}
                                for s, sub in subsets.items() for name, fn in (("O-test", o_test), ("O-mut", o_mut))])
    _write(out / "upper_curve.csv", [{"r": r / 100, "upper": upper(d, ids, r / 100)} for r in range(101)])
    rows = []
    for s in summaries:
        for split, name in (("main", "table.csv"), ("loso", "loso.csv")):
            with open(Path(s) / name, encoding="utf-8") as f:
                rows += [dict(g, source=str(s)) for g in gap(d, ids, list(csv.DictReader(f)), split)]
    _write(out / "gap.csv", rows)
    (out / "params.json").write_text(json.dumps({
        "plan": PLAN, "decision": "#51 (thăm dò, thêm sau khi đã thấy kết quả D2)", "data": str(run_dir),
        "build_nocache": str(build_dir), "summaries": [str(s) for s in summaries], "n_mutants": len(ids),
        "n_T": len(subsets["T"]), "n_fail": len(_fails(d, ids)), "W": d.W, "t_start": d.t_start,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\n{platform.platform()}\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])

"""#51 (thăm dò, thêm sau khi đã thấy D2): chạy RF trong đúng khung D2 (run.run_seed) trên một seed; #57 dùng lại cho B3k.

Dòng b0/b1/b1h/b2 chạy lại để kiểm trùng với run D2 cùng seed và kiểu chia (DoD 2); lệch thì thoát mã khác 0 và ghi
mismatches.txt, run giữ nguyên.
Chạy (PYTHONHASHSEED=0): python run_extra.py RUN_DIR BUILD_NOCACHE_DIR DEMO OUT_DIR SEED D2_RUN_DIR [loso|main] [rf|b3k]
"""
import csv
import json
import math
import os
import platform
import sys
from pathlib import Path

from data import load
from run import GRIDS, TUNE_TARGET, _write, run_seed

LABELS = {"rf": ("plans/2026-10-07-d2-rf-oracle.md", "#51 (thăm dò, thêm sau khi đã thấy kết quả D2)"),
          "b3k": ("plans/2026-10-08-phan-tich-o4-o5.md", "#57 (O5, mô tả hậu nghiệm, thêm sau khi đã thấy kết quả D2)")}
BASE = ("b0", "b1", "b1h", "b2")


def _same(a, b):
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)


def same_rows(old, new, keys):
    """Dòng CSV (chuỗi) → danh sách chỗ lệch; bỏ qua cột time_*. R1: số thực lệch ở mức làm tròn hash seed thì coi là khớp."""
    o, n = ({tuple(r[k] for k in keys): r for r in rows} for rows in (old, new))
    out = [f"thiếu {k}" for k in o.keys() - n.keys()] + [f"thừa {k}" for k in n.keys() - o.keys()]
    for k in o.keys() & n.keys():
        out += [f"{k} {c}: {o[k][c]} != {n[k].get(c)}" for c in o[k]
                if not c.startswith("time_") and not _same(o[k][c], n[k].get(c, ""))]
    return out


def _base(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["method"] in BASE]


def main(run_dir, build_dir, demo, out, seed, d2, mode="", method="rf"):
    import sklearn
    out, seed, loso, d2 = Path(out), int(seed), mode == "loso", Path(d2)
    out.mkdir(parents=True, exist_ok=True)
    d = load(run_dir, build_dir, demo)
    rows, per = run_seed(d, seed, loso=loso, methods=(method,))
    _write(out / "table.csv", rows)
    _write(out / "per_mutant.csv", per)
    bad = same_rows(_base(d2 / "table.csv"), _base(out / "table.csv"), ("fold", "method", "op")) + \
        same_rows(_base(d2 / "per_mutant.csv"), _base(out / "per_mutant.csv"), ("fold", "method", "op", "mutant_id"))
    (out / "params.json").write_text(json.dumps({
        "plan": LABELS[method][0], "decision": LABELS[method][1], "data": str(run_dir),
        "build_nocache": str(build_dir), "seed": seed, "split": "loso" if loso else "C4 K=5", "methods": [method],
        "grids": {k: [str(x) for x in GRIDS[k]] for k in ("b2", method)}, "tune_target": TUNE_TARGET,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "d2_ref": str(d2), "baseline_mismatches": len(bad),
        "W": d.W, "t_start": d.t_start, "n_mutants": len(d.mutants), "n_tests": len(d.tests)},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\nscikit-learn {sklearn.__version__}\n"
                                 f"{platform.platform()}\n", encoding="utf-8")
    if bad:
        (out / "mismatches.txt").write_text("\n".join(bad), encoding="utf-8")
        sys.exit(f"kiểm trùng baseline lệch {len(bad)} chỗ: xem {out / 'mismatches.txt'}")


if __name__ == "__main__":
    main(*sys.argv[1:9])

"""G2a (#88): chạy lại D2 với phép chia theo nhóm gần trùng (plans/2026-10-15-g2a-chia-theo-nhom-gan-trung.md).

Gắn khoá nhóm vào từng mutant (m["group"]); data.folds và features.features đọc khoá đó, nên fold ngoài, fold trong,
nhóm băm của fail_rate và lần fit lại đều chia theo nhóm. Mọi thứ khác như run.py.

Chạy: python run_nhom.py RUN_DIR BUILD_NOCACHE_DIR DEMO OUT_DIR SEED {nhom|nhom_gia|hoiquy}
  hoiquy: fold 0 của SEED, không gắn nhóm; so từng dòng per_mutant với run D2 (DoD 4).
"""
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import run
from cell5 import near_dup_key
from data import folds, load

ROOT = Path(__file__).resolve().parents[3]
PLAN = "plans/2026-10-15-g2a-chia-theo-nhom-gan-trung.md"
D2 = ROOT / "results" / "ts02_eval"


def near_dup_groups(run_dir):
    """Khoá gần trùng của #61 dạng chuỗi 'loại|service|site', đọc site_id từ mutants.csv như cell5.py."""
    with open(Path(run_dir) / "mutants" / "mutants.csv", encoding="utf-8") as fh:
        return {r["mutant_id"]: "|".join(near_dup_key(r)) for r in csv.DictReader(fh)}


def fake_groups(mutants, real, seed):
    """Nhóm giả: trong từng loại (thứ tự sắp xếp), xáo mutant bằng một random.Random(seed) dùng chung rồi cắt theo cỡ
    các nhóm thật của loại đó, từ lớn đến nhỏ."""
    rng, out = random.Random(seed), {}
    for t in sorted({m["type"] for m in mutants}):
        ids = sorted(m["mutant_id"] for m in mutants if m["type"] == t)
        sizes = sorted(Counter(real[i] for i in ids).values(), reverse=True)
        rng.shuffle(ids)
        pos = 0
        for j, n in enumerate(sizes):
            out.update({i: f"{t}|gia|{j}" for i in ids[pos:pos + n]})
            pos += n
    return out


def balance_ok(mutants, splits):
    """Tiêu chí cân bằng chốt trước: cỡ fold test trong [0,8; 1,2]·n/k; ở mỗi tầng (loại, có fail), số mutant của fold
    lệch khỏi 1/k tổng của tầng không quá số mutant lớn nhất mà một nhóm có trong tầng. Trả (đạt, lệch lớn nhất/giới hạn)."""
    k, n = len(splits), len(mutants)
    st = {m["mutant_id"]: f"{m['type']}|{m['has_fail']}" for m in mutants}
    tot, cap = Counter(st.values()), defaultdict(int)
    for (s, _), c in Counter((st[m["mutant_id"]], m["group"]) for m in mutants).items():
        cap[s] = max(cap[s], c)
    ok, worst = True, 0.0
    for _, te in splits:
        ok &= 0.8 * n / k <= len(te) <= 1.2 * n / k
        c = Counter(st[i] for i in te)
        for s in tot:
            dev = abs(c[s] - tot[s] / k)
            ok &= dev <= cap[s]
            worst = max(worst, dev / cap[s])
    return ok, worst


def _git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, encoding="utf-8", check=True).stdout


def provenance(out):
    """A4: HEAD, cờ cây bẩn, SHA-256 của git diff HEAD (cả repo và riêng experiments/), file chưa theo dõi đã dùng.
    Ghi kèm code.diff (diff của experiments/) để tái hiện được khi chưa commit."""
    diff_all, diff_exp = _git("diff", "HEAD"), _git("diff", "HEAD", "--", "experiments")
    (out / "code.diff").write_text(diff_exp, encoding="utf-8")
    here = Path(__file__).resolve().parent
    untracked = [f for f in _git("ls-files", "--others", "--exclude-standard", "--", str(here)).split()
                 if f.endswith(".py") and not Path(f).name.startswith("test_")]
    return {"git_head": _git("rev-parse", "HEAD").strip(), "dirty": bool(_git("status", "--porcelain").strip()),
            "diff_head_sha256": hashlib.sha256(diff_all.encode()).hexdigest(),
            "diff_head_experiments_sha256": hashlib.sha256(diff_exp.encode()).hexdigest(),
            "untracked_py_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in untracked}}


def same_row(got, want):
    """Trường không phải số thực khớp tuyệt đối; cost, cost_b0prime khớp với rel_tol 1e-9 như phép kiểm tái lập #51,
    vì e2e_cost cộng trên một set nên thứ tự cộng theo PYTHONHASHSEED, mà run D2 không ghi giá trị này."""
    fl = ("cost", "cost_b0prime")
    return got.keys() == want.keys() and all(got[k] == want[k] for k in got if k not in fl) and all(
        math.isclose(float(got[k]), float(want[k]), rel_tol=1e-9, abs_tol=1e-12) for k in fl)


def _regress(d, seed, out):
    """DoD 4: code đã sửa, không gắn nhóm, fold 0 → per_mutant khớp từng dòng với run D2 cùng seed."""
    orig = run.folds
    run.folds = lambda ms, s, k=5: orig(ms, s, k)[:1] if k == 5 else orig(ms, s, k)
    try:
        _, per = run.run_seed(d, seed)
    finally:
        run.folds = orig
    out.mkdir(parents=True, exist_ok=True)
    run._write(out / "per_mutant.csv", per)
    src = next(p for p in sorted(D2.glob(f"*_seed{seed}")) if p.is_dir())
    want = [r for r in csv.DictReader(open(src / "per_mutant.csv", encoding="utf-8")) if r["fold"] == "0"]
    got = [r for r in csv.DictReader(open(out / "per_mutant.csv", encoding="utf-8"))]
    bad = [i for i, (g, w) in enumerate(zip(got, want)) if not same_row(g, w)]
    exact = sum(g == w for g, w in zip(got, want))
    (out / "params.json").write_text(json.dumps({
        "plan": PLAN, "decision": "#88", "check": "DoD 4: fold 0, không gắn nhóm, so với D2", "d2": str(src),
        "n_rows": len(got), "n_rows_d2": len(want), "mismatches": len(bad), "byte_identical_rows": exact,
        "rule": same_row.__doc__, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), **provenance(out)},
        indent=1, ensure_ascii=False), encoding="utf-8")
    assert len(got) == len(want) and not bad, (len(got), len(want), bad[:3])
    print(f"hồi quy: fold 0 seed {seed}: {len(got)} dòng khớp {src.name}; {exact} dòng trùng từng byte")


def main(run_dir, build_dir, demo, out, seed, kind):
    import lightgbm
    import sklearn
    seed, out = int(seed), Path(out)
    d = load(run_dir, build_dir, demo)
    if kind == "hoiquy":
        return _regress(d, seed, out)
    real = near_dup_groups(run_dir)
    grp = {"nhom": real, "nhom_gia": fake_groups(d.mutants, real, seed)}[kind]
    for m in d.mutants:
        m["group"] = grp[m["mutant_id"]]
    splits = folds(d.mutants, seed)
    ok, worst = balance_ok(d.mutants, splits)
    assert ok, f"tiêu chí cân bằng không đạt ở seed {seed}: {worst:.2f}"  # dừng và hỏi lại, không đổi seed
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance(out)
    rows, per = run.run_seed(d, seed)
    run._write(out / "table.csv", rows)
    run._write(out / "per_mutant.csv", per)
    run._write(out / "folds.csv", [{"seed": seed, "fold": f, "mutant_id": i, "group": grp[i]}
                                   for f, (_, te) in enumerate(splits) for i in te])
    (out / "params.json").write_text(json.dumps({
        "plan": PLAN, "decision": "#88", "prereg": "docs/preregistration.md C4, Bổ sung sau khoá (#88)",
        "data": str(run_dir), "build_nocache": str(build_dir), "demo": str(demo), "seed": seed, "split": kind,
        "groups": "cell5.near_dup_key, 'loại|service|site'" if kind == "nhom"
        else "fake_groups: cùng phân bố cỡ nhóm trong từng loại, random.Random(seed)",
        "n_groups": len(set(grp[m["mutant_id"]] for m in d.mutants)), "balance_worst_over_cap": worst,
        "grids": {k: [str(x) for x in v] for k, v in run.GRIDS.items()}, "tune_target": run.TUNE_TARGET,
        "W": d.W, "t_start": d.t_start, "n_mutants": len(d.mutants), "n_tests": len(d.tests), **prov},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\nlightgbm {lightgbm.__version__}\n"
                                 f"scikit-learn {sklearn.__version__}\n{platform.platform()}\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:7])

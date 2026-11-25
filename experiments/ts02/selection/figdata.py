"""D2-9 số liệu cho hình (#49, chỉ mô tả): H1 đường tiết kiệm theo recall, H4 gain của LightGBM.

- H1: cấu hình mỗi fold lấy từ table.csv (không tune lại); τ chung quét trên điểm fold test, gộp các fold của seed.
- H4: fit lại LGBM với cấu hình đã chọn trên fold train, gain chuẩn hoá tổng 1.

Chạy: python figdata.py RUN_DIR BUILD_NOCACHE_DIR DEMO RESULTS_DIR OUT_DIR
"""
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

from data import COLLECTOR, folds, load
from features import FEATURES, features
from models import _model
from run import _by_id, _scores, _write

SCORED = ("b3", "lr", "lgbm", "lgbm_nofr")


def curve(d, scores, ids, taus):
    """[(τ, tiết kiệm, recall theo mutant)] trên tập mutant ids với một ngưỡng chung."""
    ms, lab = _by_id(d), {(r["mutant_id"], r["test_id"]) for r in d.labels if r["label"]}
    S = np.array([[scores[i, t] for t in d.tests] for i in ids])
    L = np.array([[(i, t) in lab for t in d.tests] for i in ids])
    dur = np.array([0.0 if t == COLLECTOR else d.dur[t] for t in d.tests])
    build = np.array([d.build_nocache[ms[i]["service"]] for i in ids])
    full = d.t_start + d.W + dur.sum()
    base, fail = (build + full).sum(), L.any(axis=1)
    out = []
    for tau in taus:
        ch = S >= tau
        e2e = ch.any(axis=1)
        c = build + e2e * (d.t_start + d.W + (ch * dur).sum(axis=1))
        out.append((tau, 1 - c.sum() / base, (ch & L).any(axis=1)[fail].mean()))
    return out


def importance(d, kind, cfg, train, seed):
    drop = ("fail_rate",) if kind == "lgbm_nofr" else ()
    X, keys = features(d, train, train, drop)
    lab = {(r["mutant_id"], r["test_id"]): r["label"] for r in d.labels}
    g = _model(kind, cfg, seed).fit(X, np.array([lab[k] for k in keys])).booster_.feature_importance("gain")
    return dict(zip([f for f in FEATURES if f not in drop], g / g.sum()))


def main(run_dir, build_dir, demo, results, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(run_dir, build_dir, demo)
    curves, imps, src = [], [], sorted(glob.glob(str(Path(results) / "ts02_eval/*_seed[0-4]")))
    for e in src:
        seed = json.loads((Path(e) / "params.json").read_text(encoding="utf-8"))["seed"]
        cfg = {(int(r["fold"]), r["method"]): json.loads(r["config"])
               for r in csv.DictReader(open(Path(e) / "table.csv", encoding="utf-8")) if r["method"] in SCORED}
        splits = folds(d.mutants, seed, 5)
        for m in SCORED:
            sc, ids = {}, []
            for f, (train, test) in enumerate(splits):
                sc |= _scores(m, cfg[f, m], d, train, test, seed)
                ids += test
                if m in ("lgbm", "lgbm_nofr"):
                    imps += [{"seed": seed, "fold": f, "method": m, "feature": k, "gain": v}
                             for k, v in importance(d, m, cfg[f, m], train, seed).items()]
            taus = [-math.inf] + sorted(set(np.quantile(list(sc.values()), np.linspace(0, 1, 401)))) + [math.inf]
            curves += [{"seed": seed, "method": m, "tau": t, "savings": s, "recall_mutant": r}
                       for t, s, r in curve(d, sc, ids, taus)]
    _write(out / "h1_curves.csv", curves)
    _write(out / "h4_gain.csv", imps)
    (out / "params.json").write_text(json.dumps({"decision": "#49 Q3, H4", "eval": src, "data": str(run_dir),
                                                 "tau_grid": "401 quantile điểm fold test + ±inf"},
                                                indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:6])

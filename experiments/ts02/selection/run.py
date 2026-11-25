"""D2-6/D2-8: chạy B0–B4 trên một seed theo tiền đăng ký C4–C6 (#39, #40, #41).

- Mỗi fold ngoài (C4, K = 5): tune trên fold train bằng CV trong (4 fold theo mutant), báo số trên fold test.
- Điểm vận hành của B3/B4: t95, t99, và khớp recall của B1/B1h/B2 trên validation (C5 "so sánh chính ở cùng recall").
- Ghi: table.csv (một dòng mỗi phương pháp × điểm vận hành × fold) và per_mutant.csv (chi phí từng mutant, cho bootstrap).

Chạy: python run.py RUN_DIR BUILD_NOCACHE_DIR DEMO OUT_DIR SEED [loso]
"""
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

from baselines import b0, b1, b1h, b2, b3_scores, b3_table, b3k_scores
from data import folds, load, loso_folds
from evaluate import Decision, b0prime, cost, metrics
from sklearn.metrics import average_precision_score, roc_auc_score

from models import fit_predict
from threshold import pick_threshold, select

METHODS = ["b0", "b1", "b1h", "b2", "b3", "lr", "lgbm", "lgbm_nofr"]  # lgbm_nofr: phân tích phụ #47 Q1
GRIDS = {  # C3, C6
    "b2": [(k, h) for k in (1, 2, 3, None) for h in ("callers", "undirected")],
    "b3": [0.5, 1.0, 2.0],
    "lgbm": [{"num_leaves": nl, "learning_rate": lr, "n_estimators": ne, "min_child_samples": mc, "class_weight": cw}
             for nl in (15, 31) for lr in (0.05, 0.1) for ne in (100, 300) for mc in (10, 20)
             for cw in (None, "balanced")],  # #47 Q2
    "lr": [{"C": c, "class_weight": cw} for c in (1e-3, 1e-2, 1e-1, 1, 10, 100) for cw in (None, "balanced")],
}
GRIDS["lgbm_nofr"] = GRIDS["lgbm"]
GRIDS["rf"] = [  # #51 thăm dò, chốt trước khi chạy RF; 32 cấu hình như lgbm
    {"n_estimators": ne, "max_depth": md, "min_samples_leaf": ml, "max_features": mf, "class_weight": cw}
    for ne in (100, 300) for md in (None, 10) for ml in (1, 5) for mf in ("sqrt", 0.5) for cw in (None, "balanced")]
GRIDS["b3k"] = GRIDS["b3"]  # #57 O5, mô tả hậu nghiệm; ngoài METHODS và họ Holm
TUNE_TARGET = 0.95


def _by_id(d):
    return {m["mutant_id"]: m for m in d.mutants}


def _scores(method, cfg, d, train, pred, seed):
    if method == "b3":
        tab = b3_table(train, d.labels, {m["mutant_id"]: m["service"] for m in d.mutants}, cfg)
        ms = _by_id(d)
        return {(mid, t): v for mid in pred for t, v in b3_scores(tab, ms[mid]["service"], d.tests, cfg).items()}
    if method == "b3k":
        ms = _by_id(d)
        key = {i: (m["service"], tuple(m["record"]["kinds"])) for i, m in ms.items()}
        ts = b3_table(train, d.labels, {i: m["service"] for i, m in ms.items()}, cfg)
        tk = b3_table(train, d.labels, key, cfg)
        return {(mid, t): v for mid in pred
                for t, v in b3k_scores(ts, tk, ms[mid]["service"], key[mid], d.tests, cfg).items()}
    return fit_predict(method, cfg, d, train, pred, seed)


def _decide(d, scores, ids, tau):
    ms, out = _by_id(d), {}
    for mid in ids:
        tests, e2e = select({t: scores[mid, t] for t in d.tests}, tau)
        out[mid] = Decision({ms[mid]["service"]}, e2e, tests)
    return out


def _best(d, scores, ids):
    """mutant fail → điểm cao nhất trên test mang nhãn fail (đầu vào pick_threshold)."""
    ids, best = set(ids), {}
    for r in d.labels:
        if r["label"] and r["mutant_id"] in ids:
            best[r["mutant_id"]] = max(best.get(r["mutant_id"], -math.inf), scores[r["mutant_id"], r["test_id"]])
    return best


def _better(cand, cur, target):
    """Ruling D2-6: chọn cấu hình đạt recall mục tiêu có tiết kiệm cao nhất; không cấu hình nào đạt thì recall cao nhất.
    Hoà giữ cấu hình đến trước trong grid."""
    if cur is None:
        return True
    ok_c, ok_b = cand["recall_mutant"] >= target, cur["recall_mutant"] >= target
    if ok_c != ok_b:
        return ok_c
    key = "savings" if ok_c else "recall_mutant"
    return cand[key] > cur[key]


def _oof(method, cfg, d, train, seed):
    ms, sc = _by_id(d), {}
    for itr, iva in folds([ms[i] for i in train], seed, k=4):
        sc |= _scores(method, cfg, d, itr, iva, seed)
    return sc


def _tune_scored(method, d, train, seed, grid):
    best = None
    for cfg in grid:
        sc = _oof(method, cfg, d, train, seed)
        m = metrics(_decide(d, sc, train, pick_threshold(_best(d, sc, train), TUNE_TARGET)), d)
        if _better(m, best and best[1], TUNE_TARGET):
            best = (cfg, m, sc)
    return best[0], best[2]


def _tune_b2(d, train, grid):
    ms, best = _by_id(d), None
    for k, h in grid:
        m = metrics({i: b2(ms[i], d, d.edges, k, h) for i in train}, d)
        if _better(m, best and best[1], TUNE_TARGET):
            best = ((k, h), m)
    return best[0]


def diag(scores, pos):
    """#47 Q3: PR-AUC (average precision) và ROC-AUC mức dòng; chỉ chẩn đoán, không dùng để tune hay chọn ngưỡng."""
    keys = list(scores)
    y = [int(k in pos) for k in keys]
    if len(set(y)) < 2:
        return math.nan, math.nan
    s = [scores[k] for k in keys]
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def _record(d, method, op, fold, seed, cfg, tau, dec, t_fit, t_pred, rows, per, dg=(None, None)):
    lab = {(r["mutant_id"], r["test_id"]) for r in d.labels if r["label"]}
    ms = _by_id(d)
    rows.append({"seed": seed, "fold": fold, "method": method, "op": op, "config": json.dumps(cfg), "tau": tau,
                 **metrics(dec, d), "time_fit_s": round(t_fit, 3), "time_pred_s": round(t_pred, 3),
                 "pr_auc": dg[0], "roc_auc": dg[1]})
    for mid, x in dec.items():
        m = ms[mid]
        per.append({"seed": seed, "fold": fold, "method": method, "op": op, "mutant_id": mid, "type": m["type"],
                    "service": m["service"], "in_scope": m["record"]["in_scope"], "has_fail": m["has_fail"],
                    "caught": int(x.e2e and any((mid, t) in lab for t in x.tests)), "e2e": int(x.e2e),
                    "cost": cost(x, d), "cost_b0prime": cost(b0prime(m, d), d)})


def run_seed(d, seed, grids=GRIDS, k=5, loso=False, methods=("b3", "lr", "lgbm", "lgbm_nofr")):
    ms, rows, per = _by_id(d), [], []
    splits = [(tr, te) for _, tr, te in loso_folds(d.mutants)] if loso else folds(d.mutants, seed, k)
    for fold, (train, test) in enumerate(splits):
        rec = {}
        for name, fn in [("b0", b0), ("b1", b1), ("b1h", b1h)]:
            t0 = time.perf_counter()
            dec = {i: fn(ms[i], d) for i in test}
            _record(d, name, "natural", fold, seed, None, None, dec, 0.0, time.perf_counter() - t0, rows, per)
            rec[name] = metrics({i: fn(ms[i], d) for i in train}, d)["recall_mutant"]
        t0 = time.perf_counter()
        kh = _tune_b2(d, train, grids["b2"])
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        dec = {i: b2(ms[i], d, d.edges, *kh) for i in test}
        _record(d, "b2", "natural", fold, seed, kh, None, dec, t_fit, time.perf_counter() - t0, rows, per)
        rec["b2"] = metrics({i: b2(ms[i], d, d.edges, *kh) for i in train}, d)["recall_mutant"]
        ops = {"t95": 0.95, "t99": 0.99, **{f"match_{b}": rec[b] for b in ("b1", "b1h", "b2")}}
        pos = {(r["mutant_id"], r["test_id"]) for r in d.labels if r["label"]}
        for method in methods:
            t0 = time.perf_counter()
            cfg, oof = _tune_scored(method, d, train, seed, grids[method])
            best = _best(d, oof, train)
            t_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            sc = _scores(method, cfg, d, train, test, seed)
            t_pred = time.perf_counter() - t0
            dg = diag(sc, pos)
            for op, target in ops.items():
                tau = pick_threshold(best, 1.0 if math.isnan(target) else target)
                _record(d, method, op, fold, seed, cfg, tau, _decide(d, sc, test, tau), t_fit, t_pred, rows, per, dg)
    return rows, per


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(run_dir, build_dir, demo, out, seed, mode=""):
    import lightgbm
    import sklearn
    out, seed, loso = Path(out), int(seed), mode == "loso"
    out.mkdir(parents=True, exist_ok=True)
    d = load(run_dir, build_dir, demo)
    rows, per = run_seed(d, seed, loso=loso)
    _write(out / "table.csv", rows)
    _write(out / "per_mutant.csv", per)
    (out / "params.json").write_text(json.dumps({
        "plan": "plans/2026-09-30-d2-mo-hinh-va-danh-gia.md", "prereg": "docs/preregistration.md (#34, sửa #39 #42)",
        "data": str(run_dir), "build_nocache": str(build_dir), "seed": seed, "split": "loso" if loso else "C4 K=5",
        "grids": {k: [str(x) for x in v] for k, v in GRIDS.items()}, "tune_target": TUNE_TARGET,
        "W": d.W, "t_start": d.t_start, "n_mutants": len(d.mutants), "n_tests": len(d.tests)},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\nlightgbm {lightgbm.__version__}\n"
                                 f"scikit-learn {sklearn.__version__}\n{platform.platform()}\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:7])

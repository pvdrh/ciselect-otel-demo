"""S11 phần P (#88): độ rải theo fold của recall và τ ở D2 (plans/2026-10-15-s11-dat-nguong.md, phần P).

- Mốc nhiễu nhị thức: recall thật đúng bằng mục tiêu; mỗi mutant có fail của fold được bắt độc lập với xác suất bằng mục
  tiêu; giữ số mutant có fail của từng fold D2; 20 000 lần, numpy.random.default_rng(0). Mốc chỉ phụ thuộc điểm vận hành.
- Mục tiêu của match_*: recall của B1, B1h, B2 trên fold train ngoài (run.py dòng 149–157), tính lại tất định.
- τ theo class_weight và theo cấu hình đã chọn (cột config của table.csv).
Hậu nghiệm, mô tả; không fit lại, không p, không Holm.

Chạy: python nguong.py RUN_DIR BUILD_NOCACHE_DIR DEMO RESULTS_DIR OUT_DIR
"""
import json
import platform
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

import run
from baselines import b1, b1h, b2
from data import folds, load
from evaluate import metrics
from run_nhom import provenance
from summarize import _load

PLAN = "plans/2026-10-15-s11-dat-nguong.md"
METHODS = ("b3", "lr", "lgbm", "lgbm_nofr")
OPS = ("t95", "t99", "match_b1", "match_b2")  # match_b1h trùng match_b1 ở mọi dòng (assert trong main)
N_SIM = 20_000
Band = namedtuple("Band", "mad_lo mad_hi min_lo min_hi below_lo below_hi mad_mean")


def noise_band(n_fail, targets, n=N_SIM, seed=0):
    """Khoảng 5–95 % của MAD so với mục tiêu, recall nhỏ nhất và tỉ lệ fold dưới mục tiêu, khi recall thật = mục tiêu."""
    nf, t = np.array(n_fail), np.array(targets, dtype=float)
    sims = np.random.default_rng(seed).binomial(nf[None, :], t[None, :], size=(n, len(nf))) / nf[None, :]
    mad, mn, below = np.abs(sims - t).mean(1), sims.min(1), (sims < t).mean(1)
    q = lambda x: (float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95)))  # noqa: E731
    return Band(*q(mad), *q(mn), *q(below), float(mad.mean()))


def observed(per, targets):
    """per: per_mutant của một (phương pháp, điểm vận hành); targets: (seed, fold) → mục tiêu. Fold theo thứ tự (seed, fold)."""
    g = per[per.has_fail == 1].groupby(["seed", "fold"]).caught.agg(["mean", "size"])
    t = np.array([targets[k] for k in g.index], dtype=float)
    r = g["mean"].to_numpy()
    return {"n_fail": g["size"].tolist(), "targets": t.tolist(), "recall_min": float(r.min()), "recall_max": float(r.max()),
            "mad": float(np.abs(r - t).mean()), "below": int((r < t).sum()), "n_folds": len(g)}


def _cw(config):
    v = json.loads(config)
    return str(v.get("class_weight")) if isinstance(v, dict) else "—"


def tau_by_config(tab):
    t = tab.assign(class_weight=tab.config.map(_cw))
    out = []
    for by in ("class_weight", "config"):
        g = t.groupby(["method", "op", by]).agg(
            n=("tau", "size"), tau_min=("tau", "min"), tau_max=("tau", "max"), tau_median=("tau", "median"),
            recall_min=("recall_mutant", "min"), recall_max=("recall_mutant", "max"), recall_mean=("recall_mutant", "mean"))
        out.append(g.reset_index().rename(columns={by: "key"}).assign(by=by))
    return pd.concat(out, ignore_index=True)


def match_targets(d, seed):
    """(seed, fold) → (recall trên fold train ngoài của b1, b1h, b2; cấu hình B2 đã tune), như run.run_seed."""
    ms, out = {m["mutant_id"]: m for m in d.mutants}, {}
    for f, (train, _) in enumerate(folds(d.mutants, seed)):
        rec = {name: metrics({i: fn(ms[i], d) for i in train}, d)["recall_mutant"] for name, fn in (("b1", b1), ("b1h", b1h))}
        kh = run._tune_b2(d, train, run.GRIDS["b2"])
        rec["b2"] = metrics({i: b2(ms[i], d, d.edges, *kh) for i in train}, d)["recall_mutant"]
        out[seed, f] = (rec, kh)
    return out


def main(run_dir, build_dir, demo, results, out):
    res, out = Path(results), Path(out)
    per, src = _load(str(res / "ts02_eval/*_seed[0-4]"), "per_mutant.csv")
    tab, _ = _load(str(res / "ts02_eval/*_seed[0-4]"), "table.csv")
    assert len(src) == 5 and sorted(per.seed.unique()) == list(range(5))
    d = load(run_dir, build_dir, demo)
    tg = {}
    for seed in range(5):
        for (s, f), (rec, kh) in match_targets(d, seed).items():
            cfg = tab[(tab.seed == s) & (tab.fold == f) & (tab.method == "b2")].config
            assert len(cfg) == 1 and json.loads(cfg.iloc[0]) == list(kh), (s, f)  # tái hiện fold và B2 của D2
            tg[s, f] = {"t95": 0.95, "t99": 0.99, **{f"match_{b}": rec[b] for b in ("b1", "b1h", "b2")}}
    a, b = (tab[tab.op == o].set_index(["seed", "fold", "method"])[["tau", "recall_mutant", "savings"]].sort_index()
            for o in ("match_b1", "match_b1h"))
    assert a.equals(b) and all(v["match_b1"] == v["match_b1h"] for v in tg.values())

    rows = []
    for op in OPS:
        targets = {k: v[op] for k, v in tg.items()}
        obs = {m: observed(per[(per.method == m) & (per.op == op)], targets) for m in METHODS}
        nf = obs[METHODS[0]]["n_fail"]
        assert all(o["n_fail"] == nf and o["n_folds"] == 25 for o in obs.values())  # cùng fold test cho mọi phương pháp
        band = noise_band(nf, obs[METHODS[0]]["targets"])
        for m, o in obs.items():
            rows.append({"method": m, "op": op, "n_folds": o["n_folds"], "n_fail_min": min(nf), "n_fail_max": max(nf),
                         "target_min": min(o["targets"]), "target_max": max(o["targets"]),
                         "recall_min": o["recall_min"], "recall_max": o["recall_max"], "mad": o["mad"],
                         "below": o["below"], "frac_below": o["below"] / o["n_folds"], **band._asdict(),
                         "mad_vs_band": "vượt" if o["mad"] > band.mad_hi else "dưới" if o["mad"] < band.mad_lo else "trong"})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "noise.csv", index=False)
    tau_by_config(tab[tab.method.isin(METHODS)]).to_csv(out / "tau_cfg.csv", index=False)
    pd.DataFrame([{"seed": s, "fold": f, **v} for (s, f), v in sorted(tg.items())]).to_csv(out / "targets.csv", index=False)
    (out / "params.json").write_text(json.dumps({
        "plan": PLAN + " (phần P)", "decision": "#88", "prereg": "docs/preregistration.md C5, Bổ sung sau khoá (#88)",
        "eval": src, "n_sim": N_SIM, "rng": "numpy.random.default_rng(0), một lần cho mỗi điểm vận hành",
        "quantiles": [0.05, 0.95], "methods": METHODS, "ops": OPS,
        "match_b1h": "trùng match_b1 ở mọi dòng của table.csv và mọi mục tiêu (assert); không ghi riêng",
        "targets": "t95, t99 cố định; match_*: recall trên fold train ngoài, tính lại; B2 đã tune khớp cột config của D2",
        **provenance(out)}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\npandas {pd.__version__}\nnumpy {np.__version__}\n"
                                 f"{platform.platform()}\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:6])

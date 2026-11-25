"""D2-9: tổng hợp 5 seed D2-8 (K = 5) và LOSO thành summary/ (C5, #40, #41, #47, #48, #49).

- Mỗi seed: tiết kiệm = 1 − Σ cost / Σ cost(B0′), recall theo mutant gộp các fold test; báo trung bình ± độ lệch chuẩn theo seed.
- Kiểm định (#49 Q1): trung bình chi phí và tỉ lệ bắt của từng mutant qua 5 seed, bootstrap cụm mutant một lần; Holm trên 5 cặp.
- #48: khoảng tin cậy hiệu recall, trội Pareto (mô tả, bổ sung sau khi xem seed 0).

Chạy: python summarize.py RESULTS_DIR OUT_DIR
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

from stats import bootstrap_diff, bootstrap_recall_diff, holm, recall_ci

PAIRS = [("b1", "match_b1", "b1", "natural"), ("b1h", "match_b1h", "b1h", "natural"),  # #49 Q2
         ("b2", "match_b2", "b2", "natural"), ("b3", "t95", "b3", "t95"), ("lr", "t95", "lr", "t95")]
NAN = float("nan")


def _one(x):
    f = x.has_fail == 1
    return pd.Series({"savings": 1 - x.cost.sum() / x.cost_b0prime.sum(),
                      "recall_mutant": x.caught[f].sum() / f.sum() if f.any() else NAN,
                      "n_fail": f.sum(), "n": len(x)})


def pooled(per, keys=("method", "op")):
    """Mỗi seed gộp các fold test, rồi trung bình ± độ lệch chuẩn theo seed."""
    keys = list(keys)
    s = per.groupby(keys + ["seed"], sort=False)[["has_fail", "caught", "cost", "cost_b0prime"]].apply(_one)
    out = s.groupby(keys, sort=False).agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.drop(columns=["n_fail_std", "n_std"]).reset_index()


def _avg(per):
    return per.groupby(["method", "op", "mutant_id"]).agg(cost=("cost", "mean"), caught=("caught", "mean"),
                                                          has_fail=("has_fail", "first"), base=("cost_b0prime", "first"))


def main_tests(per, pairs=PAIRS, n=10_000, left="lgbm"):
    avg, rows = _avg(per), []
    for name, op, other, oop in pairs:
        a, b = avg.loc[(left, op)], avg.loc[(other, oop)].reindex(avg.loc[(left, op)].index)
        assert b.cost.notna().all() and len(a) == len(b)  # cặp phải trên cùng tập mutant
        f = a.has_fail == 1
        lo, hi, p = bootstrap_diff(list(a.cost), list(b.cost), list(a.base), n=n)
        rlo, rhi = bootstrap_recall_diff(list(a.caught[f]), list(b.caught[f]), n=n)
        ds, dr = (b.cost.sum() - a.cost.sum()) / a.base.sum(), (a.caught[f] - b.caught[f]).mean()
        rows.append({"pair": f"{left}[{op}] vs {other}[{oop}]", f"savings_{left}": 1 - a.cost.sum() / a.base.sum(),
                     "savings_other": 1 - b.cost.sum() / a.base.sum(), f"recall_{left}": a.caught[f].mean(),
                     "recall_other": b.caught[f].mean(), "d_savings": ds, "ci_lo": lo, "ci_hi": hi, "p": p,
                     "d_recall": dr, "r_ci_lo": rlo, "r_ci_hi": rhi,
                     "pareto": bool(ds >= 0 and dr >= 0), "pareto_other": bool(ds <= 0 and dr <= 0)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(list(out.p))
    return out


def by_seed(per, pairs=PAIRS, left="lgbm"):
    """#49 Q1 độ nhạy: hiệu tiết kiệm và hiệu recall (ước lượng điểm) của từng cặp theo từng seed."""
    rows = []
    for seed, g in per.groupby("seed"):
        for name, op, other, oop in pairs:
            a, b = g[(g.method == left) & (g.op == op)], g[(g.method == other) & (g.op == oop)]
            fa, fb = a[a.has_fail == 1], b[b.has_fail == 1]
            rows.append({"seed": seed, "pair": f"{left}[{op}] vs {other}[{oop}]",
                         "d_savings": (b.cost.sum() - a.cost.sum()) / a.cost_b0prime.sum(),
                         "d_recall": fa.caught.mean() - fb.caught.mean()})
    return pd.DataFrame(rows)


def recall_cis(per, n=10_000):
    """#41: khoảng tin cậy 95% recall theo mutant, tỉ lệ bắt trung bình qua seed."""
    avg = _avg(per)
    return pd.DataFrame([{"method": m, "op": op, **dict(zip(("r_lo", "r_hi"), recall_ci(list(g.caught[g.has_fail == 1]), n=n)))}
                         for (m, op), g in avg.groupby(level=[0, 1], sort=False)])


def extras(tab, per):
    """Từ table.csv: sót xuyên service, (ii)/(iii), thời gian, PR/ROC-AUC — gộp fold theo seed rồi trung bình ± sd."""
    base = per.groupby(["seed", "fold", "method", "op"]).cost_b0prime.sum().rename("base").reset_index()
    t = tab.merge(base, on=["seed", "fold", "method", "op"])
    t["cm_n"], t["ii_b"], t["iii_b"] = t.cross_missed.fillna(0) * t.n_cross, t.savings_ii * t.base, t.savings_iii * t.base
    s = t.groupby(["method", "op", "seed"], sort=False).agg(
        cm_n=("cm_n", "sum"), n_cross=("n_cross", "sum"), ii_b=("ii_b", "sum"), iii_b=("iii_b", "sum"), base=("base", "sum"),
        time_fit_s=("time_fit_s", "mean"), time_pred_s=("time_pred_s", "mean"), pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"))
    s["cross_missed"], s["savings_ii"], s["savings_iii"] = s.cm_n / s.n_cross, s.ii_b / s.base, s.iii_b / s.base
    cols = ["cross_missed", "savings_ii", "savings_iii", "time_fit_s", "time_pred_s", "pr_auc", "roc_auc"]
    out = s[cols].groupby(["method", "op"], sort=False).agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def _load(pattern, name):
    frames = []
    for d in sorted(glob.glob(pattern)):
        f = pd.read_csv(Path(d) / name)
        f["seed"] = json.loads((Path(d) / "params.json").read_text(encoding="utf-8"))["seed"]
        frames.append(f)
    return pd.concat(frames, ignore_index=True), sorted(glob.glob(pattern))


def main(results, out):
    r, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per, src = _load(str(r / "ts02_eval/*_seed[0-4]"), "per_mutant.csv")
    tab, _ = _load(str(r / "ts02_eval/*_seed[0-4]"), "table.csv")
    lper, lsrc = _load(str(r / "ts02_eval_loso/*_seed[0-4]"), "per_mutant.csv")
    cls, csrc = _load(str(r / "ts02_cls/*_seed[0-4]"), "cls_report.csv")
    assert len(src) == len(lsrc) == len(csrc) == 5 and sorted(per.seed.unique()) == list(range(5))
    pooled(per).merge(extras(tab, per), on=["method", "op"]).merge(recall_cis(per), on=["method", "op"]) \
        .to_csv(out / "table.csv", index=False)
    main_tests(per).to_csv(out / "main_tests.csv", index=False)
    by_seed(per).to_csv(out / "main_tests_by_seed.csv", index=False)
    pooled(per[per.in_scope]).to_csv(out / "subset_T.csv", index=False)  # #40
    pooled(per, ("method", "op", "type")).to_csv(out / "by_type.csv", index=False)
    pooled(per, ("method", "op", "service")).to_csv(out / "by_service.csv", index=False)
    pooled(lper).to_csv(out / "loso.csv", index=False)
    pooled(lper, ("method", "op", "service")).to_csv(out / "loso_by_service.csv", index=False)
    num = [c for c in cls.columns if c not in ("method", "op", "seed")]
    c = cls.groupby(["method", "op"], sort=False)[num].agg(["mean", "std"])
    c.columns = [f"{a}_{b}" for a, b in c.columns]
    c.reset_index().to_csv(out / "cls_report.csv", index=False)  # #48
    (out / "params.json").write_text(json.dumps({
        "plan": "plans/2026-09-30-d2-mo-hinh-va-danh-gia.md (D2-9)", "decisions": "#40 #41 #47 #48 #49",
        "eval": src, "loso": lsrc, "cls": csrc, "bootstrap_n": 10_000, "bootstrap_seed": 0, "pairs": PAIRS},
        indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:3])

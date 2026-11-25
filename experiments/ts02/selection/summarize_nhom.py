"""G2a (#88): tóm tắt phép chia theo nhóm gần trùng (plans/2026-10-15-g2a-chia-theo-nhom-gan-trung.md, bước 5).

- table.csv: như summarize.py (tiết kiệm, recall, (ii)/(iii), khoảng tin cậy recall) cùng giây mỗi mutant, cho ba phép chia
  (chinh: D2; nhom; nhom_gia).
- main_tests.csv: họ 5 cặp ở hai mức bootstrap (theo mutant như C5; theo cụm gần trùng như #61), Holm riêng mỗi mức;
  nhãn theo #58; nhãn chỉ coi là đổi khi đổi ở cả hai mức.
- paired.csv: hiệu ghép theo mutant (nhom − chinh, nhom − nhom_gia), khoảng bootstrap theo cụm gần trùng.
- strata.csv: theo tầng (ba nhóm lớn; phần còn lại) và theo cỡ nhóm (1; 2–9; ≥ 16).
- folds.csv: cơ cấu fold ngoài; fold_recall.csv: recall theo fold của B4-LGBM, B4-LR ở t95, t99 với mốc nhiễu nhị thức.
Trước khi ghi: phép chia chính tính lại phải trùng #49 (ts02_summary) và #61 (ts02_cell5); B0, B1, B1h trùng D2.

Chạy: python summarize_nhom.py RESULTS_DIR RUN_DIR OUT_DIR
"""
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from cell5 import _cols, label, point
from desc57 import boot_cells
from nguong import noise_band
from run_nhom import near_dup_groups, provenance
from stats import holm
from summarize import PAIRS, _avg, _load, extras, main_tests, pooled, recall_cis

PLAN = "plans/2026-10-15-g2a-chia-theo-nhom-gan-trung.md"
SPLITS = {"chinh": "ts02_eval", "nhom": "ts02_eval_nhom", "nhom_gia": "ts02_eval_nhom_gia"}
SUMMARY = "ts02_summary/20261005-195113_seed0-4"
CELL5 = "ts02_cell5/20261008-180616_seed0-4/cell_tests.csv"
N, SEED = 10_000, 0
TARGET = {"t95": 0.95, "t99": 0.99}


def binom_band(n, p, q=(0.05, 0.95)):
    """Khoảng 5–95 % của recall một fold có n mutant có fail khi recall thật bằng p."""
    return tuple(float(binom.ppf(x, n, p) / n) for x in q)


def changed(new, main):
    """Cách đọc chốt trước: nhãn chỉ coi là đổi khi đổi ở cả hai mức bootstrap."""
    return all(new[lv] != main[lv] for lv in ("mutant", "near_dup"))


def size_strata(groups):
    """mutant → (tầng, cỡ nhóm). Ba nhóm lớn là ba nhóm có ≥ 16 mutant; không có nhóm cỡ 10–15."""
    sz = Counter(groups.values())
    big = {g for g, n in sz.items() if n >= 16}
    assert len(big) == 3 and not any(10 <= n <= 15 for n in sz.values())
    return {i: ("ba nhóm lớn" if g in big else "phần còn lại", "1" if sz[g] == 1 else "2–9" if sz[g] <= 9 else "≥ 16")
            for i, g in groups.items()}


def same_baselines(per_new, per_main):
    """B0, B1, B1h không học từ train: theo (seed, mutant), caught và e2e trùng, cost khớp rel_tol 1e-9."""
    k = ["seed", "method", "op", "mutant_id"]
    a, b = (p[p.method.isin(["b0", "b1", "b1h"])].set_index(k).sort_index() for p in (per_new, per_main))
    return bool(a.index.equals(b.index) and (a.caught == b.caught).all() and (a.e2e == b.e2e).all()
                and np.allclose(a.cost, b.cost, rtol=1e-9, atol=1e-12))


def contrast(per_x, per_y, keys, n=N):
    """Hiệu ghép theo mutant x − y (tỉ lệ bắt và chi phí trung bình qua seed) mỗi (phương pháp, điểm vận hành);
    khoảng bootstrap theo cụm (desc57.boot_cells)."""
    ax, ay, rows = _avg(per_x), _avg(per_y), []
    for (m, op), a in ax.groupby(level=[0, 1], sort=False):
        a = a.droplevel([0, 1])
        b = ay.loc[(m, op)].reindex(a.index)
        assert b.cost.notna().all()
        cols, k = _cols(a, b), [keys[i] for i in a.index]
        pt = point(*cols[:5], cols[5], k)
        (lo, hi), (rlo, rhi), *_ = boot_cells(*cols, k, n=n, seed=SEED)
        rows.append({"method": m, "op": op, "savings_x": pt["savings_a"], "savings_y": pt["savings_b"],
                     "d_savings": pt["d_savings"], "ci_lo": lo, "ci_hi": hi, "recall_x": pt["recall_a"],
                     "recall_y": pt["recall_b"], "d_recall": pt["d_recall"], "r_ci_lo": rlo, "r_ci_hi": rhi})
    return pd.DataFrame(rows)


def pair_tests(per, keys):
    """Họ 5 cặp: mức mutant như summarize.main_tests (#49); mức cụm gần trùng như #61. Holm riêng mỗi mức."""
    mt = main_tests(per).rename(columns={"savings_lgbm": "savings_a", "savings_other": "savings_b",
                                         "recall_lgbm": "recall_a", "recall_other": "recall_b"})
    mt = mt.drop(columns=["pareto", "pareto_other"]).assign(level="mutant")
    avg, nd = _avg(per), []
    for _, op, other, oop in PAIRS:
        a = avg.loc[("lgbm", op)]
        b = avg.loc[(other, oop)].reindex(a.index)
        cols, k = _cols(a, b), [keys[i] for i in a.index]
        (lo, hi), (rlo, rhi), *_, p, _, _ = boot_cells(*cols, k, n=N, seed=SEED)
        nd.append({"pair": f"lgbm[{op}] vs {other}[{oop}]", **point(*cols[:5], cols[5], k), "ci_lo": lo, "ci_hi": hi,
                   "p": p, "r_ci_lo": rlo, "r_ci_hi": rhi, "level": "near_dup"})
    nd = pd.DataFrame(nd)
    nd["p_holm"] = holm(list(nd.p))
    t = pd.concat([mt, nd], ignore_index=True)
    t["label"] = [label(r.p_holm, r.ci_lo, r.ci_hi, r.d_savings, r.r_ci_lo, r.r_ci_hi) for r in t.itertuples()]
    return t


def fold_structure(per):
    b = per[per.method == "b0"]
    s = b.groupby(["seed", "fold"]).agg(n=("mutant_id", "size"), n_fail=("has_fail", "sum"))
    return s.join(b.groupby(["seed", "fold", "type"]).size().unstack(fill_value=0).add_prefix("n_")).reset_index()


def fold_recall(per):
    rows, mad = [], []
    for m in ("lgbm", "lr"):
        for op, t in TARGET.items():
            g = per[(per.method == m) & (per.op == op) & (per.has_fail == 1)].groupby(["seed", "fold"]).caught
            g = g.agg(["mean", "size"])
            for (s, f), r in g.iterrows():
                rows.append({"method": m, "op": op, "seed": s, "fold": f, "n_fail": int(r["size"]), "recall": r["mean"],
                             **dict(zip(("band_lo", "band_hi"), binom_band(int(r["size"]), t)))})
            b = noise_band(g["size"].tolist(), [t] * len(g))
            mad.append({"method": m, "op": op, "mad": float((g["mean"] - t).abs().mean()), "mad_lo": b.mad_lo,
                        "mad_hi": b.mad_hi, "below": int((g["mean"] < t).sum()), "n_folds": len(g)})
    return pd.DataFrame(rows), pd.DataFrame(mad)


def _close(a, b):
    return np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-12, atol=1e-12)


def main(results, run_dir, out):
    assert sys.version_info < (3, 12)  # như cell5.py: từ 3.12 sum() cộng float theo cách khác
    res, out = Path(results), Path(out)
    per, tab, src = {}, {}, {}
    for sp, d in SPLITS.items():
        per[sp], src[sp] = _load(str(res / d / "*_seed[0-4]"), "per_mutant.csv")
        tab[sp], _ = _load(str(res / d / "*_seed[0-4]"), "table.csv")
        assert len(src[sp]) == 5 and sorted(per[sp].seed.unique()) == list(range(5)), sp
    keys = near_dup_groups(run_dir)
    for sp in ("nhom", "nhom_gia"):
        assert same_baselines(per[sp], per["chinh"]), sp  # DoD 4

    tabs = {k: [] for k in ("table", "main_tests", "strata", "folds", "fold_recall", "fold_mad")}
    st = size_strata({i: keys[i] for i in per["chinh"].mutant_id.unique()})
    for sp, p in per.items():
        cm = p.groupby(["method", "op", "seed"]).cost.mean().groupby(["method", "op"]).agg(["mean", "std"])
        cm.columns = ["cost_mean", "cost_std"]
        tabs["table"].append(pooled(p).merge(extras(tab[sp], p), on=["method", "op"]).merge(recall_cis(p), on=["method", "op"])
                             .merge(cm.reset_index(), on=["method", "op"]).assign(split=sp))
        tabs["main_tests"].append(pair_tests(p, keys).assign(split=sp))
        q = p.assign(tang=p.mutant_id.map(lambda i: st[i][0]), co=p.mutant_id.map(lambda i: st[i][1]))
        for by in ("tang", "co"):
            tabs["strata"].append(pooled(q, ("method", "op", by)).rename(columns={by: "stratum"}).assign(by=by, split=sp))
        tabs["folds"].append(fold_structure(p).assign(split=sp))
        fr, fm = fold_recall(p)
        tabs["fold_recall"].append(fr.assign(split=sp))
        tabs["fold_mad"].append(fm.assign(split=sp))
    tabs = {k: pd.concat(v, ignore_index=True) for k, v in tabs.items()}

    # Phép chia chính tính lại phải trùng kết quả đã báo (#49, #61)
    mt, ref = tabs["main_tests"], pd.read_csv(res / SUMMARY / "main_tests.csv").set_index("pair")
    c = mt[(mt.split == "chinh") & (mt.level == "mutant")].set_index("pair").loc[ref.index]
    assert _close(c[["d_savings", "ci_lo", "ci_hi", "p", "p_holm", "d_recall", "r_ci_lo", "r_ci_hi"]],
                  ref[["d_savings", "ci_lo", "ci_hi", "p", "p_holm", "d_recall", "r_ci_lo", "r_ci_hi"]])
    c5 = pd.read_csv(res / CELL5)
    c5 = c5[c5.level == "near_dup"].set_index("pair")
    c = mt[(mt.split == "chinh") & (mt.level == "near_dup")].set_index("pair").loc[c5.index]
    assert _close(c[["d_savings", "ci_lo", "ci_hi", "p", "p_holm", "r_ci_lo", "r_ci_hi"]],
                  c5[["d_savings", "ci_lo", "ci_hi", "p", "p_holm", "r_ci_lo", "r_ci_hi"]]) and (c.label == c5.label).all()
    t0 = pd.read_csv(res / SUMMARY / "table.csv").set_index(["method", "op"])
    c = tabs["table"][tabs["table"].split == "chinh"].set_index(["method", "op"]).loc[t0.index]
    assert _close(c[["savings_mean", "recall_mutant_mean", "r_lo", "r_hi"]], t0[["savings_mean", "recall_mutant_mean", "r_lo", "r_hi"]])

    lab = mt.set_index(["split", "level", "pair"]).label
    mt["changed_vs_chinh"] = [changed({lv: lab[r.split, lv, r.pair] for lv in ("mutant", "near_dup")},
                                      {lv: lab["chinh", lv, r.pair] for lv in ("mutant", "near_dup")})
                              for r in mt.itertuples()]
    tabs["paired"] = pd.concat([contrast(per["nhom"], per["chinh"], keys).assign(contrast="nhom - chinh"),
                                contrast(per["nhom"], per["nhom_gia"], keys).assign(contrast="nhom - nhom_gia")],
                               ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tabs.items():
        t.to_csv(out / f"{name}.csv", index=False)
    big = sorted({keys[i] for i, v in st.items() if v[0] == "ba nhóm lớn"})
    (out / "params.json").write_text(json.dumps({
        "plan": PLAN, "decision": "#88", "eval": src, "near_dup": "cell5.near_dup_key, 'loại|service|site'",
        "bootstrap": {"n": N, "seed": SEED, "mutant": "summarize.main_tests", "near_dup": "desc57.boot_cells"},
        "big_groups": big, "pairs": PAIRS,
        "checks": ["B0, B1, B1h của nhom và nhom_gia trùng D2 theo (seed, mutant)",
                   "phép chia chính tính lại trùng ts02_summary/main_tests.csv, table.csv (#49) và ts02_cell5 mức near_dup (#61)"],
        "note_seed_sd": "ba nhóm lớn rơi vào cùng các fold ở 5 seed, nên độ lệch chuẩn theo seed của nhom không so được với D2",
        **provenance(out)}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\npandas {pd.__version__}\nnumpy {np.__version__}\n"
                                 f"{platform.platform()}\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])

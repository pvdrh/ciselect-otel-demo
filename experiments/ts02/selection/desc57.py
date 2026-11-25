"""#57 (O4, O5) và O2L của #58: phân tích mô tả theo `plans/2026-10-08-phan-tich-o4-o5.md` bản 3 (duyệt ở #59).

Không p, không Holm; không tune hay fit lại để lấy số. Quyết định của B3, B3k, B4-LGBM ở t95 được dựng lại bằng
cls_report.rebuild (hàm này tự kiểm savings/recall với table.csv) chỉ để so tập test (o5_agree).
Mọi assert (E-1 đến E-11) chạy trước khi ghi file.

Chạy (PYTHONHASHSEED=0): python desc57.py RESULTS_DIR RUN_DIR BUILD_NOCACHE_DIR DEMO OUT_DIR
"""
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from cls_report import rebuild
from data import folds, load, meta_of, touches
from evaluate import e2e_cost
from extra_desc import _sign
from summarize import _avg, _load, by_seed, extras, main_tests, pooled

PLAN = "plans/2026-10-08-phan-tich-o4-o5.md"
LABEL = "#57, phân tích mô tả, thêm sau khi đã thấy kết quả D2; phụ thuộc cơ cấu mẫu mutant (M0 76/353, có fail 144/353)"
LABEL_O2L = ("#58, phân tích mô tả, thêm sau khi đã thấy kết quả D2; phụ thuộc cơ cấu mẫu mutant (M0 76/353, có fail "
             "144/353); bỏ khi tính, không huấn luyện lại; nhóm được chọn sau khi thấy kết quả D2; điểm match chỉ khớp "
             "mục tiêu trên toàn mẫu")
TEL_VARS = {"OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_COLLECTOR_HOST"}
O5_PAIRS = [("lgbm", "t95", "b3k", "t95"), ("lgbm", "t99", "b3k", "t99"),
            ("b3k", "t95", "b3", "t95"), ("b3k", "t99", "b3", "t99")]
SUMMARY = "ts02_summary/20261005-195113_seed0-4"
DROP = "ts02_extra_desc/20261007-182150_seed0-4/drop_service.csv"
PREDICTION = ("Dự đoán ghi trong plan trước khi chạy: hiệu tiết kiệm LGBM − B3k ở t95 nhỏ hơn hiệu LGBM − B3 "
              "(0,048)")
N, SEED = 10_000, 0


def groups(rows, labeled):
    """O2L: G_tel = M3 đổi đích telemetry của S; G_lg = mutant của load-generator (chỉ mutant có nhãn)."""
    tel = {r["mutant_id"] for r in rows
           if r["mutant_id"] in labeled and r["type"] == "M3" and r["site_id"].split(":")[2] in TEL_VARS}
    lg = {r["mutant_id"] for r in rows if r["mutant_id"] in labeled and r["service"] == "load-generator"}
    return tel, lg


def miss_split(per, keys=("method", "op")):
    """O4: sót = có fail và không bắt; (ii) không chạy e2e; (iii) chạy e2e nhưng không chọn test fail nào."""
    keys = list(keys)
    f = per[per.has_fail == 1]
    assert not ((f.e2e == 0) & (f.caught == 1)).any()  # E-4: caught = 0 khi không chạy e2e
    g = f.assign(n_fail=1, miss=(f.caught == 0).astype(int), miss_ii=(f.e2e == 0).astype(int),
                 miss_iii=((f.e2e == 1) & (f.caught == 0)).astype(int))
    s = g.groupby(keys + ["seed"], sort=False)[["n_fail", "miss", "miss_ii", "miss_iii"]].sum()
    out = s.groupby(level=keys, sort=False).sum()
    out["rate_ii"], out["rate_iii"] = out.miss_ii / out.n_fail, out.miss_iii / out.n_fail
    out["share_iii"] = (out.miss_iii / out.miss).where(out.miss > 0)  # số chính: gộp 5 seed
    for c in ("miss", "miss_ii", "miss_iii"):
        out[f"{c}_mean"] = s[c].groupby(level=keys, sort=False).mean()
        out[f"{c}_std"] = s[c].groupby(level=keys, sort=False).std()
    out["share_iii_seed_mean"] = (s.miss_iii / s.miss).where(s.miss > 0).groupby(level=keys, sort=False).mean()
    out["n_seed_no_miss"] = (s.miss == 0).groupby(level=keys, sort=False).sum()
    return out.reset_index()


def cross_only(d):
    """Mutant có fail mà mọi nhãn fail đều ở test không chạm S (§3.2)."""
    fails, svc = defaultdict(list), {m["mutant_id"]: m["service"] for m in d.mutants}
    for r in d.labels:
        if r["label"]:
            fails[r["mutant_id"]].append(r["test_id"])
    return {m for m, ts in fails.items() if not any(touches(meta_of(t), svc[m]) for t in ts)}


def _ci(xs):
    xs = sorted(xs)
    return (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]) if xs else (math.nan, math.nan)


def boot_cells(ca, cb, base, ka, kb, f, cell, n=N, seed=SEED):
    """O5: bootstrap cặp theo cụm là ô; hiệu tiết kiệm a − b = (Σb − Σa)/Σbase, hiệu recall a − b trên mutant fail.
    Trả ((lo, hi), (rlo, rhi), n_r = số lần lấy mẫu có mutant fail, p, q, q_r). #61 thêm p = min(1, 2q) như
    stats.bootstrap_diff, q = min(#{d ≤ 0}, #{d ≥ 0})/n của hiệu tiết kiệm, q_r như vậy cho hiệu recall trên n_r."""
    agg = defaultdict(lambda: [0.0] * 6)
    for i, c in enumerate(cell):
        v = agg[c]
        v[0], v[1], v[2] = v[0] + ca[i], v[1] + cb[i], v[2] + base[i]
        if f[i]:
            v[3], v[4], v[5] = v[3] + ka[i], v[4] + kb[i], v[5] + 1
    cs, rng, ds, dr = list(agg.values()), random.Random(seed), [], []
    for _ in range(n):
        t = [0.0] * 6
        for _ in cs:
            v = cs[rng.randrange(len(cs))]
            for j in range(6):
                t[j] += v[j]
        ds.append((t[1] - t[0]) / t[2])
        if t[5]:
            dr.append((t[3] - t[4]) / t[5])
    q = min(sum(x <= 0 for x in ds), sum(x >= 0 for x in ds)) / n
    q_r = min(sum(x <= 0 for x in dr), sum(x >= 0 for x in dr)) / len(dr) if dr else math.nan
    return _ci(ds), _ci(dr), len(dr), min(1.0, 2 * q), q, q_r


def in_ci(theta_sub, lo, hi, theta_full):
    """Tiêu chí của #53 (B): θ của tập nằm trong khoảng tin cậy của toàn mẫu và cùng dấu."""
    return _sign(theta_sub) == _sign(theta_full) and lo <= theta_sub <= hi


def agree(a, b):
    """(cùng quyết định e2e, cùng tập test); khi cả hai bỏ e2e thì tập test không có nghĩa."""
    same = a.e2e == b.e2e
    return same, same and (not a.e2e or a.tests == b.tests)


def cell_key(m):
    return m["service"], tuple(m["record"]["kinds"])


def cell_counts(mutants, splits):
    """Số mutant ở phía test có ô (S, loại file) rỗng, hoặc chỉ 1–2 mutant, ở phía train."""
    ms, c = {m["mutant_id"]: m for m in mutants}, Counter()
    for train, test in splits:
        n = Counter(cell_key(ms[i]) for i in train)
        for i in test:
            k = n[cell_key(ms[i])]
            c["empty" if k == 0 else "small" if k <= 2 else "other"] += 1
    return {"empty": c["empty"], "small": c["small"]}


def train_cell_size(mutants, seed):
    ms, out = {m["mutant_id"]: m for m in mutants}, {}
    for train, test in folds(mutants, seed, 5):
        n = Counter(cell_key(ms[i]) for i in train)
        out |= {i: n[cell_key(ms[i])] for i in test}
    return out


def subsets(per, tel, lg):
    return {"all": per, "-tel": per[~per.mutant_id.isin(tel)], "-lg": per[~per.mutant_id.isin(lg)],
            "-both": per[~per.mutant_id.isin(tel | lg)]}


def o2l_rq1(subs, cross):
    rows = []
    for name, p in subs.items():
        b1 = p[(p.method == "b1") & (p.op == "natural")]
        fail = set(b1[b1.has_fail == 1].mutant_id)
        co = cross & fail
        for _, g in b1.groupby("seed"):
            assert set(g[(g.has_fail == 1) & (g.caught == 0)].mutant_id) == co  # §3.2: B1 sót đúng nhóm này
        pl = pooled(p[p.method.isin(["b1", "b1h"]) & (p.op == "natural")]).set_index("method")
        rows.append({"subset": name, "n": b1.mutant_id.nunique(), "n_fail": len(fail), "n_cross_only": len(co),
                     "share_cross_only": len(co) / len(fail),
                     **{f"{m}_{k}": pl.loc[m, k] for m in ("b1", "b1h")
                        for k in ("savings_mean", "savings_std", "recall_mutant_mean", "recall_mutant_std")}})
    return pd.DataFrame(rows)


def o2l_rq2(subs):
    rows, k = [], ["seed", "mutant_id"]
    for name, p in subs.items():
        a = p[(p.method == "b1") & (p.op == "natural")].set_index(k)
        b = p[(p.method == "b2") & (p.op == "natural")].set_index(k).reindex(a.index)
        rec = ((a.has_fail == 1) & (a.caught == 0) & (b.caught == 1)).groupby(level="seed").sum()
        sv = lambda x: 1 - x.cost.groupby(level="seed").sum() / x.cost_b0prime.groupby(level="seed").sum()  # noqa: E731
        s1, s2 = sv(a), sv(b)
        rows.append({"subset": name, "b2_recovers_mean": rec.mean(), "b2_recovers_min": int(rec.min()),
                     "b2_recovers_max": int(rec.max()), "savings_b1_mean": s1.mean(), "savings_b2_mean": s2.mean(),
                     "d_savings_b1_b2_mean": (s1 - s2).mean(), "d_savings_b1_b2_std": (s1 - s2).std()})
    return pd.DataFrame(rows)


def o2l_rq3(subs):
    full, out = None, []
    for name, p in subs.items():
        t = main_tests(p).drop(columns=["p", "p_holm"])
        full = t.set_index("pair") if full is None else full  # tập đầu là toàn mẫu
        for _, r in t.iterrows():
            f = full.loc[r.pair]
            out.append({"subset": name, **r.to_dict(), "width": r.ci_hi - r.ci_lo, "r_width": r.r_ci_hi - r.r_ci_lo,
                        "abs_dev_savings": abs(r.d_savings - f.d_savings), "abs_dev_recall": abs(r.d_recall - f.d_recall),
                        "in_full_ci_savings": in_ci(r.d_savings, f.ci_lo, f.ci_hi, f.d_savings),
                        "in_full_ci_recall": in_ci(r.d_recall, f.r_ci_lo, f.r_ci_hi, f.d_recall),
                        "ci_contains0_savings": bool(r.ci_lo <= 0 <= r.ci_hi),
                        "ci_contains0_recall": bool(r.r_ci_lo <= 0 <= r.r_ci_hi)})
    return pd.DataFrame(out)


def o5_pairs(per, cell_of, E):
    avg, rows = _avg(per), []
    for left in ("lgbm", "b3k"):
        prs = [(o, op, o, oop) for lft, op, o, oop in O5_PAIRS if lft == left]
        mt, bs = main_tests(per, pairs=prs, left=left).set_index("pair"), by_seed(per, pairs=prs, left=left)
        for _, op, other, oop in prs:
            name = f"{left}[{op}] vs {other}[{oop}]"
            a = avg.loc[(left, op)]
            b = avg.loc[(other, oop)].reindex(a.index)
            f = list((a.has_fail == 1).astype(int))
            (lo, hi), (rlo, rhi), m, *_ = boot_cells(list(a.cost), list(b.cost), list(a.base), list(a.caught),
                                                 list(b.caught), f, [cell_of[i] for i in a.index])
            r, s = mt.loc[name], bs[bs.pair == name]
            rows.append({"pair": name, "savings_a": r[f"savings_{left}"], "savings_b": r.savings_other,
                         "recall_a": r[f"recall_{left}"], "recall_b": r.recall_other,
                         "d_savings": r.d_savings, "cell_ci_lo": lo, "cell_ci_hi": hi,
                         "mut_ci_lo": r.ci_lo, "mut_ci_hi": r.ci_hi,
                         "d_recall": r.d_recall, "cell_r_ci_lo": rlo, "cell_r_ci_hi": rhi,
                         "mut_r_ci_lo": r.r_ci_lo, "mut_r_ci_hi": r.r_ci_hi, "cell_boot_recall_n": m,
                         "seed_d_savings_min": s.d_savings.min(), "seed_d_savings_max": s.d_savings.max(),
                         "seed_d_recall_min": s.d_recall.min(), "seed_d_recall_max": s.d_recall.max(),
                         "seeds_same_sign_savings": int((s.d_savings.map(_sign) == _sign(r.d_savings)).sum()),
                         "seeds_same_sign_recall": int((s.d_recall.map(_sign) == _sign(r.d_recall)).sum()),
                         "e2e_decisions_per_seed": r.d_savings * a.base.sum() / E,
                         "cell_ci_contains0_savings": bool(lo <= 0 <= hi), "cell_ci_contains0_recall": bool(rlo <= 0 <= rhi)})
    return pd.DataFrame(rows)


def o5_by_cell(per, cell_of):
    avg = _avg(per[per.op == "t95"]).reset_index()
    avg["cell"] = avg.mutant_id.map(lambda i: "|".join([cell_of[i][0], *cell_of[i][1]]))
    rows = []
    for c, g in avg.groupby("cell"):
        x = g[g.method == "lgbm"]
        r = {"cell": c, "n": len(x), "n_fail": int(x.has_fail.sum()), "base": x.base.sum()}
        for m in ("lgbm", "b3k", "b3"):
            y = g[g.method == m]
            r |= {f"cost_{m}": y.cost.sum(), f"caught_{m}": y.caught[y.has_fail == 1].sum()}
        rows.append(r)
    return pd.DataFrame(rows)


def o5_agree(d, decs, size):
    ms, rows = {m["mutant_id"]: m for m in d.mutants}, []
    for a, b in (("lgbm", "b3k"), ("b3", "b3k")):
        recs = [{"type": ms[i]["type"], "cell_size": "n_k<=3" if size[s][i] <= 3 else "n_k>3",
                 **dict(zip(("same_e2e", "same_tests"), agree(x, decs[b, s][i])))}
                for s in range(5) for i, x in decs[a, s].items()]
        df = pd.DataFrame(recs)
        for by, gs in (("all", [("all", df)]), ("type", df.groupby("type")), ("cell_size", df.groupby("cell_size"))):
            rows += [{"pair": f"{a}[t95] vs {b}[t95]", "by": by, "group": k, "n": len(x),
                      "same_e2e": x.same_e2e.mean(), "same_tests": x.same_tests.mean()} for k, x in gs]
    return pd.DataFrame(rows)


def _rows(path, methods):
    with open(Path(path) / "table.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["method"] in methods and r["op"] == "t95"]


def _close(a, b):
    return all(math.isclose(x, y, rel_tol=0, abs_tol=1e-12) or (math.isnan(x) and math.isnan(y)) for x, y in zip(a, b))


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()


def main(results, run_dir, build_dir, demo, out):
    res, out = Path(results), Path(out)
    d = load(run_dir, build_dir, demo)
    E = e2e_cost(d.tests, d)
    per2, src2 = _load(str(res / "ts02_eval/*_seed[0-4]"), "per_mutant.csv")
    tab2, _ = _load(str(res / "ts02_eval/*_seed[0-4]"), "table.csv")
    perk, srck = _load(str(res / "ts02_eval_b3k/*_seed[0-4]"), "per_mutant.csv")
    tabk, _ = _load(str(res / "ts02_eval_b3k/*_seed[0-4]"), "table.csv")
    perl, srcl = _load(str(res / "ts02_eval_loso/*_seed[0-4]"), "per_mutant.csv")
    perk, tabk = perk[perk.method == "b3k"], tabk[tabk.method == "b3k"]  # dòng b0–b2 của run B3k lặp lại D2
    per, tab = pd.concat([per2, perk], ignore_index=True), pd.concat([tab2, tabk], ignore_index=True)
    st, mt = pd.read_csv(res / SUMMARY / "table.csv"), pd.read_csv(res / SUMMARY / "main_tests.csv")
    assert len(src2) == len(srck) == len(srcl) == 5 and sorted(perk.seed.unique()) == list(range(5))
    for s in srck:  # E-5
        assert json.loads((Path(s) / "params.json").read_text(encoding="utf-8"))["baseline_mismatches"] == 0, s
    n, n_fail = per.mutant_id.nunique(), int(per.drop_duplicates("mutant_id").has_fail.sum())
    assert (n, n_fail) == (353, 144)

    # O4 (E-1 đến E-4)
    o4, o4l = miss_split(per), miss_split(perl)
    o4t = miss_split(per, keys=("method", "op", "type"))
    for tbl, p in ((o4, per), (o4l, perl)):
        assert (p[p.has_fail == 1].groupby(["method", "op", "seed"]).size() == 144).all()  # E-4
        x = tbl.set_index(["method", "op"])
        assert x.loc[("b0", "natural"), "miss"] == 0  # E-3
        assert all(x.loc[(m, "natural"), "miss_ii"] == 0 for m in ("b1", "b1h", "b2"))  # E-2
        b1, b1h = (p[(p.method == m) & (p.has_fail == 1)].groupby("seed").caught.sum() for m in ("b1", "b1h"))
        assert (b1 == b1h).all()
    e1 = o4.set_index(["method", "op"])
    assert [tuple(int(e1.loc[k, c]) for c in ("miss", "miss_iii")) for k in
            (("b3", "t95"), ("lgbm", "t95"), ("lr", "t95"), ("b2", "natural"))] == [(41, 28), (40, 12), (21, 5), (33, 33)]

    # O2L (E-8 đến E-11)
    with open(Path(run_dir) / "mutants" / "mutants.csv", encoding="utf-8") as f:
        tel, lg = groups(list(csv.DictReader(f)), set(per.mutant_id))
    hf = per.drop_duplicates("mutant_id").set_index("mutant_id").has_fail
    assert (len(tel), int(hf[list(tel)].sum()), len(lg), int(hf[list(lg)].sum())) == (27, 26, 14, 5)
    assert tel & lg == {"s0-0065", "s0-0183"}
    subs = subsets(per2, tel, lg)
    sizes = {k: (p.mutant_id.nunique(), int(p.drop_duplicates("mutant_id").has_fail.sum())) for k, p in subs.items()}
    assert sizes == {"all": (353, 144), "-tel": (326, 118), "-lg": (339, 139), "-both": (314, 115)}  # E-8
    r1, r2, r3 = o2l_rq1(subs, cross_only(d)), o2l_rq2(subs), o2l_rq3(subs)
    x = r1.set_index("subset").loc["-tel"]
    assert x.n_cross_only == 32 and math.isclose(x.b1_recall_mutant_mean, 86 / 118)  # E-9
    s = st.set_index(["method", "op"])
    y = r1.set_index("subset").loc["all"]
    assert _close([y.b1_savings_mean, y.b1_recall_mutant_mean, y.b1h_savings_mean],
                  [s.loc[("b1", "natural"), "savings_mean"], s.loc[("b1", "natural"), "recall_mutant_mean"],
                   s.loc[("b1h", "natural"), "savings_mean"]])  # E-10
    cols = ["d_savings", "ci_lo", "ci_hi", "d_recall", "r_ci_lo", "r_ci_hi"]
    a3 = r3[r3.subset == "all"].set_index("pair")
    assert all(_close(a3.loc[p, cols], mt.set_index("pair").loc[p, cols]) for p in mt.pair)  # E-10
    dr = pd.read_csv(res / DROP)
    dr = dr[dr.service == "load-generator"].set_index("pair")
    l3 = r3[r3.subset == "-lg"].set_index("pair")
    assert all(_close(l3.loc[p, ["d_savings", "d_recall"]], dr.loc[p, ["d_savings", "d_recall"]]) for p in dr.index)  # E-11

    # O5
    cell_of = {m["mutant_id"]: cell_key(m) for m in d.mutants}
    keep = per.method.isin(["b3", "b3k", "lr", "lgbm"])
    o5t = pooled(per[keep]).merge(extras(tab[tab.method.isin(["b3", "b3k", "lr", "lgbm"])], per[keep]),
                                  on=["method", "op"])
    t = o5t.set_index(["method", "op"])
    assert all(_close(t.loc[k, ["savings_mean", "recall_mutant_mean"]], s.loc[k, ["savings_mean", "recall_mutant_mean"]])
               for k in t.index if k[0] != "b3k")
    o5p = o5_pairs(per, cell_of, E)
    o5ty = pooled(per[per.method.isin(["b3k", "b3", "lgbm"]) & (per.op == "t95")], keys=("method", "op", "type"))
    o5c = o5_by_cell(per[per.method.isin(["b3k", "b3", "lgbm"])], cell_of)
    assert len(o5c) == 33 and o5c.n.sum() == 353
    e7 = {"outer": [], "small": [], "inner": []}
    for sd in range(5):
        splits = folds(d.mutants, sd, 5)
        c = cell_counts(d.mutants, splits)
        ms = {m["mutant_id"]: m for m in d.mutants}
        e7["outer"].append(c["empty"])
        e7["small"].append(c["small"])
        e7["inner"].append(sum(cell_counts(d.mutants, folds([ms[i] for i in tr], sd, k=4))["empty"] for tr, _ in splits))
    assert e7["outer"] == [6, 6, 3, 4, 4] and e7["small"] == [13, 23, 24, 18, 28] and all(17 <= v <= 22 for v in e7["inner"])
    decs, size = {}, {sd: train_cell_size(d.mutants, sd) for sd in range(5)}
    for sd, (p2, pk) in enumerate(zip(sorted(src2), sorted(srck))):
        rb = rebuild(d, _rows(p2, ("b3", "lgbm")) + _rows(pk, ("b3k",)), sd)
        decs |= {(m, sd): rb[m, "t95"] for m in ("b3", "lgbm", "b3k")}
    o5a = o5_agree(d, decs, size)

    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("o4_miss_split", o4), ("o4_miss_split_by_type", o4t), ("o4_miss_split_loso", o4l),
                     ("o5_table", o5t), ("o5_pairs", o5p), ("o5_agree", o5a), ("o5_by_type", o5ty),
                     ("o5_by_cell", o5c), ("o2l_rq1", r1), ("o2l_rq2", r2), ("o2l_rq3", r3)):
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "params.json").write_text(json.dumps({
        "plan": PLAN, "decision": "#57, #58 (O2L), #59 (duyệt plan bản 3)", "label": LABEL, "label_o2l": LABEL_O2L,
        "prediction_o5": PREDICTION, "sources": {"d2": src2, "b3k": srck, "loso": srcl, "summary": SUMMARY, "drop": DROP},
        "data": str(run_dir), "bootstrap": {"n": N, "seed": SEED, "o5_cell": "33 ô (S, loại file)", "mutant": "summarize"},
        "E": E, "groups": {"tel": sorted(tel), "lg": sorted(lg)}, "subset_sizes": sizes, "e7_fallback": e7,
        "checks": "E-1 đến E-11 qua (assert trước khi ghi)", "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "git_head": _git("rev-parse", "HEAD"), "tree_dirty": bool(_git("status", "--porcelain", "--", "."))},
        indent=1, ensure_ascii=False), encoding="utf-8")
    import lightgbm
    import sklearn
    (out / "env.txt").write_text(f"Python {platform.python_version()}\npandas {pd.__version__}\nlightgbm "
                                 f"{lightgbm.__version__}\nscikit-learn {sklearn.__version__}\n{platform.platform()}\n",
                                 encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:6])

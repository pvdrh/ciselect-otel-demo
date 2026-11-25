"""Plan `plans/2026-10-14-phan-tich-mo-ta-gop.md` (duyệt ở #86): P1–P9, phân tích mô tả hậu nghiệm trên run đã có.

Không fit lại mô hình, không tune lại, không chạy demo, không có p hay Holm mới. Mọi phép kiểm tái hiện (bước 1 của plan)
chạy trước khi ghi; một phép không khớp thì dừng, không ghi gì.
Chạy: python p85.py RUN_DIR BUILD_NOCACHE_DIR DEMO RESULTS_DIR OUT_DIR
"""
import csv
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import b1, b2
from cell5 import near_dup_key
from data import load, meta_of, touches
from evaluate import cost, e2e_cost
from extra_desc import LATENCY, MODES, SCOPE, _be, b0p_rows, break_even, net_table, saved, with_h
from run import GRIDS
from summarize import PAIRS, _avg, _load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_labels import unhealthy_of  # noqa: E402

PLAN = "plans/2026-10-14-phan-tich-mo-ta-gop.md"
LABEL = "#86, phân tích mô tả, thêm sau khi đã thấy kết quả D2; không fit lại, không tune lại, không p hay Holm mới"
SUMMARY = "ts02_summary/20261005-195113_seed0-4"
EXTRA = "ts02_extra_desc/20261007-182150_seed0-4"
O4 = "ts02_desc57/20261008-135736_seed0-4/o4_miss_split.csv"
GROUPS = ("M0", "im_lang", "co_fail")
P7_METHODS = [("b1", "natural"), ("b1h", "natural"), ("b2", "natural"), ("b3", "t95"), ("lr", "t95"), ("lgbm", "t95"),
              ("lgbm", "match_b2"), ("lgbm_nofr", "t95")]
P9_DRAWS = 200
NAN = float("nan")


def pair_groups(bg, mt, pairs=PAIRS, b0p_s=NAN, left="lgbm"):
    """P1: Δ_g = contrib(B4-LGBM, g) − contrib(X, g); tỉ phần của M0 + im lặng; giây mỗi mutant mà nhóm có fail góp."""
    rows, ref = [], mt.set_index("pair")
    for _, op, other, oop in pairs:
        name = f"{left}[{op}] vs {other}[{oop}]"
        a = bg[(bg.method == left) & (bg.op == op)].set_index("group").contrib_mean
        b = bg[(bg.method == other) & (bg.op == oop)].set_index("group").contrib_mean
        d = (a - b).reindex(GROUPS)
        assert math.isclose(d.sum(), ref.loc[name, "d_savings"], abs_tol=1e-9), name
        rows.append({"pair": name, **{f"d_{g}": d[g] for g in GROUPS}, "d_total": d.sum(),
                     "share_M0_im": (d["M0"] + d["im_lang"]) / d.sum(), "s_co_fail": d["co_fail"] * b0p_s})
    return pd.DataFrame(rows)


def labels_rule(table, unhealthy, service_of, tests, rule):
    """P4: nhãn theo đúng luật lượt hợp lệ của build_labels; rule=all là C2 (L), rule=any là "fail ở ít nhất một lượt" (L′)."""
    by_run, runs_of, out = defaultdict(list), defaultdict(list), {}
    for r in table:
        by_run[r["run_id"]].append(r)
    for rid, rs in by_run.items():
        runs_of[rs[0]["mutant_id"]].append(rid)
    for mid, svc in service_of.items():
        runs = sorted(runs_of.get(mid, []))
        if not runs or by_run[runs[0]][0]["status"] == "build_error":
            continue
        valid = [rid for rid in runs if not (unhealthy.get(rid) and svc not in unhealthy[rid]) and by_run[rid][0]["test_id"]]
        oc = {rid: {r["test_id"]: r["outcome"] for r in by_run[rid]} for rid in valid}
        for t in tests if valid else ():
            out[mid, t] = int(rule(oc[rid].get(t) == "fail" for rid in valid))
    return out


def near_dup_in_train(key, fails, fold):
    """P7: N(s, i) = 1 khi có j ≠ i cùng khoá gần trùng, cùng tập test fail, nằm ở fold khác i trong seed s (tức ở train)."""
    out = {}
    for (s, i), f in fold.items():
        if fails[i]:
            out[s, i] = int(any(j != i and key[j] == key[i] and fails[j] == fails[i] and fold[s, j] != f
                                for (s2, j) in fold if s2 == s))
    return out


def step_break_even(sav_a, sav_b, miss_a, miss_b, b0p_s, n, seeds=5):
    """P5: từ điểm a (t95) lên b (t99): giây thêm mỗi mutant, mutant có fail bắt thêm mỗi seed, giây trên mỗi mutant bắt thêm."""
    ds, db = (sav_a - sav_b) * b0p_s, (miss_a - miss_b) / seeds
    return ds, db, ds * n / db if db > 0 else NAN


def be_vs_ref(x, ref, pi, m_lo, m_hi):
    """P6: hàm hoà vốn của #53 (C) với bên rẻ hơn là x, mốc là ref (cùng chỉ mục mutant, đã lấy trung bình qua seed)."""
    f = ref.has_fail == 1
    dsf, dsn = (ref.cost - x.cost)[f].mean(), (ref.cost - x.cost)[~f].mean()
    m = (ref.caught - x.caught)[f].mean()
    return _be(pi * dsf + (1 - pi) * dsn, pi, m, m_lo, m_hi)


def random_addition(t1, t2, pool, rng):
    """P9: giữ t1, thêm ngẫu nhiên đều, không lặp, đúng |t2 \\ t1| test lấy ngoài t1."""
    cand = sorted(set(pool) - set(t1))
    return set(t1) | {str(t) for t in rng.choice(cand, size=len(set(t2) - set(t1)), replace=False)}


def _caught(dec, fails):
    return bool(dec.e2e and fails & dec.tests)


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()


def main(run_dir, build_dir, demo, results, out):
    res, out, run_dir = Path(results), Path(out), Path(run_dir)
    d = load(run_dir, build_dir, demo)
    E = e2e_cost(d.tests, d)
    per, src = _load(str(res / "ts02_eval/*_seed[0-4]"), "per_mutant.csv")
    tab, _ = _load(str(res / "ts02_eval/*_seed[0-4]"), "table.csv")
    mt, summ = pd.read_csv(res / SUMMARY / "main_tests.csv"), pd.read_csv(res / SUMMARY / "table.csv").set_index(["method", "op"])
    bg = pd.read_csv(res / EXTRA / "by_group.csv")
    ms = {m["mutant_id"]: m for m in d.mutants}
    n = len(ms)
    lab = {(r["mutant_id"], r["test_id"]): r["label"] for r in d.labels}
    fails = {i: frozenset(t for t in d.tests if lab[i, t]) for i in ms}
    only_cross = lambda fs: {i for i, f in fs.items() if f and all(not touches(meta_of(t), ms[i]["service"]) for t in f)}  # noqa: E731
    s32 = only_cross(fails)
    b0p_s = per.drop_duplicates("mutant_id").cost_b0prime.mean()
    assert len(src) == 5 and n == 353 and sum(map(bool, fails.values())) == 144 and len(s32) == 32
    tabs, chk = {}, {}

    # P1 (R2)
    tabs["p1_cap_theo_nhom"] = pair_groups(bg, mt, PAIRS, b0p_s)

    # P2 (R15): 8 cấu hình B2 cố định; kiểm khớp per_mutant theo cấu hình D2 đã chọn ở từng (seed, fold)
    dec = {cfg: {i: b2(m, d, d.edges, *cfg) for i, m in ms.items()} for cfg in GRIDS["b2"]}
    chosen = {(r.seed, r.fold): tuple(json.loads(r.config)) for r in tab[tab.method == "b2"].itertuples()}
    rb2 = per[per.method == "b2"]
    for r in rb2.itertuples():
        x = dec[chosen[r.seed, r.fold]][r.mutant_id]
        assert math.isclose(cost(x, d), r.cost, rel_tol=0, abs_tol=1e-9) and _caught(x, fails[r.mutant_id]) == bool(r.caught)
    b2c = rb2[rb2.mutant_id.isin(s32)].groupby("seed").caught.sum().mean()
    b2r = rb2[rb2.has_fail == 1].groupby("seed").caught.mean().mean()
    assert math.isclose(b2c, 25.4) and math.isclose(round(b2r, 3), 0.954)
    chk["p2"] = f"cấu hình D2 đã chọn tái hiện đúng {len(rb2)} dòng per_mutant của B2; 25,4/32; recall {b2r:.4f}"
    sig, rows = {}, []
    for cfg, dd in dec.items():
        key = tuple(sorted((i, tuple(sorted(x.tests))) for i, x in dd.items()))
        rows.append({"k": "inf" if cfg[0] is None else cfg[0], "huong": cfg[1],
                     "bat_lai_32": sum(_caught(dd[i], fails[i]) for i in s32),
                     "recall": np.mean([_caught(dd[i], fails[i]) for i in ms if fails[i]]),
                     "ti_le_test": np.mean([len(dd[i].tests) / len(d.tests) for i in ms]),
                     "giay_moi_mutant": np.mean([cost(dd[i], d) for i in ms]),
                     "tiet_kiem": 1 - sum(cost(dd[i], d) for i in ms) / (b0p_s * n),
                     "so_fold_d2_chon": sum(c == cfg for c in chosen.values()), "trung_voi": sig.setdefault(key, str(cfg))})
    tabs["p2_b2_cau_hinh"] = pd.DataFrame(rows)

    # P3 (R14): mode b0p trong cận trên hiệu quả ròng; kiểm các mode cũ khớp net_effect.csv
    sc = json.loads((res / SCOPE).read_text(encoding="utf-8"))
    nh = sc["n"] - sc["bot"]
    classes = {"300": (sc["reason"]["ok"] / sc["n"], (sc["n"] - sc["reason"]["ok"] - sc["reason"]["docs_only"]) / sc["n"]),
               "104": (sc["reason_human"]["ok"] / nh, (nh - sc["reason_human"]["ok"] - sc["reason_human"]["docs_only"]) / nh)}
    lat = pd.read_csv(res / LATENCY).set_index(["mode", "group"]).median_s
    T, b0 = per[per.in_scope], per[per.method == "b0"]
    sv = {mode: saved(T[(T.method == m) & (T.op == op)], b0, E) for mode, (m, op) in MODES.items()}
    sv["oracle"] = saved(with_h(b0p_rows(T), set(T.mutant_id), E), b0, E)
    sv["b0p"] = saved(b0p_rows(T), b0, E)
    lat_m = {mode: (lat[mode, "in_scope"], lat[mode, "out_of_scope"]) for mode in MODES}
    lat_m["oracle"] = lat_m["b0p"] = (lat["b0p", "in_scope"], lat["b0p", "out_of_scope"])
    ne = net_table(sv, classes, float((b0.cost - b0.cost_b0prime).mean()), lat_m)
    keys = ["lop", "moc", "fallback", "mode", "c_job"]
    old = pd.read_csv(res / EXTRA / "net_effect.csv").astype({"lop": str})
    both = old.merge(ne[ne["mode"] != "b0p"], on=keys, suffixes=("_old", ""), validate="1:1")
    assert len(both) == len(old) and np.allclose(both.net_old, both.net, rtol=0, atol=1e-9)
    chk["p3"] = f"{len(old)} dòng net_effect.csv tái hiện đúng"
    tabs["p3_net_b0p"] = ne[ne["mode"] == "b0p"]

    # P4 (S10): nhãn L′ = fail ở ít nhất một lượt hợp lệ; RQ1, RQ2 dưới L và L′ (không huấn luyện lại B3, B4)
    table = list(csv.DictReader(open(run_dir / "table.csv", encoding="utf-8")))
    svc = {i: m["service"] for i, m in ms.items()}
    unh = unhealthy_of(run_dir)
    lab_all, lab_any = labels_rule(table, unh, svc, d.tests, all), labels_rule(table, unh, svc, d.tests, any)
    assert lab_all == lab
    chk["p4"] = "luật lượt hợp lệ tái hiện đúng labels.csv (L)"
    folds_of = per[per.method == "b1"].set_index(["seed", "mutant_id"]).fold.to_dict()
    rows = []
    for name, lb in (("L", lab_all), ("L′", lab_any)):
        fs = {i: frozenset(t for t in d.tests if lb[i, t]) for i in ms}
        oc, nf = only_cross(fs), sum(map(bool, fs.values()))
        rec_b1 = np.mean([_caught(b1(ms[i], d), fs[i]) for i in ms if fs[i]])
        per_seed = [np.mean([_caught(dec[chosen[s, folds_of[s, i]]][i], fs[i]) for i in ms if fs[i]]) for s in range(5)]
        per_seed32 = [sum(_caught(dec[chosen[s, folds_of[s, i]]][i], fs[i]) for i in oc) for s in range(5)]
        rows.append({"nhan": name, "mutant_co_fail": nf, "chi_lo_qua_service_khac": len(oc), "recall_b1": rec_b1,
                     "recall_b2_d2": np.mean(per_seed), "b2_d2_bat_lai_chi_lo_qua_service_khac": np.mean(per_seed32),
                     **{f"recall_b2_{'inf' if k is None else k}_{h}": np.mean([_caught(dec[k, h][i], fs[i]) for i in ms if fs[i]])
                        for k, h in GRIDS["b2"]}})
    tabs["p4_nhan_mot_luot"] = pd.DataFrame(rows)
    r0 = rows[0]
    assert (r0["mutant_co_fail"], r0["chi_lo_qua_service_khac"]) == (144, 32) and round(r0["recall_b1"], 3) == 0.778 \
        and math.isclose(r0["recall_b2_d2"], b2r)

    # P5 (S27): t95 → t99
    o4 = pd.read_csv(res / O4).set_index(["method", "op"])
    rows = []
    for m in ("b3", "lr", "lgbm"):
        for op in ("t95", "t99"):
            assert math.isclose(summ.loc[(m, op), "recall_mutant_mean"] * 720, 720 - o4.loc[(m, op), "miss"], abs_tol=1e-6)
        ds, db, be = step_break_even(summ.loc[(m, "t95"), "savings_mean"], summ.loc[(m, "t99"), "savings_mean"],
                                     o4.loc[(m, "t95"), "miss"], o4.loc[(m, "t99"), "miss"], b0p_s, n)
        rows.append({"method": m, "giay_them_moi_mutant": ds, "bat_them_moi_seed": db, "giay_moi_mutant_bat_them": be,
                     "lan_e2e_day_du": be / E})
    tabs["p5_t95_t99"] = pd.DataFrame(rows)

    # P6 (S38): B1, B2 so với B0′; kiểm hàm bằng cặp B4-LGBM–B4-LR của break_even.csv
    pi = 144 / 353
    avg = _avg(pd.concat([per, b0p_rows(per)], ignore_index=True))
    lt, lr_ = avg.loc[("lgbm", "t95")], avg.loc[("lr", "t95")]
    ref = mt.set_index("pair").loc["lgbm[t95] vs lr[t95]"]
    got = be_vs_ref(lt, lr_.reindex(lt.index), pi, -ref.r_ci_hi, -ref.r_ci_lo)
    old_be = pd.read_csv(res / EXTRA / "break_even.csv").set_index(["pair", "pi"]).loc[("lgbm[t95] vs lr[t95]", "thiet_ke")]
    assert np.allclose(got[1:], [old_be.be_s, old_be.be_lo, old_be.be_hi], rtol=1e-12, atol=0)
    assert np.allclose(break_even(per, mt, E).set_index(["pair", "pi"]).be_s.dropna(),
                       pd.read_csv(res / EXTRA / "break_even.csv").set_index(["pair", "pi"]).be_s.dropna())
    chk["p6"] = f"hàm hoà vốn tái hiện {old_be.be_s:.1f} s của cặp B4-LGBM–B4-LR"
    rows, bp = [], avg.loc[("b0p", "natural")]
    for m in ("b1", "b2"):
        x = avg.loc[(m, "natural")].reindex(bp.index)
        lo, hi = summ.loc[(m, "natural"), "r_lo"], summ.loc[(m, "natural"), "r_hi"]
        case, be, be_lo, be_hi = be_vs_ref(x, bp, pi, 1 - hi, 1 - lo)
        rows.append({"method": m, "case": case, "giay_tiet_kiem_moi_mutant": (bp.cost - x.cost).mean(),
                     "lot_them_tren_144": (bp.caught - x.caught)[bp.has_fail == 1].sum(), "be_s": be, "be_lo": be_lo,
                     "be_hi": be_hi, "be_lan_e2e": be / E})
    tabs["p6_hoa_von_b0p"] = pd.DataFrame(rows)

    # P7 (G2 b): có hay không bản gần trùng cùng tập fail ở fold train
    mrow = {m["mutant_id"]: m for m in csv.DictReader(open(run_dir / "mutants" / "mutants.csv", encoding="utf-8"))}
    key = {i: near_dup_key(mrow[i]) for i in ms}
    nd = near_dup_in_train(key, fails, folds_of)
    n98 = sum(any(j != i and key[j] == key[i] and fails[j] == fails[i] for j in ms) for i in ms if fails[i])
    assert (n98, sum(nd.values()), len(nd)) == (98, 436, 720)
    chk["p7"] = "98/144 mutant có bản gần trùng cùng tập fail; 436/720 lượt có bản đó ở fold train"
    pf = per[per.has_fail == 1].assign(nd=lambda x: [nd[s, i] for s, i in zip(x.seed, x.mutant_id)])
    rows, rows_t = [], []
    for m, op in P7_METHODS:
        r = pf[(pf.method == m) & (pf.op == op)]
        g = r.groupby("nd").caught.agg(["size", "mean"])
        rows.append({"method": m, "op": op, "luot_co": g.loc[1, "size"], "recall_co": g.loc[1, "mean"],
                     "luot_khong": g.loc[0, "size"], "recall_khong": g.loc[0, "mean"], "hieu": g.loc[1, "mean"] - g.loc[0, "mean"]})
        for (t, v), gg in r.groupby(["type", "nd"]).caught.agg(["size", "mean"]).iterrows():
            rows_t.append({"method": m, "op": op, "type": t, "nd": v, "luot": gg["size"], "recall": gg["mean"]})
    tabs["p7_gan_trung"], tabs["p7_gan_trung_theo_loai"] = pd.DataFrame(rows), pd.DataFrame(rows_t)

    # P8 (R3): tỉ trọng phần cố định trong thời gian e2e của từng lượt
    runs = defaultdict(list)
    for r in table:
        runs[r["run_id"]].append(r)
    rows = []
    for rid, rs in runs.items():
        if rs[0]["status"] == "build_error" or not rs[0]["test_id"]:
            continue
        col = [r for r in rs if r["test_id"] == "test_collector::test_collector_healthy"]
        t0, tt = float(rs[0]["t_start"]), float(rs[0]["t_test_tel"])
        rows.append({"run_id": rid, "nhom": "sạch" if rs[0]["mutant_id"] == "clean" else (
            "có test fail" if any(r["outcome"] == "fail" for r in rs) else "không test fail"),
                     "ti_trong": (t0 + float(col[0]["duration_s"])) / (t0 + tt), "e2e_s": t0 + tt})
    p8 = pd.DataFrame(rows)
    tabs["p8_ti_trong_co_dinh"] = p8.groupby("nhom").agg(
        luot=("ti_trong", "size"), trung_vi=("ti_trong", "median"), q1=("ti_trong", lambda x: x.quantile(0.25)),
        q3=("ti_trong", lambda x: x.quantile(0.75)), e2e_trung_vi_s=("e2e_s", "median")).reset_index()

    # P9 (R17, G1 (a), chọn trước khi tính ở #86): thêm ngẫu nhiên đúng số test B2 thêm, giữ tập test của B1
    t1 = {i: b1(ms[i], d).tests for i in ms}
    order = [(s, f, i) for s in range(5) for f in range(5)
             for i in sorted(per[(per.method == "b2") & (per.seed == s) & (per.fold == f)].mutant_id)]
    draws = []
    for r in range(P9_DRAWS):
        rng, acc = np.random.default_rng(r), defaultdict(lambda: [0, 0, 0.0])
        for s, f, i in order:
            tt = random_addition(t1[i], dec[chosen[s, f]][i].tests, d.tests, rng)
            x = type(dec[chosen[s, f]][i])({ms[i]["service"]}, True, tt)
            c = _caught(x, fails[i])
            a = acc[s]
            a[0] += c and i in s32
            a[1] += c and bool(fails[i])
            a[2] += cost(x, d)
        draws.append({"draw": r, "bat_lai_32": np.mean([acc[s][0] for s in range(5)]),
                      "recall": np.mean([acc[s][1] / 144 for s in range(5)]),
                      "giay_moi_mutant": np.mean([acc[s][2] / n for s in range(5)])})
    dr = pd.DataFrame(draws)
    ref9 = {"B2 (D2)": (b2c, b2r, rb2.groupby("seed").cost.mean().mean()),
            "B1": (0.0, summ.loc[("b1", "natural"), "recall_mutant_mean"], per[per.method == "b1"].cost.mean()),
            "B0′": (32.0, 1.0, b0p_s)}
    rows = [{"ben": k, "bat_lai_32": v[0], "recall": v[1], "giay_moi_mutant": v[2]} for k, v in ref9.items()]
    rows.append({"ben": f"ngẫu nhiên, trung bình {P9_DRAWS} lần rút", **dr[["bat_lai_32", "recall", "giay_moi_mutant"]].mean()})
    rows += [{"ben": f"ngẫu nhiên, phân vị {q}", **dr[["bat_lai_32", "recall", "giay_moi_mutant"]].quantile(q / 100)}
             for q in (2.5, 97.5)]
    tabs["p9_ngau_nhien"], tabs["p9_cac_lan_rut"] = pd.DataFrame(rows), dr
    chk["p9"] = f"{P9_DRAWS} lần rút; mỗi mutant thêm đúng |B2 \\ B1| test"

    summary = pd.concat([
        tabs["p1_cap_theo_nhom"].assign(muc="P1")[["muc", "pair", "share_M0_im", "s_co_fail"]].rename(
            columns={"pair": "dong", "share_M0_im": "gia_tri_1", "s_co_fail": "gia_tri_2"}),
        tabs["p5_t95_t99"].assign(muc="P5")[["muc", "method", "giay_moi_mutant_bat_them", "lan_e2e_day_du"]].rename(
            columns={"method": "dong", "giay_moi_mutant_bat_them": "gia_tri_1", "lan_e2e_day_du": "gia_tri_2"}),
        tabs["p6_hoa_von_b0p"].assign(muc="P6")[["muc", "method", "be_s", "be_lan_e2e"]].rename(
            columns={"method": "dong", "be_s": "gia_tri_1", "be_lan_e2e": "gia_tri_2"}),
        tabs["p9_ngau_nhien"].assign(muc="P9")[["muc", "ben", "bat_lai_32", "recall"]].rename(
            columns={"ben": "dong", "bat_lai_32": "gia_tri_1", "recall": "gia_tri_2"})], ignore_index=True)

    out.mkdir(parents=True, exist_ok=True)  # mọi assert đã qua mới ghi
    for name, t in tabs.items():
        t.to_csv(out / f"{name}.csv", index=False)
    summary.to_csv(out / "table.csv", index=False)
    (out / "params.json").write_text(json.dumps({
        "plan": PLAN, "decision": "#86", "label": LABEL, "data": str(run_dir), "build_nocache": str(build_dir),
        "demo": str(demo), "eval": src, "summary": str(res / SUMMARY), "extra_desc": str(res / EXTRA), "o4": str(res / O4),
        "E": E, "b0p_s": b0p_s, "pi": pi, "p7_methods": P7_METHODS, "p9_draws": P9_DRAWS,
        "p9_rng": "numpy default_rng(r), r = 0..199; duyệt seed 0–4, fold 0–4, mutant theo mutant_id",
        "kiem_tai_hien": chk, "p85_py_blob": _git("hash-object", "p85.py"), "git_head": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain", "--", "."), "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\npandas {pd.__version__}\nnumpy {np.__version__}\n"
                                 f"{platform.platform()}\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:6])

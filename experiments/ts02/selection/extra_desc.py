"""#53 (mô tả, thêm sau khi đã thấy kết quả D2): bộ lọc h trực giao, phân rã theo nhóm, bỏ một service, cận trên hiệu quả
ròng mỗi lần chạy CI, bộ cột quy đổi, độ nhạy theo chi phí cố định (E3).

Plan `plans/2026-10-07-phan-tich-mo-ta-53.md` (bản 2, duyệt #55). Hậu xử lý run có sẵn: không fit lại, không tune lại,
không p hay Holm mới.

Chạy: python extra_desc.py RUN_DIR BUILD_NOCACHE_DIR DEMO RESULTS_DIR OUT_DIR
"""
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import b1
from data import COLLECTOR, load
from evaluate import b0prime, cost, e2e_cost, metrics
from oracle import o_mut, o_test
from summarize import PAIRS, _avg, _load, pooled

PLAN = "plans/2026-10-07-phan-tich-mo-ta-53.md"
LABEL = "#53, phân tích mô tả, thêm sau khi đã thấy kết quả D2; phụ thuộc cơ cấu mẫu mutant (M0 76/353, có fail 144/353)"
X = [("b0p", "natural"), ("b1", "natural"), ("b2", "natural"), ("b3", "t95"), ("lr", "t95"), ("lgbm", "t95"),
     ("lgbm", "match_b2")]
CONTRASTS = [(("b2", "natural"), ("b0p", "natural")), (("lgbm", "t95"), ("b2", "natural")),
             (("lgbm", "match_b2"), ("b2", "natural"))]
GROUPS = ("M0", "im_lang", "co_fail")
MODES = {"default": ("lgbm", "match_b2"), "b2": ("b2", "natural"), "b1h": ("b1h", "natural")}  # default ≈ default_mode.json
C1 = {"c1_thap": 75 + 8.6, "c1_cao": 78 + 9.2}  # job select, venv không cache: t7.md (pip 75–78 s; select_step 8,6–9,2 s)
PI = {"thiet_ke": 144 / 353, "0.10": 0.10, "0.01": 0.01}
FS = (0.0, 0.5, 1.0)
SUMMARY = "ts02_summary/20261005-195113_seed0-4"
SCOPE = "ts02_tool_scope/20261002-140142_seed0/summary.json"
LATENCY = "ts02_tool_ops/20261006-235406_seed0/latency.csv"
DEFAULT = "ts02_tool_model/20261005-223232_seed0/default_mode.json"
NAN = float("nan")


def _ms(t):
    """Trung bình ± độ lệch chuẩn theo seed (như summarize.pooled)."""
    return {f"{c}_{k}": getattr(t[c], k)() for c in t.columns for k in ("mean", "std")}


def b0p_rows(per):
    """B0′ dựng từ dòng b0: build S + toàn bộ e2e (cost_b0prime), bắt mọi mutant có fail."""
    return per[per.method == "b0"].assign(method="b0p", cost=lambda x: x.cost_b0prime, caught=lambda x: x.has_fail, e2e=1)


def with_h(rows, co, E):
    """Bộ lọc h (#32) như yếu tố trực giao: mutant chỉ sửa comment thì chỉ build S, bỏ e2e; caught giữ nguyên (E-1)."""
    m = rows.mutant_id.isin(co)
    return rows.assign(cost=rows.cost.where(~m, rows.cost_b0prime - E), e2e=rows.e2e.where(~m, 0))


def _group(x):
    return x.type.where(x.type == "M0", x.has_fail.astype(int).map({0: "im_lang", 1: "co_fail"}))


def _seed(rows):
    """Mỗi seed: savings, recall theo mutant (công thức summarize._one), đóng góp của từng nhóm vào savings."""
    def one(x):
        base, f, g = x.cost_b0prime.sum(), x.has_fail == 1, _group(x)
        return pd.Series({"savings": 1 - x.cost.sum() / base, "recall": x.caught[f].sum() / f.sum() if f.any() else NAN,
                          **{f"contrib_{k}": (x.cost_b0prime - x.cost)[g == k].sum() / base for k in GROUPS}})
    return rows.groupby("seed")[["type", "has_fail", "caught", "cost", "cost_b0prime"]].apply(one)


def h_factor(per, co, E, xs=X, contrasts=CONTRASTS):
    """(A) X và X+h; đóng góp của M0 và M1–M3 vào savings(X+h); hiệu ghép cặp theo seed giữa các X."""
    per, tabs = pd.concat([per, b0p_rows(per)]), {}
    for m, op in xs:
        r = per[(per.method == m) & (per.op == op)]
        a, h = _seed(r), _seed(with_h(r, co, E))
        assert (a.recall == h.recall).all()  # E-2
        tabs[m, op] = pd.DataFrame({"savings": a.savings, "recall": a.recall, "savings_h": h.savings, "recall_h": h.recall,
                                    "delta": h.savings - a.savings, "contrib_M0_h": h.contrib_M0,
                                    "contrib_M1_M3_h": h.contrib_im_lang + h.contrib_co_fail})
    rows = [{"x": f"{m}[{op}]", "kind": "X", **_ms(t)} for (m, op), t in tabs.items()]
    rows += [{"x": f"{a[0]}[{a[1]}] − {b[0]}[{b[1]}]", "kind": "hiệu", **_ms(tabs[a] - tabs[b])} for a, b in contrasts]
    return pd.DataFrame(rows)


def by_group(per):
    """(B) Tiết kiệm trong từng nhóm (pooled) và đóng góp của nhóm vào savings cả tập; các nhóm cộng lại bằng savings."""
    inside = pooled(per.assign(group=_group(per)), ("method", "op", "group"))
    rows = []
    for (m, op), r in per.groupby(["method", "op"], sort=False):
        t = _seed(r)
        rows += [{"method": m, "op": op, "group": g, **_ms(t[[f"contrib_{g}", "savings"]].set_axis(
            ["contrib", "savings_total"], axis=1))} for g in GROUPS]
    return pd.DataFrame(rows).merge(inside, on=["method", "op", "group"], how="left")


def _theta(a, b):
    f = a.has_fail == 1
    return (b.cost.sum() - a.cost.sum()) / a.base.sum(), (a.caught[f] - b.caught[f]).mean()


def drop_service(per, mt, pairs=PAIRS, left="lgbm"):
    """(B) Bỏ một service khỏi tập đánh giá (không train lại; khác LOSO), ước lượng điểm đúng công thức main_tests."""
    avg, svc, rows = _avg(per), per.groupby("mutant_id").service.first(), []
    for _, op, other, oop in pairs:
        name = f"{left}[{op}] vs {other}[{oop}]"
        a = avg.loc[(left, op)]
        b, ref, sv = avg.loc[(other, oop)].reindex(a.index), mt.set_index("pair").loc[name], svc.reindex(a.index)
        ts, tr = _theta(a, b)
        assert math.isclose(ts, ref.d_savings, rel_tol=1e-9) and math.isclose(tr, ref.d_recall, rel_tol=1e-9, abs_tol=1e-12)
        for s in sorted(sv.unique()):
            k = sv != s
            ds, dr = _theta(a[k], b[k])
            rows.append({"pair": name, "service": s, "n_bo": int((~k).sum()), "n_fail_bo": int(a.has_fail[~k].sum()),
                         "d_savings": ds, "d_recall": dr, "theta_savings": ts, "theta_recall": tr,
                         "dev_savings": abs(ds - ts), "dev_recall": abs(dr - tr),
                         "in_ci_savings": bool(ref.ci_lo <= ds <= ref.ci_hi),
                         "in_ci_recall": bool(ref.r_ci_lo <= dr <= ref.r_ci_hi), "ci_lo": ref.ci_lo, "ci_hi": ref.ci_hi,
                         "r_ci_lo": ref.r_ci_lo, "r_ci_hi": ref.r_ci_hi, "d_seconds": (b.cost.sum() - a.cost.sum()) / len(a)})
    return pd.DataFrame(rows)


def _sign(x):
    return int(x > 0) - int(x < 0)


def robust(theta, thetas, lo, hi):
    """Tiêu chí đặt trước cho d_savings: mọi θ₋s giữ dấu của θ và nằm trong KTC 95 % gốc."""
    return all(_sign(t) == _sign(theta) and lo <= t <= hi for t in thetas)


def jack_summary(detail):
    rows = []
    for name, g in detail.groupby("pair", sort=False):
        r = {"pair": name, "d_seconds": g.d_seconds.iloc[0]}
        for k in ("savings", "recall"):
            t, S = g[f"d_{k}"], len(g)
            r |= {f"theta_{k}": g[f"theta_{k}"].iloc[0], f"min_{k}": t.min(), f"max_{k}": t.max(),
                  f"n_out_ci_{k}": int((~g[f"in_ci_{k}"].astype(bool)).sum()),
                  f"se_jack_{k}": math.sqrt((S - 1) / S * ((t - t.mean()) ** 2).sum())}
        r["robust_savings"] = robust(r["theta_savings"], list(g.d_savings), g.ci_lo.iloc[0], g.ci_hi.iloc[0])
        rows.append(r)
    return pd.DataFrame(rows)


def e2e_share(per, E):
    """E-3: tỉ trọng e2e trong B0′ = n·E/ΣB0′; savings_e2e = savings / tỉ trọng này với mọi phương pháp build đúng S."""
    s = per.drop_duplicates(["seed", "mutant_id"]).groupby("seed").cost_b0prime.agg(["size", "sum"])
    v = s["size"] * E / s["sum"]
    assert v.max() - v.min() < 1e-12
    return float(v.iloc[0])


def saved(rows, b0, E):
    """(C) Giây tiết kiệm trung bình mỗi mutant: (ii), (iii) so với B0′; (i) = B0 − B0′, chỉ cộng khi so với R_b."""
    x = rows.merge(b0[["seed", "mutant_id", "cost"]].rename(columns={"cost": "cost_b0"}), on=["seed", "mutant_id"],
                   validate="m:1")
    assert len(x) == len(rows)
    x = x.assign(i=x.cost_b0 - x.cost_b0prime, ii=(x.e2e == 0) * E, iii=(x.e2e == 1) * (x.cost_b0prime - x.cost))
    return x.groupby("seed")[["i", "ii", "iii"]].mean().mean().to_dict()


def net_table(sv, classes, pen_tool, lat):
    """(C) net = p·s_saved − r·phạt − c_job; cận trên, < 0 thì "thiệt". lat: mode → (trung vị in_scope, out_of_scope)."""
    rows = []
    for name, (p, r) in classes.items():
        for ref in ("R_a", "R_b"):
            for fb in ("F_tool", "F_ci"):
                pen = pen_tool if (ref, fb) == ("R_a", "F_tool") else 0.0
                for mode, s in sv.items():
                    s_i = s["i"] if ref == "R_b" else 0.0
                    for cj, c in {**C1, "c2": p * lat[mode][0] + (1 - p) * lat[mode][1]}.items():
                        v = p * (s_i + s["ii"] + s["iii"]) - r * pen - c
                        rows.append({"lop": name, "p": p, "r": r, "moc": ref, "fallback": fb, "mode": mode, "c_job": cj,
                                     "c_job_s": c, "s_i": s_i, "s_ii": s["ii"], "s_iii": s["iii"],
                                     "s_saved": s_i + s["ii"] + s["iii"], "phat": pen, "net": v, "s_star": c / p,
                                     "doc": "thiệt" if v < 0 else "không xác định dấu",
                                     "duong_gang": "select chạy trước build (needs: select)"})
    return pd.DataFrame(rows)


def _be(num, pi, m, lo, hi):
    """Trường hợp nhìn từ phía LGBM; hoà vốn = giây tiết kiệm trên mỗi mutant có fail lọt thêm; khoảng theo KTC của m."""
    if num > 0 and m > 0:
        case = "đánh đổi"
    elif num < 0 and m < 0:
        case = "đánh đổi ngược"
    elif num >= 0 and m <= 0:
        case = "ngang" if num == m == 0 else "trội"
    else:
        case = "bị trội"
    be = num / (pi * m) if case.startswith("đánh đổi") else NAN
    if num > 0 and hi > 0:
        rng = (num / (pi * hi), num / (pi * lo) if lo > 0 else math.inf)
    elif num < 0 and lo < 0:
        rng = (num / (pi * lo), num / (pi * hi) if hi < 0 else math.inf)
    else:
        rng = (NAN, NAN)
    return case, be, *rng


def break_even(per, mt, E, pairs=PAIRS, left="lgbm", pis=PI):
    """(C) Hoà vốn của đánh đổi recall – tiết kiệm cho mọi cặp #49, không lọc theo dấu; BE(π) = Δs(π)/(π·m_f)."""
    avg, rows = _avg(per), []
    for _, op, other, oop in pairs:
        name = f"{left}[{op}] vs {other}[{oop}]"
        a = avg.loc[(left, op)]
        b, ref, f = avg.loc[(other, oop)].reindex(a.index), mt.set_index("pair").loc[name], a.has_fail == 1
        m, lo, hi, nf = -ref.d_recall, -ref.r_ci_hi, -ref.r_ci_lo, int(f.sum())
        assert math.isclose(m, -(a.caught[f] - b.caught[f]).mean(), rel_tol=1e-9, abs_tol=1e-12)
        dsf, dsn = (b.cost - a.cost)[f].mean(), (b.cost - a.cost)[~f].mean()
        for pn, pi in pis.items():
            num = pi * dsf + (1 - pi) * dsn
            case, be, be_lo, be_hi = _be(num, pi, m, lo, hi)
            rows.append({"pair": name, "pi": pn, "pi_val": pi, "ds_f": dsf, "ds_nf": dsn, "ds_pi": num, "m_f": m,
                         "m_lo": lo, "m_hi": hi, "n_fail": nf, "lot_them": m * nf, "lot_them_lo": lo * nf,
                         "lot_them_hi": hi * nf, "case": case, "be_s": be, "be_e2e": be / E, "be_phut": be / 60,
                         "be_lo": be_lo, "be_hi": be_hi, "be_lo_e2e": be_lo / E, "be_hi_e2e": be_hi / E,
                         "be_lo_phut": be_lo / 60, "be_hi_phut": be_hi / 60})
    return pd.DataFrame(rows)


def dur_sel(rows, E, F):
    """Σ thời lượng test đã chọn (không gồm collector) khi chạy e2e; chỉ đúng với phương pháp build đúng S (bỏ B0)."""
    return (rows.cost - (rows.cost_b0prime - E) - F).where(rows.e2e == 1, 0.0)


def conversion(per, cls, E, F, W):
    """(D) Bộ cột mô tả, không fit lại. Bỏ B0 (build mọi service nên dur_sel không áp dụng)."""
    D, rows = E - F, []
    per = per[per.method != "b0"]

    def one(y):
        ds, f = dur_sel(y, E, F), y.has_fail == 1
        return pd.Series({"skip_rate": (y.e2e == 0).mean(), "tts_c0": 1 - ds.sum() / (len(y) * D),
                          "tts_cW": 1 - (y.e2e * (W + ds)).sum() / (len(y) * (W + D)),
                          "savings": 1 - y.cost.sum() / y.cost_b0prime.sum(),
                          "recall_mutant": y.caught[f].sum() / f.sum() if f.any() else NAN})
    for (m, op), r in per.groupby(["method", "op"], sort=False):
        for hf in ("all", 0, 1):
            x = r if hf == "all" else r[r.has_fail == hf]
            t = x.groupby("seed")[["e2e", "cost", "cost_b0prime", "caught", "has_fail"]].apply(one)
            c = None if cls is None or hf != "all" else cls[(cls.method == m) & (cls.op == op)].set_index("seed")
            if c is not None and len(c):
                assert c.index.equals(t.index)  # đủ seed, không lặng lẽ bỏ seed thiếu
                tot, g = c.support_0 + c.support_1, r.groupby("seed")
                t["sel_frac"] = (c.tp + c.fp) / tot  # có tính collector
                t["sel_frac_e2e"] = (c.tp + c.fp) / (tot * g.e2e.sum() / g.size())
                t["recall_test"] = c.recall_1
            rows.append({"method": m, "op": op, "has_fail": str(hf), "n": len(x) / x.seed.nunique(), **_ms(t)})
    return pd.DataFrame(rows)


def fixed_cost(per, E, F, fs=FS):
    """(E3) cost_f = (B0′ − E) + e2e·(f·F + dur_sel); B0′_f = (B0′ − E) + f·F + D. Bỏ B0."""
    rows = []
    for (m, op), r in per[per.method != "b0"].groupby(["method", "op"], sort=False):
        ds, build = dur_sel(r, E, F), r.cost_b0prime - E
        for f in fs:
            g = pd.DataFrame({"seed": r.seed, "c": build + r.e2e * (f * F + ds), "b": build + f * F + E - F}) \
                .groupby("seed")[["c", "b"]].sum()
            s = 1 - g.c / g.b
            rows.append({"method": m, "op": op, "f": f, "savings_mean": s.mean(), "savings_std": s.std()})
    return pd.DataFrame(rows)


def oracle_rows(d):
    """Cận trên lý tưởng #51 (O-test, O-mut) dưới dạng dòng per_mutant, để tính lại theo (E3)."""
    ids, rows = [m["mutant_id"] for m in d.mutants], []
    for name, fn in (("O-test", o_test), ("O-mut", o_mut)):
        dec = fn(d, ids)
        rows += [{"method": name, "op": "natural", "mutant_id": m["mutant_id"], "type": m["type"], "service": m["service"],
                  "in_scope": m["record"]["in_scope"], "has_fail": m["has_fail"], "caught": m["has_fail"],
                  "e2e": int(dec[m["mutant_id"]].e2e), "cost": cost(dec[m["mutant_id"]], d),
                  "cost_b0prime": cost(b0prime(m, d), d)} for m in d.mutants]
    return pd.DataFrame(rows)


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()


def main(run_dir, build_dir, demo, results, out):
    res, out = Path(results), Path(out)
    d = load(run_dir, build_dir, demo)
    E, F = e2e_cost(d.tests, d), d.t_start + d.W
    per, src = _load(str(res / "ts02_eval/*_seed[0-4]"), "per_mutant.csv")
    cls, csrc = _load(str(res / "ts02_cls/*_seed[0-4]"), "cls_report.csv")
    mt = pd.read_csv(res / SUMMARY / "main_tests.csv")
    n, n_fail = per.mutant_id.nunique(), int(per.drop_duplicates("mutant_id").has_fail.sum())
    assert len(src) == len(csrc) == 5 and sorted(per.seed.unique()) == list(range(5)) and (n, n_fail) == (353, 144)

    # Kiểm trước khi tính (điểm dừng của plan)
    co = {m["mutant_id"] for m in d.mutants if m["record"]["comment_only"]}
    assert co == set(per[per.type == "M0"].mutant_id) and len(co) == 76 and per[per.mutant_id.isin(co)].has_fail.max() == 0
    key = ["seed", "fold", "mutant_id"]
    h1 = with_h(per[per.method == "b1"], co, E).set_index(key).sort_index()
    ref = per[per.method == "b1h"].set_index(key).sort_index()
    assert h1.index.equals(ref.index) and np.allclose(h1.cost, ref.cost, rtol=1e-9, atol=0) \
        and (h1.caught == ref.caught).all() and (h1.e2e == ref.e2e).all()  # B1 + h = b1h
    ds = dur_sel(per[per.method != "b0"], E, F)
    assert ds.min() > -1e-9 and ds.max() < E - F + 1e-9
    ms = {m["mutant_id"]: m for m in d.mutants}
    r1 = per[per.method == "b1"]  # DoD (D): dur_sel khớp tập test thật của B1 trên mọi mutant
    want = r1.mutant_id.map(lambda i: sum(d.dur[t] for t in b1(ms[i], d).tests if t != COLLECTOR))
    assert np.allclose(dur_sel(r1, E, F), want, rtol=0, atol=1e-9)
    assert ((cls.support_0 + cls.support_1) == len(d.tests) * n).all()
    b0c = cls[cls.method == "b0"]
    assert ((b0c.tp + b0c.fp) == (b0c.support_0 + b0c.support_1)).all()  # B0 (và B0′) chọn mọi test
    cls_full = pd.concat([cls, b0c.assign(method="b0p")])
    full = pd.concat([per, b0p_rows(per)], ignore_index=True)
    have = cls_full.groupby(["method", "op"]).seed.nunique()
    assert all(have.get(k, 0) == 5 for k in full[full.method != "b0"].groupby(["method", "op"]).groups)

    tabs = {}
    tabs["h_factor"] = h_factor(per, co, E)  # (A)
    g = by_group(per)  # (B)
    chk = g.groupby(["method", "op"]).agg(c=("contrib_mean", "sum"), s=("savings_total_mean", "first"))
    assert np.allclose(chk.c, chk.s, rtol=1e-9, atol=1e-12)
    tabs["by_group"] = g
    tabs["drop_service"] = drop_service(per, mt)
    tabs["drop_service_summary"] = jack_summary(tabs["drop_service"])
    sc = json.loads((res / SCOPE).read_text(encoding="utf-8"))  # (C)
    nh = sc["n"] - sc["bot"]
    classes = {"300": (sc["reason"]["ok"] / sc["n"], (sc["n"] - sc["reason"]["ok"] - sc["reason"]["docs_only"]) / sc["n"]),
               "104": (sc["reason_human"]["ok"] / nh,
                       (nh - sc["reason_human"]["ok"] - sc["reason_human"]["docs_only"]) / nh)}
    assert classes == {"300": (11 / 300, 277 / 300), "104": (11 / 104, 81 / 104)}
    lat = pd.read_csv(res / LATENCY).set_index(["mode", "group"]).median_s
    T, b0 = per[per.in_scope], per[per.method == "b0"]
    sv = {mode: saved(T[(T.method == m) & (T.op == op)], b0, E) for mode, (m, op) in MODES.items()}
    sv["oracle"] = saved(with_h(b0p_rows(T), set(T.mutant_id), E), b0, E)
    lat_m = {mode: (lat[mode, "in_scope"], lat[mode, "out_of_scope"]) for mode in MODES}
    lat_m["oracle"] = (lat["b0p", "in_scope"], lat["b0p", "out_of_scope"])  # chi phí CLI tối thiểu (mode không chọn gì)
    pen = float((b0.cost - b0.cost_b0prime).mean())
    tabs["net_effect"] = net_table(sv, classes, pen, lat_m)
    tabs["break_even"] = break_even(per, mt, E)
    tabs["conversion"] = conversion(full, cls_full, E, F, d.W)  # (D)
    ids = [m["mutant_id"] for m in d.mutants]  # (E3)
    orc = pd.concat([oracle_rows(d).assign(seed=s, fold=-1) for s in range(5)], ignore_index=True)
    allr = pd.concat([full, orc], ignore_index=True)
    fc = fixed_cost(allr, E, F)
    one = fc[fc.f == 1].set_index(["method", "op"]).savings_mean
    p1 = pooled(allr[allr.method != "b0"]).set_index(["method", "op"]).savings_mean.reindex(one.index)
    assert np.allclose(one, p1, rtol=1e-9, atol=1e-12)  # f = 1 là savings chính
    assert math.isclose(one["O-test", "natural"], metrics(o_test(d, ids), d)["savings"], rel_tol=1e-9)
    assert math.isclose(one["O-mut", "natural"], metrics(o_mut(d, ids), d)["savings"], rel_tol=1e-9)
    tabs["fixed_cost_sens"] = fc

    out.mkdir(parents=True, exist_ok=True)  # mọi assert đã qua mới ghi
    for name, t in tabs.items():
        t.to_csv(out / f"{name}.csv", index=False)
    big = max(d.dur, key=d.dur.get)
    (out / "params.json").write_text(json.dumps({
        "plan": PLAN, "decision": "#53 (plan duyệt ở #55)", "label": LABEL, "data": str(run_dir), "build_nocache": str(build_dir),
        "demo": str(demo), "eval": src, "cls": csrc, "summary": str(res / SUMMARY), "scope": str(res / SCOPE),
        "latency": str(res / LATENCY), "default_mode_file": str(res / DEFAULT),
        "default_mode": json.loads((res / DEFAULT).read_text(encoding="utf-8"))["mode"],
        "E": E, "t_start": d.t_start, "W": d.W, "D": E - F, "e2e_share": e2e_share(per, E),
        "largest_test": {"test": big, "dur": d.dur[big], "share_of_D": d.dur[big] / (E - F)},
        "n_mutants": n, "n_fail": n_fail, "n_comment_only": len(co), "classes_p_r": classes,
        "docs_only": "12/300, 12/104: công cụ không build, không e2e; chênh so với mốc giả định 0",
        "unit": "commit ≈ PR (squash-merge, bước 0 của plan)", "c1": C1,
        "c1_source": "docs/2026-10-06-cong-cu-van-hanh/t7.md dòng 22–24 (venv + pip 75–78 s; select_step 8,6–9,2 s)",
        "oracle_c2_mode": "b0p", "penalty_F_tool_vs_R_a": pen, "pi": PI, "fs": FS, "modes": MODES, "x": X,
        "contrasts": CONTRASTS, "github_billing": "quy tắc làm tròn phút: cần user tra, không phải số đo",
        "comments_py_blob": _git("hash-object", "comments.py"), "extra_desc_py_blob": _git("hash-object", "extra_desc.py"),
        "git_head": _git("rev-parse", "HEAD"), "dirty": _git("status", "--porcelain", "--", "."),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\npandas {pd.__version__}\nnumpy {np.__version__}\n"
                                 f"{platform.platform()}\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:6])

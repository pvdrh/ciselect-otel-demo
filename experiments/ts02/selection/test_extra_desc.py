import math

import pandas as pd
import pytest

from data import COLLECTOR, Data
from extra_desc import (_be, _group, b0p_rows, break_even, by_group, conversion, drop_service, e2e_share, fixed_cost,
                        h_factor, jack_summary, net_table, oracle_rows, robust, saved, with_h)

# 4 mutant × 2 seed, tính tay. Build: a 40 s, b 60 s. E = t_start + W + D = 90 + 10 = 100 s.
# m0 M0 (chỉ comment), m1 M2 có fail, m2 M1 im lặng (ngoài T), m3 M3 có fail. B0′ = 140, 140, 160, 160 (Σ 600); B0 = 200.
E, F, W = 100.0, 90.0, 20.0
MUT = {"m0": ("M0", "a", 0, True), "m1": ("M2", "a", 1, True), "m2": ("M1", "b", 0, False), "m3": ("M3", "b", 1, True)}
B0P = {"m0": 140.0, "m1": 140.0, "m2": 160.0, "m3": 160.0}
# phương pháp → mutant → (e2e, cost, caught seed 0, caught seed 1)
METH = {("b0", "natural"): {m: (1, 200.0, f, 0 if m == "m1" else f) for m, (_, _, f, _) in MUT.items()},  # m1 seed 1: caught ≠ has_fail
        ("b1", "natural"): {"m0": (1, 132.0, 0, 0), "m1": (1, 132.0, 1, 1), "m2": (1, 153.0, 0, 0), "m3": (1, 153.0, 1, 1)},
        ("b1h", "natural"): {"m0": (0, 40.0, 0, 0), "m1": (1, 132.0, 1, 1), "m2": (1, 153.0, 0, 0), "m3": (1, 153.0, 1, 1)},
        ("lgbm", "t95"): {"m0": (0, 40.0, 0, 0), "m1": (1, 135.0, 1, 0), "m2": (0, 60.0, 0, 0), "m3": (1, 154.0, 1, 1)}}


def _per():
    rows = []
    for (method, op), ms in METH.items():
        for mid, (e2e, cost, c0, c1) in ms.items():
            typ, svc, fail, ins = MUT[mid]
            rows += [{"seed": s, "fold": 0, "method": method, "op": op, "mutant_id": mid, "type": typ, "service": svc,
                      "in_scope": ins, "has_fail": fail, "caught": c, "e2e": e2e, "cost": cost, "cost_b0prime": B0P[mid]}
                     for s, c in ((0, c0), (1, c1))]
    return pd.DataFrame(rows)


PER = _per()
PAIR = [("b1", "t95", "b1", "natural")]  # lgbm[t95] vs b1[natural]
MT = pd.DataFrame([{"pair": "lgbm[t95] vs b1[natural]", "d_savings": 181 / 600, "ci_lo": 0.25, "ci_hi": 0.31,
                    "d_recall": -0.25, "r_ci_lo": -0.5, "r_ci_hi": 0.1}])


def _row(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    assert m.sum() == 1, kw
    return df[m].iloc[0]


def test_b0p_is_build_s_plus_full_e2e_and_catches_every_fail():
    r = b0p_rows(PER).set_index(["seed", "mutant_id"])
    assert (r.method == "b0p").all() and (r.e2e == 1).all()
    assert r.loc[(0, "m2"), "cost"] == 160.0 and r.loc[(1, "m1"), "caught"] == 1 and r.loc[(1, "m0"), "caught"] == 0


def test_h_skips_e2e_only_on_comment_only_and_b1_plus_h_is_b1h():
    b1 = PER[PER.method == "b1"]
    h = with_h(b1, {"m0"}, E).set_index(["seed", "mutant_id"])
    assert h.loc[(0, "m0"), "cost"] == 40.0 and h.loc[(0, "m0"), "e2e"] == 0  # B0′ − E = build S
    b1h = PER[PER.method == "b1h"].set_index(["seed", "mutant_id"])
    assert (h.cost == b1h.cost).all() and (h.caught == b1h.caught).all()


def test_h_factor_by_hand():
    t = h_factor(PER, {"m0"}, E, xs=[("b0p", "natural"), ("b1", "natural"), ("lgbm", "t95")],
                 contrasts=[(("lgbm", "t95"), ("b1", "natural"))]).set_index("x")
    b1 = t.loc["b1[natural]"]
    assert b1.savings_mean == pytest.approx(30 / 600) and b1.savings_h_mean == pytest.approx(122 / 600)
    assert b1.recall_h_mean == b1.recall_mean == 1.0
    assert b1.contrib_M0_h_mean == pytest.approx(100 / 600) and b1.contrib_M1_M3_h_mean == pytest.approx(22 / 600)
    assert t.loc["b0p[natural]", "delta_mean"] == pytest.approx(E / 600)  # chạy đủ e2e ở M0 → Δ = E/ΣB0′
    assert t.loc["lgbm[t95]", "delta_mean"] == 0.0  # đã bỏ e2e ở m0
    c = t.loc["lgbm[t95] − b1[natural]"]
    assert c.savings_h_mean == pytest.approx((211 - 122) / 600) and c.recall_h_mean == pytest.approx(-0.25)
    assert c.kind == "hiệu"
    assert c.savings_h_std == 0.0 and c.recall_h_std == pytest.approx(0.5 / 2 ** 0.5)  # độ lệch chuẩn mẫu của [0; −0,5] theo seed


def test_group_contributions_sum_to_savings():
    g = by_group(PER).set_index(["method", "op", "group"])
    b1 = g.loc[("b1", "natural")]
    assert b1.loc["M0", "contrib_mean"] == pytest.approx(8 / 600)
    assert b1.loc["im_lang", "contrib_mean"] == pytest.approx(7 / 600)
    assert b1.loc["co_fail", "contrib_mean"] == pytest.approx(15 / 600)
    assert b1.loc["co_fail", "savings_mean"] == pytest.approx(1 - 285 / 300)  # tiết kiệm trong nhóm
    for (m, op), x in g.groupby(level=[0, 1]):
        assert x.contrib_mean.sum() == pytest.approx(x.savings_total_mean.iloc[0])


def test_drop_service_by_hand_and_no_drop_matches_main_tests():
    d = drop_service(PER, MT, PAIR)
    a, b = _row(d, service="a"), _row(d, service="b")
    assert (a.n_bo, a.n_fail_bo) == (2, 1)
    assert a.d_savings == pytest.approx(92 / 320) and a.d_recall == pytest.approx(0.0)
    assert b.d_savings == pytest.approx(89 / 280) and b.d_recall == pytest.approx(-0.5)
    assert a.dev_savings == pytest.approx(abs(92 / 320 - 181 / 600))
    assert bool(a.in_ci_savings) and not bool(b.in_ci_savings)  # 0,3179 > 0,31
    with pytest.raises(AssertionError):  # ước lượng điểm không khớp main_tests → dừng
        drop_service(PER, MT.assign(d_savings=0.5), PAIR)


def test_robust_and_jackknife_summary():
    assert robust(0.3, [0.28, 0.32], 0.25, 0.35) and not robust(0.3, [0.28, 0.36], 0.25, 0.35)
    assert not robust(0.01, [0.02, -0.001], -0.05, 0.05)  # đổi dấu
    s = jack_summary(drop_service(PER, MT, PAIR)).iloc[0]
    assert s.se_jack_savings == pytest.approx(abs(89 / 280 - 92 / 320) / 2)  # S = 2: √(½·Σ(θ₋s − θ̄)²)
    assert s.n_out_ci_savings == 1 and not s.robust_savings
    assert s.d_seconds == pytest.approx((570 - 389) / 4)


def test_e2e_share_converts_savings():
    share = e2e_share(PER, E)
    assert share == pytest.approx(400 / 600)
    b1 = PER[(PER.method == "b1") & (PER.seed == 0)]
    s, s_e2e = 1 - b1.cost.sum() / b1.cost_b0prime.sum(), 1 - (b1.cost - (b1.cost_b0prime - E)).sum() / (len(b1) * E)
    assert s_e2e == pytest.approx(s / share)


def test_saved_on_T_split_i_ii_iii():
    lg = PER[(PER.method == "lgbm") & PER.in_scope]
    s = saved(lg, PER[PER.method == "b0"], E)
    assert s["ii"] == pytest.approx(100 / 3) and s["iii"] == pytest.approx(11 / 3) and s["i"] == pytest.approx(160 / 3)
    oracle = with_h(b0p_rows(PER[PER.in_scope]), set(MUT), E)
    assert saved(oracle, PER[PER.method == "b0"], E)["ii"] == pytest.approx(E)


def test_net_table_formula_and_reading_rule():
    t = net_table({"lgbm": {"i": 160 / 3, "ii": 100 / 3, "iii": 11 / 3}}, {"x": (0.1, 0.8)}, pen_tool=50.0,
                  lat={"lgbm": (2.0, 1.0)})
    assert len(t) == 2 * 2 * 3
    r = _row(t, moc="R_a", fallback="F_tool", c_job="c1_thap")
    assert r.net == pytest.approx(0.1 * 37 - 0.8 * 50 - 83.6) and r.doc == "thiệt" and r.s_star == pytest.approx(836.0)
    assert _row(t, moc="R_b", fallback="F_tool", c_job="c1_thap").phat == 0.0
    r = _row(t, moc="R_b", fallback="F_ci", c_job="c2")
    assert r.c_job_s == pytest.approx(0.1 * 2.0 + 0.9 * 1.0) and r.phat == 0.0
    assert r.net == pytest.approx(0.1 * (37 + 160 / 3) - 1.1) and r.doc == "không xác định dấu"


def test_break_even_by_hand_and_cases():
    t = break_even(PER, MT, E, PAIR, pis={"half": 0.5})
    r = t.iloc[0]
    assert r.ds_f == pytest.approx(-2.0) and r.ds_nf == pytest.approx(92.5) and r.m_f == pytest.approx(0.25)
    assert r.case == "đánh đổi" and r.be_s == pytest.approx(45.25 / 0.125) and r.be_e2e == pytest.approx(3.62)
    assert r.be_lo == pytest.approx(45.25 / 0.25) and math.isinf(r.be_hi)  # KTC của m chứa giá trị ≤ 0
    assert r.be_lo_phut == pytest.approx(181 / 60) and math.isinf(r.be_hi_phut)
    assert r.lot_them == pytest.approx(0.25 * 2)  # m_f × số mutant có fail


def test_conversion_b0p_zero_and_skip_all_one():
    per = pd.concat([PER, b0p_rows(PER), with_h(b0p_rows(PER), set(MUT), E).assign(method="skip")])
    cls = pd.DataFrame([{"seed": s, "method": "b0p", "op": "natural", "tp": 10, "fp": 30, "support_0": 30, "support_1": 10,
                         "recall_1": 1.0} for s in (0, 1)])  # B0′ chọn mọi test
    c = conversion(per, cls, E, F, W)
    b = _row(c, method="b0p", has_fail="all")
    assert (b.skip_rate_mean, b.tts_c0_mean, b.tts_cW_mean, b.sel_frac_mean, b.sel_frac_e2e_mean) == (0.0, 0.0, 0.0, 1.0, 1.0)
    s = _row(c, method="skip", has_fail="all")
    assert (s.skip_rate_mean, s.tts_c0_mean, s.tts_cW_mean) == (1.0, 1.0, 1.0)
    lg = _row(c, method="lgbm", has_fail="all")  # dur_sel: m1 5 s, m3 4 s (154 − 60 − 90); n·D = 40
    assert lg.tts_c0_mean == pytest.approx(1 - 9 / 40) and lg.tts_cW_mean == pytest.approx(1 - (2 * W + 9) / (4 * (W + 10)))
    assert _row(c, method="lgbm", has_fail="1").skip_rate_mean == 0.0
    assert "b0" not in set(c.method)


def test_conversion_selection_fraction_from_cls_report():
    cls = pd.DataFrame([{"seed": s, "method": "lgbm", "op": "t95", "tp": 3, "fp": 5, "support_0": 30, "support_1": 10,
                         "recall_1": 0.3} for s in (0, 1)])
    r = _row(conversion(PER, cls, E, F, W), method="lgbm", has_fail="all")
    assert r.sel_frac_mean == pytest.approx(8 / 40) and r.sel_frac_e2e_mean == pytest.approx(8 / (40 * 2 / 4))
    assert r.recall_test_mean == pytest.approx(0.3)


def test_fixed_cost_f1_is_main_savings_and_f0_by_hand():
    t = fixed_cost(PER, E, F, fs=(0.0, 1.0)).set_index(["method", "op", "f"])
    assert t.loc[("lgbm", "t95", 1.0), "savings_mean"] == pytest.approx(211 / 600)
    assert t.loc[("lgbm", "t95", 0.0), "savings_mean"] == pytest.approx(1 - 209 / 240)
    assert ("b0", "natural", 1.0) not in t.index


def test_oracle_rows_cost_and_dur():
    tests = [COLLECTOR, "test_logs::test_service_has_logs[a]", "test_logs::test_service_has_logs[b]"]
    _, ta, tb = tests
    muts = [{"mutant_id": "m1", "type": "M2", "service": "a", "has_fail": 1, "record": {"in_scope": True}},
            {"mutant_id": "m2", "type": "M1", "service": "b", "has_fail": 0, "record": {"in_scope": False}}]
    lab = [{"mutant_id": m, "test_id": t, "label": int(m == "m1" and t != COLLECTOR)} for m in ("m1", "m2") for t in tests]
    d = Data(muts, tests, lab, {ta: 2.0, tb: 3.0}, W=20.0, t_start=70.0, build_nocache={"a": 40.0, "b": 60.0})
    r = oracle_rows(d).set_index(["method", "mutant_id"])
    assert r.loc[("O-test", "m1"), "cost"] == 40 + 90 + 5 and r.loc[("O-mut", "m1"), "cost"] == 40 + 90 + 2
    assert r.loc[("O-test", "m2"), "cost"] == 60.0 and r.loc[("O-test", "m2"), "e2e"] == 0
    assert r.loc[("O-mut", "m1"), "cost_b0prime"] == 40 + 95 and r.loc[("O-mut", "m1"), "caught"] == 1


def test_jackknife_se_uses_s_minus_1_over_s():
    """S = 3 để phân biệt SE jackknife với độ lệch chuẩn quần thể."""
    det = pd.DataFrame([{"pair": "p", "service": s, "d_savings": v, "d_recall": 0.0, "theta_savings": 0.25,
                         "theta_recall": 0.0, "in_ci_savings": True, "in_ci_recall": True, "ci_lo": 0.0, "ci_hi": 1.0,
                         "d_seconds": 1.0} for s, v in zip("abc", (0.1, 0.2, 0.4))])
    th = [0.1, 0.2, 0.4]
    m = sum(th) / 3
    assert jack_summary(det).iloc[0].se_jack_savings == pytest.approx((2 / 3 * sum((x - m) ** 2 for x in th)) ** 0.5)


def test_be_sign_cases():
    assert _be(-1.0, 0.5, 0.2, 0.1, 0.3)[0] == "bị trội"
    assert _be(1.0, 0.5, -0.2, -0.3, -0.1)[0] == "trội"
    assert _be(0.0, 0.5, 0.0, -0.1, 0.1)[0] == "ngang"
    case, be, lo, hi = _be(-1.0, 0.5, -0.2, -0.4, 0.1)
    assert case == "đánh đổi ngược" and be == pytest.approx(10.0)
    assert lo == pytest.approx(-1.0 / (0.5 * -0.4)) and math.isinf(hi)


def test_group_handles_bool_has_fail():
    x = pd.DataFrame({"type": ["M0", "M1", "M2"], "has_fail": [False, False, True]})
    assert list(_group(x)) == ["M0", "im_lang", "co_fail"]


def test_saved_rejects_duplicated_b0_rows():
    b0 = PER[PER.method == "b0"]
    with pytest.raises(Exception):
        saved(PER[PER.method == "lgbm"], pd.concat([b0, b0]), E)

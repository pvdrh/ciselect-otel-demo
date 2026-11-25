"""#61 (plan 2026-10-08-bootstrap-theo-o-ho-5-cap.md bản 3): test trên dữ liệu tính tay cho cell5."""
import math
import random
from pathlib import Path

import pandas as pd
import pytest

from cell5 import diag, label, near_dup_key, near_r, near_s, point, s_label, se_p
from desc57 import boot_cells

MAIN = Path(__file__).resolve().parents[3] / "results/ts02_summary/20261005-195113_seed0-4/main_tests.csv"


def test_near_dup_key_uses_third_part_of_site_id():
    rows = [{"type": "M1", "service": "cart", "site_id": "M1:cart:CartItem.quantity"},  # M1 trên proto: field
            {"type": "M2", "service": "cart", "site_id": "M2:cart:src/cart/a.cs:12"},
            {"type": "M3", "service": "cart", "site_id": "M3:cart:FLAGD_HOST:port"}]
    assert [near_dup_key(r) for r in rows] == [("M1", "cart", "CartItem.quantity"), ("M2", "cart", "src/cart/a.cs"),
                                               ("M3", "cart", "FLAGD_HOST")]


@pytest.mark.parametrize("p_holm, lo, hi, d, want", [
    (0.001, 0.01, 0.05, 0.03, "S+"), (0.001, -0.05, -0.01, -0.03, "S-"),
    (0.001, 0.0, 0.05, 0.03, "S0"),   # khoảng đóng: lo = 0 là chứa 0
    (0.05, 0.01, 0.05, 0.03, "S0"),   # p sau Holm phải < 0,05
    (0.001, -0.01, 0.05, 0.02, "S0")])
def test_s_label(p_holm, lo, hi, d, want):
    assert s_label(p_holm, lo, hi, d) == want


def test_s_label_direction_must_match_point_estimate():
    with pytest.raises(AssertionError):
        s_label(0.001, 0.01, 0.05, -0.02)


def test_label_covers_nine_combinations():
    S = {"S+": (0.01, 0.05, 0.03), "S-": (-0.05, -0.01, -0.03), "S0": (0.0, 0.05, 0.03)}
    R = {"R+": (0.01, 0.02), "R-": (-0.02, -0.01), "R0": (-0.02, 0.0)}
    assert {label(0.001, *S[s], *R[r]) for s in S for r in R} == {f"{s} {r}" for s in S for r in R}


@pytest.mark.skipif(not MAIN.exists(), reason="cần ts02_summary trong results/ (repo, hoặc gói sau recompute)")
def test_labels_on_main_tests():
    mt = pd.read_csv(MAIN, float_precision="round_trip")
    labs = dict(zip(mt.pair, (label(r.p_holm, r.ci_lo, r.ci_hi, r.d_savings, r.r_ci_lo, r.r_ci_hi) for r in mt.itertuples())))
    assert labs.pop("lgbm[t95] vs lr[t95]") == "S+ R-" and list(labs.values()) == ["S+ R0"] * 4


def test_p_below_005_implies_ci_excludes_zero():
    # E-5: đối ngẫu của p và chỉ số phân vị (n = 10 000 như run thật)
    rng, hits = random.Random(3), 0
    for shift in (-3, -1.5, -0.5, 0, 0.5, 1.5, 3):
        ca = [rng.uniform(0, 10) for _ in range(12)]
        cb = [x + shift + rng.uniform(-4, 4) for x in ca]
        (lo, hi), _, _, p, _, _ = boot_cells(ca, cb, [20.0] * 12, [0] * 12, [0] * 12, [0] * 12, list(range(12)))
        if p < 0.05:
            hits += 1
            assert not lo <= 0 <= hi
    assert hits >= 2


def test_se_p():
    assert se_p(0.0, 10_000) == 0 and math.isclose(se_p(0.0245, 10_000), 2 * math.sqrt(0.0245 * 0.9755 / 10_000))


def test_near_s_flags_only_when_holm_side_flips():
    ps = [0.0, 0.0, 0.0, 0.0, 0.049]  # Holm của cặp cuối = 0,049
    assert near_s(ps, 4, se_p(0.0245, 10_000))  # 0,049 + 2·SE > 0,05
    assert not near_s(ps, 0, se_p(0.0, 10_000))
    assert not near_s(ps[:4] + [0.01], 4, se_p(0.005, 10_000))


def test_near_s_flags_upward_side_too():
    # review 1b: p = 0,052 (khoảng vừa chứa 0) cách α dưới 2·SE cũng phải có cờ
    assert near_s([0.0, 0.0, 0.0, 0.0, 0.052], 4, se_p(0.026, 10_000))


def test_near_s_uses_holm_and_two_se():
    # hệ số Holm 2: 0,023·2 = 0,046 < α, (0,023 + 2·0,0012)·2 > α; so p thô, hoặc ±1·SE, thì không có cờ
    assert near_s([0.0, 0.0, 0.0, 0.023, 0.3], 3, 0.0012)


def test_near_r_on_probability_scale():
    thr = 2 * math.sqrt(0.025 * 0.975 / 10_000)
    assert near_r(0.025 + thr / 2, 10_000) and near_r(0.025 - thr / 2, 10_000)
    assert not near_r(0.025 + 2 * thr, 10_000) and not near_r(0.0, 10_000)


def test_diag_counts_and_top5_share():
    keys = ["A", "A", "B", "C", "D", "E", "F"]  # |Σ(b − a)| theo cụm: 2, 2, 3, 4, 5, 6
    got = diag([0] * 7, [1, 1, 2, 3, 4, 5, 6], [1, 0, 0, 0, 0, 0, 1], keys)
    assert got == {"n_clusters": 6, "n_singleton": 5, "n_fail_clusters": 2, "top5_share": pytest.approx(20 / 22)}


def test_point_from_cluster_sums():
    got = point([1.0, 2.0, 3.0], [2.0, 2.0, 5.0], [4.0, 4.0, 8.0], [1, 0, 0.6], [0, 0, 1], [1, 0, 1], ["A", "B", "A"])
    assert got == pytest.approx({"savings_a": 1 - 6 / 16, "savings_b": 1 - 9 / 16, "recall_a": 0.8, "recall_b": 0.5,
                                 "d_savings": 3 / 16, "d_recall": 0.3})

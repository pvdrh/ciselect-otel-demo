"""S11 phần P (#88): mốc nhiễu nhị thức của recall theo fold, chỉ số quan sát, τ theo cấu hình."""
import json

import pandas as pd
import pytest

from nguong import noise_band, observed, tau_by_config


def test_noise_band_degenerate_targets():
    b = noise_band([28, 30], [1.0, 1.0], n=500)
    assert (b.mad_lo, b.mad_hi, b.min_lo, b.min_hi, b.below_lo, b.below_hi) == (0, 0, 1, 1, 0, 0)


def test_noise_band_single_fold_half():
    """1 mutant có fail, mục tiêu 0,5: recall là 0 hoặc 1, nên MAD luôn 0,5; tỉ lệ fold dưới mục tiêu là 0 hoặc 1."""
    b = noise_band([1], [0.5], n=2000)
    assert (b.mad_lo, b.mad_hi) == (0.5, 0.5)
    assert (b.min_lo, b.min_hi, b.below_lo, b.below_hi) == (0, 1, 0, 1)


def test_noise_band_deterministic_and_seeded():
    assert noise_band([29, 30, 28], [0.95, 0.95, 0.95]) == noise_band([29, 30, 28], [0.95, 0.95, 0.95])


def test_observed_by_hand():
    per = pd.DataFrame({"seed": [0] * 5, "fold": [0, 0, 0, 1, 1], "has_fail": [1, 1, 0, 1, 1],
                        "caught": [1, 0, 0, 1, 1]})
    o = observed(per, {(0, 0): 0.9, (0, 1): 0.9})
    assert o["n_fail"] == [2, 2]
    assert (o["recall_min"], o["recall_max"]) == (0.5, 1.0)
    assert o["mad"] == pytest.approx((0.4 + 0.1) / 2)
    assert o["below"] == 1


def test_tau_by_config_splits_class_weight():
    cfg = lambda cw: json.dumps({"num_leaves": 15, "class_weight": cw})  # noqa: E731
    tab = pd.DataFrame({"method": ["lgbm"] * 3 + ["b3"], "op": ["t95"] * 4, "seed": 0, "fold": [0, 1, 2, 0],
                        "config": [cfg(None), cfg("balanced"), cfg(None), "1.0"], "tau": [0.01, 0.5, 0.1, 0.3],
                        "recall_mutant": [0.9, 1.0, 0.8, 0.95]})
    t = tau_by_config(tab).set_index(["by", "method", "op", "key"])
    r = t.loc[("class_weight", "lgbm", "t95", "None")]
    assert (r.n, r.tau_min, r.tau_max, r.recall_min, r.recall_max) == (2, 0.01, 0.1, 0.8, 0.9)
    assert t.loc[("class_weight", "lgbm", "t95", "balanced")].n == 1
    assert t.loc[("class_weight", "b3", "t95", "—")].tau_max == 0.3
    assert t.loc[("config", "lgbm", "t95", cfg(None))].n == 2

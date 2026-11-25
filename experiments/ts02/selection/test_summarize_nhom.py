"""G2a (#88): các hàm của bước tóm tắt."""
import pandas as pd
import pytest

from summarize_nhom import binom_band, changed, contrast, same_baselines, size_strata


def test_binom_band_by_hand():
    assert binom_band(1, 0.5) == (0.0, 1.0)
    assert binom_band(10, 1.0) == (1.0, 1.0)
    lo, hi = binom_band(29, 0.95)
    assert lo < 0.95 <= hi and lo == pytest.approx(25 / 29)


def test_label_changes_only_when_both_levels_change():
    main = {"mutant": "S+ R0", "near_dup": "S+ R0"}
    assert changed({"mutant": "S0 R0", "near_dup": "S0 R0"}, main)
    assert not changed({"mutant": "S0 R0", "near_dup": "S+ R0"}, main)
    assert not changed(main, main)


def test_size_strata_bins_and_big_three():
    g = {f"a{i}": "A" for i in range(16)} | {f"b{i}": "B" for i in range(33)} | {f"c{i}": "C" for i in range(34)}
    g |= {"s": "S", "p1": "P", "p2": "P", "q1": "Q", "q2": "Q", "q3": "Q"}
    st = size_strata(g)
    assert st["a0"] == ("ba nhóm lớn", "≥ 16") and st["s"] == ("phần còn lại", "1") and st["q3"] == ("phần còn lại", "2–9")
    with pytest.raises(AssertionError):
        size_strata(g | {f"x{i}": "X" for i in range(12)})  # cỡ 10–15 nằm ngoài các khoảng đã chốt


def _per(cost=(10.0, 20.0), caught=(1, 0)):
    return pd.DataFrame({"seed": [0, 0], "fold": [0, 1], "method": ["lgbm", "lgbm"], "op": ["t95", "t95"],
                         "mutant_id": ["m1", "m2"], "has_fail": [1, 1], "caught": list(caught), "e2e": [1, 1],
                         "cost": list(cost), "cost_b0prime": [30.0, 30.0]})


def test_same_baselines_ignores_fold_and_float_rounding():
    a = _per().assign(method="b1", op="natural")
    b = a.assign(fold=[3, 2], cost=[10.000000000001, 20.0])
    assert same_baselines(a, b)
    assert not same_baselines(a, b.assign(caught=[0, 0]))


def test_contrast_identical_splits_is_zero():
    c = contrast(_per(), _per(), {"m1": "g1", "m2": "g2"}, n=200).iloc[0]
    assert (c.d_savings, c.ci_lo, c.ci_hi, c.d_recall, c.r_ci_lo, c.r_ci_hi) == (0, 0, 0, 0, 0, 0)
    d = contrast(_per(cost=(5.0, 20.0)), _per(), {"m1": "g1", "m2": "g2"}, n=200).iloc[0]
    assert d.d_savings == pytest.approx(5 / 60)  # (Σy − Σx)/Σbase

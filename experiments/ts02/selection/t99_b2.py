"""R14 của phản biện lượt 2 (#112): ba cặp ở t99 so với B2 trên ba phép chia của #88 (hậu nghiệm, mô tả, ngoài họ 5 cặp).

Hiệu = phương pháp ở t99 − B2. Mức mutant như summarize.main_tests (#49); mức cụm gần trùng như
summarize_nhom.pair_tests (#61, #88). Holm riêng trên 3 cặp của mỗi (phép chia, mức); nhãn theo cell5.label.
Trước khi ghi: cặp B4-LGBM[match_b2] − B2 của mỗi phép chia, tính bằng cùng hàm, phải trùng main_tests.csv của #88.

Chạy: python t99_b2.py RESULTS_DIR RUN_DIR OUT_DIR
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cell5 import _cols, label, point
from desc57 import boot_cells
from run_nhom import near_dup_groups, provenance
from stats import holm
from summarize import _avg, _load, main_tests
from summarize_nhom import N, SEED, SPLITS, _close

PAIRS = [(m, "t99", "b2", "natural") for m in ("b3", "lr", "lgbm")]
CHECK = [("lgbm", "match_b2", "b2", "natural")]
NHOM = "ts02_summary_nhom/20261015-175342_seed0-4/main_tests.csv"
COLS = ["d_savings", "ci_lo", "ci_hi", "p", "d_recall", "r_ci_lo", "r_ci_hi"]


def pair_rows(per, keys, pairs):
    """Mỗi cặp hai dòng: mức mutant (main_tests với left là phương pháp của cặp) và mức cụm gần trùng (boot_cells)."""
    avg, rows = _avg(per), []
    for m, op, other, oop in pairs:
        mt = main_tests(per, [(m, op, other, oop)], n=N, left=m).iloc[0]
        rows.append({"pair": mt.pair, "level": "mutant", "savings_a": mt[f"savings_{m}"], "savings_b": mt.savings_other,
                     "recall_a": mt[f"recall_{m}"], "recall_b": mt.recall_other, **mt[COLS].to_dict()})
        a = avg.loc[(m, op)]
        cols, k = _cols(a, avg.loc[(other, oop)].reindex(a.index)), [keys[i] for i in a.index]
        (lo, hi), (rlo, rhi), *_, p, _, _ = boot_cells(*cols, k, n=N, seed=SEED)
        rows.append({"pair": mt.pair, "level": "near_dup", **point(*cols[:5], cols[5], k), "ci_lo": lo, "ci_hi": hi,
                     "p": p, "r_ci_lo": rlo, "r_ci_hi": rhi})
    t = pd.DataFrame(rows)
    t["p_holm"] = t.groupby("level").p.transform(lambda s: holm(list(s)))
    t["label"] = [label(r.p_holm, r.ci_lo, r.ci_hi, r.d_savings, r.r_ci_lo, r.r_ci_hi) for r in t.itertuples()]
    return t


def main(results, run_dir, out):
    assert sys.version_info < (3, 12)  # như cell5.py: từ 3.12 sum() cộng float theo cách khác
    res, out = Path(results), Path(out)
    keys, ref = near_dup_groups(run_dir), pd.read_csv(res / NHOM, float_precision="round_trip")
    tabs, src = [], {}
    for sp, d in SPLITS.items():
        per, src[sp] = _load(str(res / d / "*_seed[0-4]"), "per_mutant.csv")
        assert len(src[sp]) == 5 and sorted(per.seed.unique()) == list(range(5)), sp
        chk = pair_rows(per, keys, CHECK).set_index("level")
        want = ref[(ref.split == sp) & (ref.pair == chk.pair.iloc[0])].set_index("level").loc[chk.index]
        assert _close(chk[COLS], want[COLS]), sp
        tabs.append(pair_rows(per, keys, PAIRS).assign(split=sp))
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(tabs, ignore_index=True).to_csv(out / "table.csv", index=False)
    (out / "params.json").write_text(json.dumps({
        "decision": "#112", "review": "docs/2026-10-23-phan-bien-toan-van-2/tong-hop.md, roadmap R14",
        "eval": src, "near_dup": "cell5.near_dup_key, 'loại|service|site'", "pairs": PAIRS,
        "bootstrap": {"n": N, "seed": SEED, "mutant": "summarize.main_tests", "near_dup": "desc57.boot_cells"},
        "holm": "riêng trên 3 cặp của mỗi (phép chia, mức); ngoài họ 5 cặp đã tiền đăng ký",
        "checks": [f"cặp {CHECK[0]} tính lại trùng {NHOM} ở cả ba phép chia, hai mức (cột {', '.join(COLS)})"],
        **provenance(out)}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\npandas {pd.__version__}\nnumpy {np.__version__}\n"
                                 f"{platform.platform()}\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])

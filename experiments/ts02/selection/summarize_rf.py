"""#51 (thăm dò, thêm sau khi đã thấy D2) bước 8: tổng hợp 5 seed RF, cùng cách gộp như summarize.py (D2-9).

Cặp kiểm định: 7 cặp thăm dò, Holm riêng trong 7 cặp, ngoài họ #49. Dòng baseline của run RF chỉ dùng để kiểm trùng
(run_extra.py), không ghép vào per, để không đếm hai lần.
Chạy: python summarize_rf.py RESULTS_DIR OUT_DIR
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

from summarize import _load, by_seed, extras, main_tests, pooled, recall_cis

PAIRS = [("b1", "match_b1", "b1", "natural"), ("b1h", "match_b1h", "b1h", "natural"),
         ("b2", "match_b2", "b2", "natural"), ("b3", "t95", "b3", "t95"), ("lr", "t95", "lr", "t95"),
         ("lgbm", "t95", "lgbm", "t95"), ("lgbm", "match_b1", "lgbm", "match_b1")]
LABEL = "#51 phân tích thăm dò, thêm sau khi đã thấy kết quả D2"


def _cls(cls):
    num = [c for c in cls.columns if c not in ("method", "op", "seed")]
    c = cls.groupby(["method", "op"], sort=False)[num].agg(["mean", "std"])
    c.columns = [f"{a}_{b}" for a, b in c.columns]
    return c.reset_index()


def main(results, out):
    r, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for k, name, f in (("per", "ts02_eval_rf", "per_mutant.csv"), ("tab", "ts02_eval_rf", "table.csv"),
                       ("lper", "ts02_eval_rf_loso", "per_mutant.csv"), ("d2per", "ts02_eval", "per_mutant.csv"),
                       ("cls", "ts02_cls_rf", "cls_report.csv"), ("lcls", "ts02_cls_rf_loso", "cls_report.csv"),
                       ("d2lcls", "ts02_cls_loso", "cls_report.csv")):
        df, src = _load(str(r / name / "*_seed[0-4]"), f)
        assert len(src) == 5 and sorted(df.seed.unique()) == list(range(5)), (name, src)
        runs[k] = (df[df.method == "rf"] if k in ("per", "tab", "lper") else df, src)
    per, tab, lper = runs["per"][0], runs["tab"][0], runs["lper"][0]
    pooled(per).merge(extras(tab, per), on=["method", "op"]).merge(recall_cis(per), on=["method", "op"]) \
        .to_csv(out / "table.csv", index=False)
    pooled(lper).to_csv(out / "loso.csv", index=False)
    pooled(per[per.in_scope]).to_csv(out / "subset_T.csv", index=False)  # #40
    both = pd.concat([runs["d2per"][0], per], ignore_index=True)
    assert not both.duplicated(["seed", "method", "op", "mutant_id"]).any()
    main_tests(both, PAIRS, left="rf").assign(family="7 cặp thăm dò #51, Holm riêng, ngoài #49") \
        .to_csv(out / "pairs.csv", index=False)
    by_seed(both, PAIRS, left="rf").to_csv(out / "pairs_by_seed.csv", index=False)
    pd.concat([_cls(runs["cls"][0]).assign(split="main"), _cls(runs["lcls"][0]).assign(split="loso")]) \
        .to_csv(out / "cls_report.csv", index=False)
    _cls(runs["d2lcls"][0]).to_csv(out / "cls_report_loso_d2.csv", index=False)  # R8: gộp run D2 đã có
    (out / "params.json").write_text(json.dumps({
        "plan": "plans/2026-10-07-d2-rf-oracle.md (bước 8)", "label": LABEL,
        "sources": {k: v[1] for k, v in runs.items()}, "bootstrap_n": 10_000, "bootstrap_seed": 0, "pairs": PAIRS,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:3])

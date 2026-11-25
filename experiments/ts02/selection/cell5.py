"""#61: độ nhạy của họ 5 cặp (#49) theo đơn vị lấy mẫu lại, theo `plans/2026-10-08-bootstrap-theo-o-ho-5-cap.md` bản 3.

Phân tích độ nhạy hậu nghiệm, không thay kết quả chính. Cùng phép bootstrap của C5 (desc57.boot_cells), chỉ đổi khoá cụm:
dòng đối chứng (mỗi mutant một cụm), cụm gần trùng, cụm (S, loại file), service; p và Holm riêng ở từng mức.
Mọi assert (E-1 đến E-4) chạy trước khi ghi file.

Chạy (PYTHONHASHSEED=0, Python < 3.12): python cell5.py RESULTS_DIR RUN_DIR BUILD_NOCACHE_DIR DEMO OUT_DIR
"""
import csv
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from data import load
from desc57 import N, SEED, _git, boot_cells, cell_key
from stats import bootstrap_diff, holm
from summarize import PAIRS, _avg, _load

PLAN = "plans/2026-10-08-bootstrap-theo-o-ho-5-cap.md"
STEP0 = "0bb759d"  # commit bước 0 của plan (#61, tiền đăng ký, chương 3)
LABEL = ("#60 mục 3 (plan duyệt ở #61), phân tích độ nhạy hậu nghiệm, thêm sau khi đã thấy kết quả D2, O5 và #53; không "
         "thay kết quả chính; phụ thuộc cơ cấu mẫu mutant (M0 76/353, có fail 144/353)")
PREDICTIONS = [
    "Ở hiệu tiết kiệm của cả 5 cặp, độ rộng khoảng không giảm theo thứ tự mutant ≤ gần trùng ≤ (S, loại file) ≤ service. "
    "Thông tin thấp.",
    "Ở cặp B4-LGBM–B4-LR, nhãn R của dòng đối chứng là R−, còn ở mức (S, loại file) và mức service là R0. Thông tin thấp, "
    "vì cận trên theo mutant là −1/720.",
    "Ở mức service, khoảng tiết kiệm của cặp B4-LGBM–B1h chứa 0. Dự đoán này suy từ #53 (SE jackknife 0,0417; xấp xỉ "
    "chuẩn cho [−0,010; 0,153]), nên gần như một lần tung đồng xu."]
SUMMARY = "ts02_summary/20261005-195113_seed0-4"
O5 = "ts02_desc57/20261008-135736_seed0-4/o5_pairs.csv"
LEVELS = ("control", "near_dup", "cell", "service")
ALPHA = 0.05


def near_dup_key(m):
    """(loại, service, phần thứ ba của site_id): file; field với M1 trên proto; biến môi trường với M3 (U1a)."""
    return m["type"], m["service"], m["site_id"].split(":")[2]


def s_label(p_holm, lo, hi, d):
    if p_holm >= ALPHA or lo <= 0 <= hi:
        return "S0"
    s = "+" if lo > 0 else "-"
    assert (d > 0) == (s == "+"), (lo, hi, d)
    return "S" + s


def r_label(lo, hi):
    return "R0" if lo <= 0 <= hi else "R+" if lo > 0 else "R-"


def label(p_holm, lo, hi, d, rlo, rhi):
    return f"{s_label(p_holm, lo, hi, d)} {r_label(rlo, rhi)}"


def se_p(q, n):
    return 2 * math.sqrt(q * (1 - q) / n)


def near_s(ps, i, se):
    """Cờ "sát biên": nhãn S của cặp i đổi khi thay p_i bằng p_i ± 2·SE, các p khác giữ, Holm tính lại.
    Nhãn là S± đúng khi p sau Holm < α (p < α thì khoảng không chứa 0, E-5), và khoảng đi cùng p, nên xét p sau Holm."""
    side = holm(ps)[i] < ALPHA
    return any((holm(ps[:i] + [min(1.0, max(0.0, x))] + ps[i + 1:])[i] < ALPHA) != side
               for x in (ps[i] - 2 * se, ps[i] + 2 * se))


def near_r(q_r, n_r):
    return abs(q_r - 0.025) < 2 * math.sqrt(0.025 * 0.975 / n_r)


def _sums(keys, *cols):
    agg = defaultdict(lambda: [0.0] * len(cols))
    for i, k in enumerate(keys):
        for j, c in enumerate(cols):
            agg[k][j] += c[i]
    return list(agg.values())


def diag(ca, cb, f, keys):
    v = _sums(keys, [1] * len(keys), f, [b - a for a, b in zip(ca, cb)])
    s = sorted((abs(x[2]) for x in v), reverse=True)
    return {"n_clusters": len(v), "n_singleton": sum(x[0] == 1 for x in v), "n_fail_clusters": sum(x[1] > 0 for x in v),
            "top5_share": sum(s[:5]) / sum(s) if sum(s) else math.nan}


def point(ca, cb, base, ka, kb, f, keys):
    """E-1: ước lượng điểm cộng từ tổng theo cụm."""
    t = [sum(x) for x in zip(*_sums(keys, ca, cb, base, [k * g for k, g in zip(ka, f)], [k * g for k, g in zip(kb, f)], f))]
    return {"savings_a": 1 - t[0] / t[2], "savings_b": 1 - t[1] / t[2], "recall_a": t[3] / t[5], "recall_b": t[4] / t[5],
            "d_savings": (t[1] - t[0]) / t[2], "d_recall": (t[3] - t[4]) / t[5]}


def _cols(a, b):
    return (list(a.cost), list(b.cost), list(a.base), list(a.caught), list(b.caught), list((a.has_fail == 1).astype(int)))


def main(results, run_dir, build_dir, demo, out):
    assert sys.version_info < (3, 12)  # E-4: từ 3.12 sum() cộng float theo cách khác
    res, out = Path(results), Path(out)
    per, src = _load(str(res / "ts02_eval/*_seed[0-4]"), "per_mutant.csv")
    perk, srck = _load(str(res / "ts02_eval_b3k/*_seed[0-4]"), "per_mutant.csv")
    assert len(src) == len(srck) == 5 and sorted(per.seed.unique()) == sorted(perk.seed.unique()) == list(range(5))
    mt = pd.read_csv(res / SUMMARY / "main_tests.csv", float_precision="round_trip").set_index("pair")
    o5 = pd.read_csv(res / O5, float_precision="round_trip").set_index("pair")
    d = load(run_dir, build_dir, demo)
    with open(Path(run_dir) / "mutants" / "mutants.csv", encoding="utf-8") as fh:
        nd = {r["mutant_id"]: near_dup_key(r) for r in csv.DictReader(fh)}
    key_of = {"near_dup": nd, "cell": {m["mutant_id"]: cell_key(m) for m in d.mutants},
              "service": {m["mutant_id"]: m["service"] for m in d.mutants}}
    avg = _avg(per)

    # E-2: boot_cells sau khi sửa cho lại đúng khoảng theo (S, loại file) của O5
    ak = _avg(pd.concat([per, perk[perk.method == "b3k"]], ignore_index=True))
    a, b = ak.loc[("lgbm", "t95")], ak.loc[("b3k", "t95")].reindex(ak.loc[("lgbm", "t95")].index)
    (lo, hi), (rlo, rhi), *_ = boot_cells(*_cols(a, b), [key_of["cell"][i] for i in a.index])
    r = o5.loc["lgbm[t95] vs b3k[t95]"]
    assert (lo, hi, rlo, rhi) == (r.cell_ci_lo, r.cell_ci_hi, r.cell_r_ci_lo, r.cell_r_ci_hi)

    rows = []
    for _, op, other, oop in PAIRS:
        name = f"lgbm[{op}] vs {other}[{oop}]"
        a = avg.loc[("lgbm", op)]
        b = avg.loc[(other, oop)].reindex(a.index)
        ids, cols, m = list(a.index), _cols(a, b), mt.loc[name]
        assert len(ids) == 353 and sum(cols[5]) == 144
        keys = {"control": ids, **{lv: [key_of[lv][i] for i in ids] for lv in LEVELS[1:]}}
        ref = {"savings_a": m.savings_lgbm, "savings_b": m.savings_other, "recall_a": m.recall_lgbm,
               "recall_b": m.recall_other, "d_savings": m.d_savings, "d_recall": m.d_recall}
        rows.append({"pair": name, "level": "mutant", **ref, **diag(cols[0], cols[1], cols[5], ids), "ci_lo": m.ci_lo,
                     "ci_hi": m.ci_hi, "p": m.p, "p_holm": m.p_holm, "r_ci_lo": m.r_ci_lo, "r_ci_hi": m.r_ci_hi})
        for lv in LEVELS:
            pt = point(*cols[:5], cols[5], keys[lv])
            assert all(math.isclose(pt[k], ref[k], rel_tol=0, abs_tol=1e-12) for k in ref), (name, lv)  # E-1
            (lo, hi), (rlo, rhi), n_r, p, q, q_r = boot_cells(*cols, keys[lv])
            if lv == "control":  # E-4
                assert (lo, hi, p) == bootstrap_diff(*cols[:3], n=N, seed=SEED) == (m.ci_lo, m.ci_hi, m.p), name
            rows.append({"pair": name, "level": lv, **pt, **diag(cols[0], cols[1], cols[5], keys[lv]), "ci_lo": lo,
                         "ci_hi": hi, "p": p, "q": q, "se_p": se_p(q, N), "r_ci_lo": rlo, "r_ci_hi": rhi, "n_r": n_r,
                         "q_r": q_r})

    t = pd.DataFrame(rows)
    for lv in LEVELS:
        x = t.level == lv
        t.loc[x, "p_holm"] = holm(list(t.p[x]))
        ps = list(t.p[x])
        t.loc[x, "near_s"] = [near_s(ps, i, r.se_p) for i, r in enumerate(t[x].itertuples())]
        t.loc[x, "near_r"] = [near_r(r.q_r, r.n_r) for r in t[x].itertuples()]
    g = t.set_index(["pair", "level"])
    t["width_ratio_s"] = [(r.ci_hi - r.ci_lo) / (g.loc[(r.pair, "mutant"), "ci_hi"] - g.loc[(r.pair, "mutant"), "ci_lo"])
                          for r in t.itertuples()]
    t["width_ratio_r"] = [(r.r_ci_hi - r.r_ci_lo) / (g.loc[(r.pair, "control"), "r_ci_hi"] - g.loc[(r.pair, "control"), "r_ci_lo"])
                          for r in t.itertuples()]
    t["s_label"] = [s_label(r.p_holm, r.ci_lo, r.ci_hi, r.d_savings) for r in t.itertuples()]
    t["r_label"] = [r_label(r.r_ci_lo, r.r_ci_hi) for r in t.itertuples()]
    t["label"] = t.s_label + " " + t.r_label

    # E-3: số cụm, cụm có fail, theo loại ở mức gần trùng; ba mức lồng nhau
    ids = list(avg.loc[("lgbm", "t95")].index)
    fail = set(avg.loc[("lgbm", "t95")].query("has_fail == 1").index)
    levels = {}
    for lv in LEVELS[1:]:
        c = Counter(key_of[lv][i] for i in ids)
        levels[lv] = {"n_clusters": len(c), "n_fail_clusters": len({key_of[lv][i] for i in fail}),
                      "size_min": min(c.values()), "size_max": max(c.values()),
                      "n_singleton": sum(v == 1 for v in c.values())}
    by_type = dict(sorted(Counter(k[0] for k in {nd[i] for i in ids}).items()))
    assert [(levels[lv]["n_clusters"], levels[lv]["n_fail_clusters"]) for lv in LEVELS[1:]] == [(142, 64), (33, 29), (14, 14)]
    assert by_type == {"M0": 19, "M1": 60, "M2": 15, "M3": 48}
    for lo_lv, hi_lv in (("near_dup", "cell"), ("cell", "service")):
        up = defaultdict(set)
        for i in ids:
            up[key_of[lo_lv][i]].add(key_of[hi_lv][i])
        assert all(len(v) == 1 for v in up.values()), (lo_lv, hi_lv)

    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "cell_tests.csv", index=False)
    (out / "params.json").write_text(json.dumps({
        "plan": PLAN, "decision": "#60 mục 3, #61 (duyệt plan bản 3)", "label": LABEL, "predictions": PREDICTIONS,
        "sources": {"d2": src, "b3k": srck, "summary": SUMMARY, "o5": O5}, "data": str(run_dir),
        "bootstrap": {"n": N, "seed": SEED, "function": "desc57.boot_cells", "alpha": ALPHA}, "levels": levels,
        "near_dup_by_type": by_type, "checks": "E-1 đến E-4 qua (assert trước khi ghi)",
        "input_parse": ("per_mutant.csv đọc qua summarize._load (parser mặc định của pandas, như summarize và desc57 "
                        "đã dùng để ra main_tests.csv, o5_pairs.csv); main_tests.csv, o5_pairs.csv đọc round_trip"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "python": platform.python_version(),
        "git_head": _git("rev-parse", "HEAD"), "step0_commit": _git("rev-parse", STEP0),
        "tree_dirty": bool(_git("status", "--porcelain", "--", "."))}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\npandas {pd.__version__}\n{platform.platform()}\n",
                                 encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:6])

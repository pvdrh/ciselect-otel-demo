"""Quy tắc chọn chế độ mặc định của ciselect từ bảng tổng hợp và bảng kiểm định của phần đánh giá.

Đây là cấu hình sản phẩm, không phải kết luận rằng chiến lược nào tốt hơn.
- Ứng viên: cận trên khoảng tin cậy 95% của recall theo mutant ≥ 0,95. Chiến lược luật lấy recall tự nhiên,
  mô hình lấy recall ở t95. B0′ luôn là ứng viên.
- Bước 1: LGBM là ứng viên và hơn mọi ứng viên khác có ý nghĩa (khoảng tin cậy của hiệu tiết kiệm > 0 và p Holm < 0,05)
  thì chọn lgbm ở r* = max(0,95; recall tự nhiên lớn nhất trong các ứng viên B1, B1h, B2).
- Bước 2: nếu không, chọn ứng viên khác có tiết kiệm cao nhất. Hoà trong 0,5 điểm % thì chọn chiến lược đơn giản hơn,
  theo thứ tự B1 < B1h < B2 < B3 < B4-LR.
- Bước 3: chỉ còn B0′ thì chọn B0′.
"""
import re

SIMPLE = ["b1", "b1h", "b2", "b3", "lr"]
RULES = ("b1", "b1h", "b2")
OP = {m: ("natural" if m in RULES else "t95") for m in SIMPLE + ["lgbm"]}
TIE = 0.005


def apply_rule41(table, main_tests):
    t = table.set_index(["method", "op"])
    row = lambda m: t.loc[(m, OP[m])]  # noqa: E731
    cand = [m for m in SIMPLE + ["lgbm"] if row(m).r_hi >= 0.95]
    sig = {re.search(r" vs (\w+)\[", r.pair).group(1): bool(r.ci_lo > 0 and r.p_holm < 0.05)
           for r in main_tests.itertuples()}
    used = {"candidates": {m: float(row(m).r_hi) for m in SIMPLE + ["lgbm"]}, "significant_vs": sig}
    others = [m for m in cand if m != "lgbm"]
    if "lgbm" in cand and all(sig[m] for m in others):
        r_star = max([0.95] + [float(row(m).recall_mutant_mean) for m in others if m in RULES])
        return {"step": 1, "mode": "lgbm", "recall_target": r_star, **used}
    if others:
        best = max(float(row(m).savings_mean) for m in others)
        mode = next(m for m in SIMPLE if m in others and best - float(row(m).savings_mean) <= TIE)
        return {"step": 2, "mode": mode, "recall_target": None if mode in RULES else 0.95, **used}
    return {"step": 3, "mode": "b0p", "recall_target": None, **used}

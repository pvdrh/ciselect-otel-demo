"""D2-2: chọn ngưỡng cho B3/B4 (C5). "Ngưỡng nhỏ nhất đạt recall" hiểu là ngưỡng chặt nhất (tập test nhỏ nhất)
vẫn đạt recall theo mutant ≥ mục tiêu."""
import math


def pick_threshold(best, target):
    """best: mutant fail → điểm cao nhất trên các test mang nhãn fail của nó. Mutant được bắt khi điểm đó ≥ ngưỡng."""
    if not best:
        return -math.inf  # không có mutant fail để hiệu chỉnh: chạy hết (bảo thủ)
    k = math.ceil(target * len(best) - 1e-9)
    if k <= 0:
        return math.inf  # review D2-7 #2: không cần bắt mutant nào → không chọn test
    return sorted(best.values(), reverse=True)[k - 1]


def select(scores, tau):
    """scores: test → điểm. Trả (tập test chọn, có chạy e2e). (ii): chạy e2e khi có ≥ 1 test vượt ngưỡng (C6)."""
    chosen = {t for t, s in scores.items() if s >= tau}
    return chosen, bool(chosen)

"""V2 (plan C, C-3): chọn N mutant từ danh sách vị trí ứng viên, theo seed.

- Số mutant mỗi loại = N × tỉ lệ (làm tròn theo phần dư lớn nhất), tỉ lệ lấy từ tiền đăng ký C2.
- Trong mỗi loại: phân tầng theo service, rút xoay vòng qua các service (thứ tự xáo theo seed)
  để số mutant mỗi service lệch nhau ≤ 1; service hết vị trí thì bỏ qua lượt.
"""
import random
from collections import defaultdict


def quotas(n, props):
    raw = {t: n * p for t, p in props.items()}
    q = {t: int(v) for t, v in raw.items()}
    for t in sorted(raw, key=lambda t: raw[t] - q[t], reverse=True)[: n - sum(q.values())]:
        q[t] += 1
    return q


def sample(sites, n, props, seed):
    rng = random.Random(seed)
    out = []
    for t, k in sorted(quotas(n, props).items()):
        pools = defaultdict(list)
        for r in sorted((r for r in sites if r["type"] == t), key=lambda r: r["site_id"]):
            pools[r["service"]].append(r)
        order = sorted(pools)
        rng.shuffle(order)
        for svc in order:
            rng.shuffle(pools[svc])
        got = []
        while len(got) < k and any(pools.values()):
            for svc in order:
                if pools[svc] and len(got) < k:
                    got.append(pools[svc].pop())
        assert len(got) == k, f"{t}: chỉ có {len(got)}/{k} vị trí ứng viên"
        out += got
    return out

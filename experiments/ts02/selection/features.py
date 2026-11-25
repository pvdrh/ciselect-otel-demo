"""D2-5: đặc trưng (mutant, test) theo C6 (số dòng/file theo #42). Chỉ dùng thông tin CI biết trước khi chạy test;
tỉ lệ fail chỉ tính từ fold train."""
import math
import zlib
from collections import defaultdict, deque

import numpy as np

from allowlist import SERVICES
from data import ALL, meta_of, touches

SVCS = sorted(SERVICES)
BIG = 99.0  # khoảng cách ∞ (không nối) mã hoá thành số lớn
LAPLACE_A = 1.0
ENC_FOLDS = 5  # review D2-7 #1: mã hoá tỉ lệ fail out-of-fold cho dòng train
FEATURES = ([f"svc_{s}" for s in SVCS] + ["kind_proto", "kind_config", "kind_code", "n_files", "n_lines",
            "dist_up", "dist_down", "dist_undir", "touches"]
            + [f"tkind_{k}" for k in ["service", "edge", "collector"]]
            + [f"sig_{s}" for s in ["traces", "metrics", "logs", "none"]] + ["fail_rate", "dur"])


def _bfs(nbr, s):
    dist, q = {s: 0}, deque([s])
    while q:
        u = q.popleft()
        for v in nbr[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def distances(edges, s, targets):
    """(ngược chiều gọi, xuôi chiều gọi, vô hướng): số bước nhỏ nhất từ s tới một service trong targets."""
    up, down, und = defaultdict(set), defaultdict(set), defaultdict(set)
    for a, b in edges:  # a gọi b
        up[b].add(a)
        down[a].add(b)
        und[a].add(b)
        und[b].add(a)
    return tuple(min((_bfs(n, s).get(t, math.inf) for t in targets), default=math.inf) for n in (up, down, und))


def features(d, train_ids, row_ids, drop=()):
    """Ma trận đặc trưng cho mọi (mutant ∈ row_ids, test); keys cùng thứ tự hàng. drop: tên cột bỏ đi (#47 Q1)."""
    train = set(train_ids)
    grp = {m["mutant_id"]: m.get("group", m["mutant_id"]) for m in d.mutants}  # G2a (#88): băm theo nhóm khi có
    enc = lambda mid: zlib.crc32(grp[mid].encode()) % ENC_FOLDS  # noqa: E731
    n, fails = defaultdict(int), defaultdict(int)  # khoá (fold mã hoá, ...) và tổng (None, ...)
    svc_of = {m["mutant_id"]: m["service"] for m in d.mutants}
    for m in d.mutants:
        if m["mutant_id"] in train:
            for f in (None, enc(m["mutant_id"])):
                n[f, m["service"]] += 1
    for r in d.labels:
        if r["mutant_id"] in train:
            for f in (None, enc(r["mutant_id"])):
                fails[f, svc_of[r["mutant_id"]], r["test_id"]] += r["label"]

    def rate(mid, s, t):
        """Dòng train: chỉ dùng mutant ở fold mã hoá khác (không tự mã hoá nhãn của mình); dòng dự đoán: cả train."""
        k, nn = fails[None, s, t], n[None, s]
        if mid in train:
            k, nn = k - fails[enc(mid), s, t], nn - n[enc(mid), s]
        return (k + LAPLACE_A) / (nn + 2 * LAPLACE_A)

    meta = {t: meta_of(t) for t in d.tests}
    rows, keys = [], []
    by_id = {m["mutant_id"]: m for m in d.mutants}
    for mid in row_ids:
        m = by_id[mid]
        s, rec = m["service"], m["record"]
        base = [float(s == x) for x in SVCS] + [float(k in rec["kinds"]) for k in ["proto", "config", "code"]] \
            + [rec["n_files"], rec["n_lines"]]
        for t in d.tests:
            mt = meta[t]
            dist = (0, 0, 0) if mt["services"] is ALL else distances(d.edges, s, mt["services"])
            rows.append(base + [min(x, BIG) for x in dist] + [float(touches(mt, s))]
                        + [float(mt["kind"] == k) for k in ["service", "edge", "collector"]]
                        + [float(mt["signal"] == g) for g in ["traces", "metrics", "logs", "none"]]
                        + [rate(mid, s, t), d.dur.get(t, 0.0)])
            keys.append((mid, t))
    X = np.array(rows, dtype=float)
    return np.delete(X, [FEATURES.index(c) for c in drop], axis=1), keys

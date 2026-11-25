"""D2-3: baseline B0, B1, B1h, B2, B3 (tiền đăng ký C1, C3, C6; #32). Mỗi hàm trả Decision (i)–(iii)."""
from collections import defaultdict

from data import meta_of, touches
from evaluate import Decision


def reach(edges, s, k, direction):
    """Service cách s tối đa k bước (k = None: ∞). callers: đi ngược chiều gọi; undirected: cả hai chiều."""
    nbr = defaultdict(set)
    for a, b in edges:
        nbr[b].add(a)
        if direction == "undirected":
            nbr[a].add(b)
    seen, frontier, step = {s}, {s}, 0
    while frontier and (k is None or step < k):
        frontier = {n for f in frontier for n in nbr[f]} - seen
        seen |= frontier
        step += 1
    return seen - {s}


def _touching(d, svcs):
    if d.test_services is not None:  # repo khác OTel Demo: ánh xạ đọc từ catalog.json (plan 2026-11-21)
        return {t for t in d.tests if svcs & set(d.test_services.get(t, ()))}
    return {t for t in d.tests if any(touches(meta_of(t), s) for s in svcs)}


def b0(m, d):
    return Decision(set(d.build_nocache), True, set(d.tests))


def b1(m, d):
    tests = _touching(d, {m["service"]})
    return Decision({m["service"]}, bool(tests), tests)


def b1h(m, d):
    return Decision({m["service"]}, False, set()) if m["record"]["comment_only"] else b1(m, d)


def b2(m, d, edges, k, direction):
    tests = _touching(d, {m["service"]} | reach(edges, m["service"], k, direction))
    return Decision({m["service"]}, bool(tests), tests)


def b3_table(train_ids, labels, service_of, a):
    """Đồng-fail chỉ từ fold train: số mutant đổi S và số lần test t fail khi đổi S."""
    train, n, fails = set(train_ids), defaultdict(set), defaultdict(int)
    for r in labels:
        if r["mutant_id"] in train:
            s = service_of[r["mutant_id"]]
            n[s].add(r["mutant_id"])
            fails[s, r["test_id"]] += r["label"]
    return {"n": {s: len(v) for s, v in n.items()}, "fails": dict(fails)}


def b3_scores(table, s, tests, a):
    n = table["n"].get(s, 0)
    return {t: (table["fails"].get((s, t), 0) + a) / (n + 2 * a) for t in tests}


def b3k_scores(tab_s, tab_k, s, k, tests, a):
    """#57 O5: B3 theo khoá k = (S, loại file); prior là tỉ lệ (Laplace a) của S không gồm ô k, = 0,5 khi rỗng.
    Ô rỗng → điểm B3 của S; S chỉ một loại file → trùng B3."""
    n_k = tab_k["n"].get(k, 0)
    n_o = tab_s["n"].get(s, 0) - n_k
    out = {}
    for t in tests:
        f_k = tab_k["fails"].get((k, t), 0)
        p = (tab_s["fails"].get((s, t), 0) - f_k + a) / (n_o + 2 * a)
        out[t] = (f_k + 2 * a * p) / (n_k + 2 * a)
    return out

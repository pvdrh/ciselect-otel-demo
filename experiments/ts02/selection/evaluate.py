"""D2-2: chi phí và metric (tiền đăng ký C0 sửa bởi #39, C1, C5)."""
from dataclasses import dataclass, field

from data import COLLECTOR, meta_of, touches


@dataclass
class Decision:
    build: set  # (i)
    e2e: bool  # (ii)
    tests: set = field(default_factory=set)  # (iii)


def e2e_cost(tests, d):
    return d.t_start + d.W + sum(d.dur[t] for t in tests if t != COLLECTOR)


def cost(dec, d):
    return sum(d.build_nocache[s] for s in dec.build) + (e2e_cost(dec.tests, d) if dec.e2e else 0.0)


def b0prime(m, d):
    """B0′: build S + chạy toàn bộ e2e (C5)."""
    return Decision({m["service"]}, True, set(d.tests))


def metrics(decisions, d):
    """decisions: mutant_id → Decision, trên các mutant của một fold test."""
    lab = {(r["mutant_id"], r["test_id"]): r["label"] for r in d.labels}
    ms = [m for m in d.mutants if m["mutant_id"] in decisions]
    base = sum(cost(b0prime(m, d), d) for m in ms)
    got = ii = iii = 0.0
    fail = caught = n_lab = n_hit = cross = cross_miss = 0
    for m in ms:
        dec, mid = decisions[m["mutant_id"]], m["mutant_id"]
        got += cost(dec, d)
        if dec.e2e:
            iii += e2e_cost(d.tests, d) - e2e_cost(dec.tests, d)
        else:
            ii += e2e_cost(d.tests, d)
        fails = [t for t in d.tests if lab[mid, t]]
        hit = dec.e2e and any(t in dec.tests for t in fails)
        n_lab += len(fails)
        n_hit += sum(t in dec.tests for t in fails) if dec.e2e else 0
        if fails:
            fail += 1
            caught += hit
            if any(not touches(meta_of(t), m["service"]) for t in fails):
                cross += 1
                cross_miss += not hit
    nan = float("nan")
    return {"n_mutants": len(ms), "n_fail": fail, "savings": 1 - got / base,
            "recall_mutant": caught / fail if fail else nan, "recall_test": n_hit / n_lab if n_lab else nan,
            "cross_missed": cross_miss / cross if cross else nan, "n_cross": cross,
            "savings_ii": ii / base, "savings_iii": iii / base}

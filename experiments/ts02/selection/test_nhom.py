"""G2a (#88): chia theo nhóm gần trùng ở mọi chỗ D2 chia mutant; nhóm giả; tiêu chí cân bằng."""
import copy
import csv
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from sklearn.model_selection import StratifiedKFold

from data import UNITS, folds
from features import FEATURES, features
from run import _oof, run_seed
from run_nhom import balance_ok, fake_groups, near_dup_groups
from test_models import TE, TP
from test_run import TINY, _data

D1 = Path(__file__).resolve().parents[3] / "results" / "ts02_dataset" / "20261001-213236_seed0"
need_d1 = pytest.mark.skipif(not D1.exists(), reason="cần run D1-7")
FR = FEATURES.index("fail_rate")


def _grouped():
    d = _data()
    for i, m in enumerate(d.mutants):  # nhóm cỡ 2–3, cùng loại và service
        m["group"] = f"{m['type']}|{m['service']}|{i // 10}"
    return d


def _flip_mates(d, mid="m10"):
    """Đổi nhãn TP, TE của các mutant cùng nhóm với mid; has_fail giữ nguyên (TP 1 → 0, TE 0 → 1)."""
    d2 = copy.deepcopy(d)
    g = next(m["group"] for m in d2.mutants if m["mutant_id"] == mid)
    mates = {m["mutant_id"] for m in d2.mutants if m["group"] == g and m["mutant_id"] != mid}
    for r in d2.labels:
        if r["mutant_id"] in mates and r["test_id"] in (TP, TE):
            r["label"] = 1 - r["label"]
    fail = defaultdict(int)
    for r in d2.labels:
        fail[r["mutant_id"]] |= r["label"]
    assert mates and all(fail[m["mutant_id"]] == m["has_fail"] for m in d2.mutants)
    return d2, mates


def _rate(d, train, mid, t, key):
    """fail_rate của dòng train (mid, t) tính tay: cùng service, bỏ mutant cùng nhóm băm crc32(key) % 5, Laplace a = 1."""
    enc = lambda x: zlib.crc32(key(x).encode()) % 5  # noqa: E731
    svc = {m["mutant_id"]: m["service"] for m in d.mutants}
    lab = {(r["mutant_id"], r["test_id"]): r["label"] for r in d.labels}
    pool = [x for x in train if svc[x] == svc[mid] and enc(x) != enc(mid)]
    return (sum(lab[x, t] for x in pool) + 1) / (len(pool) + 2)


def test_folds_without_group_is_old_split():
    d = _data()
    ids, st = [m["mutant_id"] for m in d.mutants], [f"{m['type']}|{m['has_fail']}" for m in d.mutants]
    for seed in range(3):
        old = [([ids[i] for i in tr], [ids[i] for i in te])
               for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, st)]
        assert folds(d.mutants, seed) == old


def test_fail_rate_hash_key_is_group_when_present():
    for d, key in [(_data(), lambda x: x), (_grouped(), None)]:
        grp = {m["mutant_id"]: m.get("group", m["mutant_id"]) for m in d.mutants}
        ids = [m["mutant_id"] for m in d.mutants]
        X, keys = features(d, ids, ids)
        for (mid, t), row in zip(keys, X):
            assert row[FR] == pytest.approx(_rate(d, ids, mid, t, key or grp.get), abs=1e-12)


def test_group_folds_keep_groups_and_differ_by_seed():
    d = _grouped()
    g = {m["mutant_id"]: m["group"] for m in d.mutants}
    runs = []
    for seed in range(3):
        sp = folds(d.mutants, seed)
        assert sorted(i for _, te in sp for i in te) == sorted(g)
        assert all(not {g[i] for i in tr} & {g[i] for i in te} for tr, te in sp)
        runs.append(sorted(sorted(te) for _, te in sp))
    assert runs[0] != runs[1] or runs[0] != runs[2]


def test_labels_of_group_mates_do_not_reach_m():
    """Đổi nhãn mutant cùng nhóm với m không đổi: fail_rate của dòng train của m, điểm out-of-fold của m (B3, LGBM),
    và quyết định của m ở fold test."""
    d = _grouped()
    d2, _ = _flip_mates(d)
    ids, m = [x["mutant_id"] for x in d.mutants], "m10"
    (X1, k1), (X2, _) = features(d, ids, [m]), features(d2, ids, [m])
    assert (X1 == X2).all() and k1[0][0] == m
    for method, cfg in [("b3", TINY["b3"][0]), ("lgbm", TINY["lgbm"][0])]:
        o1, o2 = _oof(method, cfg, d, ids, 0), _oof(method, cfg, d2, ids, 0)
        assert all(o1[m, t] == o2[m, t] for t in d.tests), method
    p1 = [r for r in run_seed(d, 0, TINY, 5)[1] if r["mutant_id"] == m]
    p2 = [r for r in run_seed(d2, 0, TINY, 5)[1] if r["mutant_id"] == m]
    assert len(p1) == len(p2) > 0 and p1 == p2


def test_balance_ok_by_hand():
    ms = [{"mutant_id": i, "type": "M0", "has_fail": 0, "group": g} for i, g in zip("abcd", ["g1", "g1", "g2", "g3"])]
    assert balance_ok(ms, [(["c", "d"], ["a", "b"]), (["a", "b"], ["c", "d"])]) == (True, 0.0)
    assert balance_ok(ms, [(["d"], ["a", "b", "c"]), (["a", "b", "c"], ["d"])]) == (False, 0.5)  # cỡ 3 > 1,2·2; lệch 1, cap 2


def _real():
    st = {r["mutant_id"]: r["status"] for r in csv.DictReader(open(D1 / "mutant_status.csv", encoding="utf-8"))}
    fail = defaultdict(int)
    for r in csv.DictReader(open(D1 / "labels.csv", encoding="utf-8")):
        if st.get(r["mutant_id"]) in UNITS:
            fail[r["mutant_id"]] |= int(r["label"])
    g = near_dup_groups(D1)
    return [{"mutant_id": r["mutant_id"], "type": r["type"], "has_fail": fail[r["mutant_id"]], "group": g[r["mutant_id"]]}
            for r in csv.DictReader(open(D1 / "mutants" / "mutants.csv", encoding="utf-8")) if st.get(r["mutant_id"]) in UNITS]


def _uncut(by, splits):
    return all(not {by[i]["group"] for i in tr} & {by[i]["group"] for i in te} for tr, te in splits)


@need_d1
def test_real_groups_split_balanced_outer_and_inner():
    ms = _real()
    by = {m["mutant_id"]: m for m in ms}
    assert (len(ms), len({m["group"] for m in ms}), sum(m["has_fail"] for m in ms)) == (353, 142, 144)
    tests = []
    for seed in range(5):
        outer = folds(ms, seed)
        assert balance_ok(ms, outer)[0] and _uncut(by, outer), seed
        for tr, _ in outer:
            sub = [by[i] for i in tr]
            inner = folds(sub, seed, k=4)
            assert balance_ok(sub, inner)[0] and _uncut(by, inner), seed
        tests.append(sorted(sorted(te) for _, te in outer))
    assert len({str(t) for t in tests}) == 5  # thành phần fold khác nhau giữa các seed


@need_d1
def test_fake_groups_keep_size_distribution_per_type():
    ms = _real()
    real = {m["mutant_id"]: m["group"] for m in ms}
    for seed in range(5):
        fg = fake_groups(ms, real, seed)
        assert fg == fake_groups(ms, real, seed) and set(fg) == set(real)
        for t in {m["type"] for m in ms}:
            ids = [m["mutant_id"] for m in ms if m["type"] == t]
            assert sorted(Counter(fg[i] for i in ids).values()) == sorted(Counter(real[i] for i in ids).values())
            assert all(fg[i].startswith(t + "|") for i in ids)
    assert fake_groups(ms, real, 0) != fake_groups(ms, real, 1)


def test_same_row_tolerates_only_float_rounding_of_cost():
    from run_nhom import same_row
    w = {"method": "lgbm", "caught": "1", "e2e": "1", "cost": "291.49199999999996", "cost_b0prime": "297.46799999999996"}
    assert same_row(dict(w, cost="291.492"), w)
    assert not same_row(dict(w, caught="0"), w)
    assert not same_row(dict(w, cost="291.4923"), w)

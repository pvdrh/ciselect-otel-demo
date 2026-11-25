from collections import Counter

from sample import sample

SITES = [{"site_id": f"{t}-{s}-{i}", "type": t, "service": s}
         for t in ["M0", "M1", "M2", "M3"] for s in ["a", "b", "c"] for i in range(20)]
PROPS = {"M0": 0.2, "M1": 0.3, "M2": 0.3, "M3": 0.2}


def test_same_seed_same_list_different_seed_different_list():
    assert sample(SITES, 40, PROPS, seed=1) == sample(SITES, 40, PROPS, seed=1)
    assert sample(SITES, 40, PROPS, seed=1) != sample(SITES, 40, PROPS, seed=2)


def test_type_counts_follow_proportions():
    assert Counter(r["type"] for r in sample(SITES, 40, PROPS, seed=0)) == {"M0": 8, "M1": 12, "M2": 12, "M3": 8}


def test_services_balanced_within_each_type():
    got = Counter((r["type"], r["service"]) for r in sample(SITES, 40, PROPS, seed=0))
    for t in PROPS:
        per_svc = [got[(t, s)] for s in "abc"]
        assert max(per_svc) - min(per_svc) <= 1


def test_no_site_drawn_twice_and_short_service_does_not_block_quota():
    sites = [r for r in SITES if not (r["type"] == "M1" and r["service"] == "a" and r["site_id"][-1] != "0")]
    out = sample(sites, 40, PROPS, seed=0)
    assert len({r["site_id"] for r in out}) == len(out) == 40

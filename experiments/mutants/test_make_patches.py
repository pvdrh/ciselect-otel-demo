import subprocess

from make_patches import make_patch


def repo(tmp_path):
    (tmp_path / "a.go").write_text("x\nif ok {\ny\n")
    (tmp_path / "r.php").write_text("$o['k']; f($o['k']);\n")
    for c in (["git", "init", "-q"], ["git", "add", "."], ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"]):
        subprocess.run(c, cwd=tmp_path, check=True)
    return tmp_path


def git(tmp_path, *a):
    return subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, text=True)


def test_line_site_gives_patch_that_applies_and_leaves_repo_clean(tmp_path):
    d = repo(tmp_path)
    p = make_patch(d, {"file": "a.go", "line": "2", "old": "if ok {", "new": "if !(ok) {", "all": "", "regen": ""})
    assert git(d, "status", "--porcelain").stdout == ""
    assert "+if !(ok) {" in p["patch"] and p["files_changed"] == 1 and p["lines_changed"] == 2
    (d / "m.patch").write_text(p["patch"])
    assert git(d, "apply", "--check", "m.patch").returncode == 0


def test_all_site_replaces_every_occurrence(tmp_path):
    d = repo(tmp_path)
    p = make_patch(d, {"file": "r.php", "line": "", "old": "'k'", "new": "'kX'", "all": "True", "regen": ""})
    assert p["patch"].count("'kX'") == 2


def test_regen_runs_after_edit_and_its_output_is_in_patch(tmp_path):
    d = repo(tmp_path)
    calls = []

    def regen(demo, lang, service):
        calls.append((lang, service, (demo / "a.go").read_text()))
        (demo / "gen.txt").write_text("generated\n")

    p = make_patch(d, {"file": "a.go", "line": "2", "old": "if ok {", "new": "if !(ok) {", "all": "", "regen": "go",
                       "service": "checkout"}, regen=regen)
    assert calls == [("go", "checkout", "x\nif !(ok) {\ny\n")]
    assert "gen.txt" in p["patch"] and p["files_changed"] == 2
    assert git(d, "status", "--porcelain").stdout == ""


def test_stale_site_is_rejected(tmp_path):
    d = repo(tmp_path)
    try:
        make_patch(d, {"file": "a.go", "line": "1", "old": "if ok {", "new": "z", "all": "", "regen": ""})
    except AssertionError:
        return
    raise AssertionError("phải từ chối khi dòng không khớp")


from make_patches import heldout

SITES = [{"site_id": f"M{t}:x:{i}", "type": f"M{t}"} for t in (0, 2) for i in range(3)]


def test_heldout_excludes_used_and_puts_m0_last():
    out = heldout(SITES, used={"M2:x:0", "M0:x:1"})
    assert sorted(s["site_id"] for s in out) == ["M0:x:0", "M0:x:2", "M2:x:1", "M2:x:2"]
    assert [s["type"] for s in out] == ["M2", "M2", "M0", "M0"]


def test_heldout_order_is_seeded():
    assert heldout(SITES, set(), 0) == heldout(list(reversed(SITES)), set(), 0)

import collections
import csv
import subprocess
from pathlib import Path

import pytest

from allowlist import build
from compose_owner import owner_by_line
from diff import changes, parse
from scope import scope

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results" / "ts02_dataset" / "20261001-213236_seed0" / "mutants"
DEMO = ROOT / "data" / "raw" / "opentelemetry-demo"

COMPOSE = """\
x-default-logging: &logging
  driver: "json-file"

services:
  cart:
    image: cart
    deploy:
      resources:
        limits:
          memory: 20M
    environment:
      - CART_PORT
      - FLAGD_HOST
    depends_on:
      - valkey
  checkout:
    image: checkout
    environment:
      - CART_ADDR
"""
OWN = owner_by_line(COMPOSE)
ALLOW = {"files": {
    "src/cart/src/services/CartService.cs": {"service": "cart", "generated": False},
    "src/checkout/main.go": {"service": "checkout", "generated": False},
    "src/checkout/genproto/oteldemo/demo.pb.go": {"service": "checkout", "generated": True},
}}


def d(path, body, header="@@ -10,1 +10,1 @@"):
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{header}\n{body}\n"


EDIT_CART = d("src/cart/src/services/CartService.cs", "-a\n+b")
EDIT_CHECKOUT = d("src/checkout/main.go", "-a\n+b")


def run(text, max_lines=None):
    return scope(parse(text), ALLOW, OWN, max_lines)


def test_parse_counts_lines_and_old_positions():
    (c,) = parse(d("f.go", " ctx\n-old\n+new\n+new2\n ctx", "@@ -5,3 +5,4 @@"))
    assert (c.path, c.added, c.removed) == ("f.go", 2, 1)
    assert c.old_lines == [6, 6, 6]  # dòng xoá ở 6; dòng thêm neo vào dòng cũ ngay trước (6)


def test_parse_binary_file():
    (c,) = parse("diff --git a/img.png b/img.png\nBinary files a/img.png and b/img.png differ\n")
    assert (c.path, c.added, c.removed) == ("img.png", 0, 0)


def test_owner_by_line_marks_environment_only():
    at = lambda n: OWN[n - 1]  # số dòng 1-based trong COMPOSE
    assert at(11) == ("cart", True)        # "    environment:"
    assert at(13) == ("cart", True)        # "      - FLAGD_HOST"
    assert at(10) == ("cart", False)       # "          memory: 20M"
    assert at(14) == ("cart", False)       # "    depends_on:"
    assert at(19) == ("checkout", True)    # "      - CART_ADDR"
    assert at(2) == (None, False)          # x-default-logging


def test_single_service_code_in_scope():
    s = run(EDIT_CART)
    assert (s.in_scope, s.service, s.reason, s.n_files, s.n_lines) == (True, "cart", "ok", 1, 2)


def test_docs_only():
    s = run(d("README.md", "-a\n+b") + d(".chloggen/x.yaml", "+c"))
    assert (s.in_scope, s.service, s.reason) == (True, None, "docs_only")


def test_empty_diff_is_docs_only():
    assert run("").reason == "docs_only"


def test_docs_ignored_next_to_code():
    assert run(EDIT_CART + d("CHANGELOG.md", "+x")).service == "cart"


def test_file_not_in_allowlist():
    s = run(d("src/cart/src/Program.cs", "-a\n+b"))
    assert (s.in_scope, s.reason) == (False, "file_not_in_scope:src/cart/src/Program.cs")


def test_multi_service():
    assert run(EDIT_CART + EDIT_CHECKOUT).reason == "multi_service"


def test_rename_across_services_is_multi_service():
    # git diff --no-renames: đổi tên = xoá ở service này + thêm ở service kia
    text = (d("src/cart/src/services/CartService.cs", "-a", "@@ -1,1 +0,0 @@")
            + d("src/checkout/main.go", "+a", "@@ -0,0 +1,1 @@"))
    assert run(text).reason == "multi_service"


def test_proto_with_own_stub():
    s = run(d("pb/demo.proto", "-a\n+b") + d("src/checkout/genproto/oteldemo/demo.pb.go", "-a\n+b\n-c\n+d"))
    assert (s.in_scope, s.service, s.n_files, s.n_lines) == (True, "checkout", 1, 2)  # stub không tính (#42)


def test_proto_without_stub():
    assert run(d("pb/demo.proto", "-a\n+b")).reason == "proto_without_stub"
    assert run(d("pb/demo.proto", "-a\n+b") + EDIT_CHECKOUT).reason == "proto_without_stub"


def test_compose_env_line_of_one_service():
    s = run(d("compose.yaml", "-      - FLAGD_HOST\n+      - FLAGD_HOST=x", "@@ -13,1 +13,1 @@"))
    assert (s.in_scope, s.service) == (True, "cart")


def test_compose_added_line_uses_previous_line():
    s = run(d("compose.yaml", "+      - NEW_VAR", "@@ -11,0 +12,1 @@"))  # chèn sau dòng 11 "environment:"
    assert s.service == "cart"


def test_compose_non_env_key_out_of_scope():
    s = run(d("compose.yaml", "-          memory: 20M\n+          memory: 40M", "@@ -10,1 +10,1 @@"))
    assert (s.in_scope, s.reason) == (False, "file_not_in_scope:compose.yaml")


def test_compose_hunk_across_two_services():
    text = (d("compose.yaml", "-      - FLAGD_HOST\n+      - FLAGD_HOST=x", "@@ -13,1 +13,1 @@")
            + "@@ -19,1 +19,1 @@\n-      - CART_ADDR\n+      - CART_ADDR=y\n")
    assert run(text).reason == "multi_service"


def test_compose_mode_only_change_falls_back_to_b0():
    # đổi quyền tệp, không đổi dòng nào: không quy được về service nào → B0 (trước đây KeyError ở svcs.pop())
    s = run("diff --git a/compose.yaml b/compose.yaml\nold mode 100644\nnew mode 100755\n")
    assert (s.in_scope, s.service, s.reason) == (False, None, "no_service")


def test_max_lines_optional():
    big = d("src/cart/src/services/CartService.cs", "\n".join(["+x"] * 30), "@@ -10,0 +11,30 @@")
    assert run(big).in_scope
    assert run(big, max_lines=10).reason == "too_large"


def test_changes_from_git_repo(tmp_path):
    g = ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(g[:3] + ["init", "-q"], check=True)
    f = tmp_path / "src" / "cart" / "src" / "services" / "CartService.cs"
    f.parent.mkdir(parents=True)
    f.write_text("a\nb\nc\n")
    subprocess.run(g + ["add", "."], check=True)
    subprocess.run(g + ["commit", "-qm", "base"], check=True)
    f.write_text("a\nB\nc\n")
    subprocess.run(g + ["commit", "-qam", "head"], check=True)
    (c,) = changes(tmp_path, "HEAD~1", "HEAD")
    assert (c.path, c.added, c.removed, c.old_lines) == ("src/cart/src/services/CartService.cs", 1, 1, [2, 2])


@pytest.mark.skipif(not RUN.exists() or not DEMO.exists(), reason="cần run D1-7 và clone demo")
def test_all_d1_7_patches():
    """#40 và #42 trên 380 patch thật: 39 proto-only bị chặn, còn lại quy đúng service; phân bố số dòng không gồm stub."""
    allow = build(RUN)
    base = subprocess.run(["git", "-C", str(DEMO), "show", "3.1.0:compose.yaml"], capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout
    own = owner_by_line(base)
    reasons, lines, wrong = collections.Counter(), collections.Counter(), []
    for m in csv.DictReader(open(RUN / "mutants.csv", encoding="utf-8")):
        s = scope(parse((RUN / "patches" / f"{m['mutant_id']}.patch").read_text(encoding="utf-8")), allow, own)
        reasons[s.reason] += 1
        if s.in_scope:
            lines[s.n_lines] += 1
            if s.service != m["service"]:
                wrong.append((m["mutant_id"], m["service"], s.service))
    assert reasons == {"ok": 341, "proto_without_stub": 39}
    assert wrong == []
    assert lines == {1: 18, 2: 321, 4: 1, 10: 1}
    gen = {p for p, v in allow["files"].items() if v["generated"]}
    assert "src/shipping/src/shipping_service/shipping_types.rs" not in gen
    assert {p.split("/")[1] for p in gen} >= {"checkout", "product-catalog", "currency", "recommendation"}


# Plan 2026-11-21 Task 2.1: danh sách cho phép theo tiền tố thư mục (repo khác OTel Demo)
from diff import Change  # noqa: E402
from scope import prefix_owner  # noqa: E402

PRE = {"files": {}, "prefixes": {"src/ledger/": "x-shared", "src/ledger/writer/": "writer", "src/web/": "web"}}


def test_prefix_owner_longest_match():
    assert prefix_owner("src/ledger/writer/A.java", PRE) == "writer"
    assert prefix_owner("src/other/a.py", PRE) is None


def test_prefix_owner_suffix_filter():
    src_only = {**PRE, "suffixes": [".py", ".java"]}
    assert prefix_owner("src/web/app.py", src_only) == "web"
    assert prefix_owner("src/web/requirements.txt", src_only) is None


def test_scope_with_prefixes_decides_one_service():
    sc = scope([Change("src/web/app.py", 1, 1, [3, 3], ["a"], ["b"])], PRE, [])
    assert (sc.in_scope, sc.service, sc.reason) == (True, "web", "ok")


def test_scope_with_prefixes_two_services_is_multi():
    ch = [Change("src/web/app.py", 1, 0, [3]), Change("src/ledger/writer/A.java", 1, 0, [3])]
    assert scope(ch, PRE, []).reason == "multi_service"


def test_scope_without_prefixes_unchanged():
    assert scope([Change("src/web/app.py", 1, 0, [3])], {"files": {}}, []).reason == "file_not_in_scope:src/web/app.py"


def test_is_bot_counts_renovate_and_bot_accounts():
    from scope_upstream import is_bot
    assert is_bot("dependabot[bot]") and is_bot("Mend Renovate") and is_bot("OpenTelemetry Bot")
    assert not is_bot("Olivier Bourgeois")

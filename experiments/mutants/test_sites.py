from sites import comment_site, env_sites, negate_if, proto_sites

COMPOSE = """
services:
  front:
    build: {context: .}
    environment:
      - CART_ADDR
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://${COLLECTOR_HOST}:4318
      - FRONT_PORT
  cart:
    build: {context: .}
  collector:
    image: otel
"""
ENV = "CART_PORT=7070\nCART_ADDR=cart:${CART_PORT}\nCOLLECTOR_HOST=collector\nFRONT_PORT=8080\n"

PROTO = """syntax = "proto3";
service FooService {
    rpc Get(GetReq) returns (Resp) {}
}
message GetReq {
    Item item = 1;
    string id = 2;
}
message Item {
    string name = 1;
}
message Resp {
    repeated Item items = 3;
}
message Other {
    string unused = 1;
}
"""


def test_env_sites_wrong_host_and_wrong_port_for_addresses_only(tmp_path):
    (tmp_path / "compose.yaml").write_text(COMPOSE)
    (tmp_path / ".env").write_text(ENV)
    got = {(s["service"], s["new"].strip()) for s in env_sites(tmp_path / "compose.yaml", tmp_path / ".env")}
    assert got == {
        ("front", "- CART_ADDR=cart-x:7070"), ("front", "- CART_ADDR=cart:1"),
        ("front", "- OTEL_EXPORTER_OTLP_ENDPOINT=http://collector-x:4318"),
        ("front", "- OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:1"),
    }


def test_proto_sites_cover_closure_of_service_messages_and_move_field_number(tmp_path):
    got = {s["site_id"]: s for s in proto_sites(PROTO, "foo", "FooService") if not s["site_id"].endswith(":type")}
    assert set(got) == {"M1:foo:GetReq.item", "M1:foo:GetReq.id", "M1:foo:Item.name", "M1:foo:Resp.items"}
    assert got["M1:foo:GetReq.item"]["old"] == "    Item item = 1;"
    assert got["M1:foo:GetReq.item"]["new"] == "    Item item = 3;"  # max số field của message + 1


def test_negate_if_per_language():
    assert negate_if("  if (x > 0) {", "java") == "  if (!(x > 0)) {"
    assert negate_if("\tif err := f(); err != nil {", "go") == "\tif err := f(); !(err != nil) {"
    assert negate_if("    if n > 3 {", "rust") == "    if !(n > 3) {"
    assert negate_if("    elif a and b:", "python") == "    elif not (a and b):"
    assert negate_if("  if data.ok", "ruby") == "  if !(data.ok)"
    assert negate_if("    if let Some(x) = y {", "rust") is None
    assert negate_if("  if (a &&", "js") is None  # điều kiện nhiều dòng: bỏ


def test_comment_site_only_real_comments():
    assert comment_site("  // compute total", "go") == "  // compute total (note)"
    assert comment_site("# load config", "python") == "# load config (note)"
    assert comment_site("#include <x>", "cpp") is None
    assert comment_site("    #[serde(alias = \"a\")]", "rust") is None
    assert comment_site("x = 1  # trailing", "python") is None


DUP_COMPOSE = COMPOSE.replace("  cart:\n    build: {context: .}\n", "  cart:\n    build: {context: .}\n    environment:\n      - OTEL_EXPORTER_OTLP_ENDPOINT=http://${COLLECTOR_HOST}:4318\n")
DUP_PROTO = PROTO.replace("message Item {\n    string name = 1;", "message Item {\n    string id = 2;\n    string name = 1;")


def test_every_site_points_at_its_line_and_ids_are_unique(tmp_path):
    (tmp_path / "compose.yaml").write_text(DUP_COMPOSE)
    (tmp_path / ".env").write_text(ENV)
    for sites, text in [(env_sites(tmp_path / "compose.yaml", tmp_path / ".env"), DUP_COMPOSE),
                        (proto_sites(DUP_PROTO, "foo", "FooService"), DUP_PROTO)]:
        lines = text.splitlines()
        assert all(lines[s["line"] - 1] == s["old"] for s in sites)
        assert len({s["site_id"] for s in sites}) == len(sites)
    assert {s["service"] for s in env_sites(tmp_path / "compose.yaml", tmp_path / ".env")} == {"front", "cart"}


def test_tool_directives_are_not_comments():
    assert comment_site("//go:generate go run x", "go") is None
    assert comment_site("// +build linux", "go") is None


def test_proto_sites_also_change_scalar_type():
    got = {s["site_id"]: s["new"] for s in proto_sites(PROTO, "foo", "FooService")}
    assert got["M1:foo:GetReq.id:type"] == "    int64 id = 2;"
    assert got["M1:foo:Item.name:type"] == "    int64 name = 1;"
    assert "M1:foo:GetReq.item:type" not in got  # field kiểu message: không đổi kiểu


def test_proto_fields_with_aligned_spacing_count_toward_max():  # pilot s1-0020: "int32  quantity = 2;" bị bỏ sót
    proto = PROTO.replace("message Item {\n    string name = 1;", "message Item {\n    string name = 1;\n    int32  qty = 2;")
    got = {s["site_id"]: s["new"] for s in proto_sites(proto, "foo", "FooService")}
    assert got["M1:foo:Item.name"] == "    string name = 3;"
    assert got["M1:foo:Item.qty"] == "    int32  qty = 3;"
    assert got["M1:foo:Item.qty:type"] == "    string  qty = 2;"

"""D1-1 + D1-2 (gộp, ruling D1): liệt kê vị trí đột biến kèm phép thay cụ thể (file, old, new).

Mỗi site là dict: site_id, type (M0–M3), service, language, file, old, new, regen (ngôn ngữ cần sinh lại stub, M1 proto).
`line` là số dòng (từ 1) của `old` trong file; patch = thay dòng đó bằng `new`. Riêng site có all=True
(khoá JSON) thì old/new là literal, thay mọi lần xuất hiện, không có `line`.

Chạy: python sites.py DEMO_DIR OUT_CSV
"""
import csv
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))
from extract import host_of, load_env, resolve  # noqa: E402

# Service trong phạm vi đột biến: service ứng dụng có test telemetry (C1). flagd-ui, telemetry-docs không có
# test nào nên bị loại. File handler liệt kê tay; proto = tên service trong pb/demo.proto (server dùng stub).
SERVICES = {
    "ad": ("java", ["src/ad/src/main/java/oteldemo/AdService.java"], "AdService", None),
    "cart": ("csharp", ["src/cart/src/services/CartService.cs", "src/cart/src/cartstore/ValkeyCartStore.cs"], "CartService", None),
    "checkout": ("go", ["src/checkout/main.go"], "CheckoutService", "go"),
    "currency": ("cpp", ["src/currency/src/server.cpp"], "CurrencyService", "cpp"),
    "email": ("ruby", ["src/email/email_server.rb"], None, None),
    "frontend": ("ts", [f"src/frontend/gateways/rpc/{g}.gateway.ts" for g in
                        ["Ad", "Cart", "Checkout", "Currency", "ProductCatalog", "Recommendations"]]
                 + ["src/frontend/gateways/http/Shipping.gateway.ts"], None, None),
    "frontend-proxy": ("config", [], None, None),
    "image-provider": ("config", [], None, None),
    "load-generator": ("python", ["src/load-generator/locustfile.py"], None, None),
    "payment": ("js", ["src/payment/charge.js", "src/payment/index.js"], "PaymentService", None),
    "product-catalog": ("go", ["src/product-catalog/main.go"], "ProductCatalogService", "go"),
    "quote": ("php", ["src/quote/app/routes.php"], None, None),
    "recommendation": ("python", ["src/recommendation/recommendation_server.py"], "RecommendationService", "python"),
    "shipping": ("rust", ["src/shipping/src/shipping_service.rs", "src/shipping/src/shipping_service/quote.rs"], None, None),
}

SCALAR_SWAP = {"string": "int64", "int32": "string", "int64": "string", "uint32": "string", "uint64": "string"}
C_LIKE = {"java", "csharp", "cpp", "js", "ts", "php"}
COMMENT = {"python": "#", "ruby": "#", "php": "//"} | {l: "//" for l in ["java", "csharp", "cpp", "js", "ts", "go", "rust"]}


def negate_if(line, lang):
    """M2: đảo điều kiện của một câu if một dòng; None nếu không áp được."""
    if lang in C_LIKE:
        m = re.match(r"^(\s*(?:\}\s*)?(?:else\s+)?if\s*)\((.*)\)(\s*\{?\s*)$", line)
        return f"{m[1]}(!({m[2]})){m[3]}" if m and m[2].count("(") == m[2].count(")") else None
    if lang in ("go", "rust"):
        m = re.match(r"^(\s*(?:\}\s*)?(?:else\s+)?if\s+)(.+?)(\s*\{\s*)$", line)
        if not m or m[2].startswith("let "):
            return None
        init, _, cond = m[2].rpartition("; ")
        return f"{m[1]}{init + '; ' if init else ''}!({cond}){m[3]}"
    if lang == "python":
        m = re.match(r"^(\s*(?:el)?if\s+)(.+):(\s*)$", line)
        return f"{m[1]}not ({m[2]}):{m[3]}" if m else None
    if lang == "ruby":
        m = re.match(r"^(\s*(?:els)?if\s+)(.+?)(\s*)$", line)
        return f"{m[1]}!({m[2]}){m[3]}" if m else None
    return None


def comment_site(line, lang):
    """M0: thêm chữ vào cuối một dòng chỉ chứa comment; None nếu không phải dòng comment."""
    mark = COMMENT.get(lang)
    s = line.strip()
    if not mark or not s.startswith(mark) or s.startswith(("#include", "#[", "#!", "#define", "#pragma", "//go:", "// +build")):
        return None
    return line.rstrip() + " (note)"


def env_sites(compose_path, env_path):
    """M3: biến env của service ứng dụng có giá trị trỏ tới container khác → sai host hoặc sai cổng."""
    text = Path(compose_path).read_text(encoding="utf-8")
    services = yaml.safe_load(text)["services"]
    env = load_env(env_path)
    out = []
    name = section = None
    for no, raw in enumerate(text.splitlines(), 1):
        if m := re.match(r"^  ([\w-]+):\s*$", raw):
            name, section = m[1], None
            continue
        if m := re.match(r"^    (\w+):", raw):
            section = m[1]
            continue
        m = re.match(r"^      - (.+)$", raw)
        if not m or section != "environment" or "build" not in services.get(name, {}):
            continue
        key, eq, val = m[1].partition("=")
        value = resolve(val if eq else env.get(key, ""), env)
        host = host_of(value)
        if host not in services or host == name:
            continue
        for op, new_value in [("host", value.replace(host, host + "-x", 1)),
                              ("port", re.sub(r":\d+", ":1", value, count=1) if re.search(r":\d+", value) else None)]:
            if new_value:
                out.append({"site_id": f"M3:{name}:{key}:{op}", "type": "M3", "service": name, "language": "config",
                            "file": "compose.yaml", "line": no, "old": raw, "new": f"      - {key}={new_value}"})
    return out


def proto_sites(proto, service, proto_service):
    """M1: mọi field của message nằm trong bao đóng các rpc của proto_service; đổi số field thành max + 1."""
    found = list(re.finditer(r"^message (\w+) \{\n(.*?)^\}", proto, re.M | re.S))
    messages = {m[1]: m[2] for m in found}
    first_line = {m[1]: proto[:m.start(2)].count("\n") + 1 for m in found}
    body = re.search(rf"^service {proto_service} \{{\n(.*?)^\}}", proto, re.M | re.S)[1]
    todo = re.findall(r"rpc \w+\((\w+)\) returns \((\w+)\)", body)
    todo, seen = [t for pair in todo for t in pair], set()
    while todo:
        m = todo.pop()
        if m in messages and m not in seen:
            seen.add(m)
            todo += re.findall(r"^\s*(?:repeated\s+)?(\w+)\s+\w+ = \d+;", messages[m], re.M)
    out = []
    for m in sorted(seen):
        rows = messages[m].splitlines()
        fields = [(i, f) for i, row in enumerate(rows)
                  if (f := re.match(r"^(\s*(?:repeated\s+)?)(\w+)(\s+)(\w+) = (\d+);", row))]
        top = max(int(f[5]) for _, f in fields)
        for i, f in fields:
            site = {"type": "M1", "service": service, "language": "proto", "file": "pb/demo.proto",
                    "line": first_line[m] + i, "old": rows[i]}
            out.append({**site, "site_id": f"M1:{service}:{m}.{f[4]}",
                        "new": rows[i].replace(f"= {f[5]};", f"= {top + 1};", 1)})
            if f[2] in SCALAR_SWAP:  # đổi kiểu: khác wire type (varint ↔ length-delimited)
                out.append({**site, "site_id": f"M1:{service}:{m}.{f[4]}:type",
                            "new": rows[i].replace(f"{f[1]}{f[2]}{f[3]}", f"{f[1]}{SCALAR_SWAP[f[2]]}{f[3]}", 1)})
    return out


def json_sites(demo):
    """M1 cho service HTTP/JSON: đổi khoá JSON mà service đọc."""
    out = []
    q = (demo / "src/quote/app/routes.php").read_text(encoding="utf-8")
    for k in sorted(set(re.findall(r"\$jsonObject\['(\w+)'\]", q))):
        out.append(("quote", "php", "src/quote/app/routes.php", None, f"'{k}'", f"'{k}X'", True))
    e = (demo / "src/email/email_server.rb").read_text(encoding="utf-8")
    for k in sorted(set(re.findall(r"\bdata\.(\w+)", e))):
        out.append(("email", "ruby", "src/email/email_server.rb", None, f"data.{k}", f"data.{k}X", True))
    rel = "src/shipping/src/shipping_service/shipping_types.rs"
    for no, line in enumerate((demo / rel).read_text(encoding="utf-8").splitlines(), 1):
        m = re.match(r"^(\s*)pub (\w+): ", line)
        if m:
            out.append(("shipping", "rust", rel, no, line, f'{m[1]}#[serde(rename = "{m[2]}X")]\n{line}', False))
    return [{"site_id": f"M1:{svc}:{f}:{no or old}", "type": "M1", "service": svc, "language": lang,
             "file": f, "line": no, "old": old, "new": new, "all": all_} for svc, lang, f, no, old, new, all_ in out]


def code_sites(demo):
    """M2 (đảo if) và M0 (sửa comment) trên file handler; chỉ giữ dòng xuất hiện đúng 1 lần trong file."""
    out = []
    for svc, (lang, files, _, _) in SERVICES.items():
        for rel in files:
            lines = (demo / rel).read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines, 1):
                for typ, new in [("M2", negate_if(line, lang)), ("M0", comment_site(line, lang))]:
                    if new:
                        out.append({"site_id": f"{typ}:{svc}:{rel}:{i}", "type": typ, "service": svc, "language": lang,
                                    "file": rel, "line": i, "old": line, "new": new})
    return out


def all_sites(demo):
    demo = Path(demo)
    proto = (demo / "pb/demo.proto").read_text(encoding="utf-8")
    out = code_sites(demo) + json_sites(demo) + env_sites(demo / "compose.yaml", demo / ".env")
    for svc, (_, _, ps, regen) in SERVICES.items():
        if ps:
            out += [{**s, "regen": regen} for s in proto_sites(proto, svc, ps)]
    return [s for s in out if s["service"] in SERVICES]


def main(demo, out_csv):
    sites = all_sites(demo)
    cols = ["site_id", "type", "service", "language", "file", "line", "old", "new", "all", "regen"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sites)
    from collections import Counter
    c = Counter((s["type"], s["service"]) for s in sites)
    for t in ["M0", "M1", "M2", "M3"]:
        print(t, sum(v for (tt, _), v in c.items() if tt == t), {s: v for (tt, s), v in sorted(c.items()) if tt == t})


if __name__ == "__main__":
    main(*sys.argv[1:3])

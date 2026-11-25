"""V1 (plan C, C3a): trích đồ thị gọi giữa service từ compose + .env, không dùng SERVICE_EDGES.

Nút = service có `build:` (service ứng dụng). Hai nguồn cạnh (s, t) = "s gọi t":
- env: biến môi trường của s chứa địa chỉ có host là t (giá trị inline hoặc lấy từ .env, nội suy ${VAR});
- depends_on: s khai báo depends_on t.
Proto không cho thêm cạnh: mọi service dùng chung pb/demo.proto (b0-notes).

Chạy: python extract.py DEMO_DIR OUT_DIR → OUT_DIR/table.csv (precision/recall so với SERVICE_EDGES scope minimal).
"""
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml


def load_env(path):
    env = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def resolve(value, env):
    for _ in range(10):  # nội suy lồng nhau, ví dụ LOCUST_HOST=http://${FRONTEND_PROXY_ADDR}
        value = re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), ""), value)
    return value


def host_of(value):
    return urlparse(value if "://" in value else "//" + value).hostname


def extract_edges(compose_path, env_path):
    services = yaml.safe_load(Path(compose_path).read_text(encoding="utf-8"))["services"]
    env = load_env(env_path)
    apps = {n for n, s in services.items() if "build" in s}
    edges = {"env": set(), "depends_on": set()}
    for name in apps:
        s = services[name]
        for item in s.get("environment", []):
            key, _, val = item.partition("=")
            host = host_of(resolve(val if _ else env.get(key, ""), env))
            if host in apps and host != name:
                edges["env"].add((name, host))
        edges["depends_on"] |= {(name, t) for t in s.get("depends_on", {}) if t in apps}
    return edges


def k8s_edges(docs):
    """Plan 2026-11-21: cạnh s → t của manifest k8s, cùng luật "giá trị env có host là t" như extract_edges.

    Nút = Deployment, StatefulSet. Env của container: `value` viết thẳng và ConfigMap nạp qua `envFrom.configMapRef`.
    Bỏ tiền tố `jdbc:` trước khi lấy host. Tài liệu trùng tên gộp theo tên. Không đọc initContainers.
    """
    docs = [x for x in docs if x]
    cms = {x["metadata"]["name"]: x.get("data") or {} for x in docs if x.get("kind") == "ConfigMap"}
    work = {x["metadata"]["name"]: x["spec"]["template"]["spec"] for x in docs
            if x.get("kind") in ("Deployment", "StatefulSet")}
    edges = set()
    for s, pod in work.items():
        for c in pod.get("containers", []):
            vals = [e.get("value", "") for e in c.get("env", [])]
            vals += [v for r in c.get("envFrom", []) if "configMapRef" in r
                     for v in cms.get(r["configMapRef"]["name"], {}).values()]
            edges |= {(s, h) for v in vals if (h := host_of(str(v).removeprefix("jdbc:"))) in work and h != s}
    return edges


def main(demo, out):
    import csv
    sys.path.insert(0, str(Path(demo) / "test" / "telemetry"))
    from services import edges_for_scope  # chỉ dùng làm đáp án để chấm, B2 không đọc file này (C1, R4)

    truth = set(edges_for_scope("minimal"))
    e = extract_edges(Path(demo) / "compose.yaml", Path(demo) / ".env")
    e["union"] = e["env"] | e["depends_on"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "n_edges", "tp", "precision", "recall", "missed", "extra"])
        for src, got in e.items():
            tp = len(got & truth)
            w.writerow([src, len(got), tp, round(tp / len(got), 3) if got else 0, round(tp / len(truth), 3),
                        ";".join(f"{p}->{c}" for p, c in sorted(truth - got)),
                        ";".join(f"{p}->{c}" for p, c in sorted(got - truth))])
    print((out / "table.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(*sys.argv[1:3])

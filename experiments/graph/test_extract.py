from extract import extract_edges

COMPOSE = """
services:
  front:
    build: {context: .}
    depends_on: {cart: {condition: service_started}, db: {condition: service_started}}
    environment:
      - CART_ADDR
      - PAY_ADDR=http://pay:9000
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://${COLLECTOR_HOST}:4318
  cart:
    build: {context: .}
    environment:
      - DB_ADDR
  pay:
    build: {context: .}
  db:
    image: postgres
  collector:
    image: otel
"""
ENV = "CART_PORT=7070\nCART_ADDR=cart:${CART_PORT}\nDB_ADDR=db:5432\nCOLLECTOR_HOST=collector\n"


def write(tmp_path):
    (tmp_path / "compose.yaml").write_text(COMPOSE)
    (tmp_path / ".env").write_text(ENV)
    return extract_edges(tmp_path / "compose.yaml", tmp_path / ".env")


def test_env_edges_resolve_dotenv_and_inline_values(tmp_path):
    assert write(tmp_path)["env"] == {("front", "cart"), ("front", "pay")}


def test_depends_on_edges_only_between_built_services(tmp_path):
    assert write(tmp_path)["depends_on"] == {("front", "cart")}


# Plan 2026-11-21 Task 2.3: đồ thị từ manifest k8s (repo khác OTel Demo)
import yaml  # noqa: E402

from extract import k8s_edges  # noqa: E402

K8S = """
apiVersion: v1
kind: ConfigMap
metadata: {name: api-config}
data: {API_ADDR: "api:8080", DB_ADDR: "db:5432"}
---
apiVersion: v1
kind: ConfigMap
metadata: {name: db-config}
data: {URL: "jdbc:postgresql://db:5432/x"}
---
kind: Deployment
metadata: {name: web}
spec: {template: {spec: {containers: [{name: c, envFrom: [{configMapRef: {name: api-config}}]}]}}}
---
kind: Deployment
metadata: {name: api}
spec: {template: {spec: {containers: [{name: c, envFrom: [{configMapRef: {name: db-config}}],
       env: [{name: SELF, value: "api:8080"}, {name: S, valueFrom: {secretKeyRef: {name: k, key: v}}}]}]}}}
---
kind: StatefulSet
metadata: {name: db}
spec: {template: {spec: {containers: [{name: c, envFrom: [{configMapRef: {name: db-config}}]}]}}}
---
kind: Service
metadata: {name: web}
---
"""


def test_k8s_edges_configmap_jdbc_self_and_secret():
    assert k8s_edges(list(yaml.safe_load_all(K8S))) == {("web", "api"), ("web", "db"), ("api", "db")}

"""Gán mỗi dòng của compose.yaml cho (service, có nằm trong mục environment không) — luật #40."""


def owner_by_line(text):
    out, in_services, svc, env = [], False, None, False
    for line in text.splitlines():
        s = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if s and not s.startswith("#"):
            if indent == 0:
                in_services, svc, env = s.startswith("services:"), None, False
            elif in_services and indent == 2:
                svc, env = s.split(":")[0], False
            elif svc and indent == 4:
                env = s.startswith("environment:")
        out.append((svc, env and svc is not None))
    return out

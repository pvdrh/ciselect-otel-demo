"""Đọc JUnit XML (pytest --junitxml / cypress junit reporter) thành các dòng (oracle, test_id, outcome, duration_s)."""
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_junit(path, oracle):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for tc in ET.parse(path).iter("testcase"):
        if tc.find("failure") is not None or tc.find("error") is not None:
            outcome = "fail"
        elif tc.find("skipped") is not None:
            outcome = "skip"
        else:
            outcome = "pass"
        test_id = f"{tc.get('classname', '')}::{tc.get('name', '')}"
        rows.append((oracle, test_id, outcome, float(tc.get("time") or 0)))
    return rows

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("ja", Path("scripts/judge_agreement.py"))
ja = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ja)


def test_kappa_perfect_and_chance():
    assert ja.cohen_kappa(["PASS", "FAIL"], ["PASS", "FAIL"]) == 1.0
    assert (
        abs(ja.cohen_kappa(["PASS", "PASS", "FAIL", "FAIL"], ["PASS", "FAIL", "PASS", "FAIL"]))
        < 1e-9
    )

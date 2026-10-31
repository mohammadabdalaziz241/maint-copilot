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


def test_wilson_and_adjustment():
    lo, hi = ja.wilson(5, 20)
    assert 0.10 < lo < 0.12 and 0.46 < hi < 0.48
    adj = ja.adjusted_pass_rate(0.5, up=(5, 20), down=(0, 20))
    assert adj["estimate"] == 0.625
    assert adj["range95"][0] < 0.625 < adj["range95"][1]

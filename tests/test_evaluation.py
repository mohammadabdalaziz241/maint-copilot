import numpy as np

from maint_copilot.evaluation import paired_bootstrap, rrf, score_run, summarise


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]], k=60)
    assert set(fused[:2]) == {1, 2}
    assert fused.index(3) > 1 and fused.index(4) > 1


def test_rrf_is_deterministic_on_ties():
    assert rrf([[5, 6]], k=1) == [5, 6]
    assert rrf([[6], [5]], k=1) == [5, 6]


def test_summary_and_paired_bootstrap():
    qrels = {1: {10: 1}, 2: {20: 2}}
    a = score_run({1: [10], 2: [99]}, qrels)
    b = score_run({1: [10], 2: [20]}, qrels)
    s = summarise(b, [1, 2])
    assert s["ndcg@10"]["mean"] == 1.0
    d = paired_bootstrap(a, b, [1, 2])
    assert np.isclose(d["mean_diff"], 0.5)
    assert d["ci95"][0] <= 0.5 <= d["ci95"][1]

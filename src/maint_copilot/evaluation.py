"""Shared scoring: per-query metrics, bootstrap confidence intervals, paired comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from maint_copilot.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

METRICS = ("ndcg@10", "recall@10", "recall@100", "mrr")


def score_run(
    run: Mapping[int, Sequence[int]], qrels: Mapping[int, Mapping[int, int]]
) -> dict[int, dict[str, float]]:
    """run: query_id -> ranked page ids. Returns query_id -> metric -> value."""
    out = {}
    for qid, ranked in run.items():
        rels = qrels.get(qid, {})
        out[qid] = {
            "ndcg@10": ndcg_at_k(ranked, rels, 10),
            "recall@10": recall_at_k(ranked, rels, 10),
            "recall@100": recall_at_k(ranked, rels, 100),
            "mrr": reciprocal_rank(ranked, rels),
        }
    return out


def bootstrap_ci(values: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(per_query: Mapping[int, Mapping[str, float]], ids: Sequence[int]) -> dict:
    out: dict = {"n": len(ids)}
    for m in METRICS:
        v = np.array([per_query[i][m] for i in ids])
        lo, hi = bootstrap_ci(v)
        out[m] = {"mean": round(float(v.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def paired_bootstrap(
    a: Mapping[int, Mapping[str, float]],
    b: Mapping[int, Mapping[str, float]],
    ids: Sequence[int],
    metric: str = "ndcg@10",
    n: int = 10_000,
    seed: int = 0,
) -> dict:
    """Mean difference b - a with a 95% CI, resampling the same queries for both systems."""
    diff = np.array([b[i][metric] - a[i][metric] for i in ids])
    lo, hi = bootstrap_ci(diff, n=n, seed=seed)
    return {
        "metric": metric,
        "mean_diff": round(float(diff.mean()), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def rrf(runs: Sequence[Sequence[int]], k: int = 60, depth: int = 100) -> list[int]:
    """Reciprocal rank fusion (Cormack et al., 2009) of several ranked lists."""
    scores: dict[int, float] = {}
    for ranked in runs:
        for rank, doc in enumerate(ranked[:depth]):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: (-scores[d], d))

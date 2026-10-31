"""Ranking metrics with graded relevance, implemented directly so they can be unit-tested."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[int], rels: Mapping[int, int], k: int = 10) -> float:
    """nDCG@k with linear gains (gain = relevance grade), as used by trec_eval / MTEB."""
    if not rels:
        return 0.0
    gains = [rels.get(doc, 0) for doc in ranked[:k]]
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[int], rels: Mapping[int, int], k: int) -> float:
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[int], rels: Mapping[int, int]) -> float:
    for i, doc in enumerate(ranked):
        if rels.get(doc, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

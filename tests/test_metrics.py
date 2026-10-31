import math

import pytest

from maint_copilot.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_perfect_ranking_scores_one():
    rels = {1: 2, 2: 1}
    assert ndcg_at_k([1, 2, 3], rels) == pytest.approx(1.0)


def test_graded_ndcg_matches_hand_calculation():
    # ranked: d3 (0), d2 (1), d1 (2); ideal: 2, 1
    rels = {1: 2, 2: 1}
    dcg = 1 / math.log2(3) + 2 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([3, 2, 1], rels) == pytest.approx(dcg / idcg)


def test_cutoff_ignores_documents_beyond_k():
    assert ndcg_at_k([9, 8, 1], {1: 1}, k=2) == 0.0


def test_recall_and_rr():
    rels = {1: 1, 2: 2, 3: 1}
    assert recall_at_k([5, 2, 1], rels, k=2) == pytest.approx(1 / 3)
    assert reciprocal_rank([5, 6, 3], rels) == pytest.approx(1 / 3)
    assert reciprocal_rank([5, 6], rels) == 0.0


def test_no_judgements_scores_zero():
    assert ndcg_at_k([1, 2], {}) == 0.0
    assert recall_at_k([1, 2], {}, k=2) == 0.0

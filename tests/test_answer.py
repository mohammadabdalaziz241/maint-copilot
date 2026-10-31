import pytest
from pydantic import ValidationError

from maint_copilot.answer import _answer_schema, citation_scores, format_context
from maint_copilot.llm import Usage


def test_citation_scores():
    rels = {10: 2, 11: 1}
    assert citation_scores([10, 99], rels) == {
        "citation_precision": 0.5,
        "citation_hit": 1.0,
        "n_cited": 2,
    }
    assert citation_scores([10, 10], rels)["n_cited"] == 1
    assert citation_scores([], rels) == {
        "citation_precision": None,
        "citation_hit": 0.0,
        "n_cited": 0,
    }


def test_context_labels_pages_and_handles_empty_text():
    ctx = format_context(
        [{"corpus_id": 5, "doc_id": "TO_1", "page_number_in_doc": 3, "markdown": ""}]
    )
    assert '<page id="P5" source="TO_1, page 3">' in ctx
    assert "no extractable text" in ctx


def test_cost_uses_price_table():
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert u.cost_usd("claude-haiku-5-5") == 0.60
    assert u.cost_usd("unknown-model") is None


def test_schema_only_accepts_offered_page_ids():
    schema = _answer_schema([5, 9])
    ok = schema(answer="x", citations=["P5"], insufficient_evidence=False)
    assert ok.citations == ["P5"]
    with pytest.raises(ValidationError):
        schema(answer="x", citations=["P44"], insufficient_evidence=False)

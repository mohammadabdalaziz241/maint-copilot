import pytest
from pydantic import ValidationError

from maint_copilot.faithfulness import _schema, quote_on_page, summarise_claims

PAGE = (
    "| Item | Torque |\n|---|---|\n| Flange bolt | 35 in-lb |\nWARNING: Ground the aircraft first."
)


def test_quote_matching_ignores_formatting_but_not_content():
    assert quote_on_page("Flange bolt 35 in-lb", PAGE)
    assert quote_on_page("warning: ground the aircraft", PAGE)
    assert quote_on_page("Flange bolt ... Ground the aircraft", PAGE)
    assert not quote_on_page("Flange bolt 45 in-lb", PAGE)
    assert not quote_on_page("", PAGE)


def test_schema_restricts_page_ids():
    s = _schema([7])
    ok = {"claim": "c", "kind": "fact", "verdict": "SUPPORTED", "page_id": "P7", "evidence": "x"}
    assert s(claims=[ok]).claims[0].page_id == "P7"
    with pytest.raises(ValidationError):
        s(claims=[{**ok, "page_id": "P8"}])


def test_summary_counts():
    claims = [
        {
            "kind": "fact",
            "verdict": "SUPPORTED",
            "quote_verified": True,
            "page_cited_by_answer": True,
        },
        {
            "kind": "fact",
            "verdict": "UNSUPPORTED",
            "quote_verified": False,
            "page_cited_by_answer": False,
        },
        {
            "kind": "fact",
            "verdict": "PARTIALLY_SUPPORTED",
            "quote_verified": False,
            "page_cited_by_answer": True,
        },
        {
            "kind": "scope_note",
            "verdict": "SUPPORTED",
            "quote_verified": False,
            "page_cited_by_answer": False,
        },
    ]
    s = summarise_claims(claims)
    assert s["n_fact_claims"] == 3 and s["n_scope_notes"] == 1
    assert s["supported_share"] == pytest.approx(1 / 3)
    assert s["unsupported_count"] == 1 and not s["fully_grounded"]
    assert s["quote_verified_share"] == 0.5
    assert s["correctly_cited_share"] == 1.0

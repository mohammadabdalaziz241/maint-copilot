"""Claim-level faithfulness: is each statement in an answer supported by the pages shown?

One call to a verifier model (different from the answerer) splits an answer into atomic claims and
labels each against the context pages, naming the supporting page and quoting its evidence. The
verifier's quotes are then checked programmatically against the page text, so a claim backed by a
quote that does not exist on the page is caught rather than trusted.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field, create_model

from maint_copilot.answer import format_context
from maint_copilot.llm import Usage

VERIFIER_SYSTEM = """You check whether an answer about U.S. Air Force technical orders is \
supported by the pages it was written from.

1. Split the answer into atomic claims: one checkable fact each (a number with its unit, a step, \
a requirement, a warning, a definition). Do not merge facts; do not invent claims the answer does \
not make.
2. Mark a claim `scope_note` if it only describes what the pages do or do not contain \
(e.g. "the pages do not specify X"); otherwise `fact`.
3. For each `fact`, decide against the pages only:
   SUPPORTED: a page states it, allowing paraphrase; numbers, units and codes must match exactly.
   PARTIALLY_SUPPORTED: a page supports part of it, but a detail is added, broadened or altered.
   UNSUPPORTED: no page states it, or a page contradicts it.
4. Give the id of the page that best supports the claim, and copy a short evidence quote (at most \
30 words) verbatim from that page. For UNSUPPORTED claims give no page and no quote.
Use outside knowledge only to understand terms, never as support."""


def _schema(page_ids: Sequence[int]) -> type[BaseModel]:
    page_lit = Literal[tuple(f"P{i}" for i in page_ids)]  # type: ignore[valid-type]
    claim = create_model(
        "ClaimCheck",
        claim=(str, Field(description="One atomic claim from the answer.")),
        kind=(Literal["fact", "scope_note"], ...),
        verdict=(Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"], ...),
        page_id=(page_lit | None, Field(description="Supporting page id, or null.")),
        evidence=(str, Field(description="Verbatim quote from that page, or empty.")),
    )
    return create_model("FaithfulnessCheck", claims=(list[claim], ...))


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def quote_on_page(quote: str, page_text: str) -> bool:
    """True if the quote's alphanumeric content occurs in the page (ignores spacing, punctuation,
    markdown table pipes and case). Quotes that use an ellipsis are checked piece by piece."""
    pieces = [p for p in re.split(r"\.\.\.|…", quote) if _normalise(p)]
    if not pieces:
        return False
    page = _normalise(page_text)
    return all(_normalise(p) in page for p in pieces)


def check_answer(
    client,
    model: str,
    question: str,
    answer: str,
    pages: Sequence[Mapping],
    citations: Sequence[int],
) -> dict:
    ids = [int(p["corpus_id"]) for p in pages]
    text_by_id = {int(p["corpus_id"]): p["markdown"] for p in pages}
    resp = client.messages.parse(
        model=model,
        max_tokens=6000,
        system=VERIFIER_SYSTEM,
        output_format=_schema(ids),
        messages=[
            {
                "role": "user",
                "content": f"{format_context(pages)}\n\n<question>{question}</question>\n"
                f"<answer>{answer}</answer>",
            }
        ],
    )
    out = resp.parsed_output
    claims = []
    for c in out.claims if out else []:
        pid = int(c.page_id.lstrip("P")) if c.page_id else None
        claims.append(
            {
                "claim": c.claim,
                "kind": c.kind,
                "verdict": c.verdict,
                "page_id": pid,
                "evidence": c.evidence,
                "quote_verified": bool(
                    pid and c.evidence and quote_on_page(c.evidence, text_by_id[pid])
                ),
                "page_cited_by_answer": pid in set(citations) if pid else False,
            }
        )
    return {
        "claims": claims,
        "verifier_model": model,
        "stop_reason": resp.stop_reason,
        "usage": Usage.from_response(resp.usage).__dict__,
        **summarise_claims(claims),
    }


def summarise_claims(claims: Sequence[Mapping]) -> dict:
    facts = [c for c in claims if c["kind"] == "fact"]
    n = len(facts)
    supported = [c for c in facts if c["verdict"] == "SUPPORTED"]
    unsupported = [c for c in facts if c["verdict"] == "UNSUPPORTED"]
    with_page = [c for c in facts if c["verdict"] != "UNSUPPORTED"]
    return {
        "n_fact_claims": n,
        "n_scope_notes": len(claims) - n,
        "supported_share": len(supported) / n if n else None,
        "unsupported_count": len(unsupported),
        "partially_supported_count": sum(1 for c in facts if c["verdict"] == "PARTIALLY_SUPPORTED"),
        "fully_grounded": n > 0 and not unsupported,
        "quote_verified_share": (
            sum(c["quote_verified"] for c in with_page) / len(with_page) if with_page else None
        ),
        "correctly_cited_share": (
            sum(c["page_cited_by_answer"] for c in with_page) / len(with_page)
            if with_page
            else None
        ),
    }

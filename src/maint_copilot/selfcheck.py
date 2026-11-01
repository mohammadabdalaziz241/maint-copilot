"""Self-check step for the agent: keep only claims the pages it read support.

Runs after the agent submits. The agent's own model checks each claim of the draft against the
pages read, quoting evidence, and rewrites the answer without unsupported content. Quotes are
verified against the page text in code; the rate is logged, not trusted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, ValidationError, create_model

from maint_copilot.answer import format_context
from maint_copilot.faithfulness import quote_on_page
from maint_copilot.llm import Usage

SELFCHECK_SYSTEM = """You check a draft answer about U.S. Air Force technical orders before it \
is shown to a maintenance technician. Use only the pages provided.

1. List every factual claim in the draft (one checkable fact each). For each, decide whether a \
page states it, allowing paraphrase; numbers, units, part numbers and codes must match exactly. \
If supported, give the page id and copy a short quote (at most 30 words) verbatim from it.
2. Write `revised_answer`: the draft with unsupported claims removed, or narrowed to what the page \
actually says. Do not add facts that were not in the draft. Keep the draft's structure and \
WARNING/CAUTION text where supported. If removing a claim leaves part of the question unanswered, \
say in one short sentence that the pages read do not state it.
3. `citations`: the ids of the pages that support the revised answer."""


def _schema(page_ids: Sequence[int]):
    page = Literal[tuple(f"P{i}" for i in page_ids)]  # type: ignore[valid-type]
    claim = create_model(
        "DraftClaim",
        claim=(str, ...),
        supported=(bool, ...),
        page_id=(page | None, Field(description="Supporting page id, or null.")),
        quote=(str, Field(description="Verbatim quote from that page, or empty.")),
    )
    return create_model(
        "SelfCheck",
        claims=(list[claim], ...),
        revised_answer=(str, ...),
        citations=(list[page], ...),
    )


def self_check(client, model: str, question: str, draft: str, pages: Sequence[Mapping]) -> dict:
    ids = [int(p["corpus_id"]) for p in pages]
    text = {int(p["corpus_id"]): p["markdown"] for p in pages}
    content = f"{format_context(pages)}\n\n<question>{question}</question>\n<draft>{draft}</draft>"
    resp, out, error = None, None, None
    for max_tokens in (6000, 12000):
        try:
            resp = client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=SELFCHECK_SYSTEM,
                output_format=_schema(ids),
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": content}],
            )
            out, error = resp.parsed_output, None
            if out is not None:
                break
            error = f"no parsed output (stop_reason={resp.stop_reason})"
        except ValidationError as err:
            error = f"self-check output invalid: {str(err).splitlines()[0][:120]}"
    usage = Usage.from_response(resp.usage).__dict__ if resp else Usage().__dict__
    if out is None or not out.revised_answer.strip():
        return {"applied": False, "error": error or "empty revision", "usage": usage}
    claims = []
    for c in out.claims:
        pid = int(c.page_id.lstrip("P")) if c.page_id else None
        claims.append(
            {
                "claim": c.claim,
                "supported": c.supported,
                "page_id": pid,
                "quote_verified": bool(pid and c.quote and quote_on_page(c.quote, text[pid])),
            }
        )
    supported = [c for c in claims if c["supported"]]
    return {
        "applied": True,
        "error": None,
        "revised_answer": out.revised_answer,
        "citations": list(dict.fromkeys(int(c.lstrip("P")) for c in out.citations)),
        "claims_checked": len(claims),
        "claims_removed": len(claims) - len(supported),
        "quote_verified_share": (
            sum(c["quote_verified"] for c in supported) / len(supported) if supported else None
        ),
        "claims": claims,
        "usage": usage,
    }

"""Summary statistics for a set of judged answer records (shared by pipeline and agent runs)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from maint_copilot.evaluation import bootstrap_ci
from maint_copilot.llm import PRICE_SOURCE, Usage


def mean_ci(values) -> dict:
    v = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(v)
    return {"mean": round(float(v.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(v)}


def summarise_records(records: Sequence[dict], answer_model: str, judge_model: str) -> dict:
    passed = [r["judge"]["verdict"] == "PASS" for r in records]
    answered = [r for r in records if not r["insufficient_evidence"]]
    ans_usage = Usage(**{k: sum(r["usage"][k] for r in records) for k in Usage().__dict__})
    judge_usage = Usage(
        **{k: sum(r["judge"]["usage"][k] for r in records) for k in Usage().__dict__}
    )
    lat = np.array([r["latency_s"] for r in records])
    by_type: dict[str, list[bool]] = {}
    for r, p in zip(records, passed, strict=True):
        for t in r["query_types"]:
            by_type.setdefault(t, []).append(p)
    return {
        "judge_pass_rate": mean_ci(passed),
        "pass_rate_by_query_type": {
            t: round(float(np.mean(v)), 3) for t, v in sorted(by_type.items())
        },
        "context_contains_relevant_page": mean_ci([r["context_has_relevant"] for r in records]),
        "abstention_rate": round(1 - len(answered) / len(records), 4),
        "citation_precision_when_citing": mean_ci(
            [r["citation_precision"] for r in records if r["citation_precision"] is not None]
        ),
        "citation_hit_rate": mean_ci([r["citation_hit"] for r in records]),
        "invalid_citations_total": sum(len(r.get("invalid_citations", [])) for r in records),
        "answer_errors": sum(1 for r in records if r.get("error")),
        "answer_latency_s": {
            "p50": round(float(np.median(lat)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
        },
        "tokens": {"answer": ans_usage.__dict__, "judge": judge_usage.__dict__},
        "estimated_cost_usd": {
            "answer": round(ans_usage.cost_usd(answer_model) or 0, 4),
            "judge": round(judge_usage.cost_usd(judge_model) or 0, 4),
            "price_source": PRICE_SOURCE,
        },
    }

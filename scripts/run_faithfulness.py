"""Check claim-level faithfulness of a cached answer run.

Reads an answer or agent results file, re-builds each answer's context pages, and asks the verifier
model to label every claim. Results are cached per query in data/cache/faithfulness/<stem>_<model>/.
Writes evals/results/faithfulness_<stem>.json, where <stem> is the input name without "answers_".
Usage:
    python scripts/run_faithfulness.py evals/results/answers_dev_rerank_k10_v3.json
    python scripts/run_faithfulness.py evals/results/agent_sealed_a1.1_r0.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from maint_copilot.data import load_benchmark
from maint_copilot.evaluation import bootstrap_ci
from maint_copilot.faithfulness import check_answer
from maint_copilot.llm import PRICE_SOURCE, Usage, client


def mean_ci(values) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return {"mean": None, "ci95": None, "n": 0}
    lo, hi = bootstrap_ci(v)
    return {"mean": round(float(v.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(v)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--verifier", default="claude-sonnet-5-5")
    ap.add_argument("--limit", type=int, default=0, help="only the first N queries (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    src = Path(args.answers)
    run = json.loads(src.read_text())
    records = run["records"][: args.limit or None]
    stem = src.stem.removeprefix("answers_")
    cache = Path("data/cache/faithfulness") / f"{stem}_{args.verifier}"
    cache.mkdir(parents=True, exist_ok=True)
    pages = load_benchmark().corpus.set_index("corpus_id")
    llm = client()

    def check(rec: dict) -> dict:
        path = cache / f"{rec['query_id']}.json"
        if path.exists():
            return json.loads(path.read_text())
        if rec.get("error") or not rec["answer"].strip():
            out = {"skipped": "no answer"}
        elif rec["insufficient_evidence"]:
            out = {"skipped": "declined"}
        elif not rec["context_page_ids"]:  # nothing to check against (added after sealed run)
            out = {"skipped": "verifier failed", "error": "answer submitted without reading a page"}
        else:
            ctx = [
                {"corpus_id": i, **pages.loc[i, ["doc_id", "page_number_in_doc", "markdown"]]}
                for i in rec["context_page_ids"]
            ]
            out = check_answer(
                llm, args.verifier, rec["query"], rec["answer"], ctx, rec["citations"]
            )
        out = {"query_id": rec["query_id"], "judge_verdict": rec["judge"]["verdict"], **out}
        path.write_text(json.dumps(out))
        return out

    with ThreadPoolExecutor(args.workers) as pool:
        checks = list(pool.map(check, records))

    done = [c for c in checks if "skipped" not in c]
    claims = [cl for c in done for cl in c["claims"]]
    facts = [cl for cl in claims if cl["kind"] == "fact"]
    backed = [cl for cl in facts if cl["verdict"] != "UNSUPPORTED"]
    usage = Usage(**{k: sum(c["usage"][k] for c in done) for k in Usage().__dict__})

    def by_verdict(v: str) -> dict:
        sub = [c for c in done if c["judge_verdict"] == v and c["n_fact_claims"]]
        return {
            "answers": len(sub),
            "supported_share": mean_ci([c["supported_share"] for c in sub]),
            "fully_grounded": mean_ci([c["fully_grounded"] for c in sub]),
        }

    summary = {
        "answers_file": str(src),
        "answer_prompt_version": run["summary"].get("prompt_version"),
        "verifier_model": args.verifier,
        "answers_checked": len(done),
        "answers_skipped": {
            k: sum(1 for c in checks if c.get("skipped") == k)
            for k in ("declined", "no answer", "verifier failed")
        },
        "per_answer": {
            "supported_share": mean_ci([c["supported_share"] for c in done]),
            "fully_grounded": mean_ci([c["fully_grounded"] for c in done if c["n_fact_claims"]]),
            "unsupported_claims_mean": round(
                float(np.mean([c["unsupported_count"] for c in done])), 3
            ),
        },
        "per_claim": {
            "fact_claims": len(facts),
            "scope_notes": len(claims) - len(facts),
            "supported": sum(cl["verdict"] == "SUPPORTED" for cl in facts),
            "partially_supported": sum(cl["verdict"] == "PARTIALLY_SUPPORTED" for cl in facts),
            "unsupported": sum(cl["verdict"] == "UNSUPPORTED" for cl in facts),
            "verifier_quote_found_on_page": round(
                sum(cl["quote_verified"] for cl in backed) / max(len(backed), 1), 4
            ),
            "supporting_page_was_cited": round(
                sum(cl["page_cited_by_answer"] for cl in backed) / max(len(backed), 1), 4
            ),
        },
        "by_judge_verdict": {"PASS": by_verdict("PASS"), "FAIL": by_verdict("FAIL")},
        "tokens": usage.__dict__,
        "estimated_cost_usd": {
            "verifier": round(usage.cost_usd(args.verifier) or 0, 4),
            "price_source": PRICE_SOURCE,
        },
    }
    if not args.limit:
        out = Path(f"evals/results/faithfulness_{stem}.json")
        out.write_text(json.dumps({"summary": summary, "checks": checks}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

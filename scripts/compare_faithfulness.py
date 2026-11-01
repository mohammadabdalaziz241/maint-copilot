"""Paired faithfulness comparison: agent vs hybrid baseline, same verifier, same questions.

Writes evals/results/faithfulness_agent_vs_baseline_<split>_<agent version>.json. Usage:
    python scripts/compare_faithfulness.py --agent-version a2
    python scripts/compare_faithfulness.py --agent-version a1.1 --split sealed
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from maint_copilot.evaluation import bootstrap_ci

_spec = importlib.util.spec_from_file_location("ja", Path(__file__).with_name("judge_agreement.py"))
_ja = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ja)


def claim_rates(checks: list[dict]) -> dict:
    facts = [cl for c in checks if "claims" in c for cl in c["claims"] if cl["kind"] == "fact"]
    n = len(facts)
    out = {"fact_claims": n}
    for v in ("UNSUPPORTED", "PARTIALLY_SUPPORTED"):
        k = sum(cl["verdict"] == v for cl in facts)
        lo, hi = _ja.wilson(k, n)
        out[v.lower()] = {
            "count": k,
            "rate": round(k / n, 4),
            "wilson95": [round(lo, 4), round(hi, 4)],
        }
    return out


def paired(a: dict, b: dict, ids: list[int], key: str) -> dict:
    d = np.array([float(a[i][key]) - float(b[i][key]) for i in ids])
    lo, hi = bootstrap_ci(d)
    return {"agent_minus_baseline": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-version", required=True)
    ap.add_argument("--split", choices=["dev", "sealed"], default="dev")
    args = ap.parse_args()
    res = Path("evals/results")
    agent_file = res / f"faithfulness_agent_{args.split}_{args.agent_version}_r0.json"
    base_file = res / f"faithfulness_{args.split}_hybrid_k10_v3.json"
    ag, bs = json.loads(agent_file.read_text()), json.loads(base_file.read_text())
    a = {c["query_id"]: c for c in ag["checks"] if "claims" in c and c["n_fact_claims"]}
    b = {c["query_id"]: c for c in bs["checks"] if "claims" in c and c["n_fact_claims"]}
    ids = sorted(set(a) & set(b))
    result = {
        "split": args.split,
        "agent_version": args.agent_version,
        "verifier": ag["summary"]["verifier_model"],
        "paired_questions": len(ids),
        "supported_share_per_answer": {
            "agent": round(float(np.mean([a[i]["supported_share"] for i in ids])), 4),
            "baseline": round(float(np.mean([b[i]["supported_share"] for i in ids])), 4),
            **paired(a, b, ids, "supported_share"),
        },
        "answers_with_no_unsupported_claim": {
            "agent": round(float(np.mean([a[i]["fully_grounded"] for i in ids])), 4),
            "baseline": round(float(np.mean([b[i]["fully_grounded"] for i in ids])), 4),
            **paired(a, b, ids, "fully_grounded"),
        },
        "unsupported_claims_per_answer": {
            "agent": round(float(np.mean([a[i]["unsupported_count"] for i in ids])), 3),
            "baseline": round(float(np.mean([b[i]["unsupported_count"] for i in ids])), 3),
            **paired(a, b, ids, "unsupported_count"),
        },
        "fact_claims_per_answer": {
            "agent": round(float(np.mean([a[i]["n_fact_claims"] for i in ids])), 2),
            "baseline": round(float(np.mean([b[i]["n_fact_claims"] for i in ids])), 2),
        },
        "claim_level_all_checked_answers": {
            "agent": claim_rates(ag["checks"]),
            "baseline": claim_rates(bs["checks"]),
        },
        "caveat": "Verifier misses some unsupported claims (human spot check: 1 of 11 'supported' "
        "claims was unsupported), so absolute rates are underestimates; the paired difference uses "
        "the same verifier for both arms.",
    }
    out = res / f"faithfulness_agent_vs_baseline_{args.split}_{args.agent_version}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

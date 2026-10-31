"""Apply the pre-registered Phase 3 comparison: agent vs non-agentic baseline on dev.

Baseline: evals/results/answers_dev_hybrid_k10_v3.json (same retriever, same answer rules).
Writes evals/results/agent_vs_baseline_dev.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from maint_copilot.evaluation import bootstrap_ci


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-version", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    base = json.loads(Path("evals/results/answers_dev_hybrid_k10_v3.json").read_text())
    runs = [
        json.loads(Path(f"evals/results/agent_dev_{args.agent_version}_r{i}.json").read_text())
        for i in range(args.repeats)
    ]
    b = {r["query_id"]: r for r in base["records"]}
    a0 = {r["query_id"]: r for r in runs[0]["records"]}
    ids = sorted(b)
    passed = lambda r: r["judge"]["verdict"] == "PASS"  # noqa: E731

    diff = np.array([passed(a0[i]) - passed(b[i]) for i in ids], float)
    lo, hi = bootstrap_ci(diff)
    groups = {
        "retrieval_missed_in_baseline": [i for i in ids if not b[i]["context_has_relevant"]],
        "summary_questions": None,
    }
    import pyarrow.parquet as pq

    src = (
        pq.read_table(
            "data/vidore_v3_industrial/queries/queries.parquet", columns=["query_id", "source_type"]
        )
        .to_pandas()
        .set_index("query_id")
    )
    groups["summary_questions"] = [i for i in ids if src.loc[i, "source_type"] == "summary"]

    def rate(recs, sel) -> float:
        return round(float(np.mean([passed(recs[i]) for i in sel])), 3)

    per_rep = [{r["query_id"]: r for r in run["records"]} for run in runs]
    pass_all = [all(passed(rep[i]) for rep in per_rep) for i in ids]
    cost_b = base["summary"]["estimated_cost_usd"]["answer"] / len(ids)
    cost_a = np.mean([run["summary"]["estimated_cost_usd"]["answer"] for run in runs]) / len(ids)
    result = {
        "baseline": "hybrid top 10, prompt v3, claude-haiku-5-5",
        "agent": runs[0]["summary"]["config"],
        "n_queries": len(ids),
        "pass_rate": {
            "baseline": rate(b, ids),
            "agent_r0": rate(a0, ids),
            "agent_each_repeat": [rate(p, ids) for p in per_rep],
        },
        "paired_diff_agent_r0_minus_baseline": {
            "mean": round(float(diff.mean()), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        },
        "flips": {"fail_to_pass": int((diff > 0).sum()), "pass_to_fail": int((diff < 0).sum())},
        "pass_hat_k": {
            "k": len(runs),
            "agent_all_repeats_pass": round(float(np.mean(pass_all)), 3),
        },
        "groups": {
            name: {"n": len(sel), "baseline": rate(b, sel), "agent_r0": rate(a0, sel)}
            for name, sel in groups.items()
        },
        "answer_cost_usd_per_question": {
            "baseline": round(cost_b, 5),
            "agent": round(float(cost_a), 5),
            "ratio": round(float(cost_a / cost_b), 2),
        },
        "latency_p50_s": {
            "baseline": base["summary"]["answer_latency_s"]["p50"],
            "agent": runs[0]["summary"]["answer_latency_s"]["p50"],
        },
    }
    result["success_rule_met"] = bool(
        lo > 0 and result["answer_cost_usd_per_question"]["ratio"] <= 3
    )
    Path("evals/results/agent_vs_baseline_dev.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

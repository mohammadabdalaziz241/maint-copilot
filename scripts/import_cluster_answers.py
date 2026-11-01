"""Judge open-weight answers from the cluster and compare them with the Haiku baseline.

Reads the cluster's JSONL (single-stream run, and optionally the concurrency run for throughput),
judges every answer with the frozen judge (cached per query), and applies the pre-registered
non-inferiority margin from docs/open_weight_design.md. Usage:
    python scripts/import_cluster_answers.py --name gpt-oss-20b \
        --answers evals/cluster/gpt-oss-20b.jsonl --throughput evals/cluster/gpt-oss-20b_c8.jsonl
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from maint_copilot.answer import citation_scores
from maint_copilot.data import load_benchmark
from maint_copilot.evaluation import bootstrap_ci
from maint_copilot.judge import judge_answer
from maint_copilot.llm import Usage, client
from maint_copilot.reporting import summarise_records

BASELINE = Path("evals/results/answers_dev_hybrid_k10_v3.json")
MARGIN = -0.10
# Cloud prices for one 24 GB GPU (RTX 4090 class), USD/hour: Vast.ai from $0.14, RunPod $0.74
# (third-party summary dated 17 Oct 2026, not verified on the providers' pages).
GPU_USD_PER_HOUR = (0.14, 0.74)


def load_jsonl(path: Path) -> list[dict]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        rows[r["query_id"]] = r  # last write wins if a query was retried
    return list(rows.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--throughput")
    ap.add_argument("--judge", default="claude-sonnet-5-5")
    args = ap.parse_args()

    bench = load_benchmark()
    queries = bench.queries.set_index("query_id")
    rows = load_jsonl(Path(args.answers))
    cache = Path("data/cache/answers") / f"cluster_{args.name}_judge"
    cache.mkdir(parents=True, exist_ok=True)
    llm = client()

    def build(r: dict) -> dict:
        qid = r["query_id"]
        path = cache / f"{qid}.json"
        cited = [int(c.lstrip("P")) for c in r.get("citations", [])]
        if path.exists():
            verdict = json.loads(path.read_text())
        elif r.get("error") or not r.get("answer", "").strip():
            verdict = {
                "verdict": "FAIL",
                "reasoning": "no valid answer produced",
                "missing_or_wrong": [],
                "judge_model": None,
                "usage": Usage().__dict__,
            }
            path.write_text(json.dumps(verdict))
        else:
            row = queries.loc[qid]
            verdict = judge_answer(llm, args.judge, row["query"], row["answer"], r["answer"])
            path.write_text(json.dumps(verdict))
        rels = bench.qrels[qid]
        return {
            "query_id": qid,
            "query": queries.loc[qid, "query"],
            "query_types": list(queries.loc[qid, "query_types"]),
            "reference": queries.loc[qid, "answer"],
            "answer": r.get("answer", ""),
            "citations": cited,
            "invalid_citations": r.get("invalid_citations", []),
            "insufficient_evidence": bool(r.get("insufficient_evidence")),
            "error": r.get("error"),
            "context_page_ids": r["page_ids"],
            "model": r.get("served_model", args.name),
            "usage": Usage(r.get("prompt_tokens") or 0, r.get("completion_tokens") or 0).__dict__,
            "reasoning_tokens": r.get("reasoning_tokens"),
            "finish_reason": r.get("finish_reason"),
            "latency_s": r["latency_s"],
            "judge": verdict,
            "context_has_relevant": any(rels.get(p, 0) > 0 for p in r["page_ids"]),
            **citation_scores(cited, rels),
        }

    with ThreadPoolExecutor(4) as pool:
        records = sorted(pool.map(build, rows), key=lambda x: x["query_id"])

    base = {r["query_id"]: r for r in json.loads(BASELINE.read_text())["records"]}
    ow = {r["query_id"]: r for r in records}
    ids = sorted(set(base) & set(ow))
    passed = lambda r: r["judge"]["verdict"] == "PASS"  # noqa: E731
    diff = np.array([passed(ow[i]) - passed(base[i]) for i in ids], float)
    lo, hi = bootstrap_ci(diff)
    verdict = "comparable" if lo > MARGIN else ("worse" if hi < 0 else "inconclusive")

    summary = {
        "name": args.name,
        "served_model": rows[0].get("served_model"),
        "gpu": rows[0].get("gpu"),
        "vllm": rows[0].get("vllm"),
        **summarise_records(records, args.name, args.judge),
    }
    haiku_cost = json.loads(BASELINE.read_text())["summary"]["estimated_cost_usd"]["answer"] / len(
        base
    )
    cost = {"haiku_answer_usd_per_1000_questions": round(haiku_cost * 1000, 3)}
    sequential_s = float(np.mean([r["latency_s"] for r in records]))
    cost["gpu_hours_per_1000_sequential"] = round(sequential_s * 1000 / 3600, 3)
    if args.throughput:
        tp = json.loads(Path(args.throughput + ".summary.json").read_text())
        qpm = tp["questions_per_min_this_session"]
        cost["throughput_questions_per_min"] = qpm
        cost["throughput_concurrency"] = tp["concurrency"]
        if qpm:
            hours = 1000 / (qpm * 60)
            cost["gpu_hours_per_1000_batched"] = round(hours, 3)
            cost["gpu_usd_per_1000_batched"] = [round(hours * p, 3) for p in GPU_USD_PER_HOUR]
    cost["gpu_usd_per_1000_sequential"] = [
        round(cost["gpu_hours_per_1000_sequential"] * p, 3) for p in GPU_USD_PER_HOUR
    ]
    cost["gpu_price_range_usd_per_hour"] = list(GPU_USD_PER_HOUR)
    cost["gpu_price_source"] = "third-party summary, 17 Oct 2026, unverified"

    comparison = {
        "paired_questions": len(ids),
        "pass_rate": {
            "haiku": round(float(np.mean([passed(base[i]) for i in ids])), 4),
            args.name: round(float(np.mean([passed(ow[i]) for i in ids])), 4),
        },
        "paired_diff_open_minus_haiku": {
            "mean": round(float(diff.mean()), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        },
        "flips": {"fail_to_pass": int((diff > 0).sum()), "pass_to_fail": int((diff < 0).sum())},
        "non_inferiority_margin": MARGIN,
        "verdict": verdict,
        "latency_s": {
            "haiku_p50": json.loads(BASELINE.read_text())["summary"]["answer_latency_s"],
            "open_weight": summary["answer_latency_s"],
        },
        "cost": cost,
    }
    out = Path(f"evals/results/answers_dev_hybrid_k10_v3_{args.name}.json")
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    Path(f"evals/results/open_weight_vs_haiku_dev_{args.name}.json").write_text(
        json.dumps(comparison, indent=1)
    )
    print(json.dumps(comparison, indent=1))


if __name__ == "__main__":
    main()

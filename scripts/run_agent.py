"""Run the agent on the dev split, judge its answers, and summarise (optionally repeated runs).

Each repeat is cached per query in data/cache/agent/<config>/r<i>/. Writes
evals/results/agent_dev_<version>_r<i>.json for every repeat. Usage:
    python scripts/run_agent.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from maint_copilot.agent import (
    AGENT_VERSION,
    AGENT_VERSION_SELFCHECK,
    Budget,
    Library,
    run_agent,
)
from maint_copilot.answer import citation_scores
from maint_copilot.data import load_benchmark
from maint_copilot.judge import judge_answer
from maint_copilot.llm import Usage, client
from maint_copilot.reporting import summarise_records
from maint_copilot.retrieval import BM25Retriever, ChunkIndex, DenseRetriever, HybridRetriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--judge", default="claude-sonnet-5-5")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tool-calls", type=int, default=8)
    ap.add_argument("--self-check", action="store_true", help="agent v2: self-check step")
    args = ap.parse_args()

    bench = load_benchmark()
    dev = [int(q) for q in json.loads(Path("evals/splits/split_v1.json").read_text())["dev"]]
    queries = bench.queries.set_index("query_id")
    embedder = SentenceTransformer("models/bge-small-en-v1.5", device="cpu", local_files_only=True)
    hybrid = HybridRetriever(
        BM25Retriever(bench.corpus),
        DenseRetriever(embedder, ChunkIndex.load(Path("data/cache/bge-small-en-v1.5/chunk384"))),
    )
    lock = threading.Lock()

    def search(q: str, k: int) -> list[int]:
        with lock:
            return hybrid.search([q], k)[0]

    lib = Library(bench.corpus, search)
    llm = client()
    budget = Budget(max_tool_calls=args.max_tool_calls)
    version = AGENT_VERSION_SELFCHECK if args.self_check else AGENT_VERSION
    config = f"{version}_{args.model}_tc{args.max_tool_calls}"

    for rep in range(args.repeats):
        cache = Path("data/cache/agent") / config / f"r{rep}"
        cache.mkdir(parents=True, exist_ok=True)

        def run_one(qid: int, cache: Path = cache) -> dict:
            path = cache / f"{qid}.json"
            if path.exists():
                return json.loads(path.read_text())
            row = queries.loc[qid]
            out = run_agent(
                llm,
                args.model,
                row["query"],
                lib,
                budget,
                price=lambda u: u.cost_usd(args.model),
                self_check=args.self_check,
            )
            if out["error"]:
                verdict = {
                    "verdict": "FAIL",
                    "reasoning": "no answer produced",
                    "missing_or_wrong": [],
                    "judge_model": None,
                    "usage": Usage().__dict__,
                }
            else:
                verdict = judge_answer(llm, args.judge, row["query"], row["answer"], out["answer"])
            rels = bench.qrels[qid]
            rec = {
                "query_id": qid,
                "query": row["query"],
                "query_types": list(row["query_types"]),
                "reference": row["answer"],
                **out,
                "judge": verdict,
                "context_has_relevant": any(rels.get(p, 0) > 0 for p in out["context_page_ids"]),
                "search_found_relevant": any(
                    rels.get(p, 0) > 0 for p in out["pages_seen_in_search"]
                ),
                **citation_scores(out["citations"], rels),
            }
            path.write_text(json.dumps(rec))
            return rec

        with ThreadPoolExecutor(args.workers) as pool:
            records = list(pool.map(run_one, dev))

        calls = np.array([r["tool_calls"] for r in records])
        causes: dict[str, int] = {}
        for r in records:
            causes[r["stop_cause"]] = causes.get(r["stop_cause"], 0) + 1
        summary = {
            "split": "dev",
            "config": config,
            "repeat": rep,
            "agent_version": version,
            "answer_model": args.model,
            "judge_model": args.judge,
            "budget": budget.__dict__,
            **summarise_records(records, args.model, args.judge),
            "agent": {
                "tool_calls_mean": round(float(calls.mean()), 2),
                "tool_calls_p95": float(np.percentile(calls, 95)),
                "pages_read_mean": round(
                    float(np.mean([len(r["context_page_ids"]) for r in records])), 2
                ),
                "stop_causes": causes,
                "search_found_relevant": round(
                    float(np.mean([r["search_found_relevant"] for r in records])), 4
                ),
            },
        }
        out = Path(f"evals/results/agent_dev_{version}_r{rep}.json")
        out.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
        s = summary
        print(
            f"repeat {rep}: pass {s['judge_pass_rate']['mean']} {s['judge_pass_rate']['ci95']} | "
            f"ctx∋rel {s['context_contains_relevant_page']['mean']} | tool calls "
            f"{s['agent']['tool_calls_mean']} | stops {causes} | "
            f"${s['estimated_cost_usd']['answer']} answer"
        )


if __name__ == "__main__":
    main()

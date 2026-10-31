"""Answer dev queries from retrieved pages, judge them, and score citations.

Contexts:
  rerank  top-k of the hybrid list after cross-encoder reranking (top 20 reranked)
  hybrid  top-k of the hybrid RRF list, no reranking
  oracle  the benchmark's relevant pages (grade 2 first), an upper bound for the answering step

Every query's answer and verdict is cached in data/cache/answers/<config>/, so reruns cost nothing.
Writes evals/results/answers_dev_<config>.json. Usage:
    python scripts/run_answers.py --context rerank --k 5
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from maint_copilot.answer import PROMPT_VERSION, answer_question, citation_scores
from maint_copilot.data import load_benchmark
from maint_copilot.judge import judge_answer
from maint_copilot.llm import Usage, client
from maint_copilot.reporting import summarise_records

RERANK_CACHE = Path("data/cache/rerank/bge-reranker-base_top20")
CANDIDATES = Path("data/cache/rerank/candidates_dev_hybrid.json")


def context_ids(mode: str, qid: int, k: int, qrels, candidates) -> list[int]:
    if mode == "oracle":
        rels = qrels[qid]
        return sorted(rels, key=lambda c: (-rels[c], c))[:k]
    if mode == "hybrid":
        return candidates[qid][:k]
    scores = {int(p): s for p, s in json.loads((RERANK_CACHE / f"{qid}.json").read_text()).items()}
    reranked = sorted(scores, key=lambda p: -scores[p])
    return (reranked + [p for p in candidates[qid] if p not in scores])[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", choices=["rerank", "hybrid", "oracle"], default="rerank")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--judge", default="claude-sonnet-5-5")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--thinking", action="store_true", help="adaptive thinking for the answerer")
    ap.add_argument(
        "--prompt-version",
        default=PROMPT_VERSION,
        help="summarise a cached run made with an earlier prompt version (no API calls)",
    )
    args = ap.parse_args()

    bench = load_benchmark()
    dev = json.loads(Path("evals/splits/split_v1.json").read_text())["dev"]
    queries = bench.queries.set_index("query_id").loc[dev]
    pages = bench.corpus.set_index("corpus_id")
    candidates = {int(k): v for k, v in json.loads(CANDIDATES.read_text()).items()}
    config = f"{args.context}_k{args.k}_{args.model}_{args.prompt_version}" + (
        "_thinking" if args.thinking else ""
    )
    cache = Path("data/cache/answers") / config
    cache.mkdir(parents=True, exist_ok=True)
    llm = client()

    def run_one(qid: int) -> dict:
        path = cache / f"{qid}.json"
        if path.exists():
            return json.loads(path.read_text())
        if args.prompt_version != PROMPT_VERSION:
            raise SystemExit(f"{path} missing; cannot re-run an old prompt version")
        row = queries.loc[qid]
        ids = context_ids(args.context, qid, args.k, bench.qrels, candidates)
        ctx = [
            {"corpus_id": i, **pages.loc[i, ["doc_id", "page_number_in_doc", "markdown"]]}
            for i in ids
        ]
        ans = answer_question(llm, args.model, row["query"], ctx, thinking=args.thinking)
        if ans["error"]:
            print(f"query {qid}: answer failed ({ans['error']}); scored as FAIL")
            verdict = {
                "verdict": "FAIL",
                "reasoning": "no answer produced",
                "missing_or_wrong": [],
                "judge_model": None,
                "usage": Usage().__dict__,
            }
        else:
            verdict = judge_answer(llm, args.judge, row["query"], row["answer"], ans["answer"])
        rec = {
            "query_id": qid,
            "query": row["query"],
            "query_types": list(row["query_types"]),
            "reference": row["answer"],
            **ans,
            "judge": verdict,
            "context_has_relevant": any(bench.qrels[qid].get(i, 0) > 0 for i in ids),
            **citation_scores(ans["citations"], bench.qrels[qid]),
        }
        path.write_text(json.dumps(rec))
        return rec

    with ThreadPoolExecutor(args.workers) as pool:
        records = list(pool.map(run_one, [int(q) for q in dev]))

    summary = {
        "split": "dev",
        "config": config,
        "context": args.context,
        "k": args.k,
        "answer_model": args.model,
        "prompt_version": args.prompt_version,
        "judge_model": args.judge,
        "judge_calibrated": False,
        **summarise_records(records, args.model, args.judge),
    }
    suffix = "_thinking" if args.thinking else ""
    out = Path(
        f"evals/results/answers_dev_{args.context}_k{args.k}_{args.prompt_version}{suffix}.json"
    )
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

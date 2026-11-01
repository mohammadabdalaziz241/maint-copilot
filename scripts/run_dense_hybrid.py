"""Score BM25, dense and hybrid RRF retrieval on one split (dev by default).

Uses the shared retrieval module (the same code path as the agent's search tool). Reads cached
embeddings written by scripts/embed_corpus.py. Writes evals/results/dense_hybrid_<split>.json and
the hybrid top-100 candidates (data/cache/rerank/candidates_<split>_hybrid.json) if not yet cached.
On the sealed split only the frozen systems are scored (BM25, dense chunks, hybrid RRF of the two);
see docs/sealed_test_design.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from maint_copilot.data import load_benchmark, load_split
from maint_copilot.evaluation import paired_bootstrap, rrf, score_run, summarise
from maint_copilot.retrieval import BGE_QUERY_PROMPT, BM25Retriever, ChunkIndex, DenseRetriever

DEPTH = 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bge-small-en-v1.5")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--split", choices=["dev", "sealed"], default="dev")
    args = ap.parse_args()

    bench = load_benchmark()
    ids = load_split(args.split)
    q = bench.queries[bench.queries["query_id"].isin(ids)].reset_index(drop=True)
    qids, texts = q["query_id"].astype(int).tolist(), q["query"].tolist()

    runs: dict[str, dict[int, list[int]]] = {}
    runs["bm25"] = dict(zip(qids, BM25Retriever(bench.corpus).search(texts, DEPTH), strict=True))

    model = SentenceTransformer(args.model, device="cpu", local_files_only=True)
    base = Path("data/cache") / Path(args.model).name
    timing = {}
    variants = [("dense_page", base / "page"), ("dense_chunk", base / f"chunk{args.max_tokens}")]
    if args.split == "sealed":
        variants = variants[1:]
    for name, folder in variants:
        if (folder / "manifest.json").exists():
            dense = DenseRetriever(model, ChunkIndex.load(folder))
            t0 = time.perf_counter()
            runs[name] = dict(zip(qids, dense.search(texts, DEPTH), strict=True))
            timing[name] = round((time.perf_counter() - t0) / len(qids) * 1000, 2)

    for dense in ("dense_page", "dense_chunk"):
        if dense in runs:
            runs[f"hybrid_rrf(bm25,{dense})"] = {
                qid: rrf([runs["bm25"][qid], runs[dense][qid]], k=60, depth=DEPTH)[:DEPTH]
                for qid in qids
            }

    cand_file = Path(f"data/cache/rerank/candidates_{args.split}_hybrid.json")
    if not cand_file.exists():
        cand_file.parent.mkdir(parents=True, exist_ok=True)
        hybrid = runs["hybrid_rrf(bm25,dense_chunk)"]
        cand_file.write_text(json.dumps({str(k): v for k, v in hybrid.items()}))

    scores = {name: score_run(run, bench.qrels) for name, run in runs.items()}
    result = {
        "split": args.split,
        "n_queries": len(qids),
        "embedding_model": args.model,
        "query_prompt": BGE_QUERY_PROMPT,
        "chunk_max_tokens": args.max_tokens,
        "rrf_k": 60,
        "dense_query_ms_mean_cpu": timing,
        "systems": {name: summarise(s, qids) for name, s in scores.items()},
        "vs_bm25_ndcg@10": {
            name: paired_bootstrap(scores["bm25"], s, qids)
            for name, s in scores.items()
            if name != "bm25"
        },
        "per_query_ndcg@10": {
            name: {str(k): round(v["ndcg@10"], 4) for k, v in s.items()}
            for name, s in scores.items()
        },
    }
    Path(f"evals/results/dense_hybrid_{args.split}.json").write_text(json.dumps(result, indent=1))
    for name, s in result["systems"].items():
        d = result["vs_bm25_ndcg@10"].get(name)
        delta = f"  Δ vs BM25 {d['mean_diff']:+.3f} {d['ci95']}" if d else ""
        nd = s["ndcg@10"]
        r100 = s["recall@100"]["mean"]
        print(f"{name:30s} nDCG@10 {nd['mean']:.3f} {nd['ci95']} R@100 {r100:.3f}{delta}")


if __name__ == "__main__":
    main()

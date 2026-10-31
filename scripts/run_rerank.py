"""Rerank the hybrid retriever's top-N pages with a cross-encoder, on the dev split.

A page's rerank score is the maximum over its chunks (the same chunks the dense index uses), so
content beyond the model's 512-token window is not lost. Pair scores are cached per query in
data/cache/rerank/, so the run can be resumed across short sandbox sessions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from maint_copilot.data import load_benchmark
from maint_copilot.evaluation import paired_bootstrap, score_run, summarise
from maint_copilot.retrieval import BM25Retriever, ChunkIndex, DenseRetriever, HybridRetriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="models/bge-small-en-v1.5")
    ap.add_argument("--reranker", default="models/bge-reranker-base")
    ap.add_argument("--chunks", default="data/cache/bge-small-en-v1.5/chunk384")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--time-budget", type=float, default=0)
    args = ap.parse_args()

    bench = load_benchmark()
    dev = json.loads(Path("evals/splits/split_v1.json").read_text())["dev"]
    q = bench.queries[bench.queries["query_id"].isin(dev)].reset_index(drop=True)
    qids, texts = q["query_id"].astype(int).tolist(), q["query"].tolist()

    index = ChunkIndex.load(Path(args.chunks))
    cand_file = Path("data/cache/rerank/candidates_dev_hybrid.json")
    if cand_file.exists():
        candidates = {int(k): v for k, v in json.loads(cand_file.read_text()).items()}
    else:
        embedder = SentenceTransformer(args.embedder, device="cpu", local_files_only=True)
        hybrid = HybridRetriever(BM25Retriever(bench.corpus), DenseRetriever(embedder, index))
        candidates = dict(zip(qids, hybrid.search(texts, k=100), strict=True))
        cand_file.parent.mkdir(parents=True, exist_ok=True)
        cand_file.write_text(json.dumps({str(k): v for k, v in candidates.items()}))

    cache_dir = Path("data/cache/rerank") / f"{Path(args.reranker).name}_top{args.top_n}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    reranker = CrossEncoder(args.reranker, device="cpu", local_files_only=True, max_length=512)
    page_md = dict(
        zip(bench.corpus["corpus_id"].astype(int), bench.corpus["markdown"], strict=True)
    )
    chunks_by_page: dict[int, list[str]] = {}
    for text, pid in zip(index.texts, index.page_ids, strict=True):
        chunks_by_page.setdefault(int(pid), []).append(text)

    start, n_pairs, pair_time = time.perf_counter(), 0, 0.0
    for qid, query in zip(qids, texts, strict=True):
        cache = cache_dir / f"{qid}.json"
        if cache.exists():
            continue
        if args.time_budget and time.perf_counter() - start > args.time_budget:
            done = len(list(cache_dir.glob("*.json")))
            print(f"time budget reached: {done}/{len(qids)} queries reranked; resume to continue")
            return
        pairs, owners = [], []
        for pid in candidates[qid][: args.top_n]:
            for chunk in chunks_by_page.get(pid) or [page_md[pid][:2000]]:
                pairs.append((query, chunk))
                owners.append(pid)
        t0 = time.perf_counter()
        scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        pair_time += time.perf_counter() - t0
        n_pairs += len(pairs)
        best: dict[int, float] = {}
        for pid, s in zip(owners, scores, strict=True):
            best[pid] = max(best.get(pid, float("-inf")), float(s))
        cache.write_text(json.dumps({str(k): v for k, v in best.items()}))

    hybrid_run = {qid: candidates[qid] for qid in qids}
    reranked_run = {}
    for qid in qids:
        best = {int(k): v for k, v in json.loads((cache_dir / f"{qid}.json").read_text()).items()}
        head = sorted(best, key=lambda p: -best[p])
        reranked_run[qid] = head + [p for p in candidates[qid] if p not in best]

    s_h, s_r = score_run(hybrid_run, bench.qrels), score_run(reranked_run, bench.qrels)
    result = {
        "split": "dev",
        "n_queries": len(qids),
        "candidates": f"hybrid RRF(bm25, dense chunks) top {args.top_n}",
        "reranker": args.reranker,
        "page_score": "max over the page's chunks (<=384 tokens)",
        "rerank_ms_per_pair_this_session": round(pair_time / n_pairs * 1000, 1)
        if n_pairs
        else None,
        "systems": {"hybrid": summarise(s_h, qids), "hybrid+rerank": summarise(s_r, qids)},
        "rerank_vs_hybrid_ndcg@10": paired_bootstrap(s_h, s_r, qids),
        "per_query_ndcg@10": {
            "hybrid": {str(k): round(v["ndcg@10"], 4) for k, v in s_h.items()},
            "hybrid+rerank": {str(k): round(v["ndcg@10"], 4) for k, v in s_r.items()},
        },
    }
    out = Path(f"evals/results/rerank_dev_top{args.top_n}.json")
    out.write_text(json.dumps(result, indent=1))
    for name, s in result["systems"].items():
        nd, mrr = s["ndcg@10"], s["mrr"]["mean"]
        print(f"{name:16s} nDCG@10 {nd['mean']:.3f} {nd['ci95']}  MRR {mrr:.3f}")
    d = result["rerank_vs_hybrid_ndcg@10"]
    print(f"rerank - hybrid: {d['mean_diff']:+.3f} {d['ci95']}")


if __name__ == "__main__":
    main()

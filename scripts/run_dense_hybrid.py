"""Score BM25, dense (page and chunk variants) and hybrid RRF on the dev split.

Reads cached embeddings written by scripts/embed_corpus.py. Writes
evals/results/dense_hybrid_dev.json. The sealed split is not touched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
from sentence_transformers import SentenceTransformer

from maint_copilot.data import load_benchmark
from maint_copilot.evaluation import paired_bootstrap, rrf, score_run, summarise

BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
DEPTH = 100


def load_cache(folder: Path) -> tuple[np.ndarray, np.ndarray] | None:
    manifest = folder / "manifest.json"
    if not manifest.exists():
        return None
    m = json.loads(manifest.read_text())
    shards = sorted(folder.glob("emb_*.npy"))
    emb = np.concatenate([np.load(s) for s in shards]) if shards else np.empty((0, 0))
    if len(emb) != m["n_texts"]:
        raise SystemExit(f"{folder}: {len(emb)} of {m['n_texts']} embedded; finish embedding first")
    return emb, np.array(m["page_ids"])


def dense_run(emb, page_ids, q_emb, qids) -> dict[int, list[int]]:
    sims = q_emb @ emb.T
    run = {}
    for qid, row in zip(qids, sims, strict=True):
        best: dict[int, float] = {}
        for idx in np.argsort(-row)[: DEPTH * 8]:
            pid = int(page_ids[idx])
            if pid not in best:
                best[pid] = float(row[idx])
            if len(best) == DEPTH:
                break
        run[int(qid)] = sorted(best, key=lambda p: -best[p])
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bge-small-en-v1.5")
    ap.add_argument("--max-tokens", type=int, default=384)
    args = ap.parse_args()

    bench = load_benchmark()
    dev = json.loads(Path("evals/splits/split_v1.json").read_text())["dev"]
    q = bench.queries[bench.queries["query_id"].isin(dev)].reset_index(drop=True)
    qids = q["query_id"].astype(int).tolist()
    corpus_ids = bench.corpus["corpus_id"].astype(int).tolist()

    runs: dict[str, dict[int, list[int]]] = {}
    stem = Stemmer.Stemmer("english")
    bm = bm25s.BM25()
    bm.index(bm25s.tokenize(bench.corpus["markdown"].tolist(), stopwords="en", stemmer=stem))
    idx, _ = bm.retrieve(bm25s.tokenize(q["query"].tolist(), stopwords="en", stemmer=stem), k=DEPTH)
    runs["bm25"] = {qid: [corpus_ids[j] for j in row] for qid, row in zip(qids, idx, strict=True)}

    model = SentenceTransformer(args.model, device="cpu", local_files_only=True)
    t0 = time.perf_counter()
    q_emb = model.encode_query(
        q["query"].tolist(), prompt=BGE_QUERY_PROMPT, normalize_embeddings=True
    )
    q_ms = (time.perf_counter() - t0) / len(qids) * 1000

    base = Path("data/cache") / Path(args.model).name
    for name, folder in (
        ("dense_page", base / "page"),
        ("dense_chunk", base / f"chunk{args.max_tokens}"),
    ):
        cached = load_cache(folder)
        if cached is not None:
            runs[name] = dense_run(*cached, q_emb, qids)

    for dense in ("dense_page", "dense_chunk"):
        if dense in runs:
            runs[f"hybrid_rrf(bm25,{dense})"] = {
                qid: rrf([runs["bm25"][qid], runs[dense][qid]], k=60, depth=DEPTH) for qid in qids
            }

    scores = {name: score_run(run, bench.qrels) for name, run in runs.items()}
    result = {
        "split": "dev",
        "n_queries": len(qids),
        "embedding_model": args.model,
        "query_prompt": BGE_QUERY_PROMPT,
        "chunk_max_tokens": args.max_tokens,
        "rrf_k": 60,
        "query_embedding_ms_mean": round(q_ms, 2),
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
    out = Path("evals/results/dense_hybrid_dev.json")
    out.write_text(json.dumps(result, indent=1))
    for name, s in result["systems"].items():
        d = result["vs_bm25_ndcg@10"].get(name)
        print(
            f"{name:32s} nDCG@10 {s['ndcg@10']['mean']:.3f} {s['ndcg@10']['ci95']}  "
            f"R@100 {s['recall@100']['mean']:.3f}  "
            + (f"Δ vs BM25 {d['mean_diff']:+.3f} {d['ci95']}" if d else "")
        )


if __name__ == "__main__":
    main()

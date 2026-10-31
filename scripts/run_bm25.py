"""Untuned BM25 baseline over page markdown, scored with the project's own metrics.

Writes evals/results/bm25_baseline.json with config, library versions, aggregate scores and
per-query scores for the full English set and the dev split. The sealed split is not scored here.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from maint_copilot.data import load_benchmark
from maint_copilot.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

K = 100
OUT = Path("evals/results/bm25_baseline.json")


def bootstrap_ci(values: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(per_query: dict[int, dict[str, float]], ids: list[int]) -> dict:
    out = {"n": len(ids)}
    for m in ("ndcg@10", "recall@10", "recall@100", "mrr"):
        v = np.array([per_query[i][m] for i in ids])
        lo, hi = bootstrap_ci(v)
        out[m] = {"mean": round(float(v.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def main() -> None:
    bench = load_benchmark()
    split = json.loads(Path("evals/splits/split_v1.json").read_text())
    stemmer = Stemmer.Stemmer("english")
    corpus_ids = bench.corpus["corpus_id"].astype(int).tolist()

    t0 = time.perf_counter()
    tokens = bm25s.tokenize(bench.corpus["markdown"].tolist(), stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(tokens)
    index_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_tokens = bm25s.tokenize(bench.queries["query"].tolist(), stopwords="en", stemmer=stemmer)
    idx, _ = retriever.retrieve(q_tokens, k=K)
    query_ms = (time.perf_counter() - t0) / len(bench.queries) * 1000

    per_query: dict[int, dict[str, float]] = {}
    for qid, row in zip(bench.queries["query_id"].astype(int), idx, strict=True):
        ranked = [corpus_ids[j] for j in row]
        rels = bench.qrels.get(qid, {})
        per_query[qid] = {
            "ndcg@10": ndcg_at_k(ranked, rels, 10),
            "recall@10": recall_at_k(ranked, rels, 10),
            "recall@100": recall_at_k(ranked, rels, 100),
            "mrr": reciprocal_rank(ranked, rels),
        }

    # Replication of the published multilingual figure: same index, queries in all six languages.
    languages = ["english", "french", "german", "spanish", "italian", "portuguese"]
    per_language = {}
    for lang in languages:
        lb = load_benchmark(language=lang)
        lt = bm25s.tokenize(lb.queries["query"].tolist(), stopwords="en", stemmer=stemmer)
        lidx, _ = retriever.retrieve(lt, k=10)
        scores = [
            ndcg_at_k([corpus_ids[j] for j in row], lb.qrels.get(int(qid), {}), 10)
            for qid, row in zip(lb.queries["query_id"], lidx, strict=True)
        ]
        per_language[lang] = round(float(np.mean(scores)), 4)
    replication = {
        "ndcg@10_by_query_language": per_language,
        "ndcg@10_mean_over_languages": round(float(np.mean(list(per_language.values()))), 4),
        "published_bm25s_industrial": 0.156,
        "published_source": "ViDoRe V3, arXiv 2601.08620, Table 1",
    }

    all_ids = sorted(per_query)
    result = {
        "system": (
            "bm25s defaults (k1=1.5, b=0.75, method=lucene), English stopwords, Snowball stemmer"
        ),
        "text_field": "corpus.markdown (as shipped in the dataset)",
        "versions": {"bm25s": bm25s.__version__, "python": platform.python_version()},
        "timing": {"index_seconds": round(index_s, 2), "query_ms_mean": round(query_ms, 2)},
        "full_english": summarise(per_query, all_ids),
        "dev": summarise(per_query, split["dev"]),
        "multilingual_replication": replication,
        "per_query": {str(k): {m: round(v, 4) for m, v in d.items()} for k, d in per_query.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {k: result[k] for k in ("system", "full_english", "dev", "multilingual_replication")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()

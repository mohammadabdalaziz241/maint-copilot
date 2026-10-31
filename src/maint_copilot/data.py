"""Load the locally downloaded ViDoRe V3 Industrial snapshot (no network access)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_ROOT = Path("data/vidore_v3_industrial")


def _read(root: Path, split: str, columns: list[str]) -> pd.DataFrame:
    files = sorted((root / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files under {root / split}; download the dataset first."
        )
    return pd.concat(
        [pq.read_table(f, columns=columns).to_pandas() for f in files], ignore_index=True
    )


@dataclass(frozen=True)
class Benchmark:
    corpus: pd.DataFrame  # corpus_id, doc_id, page_number_in_doc, markdown
    queries: pd.DataFrame  # query_id, query, answer, query_types, query_format, content_type
    qrels: dict[int, dict[int, int]]  # query_id -> {corpus_id: graded relevance}


def load_benchmark(root: Path = DEFAULT_ROOT, language: str = "english") -> Benchmark:
    """Return corpus pages, queries in one language, and graded relevance judgements."""
    corpus = _read(root, "corpus", ["corpus_id", "doc_id", "page_number_in_doc", "markdown"])
    corpus["markdown"] = corpus["markdown"].fillna("")
    queries = _read(
        root,
        "queries",
        [
            "query_id",
            "query",
            "language",
            "answer",
            "query_types",
            "query_format",
            "content_type",
        ],
    )
    queries = queries[queries["language"] == language].reset_index(drop=True)
    rels = _read(root, "qrels", ["query_id", "corpus_id", "score"])
    rels = rels[rels["query_id"].isin(set(queries["query_id"]))]
    qrels: dict[int, dict[int, int]] = {}
    for qid, cid, score in rels.itertuples(index=False):
        qrels.setdefault(int(qid), {})[int(cid)] = int(score)
    return Benchmark(corpus=corpus, queries=queries, qrels=qrels)

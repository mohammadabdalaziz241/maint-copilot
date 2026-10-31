"""Page retrieval: BM25, dense (chunk embeddings, max-pooled per page) and hybrid RRF.

One code path for experiments and for the agent's `search_pages` tool.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import Stemmer

from maint_copilot.evaluation import rrf

BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


class BM25Retriever:
    def __init__(self, corpus: pd.DataFrame) -> None:
        self._stem = Stemmer.Stemmer("english")
        self._ids = corpus["corpus_id"].astype(int).tolist()
        self._bm = bm25s.BM25()
        self._bm.index(
            bm25s.tokenize(corpus["markdown"].tolist(), stopwords="en", stemmer=self._stem),
            show_progress=False,
        )

    def search(self, queries: Sequence[str], k: int = 100) -> list[list[int]]:
        tokens = bm25s.tokenize(
            list(queries), stopwords="en", stemmer=self._stem, show_progress=False
        )
        idx, _ = self._bm.retrieve(tokens, k=k, show_progress=False)
        return [[self._ids[j] for j in row] for row in idx]


@dataclass
class ChunkIndex:
    embeddings: np.ndarray  # (n_chunks, dim), L2-normalised
    page_ids: np.ndarray  # (n_chunks,)
    texts: list[str]

    @classmethod
    def load(cls, folder: Path) -> ChunkIndex:
        m = json.loads((folder / "manifest.json").read_text())
        emb = np.concatenate([np.load(s) for s in sorted(folder.glob("emb_*.npy"))])
        if len(emb) != m["n_texts"]:
            raise RuntimeError(f"{folder}: {len(emb)} of {m['n_texts']} texts embedded")
        return cls(emb, np.asarray(m["page_ids"]), m["texts"])

    def chunks_for_page(self, page_id: int) -> list[str]:
        return [t for t, p in zip(self.texts, self.page_ids, strict=True) if p == page_id]


class DenseRetriever:
    def __init__(self, model, index: ChunkIndex, query_prompt: str = BGE_QUERY_PROMPT) -> None:
        self.model, self.index, self.query_prompt = model, index, query_prompt

    def search(self, queries: Sequence[str], k: int = 100) -> list[list[int]]:
        q = self.model.encode_query(
            list(queries), prompt=self.query_prompt, normalize_embeddings=True
        )
        sims = q @ self.index.embeddings.T
        runs = []
        for row in sims:
            best: dict[int, float] = {}
            for i in np.argsort(-row):
                pid = int(self.index.page_ids[i])
                best.setdefault(pid, float(row[i]))
                if len(best) == k:
                    break
            runs.append(sorted(best, key=lambda p: -best[p]))
        return runs


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60) -> None:
        self.bm25, self.dense, self.rrf_k = bm25, dense, rrf_k

    def search(self, queries: Sequence[str], k: int = 100) -> list[list[int]]:
        lex, sem = self.bm25.search(queries, k), self.dense.search(queries, k)
        return [rrf([a, b], k=self.rrf_k, depth=k)[:k] for a, b in zip(lex, sem, strict=True)]

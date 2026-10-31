"""Embed the corpus with a local sentence-transformers model, resumably.

Two variants:
  chunk  structure-based chunks of <= --max-tokens model tokens (page score = max over its chunks)
  page   whole page, truncated by the model at its maximum sequence length

Embeddings are written in shards to data/cache/<model>/<variant>/, so an interrupted run resumes
where it stopped. Usage:
    python scripts/embed_corpus.py --model models/bge-small-en-v1.5 --variant chunk
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from maint_copilot.chunking import chunk_page
from maint_copilot.data import load_benchmark

SHARD = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", choices=["chunk", "page"], default="chunk")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--time-budget", type=float, default=0, help="seconds; 0 = run to the end")
    args = ap.parse_args()

    model = SentenceTransformer(args.model, device="cpu", local_files_only=True)
    tok = model.tokenizer
    out = (
        Path("data/cache")
        / Path(args.model).name
        / (f"chunk{args.max_tokens}" if args.variant == "chunk" else "page")
    )
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        corpus = load_benchmark().corpus
        texts, page_ids = [], []
        for cid, md in zip(corpus["corpus_id"], corpus["markdown"], strict=True):
            if args.variant == "page":
                if md.strip():
                    texts.append(md)
                    page_ids.append(int(cid))
                continue

            def length(text: str) -> int:
                return len(tok(text, add_special_tokens=False)["input_ids"])

            for ch in chunk_page(int(cid), md, args.max_tokens, length):
                texts.append(ch.text)
                page_ids.append(ch.page_id)
        manifest = {
            "model": args.model,
            "variant": args.variant,
            "max_tokens": args.max_tokens if args.variant == "chunk" else model.max_seq_length,
            "n_texts": len(texts),
            "n_pages_covered": len(set(page_ids)),
            "page_ids": page_ids,
            "texts": texts,
        }
        manifest_path.write_text(json.dumps(manifest))
        print(f"prepared {len(texts)} texts covering {len(set(page_ids))} pages")

    texts, n = manifest["texts"], manifest["n_texts"]
    start = time.perf_counter()
    for s in range(0, n, SHARD):
        shard = out / f"emb_{s:06d}.npy"
        if shard.exists():
            continue
        if args.time_budget and time.perf_counter() - start > args.time_budget:
            print(f"time budget reached; resume to continue (done up to {s}/{n})")
            return
        t0 = time.perf_counter()
        emb = model.encode_document(
            texts[s : s + SHARD],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(shard, emb)
        print(f"shard {s}-{s + len(emb)} of {n} in {time.perf_counter() - t0:.1f}s", flush=True)
    print("complete")


if __name__ == "__main__":
    main()

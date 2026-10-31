"""Blind human spot check of the faithfulness verifier.

Samples every claim the verifier marked UNSUPPORTED plus a random set it marked SUPPORTED (v3,
retrieved top 10). For each claim, the evidence shown is chosen by the same verdict-blind rule:
the two context pages whose best-matching ~120-word window shares the most content words with the
claim. Writes evals/calibration/faithfulness_to_label_v1.csv, a companion markdown file with the
full page text, and keeps the verifier's labels in faithfulness_verdicts_v1.json.
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

from maint_copilot.data import load_benchmark

SOURCE = Path("evals/results/faithfulness_dev_rerank_k10_v3.json")
ANSWERS = Path("evals/results/answers_dev_rerank_k10_v3.json")
N_SUPPORTED = 11
SEED = 20261031
WINDOW, STEP = 120, 15
STOP = set(
    "the a an of to and or in on for is are be by with as at that this from it its not".split()
)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9.\-/]*", text.lower())


def best_window(claim: str, page: str) -> tuple[int, str]:
    target = {w for w in words(claim) if w not in STOP}
    tokens = page.split()
    best, best_text = -1, ""
    for start in range(0, max(1, len(tokens) - WINDOW + 1), STEP):
        chunk = " ".join(tokens[start : start + WINDOW])
        score = len(target & set(words(chunk)))
        if score > best:
            best, best_text = score, chunk
    return best, best_text


def main() -> None:
    out_csv = Path("evals/calibration/faithfulness_to_label_v1.csv")
    if out_csv.exists():
        raise SystemExit(f"{out_csv} exists; the sample is frozen.")
    checks = {c["query_id"]: c for c in json.loads(SOURCE.read_text())["checks"] if "claims" in c}
    answers = {r["query_id"]: r for r in json.loads(ANSWERS.read_text())["records"]}
    pages = load_benchmark().corpus.set_index("corpus_id")

    pool = [
        (qid, i, cl)
        for qid, c in checks.items()
        for i, cl in enumerate(c["claims"])
        if cl["kind"] == "fact"
    ]
    rng = random.Random(SEED)
    unsupported = [p for p in pool if p[2]["verdict"] == "UNSUPPORTED"]
    supported = rng.sample([p for p in pool if p[2]["verdict"] == "SUPPORTED"], N_SUPPORTED)
    sample = unsupported + supported
    rng.shuffle(sample)

    verdicts, md = {}, ["# Faithfulness spot check: full page text", ""]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "item",
                "query_id",
                "question",
                "claim",
                "page_A",
                "excerpt_A",
                "page_B",
                "excerpt_B",
                "your_label",
                "note",
            ]
        )
        for item, (qid, idx, cl) in enumerate(sample, 1):
            ctx = answers[qid]["context_page_ids"]
            scored = sorted(
                ((*best_window(cl["claim"], pages.loc[p, "markdown"]), p) for p in ctx),
                key=lambda t: -t[0],
            )[:2]
            (_, ex_a, pa), (_, ex_b, pb) = scored
            w.writerow(
                [
                    item,
                    qid,
                    answers[qid]["query"],
                    cl["claim"],
                    f"P{pa}",
                    ex_a,
                    f"P{pb}",
                    ex_b,
                    "",
                    "",
                ]
            )
            verdicts[str(item)] = {"query_id": qid, "claim_index": idx, "verifier": cl["verdict"]}
            md += [f"## Item {item}", ""]
            for p in (pa, pb):
                where = f"{pages.loc[p, 'doc_id']}, page {pages.loc[p, 'page_number_in_doc']}"
                md += [
                    f"### P{p} ({where})",
                    "",
                    pages.loc[p, "markdown"],
                    "",
                ]
    Path("evals/calibration/faithfulness_pages_v1.md").write_text("\n".join(md), encoding="utf-8")
    Path("evals/calibration/faithfulness_verdicts_v1.json").write_text(
        json.dumps({"source": str(SOURCE), "seed": SEED, "items": verdicts}, indent=1)
    )
    print(f"wrote {len(sample)} claims ({len(unsupported)} unsupported, {N_SUPPORTED} supported)")


if __name__ == "__main__":
    main()

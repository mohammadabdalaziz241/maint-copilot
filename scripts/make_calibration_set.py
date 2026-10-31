"""Draw a blind, balanced sample of judged answers for human labelling.

Writes evals/calibration/to_label_v1.csv (no judge verdicts) and
evals/calibration/judge_verdicts_v1.json (kept separate so the labeller does not see them).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

SOURCE = Path("evals/results/answers_dev_rerank_k10_v3.json")
N_PER_CLASS = 20
SEED = 20261031


def main() -> None:
    out_csv = Path("evals/calibration/to_label_v1.csv")
    if out_csv.exists():
        raise SystemExit(f"{out_csv} exists; the calibration sample is frozen.")
    records = json.loads(SOURCE.read_text())["records"]
    rng = random.Random(SEED)
    passed = [r for r in records if r["judge"]["verdict"] == "PASS"]
    failed = [r for r in records if r["judge"]["verdict"] == "FAIL" and not r.get("error")]
    sample = rng.sample(passed, N_PER_CLASS) + rng.sample(failed, N_PER_CLASS)
    rng.shuffle(sample)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "item",
                "query_id",
                "question",
                "reference_answer",
                "candidate_answer",
                "your_label",
                "note",
            ]
        )
        for i, r in enumerate(sample, 1):
            w.writerow([i, r["query_id"], r["query"], r["reference"], r["answer"], "", ""])
    Path("evals/calibration/judge_verdicts_v1.json").write_text(
        json.dumps(
            {
                "source": str(SOURCE),
                "seed": SEED,
                "judge_model": records[0]["judge"]["judge_model"],
                "verdicts": {str(r["query_id"]): r["judge"]["verdict"] for r in sample},
            },
            indent=1,
        )
    )
    print(f"wrote {len(sample)} items to {out_csv}")


if __name__ == "__main__":
    main()

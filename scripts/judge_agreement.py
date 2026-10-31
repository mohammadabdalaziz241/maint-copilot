"""Agreement between the LLM judge and human labels on the calibration sample.

Reports raw agreement, Cohen's kappa, and the judge's true-pass and true-fail rates (how often it
agrees with the human on items the human passed or failed). The judge is trusted for reporting
only if both rates are at least 0.8 (threshold recorded in evals/PROTOCOL.md).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def cohen_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    rows = list(csv.DictReader(Path("evals/calibration/to_label_v1.csv").open(encoding="utf-8")))
    verdicts = json.loads(Path("evals/calibration/judge_verdicts_v1.json").read_text())["verdicts"]
    human, judge = [], []
    for r in rows:
        lab = r["your_label"].strip().upper()
        if lab not in {"PASS", "FAIL"}:
            continue
        human.append(lab)
        judge.append(verdicts[r["query_id"]])
    if not human:
        raise SystemExit("No labels yet: fill the your_label column with PASS or FAIL.")
    pairs = list(zip(human, judge, strict=True))
    hp = [j for h, j in pairs if h == "PASS"]
    hf = [j for h, j in pairs if h == "FAIL"]
    result = {
        "n_labelled": len(human),
        "agreement": round(sum(h == j for h, j in pairs) / len(pairs), 3),
        "cohen_kappa": round(cohen_kappa(human, judge), 3),
        "judge_true_pass_rate": round(hp.count("PASS") / len(hp), 3) if hp else None,
        "judge_true_fail_rate": round(hf.count("FAIL") / len(hf), 3) if hf else None,
        "human_pass_share": round(human.count("PASS") / len(human), 3),
        "disagreements": [
            {
                "item": r["item"],
                "query_id": r["query_id"],
                "human": r["your_label"].strip().upper(),
                "judge": verdicts[r["query_id"]],
            }
            for r in rows
            if r["your_label"].strip().upper() in {"PASS", "FAIL"}
            and r["your_label"].strip().upper() != verdicts[r["query_id"]]
        ],
    }
    Path("evals/calibration/agreement_v1.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

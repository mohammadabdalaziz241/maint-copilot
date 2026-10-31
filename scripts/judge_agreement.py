"""Agreement between the LLM judge and human labels on the calibration sample.

Reports raw agreement, Cohen's kappa, and the judge's true-pass and true-fail rates (how often it
agrees with the human on items the human passed or failed). The judge is trusted for reporting
only if both rates are at least 0.8 (threshold recorded in evals/PROTOCOL.md).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def adjusted_pass_rate(judge_pass_share: float, up: tuple[int, int], down: tuple[int, int]) -> dict:
    """Estimate the human-standard pass rate of a full run from the judge's run-level pass share.

    The calibration sample is stratified by judge verdict, so within each stratum the human label
    rate is estimated directly: up = (human PASS among judge FAIL, n judge FAIL), down = (human FAIL
    among judge PASS, n judge PASS). Interval: worst/best case of the two Wilson intervals.
    """
    f_up, f_down = up[0] / up[1], down[0] / down[1]
    (lu, hu), (ld, hd) = wilson(*up), wilson(*down)
    q = judge_pass_share

    def est(u: float, d: float) -> float:
        return q * (1 - d) + (1 - q) * u

    return {
        "estimate": round(est(f_up, f_down), 3),
        "range95": [round(est(lu, hd), 3), round(est(hu, ld), 3)],
        "method": "stratified by judge verdict; Wilson bounds on both flip rates",
    }


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
    jf = [h for h, j in pairs if j == "FAIL"]
    jp = [h for h, j in pairs if j == "PASS"]
    up, down = (jf.count("PASS"), len(jf)), (jp.count("FAIL"), len(jp))
    result["true_pass_rate_wilson95"] = [round(x, 3) for x in wilson(hp.count("PASS"), len(hp))]
    result["true_fail_rate_wilson95"] = [round(x, 3) for x in wilson(hf.count("FAIL"), len(hf))]
    result["meets_protocol_threshold"] = bool(
        result["judge_true_pass_rate"] is not None
        and result["judge_true_fail_rate"] is not None
        and result["judge_true_pass_rate"] >= 0.8
        and result["judge_true_fail_rate"] >= 0.8
    )
    source = json.loads(Path("evals/calibration/judge_verdicts_v1.json").read_text())["source"]
    run = json.loads(Path(source).read_text())["summary"]
    result["adjusted_pass_rate"] = {
        "run": source,
        "judge_pass_rate": run["judge_pass_rate"]["mean"],
        **adjusted_pass_rate(run["judge_pass_rate"]["mean"], up, down),
    }
    Path("evals/calibration/agreement_v1.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

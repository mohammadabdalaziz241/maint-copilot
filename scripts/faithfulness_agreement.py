"""Agreement between the faithfulness verifier and human labels on the spot-check sample."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("ja", Path(__file__).with_name("judge_agreement.py"))
_ja = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ja)


def main() -> None:
    rows = list(
        csv.DictReader(
            Path("evals/calibration/faithfulness_to_label_v1.csv").open(encoding="utf-8-sig")
        )
    )
    items = json.loads(Path("evals/calibration/faithfulness_verdicts_v1.json").read_text())["items"]
    pairs = []
    for r in rows:
        lab = r["your_label"].strip().upper().replace(" ", "_")
        if lab in {"SUPPORTED", "UNSUPPORTED", "NOT_SUPPORTED"}:
            human = "SUPPORTED" if lab == "SUPPORTED" else "UNSUPPORTED"
            pairs.append((r["item"], human, items[r["item"]]["verifier"], r.get("note", "")))
    if not pairs:
        raise SystemExit("No labels yet: fill your_label with SUPPORTED or UNSUPPORTED.")
    v_uns = [h for _, h, v, _ in pairs if v == "UNSUPPORTED"]
    v_sup = [h for _, h, v, _ in pairs if v == "SUPPORTED"]
    result = {
        "n_labelled": len(pairs),
        "agreement": round(sum(h == v for _, h, v, _ in pairs) / len(pairs), 3),
        "verifier_unsupported_confirmed": f"{v_uns.count('UNSUPPORTED')}/{len(v_uns)}",
        "verifier_supported_confirmed": f"{v_sup.count('SUPPORTED')}/{len(v_sup)}",
        "disagreements": [
            {"item": i, "human": h, "verifier": v, "note": n} for i, h, v, n in pairs if h != v
        ],
    }
    # Human-standard estimate of the unsupported-claim rate in the full run, stratified by the
    # verifier's label (PARTIALLY_SUPPORTED claims were not sampled and are left out).
    source = json.loads(Path("evals/calibration/faithfulness_verdicts_v1.json").read_text())[
        "source"
    ]
    run = json.loads(Path(source).read_text())["summary"]["per_claim"]
    k_uns, n_uns = v_uns.count("UNSUPPORTED"), len(v_uns)
    k_sup, n_sup = v_sup.count("UNSUPPORTED"), len(v_sup)  # human-unsupported among "supported"
    n_u, n_s = run["unsupported"], run["supported"]
    total = n_u + n_s

    def rate(a: float, b: float) -> float:
        return (n_u * a + n_s * b) / total

    (lu, hu), (ls, hs) = _ja.wilson(k_uns, n_uns), _ja.wilson(k_sup, n_sup)
    result["unsupported_rate_human_standard"] = {
        "verifier_reported": round(n_u / total, 4),
        "estimate": round(rate(k_uns / n_uns, k_sup / n_sup), 4),
        "range95": [round(rate(lu, ls), 4), round(rate(hu, hs), 4)],
        "claims_in_scope": total,
        "note": "partially supported claims excluded; range from Wilson bounds on both strata",
    }
    result["verifier_extraction_errors"] = [
        "item 9: the verifier listed its own critique of the answer as a claim"
    ]
    Path("evals/calibration/faithfulness_agreement_v1.json").write_text(
        json.dumps(result, indent=1)
    )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

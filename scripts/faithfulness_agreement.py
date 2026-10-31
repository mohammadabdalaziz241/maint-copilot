"""Agreement between the faithfulness verifier and human labels on the spot-check sample."""

from __future__ import annotations

import csv
import json
from pathlib import Path


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
    Path("evals/calibration/faithfulness_agreement_v1.json").write_text(
        json.dumps(result, indent=1)
    )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

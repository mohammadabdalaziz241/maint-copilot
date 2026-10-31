"""Create the pre-registered dev/sealed split of the English ViDoRe V3 Industrial queries.

Run once. The output file is committed and never regenerated; the SHA-256 of the sealed id list is
recorded in evals/PROTOCOL.md so any later change to the split is detectable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from maint_copilot.data import load_benchmark

SEED = 20261031
DEV_FRACTION = 0.4
OUT = Path("evals/splits/split_v1.json")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; the split is frozen.")
    q = load_benchmark().queries
    strata = q["query_types"].map(lambda v: "+".join(sorted(v)))
    rng = np.random.default_rng(SEED)
    dev: list[int] = []
    for _, ids in q.groupby(strata)["query_id"]:
        ids = sorted(int(i) for i in ids)
        rng.shuffle(ids)
        dev.extend(ids[: round(len(ids) * DEV_FRACTION)])
    dev = sorted(dev)
    sealed = sorted(set(int(i) for i in q["query_id"]) - set(dev))
    digest = hashlib.sha256(json.dumps(sealed).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "version": 1,
                "seed": SEED,
                "language": "english",
                "stratified_by": "sorted query_types",
                "dev": dev,
                "sealed": sealed,
                "sealed_sha256": digest,
            },
            indent=1,
        )
    )
    print(f"dev={len(dev)} sealed={len(sealed)} sealed_sha256={digest}")


if __name__ == "__main__":
    main()

"""Collect the sealed-test outcomes and apply the decision rules in docs/sealed_test_design.md.

Reporting only: reads the committed dev and sealed result files, makes no API calls and changes no
system. Writes evals/results/sealed_summary.json.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

RES = Path("evals/results")

_spec = importlib.util.spec_from_file_location("ja", Path(__file__).with_name("judge_agreement.py"))
_ja = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ja)


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def side(ci: list[float]) -> str:
    return "above zero" if ci[0] > 0 else "below zero" if ci[1] < 0 else "includes zero"


def retrieval_claim(dev: dict, sealed: dict) -> dict:
    d, s = dev["ci95"], sealed["ci95"]
    return {
        "dev": {"diff": dev["mean_diff"], "ci95": d},
        "sealed": {"diff": sealed["mean_diff"], "ci95": s},
        "confirmed": side(d) == side(s),
    }


def main() -> None:
    dh_dev, dh_sealed = load("dense_hybrid_dev.json"), load("dense_hybrid_sealed.json")
    rr_dev, rr_sealed = load("rerank_dev_top20.json"), load("rerank_sealed_top20.json")
    hyb = "hybrid_rrf(bm25,dense_chunk)"
    retrieval = {
        "ndcg@10_sealed": {
            "bm25": dh_sealed["systems"]["bm25"]["ndcg@10"],
            "dense_chunk": dh_sealed["systems"]["dense_chunk"]["ndcg@10"],
            "hybrid": dh_sealed["systems"][hyb]["ndcg@10"],
            "hybrid+rerank": rr_sealed["systems"]["hybrid+rerank"]["ndcg@10"],
        },
        "hybrid_vs_bm25": retrieval_claim(
            dh_dev["vs_bm25_ndcg@10"][hyb], dh_sealed["vs_bm25_ndcg@10"][hyb]
        ),
        "dense_chunk_vs_bm25": retrieval_claim(
            dh_dev["vs_bm25_ndcg@10"]["dense_chunk"], dh_sealed["vs_bm25_ndcg@10"]["dense_chunk"]
        ),
        "rerank_vs_hybrid": retrieval_claim(
            rr_dev["rerank_vs_hybrid_ndcg@10"], rr_sealed["rerank_vs_hybrid_ndcg@10"]
        ),
    }

    cmp_dev, cmp = (
        load("agent_vs_baseline_dev_a1.1.json"),
        load("agent_vs_baseline_sealed_a1.1.json"),
    )
    ci = cmp["paired_diff_agent_r0_minus_baseline"]["ci95"]
    ratio = cmp["answer_cost_usd_per_question"]["ratio"]
    agreement = json.loads(Path("evals/calibration/agreement_v1.json").read_text())
    # calibration strata (40 dev answers): human PASS among 20 judge FAIL; human FAIL among 20 PASS
    human_pass_among_fail = sum(1 for d in agreement["disagreements"] if d["human"] == "PASS")
    human_fail_among_pass = sum(1 for d in agreement["disagreements"] if d["human"] == "FAIL")
    correctness = {
        "pass_rate": cmp["pass_rate"] | {"agent": cmp["pass_rate"]["agent_r0"]},
        "paired_diff": cmp["paired_diff_agent_r0_minus_baseline"],
        "dev_paired_diff": cmp_dev["paired_diff_agent_r0_minus_baseline"],
        "cost_ratio": ratio,
        "replicates": bool(ci[0] > 0 and ratio <= 3),
        "human_standard_estimate": {
            arm: _ja.adjusted_pass_rate(
                cmp["pass_rate"][key], (human_pass_among_fail, 20), (human_fail_among_pass, 20)
            )
            for arm, key in (("baseline", "baseline"), ("agent", "agent_r0"))
        },
        "groups": cmp["groups"],
        "flips": cmp["flips"],
    }
    correctness["pass_rate"].pop("agent_each_repeat", None)
    correctness["pass_rate"].pop("agent_r0", None)

    f_dev = load("faithfulness_agent_vs_baseline_dev_a1.1.json")
    f = load("faithfulness_agent_vs_baseline_sealed_a1.1.json")
    fci = f["supported_share_per_answer"]["ci95"]
    faithfulness = {
        "paired_questions": f["paired_questions"],
        "supported_share_per_answer": f["supported_share_per_answer"],
        "dev_supported_share_per_answer": f_dev["supported_share_per_answer"],
        "unsupported_claims_per_answer": f["unsupported_claims_per_answer"],
        "answers_with_no_unsupported_claim": f["answers_with_no_unsupported_claim"],
        "claim_level": f["claim_level_all_checked_answers"],
        "agent_less_faithful_replicates": bool(fci[1] < 0),
    }

    cost = {
        name: load(file)["summary"]["estimated_cost_usd"]
        for name, file in (
            ("baseline_answers_and_judge", "answers_sealed_hybrid_k10_v3.json"),
            ("agent_answers_and_judge", "agent_sealed_a1.1_r0.json"),
            ("faithfulness_baseline", "faithfulness_sealed_hybrid_k10_v3.json"),
            ("faithfulness_agent", "faithfulness_agent_sealed_a1.1_r0.json"),
        )
    }
    total = sum(v for c in cost.values() for k, v in c.items() if k != "price_source")
    result = {
        "design": "docs/sealed_test_design.md",
        "n_queries": cmp["n_queries"],
        "retrieval": retrieval,
        "correctness": correctness,
        "faithfulness": faithfulness,
        "estimated_cost_usd": {"total": round(total, 2), "by_step": cost},
        "multiple_comparisons": "two primary comparisons, no correction applied",
    }
    (RES / "sealed_summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

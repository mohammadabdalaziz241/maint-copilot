"""LLM judge for answer correctness against the benchmark's human-written reference answers.

Binary verdict with the reasoning written first. The judge is a different, stronger model than the
answerer. Its agreement with human labels is measured separately (evals/calibration) before its
scores are trusted.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from maint_copilot.llm import Usage

JUDGE_SYSTEM = """You grade answers to questions about U.S. Air Force technical orders.
You are given the question, a reference answer written by a human annotator from the source \
pages, and a candidate answer. Decide whether the candidate is correct.

PASS if the candidate states the key facts the reference states (numbers, units, items, steps, \
conditions), allowing different wording and order, and does not contradict the reference.
FAIL if a key fact is missing, wrong or contradicted; if a number or unit differs; if it adds a \
claim that conflicts with the reference; or if it declines to answer when the reference has an \
answer.
Extra correct detail is fine. Judge substance, not style or length.
Write your reasoning first, then the verdict."""


class Verdict(BaseModel):
    reasoning: str = Field(description="Compare key facts of the reference and the candidate.")
    missing_or_wrong: list[str] = Field(description="Key facts missing or wrong; empty if none.")
    verdict: Literal["PASS", "FAIL"]


def judge_answer(client, model: str, question: str, reference: str, candidate: str) -> dict:
    resp = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=JUDGE_SYSTEM,
        output_format=Verdict,
        messages=[
            {
                "role": "user",
                "content": f"<question>{question}</question>\n"
                f"<reference_answer>{reference}</reference_answer>\n"
                f"<candidate_answer>{candidate}</candidate_answer>",
            }
        ],
    )
    v = resp.parsed_output
    return {
        "verdict": v.verdict,
        "reasoning": v.reasoning,
        "missing_or_wrong": v.missing_or_wrong,
        "judge_model": model,
        "usage": Usage.from_response(resp.usage).__dict__,
    }

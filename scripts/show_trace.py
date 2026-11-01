"""Render one agent run as Markdown: question, the retrieve-once baseline, every agent step, the
final answer, its judge verdict and its claim-level faithfulness check.

Reads committed result files and the local corpus (for page sources); makes no API calls. Usage:
    python scripts/show_trace.py --split sealed --query 218 --out docs/example_trace.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from maint_copilot.data import load_benchmark
from maint_copilot.llm import Usage

RES = Path("evals/results")


def records(name: str, key: str = "records") -> dict[int, dict]:
    return {r["query_id"]: r for r in json.loads((RES / name).read_text())[key]}


def quote(text: str) -> str:
    return "\n".join("> " + line if line else ">" for line in text.strip().splitlines())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "sealed"], default="sealed")
    ap.add_argument("--query", type=int, required=True)
    ap.add_argument("--agent-version", default="a1.1")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    q, split, ver = args.query, args.split, args.agent_version
    agent = records(f"agent_{split}_{ver}_r0.json")[q]
    base = records(f"answers_{split}_hybrid_k10_v3.json")[q]
    check = records(f"faithfulness_agent_{split}_{ver}_r0.json", "checks").get(q, {})
    pages = load_benchmark().corpus.set_index("corpus_id")

    def src(pid: int) -> str:
        return (
            f"{pages.loc[pid, 'doc_id'].replace('_', ' ')} p.{pages.loc[pid, 'page_number_in_doc']}"
        )

    def pids(ids: list) -> list[int]:
        return [int(str(p).lstrip("P")) for p in ids]

    out = [
        f"# Example agent run: held-out question {q}"
        if split == "sealed"
        else f"# Dev question {q}",
        "",
        "Rendered by `scripts/show_trace.py` from committed result files. Page text comes from "
        "ViDoRe V3 Industrial (CC BY 4.0; USAF Technical Orders, Distribution Statement A).",
        "",
        f"**Question.** {agent['query']}",
        "",
        f"**Reference answer (human-written).** {agent['reference']}",
        "",
        "## Baseline: retrieve once, answer from the top 10 pages",
        "",
        f"Relevant page among the 10 retrieved: **{'yes' if base['context_has_relevant'] else 'no'}"
        "**."
        f" Judge verdict: **{base['judge']['verdict']}**.",
        "",
        quote(base["answer"]),
        "",
        "## Agent v1",
        "",
        "| Step | Tool | Input | Result |",
        "|---|---|---|---|",
    ]
    for s in agent["trace"]:
        inp = dict(s["input"])
        if s["tool"] == "submit_answer":
            cites = ", ".join(src(p) for p in pids(inp.get("citations", [])))
            text = f"answer citing {cites or 'nothing'}"
            result = s.get("result", "")
            if result.startswith("rejected"):
                result = "**rejected by code**: cited a page not yet read"
        elif s["tool"] == "search_pages":
            text = f"`{inp['query']}`"
            later = [p for p in s["pages"] if p in agent["context_page_ids"]]
            result = f"{len(s['pages'])} pages with snippets, e.g. " + "; ".join(
                src(p) for p in s["pages"][:2]
            )
            if later:
                result += "; read later: " + "; ".join(f"**{src(p)}**" for p in later)
        else:
            text = ", ".join(src(p) for p in pids(inp.get("page_ids", [inp.get("page_id", "")])))
            result = "full text of " + "; ".join(src(p) for p in s["pages"])
        out.append(f"| {s['step']} | `{s['tool']}` | {text} | {result} |")

    usage = Usage(**agent["usage"])
    cost = usage.cost_usd(agent["model"])
    out += [
        "",
        f"{agent['tool_calls']} tool calls, {agent['latency_s']:.1f} s, about ${cost:.4f}."
        " Steps 1 and 2 were issued together in one model turn.",
        "",
        "**Final answer** (cites " + ", ".join(src(p) for p in agent["citations"]) + "). "
        f"Judge verdict: **{agent['judge']['verdict']}**.",
        "",
        quote(agent["answer"]),
        "",
    ]
    if "claims" in check:
        out += [
            "**Faithfulness check** (separate verifier model; each evidence quote is then looked "
            "up on the page in code):",
            "",
            "| Claim | Verdict | Evidence quote found on the page |",
            "|---|---|---|",
        ]
        for c in check["claims"]:
            kind = " (scope note)" if c["kind"] != "fact" else ""
            found = "yes" if c["quote_verified"] else "no"
            out.append(f"| {c['claim']}{kind} | {c['verdict']} | {found} |")
        out.append("")
    text = "\n".join(out)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

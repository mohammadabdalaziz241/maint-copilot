"""Bounded tool-use agent for maintenance questions (plain Anthropic Messages API loop).

Tools: search_pages, read_pages, read_adjacent, submit_answer. The agent may only cite pages it has
read in full; this and the budgets (tool calls, input tokens, USD) are enforced in code, not in the
prompt. Every step is recorded in a trace.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from maint_copilot.answer import ANSWER_SYSTEM, MAX_PAGE_CHARS
from maint_copilot.llm import Usage

AGENT_VERSION = "a1.1"  # a1.1: empty/truncated submissions are returned as errors

AGENT_SYSTEM = """You are a research assistant for U.S. Air Force maintenance technicians. \
You find the answer in a library of 27 technical orders using tools, then submit it.

How to work:
1. Search with the question's key terms. Read the pages whose snippets look relevant; you can \
only cite pages you have read in full.
2. If the pages you read do not answer the question, search again with different wording: \
synonyms, the likely technical order or section name, or one sub-question at a time.
3. Questions that ask to explain, summarize, compare or list often need several pages. Cover \
each part of the question. Procedures and tables often continue on the next page: use \
read_adjacent.
4. Stop when you can answer every part, or when further searching is not finding new evidence. \
Then call submit_answer.

Answer rules for submit_answer:
""" + ANSWER_SYSTEM.split("Rules:", 1)[1]

TOOLS = [
    {
        "name": "search_pages",
        "description": "Hybrid keyword + semantic search over all pages. Returns page ids, their "
        "manual and page number, and a short snippet. Does not count as reading the page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_pages",
        "description": "Return the full text of up to 5 pages by id (e.g. P1234).",
        "input_schema": {
            "type": "object",
            "properties": {
                "page_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                }
            },
            "required": ["page_ids"],
        },
    },
    {
        "name": "read_adjacent",
        "description": "Return the pages just before and/or after a page in the same manual.",
        "input_schema": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string"},
                "before": {"type": "integer", "minimum": 0, "maximum": 2, "default": 0},
                "after": {"type": "integer", "minimum": 0, "maximum": 2, "default": 1},
            },
            "required": ["page_id"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit the final answer. Cite only pages you have read in full.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
                "insufficient_evidence": {"type": "boolean"},
            },
            "required": ["answer", "citations", "insufficient_evidence"],
        },
    },
]


@dataclass
class Budget:
    max_tool_calls: int = 8
    max_input_tokens: int = 60_000
    max_usd: float = 0.05


@dataclass
class AgentState:
    read: set[int] = field(default_factory=set)
    seen_in_search: set[int] = field(default_factory=set)
    tool_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    trace: list[dict] = field(default_factory=list)


def _pid(text: str) -> int | None:
    m = re.fullmatch(r"\s*P?(\d+)\s*", str(text))
    return int(m.group(1)) if m else None


class Library:
    """Page access for the tools: corpus table plus a search function returning page ids."""

    def __init__(self, corpus: pd.DataFrame, search: Callable[[str, int], list[int]]):
        self.pages = corpus.set_index("corpus_id")
        self.search_fn = search
        self.by_doc = {
            doc: g.sort_values("page_number_in_doc")["corpus_id"].astype(int).tolist()
            for doc, g in corpus.groupby("doc_id")
        }

    def header(self, pid: int) -> str:
        row = self.pages.loc[pid]
        return f"P{pid} ({row['doc_id']}, page {row['page_number_in_doc']})"

    def snippet(self, pid: int, query: str, width: int = 400) -> str:
        text = self.pages.loc[pid, "markdown"] or "[no extractable text]"
        terms = {w for w in re.findall(r"[a-z0-9]{3,}", query.lower())}
        best, best_i = -1, 0
        for i in range(0, max(1, len(text) - width), 100):
            score = sum(t in text[i : i + width].lower() for t in terms)
            if score > best:
                best, best_i = score, i
        return " ".join(text[best_i : best_i + width].split())

    def full(self, pid: int) -> str:
        text = self.pages.loc[pid, "markdown"][:MAX_PAGE_CHARS] or "[no extractable text]"
        row = self.pages.loc[pid]
        source = f"{row['doc_id']}, page {row['page_number_in_doc']}"
        return f'<page id="P{pid}" source="{source}">\n{text}\n</page>'

    def adjacent(self, pid: int, before: int, after: int) -> list[int]:
        doc = self.pages.loc[pid, "doc_id"]
        order = self.by_doc[doc]
        i = order.index(pid)
        return [p for p in order[max(0, i - before) : i + after + 1] if p != pid]


def _run_tool(name: str, args: dict, lib: Library, st: AgentState) -> tuple[str, list[int]]:
    if name == "search_pages":
        k = max(1, min(int(args.get("k", 8)), 10))
        ids = lib.search_fn(str(args["query"]), k)
        st.seen_in_search.update(ids)
        lines = [f"{lib.header(p)}: {lib.snippet(p, args['query'])}" for p in ids]
        return "\n".join(lines) or "No results.", ids
    if name == "read_pages":
        ids = [p for p in (_pid(x) for x in args.get("page_ids", [])[:5]) if p in lib.pages.index]
        if not ids:
            return "Error: no valid page ids. Use ids like P1234 from search results.", []
        st.read.update(ids)
        requested = args.get("page_ids", [])
        note = (
            f"\n\n[Only the first 5 of {len(requested)} requested pages were returned.]"
            if len(requested) > 5
            else ""
        )
        return "\n\n".join(lib.full(p) for p in ids) + note, ids
    if name == "read_adjacent":
        pid = _pid(args.get("page_id", ""))
        if pid is None or pid not in lib.pages.index:
            return "Error: unknown page id.", []
        before = max(0, min(int(args.get("before", 0)), 2))
        after = max(0, min(int(args.get("after", 1)), 2))
        ids = lib.adjacent(pid, before, after)
        st.read.update(ids)
        return "\n\n".join(lib.full(p) for p in ids) or "No adjacent pages.", ids
    return f"Error: unknown tool {name}.", []


def run_agent(
    client,
    model: str,
    question: str,
    lib: Library,
    budget: Budget | None = None,
    price: Callable[[Usage], float | None] | None = None,
) -> dict:
    budget = budget or Budget()
    st = AgentState()
    messages: list[dict] = [{"role": "user", "content": question}]
    t_start = time.perf_counter()
    final, stop_cause, forced = None, None, False

    for _turn in range(budget.max_tool_calls + 4):  # hard ceiling on model calls
        over = (
            st.tool_calls >= budget.max_tool_calls
            or st.usage.input_tokens >= budget.max_input_tokens
            or (price is not None and (price(st.usage) or 0) >= budget.max_usd)
        )
        if over and not forced:
            forced = True
            stop_cause = stop_cause or "budget"
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=model,
            max_tokens=4000,
            system=AGENT_SYSTEM,
            tools=TOOLS,
            tool_choice={"type": "tool", "name": "submit_answer"} if forced else {"type": "auto"},
            thinking={"type": "disabled"},
            messages=messages,
        )
        u = Usage.from_response(resp.usage)
        for k in st.usage.__dict__:
            setattr(st.usage, k, getattr(st.usage, k) + getattr(u, k))
        messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:  # model answered in prose: ask for the tool
            st.trace.append({"step": len(st.trace) + 1, "tool": None, "note": "no tool call"})
            messages.append({"role": "user", "content": "Use submit_answer to give your answer."})
            forced = True
            continue
        results = []
        for call in calls:
            step = {
                "step": len(st.trace) + 1,
                "tool": call.name,
                "input": call.input,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "latency_s": round(time.perf_counter() - t0, 3),
            }
            if call.name == "submit_answer":
                answer_text = str(call.input.get("answer", "")).strip()
                if not answer_text and _turn < budget.max_tool_calls + 3:
                    hint = (
                        " Your output was cut off; keep it shorter."
                        if (resp.stop_reason == "max_tokens")
                        else ""
                    )
                    step["result"] = "rejected: empty answer"
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "is_error": True,
                            "content": "submit_answer needs a non-empty answer." + hint,
                        }
                    )
                    st.trace.append(step)
                    forced = True
                    continue
                cited = [_pid(c) for c in call.input.get("citations", [])]
                bad = [c for c in cited if c is None or c not in st.read]
                if bad and not forced:
                    msg = (
                        f"Rejected: cited pages {[f'P{c}' for c in bad]} were not read in full. "
                        f"Read them first, or cite only pages you read: "
                        f"{sorted(f'P{p}' for p in st.read)}."
                    )
                    step["result"] = "rejected: unread citations"
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": msg,
                            "is_error": True,
                        }
                    )
                    st.tool_calls += 1
                    st.trace.append(step)
                    continue
                final = {
                    "answer": str(call.input.get("answer", "")),
                    "citations": [c for c in dict.fromkeys(cited) if c in st.read],
                    "dropped_citations": [c for c in cited if c not in st.read],
                    "insufficient_evidence": bool(call.input.get("insufficient_evidence", False)),
                }
                step["result"] = "submitted"
                st.trace.append(step)
                break
            text, ids = _run_tool(call.name, call.input, lib, st)
            st.tool_calls += 1
            step["pages"] = ids
            st.trace.append(step)
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": text})
        if final is not None:
            stop_cause = stop_cause or "submitted"
            break
        messages.append({"role": "user", "content": results})

    if final is None:
        stop_cause = "no answer"
        final = {
            "answer": "",
            "citations": [],
            "dropped_citations": [],
            "insufficient_evidence": False,
        }
    return {
        **final,
        "error": None if final["answer"] else "agent produced no answer",
        "context_page_ids": sorted(st.read),
        "pages_seen_in_search": sorted(st.seen_in_search),
        "tool_calls": st.tool_calls,
        "stop_cause": stop_cause,
        "agent_version": AGENT_VERSION,
        "model": model,
        "usage": st.usage.__dict__,
        "latency_s": round(time.perf_counter() - t_start, 3),
        "trace": st.trace,
    }


def trace_to_json(result: dict) -> str:
    return json.dumps(result["trace"], indent=1)


__all__: Sequence[str] = ["Budget", "Library", "run_agent", "AGENT_VERSION", "TOOLS"]

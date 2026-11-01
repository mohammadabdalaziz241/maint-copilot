"""Send the exported prompts to the local vLLM server and record answers, tokens and latency.

Plain HTTP to the OpenAI-compatible endpoint (no SDK), JSON-schema constrained output, resumable.
Usage:
    python answer_open_weight.py --out results/<name>.jsonl [--limit 3] [--concurrency 1]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

URL = "http://localhost:8000/v1"


def environment() -> dict:
    env = {"served_model": requests.get(f"{URL}/models", timeout=30).json()["data"][0].get("root")}
    try:
        env["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        import vllm  # noqa: PLC0415

        env["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        env["env_error"] = str(e)
    return env


def ask(item: dict, max_tokens: int) -> dict:
    body = {
        "model": "local",
        "messages": [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["user"]},
        ],
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "GroundedAnswer", "schema": item["schema"]},
        },
    }
    t0 = time.perf_counter()
    rec = {"query_id": item["query_id"], "page_ids": item["page_ids"]}
    try:
        r = requests.post(f"{URL}/chat/completions", json=body, timeout=600)
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        msg = choice["message"]
        rec["finish_reason"] = choice.get("finish_reason")
        rec["raw"] = msg.get("content") or ""
        rec["reasoning_chars"] = len(msg.get("reasoning_content") or msg.get("reasoning") or "")
        usage = data.get("usage") or {}
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        rec["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        )
        parsed = json.loads(rec["raw"])
        offered = {f"P{p}" for p in item["page_ids"]}
        cites = parsed.get("citations", [])
        ok = (
            isinstance(parsed.get("answer"), str)
            and isinstance(cites, list)
            and isinstance(parsed.get("insufficient_evidence"), bool)
        )
        rec.update(
            {
                "answer": parsed.get("answer", ""),
                "citations": [c for c in cites if c in offered],
                "invalid_citations": [c for c in cites if c not in offered],
                "insufficient_evidence": bool(parsed.get("insufficient_evidence", False)),
                "error": None if ok else "schema mismatch",
            }
        )
    except Exception as e:  # noqa: BLE001
        rec.setdefault("latency_s", round(time.perf_counter() - t0, 3))
        rec.update(
            {
                "answer": "",
                "citations": [],
                "invalid_citations": [],
                "insufficient_evidence": False,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }
        )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--inputs", default=str(here / "inputs_dev_hybrid_k10.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=4000)
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.inputs, encoding="utf-8")]
    items = items[: args.limit or None]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["query_id"] for line in out.open()} if out.exists() else set()
    todo = [it for it in items if it["query_id"] not in done]
    env = environment()
    print(f"{len(todo)} to run ({len(done)} already done) | {env}")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool, out.open("a", encoding="utf-8") as f:
        for rec in pool.map(lambda it: ask(it, args.max_tokens), todo):
            rec.update({"concurrency": args.concurrency, **env})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            flag = "ok" if not rec["error"] else rec["error"][:60]
            print(
                f"q{rec['query_id']}: {rec['latency_s']}s {rec.get('completion_tokens')} tok {flag}"
            )
    wall = time.perf_counter() - t0

    recs = [json.loads(line) for line in out.open()]
    lat = [r["latency_s"] for r in recs if not r["error"]]
    summary = {
        "answers": len(recs),
        "valid": sum(1 for r in recs if not r["error"]),
        "errors": sorted({(r["error"] or "")[:60] for r in recs if r["error"]}),
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        "latency_p95_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2) if lat else None,
        "wall_s_this_session": round(wall, 1),
        "questions_per_min_this_session": round(len(todo) / wall * 60, 2) if todo else None,
        "concurrency": args.concurrency,
    }
    print(json.dumps(summary, indent=1))
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

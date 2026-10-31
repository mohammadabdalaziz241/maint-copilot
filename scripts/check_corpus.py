"""Check whether the ViDoRe V3 Industrial corpus can support the Maintenance Copilot.

Reads the locally downloaded Hugging Face snapshot (no network) and writes
docs/corpus_check.md with: licence from the dataset card, file inventory,
schema and row counts per split, query-type mix, and how much of the corpus
and query set touches rotating machinery (bearings, vibration, pumps, ...).

Usage:
    python scripts/check_corpus.py [data/vidore_v3_industrial]
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TERMS: dict[str, str] = {
    "bearing": r"\bbearings?\b",
    "vibration": r"\bvibrat\w*",
    "rotating machinery": r"\b(?:rotating|rotor|rotors|shaft|shafts|spindle)\b",
    "pump": r"\bpumps?\b",
    "turbine / engine": r"\b(?:turbines?|engines?|compressors?)\b",
    "gear / gearbox": r"\b(?:gears?|gearbox(?:es)?)\b",
    "motor / generator": r"\b(?:motors?|generators?|alternators?)\b",
    "lubrication": r"\b(?:lubricat\w*|grease|oil)\b",
    "fault / failure": r"\b(?:fault\w*|failures?|defect\w*|wear)\b",
    "inspection / maintenance": r"\b(?:inspect\w*|maintenance|overhaul\w*|repair\w*)\b",
}
FOCUS = ("bearing", "vibration", "rotating machinery")


def text_columns(schema: pa.Schema) -> list[str]:
    keep = []
    for field in schema:
        t = field.type
        if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_integer(t):
            keep.append(field.name)
        elif pa.types.is_list(t) and (
            pa.types.is_string(t.value_type) or pa.types.is_integer(t.value_type)
        ):
            keep.append(field.name)
    return keep


def load_split(root: Path, split: str):
    files = sorted((root / split).rglob("*.parquet")) if (root / split).exists() else []
    if not files:
        files = sorted(
            p for p in root.rglob("*.parquet") if split in p.parts or p.name.startswith(split)
        )
    if not files:
        return None, [], None
    schema = pq.read_schema(files[0])
    cols = text_columns(schema)
    tables = [pq.read_table(f, columns=cols) for f in files]
    return pa.concat_tables(tables).to_pandas(), files, schema


def licence_from_card(root: Path) -> str:
    card = root / "README.md"
    if not card.exists():
        return "README.md (dataset card) not found in the download"
    text = card.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^license:\s*(.+)$", text, flags=re.M)
    lines = [
        f"`license:` field: **{m.group(1).strip()}**"
        if m
        else "no `license:` field in the card front matter"
    ]
    for hit in re.findall(r"[^\n]*licen[sc]e[^\n]*", text, flags=re.I)[:8]:
        lines.append(f"> {hit.strip()[:300]}")
    return "\n".join(lines)


def coverage(series, label: str, out: list[str]) -> dict[str, int]:
    s = series.fillna("").astype(str)
    counts = {}
    out.append(f"\n| Term group | {label} matching | Share |\n|---|---|---|")
    for name, pat in TERMS.items():
        n = int(s.str.contains(pat, flags=re.I, regex=True).sum())
        counts[name] = n
        out.append(f"| {name} | {n} of {len(s)} | {n / max(len(s), 1):.1%} |")
    return counts


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/vidore_v3_industrial")
    if not root.exists():
        sys.exit(f"{root} not found. Download it first (see README).")
    out: list[str] = ["# ViDoRe V3 Industrial: corpus check", ""]
    out += ["## Licence (from the dataset card)", "", licence_from_card(root), ""]

    out += ["## Files", "", "| File | Size (MB) |", "|---|---|"]
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out.append(f"| `{p.relative_to(root)}` | {p.stat().st_size / 1e6:.1f} |")

    frames = {}
    for split in ("corpus", "queries", "qrels"):
        df, files, schema = load_split(root, split)
        frames[split] = df
        out += ["", f"## Split: {split}", ""]
        if df is None:
            out.append("not found")
            continue
        out.append(f"{len(df)} rows from {len(files)} file(s).")
        out.append("")
        out.append("Schema: " + ", ".join(f"`{f.name}`: {f.type}" for f in schema))

    q = frames.get("queries")
    if q is not None:
        for col in q.columns:
            if col.lower() in {
                "query_type",
                "query_types",
                "type",
                "query_format",
                "format",
                "language",
                "lang",
            }:
                vals = (
                    q[col].explode()
                    if q[col].map(lambda v: isinstance(v, (list, tuple))).any()
                    else q[col]
                )
                out += ["", f"### queries.{col}", "", "| Value | Count |", "|---|---|"]
                for k, v in Counter(vals.dropna().astype(str)).most_common(15):
                    out.append(f"| {k} | {v} |")
        if "language" in q.columns:
            langs = q["language"].astype(str).str.lower()
            english = q[langs.str.startswith("en")]
            out += ["", f"Coverage below uses the {len(english)} English queries only."]
            q = english
        qtext = next((c for c in ("query", "question", "text") if c in q.columns), None)
        if qtext:
            out += ["", "## Rotating-machinery coverage: queries"]
            coverage(q[qtext], "queries", out)
            pat = "|".join(TERMS[t] for t in FOCUS)
            hits = q[q[qtext].fillna("").str.contains(pat, flags=re.I, regex=True)]
            out += ["", f"### Queries mentioning {', '.join(FOCUS)} ({len(hits)})", ""]
            ans = "answer" if "answer" in q.columns else None
            for _, r in hits.head(25).iterrows():
                a = f" — *{str(r[ans])[:160]}*" if ans else ""
                out.append(f"- {str(r[qtext])[:220]}{a}")

    c = frames.get("corpus")
    if c is not None:
        ctext = next(
            (
                col
                for col in ("markdown", "text", "ocr", "content", "page_text")
                if col in c.columns
            ),
            None,
        )
        doc = next(
            (
                col
                for col in ("doc_id", "document_id", "doc_name", "filename", "source", "document")
                if col in c.columns
            ),
            None,
        )
        if doc:
            out += [
                "",
                f"## Documents (by `{doc}`)",
                "",
                f"{c[doc].nunique()} distinct documents.",
                "",
                "| Document | Pages |",
                "|---|---|",
            ]
            for k, v in c[doc].value_counts().head(40).items():
                out.append(f"| {str(k)[:120]} | {v} |")
        if ctext:
            out += ["", f"## Rotating-machinery coverage: corpus pages (`{ctext}`)"]
            coverage(c[ctext], "pages", out)
            if doc:
                pat = "|".join(TERMS[t] for t in FOCUS)
                mask = c[ctext].fillna("").str.contains(pat, flags=re.I, regex=True)
                out += [
                    "",
                    "### Documents with the most bearing/vibration/rotating-machinery pages",
                    "",
                    "| Document | Matching pages |",
                    "|---|---|",
                ]
                for k, v in c[mask][doc].value_counts().head(25).items():
                    out.append(f"| {str(k)[:120]} | {v} |")
        else:
            out += [
                "",
                "No text column in the corpus split; page text must come from our own parsing "
                "(Docling/OCR).",
            ]

    Path("docs").mkdir(exist_ok=True)
    Path("docs/corpus_check.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

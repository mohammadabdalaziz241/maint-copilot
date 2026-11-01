#!/usr/bin/env bash
# List candidate open-weight models from the Hugging Face API (exact ids, licence, gating, size).
set -euo pipefail
python3 - <<'PY'
import json, urllib.parse, urllib.request

def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)

searches = [("openai", "gpt-oss"), ("google", "gemma-4"), ("Qwen", "Qwen3.6"), ("RedHatAI", "gemma-4"),
            ("RedHatAI", "Qwen3.6"), ("Qwen", "Qwen3.6 FP8"), ("google", "gemma-4 fp8")]
seen = set()
for author, term in searches:
    q = urllib.parse.quote(term)
    try:
        hits = get(f"https://huggingface.co/api/models?author={author}&search={q}&limit=30")
    except Exception as e:  # noqa: BLE001
        print(f"search {author}/{term} failed: {e}")
        continue
    for h in hits:
        mid = h["id"]
        if mid in seen:
            continue
        seen.add(mid)
        try:
            info = get(f"https://huggingface.co/api/models/{mid}?blobs=true")
        except Exception as e:  # noqa: BLE001
            print(f"{mid}: info failed ({e})")
            continue
        lic = next((t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("license:")), "?")
        size = sum((s.get("size") or 0) for s in info.get("siblings", [])
                   if s["rfilename"].endswith((".safetensors", ".bin", ".gguf")))
        params = (info.get("safetensors") or {}).get("total")
        print(f"{mid:55s} licence={lic:12s} gated={str(info.get('gated')):6s} "
              f"weights={size / 2**30:6.1f} GiB params={params} updated={info.get('lastModified', '')[:10]}")
PY

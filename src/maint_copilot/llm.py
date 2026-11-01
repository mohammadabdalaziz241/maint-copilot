"""Anthropic client, token accounting and cost estimates.

Prices are configuration, not facts baked into results: every run stores raw token counts, and
cost is recomputed from PRICES_USD_PER_MTOK, which records where and when each price was checked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import anthropic

# USD per million tokens: (input, output). Verify against https://www.anthropic.com/pricing
# before quoting any cost; the values below are recorded with their source.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),  # third-party tracker, 19 Oct 2026; consistent with billing
    "claude-sonnet-5-5": (2.00, 10.00),  # third-party tracker, 19 Oct 2026; consistent with billing
}
PRICE_SOURCE = (
    "third-party trackers (Oct 2026); token-based estimate matched billed spend within 1.3% "
    "on 1 Nov 2026 ($21.02 estimated vs $21.29 billed)"
)


def load_api_key(env_file: Path = Path(".env")) -> str:
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return key
    for line in env_file.read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("ANTHROPIC_API_KEY not set in the environment or .env")


def client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=load_api_key(), max_retries=6)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_response(cls, usage) -> Usage:
        return cls(
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens or 0,
            usage.cache_creation_input_tokens or 0,
        )

    def cost_usd(self, model: str) -> float | None:
        if model not in PRICES_USD_PER_MTOK:
            return None
        pin, pout = PRICES_USD_PER_MTOK[model]
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6

"""Configuration: the client, the models, and the money.

Everything that talks to Nebius Token Factory goes through here.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# --- Model choice -----------------------------------------------------------------------
#
# A $20 budget makes model choice a *cost* decision — and that turns out to serve the argument
# rather than compromise it. The thesis of this design is that reliability comes from the
# scaffolding, not from a bigger base model. So the strongest possible demonstration is also
# the cheapest one: run a SMALL model with the guardrails, and show it beats a model 14x its
# price running without them. Notebook 07 does exactly that.
#
# A whole agent run on the hardest task costs $0.007 here. On GLM-5.2 it would cost $0.10.
#
# See docs/DECISIONS.md D18. Every candidate was smoke-tested on the exact two-round
# tool-calling loop this agent needs (notebooks/00_setup.ipynb) before being chosen — all six
# handled the protocol correctly, so the choice rests on cost, not on plumbing.

AGENT_MODEL = os.getenv("AGENT_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")   # $0.10/$0.30

# The verifier is deliberately a DIFFERENT model family. A model reviewing its own work shows
# self-preference bias (Zheng et al. 2023), which would make the review worthless.
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "openai/gpt-oss-120b")          # $0.15/$0.60

# The heavyweight, for the "would a bigger model just fix this?" comparison in notebook 07.
BIG_MODEL = os.getenv("BIG_MODEL", "zai-org/GLM-5.2")                        # $1.40/$4.40

# --- Prices, USD per 1M tokens ------------------------------------------------------------
# Used by the cost meter so every run prints what it actually cost. Knowing your own numbers
# cold is the difference between saying "it's cheap" and saying "$0.0073".
#
# Note the spread: the large models are 10-25x the small ones per token. That is precisely why
# the design bets on scaffolding rather than on model size — and notebook 07 tests whether that
# bet pays off, by running the *big* model with *no* guardrails against the *small* model with
# all of them.
PRICES = {
    #                                            input, output
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B":    (0.06, 0.24),
    "Qwen/Qwen3-30B-A3B-Instruct-2507":         (0.10, 0.30),   # <- the agent
    "Qwen/Qwen3-32B":                           (0.10, 0.30),
    "google/gemma-3-27b-it":                    (0.10, 0.30),
    "meta-llama/Llama-3.3-70B-Instruct":        (0.13, 0.40),
    "NousResearch/Hermes-4-70B":                (0.13, 0.40),
    "openai/gpt-oss-120b":                      (0.15, 0.60),   # <- the verifier
    "Qwen/Qwen3-235B-A22B-Instruct-2507":       (0.20, 0.60),
    "Qwen/Qwen3.5-397B-A17B":                   (0.60, 3.60),
    "moonshotai/Kimi-K2.7-Code":                (0.95, 4.00),
    "zai-org/GLM-5.2":                          (1.40, 4.40),   # <- 14x the agent, per run
    "deepseek-ai/DeepSeek-V4-Pro":              (1.75, 3.50),
}
DEFAULT_PRICE = (0.20, 0.60)   # unlisted model: assume mid-range rather than free

CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_client: OpenAI | None = None


def client() -> OpenAI:
    """The Token Factory client. OpenAI-compatible, so the `openai` package works unchanged —
    base_url + api_key is the entire integration."""
    global _client
    if _client is None:
        key = os.getenv("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError(
                "NEBIUS_API_KEY is not set. Copy .env.example to .env and paste your key.\n"
                "(You can still run every notebook without a key: set LIVE=False to replay "
                "the committed cache.)"
            )
        _client = OpenAI(base_url=BASE_URL, api_key=key, timeout=180.0, max_retries=0)
    return _client


def price_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Dollars for one call."""
    p_in, p_out = PRICES.get(model, DEFAULT_PRICE)
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000

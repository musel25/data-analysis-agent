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
# DrugDiscoveryBench benchmarked the open models, and Token Factory hosts almost exactly their
# leaderboard: GLM 5.2 (37.8%), Kimi K2.7 Code (35.3%), DeepSeek V4 Pro (31.7%), MiniMax M3
# (23.2%) — DDB Figure 7, p.22. So "which model" did not have to be a vibe.
#
# And then I deliberately did not pick the winner.
#
# The thesis of this design is that reliability comes from the scaffolding, not from a bigger
# base model. Picking the strongest available model would have made that thesis untestable —
# any success could be credited to the model. So the agent runs on Qwen3-30B-A3B, which is 14x
# CHEAPER than GLM-5.2 per token, and the claim becomes falsifiable: if the scaffolding is doing
# the work, a small model with it should hold up.
#
# A whole agent run on the hardest task costs $0.007 here. On GLM-5.2 it would cost ~$0.10.
#
# What this does NOT yet show: that the small+scaffolded agent beats a BIG model with NO
# scaffolding. That is the direct test of the thesis and I have not run it — see DECISIONS D18
# ("what I would spend the next $20 on"). Saying so is cheaper than implying otherwise.

AGENT_MODEL = os.getenv("AGENT_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")   # $0.10/$0.30

# The verifier is deliberately a DIFFERENT model family. A model reviewing its own work shows
# self-preference bias (Zheng et al. 2023), which would make the review worthless.
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "openai/gpt-oss-120b")          # $0.15/$0.60

# The heavyweight. Not used by default — it is here so the big-model-no-guardrails baseline is
# one flag away: `uv run python -m evals.run_eval --config no_guardrails --model zai-org/GLM-5.2`
BIG_MODEL = os.getenv("BIG_MODEL", "zai-org/GLM-5.2")                        # $1.40/$4.40

# --- Prices, USD per 1M tokens ------------------------------------------------------------
# Used by the cost meter so every run prints what it actually cost. Knowing your own numbers
# cold is the difference between saying "it's cheap" and saying "$0.0073".
#
# Note the spread: the large models are 10-25x the small ones per token. That spread is the whole
# reason the design bets on scaffolding rather than on model size — and it is why the agent runs
# on the cheapest capable model rather than the best one.
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

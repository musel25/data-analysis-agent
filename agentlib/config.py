"""Configuration: the provider, the models, and the money.

Everything that talks to an LLM goes through here — and there is exactly one thing to configure,
which is the whole point of D01.

    THE PROVIDER IS A BASE_URL AND A KEY. THAT IS THE ENTIRE INTEGRATION.

No framework, no adapter layer, no provider abstraction. The `openai` package speaks to anything
OpenAI-compatible, and essentially everything is. Swap the two env vars and the agent — the loop,
the ledgers, the gates, the cache, the eval harness — does not know or care:

    # Nebius Token Factory (what the 4,480-run evaluation was measured on)
    LLM_BASE_URL=https://api.tokenfactory.nebius.com/v1/
    LLM_API_KEY=...
    AGENT_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507

    # your own GPU, self-hosted on Modal (see infra/modal_vllm.py — one command to deploy)
    LLM_BASE_URL=https://<you>--research-agent-vllm-serve.modal.run/v1
    LLM_API_KEY=...
    AGENT_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507

    # a free tier, if you just want to run the notebooks
    LLM_BASE_URL=https://api.groq.com/openai/v1        AGENT_MODEL=qwen/qwen3-32b
    LLM_BASE_URL=https://openrouter.ai/api/v1          AGENT_MODEL=qwen/qwen3-30b-a3b:free
    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

⚠️ A MODEL SWAP INVALIDATES THE CACHE. The cache key hashes the model name, so pointing at a
different model means every request is a miss and the committed responses cannot replay. Keep
`AGENT_MODEL` on Qwen3-30B-A3B if you want the notebooks to replay offline for free; change it only
when you actually want to spend money on new inference.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# `NEBIUS_*` kept as a fallback so an existing .env keeps working unchanged.
BASE_URL = os.getenv("LLM_BASE_URL") or "https://api.tokenfactory.nebius.com/v1/"


def api_key() -> str | None:
    return os.getenv("LLM_API_KEY") or os.getenv("NEBIUS_API_KEY")

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

    # Google Gemini, via the OpenAI-compatible endpoint. FREE-TIER USE COSTS $0 — these are the
    # PAID-tier list prices, so the meter tells the truth the moment you upgrade rather than
    # printing a comforting zero.
    #
    # Note what these numbers say: gemini-3.5-flash at $9.00/1M output is more expensive than
    # GLM-5.2 ($4.40), the model this design deliberately refused as the "heavyweight". A Gemini
    # swap is a free-tier convenience, NOT a cheaper agent. Hence the agent runs on flash-lite.
    # And the cheap Gemini is NOT a drop-in: gemini-3.1-flash-lite could not drive the agent loop
    # at all (t1_sentinel: 17 steps, wandered off the question, submitted a null value), while
    # gemini-3.5-flash lands the exact truth in 9. Under Gemini you pay for the strong model or
    # you get nothing — which is itself a data point for the scaffolding-vs-model-size thesis:
    # the scaffolding did NOT rescue the weak Gemini the way it carries Qwen3-30B.
    "gemini-3.5-flash":                         (1.50, 9.00),   # <- the agent, under Gemini
    "gemini-3-flash-preview":                   (0.50, 3.00),
    "gemini-3.1-flash-lite":                    (0.25, 1.50),   # <- the verifier, under Gemini
    "gemini-2.5-flash-lite":                    (0.10, 0.40),
}
DEFAULT_PRICE = (0.20, 0.60)   # unlisted model: assume mid-range rather than free

# --- Self-hosted endpoints have no per-token price ----------------------------------------
#
# When the model runs on a GPU you rented (infra/modal_vllm.py), you are billed by the SECOND for
# the GPU — not by the token. Multiplying tokens by a made-up $/1M rate would print a confident
# dollar figure that corresponds to nothing, and inventing a number that looks measured is the
# single thing this project exists not to do. (DEFAULT_PRICE would have quoted $0.0113 for the
# demo run below. That number is fiction.)
#
# So: per-token cost is zero here, and the UI says "self-hosted" and quotes the GPU-hour instead.
SELF_HOSTED_HOSTS = ("modal.run", "localhost", "127.0.0.1", "0.0.0.0")
SELF_HOSTED = any(h in BASE_URL for h in SELF_HOSTED_HOSTS)
GPU_HOURLY_USD = 1.10          # Modal A10G, the GPU infra/modal_vllm.py asks for

CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_client: OpenAI | None = None


def client() -> OpenAI:
    """The LLM client. OpenAI-compatible, so the `openai` package works unchanged against Nebius,
    a self-hosted vLLM on your own GPU, or anyone's free tier — `base_url` + `api_key` is the
    entire integration. That is not a boast; it is the reason D01 refuses a framework."""
    global _client
    if _client is None:
        key = api_key()
        if not key:
            raise RuntimeError(
                "No API key. Set LLM_API_KEY (or NEBIUS_API_KEY) in .env.\n"
                "\n"
                "  You almost certainly do not need one. 7 of the 8 notebooks replay from the\n"
                "  committed cache with no key and no network:\n"
                "\n"
                "      from agentlib import set_live\n"
                "      set_live(False)\n"
                "\n"
                "  A key is only needed to (a) run notebook 01, which deliberately makes one RAW\n"
                "  uncached API call to show the unwrapped protocol, or (b) ask the agent something\n"
                "  new. Any OpenAI-compatible endpoint works — set LLM_BASE_URL. See the module\n"
                "  docstring, or `infra/modal_vllm.py` to serve the exact model on your own GPU."
            )
        _client = OpenAI(base_url=BASE_URL, api_key=key, timeout=600.0, max_retries=0)
    return _client


def price_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Dollars for one call. Zero on a self-hosted endpoint — see SELF_HOSTED above: there the
    meter is the GPU clock, not the token count, and a per-token figure would be invented."""
    if SELF_HOSTED:
        return 0.0
    p_in, p_out = PRICES.get(model, DEFAULT_PRICE)
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000

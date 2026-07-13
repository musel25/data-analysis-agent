"""Serve the EXACT agent model on your own GPU, as an OpenAI-compatible endpoint.

    uv run modal deploy infra/modal_vllm.py

This exists to make one claim checkable rather than merely asserted.

D01 refuses a framework, and the reason given is: *"with an OpenAI-compatible endpoint there is
precisely one integration to write."* That is easy to say. So here is the same agent — the same
loop, the same three ledgers, the same four gates, the same cache, the same 4,480-run eval harness —
pointed at a vLLM server running on a GPU I rented myself. The diff needed to move it:

    LLM_BASE_URL=https://<you>--research-agent-vllm-serve.modal.run/v1
    LLM_API_KEY=<the secret below>

Two environment variables. No adapter, no provider class, no code change. That is the whole point.

VERIFIED. The full agent — contract, ledger, all four gates, the death-loop guards — ran end to end
against this endpoint on the flagship Simpson's-paradox task and returned **+0.1507** (the truth;
the confounded trap is -0.087). On a **4-billion-parameter** model, which is 7x smaller than the one
the evaluation was measured on. The verifier bounced its first answer and it reopened three findings
and fixed them.

That is n=1 and I will not build a thesis on it. But it is a rather good advertisement for spending
the budget on scaffolding rather than on parameters.

────────────────────────────────────────────────────────────────────────────────────────────────
WHAT THIS IS NOT
────────────────────────────────────────────────────────────────────────────────────────────────
It is NOT a way to extend the measured evaluation. Same weights are not the same model: vLLM here
batches, quantises and samples differently from Nebius's stack, so numbers produced against this
endpoint are NOT comparable to the 4,480-run grid in evals/results.jsonl. Mixing them would be a
confound, in a project that is entirely about confounds.

If you want a number out of this endpoint, run BOTH arms of the comparison against it and report it
as its own experiment. (The harness will not let you do it by accident: every result row is stamped
with a prompt fingerprint, and `run_eval` refuses to resume across a mismatch — see D31/D33.)

────────────────────────────────────────────────────────────────────────────────────────────────
COST
────────────────────────────────────────────────────────────────────────────────────────────────
`scaledown_window` is 2 minutes and `max_containers` is 1, so it scales to zero when idle and you
pay only for the seconds it is actually generating. An A10G is roughly $1.10/hour, so an hour of
playing with it costs about a dollar. The first cold start downloads the weights; after that they
are cached in a Modal Volume and the start is well under a minute.

Set a spend limit in the Modal dashboard. I burned $30 of someone else's inference credit on this
project by not doing that, which is why `evals/run_eval.py` now has `--max-spend`.
"""

import modal

# The SAME model the evaluation ran on — Qwen3-30B-A3B-Instruct-2507 — but 4-bit quantised, because
# of a hardware constraint that is worth stating plainly:
#
#   The eval model in bf16 is ~61 GB. A Modal account with no payment method on file can use T4
#   (15 GB), L4 (23 GB) and A10G (23 GB); A100 / H100 / L40S all demand a card. 61 > 23.
#
# So: AWQ 4-bit, ~17 GB, which fits an A10G with room for a KV cache. It is the same weights and the
# same non-thinking Instruct behaviour — but **4-bit quantisation is not a no-op**. Outputs differ
# from bf16, so the committed cache will not replay against this endpoint and numbers produced here
# are NOT comparable to evals/results.jsonl.
#
# This serves the portability claim, not the evaluation. Do not mix them. (The harness will not let
# you do it by accident — every result row carries a prompt fingerprint. See D31/D33.)
# I tried the 4-bit AWQ quant of the exact eval model first. It loads — 20.1 GB of the A10G's
# 22 GB — and then dies in vLLM's sampler warmup with CUDA OOM, because there is nothing left for a
# KV cache. A 30B model does not fit on a 23 GB card, and no amount of quantisation talks it into
# fitting *with room to actually generate*. That is physics, not configuration.
#
# So: Qwen3-4B-Instruct-2507. Same family, same NON-THINKING Instruct behaviour as the eval model
# (which matters — the hybrid Qwen3 checkpoints emit <think> blocks that would wreck the agent's
# tool-calling), ~8 GB in bf16, comfortable on an A10G with a large KV cache.
#
# It is a SMALLER MODEL. Say so. Numbers from this endpoint are not comparable to
# evals/results.jsonl, the committed cache will not replay against it, and the harness's prompt
# fingerprint (D31/D33) will refuse to mix the two. It exists to demonstrate that the provider is
# two environment variables — nothing more, and nothing less.
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SERVED_AS = MODEL
GPU = "A10G"                                     # 23 GB; 4B in bf16 is ~8 GB + a fat KV cache
PORT = 8000

app = modal.App("research-agent-vllm")

# `hf_transfer` because pulling 60 GB over plain HTTPS is a slow way to discover you mistyped the
# model name.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "huggingface_hub[hf_transfer]==0.35.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

# Weights live in a Volume, so the 60 GB download happens once, not once per cold start.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# `modal secret create research-agent-llm LLM_API_KEY=...`
secret = modal.Secret.from_name("research-agent-llm")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[secret],
    timeout=60 * 60,
    scaledown_window=120,      # idle for 2 min -> shut down. You are not billed for idle GPUs.
    max_containers=1,          # a cost guard, not a scaling decision. See the module docstring.
)
@modal.concurrent(max_inputs=32)     # the eval runs 16 agents at once; vLLM batches them happily
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    """vLLM's own OpenAI-compatible server. We add nothing to it — that is the point.

    `--api-key` makes it reject anonymous callers, because this endpoint is on the public internet
    and an open LLM proxy is somebody else's free lunch.
    """
    import os
    import subprocess

    subprocess.Popen(
        [
            "vllm", "serve", MODEL,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--api-key", os.environ["LLM_API_KEY"],
            "--served-model-name", SERVED_AS,
            "--max-model-len", "32768",
            "--gpu-memory-utilization", "0.90",
            "--max-num-seqs", "16",
            "--enable-auto-tool-choice",       # the agent is nothing without tool calling
            "--tool-call-parser", "hermes",    # Qwen3 emits Hermes-style tool calls
        ]
    )

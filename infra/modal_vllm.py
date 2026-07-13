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
MODEL = "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
SERVED_AS = "Qwen/Qwen3-30B-A3B-Instruct-2507"   # so AGENT_MODEL needs no change
GPU = "A10G"                                     # 23 GB; 4-bit 30B is ~17 GB + KV cache
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
            "--max-model-len", "16384",        # 17 GB of weights in 23 GB of VRAM: be modest
            "--gpu-memory-utilization", "0.93",
            "--quantization", "compressed-tensors",
            "--enable-auto-tool-choice",       # the agent is nothing without tool calling
            "--tool-call-parser", "hermes",    # Qwen3 emits Hermes-style tool calls
        ]
    )

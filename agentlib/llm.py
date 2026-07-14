"""One function: `llm(messages, tools) -> response`.

Three things wrap the raw API call, and each earns its place:

  RETRY   — networks fail. Exponential backoff, and nothing else.
  CACHE   — every response is stored on disk, keyed by a hash of the request. With LIVE=False
            the notebooks replay from cache: they run offline, for free, deterministically, and
            a reader with no API key still sees everything work. It is also demo insurance.
            And (notebook 07) it is how you write cheap tests for a non-deterministic system.
  METER   — every call's cost is accumulated. A design that talks about budgets should be able
            to tell you what it spent, to the cent.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field

_LOCK = threading.Lock()   # the eval runs agents concurrently; the meter is shared

from . import config


@dataclass
class Meter:
    """Running token + dollar count. Reset it before a run, print it after."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, model, prompt_tokens, completion_tokens, cached=False):
        with _LOCK:
            self._add(model, prompt_tokens, completion_tokens, cached)

    def _add(self, model, prompt_tokens, completion_tokens, cached=False):
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        if cached:
            self.cache_hits += 1
            return
        cost = config.price_of(model, prompt_tokens, completion_tokens)
        self.cost_usd += cost
        m = self.by_model.setdefault(model, {"calls": 0, "cost": 0.0, "tokens": 0})
        m["calls"] += 1
        m["cost"] += cost
        m["tokens"] += prompt_tokens + completion_tokens

    def reset(self):
        self.__init__()

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def __str__(self):
        paid = self.calls - self.cache_hits
        return (f"{self.calls} calls ({paid} billed, {self.cache_hits} cached) | "
                f"{self.prompt_tokens:,} in + {self.completion_tokens:,} out "
                f"= {self.total_tokens:,} tokens | ${self.cost_usd:.4f}")


METER = Meter()

# Flip to False and every notebook replays from the committed cache: no key, no network, no cost.
LIVE = True


def set_live(live: bool) -> None:
    """Toggle live API calls. Use this rather than assigning to the flag from outside.

    Why a function and not just `agentlib.llm.LIVE = False`: `agentlib/__init__.py` exports the
    *function* `llm`, which shadows the *submodule* `agentlib.llm` on the package. So
    `import agentlib.llm as L; L.LIVE = False` binds an attribute on the function object and
    silently does nothing — the flag never moves, and you find out when the API call you thought
    was cached bills you.

    I only found this because I tested the "runs offline with no key" claim in the README instead
    of assuming it. A footgun that fails silently is worse than one that crashes.
    """
    global LIVE
    LIVE = live


def _key(model, messages, tools, temperature, nonce) -> str:
    blob = json.dumps(
        {"model": model, "messages": messages, "tools": tools, "temperature": temperature,
         "nonce": nonce},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def llm(messages, tools=None, model=None, temperature=0.0, force_tool=None, max_retries=6,
        nonce=0, meter=None):
    """Call the model. Returns the raw `message` object (so you can see `.tool_calls`).

    Deliberately thin: this is the only place the OpenAI SDK is touched, and it stays small
    enough to read in one sitting.

    `nonce` exists because of a bug I shipped and then caught in my own eval.

    The cache is keyed by a hash of the request. With temperature=0 and an identical request,
    attempt #2 and attempt #3 of an eval task hit the cache and return *the identical response*
    — at a cost of $0.0000, which is how I noticed. I thought I was measuring run-to-run
    variance across three samples. I was replaying one sample three times.

    Passing the attempt number as a nonce forces genuinely independent samples. Both papers run
    repeats for exactly this reason (GeneBench-Pro: 10 attempts + bootstrap CIs; DDB: 3 trials),
    and a benchmark that silently reports one run as three is worse than one that honestly
    reports one.
    """
    model = model or config.AGENT_MODEL
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = ({"type": "function", "function": {"name": force_tool}}
                                 if force_tool else "auto")

    cache_key = _key(model, messages, tools, temperature, nonce)
    cache_file = config.CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        payload = json.loads(cache_file.read_text())
        for m in (METER, meter):
            if m is not None:
                m.add(model, payload["usage"]["prompt_tokens"],
                      payload["usage"]["completion_tokens"], cached=True)
        return _Msg(payload["message"], meta={"cached": True, "model": model,
                                              "duration_s": 0.0, "retries": 0, "slept_s": 0.0,
                                              **payload["usage"]})

    if not LIVE:
        raise RuntimeError(
            f"LIVE=False and no cached response for this request ({cache_key}).\n"
            "The cache only covers requests made during the recorded run. Set LIVE=True "
            "(and a NEBIUS_API_KEY) to make new calls."
        )

    last_err = None
    t0 = time.monotonic()
    retries, slept = 0, 0.0
    for attempt in range(max_retries):
        try:
            resp = config.client().chat.completions.create(**kwargs)
            break
        except Exception as e:                       # noqa: BLE001 — retry anything transient
            last_err = e
            if attempt == max_retries - 1:
                raise
            # A 429 is the platform asking us to slow down. It is NOT an agent failure, and the
            # first eval run recorded seven of them as crashes — five of those against the FULL
            # agent, which quietly cost it ~1.8 points of pass rate against its own ablations.
            # Infrastructure noise must never land in the numerator. Back off hard instead.
            rate_limited = "429" in str(e) or "ratelimit" in type(e).__name__.lower()
            pause = (5 * 2 ** attempt) if rate_limited else (2 ** attempt)
            retries += 1
            slept += pause
            time.sleep(pause)
    else:                                            # pragma: no cover
        raise last_err

    msg = resp.choices[0].message
    payload = {
        "message": msg.model_dump(exclude_none=True),
        "usage": {"prompt_tokens": resp.usage.prompt_tokens,
                  "completion_tokens": resp.usage.completion_tokens},
    }
    cache_file.write_text(json.dumps(payload, indent=1))
    for m in (METER, meter):
        if m is not None:
            m.add(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    return _Msg(payload["message"], meta={"cached": False, "model": model,
                                          "duration_s": time.monotonic() - t0,
                                          "retries": retries, "slept_s": slept,
                                          **payload["usage"]})


class _Msg:
    """A tiny dict wrapper so cached and live responses behave identically.

    (Without this, a cached response is a dict and a live one is a pydantic object, and every
    call site would need to branch. One small class beats fifty `if isinstance` checks.)
    """

    def __init__(self, d: dict, meta: dict | None = None):
        self._d = d
        self.content = d.get("content") or ""
        self.tool_calls = [_ToolCall(tc) for tc in (d.get("tool_calls") or [])]
        # Per-call telemetry: duration, retries, backoff sleeps, token usage. This is how a
        # 40-second turn stops being a mystery — the loop can SAY where the time went.
        self.meta = meta or {}
        self.reasoning = d.get("reasoning") or ""   # thinking models (Gemini) return this

    def raw(self) -> dict:
        """The exact dict to append to `messages` when replaying the assistant's turn."""
        return self._d


class _ToolCall:
    def __init__(self, d: dict):
        self.id = d["id"]
        self.name = d["function"]["name"]
        self._raw_args = d["function"]["arguments"]

    @property
    def args(self) -> dict:
        """Open models sometimes emit imperfect JSON. Never let that crash the loop — a bad
        tool call should become an *observation* the model can recover from, not a traceback."""
        try:
            return json.loads(self._raw_args)
        except json.JSONDecodeError as e:
            return {"__parse_error__": f"{e}", "__raw__": self._raw_args}

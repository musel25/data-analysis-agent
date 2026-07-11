# A data-analysis agent, built from scratch

> **The claim:** a data-analysis agent is a **while loop that writes code, runs it, and reads the
> results.** Everything else is a guardrail *earned from a failure you watched happen*.

Seven notebooks build the whole thing, one mechanism at a time. Each chapter ends by making the
agent **fail on purpose**, and the next chapter adds the one thing that fixes it. Nothing is
introduced before you have watched its absence hurt.

The finished agent is **~450 lines of Python**. No LangChain, no framework, no vector store, no
multi-agent swarm. It runs on a **$0.10/1M-token open model** and costs **under a cent per
analysis**.

*(Part 2, Option A of a take-home: an agent that gets a prompt + files, explores the data, picks an
approach, writes and runs code, checks intermediate results, and returns a structured answer.)*

---

## The problem, in one picture

Ask the agent the question this clinical trial was actually run to answer:

> *"Does the treatment improve the response rate?"*

| | answer | |
|---|---|---|
| **A competent agent** — writes flawless pandas, prints real numbers | **−0.088** | *"the treatment does not work"* |
| **The truth** | **+0.150** | *the treatment works — in every severity stratum* |

Its arithmetic was perfect. It reached **the opposite conclusion**, because the trial wasn't
randomised — sicker patients were given the drug — and it never let that fact change its method.

That's **Simpson's paradox**, and it isn't my invention. It's the failure both supplied papers were
written about.

---

## What the papers actually say

Both evaluate *exactly this system*: a single LLM agent, in a sandbox, running Python against messy
scientific data. They agree on the diagnosis.

**GeneBench-Pro** (Li & Ho, OpenAI, 2026) — 129 multistage statistical-reasoning problems. Best
frontier model: **31.5%**. On **45.7% of problems it scores zero across ten attempts.**

> *"the agent **notices** the relevant local diagnostic clue but **treats it as a local data
> cleaning issue** rather than as evidence that should **change the downstream statistical
> method**."*

**DrugDiscoveryBench** (Akyürek, Tu et al., Scale AI, 2026) — 82 expert tasks, best agent **51.6%**.
They hand-classified 226 failing runs:

| Failure mode | Share |
|---|---|
| **Domain reasoning** — *"applies an incorrect scientific premise or misinterprets the data it has, even though its inputs and tools are correct"* | **54.0%** |
| Derivation error — right approach, wrong calculation | 18.6% |
| Retrieval — *"failing to read a provided file"* | 16.4% |
| Constraint — violates an explicit instruction | 7.5% |
| Final-answer slip | 3.5% |

### Read that table again.

**Over half of all failures are not coding failures.** The tools worked. The Python was correct.
The model had everything it needed.

It just didn't let what it saw change what it did.

> ## The design thesis
>
> The bottleneck is **not** code generation, **not** knowledge, **not** context length.
> It is **thread-keeping**.
>
> So don't spend the budget making the model smarter. Spend it making it **structurally impossible
> to drop the thread.**

---

## What I built on top of the base model

Three things get dropped: **the question**, **the finding**, and **the number**. Each becomes
explicit state the agent must reconcile before it is allowed to finish.

| | Mechanism | Kills |
|---|---|---|
| **Ledger 1** | **Question Contract** — estimand, *population*, units, constraints, premises. Re-rendered every turn, so it can't decay. | scope drift; *"a valid model applied to the wrong population"* |
| **Ledger 2** | **Findings Ledger** — every noticed problem is an **open obligation**. `submit_answer` is **hard-blocked** while any finding is open. Closing one requires *acting* on it, or *dismissing* it with a written reason. | **the notice–act gap** ← the centrepiece |
| **Ledger 3** | **Evidence grounding** — every number in the answer must appear in the stdout of code that actually ran. A regex. No LLM. | derivation errors; the agent contradicting its own printed output |
| **Gate 4** | **Fresh-context verifier** — a different model family that sees the code and the answer but **never the reasoning** (a reviewer who reads the transcript gets anchored, and rubber-stamps). | *"the last chance to catch the slip is at the final answer. None of the failing models caught this."* |

Plus the unglamorous parts: a persistent-namespace kernel, a deterministic data briefing, head+tail
observation truncation, a live state banner, step budgets with graceful degradation, and two
death-loop guards.

**`submit_answer` is a gated action, not a default.** Four gates, cheapest first — never pay for an
LLM call to catch something a regex would have caught.

---

## The notebooks

| | | The failure that forces the next chapter |
|---|---|---|
| **01** | Hello, model | On a *famous* dataset it recites a **memorised** answer; on an unseen one it **invents** a confident number |
| **02** | Tool calling | A tool call is just the model **emitting JSON**. One round trip — no autonomy |
| **03** | The agent loop | An agent is a `while` loop (~30 lines). But **fixed tools cap the ceiling** |
| **04** | `run_python` | One general tool; it self-debugs from tracebacks — and returns **the opposite of the truth** on real data |
| **05** | Keeping the thread | Contract + Findings Ledger. **The sign flips.** But — where did that number come from? |
| **06** | The gated exit | Grounding + verifier + structured report. But *"it worked once"* is not evidence |
| **07** | Evaluation | The benchmark, the ablations, the failure taxonomy. Every claim becomes a number |

---

## How I'd know it works

**15 tasks**, two datasets, ground truth computed programmatically. Binary, all-or-nothing grading
with pre-specified tolerances — following GeneBench-Pro, whose defence of the strictness is hard to
argue with: *"an agent that executes several intermediate steps correctly but returns the wrong
decision-relevant answer has not successfully automated the analysis."*

Two guards run before every evaluation:

- **Separation** — every trap task records its *plausible-but-wrong* answer and asserts it lies far
  outside the tolerance band. If a lazy analysis can land inside the band, the task grades nothing.
  **This guard rejected three of my own tasks**: a median that's robust to the sentinels; an age
  question independent of its own filter; a proportion unmoved by deduplication. All three looked
  perfectly reasonable. All three measured nothing.
- **Leakage** — the ground truth is a lambda only the grader ever calls, and never appears in a
  prompt.

Because every trap has a *documented* wrong answer, a failure isn't merely wrong — the harness
reports **which** wrong. Landing on the naive answer means the agent fell into the notice–act gap
*specifically*, rather than just fumbling. That's the difference between a benchmark and a
diagnostic.

**Ablations** switch each mechanism off in turn. If one doesn't pay for itself, it gets cut.

---

## Run it

```bash
uv sync
cp .env.example .env        # add a Nebius Token Factory key
uv run jupyter lab notebooks/
```

**No API key?** Every notebook replays from the committed response cache — set `LIVE = False` in
`agentlib/llm.py` and the whole series runs offline, deterministically, for free.

```bash
uv run python data/make_trial.py            # regenerate the data (re-verifies every trap)
uv run python evals/tasks.py                # inspect the benchmark + run its guards
uv run python -m evals.run_eval --ablate    # the full ablation study
```

---

## Where things are

```
docs/DESIGN.md        the proposal — architecture, the papers, the evaluation
docs/DECISIONS.md     every non-obvious choice, its alternatives, and what would change my mind
agentlib/             the agent itself (~450 lines)
notebooks/            the buildup — seven chapters
data/make_trial.py    the synthetic dataset, and the assertions that keep its traps honest
evals/                the benchmark, the grader, the ablation runner
```

Built on **Nebius Token Factory** — `base_url` + `api_key` was the entire integration. Agent:
`Qwen3-30B-A3B`. Verifier and judge: `gpt-oss-120b`, deliberately a *different family*, because a
model grading its own work shows self-preference bias.

---

## The honest part

This is a benchmark I wrote, graded by a harness I wrote, on data I generated, with guardrails
designed against failures I chose to plant. n=15, three runs each. One task flipping is a 7-point
swing.

So the claim is **not** *"this agent is reliable."*

It is: **here is a design, here is the evidence for each piece of it, and here is exactly how you'd
find out if I'm wrong.**

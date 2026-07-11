# A data-analysis agent, built from scratch

> **The claim:** a data-analysis agent is a **while loop that writes code, runs it, and reads the
> results.** Everything else is a guardrail *earned from a failure you watched happen*.

Seven notebooks build the whole thing, one mechanism at a time. Each chapter ends by making the
agent **fail on purpose**, and the next chapter adds the one thing that fixes it. Nothing is
introduced before you have watched its absence hurt.

The finished agent is **~885 lines of Python** (comments and docstrings excluded — most of the file volume is the reasoning behind each choice). No LangChain, no framework, no vector store, no
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

*(That's where I started. The evaluation later corrected me on **which half** of the thread-keeping
problem actually matters — see [the results](#the-results--including-the-one-that-inverted-my-own-thesis).
I've left the original thesis standing rather than quietly rewriting history around the answer.)*

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

### Does it generalise? (the question more runs can't answer)

Every trap in `trial.csv` is one **I** planted while designing the guardrails. Passing it only
proves they work on the failures I *already knew about* — and more runs shrink the *variance* of
that claim, not its **bias**.

So `sales.csv` is a **held-out domain**: e-commerce, not medicine. Revenue exported as text
(`"1,234.56"`), internal QA orders at `999999.99`, refunds left in the file, `-1` for a missing
age, and a Simpson's paradox on **channel × segment** instead of arm × severity.

| domain | pass rate | 95% CI | |
|---|---|---|---|
| `penguins` | 100% | `[100%, 100%]` | clean data |
| `trial` | 82% | `[61%, 97%]` | **designed against** |
| **`sales`** | **89%** | `[74%, 99%]` | **🎯 held-out domain** |

**It does better on the domain it was never tuned for.** That's the most reassuring number here.

*(Writing that dataset also found a hole in my own benchmark: `observe.py` has had a detector for
numeric-columns-stored-as-text since the first commit, and no task ever exercised it. A benchmark
built from one dataset only tests the mechanisms that dataset happens to provoke.)*

### The results — including the one that inverted my own thesis

**2,240 runs · 28 tasks · 3 domains · $8.01.** Each ablation is a *paired* bootstrap against the
full agent (10k hierarchical resamples: tasks, then runs within tasks). **If the CI crosses zero, I
can't distinguish that mechanism from doing nothing — and I say so, instead of reporting a point
estimate.**

| remove this | pass | Δ | 95% CI | verdict |
|---|---|---|---|---|
| **the deterministic data briefing** | 62% | **−26%** | `[−40%, −13%]` | **HURTS** |
| *every guardrail at once* | 69% | −19% | `[−31%, −8%]` | **HURTS** |
| **the Findings Ledger** — *my centrepiece* | 83% | −5% | `[−11%, +1%]` | no detectable effect |
| the Question Contract | 87% | −1% | `[−6%, +4%]` | no detectable effect |
| the grounding gate | 88% | −0% | `[−4%, +4%]` | no detectable effect |
| the fresh-context verifier | 88% | −0% | `[−5%, +5%]` | no detectable effect |

**Removing twenty lines of pandas is the largest effect in the study.** I built this design around
*gates*. The thing carrying it is the *detector*.

> ## A gate is only as good as the detector feeding it.
>
> The papers describe a **notice–act** gap and I read it as a failure to *act* — so I built
> machinery to force action. The ablation says the leverage is on the **notice** side.
>
> Tell the agent what's in the data, deterministically, before it starts, and **it acts on it.**
> It didn't need to be forced. It needed to be *informed*.

**And the bigger eval corrected me about my own centrepiece.** At 15 tasks × 3 runs the Findings
Ledger measured *exactly zero*, and I wrote that it "does not pay for itself." At 28 × 10 it's
**−5% [−11%, +1%]** — the only gate whose interval sits almost entirely on the "it helps" side. It
still misses 95%, so I can't claim it. But *"the ledger does nothing"* was never a finding — it was
**a twenty-point-wide confidence interval reported as a point estimate.**

**What I'd build next isn't another gate. It's more detectors.**

---

## Run it

```bash
uv sync
cp .env.example .env        # add a Nebius Token Factory key
uv run jupyter lab notebooks/
```

**No API key?** Every notebook replays from the committed response cache. Put this at the top and
the whole series runs offline, deterministically, for free:

```python
from agentlib import set_live
set_live(False)          # replay from cache; no key, no network, no cost
```

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
agentlib/             the agent itself (~885 lines of code)
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

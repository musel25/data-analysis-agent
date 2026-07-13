# A data-analysis agent, built from scratch

> **The claim:** a data-analysis agent is a **while loop that writes code, runs it, and reads the
> results.** Everything else is a guardrail *earned from a failure you watched happen*.

Seven notebooks build the whole thing, one mechanism at a time. Each chapter ends by making the
agent **fail on purpose**, and the next chapter adds the one thing that fixes it. Nothing is
introduced before you have watched its absence hurt.

The finished agent is **~1,000 lines of Python** (comments and docstrings excluded — most of the file volume is the reasoning behind each choice). No LangChain, no framework, no vector store, no
multi-agent swarm. It runs on a **$0.10/1M-token open model** and costs **under a cent per
analysis**.

*(Part 2, Option A of a take-home: an agent that gets a prompt + files, explores the data, picks an
approach, writes and runs code, checks intermediate results, and returns a structured answer.)*

*(**Part 1** — the review of the two papers, with the critical assessment — is
**[docs/PART1_REVIEW.md](docs/PART1_REVIEW.md)**. Two of its criticisms are not opinions: they are
things that happened to me while building Part 2 on the papers' own protocols.)*

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

**GeneBench-Pro** (Li & Ho, OpenAI, 2026) — 129 multistage statistical-reasoning problems. The best
configuration in the paper reaches **31.5%**. The best *mainline* model scores **28.7%** — and on
**45.7% of problems it scores zero across all ten attempts.**

> *"the agent **notices** the relevant local diagnostic clue but **treats it as a local data
> cleaning issue** rather than as evidence that should **change the downstream statistical method
> and QC pipeline**."*

**DrugDiscoveryBench** (Akyürek, Tu et al., Scale AI & Phylo, 2026) — 82 expert tasks, best agent
**51.6%**. They classified 226 failing runs:

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

**28 tasks**, three domains, ground truth computed programmatically. Binary, all-or-nothing grading
with pre-specified tolerances — following GeneBench-Pro, whose defence of the strictness is hard to
argue with: *"an agent that executes several intermediate steps correctly but returns the wrong
decision-relevant answer has not successfully automated the analysis."*

Four guards run before every evaluation, and they raise rather than warn:

- **Separation** — every trap task records its *plausible-but-wrong* answer and asserts it lies far
  outside the tolerance band. If a lazy analysis can land inside the band, the task grades nothing.
  **This guard rejected two of my own tasks** (receipts in `evals/tasks.py`): a median that's robust
  to the sentinels, and an age question independent of its own filter. Both looked perfectly
  reasonable. Both measured nothing.
- **Leakage** — the *numeric* ground truth is a lambda only the grader ever calls, and never appears
  in a prompt.

  **And that guard passed the whole time, and it was still not enough.** Two of my behavioural tasks
  had their questions — *and their answers* — written into the system prompt as illustrative
  "examples." They scored 100% and 90%. The guard never fired, because the ground truth of a
  behavioural task is not a number.

  > **A leak guard only guards the channel you pointed it at.** ([D28](docs/DECISIONS.md))

- **Contamination** *(new — this is the guard I did not have when I needed it)* — every 4-gram of
  every benchmark question is checked against the **full model-visible surface**: the system prompt,
  the tool schemas, and the verifier's rubric. Prompt examples are training data; examples drawn
  from your benchmark are training on your test set, and it never *feels* like cheating, because it
  reads as advice.

  The fix has a sting in it. My first attempt swapped the leaked examples for an abstract
  description of a false premise — and the **clean control task fell from 10/10 to 5/10**, while the
  two contaminated ones barely moved. Those examples had been leaking two answers *and* teaching the
  reflex to everything else. **Concrete examples teach; abstract descriptions don't.** The shipped
  prompt uses concrete examples about *warehouses and work shifts* — vivid, and in neither dataset.

Both bugs are written up in [D28](docs/DECISIONS.md). I mention them here rather than in a footnote,
because credibility is the only thing a benchmark has.

Because every trap has a *documented* wrong answer, a failure isn't merely wrong — the harness
reports **which** wrong. Landing on the naive answer means the agent fell into the notice–act gap
*specifically*, rather than just fumbling. That's the difference between a benchmark and a
diagnostic.

### Does it generalise?

Every trap in `trial.csv` is one **I** planted while designing the guardrails. Passing it only
proves they work on the failures I *already knew about* — and more runs shrink the *variance* of
that claim, not its **bias**.

So `sales.csv` is a **held-out domain**: e-commerce, not medicine. Revenue exported as text
(`"1,234.56"`), internal QA orders at `999999.99`, refunds left in the file, `-1` for a missing
age, and a Simpson's paradox on **channel × segment** instead of arm × severity.

| domain | pass rate | 95% CI | |
|---|---|---|---|
| `penguins` | 100% | `[100%, 100%]` | clean data |
| `trial` | 82% | `[62%, 97%]` | **designed against** |
| **`sales`** | **88%** | `[77%, 96%]` | **🎯 held-out domain** |

It holds up on the domain it was never tuned for. Which is reassuring, and which is also — read on —
**the wrong number to be reassured by.**

### The results

**4,480 runs · 28 tasks · 3 domains · 20 runs per cell · $16.13 · 0 crashes.** Each ablation is a
*paired* bootstrap against the full agent (10k hierarchical resamples: tasks, then runs within
tasks).

| remove this | pass | Δ | 95% CI | verdict |
|---|---|---|---|---|
| *(nothing — the full agent)* | **87%** | — | | |
| **the deterministic data briefing** | 60% | **−27%** | `[−40%, −15%]` | **HURTS** |
| *every guardrail at once* | 67% | **−20%** | `[−32%, −9%]` | **HURTS** |
| **the Findings Ledger** — *my centrepiece* | 82% | **−6%** | `[−11%, −1%]` | **HURTS** |
| the fresh-context verifier | 86% | −1% | `[−4%, +3%]` | no detectable effect |
| observation truncation | 87% | −1% | `[−3%, +3%]` | no detectable effect |
| the grounding gate | 87% | −1% | `[−4%, +3%]` | no detectable effect |
| the Question Contract | 88% | +0% | `[−4%, +4%]` | no detectable effect |

**Removing twenty lines of pandas is the largest effect in the study, by a factor of four.** I built
this design around *gates*. The thing carrying it is the *detector*.

> ## A gate is only as good as the detector feeding it.
>
> The papers describe a **notice–act** gap and I read it as a failure to *act* — so I built
> machinery to force action. The ablation says most of the leverage is on the **notice** side.
>
> Tell the agent what's in the data, deterministically, before it starts, and **it acts on it.**
> It did not need to be forced nearly as much as it needed to be *informed*.

*(That is a claim about **this** model, not about the papers. GeneBench-Pro is explicit that*
frontier *models notice reliably and fail at acting; my agent is a 30B open model chosen precisely
because it is weak. Which half of the scaffolding earns its keep depends on which half of the job
your base model already does for free — and only a harness will tell you which.)*

### 🚨 And then the harness caught me too

Everything above is a `95% CI`, which looks like rigour. Here is what happened when I ran the
**identical benchmark twice**, either side of a change that provably altered nothing (same error
rate, same step count, same budget-exhaustion rate):

| removing the Findings Ledger | Δ | 95% CI | verdict |
|---|---|---|---|
| run A | **−11.1%** | `[−18.6%, −4.3%]` | ***"SIGNIFICANT"*** |
| run B | **−1.8%** | `[−8.9%, +4.6%]` | *"no detectable effect"* |

**Same code. Same tasks. Opposite conclusions, from intervals that barely overlap.** At least one of
them is wrong and I cannot tell which — which means, as stated, **both are worthless.**

The bootstrap isn't miscoded. It's being asked something it can't answer: with **10 runs per cell**
it resamples from the ten outcomes it *happened to see*, so a cell that came back `1/10` has a
bootstrap distribution centred near 10% and **cannot reach the 60% the next run produced.** Near
`p=0` and `p=1` the empirical distribution is degenerate — and that is exactly where the hard tasks
live.

> **An error bar is not automatically an honest number.** It is honest only if the error bar is —
> and mine was computed from too little data to know.
>
> I built this harness to stop myself reading noise as signal. Then I read noise as signal
> **out of the harness**, twice, in opposite directions. ([D31](docs/DECISIONS.md))

**So the error bar I now report is a measurement, not a model.** Run the whole benchmark twice; the
gap between the two runs *is* the uncertainty. `uv run python -m evals.report` prints it:

| remove this | run A | run B | **spread** | pooled Δ | |
|---|---|---|---|---|---|
| the data briefing | −31% | −25% | **6 pt** | **−27%** | ✅ robust |
| every guardrail | −22% | −18% | 4 pt | −20% | ✅ robust |
| **the Findings Ledger** | **−9%** | **−3%** | **6 pt** | **−6%** | ⚠️ real, but I would not bet on the size |
| the verifier | −1% | +1% | 3 pt | −0% | — |
| observation truncation | +0% | −0% | 1 pt | +0% | — |
| the grounding gate | −0% | −1% | 1 pt | −0% | 🚩 **sign flips** |
| the Question Contract | −1% | **+3%** | 4 pt | **+1%** | 🚩 **sign flips** |

Read the *spread* column, not the point estimate. The briefing is the only thing here I would defend
without hedging.

### The failure that the average hides

`sales` scores 88%. The one task the entire design exists for scores **45%**.

| | full agent | landed on the *documented naive* answer |
|---|---|---|
| `t4_simpson` — Simpson's paradox, **medical** (designed against) | 50% | 50% |
| **`s4_simpson_sales`** — Simpson's paradox, **held-out** | **45%** | **40%** |
| `b3_ambiguous` — *"did the biomarker improve?"* | **0%** | — |

Twenty-two easy tasks carry that 90%. **On the confounded comparison — the failure both papers were
written about — the agent is a coin flip.**

*(And `s4` is the same task that read `1/10` and then `6/10` on two runs of n=10. The truth is
`9/20`. **Every strong sentence I wrote about it from a single run was noise.**)*

> ## You cannot close the *notice* gap with better advice.
>
> Sentinels, duplicates and scope land at **95–100%**, because a twenty-line profiler **detects**
> them and hands the agent an obligation it cannot walk past.
>
> Confounding lands at a coin flip, because **nothing detects it.** Nothing computes *"are these
> groups imbalanced on a third variable?"*, so nothing is seeded, so the gate has nothing to gate.
> I tried rewording the prompt to fix it. It didn't move.
>
> The ledger can force a noticed thing to *reach* a decision. It cannot make the agent **notice**.

**What I'd build next isn't another gate. It's that detector.**

### So I built it. It works. And I ran out of money before I could prove it properly.

Twenty lines (`observe._confounds`): cross-tabulate every pair of low-cardinality categorical
columns; if one's conditional distribution departs from its marginal by ≥15 points, **the groups are
not comparable** — seed it as an open finding the agent cannot submit past. It names no column, no
dataset, no domain. It finds `arm × severity` in a clinical trial and `channel × customer_segment` in
an e-commerce export **by exactly the same arithmetic.**

| | before | after |
|---|---|---|
| the demo (`t4_simpson`, temperature 0) | **−0.087** — the confounded answer, after burning all 20 steps | **+0.150** — the truth, in 13 steps, `ACTED` not `DISMISSED` |
| `t4_simpson`, 20 runs | **8/20** | **13/20** |

> ## ⚠️ Where the evidence stops — read this before the tables above
>
> **The 4,480-run grid was measured *before* the detector existed.** I built it, watched it turn the
> flagship failure into a pass, and then **exhausted my API budget** before I could re-run the grid
> with it. So:
>
> - Every number in this README is from the system **without** the detector. They are a **lower
>   bound** on the shipped code.
> - The detector is **on by default**. The committed grid is still exactly reproducible:
>   `uv run python -m evals.run_eval --ablate --runs 20 --no-confound`.
> - The evidence I *do* have: `t4_simpson` **8/20 → 13/20** (`evals/results_confound_detector.jsonl`,
>   222 clean runs), and the demo above. The held-out domain never ran. **I do not know what it does
>   there, and that is the number I most want.**
>
> I could have shipped the old code to make the table match, and said nothing. Closing the gap costs
> about $30 and one command.

That is the honest end of this project: the eval told me what to build, I built it, the first
evidence says it works, **and I am telling you exactly how far the evidence goes** — which is the
only habit in here I would actually defend.

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

**All eight of them.** Notebook `01` opens with a deliberately *raw* `client.chat.completions.create()`
call — the point of that chapter is the unwrapped protocol — so it cannot hit the cache. It now says
so out loud and replays the identical request through the cached helper instead, rather than
crashing. Verified from a fresh clone, at a foreign path, with the environment scrubbed.

*That sentence has now been false twice.* Once because the flag could not actually be flipped
([D25](docs/DECISIONS.md)), and once because the cache was keyed to **absolute file paths**, which
made it replay on precisely one computer — mine ([D30](docs/DECISIONS.md)). `test_portability()`
now asserts the prompt is byte-identical from any working directory, and it runs before every
evaluation. **A reproducibility claim only checked by the person who wrote it has not been
checked.**

```bash
uv run python data/make_trial.py            # regenerate the data (re-verifies every trap)
uv run python evals/tasks.py                # inspect the benchmark + run its guards
uv run python -m evals.run_eval --ablate    # the full ablation study
```

---

## Where things are

```
docs/PART1_REVIEW.md  Part 1 — the papers: summary + critical assessment
docs/DESIGN.md        Part 2 — the proposal: architecture, the papers, the evaluation
docs/DECISIONS.md     every non-obvious choice, its alternatives, and what would change my mind
docs/AI_USE.md        where I used AI, and the four bugs it happily wrote for me
agentlib/             the agent itself (~1,000 lines of code)
notebooks/            the buildup — seven chapters
data/make_trial.py    the synthetic dataset, and the assertions that keep its traps honest
evals/                the benchmark, the grader, the ablation runner
evals/report.py       every number in this README, printed from the results file
```

Every figure quoted here comes out of `uv run python -m evals.report`. If a number in the prose and
a number in that script disagree, **the script is right and the prose is a bug** — which is not a
hypothetical, and is why the script exists.

### Want to run it on your own GPU?

```bash
uv run modal deploy infra/modal_vllm.py     # vLLM, the same model, scale-to-zero
```

Then two environment variables — `LLM_BASE_URL` and `LLM_API_KEY` — and **nothing else changes**.
Not the loop, not the ledgers, not the gates, not the cache, not the eval harness. That is
[D01](docs/DECISIONS.md) ("no framework — there is precisely one integration to write") stated as a
command rather than as a claim. Any OpenAI-compatible endpoint works, including free tiers; see
`.env.example`.

*(Caveat, stated in the script: a Modal account with no card can't get an 80 GB GPU, so it serves a
**4-bit** quant of the eval model. Quantisation is not a no-op — numbers from that endpoint are not
comparable to the grid below, and the harness's prompt fingerprint will refuse to mix them.)*

---

Built on **Nebius Token Factory** — `base_url` + `api_key` was the entire integration. Agent:
`Qwen3-30B-A3B`, chosen because it is **14× cheaper than the best model available** and therefore
makes the thesis falsifiable rather than flattering ([D18](docs/DECISIONS.md)). Verifier and judge:
`gpt-oss-120b`, deliberately a *different family*, because a model grading its own work shows
self-preference bias.

---

## The honest part

This is a benchmark I wrote, graded by a harness I wrote, on data I generated, with guardrails
designed against failures I chose to plant. 28 tasks, ten runs each — enough to put error bars on
it, not enough to make it a fact.

Specifically, and in the order I would want them raised:

- **On the confounded comparison — the failure both papers are about — the agent is a coin flip**
  (45%). Rewording the prompt did not move it. The 90% is carried by easy tasks.
- **My confidence intervals were too narrow, and I only found out by accident.** Running the same
  benchmark twice flipped my centrepiece from "significant" to "no effect"
  ([D31](docs/DECISIONS.md)). The numbers above are pooled over 20 runs per cell, and I report the
  **between-run spread** next to every one of them, because that is the only error bar here I trust.
- **5 of 28 tasks are graded by an LLM judge I never validated.** DrugDiscoveryBench validated theirs
  at κ=1.0 against two other judges on 200 responses. I read mine's calls and they looked right,
  which is worth what it sounds like.
- **My ablation grid cannot separate the Findings Ledger from the pre-seeding it feeds** — `no_ledger`
  removes both. The −6% is for the pair, and no config in the grid pulls them apart. One extra flag
  would fix it. I didn't run it.
- **The central bet is unproven.** *"Scaffolding beats model size"* is why this runs on a model 14×
  cheaper than the best one available. But I never ran the *big* model with *no* scaffolding, so I
  have not shown it. One command, ~$15, and it's the first thing I'd spend the next budget on.

**Five of the numbers in this repo were once lies, and my own harness caught every one:** a cache that
replayed one run three times and called it three ([D24]); a `LIVE=False` flag that could not be
flipped ([D25]); two benchmark answers I had written into my own system prompt ([D28]); a prompt that
overfit to medicine without naming a single column ([D29]); and a committed cache keyed to my home
directory, so *"runs offline with no API key"* was true on exactly one computer ([D30]).

**Not one of them was findable by reading the code.** Every single one surfaced by running something
and looking hard at a number that was wrong — a `$0.0000` cost, a flag that wouldn't move, a task
scoring 100%, a cache only I could hit, a CI that changed its mind.

> That is the whole argument of this project, learned the expensive way, five times:
> **AI makes it very cheap to produce something that looks right. It does not make it any cheaper to
> find out whether it *is* right. So spend the budget on the second thing.**

So the claim is **not** *"this agent is reliable."*

It is: **here is a design, here is the evidence for each piece, here is the piece doing all the work,
here is the piece that doesn't work at all, here is the instrument I used — and here is where the
instrument itself lied to me.**

[D24]: docs/DECISIONS.md
[D25]: docs/DECISIONS.md
[D28]: docs/DECISIONS.md
[D29]: docs/DECISIONS.md
[D30]: docs/DECISIONS.md
[D31]: docs/DECISIONS.md

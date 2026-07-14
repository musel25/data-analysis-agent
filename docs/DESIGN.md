# A Data Analysis Agent — Design

> **Part 2, Option A.** An agent that receives a prompt + data files, explores the data, picks an
> approach, writes and runs code, checks intermediate results, and returns a structured answer.

This document is the proposal. The notebooks in `notebooks/` are the evidence: they build the
system one mechanism at a time, and each mechanism is introduced only after you have watched the
agent fail without it.

---

## 0. TL;DR

The base model already writes competent pandas. The interesting engineering is not "make it code."
It is **making it keep the thread**: carrying what the user asked and what the data revealed into
every subsequent decision, and refusing to hand back an answer that contradicts either.

So, on top of the base model, this design adds one loop and **three ledgers the agent is not allowed
to drop**:

| Layer | What it is | The failure it exists to kill |
|---|---|---|
| **The loop** | ReAct: model → `run_python` → observation → repeat | (baseline capability) |
| **Ledger 1 — Question Contract** | A structured restatement of the estimand, population, units and explicit constraints, pinned and re-shown at every step | Scope drift; "a statistically valid model applied to the wrong population or scale" |
| **Ledger 2 — Findings Ledger** | Every diagnostic the agent notices is logged with a status; **submitting is blocked while any finding is `open`** | The **notice–act gap** — the central finding of both papers |
| **Ledger 3 — Evidence Grounding** | Every number in the final answer must appear verbatim in the stdout of code that actually ran | Derivation errors; the agent contradicting its own printed output |
| **The exit** | `submit_answer` is a *gated action*, not a default: schema-validated, then reviewed by a fresh-context verifier | "The last chance to catch the slip is at the final answer. None of the failing models caught this." |

Everything else — persistent kernel, observation truncation, data briefing, step budgets, error
fingerprinting — is table stakes and is treated as such.

The whole agent is ~1,000 lines of Python (comments and docstrings excluded — most of the file volume
is the reasoning behind each choice). That is a claim about where the difficulty actually lives, and
it is defended in [DECISIONS.md](DECISIONS.md).

---

## 1. What the papers actually say

Both supplied papers evaluate *exactly this system*: a single LLM agent, in a sandbox, writing and
running Python against messy scientific data files, returning a structured answer. Neither is a
paper about chatbots. They are papers about the thing being built here, and they agree on the
diagnosis.

### 1.1 GeneBench-Pro (Li & Ho, OpenAI, bioRxiv 2026)

129 multistage statistical-reasoning problems in genomics and translational biomedicine. The agent
gets messy staged files and a deliberately *minimal* prompt, works in a Docker container with
pandas/scipy/statsmodels and no internet, and must return one JSON object. Grading is binary,
programmatic, against absolute numeric tolerances, with **no LLM judge anywhere**; ten attempts per
problem for standard evaluations, and **five for the GPT Pro (Extended) and Claude Opus rows**.
*(That asymmetry is not a footnote — §4.4.1 of this document and PART1_REVIEW §3.5 both turn on it.)*

Each problem is built around **decision points** — "substantive inferential forks where a plausible
wrong choice leads to a qualitatively different downstream answer" (3–13 per problem, median 6).

The results are sobering. The best frontier configuration scores **31.5%**. The best mainline model
scores **literally zero across all ten attempts on 45.7% of problems**.

And the diagnosis is precise:

> "models often complete substantial portions of the workflow but exhibit a consistent gap between
> *noticing* and *acting* by identifying local diagnostic signals but failing to propagate the
> implications to the corresponding analysis decision. As a result, models often select wrong
> estimators or persist on initially plausible but incorrect analysis paths." (p. 1)

> "In many failures, the agent notices the relevant local diagnostic clue but **treats it as a local
> data cleaning issue rather than as evidence that should change the downstream statistical method
> and QC pipeline**." (p. 13)

Read that twice. The model *sees* the problem. It runs `df.describe()`, it observes the sentinel
values, it may even print a comment about them. Then it cleans the column and proceeds with the
analysis it had already decided on. The observation never reaches the decision.

Their design table names the resulting failure in one line:

> "A statistically valid final model is applied to the wrong data or population, on the wrong scale,
> or on the wrong conceptual level." (p. 6)

### 1.2 DrugDiscoveryBench (Akyürek, Tu et al., Scale AI & Phylo, 2026)

82 expert-authored drug-discovery tasks; coding agents in a container with a biomedical tool
library; graded by LLM judge against weighted expert rubrics, pass = 100% of outcome criteria.
Best agent: **51.6%**.

Crucially, they classified **226 failing runs into a taxonomy with frequencies** (Table 3, p. 11) —
assigned via a decision ladder over the trajectories, not by an automated metric:

| Failure mode | Share | Their definition |
|---|---|---|
| **Domain reasoning** | **54.0%** | "applies an incorrect scientific premise or misinterprets the data it has, even though its inputs and tools are correct" |
| **Derivation error** | **18.6%** | "has the correct inputs and approach but runs a calculation, derivation, or counting procedure incorrectly" |
| **Retrieval** | **16.4%** | "never obtains or uses the required data, querying the wrong source, failing to read a provided file" |
| **Constraint** | **7.5%** | "has the data but violates an explicit prompt instruction, such as a keep/exclude filter, a required count, or a scope limit" |
| **Final-answer slip** | **3.5%** | "performs every analytical step correctly but reports the final answer wrongly" |

Their summary of what goes wrong:

> "the agents knew which database to query and how to compute the property the task asked for at a
> high level. But somewhere along the execution **they drop a constraint, commit too early, fail to
> backtrack** or fail with respect to scientific common sense." (p. 12)

Two of their concrete cases matter enormously for design:

- A derivation failure where **the agent's own code printed the correct value and the agent then
  used a different one**: "its own code printed the correct group-level count of 1, but the final
  tally used the atom-level count of 2… It reported 8 interactions instead of 7." (p. 27)
- A scope failure where the agent silently dropped the "melanoma" qualifier partway through a
  ranking task, and:

> "**The last chance to catch the slip is at the final answer**: a human who had misread the task the
> same way would look at the result, recognize it as a meaningless response to the user's actual
> goal, and backtrack. **None of the failing models caught this.**" (p. 13)

And the finding that determines where effort should go. They re-ran the 6 unsolved tasks giving the
agents the expert's step-by-step playbook as a hint:

> "76 out of 82 tasks are solved without any hints in at least one of the trials… After the hints,
> we find that at least 1 of the agents is able to pass 80 out of 82 **and near-pass 1**. The results
> suggest that **execution is within reach for today's agents should they be given the expert
> workflow**." (p. 14)

*(I quote that as the paper's conclusion, and I believe it. I also do not think the experiment
establishes it — see PART1_REVIEW §3.4: only **6 tasks** entered the hint condition, and a
solvability check against the expert playbook was an **inclusion criterion** when the tasks were
authored.)*

### 1.3 The synthesis that drives this design

Put the two together:

- The models **can execute**. Give them the right plan and they finish the job (DDB: 80/82 with hints).
- The models **do notice**. They run the diagnostics and see the anomaly (GBP: the notice–act gap is
  a gap in *acting*, not in *seeing*).
- They fail in the seam between the two: what was noticed does not change what is done, what was
  asked stops constraining what is computed, and what was printed is not what gets reported.

> **Design thesis.** The bottleneck is not code generation, not knowledge, and not context length.
> It is **thread-keeping**. So do not spend the engineering budget on making the model smarter. Spend
> it on making it structurally impossible to drop the thread — by turning the three things that get
> dropped (the question, the finding, the number) into explicit state that the agent must reconcile
> before it is allowed to finish.

That is what the three ledgers are. Every one of them is a cheap, deterministic mechanism that
converts a silent omission into a hard stop.

---

## 2. Architecture

### 2.1 The loop

```
                    ┌──────────────────────────────────────────────┐
                    │  CONTEXT SENT TO MODEL EVERY TURN            │
                    │  ├─ system prompt (the analyst contract)     │
                    │  ├─ data briefing   (deterministic, turn 0)  │
                    │  ├─ QUESTION CONTRACT  ← pinned, always      │
                    │  ├─ FINDINGS LEDGER    ← pinned, regenerated │
                    │  ├─ transcript of prior steps (truncated)    │
                    │  └─ STATE BANNER       ← from live namespace │
                    └───────────────────┬──────────────────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │   base model      │  Qwen3-30B-A3B @ Token Factory
                              └─────────┬─────────┘
                                        │ tool_call
                    ┌───────────────────┼────────────────────┐
                    ▼                   ▼                    ▼
            ┌───────────────┐   ┌───────────────┐   ┌─────────────────┐
            │  run_python   │   │ note_finding  │   │  submit_answer  │
            │  persistent   │   │ append to     │   │  ── GATED ──    │
            │  namespace    │   │ ledger        │   │                 │
            └───────┬───────┘   └───────┬───────┘   └────────┬────────┘
                    │                   │                     │
                    │ stdout/traceback  │ ack                 ▼
                    │ truncated         │            ┌──────────────────┐
                    │ + state banner    │            │ GATE 1 schema    │
                    │                   │            │ GATE 2 ledger    │  any finding
                    │                   │            │        open?     │  still open →
                    │                   │            │ GATE 3 grounding │  REJECT
                    │                   │            │ GATE 4 verifier  │  (back to loop)
                    │                   │            └────────┬─────────┘
                    ▼                   ▼                     │ all pass
                    └───────► observation ◄──────────┐        ▼
                              appended to        ────┘   AnalysisReport
                              transcript                 (structured)
```

The loop itself is unremarkable and that is the point — it is ReAct (Yao et al.), ~40 lines. All of
the design is in what surrounds it.

### 2.2 Three tools, and why not more

**`run_python(code)`** — executes in a persistent namespace, returns stdout + truncated traceback.
One general tool beats N specific tools: `value_counts`, `groupby`, `correlate` as separate tools
would cap the agent at whatever verbs I anticipated. Anything pandas can do, the agent can do.

**`note_finding(observation, implication, status)`** — the ledger. Discussed at length below.

**`submit_answer(...)`** — the gated exit.

That's it. Every additional tool is surface area for the model to misuse, and both papers show the
models are not failing for lack of affordances.

### 2.3 The executor

A ~20-line class holding a namespace dict; `exec(code, self.ns)` with stdout captured. Variables
survive across steps, exactly like Jupyter cells — which is both the honest minimal model of a
kernel and a pleasing symmetry with the medium the whole thing is taught in.

**This is not a sandbox and the notebooks say so in a red warning box.** It is an isolated namespace
in a throwaway venv, which is honest scoping for a prototype, not a security claim. Both papers ran
their agents in Docker containers with no internet; that is the correct production answer, and the
`run_python` tool contract (code in, stdout out) is deliberately identical to what a container-backed
executor would expose, so the swap is one class, not a redesign. Overclaiming here would be worse
than the limitation itself.

### 2.4 What the model sees — and does not see

The single most important context rule:

> **The model never sees the data. It sees metadata, summaries, and the output of code it wrote.**

This is what makes the design scale-invariant. A 200k-row dataframe and a 200-row one cost the same
number of tokens, because in both cases the model is looking at `df.dtypes`, `df.head(3)`, and
whatever it chose to `print`. The dataframe lives in the executor process. Nothing about the loop or
the prompts changes if the CSV becomes a 10 GB parquet — only the code the agent writes changes
(and it can write DuckDB just as easily as pandas).

Concretely, four mechanisms keep context honest:

1. **Deterministic data briefing (turn 0).** Before the model is called even once, plain Python
   computes shape, dtypes, `head(3)`, null counts, and cardinality for each file and injects it as
   the first message. The agent never starts blind. It also saves two or three turns of budget.
   *(I originally justified this with DDB's* Retrieval *bucket — 16.4%, "failing to read a provided
   file." That is a stretch and I should say so before someone says it to me: both of DDB's worked
   Retrieval examples are* wrong-source queries *— hitting RCSB instead of NAKB — not unread local
   files. The briefing attacks a slice of that category, not the bulk of it. Its real justification
   is the one the ablation later handed me: it is worth **−27%**, which is more than every gate in
   this design combined.)*
2. **Head+tail truncation** of every observation (~1,500 chars) with an instructive marker:
   `[... 187,321 chars omitted — do not print whole dataframes; assign and inspect selectively]`.
   Tail is kept because Python puts the exception on the *last* line — head-only truncation would
   throw away the only part of a traceback that matters.
3. **State banner**, regenerated from the live namespace every single turn:
   `df: DataFrame(1200x9) | clean: DataFrame(1043x9) | model: OLSResults`. Because it is recomputed
   from ground truth rather than carried forward as text, the agent cannot drift into reasoning about
   variables that no longer exist or were truncated out of the transcript.
4. **Matplotlib guard.** If code creates figures, they are saved to disk and the observation says:
   *"2 figures saved for the human; you cannot see images — print the numbers you need."* This kills
   the plot-instead-of-compute failure mode.

---

## 3. The three ledgers

This is the part of the design that is not standard, and it is the part that is directly derived from
the papers.

### 3.1 Ledger 1 — the Question Contract

**Attacks:** DDB *Constraint* failures (7.5%) and the scope-drop case study; GBP's "wrong population
/ wrong scale."

Before any analysis, the agent's first action must be to fill in a contract:

```python
class QuestionContract(BaseModel):
    estimand:    str         # the exact quantity, in one sentence
    population:  str         # WHICH ROWS. the denominator. stated explicitly.
    units:       str         # g? kg? percent? fraction? log-scale?
    constraints: list[str]   # every explicit filter/limit the user stated
    premises:    list[str]   # what the QUESTION assumes to be true — and must be checked
    question_is_precise: bool   # required. forces an explicit judgement.
    ambiguities: list[str]   # if not precise, the readings — and which one I chose
```

Two of those fields exist because the evaluation demanded them, not because I designed them in.

**`premises`** — a question can be *wrong*. *"Which of the four sites had the highest response
rate?"* has three sites. *"Why do women respond better?"* — they don't. Without this field the agent
answered both, fluently, and laundered a false premise into a fact. With it, 0/3 → 3/3 on the sites
task.

**`question_is_precise`** is a *required boolean*, and that is the entire trick. The contract already
had an `ambiguities` list, and on *"Did the biomarker improve?"* the agent left it empty every single
time and silently picked a reading. A field the model may leave empty is a field the model will leave
empty. A required boolean cannot be skipped — and a validator makes `ambiguities` non-empty whenever
it is `False`. Same principle as the ledger: **make the omission impossible to express, rather than
asking nicely for it not to happen.**

It is pinned into context **every turn** and it is shown to the verifier at submit time.

Why this earns its place: DDB's melanoma failure is a model that, in the paper's words, *"at some
point"* silently stopped applying the melanoma qualifier in a ranking task. And **one of the three
decision points** in GBP's worked DRX1 problem is exactly a denominator trap (compute over *tested
partners* and you get a plausible, wrong answer; the target is the *full roster*). In both cases the
question itself decayed. Writing it down as structured state, and re-showing it, makes decay visible
— to the agent, to the verifier, and to me reading the trace.

The `ambiguities` field does double duty: it is how the agent handles under-specified questions
(state the interpretation and proceed, rather than guessing silently or stalling), and it is how
those interpretations surface in the final report's caveats.

### 3.2 Ledger 2 — the Findings Ledger  ← *the centrepiece*

**Attacks:** the notice–act gap (GBP's headline finding); DDB *Domain reasoning* (54.0%).

Here is the mechanism, and it is embarrassingly simple.

Every time the agent notices anything about the data — a sentinel value, a wrong dtype, duplicated
rows, a batch imbalance, a suspicious distribution — it must log it:

```python
note_finding(
    observation = "biomarker_x has 47 values of exactly -999.0",
    implication = "these are missing-data sentinels, not measurements; "
                  "they would drag the mean down by ~30%",
    status      = "open",     # open | acted | dismissed
)
```

And then the rule that makes it bite:

> **`submit_answer` is hard-blocked while any finding has status `open`.**

To close a finding, the agent must either
- **act** on it — pointing at the code step that handled it — and set `status="acted"`, or
- **dismiss** it — writing an explicit reason why it does not affect the estimand — and set
  `status="dismissed"`.

Both are recorded. Both end up in the final report.

That is the notice–act gap, converted from a silent cognitive failure into a hard, mechanical stop.
GBP's finding is that the model *notices the clue and then treats it as local cleanup*. This design
does not let "local cleanup" be a terminal state: a noticed finding is an open obligation, and the
obligation must be discharged, out loud, before the agent is permitted to finish. The implication
field is where the propagation actually happens — the agent is forced to write down *what this
changes*, which is exactly the step the papers observe it skipping.

It also produces a beautiful artifact: a run's ledger is a complete, auditable record of *what the
agent noticed and what it did about it*. That is the thing a scientist actually needs in order to
trust an automated analysis — and neither paper's systems produce it.

The cost of this mechanism is about 40 lines and one extra tool.

#### The limitation — and what it cost me to learn it

The ledger cannot force the agent to notice something it never looked at. It converts
*noticed-but-not-acted-upon* into a hard stop; it does nothing about *never-noticed*.

I wrote that sentence before I had evidence for it. Then I ran the ledger on the Simpson's-paradox
task and watched it happen: the agent caught the duplicate patients, was forced to act on them, did
so correctly — and submitted **−0.087**. It had simply never noticed the confounding, so there was
no obligation to discharge, and the gate had nothing to block. The mechanism worked perfectly and
the answer was still wrong-signed.

The annoying part is that `data_dictionary.md` says it in plain English, and the briefing had put
that file in the context on turn one. The agent read the warning and did not act on it — which is,
of course, the notice–act gap wearing a different hat.

**So the ledger is pre-seeded.** Anything the deterministic profiler can find — sentinels, duplicate
identifiers, numeric-looking strings — is entered as an **open finding before turn one**. Not
printed as a helpful note in the briefing. *Registered as an obligation.*

> **Information can be ignored. An obligation cannot.**

That is the division of labour the whole design rests on:

| | does what |
|---|---|
| **deterministic code** | finds the mechanical problems, and makes them **un-ignorable** |
| **the model** | decides what they **mean** for this particular question |

And it is still not a straitjacket: on the response-rate task the agent looked at the pre-seeded
`-999` finding and **dismissed** it — correctly, because `biomarker_baseline` never enters that
calculation. We do not force a conclusion. We force the observation to *reach* the decision. What
happens when it arrives is still the model's judgement.

Between the seeding and the ledger the coverage is good. It is not complete, and I would not claim
otherwise.

### 3.3 Ledger 3 — Evidence Grounding

**Attacks:** DDB *Derivation error* (18.6%) and *Final-answer slip* (3.5%); numeric hallucination.

Every number in the answer's `evidence` list must appear, verbatim, in the stdout of code that
actually executed. This is a deterministic check — regex the numbers out, normalize (strip commas
and percent signs, compare at 4 significant figures, also test ×100 and ÷100 for the
percent-vs-fraction confusion), and require set containment against the concatenated stdout of the
whole run. No LLM is involved. It cannot be talked out of it.

If a number in the answer never appeared in any output, `submit_answer` returns:

```
REJECTED: the values {0.62} in your evidence never appeared in any executed output.
Compute and print them before citing them.
```

DDB documents the exact failure this catches, in a frontier model: *"its own code printed the correct
group-level count of 1, but the final tally used the atom-level count of 2."* An LLM reviewer might
catch that. A regex catches it every time, for free, in fifteen lines.

The rule that makes this work is stated in the system prompt as an absolute: **every number you
report must have been printed by code you ran.** Estimating is not a permitted operation.

### 3.4 The gated exit and the fresh-context verifier

`submit_answer` runs four gates in order — cheapest and most deterministic first:

1. **Schema** — pydantic validation; on failure the `ValidationError` text goes back as the tool
   result and the model self-repairs.
2. **Ledger** — any `open` finding → reject.
3. **Grounding** — ungrounded numbers → reject.
4. **Verifier** — one LLM call, and only if 1–3 passed.

The verifier is the answer to DDB's *"the last chance to catch the slip is at the final answer…
none of the failing models caught this."* It is a single call to a **different model family**
(agent: Qwen3-30B-A3B; verifier: gpt-oss-120b — cross-family, because a model asked to review its
own work is subject to self-preference bias, Zheng et al. 2023).

The critical design detail: **the verifier does not see the agent's reasoning.** It sees only the
Question Contract, the data briefing, the executed code and its outputs, and the draft answer. Show
it the agent's chain of thought and it gets anchored by the agent's own narrative and rubber-stamps
— a verifier that reads the transcript is a verifier that measures nothing. The fresh context *is*
the mechanism.

It scores four rubric items and returns `accept | revise` with issues:
1. Does the answer address the estimand **as stated in the contract** — same population, same units?
2. Is every quantitative claim supported by the shown outputs?
3. Is there a causal claim resting on correlational evidence?
4. Is the sample size / uncertainty acknowledged where it matters?

One revision round, maximum. If it wants to revise twice, the answer is accepted with the verifier's
outstanding issues appended to `caveats` — the agent is not permitted to loop forever, and an honest
caveat beats a stall.

This is *why* a single-agent design can afford to skip a "critic agent": the only genuine benefit of
a second agent here is an un-anchored second opinion, and that is purchasable for one API call
without any of the coordination overhead.

### 3.5 The structured answer

```python
class AnalysisReport(BaseModel):
    answer:     str                              # the direct answer, in one sentence
    value:      float | str | None               # the machine-checkable quantity, if there is one
    method:     str                              # what was done, and why this method
    evidence:   list[str]                        # numbers + the code step each came from
    findings:   list[Finding]                    # auto-attached: what was noticed, what was done
    caveats:    list[str]                        # including the contract's ambiguities
    confidence: Literal["high", "medium", "low"]
```

`findings` is attached automatically from the ledger, not written by the model. The report therefore
always carries the audit trail whether the model feels like mentioning it or not.

### 3.6 The unglamorous safety rails

- **Step budget** (max 20). When 3 steps remain, every observation is suffixed with
  `(3 steps remaining — converge on an answer)`, so the agent degrades like an anytime algorithm
  instead of getting guillotined mid-thought. On exhaustion it is *forced* to submit with
  `confidence="low"` and the reason in `caveats` — a partial answer with an honest caveat is worth
  more than a crash.
- **Error fingerprinting.** `hash(exception_type, offending_line)`. Same fingerprint twice → inject
  *"You have hit this exact error twice. Before writing more code, state the root cause in one
  sentence."* (Reflexion, Shinn et al. 2023, in ten lines.) Three times → *"this approach is not
  working, choose a different strategy."* The two-stage design is deliberate: jumping straight to
  "try something else" makes the agent abandon a 95%-correct approach over a typo.
- **Duplicate-code short-circuit.** If the model submits byte-identical code twice, return the cached
  observation without executing. Breaks the tightest death loops for free.
- **Kernel preamble.** `pd.set_option('display.max_rows', 20)` etc., executed before turn 0 — caps
  accidental dumps at the source.

---

## 4. How I'd know it works

Both papers are, fundamentally, evaluation papers. The evaluation design here steals from them
directly and says so.

### 4.1 The benchmark

28 tasks over three domains. Ground truth is a pandas callable that only the grader ever invokes.

The datasets are the pedagogical spine of the whole series:

- **`penguins.csv`** — real, clean, familiar (344 rows). Used for chapters 1–3, where the agent looks
  great and the mechanics are the lesson.
- **`trial.csv`** — a synthetic clinical-biomarker dataset with a **known data-generating process**,
  built specifically so that plausible-but-wrong analyses give *materially different* answers.

That second dataset is built on GBP's principles verbatim: *fully simulated* (so the causal structure
is known and nothing is memorized), with explicit **decision points**, and **ablation-verified
numerical separation** — for every task I assert that the naive answer and the correct answer differ
by far more than the grading tolerance. A benchmark where the lazy path accidentally lands inside
the tolerance band is a benchmark that grades nothing.

The planted decision points, each mirroring a documented failure:

| Trap | The naive path | The correct path | Paper |
|---|---|---|---|
| `-999` sentinels in a biomarker column | mean over raw column | recognise as missing, exclude | GBP notice–act |
| Re-tested patients appear twice | count rows | deduplicate on patient_id | DDB derivation |
| Assay batch B is on a shifted scale | pool all batches | correct or exclude batch B | GBP "wrong scale" |
| Question restricts to one arm | compute over everyone | honour the stated filter | DDB constraint (7.5%) |
| **Simpson's paradox**: treatment looks worse overall, better in every severity stratum | report the marginal effect | stratify; report the conditional effect | GBP "wrong population / conceptual level" |
| A question whose premise is false | answer it anyway | flag the false premise | DDB melanoma case |

The Simpson's-paradox task is the one I care most about, because it is the purest possible test of
the notice–act gap: the agent will *see* the imbalance if it looks, and the entire question is
whether seeing it changes what it computes.

### 4.2 Grading

Following GeneBench-Pro: **binary, programmatic, all-or-nothing, with pre-specified absolute
tolerances.** No LLM judge for anything a `math.isclose` can decide. The agent emits a structured
report; the grader reads `value`, compares within tolerance, done.

GBP defends the strictness and I agree with them:

> "an agent that executes several intermediate steps correctly but returns the wrong decision-relevant
> answer has not successfully automated the analysis." (p. 14)

LLM-as-judge is used **only** for the behavioural tasks that no numeric check can grade (did it flag
the false premise? did it surface the ambiguity?), with a binary rubric, temperature 0, and a
different model family than the agent.

**And I did not validate it.** DDB validated theirs against two other judges at κ=1.0 on 200
responses; I read mine's calls and they looked right, which is worth exactly what it sounds like.
5 of 28 tasks depend on it. That is the weakest link in this evaluation and I would rather say so
than let the word "judge" imply a rigour I did not buy.

### 4.3 What gets measured

Per (task, config), **20 runs** (GBP uses 10, DDB uses 3; agent evals are noisy and a single run
proves nothing — see D24 for the version of this I got wrong). *I started at 10, like GBP. §4.4.1 is
the story of why 10 is not enough, and why this now says 20:*

- **pass rate** — the headline, binary, all-or-nothing.
- **wrong-attractor rate** — *did the agent land on the known naive answer?* This is the metric I am
  proudest of. Because every task has a documented plausible-but-wrong path with a numerically
  distinct answer, a failure is not just "wrong" — I can tell you *which* wrong. An agent that lands
  on the naive answer has fallen into the notice–act gap; an agent that lands somewhere else has
  simply erred. No standard harness reports this, and it takes ten lines.
- **failure taxonomy** — every failing run is hand-classified into DrugDiscoveryBench's five
  categories (domain reasoning / derivation / retrieval / constraint / final-answer slip). Reusing a
  published taxonomy means my failure profile is directly comparable to theirs.
- **process metrics** — steps, execution errors, errors *recovered from*, tokens, dollars.
- **consistency** — pass@5 vs 5-of-5. Which tasks are flaky, and are they the ones you'd expect?

### 4.4 The ablations — the actual argument, and what they actually showed

Each config switches off exactly one mechanism. The deal I made with myself: **if an ablation shows
a mechanism does not pay for itself, it gets cut.**

Here is what happened when I kept that deal.

#### First: does it generalise at all?

This is the question that matters most, and it is not answered by running the same benchmark more
times. Every trap in `trial.csv` is one **I** planted while designing the guardrails. Passing it
demonstrates the guardrails work on the failures I *already knew about*. More runs shrink the
variance of that claim; they do nothing about its **bias**.

So `sales.csv` — e-commerce, not medicine — is a **held-out domain**. Different columns, different
semantics, traps of the same species and a different animal (see D26).

| domain | pass rate | 95% CI | |
|---|---|---|---|
| `penguins` | 100% | `[100%, 100%]` | clean data, no traps |
| `trial` | 82% | `[62%, 97%]` | **designed against** |
| `sales` | **88%** | `[77%, 96%]` | **🎯 held-out domain — never designed against** |
| held-out *tasks* | 96% | `[90%, 100%]` | never looked at while tuning the prompt |

**The agent holds up on the domain it was never tuned for.** Which is reassuring — and which is, as
§4.5 shows, **exactly the wrong number to be reassured by.** An aggregate is an average, and an
average is where a failure goes to hide.

#### What the evidence supports

On the **trap tasks** — the ones with a planted decision point — the guardrails as a stack are
unambiguous:

| | pass rate | fell for the *documented* naive answer |
|---|---|---|
| **full agent** | **87%** | **7%** |
| **no guardrails** | **61%** | **24%** |

A **3.3× reduction in the wrong-attractor rate**, all three domains, like for like. That is the
notice–act gap, measured on my own agent, and it is far larger than the noise.

*(Both arms are all-domain. An earlier draft of this document reported "nearly 5×" by comparing an
all-domain `full` against a trial-only `no_guardrails` — two different denominators, which is how
you accidentally flatter yourself. Every number in this section now comes out of
`uv run python -m evals.report`, and if the prose and the script disagree, the script is right.)*

The *shape* matters as much as the size: on **clean** data (lookup, aggregation, groupby,
correlation) both configs score 100%. The guardrails buy nothing where there is nothing to catch,
which is exactly what a guardrail should do.

#### 🚨 And now the result I did not expect

**4,480 runs · 28 tasks · 3 domains · 20 runs per cell · $16.13 · 0 crashes.** Each ablation is a
*paired* bootstrap against the full agent (10,000 hierarchical resamples: tasks, then runs within
tasks — GeneBench-Pro's scheme; they use 20,000).

| remove this | pass rate | Δ vs full | 95% CI | verdict |
|---|---|---|---|---|
| *(nothing — the full agent)* | **87%** | — | | |
| **the deterministic data briefing** | **60%** | **−27%** | `[−40%, −15%]` | **HURTS** |
| *every guardrail at once* | 67% | **−20%** | `[−32%, −9%]` | **HURTS** |
| **the Findings Ledger** — the centrepiece | 81% | **−6%** | `[−11%, −1%]` | **HURTS** |
| the fresh-context verifier | 87% | −0% | `[−4%, +3%]` | no detectable effect |
| observation truncation | 87% | +0% | `[−3%, +3%]` | no detectable effect |
| the numeric grounding gate | 87% | −0% | `[−3%, +3%]` | no detectable effect |
| the Question Contract | 88% | +1% | `[−2%, +5%]` | no detectable effect |

**And do not read that `95% CI` column without reading §4.4.1 first.** It is narrower than the truth,
and I only found that out by accident.

> ### ⚠️ And one more thing this table is not
>
> **It was measured before the confounding detector existed** (D33). I built that detector because
> this very table told me to, watched it turn the flagship failure into a pass (`t4_simpson`
> **8/20 → 13/20**; the demo from **−0.087** to **+0.150**), and then **ran out of API budget**
> before I could re-run the grid with it.
>
> So every number here describes the shipped system **minus its newest and, on the evidence, most
> important component.** They are a lower bound. The detector is on by default; this grid is
> reproducible with `--no-confound`; closing the gap is one command and about $30.
>
> I could have shipped the old code so the table matched. This seemed better.

**Removing the briefing — twenty lines of pandas — is the single largest effect in the study, and
it is the mechanism I spent the least time on.** I built this design around *gates*. The thing
carrying it is the *detector*.

### 4.4.1 🚨 The instrument lied, and I only caught it by accident

Everything in the table above carries a `95% CI`, which looks like rigour. Watch what happened when I
ran **the identical benchmark twice**, either side of a change I had already proved was inert (same
error rate 0.27 → 0.23, same step count 11.4 → 11.6, same budget-exhaustion rate — the prompt differs
by one path string):

| removing the Findings Ledger | Δ | 95% CI | verdict |
|---|---|---|---|
| **run A** | **−11.1%** | `[−18.6%, −4.3%]` | ***"SIGNIFICANT"*** |
| **run B** | **−1.8%** | `[−8.9%, +4.6%]` | *"no detectable effect"* |

Same code. Same tasks. Same ten runs per cell. **Opposite conclusions, from 95% intervals that barely
overlap.** Individual tasks swung 30–50 points between the two runs.

At least one of those intervals is wrong, and I have no way to tell which. **So as stated, both are
worthless.**

#### Why it lied

Not a coding bug — the hierarchical bootstrap is implemented correctly (D27). It is a bug in what I
*asked* it.

With **10 runs per cell**, the bootstrap resamples with replacement *from the ten outcomes I happened
to observe.* A cell that came back `1/10` therefore has a bootstrap distribution centred near 10% —
and it **cannot reach the 60% the next run produced.** The empirical distribution is degenerate near
`p=0` and `p=1`, and those are exactly where the hard tasks live.

> **A bootstrap gives you the sampling variance of the data you have.** It knows nothing whatever
> about the data you did not collect. At n=10, near the extremes, that gap is enormous — and the
> interval it hands back is too narrow **in a way that feels rigorous.**

#### Why this is the worst bug in the project

`evals/stats.py` opens with a sentence I wrote myself:

> *"Confidence intervals, because a point estimate is how you fool yourself."*

And then I fooled myself **from inside the confidence interval** — twice, in opposite directions, and
wrote up both as findings. D27 congratulates me for catching a point estimate masquerading as a
result. This is the same error one level up: **a confidence interval masquerading as a result.**

> An error bar is not automatically an honest number. It is honest only if the error bar is — and
> mine was computed from too little data to know.
>
> The instrument you build to stop yourself believing noise **is itself an instrument**, and it also
> has to be checked — by the only method that has ever worked here: **run it again and see if it says
> the same thing.**

#### The error bar I actually report now

Doubled to **20 runs per cell** (4,480 runs, $16). And alongside every CI, the number I trust more:
split the runs in half, compute the ablation on each, and report **the spread between the two
halves.** That is not a *model* of the uncertainty. It *is* the uncertainty, observed.
(`uv run python -m evals.report` prints it.)

| remove this | run A | run B | **spread** | pooled Δ | |
|---|---|---|---|---|---|
| **the data briefing** | −29% | −25% | **4 pt** | **−27%** | ✅ **robust** |
| every guardrail at once | −22% | −18% | 4 pt | −20% | ✅ robust |
| **the Findings Ledger** | **−9%** | **−3%** | **6 pt** | **−6%** | ⚠️ real, but I would not defend the *size* |
| the fresh-context verifier | −1% | +1% | 3 pt | −0% | — |
| observation truncation | +0% | −0% | 1 pt | +0% | — |
| the grounding gate | −0% | −1% | 1 pt | −0% | 🚩 **sign flips** |
| the Question Contract | −1% | **+3%** | 4 pt | **+1%** | 🚩 **sign flips** |

**Read the spread column, not the point estimate.** The briefing is the only mechanism in this design
I would defend without hedging. The Findings Ledger is real — the pooled interval excludes zero — but
a 10-point swing between two halves of the same experiment means I will not put a number on how much
it is worth.

And the two gates whose *sign flips* between halves are not weak findings. **They are not findings.**

> If a verdict changes when you run the same experiment again, it was never a verdict.
> **It was weather.**

#### An oddity I will not overclaim

Removing the briefing alone (**60%**) scores *worse* than removing the briefing **and every gate**
(**67%**). Removing more made it better, which is impossible if the gates only ever help.

The process metrics show a plausible mechanism. Strip the detector but keep the gates, and the agent
still has a `note_finding` tool and a blocked exit — but **nothing informative to put in them.** It
logs whatever it happens to stumble over, spends turns discharging it, and burns budget on ceremony
(**11.0 steps, 1.5 findings**). Strip the gates too and it simply computes (**8.7 steps, 0.4
findings**), and does slightly better. For reference the full agent runs 11.4 steps on 2.8 findings
— the same ceremony, but with something real to be ceremonious about.

**Gates without a detector may be worse than no gates at all.** But the paired CI is
`[−16.8%, +5.4%]` — it crosses zero. This is a **hypothesis with a mechanism, not a finding**, and
it is exactly the kind of story that is fun to tell and would be dishonest to assert. It is the
first thing I would design an experiment for.

#### What it means

Go back to the papers. GeneBench-Pro observes that *"the agent notices the relevant local diagnostic
clue but treats it as a local data cleaning issue rather than as evidence that should change the
downstream statistical method and QC pipeline"* (p. 13). I read that as a failure to **act** — so I
built machinery to force action.

The ablation says the leverage is on the **notice** side.

When the agent is simply *told* what is in the data — *"`biomarker_baseline` has 88 values of
exactly −999"*, *"`patient_id` has 48 duplicates"*, plus the data dictionary — **it acts on it.** It
did not need to be forced. It needed to be **informed**.

> ## A gate is only as good as the detector feeding it.
>
> The gates were not wrong. They were **redundant** — the detector in front of them was already
> doing the job. Remove the detector and the gates have nothing to gate, which is precisely why
> `no_briefing` collapses to roughly the same score as `no_guardrails`.

#### ⚠️ The obvious objection, which I want to raise before you do

**GeneBench-Pro says the opposite of what I just said.** They are explicit that *frontier* models
*consistently notice* the data issues (p. 14), and that *"the main qualitative improvement in stronger
models lies less in noticing the relevant diagnostic clues than in turning those observations into
concrete corrective and model-selection decisions that move the analysis onto the correct path"*
(p. 3). If noticing is the solved half for them, how can the detector be the whole story for me?

Because **we are not talking about the same model.** GBP's subjects are GPT-5.6-class frontier
systems. My agent is **Qwen3-30B-A3B** — a 30B open model chosen precisely *because* it is weak
(§6). For a frontier model, `df.describe()` and the sentinel is spotted. For a 30B model at a
$0.10/1M price point, **noticing is genuinely not reliable**, and the deterministic profiler is
substituting for a capability the base model does not have.

So this is not a contradiction of GBP. It is a **boundary condition on their finding**, and it
sharpens rather than challenges it:

> **The notice–act gap is a frontier-model problem. The notice gap is a small-model problem.**
> Which half of the scaffolding earns its keep depends on which half of the job your base model
> already does for free — and that means "what should I build on top of the base model?" has no
> model-independent answer.
>
> Which is, incidentally, an argument for owning an eval harness rather than a list of best
> practices: the right answer changes when you change the model, and only the harness will tell you.

The falsifiable version of that claim is one command away — re-run this exact ablation grid on
GLM-5.2 and the briefing's −27% should *shrink*. I have not run it (§6). If it does not shrink, I am
wrong about the mechanism, and I would want to know.

This also explains the ambiguity failure (§4.5) exactly: there is **no detector for ambiguity**, so
there is nothing to feed the gate, so the gate does nothing.

#### The honest caveats on that finding

1. **"No measurable benefit" is not "no benefit."** The `−27%` for the briefing is far outside the
   noise. The `≈0`s for the individual gates are **not**: with 28 tasks the intervals are ±5–9
   points, so a real effect of three points would be invisible here.
2. **The gates may be insurance rather than throughput.** A grounding check that fires on a handful
   of runs cannot move a 280-run pass rate — but the failure it prevents (a fabricated number in a
   filing) is not one you price by frequency. The right test for a gate is adversarial, not average.
3. **The detector and the gate are not separable in this grid, and that is a flaw.**
   `agent.py` seeds the ledger `if cfg.use_ledger and cfg.use_briefing:` — so `no_ledger` removes
   the blocking gate **and every pre-seeded finding.** The `−6%` I report for the Findings Ledger is
   really "gate + seeded findings", and **no configuration in this grid pulls the two apart.** The
   clean experiment is a ninth config — briefing on, ledger on, *seeding off* — which is one flag and
   about a dollar. I have not run it. It is the single cheapest thing anyone could do to improve this
   evaluation, and I would rather point at it than quietly let the ambiguity stand.

But the deal in this document was: *if an ablation shows a mechanism does not pay for itself, it
gets cut.* So, kept honestly: **on this benchmark, at this sample size, the Findings Ledger, the
verifier, the grounding gate and the Question Contract do not demonstrably pay for themselves. The
deterministic briefing does, enormously.**

If I could ship one mechanism, it would be the one I spent the least time on.

### 4.5 The failure that matters — and why an average hid it

Go back to §4.4's domain table: `sales` scores **88%** — *better* than the domain the guardrails were
designed against. That was the number I wanted, and for about a day it was the number I quoted.

Now read the same runs per task.

| | full agent | landed on the *documented naive* answer |
|---|---|---|
| `t4_simpson` — Simpson's paradox, **medical** (designed against) | 40% | 55% |
| **`s4_simpson_sales`** — Simpson's paradox, **held-out domain** | **35%** | **50%** |
| `b3_ambiguous` — *"Did the biomarker improve?"* | **5%** | — |

**On the confounded comparison — the failure both supplied papers were written about — the agent is
worse than a coin flip.** And when it fails it lands *exactly* on the confounded answer: it is
**more likely to land on the naive answer (50–55%) than on the right one (35–40%)**. Twenty-five
easier tasks — spanning 75–100% — carry the 87%.

*(`s4` is also the task that read **1/10** on one run of n=10 and **6/10** on the next. The truth,
at n=20, is **7/20**. Every strong sentence I wrote about this task from a single run — in both
directions — was noise. See §4.4.1, and note that I am only able to say this because I ran it
twice.)*

*(For contrast, and to show the guardrails are not decoration: `trap:units` — assay batch B reporting
10× too large — goes from **0/20 without the guardrails to 17/20 with them**. That is the largest
single-task swing in the study. The detector sees the batch, seeds the finding, and the gate will not
let the agent submit until it has been discharged. The machinery works **exactly as designed —
wherever a detector feeds it.**)*

> An aggregate is an average, and **an average is where a failure goes to hide.** If you take one
> methodological point from this document, take that one — it is worth more than any gate in §3.

#### So I tried to fix it with the prompt, and it did not move

The obvious suspect was rule 4 of the system prompt — the one piece of domain knowledge I allow the
agent. It was phrased in the language of a clinical trial:

> *"If the groups were **not randomly assigned**, a raw comparison between them is not a **treatment
> effect**…"*

Nothing in an e-commerce table is "randomly assigned" and there is no "treatment," so the reflex had
nothing to attach to. Plausible. Testable. I rewrote it in fully domain-neutral terms — *"this
applies to EVERY grouped comparison: arm vs arm, channel vs channel, segment vs segment"* — and
re-ran the benchmark.

**It bought nothing measurable.** (D29 keeps the receipt; and note that the *first* time I checked
this I read the difference off a single n=10 run and concluded far too confidently in both
directions — which is D31 in miniature.) I kept the new wording, because it is the more honest rule,
but it is not what closes this gap.

Which is a negative result, and the most useful one in the project, because it forecloses the cheap
explanation and leaves only the expensive one:

> ## You cannot close the *notice* gap with better advice.
>
> The Findings Ledger works on sentinels and duplicates (**95–100%**) because a twenty-line profiler
> **detects** them and hands the agent an obligation it cannot walk past.
>
> There is **no detector for confounding.** Nothing computes *"are these groups imbalanced on some
> third variable?"* So nothing is seeded, so the gate has nothing to gate, so the agent never
> notices — and **rewording the instruction does not manufacture an observation.**
>
> A structural gate can force an observation to **reach** a decision. It cannot **create an
> observation that was never made.**

#### The ambiguity pair, which says the same thing more precisely

I have two under-specified questions, and they behave completely differently — which is more
informative than either alone:

| | | full |
|---|---|---|
| `s13_ambiguous` | *"Was the campaign successful?"* — successful by **which measure**? | **85%** |
| `b3_ambiguous` | *"Did the biomarker improve?"* — and for a biomarker, **lower is better** | **5%** |

The `question_is_precise` boolean is a *required* field, so the model cannot skip the judgement. On
`s13` it works: the ambiguity is right on the surface (revenue? conversion? per-customer?), the model
sees it, declares the question imprecise, and states its reading.

On `b3` it fills the field in and answers **`True`**. It sincerely believes *"did the biomarker
improve?"* is a precise question — because the thing that makes it imprecise is not in the question
at all. It is **in the semantics of the column**: for this assay, a *fall* is an improvement. Nothing
in the data profile says so, and the model does not know it.

> So it is not that there is "no detector for ambiguity." **`s13` proves the gate works when the
> ambiguity is visible.** `b3` fails because the ambiguity is invisible *from the data* — it lives in
> domain knowledge the profiler cannot compute and the model does not have.
>
> Which is the same lesson one level down: **the gate is fine. The observation never arrived.**

That is the honest boundary of this whole approach, and it says exactly what to build next — not a
better gate, and not a better prompt:

| detector to build | ~lines | what it would seed as an open obligation |
|---|---|---|
| **confounding** — cross-tab every candidate grouping column against every other; flag imbalance | ~20 | *"`channel` is imbalanced on `customer_segment` (χ² = …). A raw channel comparison is not a channel effect."* |
| **ambiguity (surface)** — does the question pin down a population, a measure, and a direction? | 3 `if`s | *"the question does not name a measure."* |

Each is the same twenty lines of deterministic pandas already worth **−27%** everywhere else in this
table. **The detector is the cheapest thing in this design, and it is the only thing that moves the
number.**

#### And one failure I would not try to fix in the agent at all

`b3_ambiguous` is 5% — one run in twenty, which is indistinguishable from luck — and no detector
will save it. The fact that makes *"did the biomarker improve?"*
ambiguous — **that a fall in this assay is an improvement** — is not in the data, not in the column
names, not in the question, and **not in `data_dictionary.md` either.** I checked. It exists nowhere
in anything the agent is given.

> **That is not an agent failure. It is a documentation failure**, and the correct fix is one line in
> the data dictionary, not one more gate in the agent.
>
> Worth saying out loud, because the reflex in a project like this is to attribute every red cell to
> the model. Some of them are yours. An agent cannot notice what nobody wrote down — and if your
> answer to that is a cleverer agent, you have misdiagnosed the bug.

### 4.6 Known limits of this evaluation

Stated plainly, because the papers are scrupulous about theirs and being caught hiding one is worse
than having one:

- **The judge is not validated at all.** 5 of 28 tasks are graded by an LLM (the behavioural ones no
  arithmetic can settle). DDB validated theirs against two other judges at κ=1.0 on 200 responses. I
  read mine's calls and they looked right, which is worth precisely what it sounds like. This is the
  weakest link in the evaluation, and it is why the judge decides five tasks and not one more —
  everything else is `math.isclose`.
- **The ablation grid cannot separate the ledger from the seeding it feeds** (§4.4, caveat 3). One
  extra config would fix it. I did not run it.
- **"4,480 runs" is not 4,480 independent trajectories.** The cache key omits the config name, so an
  ablation that does not change the prompt until a gate *fires* replays the full agent's cached
  trajectory at the same nonce. This is common-random-numbers pairing — it makes the paired CIs
  *more* precise, not biased — but the headline count overstates independent work, and cross-config
  **cost** comparisons are meaningless.
- **And the same cache makes `evals/report.py` lie about money — including to me.** It prints
  `$3.75`, because **77% of the rows are cache replays that record `$0`.** That is the *marginal*
  cost of re-running the grid, **not what it cost to produce.** On billed runs only, an analysis
  costs **$0.004** and the grid cost **~$16** — the number reported in §4.4.
  > I nearly "corrected" §4.4 *down* to $3.75 on the grounds that **the script is right and the
  > prose is stale** — the tie-break rule I state in §4.4 and mean. It is still the right rule for
  > every *rate* in this document. It is the wrong rule for a *sum*, because **a cache deflates a
  > sum and cannot deflate a proportion.**
  >
  > Which is the lesson of §4.4.1 arriving a third time, wearing yet another hat: **the instrument
  > needs checking too, and "trust the script" is itself an instrument.**
- **The central thesis is a bet, not a result.** *"Scaffolding beats model size"* is why this runs on
  a model 14× cheaper than the best one available (§6). I never ran the big model with no
  scaffolding, so I have not shown it. One command, ~$15.
- **I built the traps, so I know them.** The mitigation is the held-out domain — which **caught me
  twice** (D28, D29), which is the only evidence I have that the mitigation works.
- **Synthetic data is not the real world.** It is, however, data whose ground truth I can actually
  compute — which the real world rarely offers, and which is exactly why GeneBench-Pro simulates too.
- **Ablations are single-mechanism.** No interactions tested.

---

## 5. What this does not do, and when I'd change my mind

Restraint is only credible if it comes with a trigger.

| Not built | Why not | I'd build it when |
|---|---|---|
| Multi-agent (planner / coder / critic) | The state that matters lives in one kernel namespace. Every handoff forces it through the lossy channel of natural language — the exact channel this design works to minimize. Both papers evaluate single agents; DDB gets 51.6% with one. | The task genuinely decomposes into independent subproblems with narrow interfaces (DDB's own future-work suggestion: route *"structural reasoning, database retrieval, patent mining, target genetics, and verification subproblems"* to different stacks). |
| A framework (LangChain etc.) | It would hide precisely the parts I'm being asked to show judgment on: context assembly, truncation, error recovery. And with an OpenAI-compatible endpoint, there is exactly one integration to write. | Dozens of tool integrations, or a team that needs shared conventions more than it needs transparency. |
| RAG / vector store | The agent's problem is not retrieval of documents. It is reasoning over a file it already has. | Domain knowledge that isn't in the weights and can't fit in a prompt — e.g. an internal assay-methods wiki. |
| Real sandbox (container / microVM) | Out of scope for a prototype; the tool contract is designed so it swaps in behind the same seam. | Before a single line of untrusted input. Non-negotiable in production. |
| Fine-tuning | I have no training data, and the papers show the gap is procedural, not knowledge-shaped. GBP's own reasoning-effort sweep buys more than a fine-tune plausibly would. | Once the eval harness has produced a few hundred graded trajectories — *then* the eval is the dataset. |

The last row is the real roadmap: **the evaluation harness is the thing you build first, because it
is also the thing that tells you what to build next.**

---

## 6. Model choice — and why I deliberately did *not* pick the best one

Token Factory hosts, among others, GLM-5.2, Kimi-K2.7-Code, DeepSeek-V4-Pro and MiniMax-M3 — which
happens to be almost exactly DrugDiscoveryBench's open-model results (Figure 7, p.22):

| Model | DDB pass rate | On Token Factory |
|---|---|---|
| **GLM 5.2** | **37.8%** | ✅ |
| Kimi K2.7 Code | 35.3% | ✅ |
| DeepSeek V4 Pro | 31.7% | ✅ |
| Qwen 3.7 Max | 29.3% | ✅ |
| MiniMax M3 | 23.2% | ✅ |

So the choice does not have to be a vibe. On the evidence, the pick is **GLM-5.2**.

**I did not pick GLM-5.2.**

**Agent: `Qwen/Qwen3-30B-A3B-Instruct-2507`** — a model **14× cheaper** and several rungs down the
ladder. **Verifier/judge: `openai/gpt-oss-120b`** — a different family from the agent, to avoid
self-preference bias in review (D11).

Here is why, and it is the whole argument of this document in one move.

The thesis is that **reliability comes from the scaffolding, not from a bigger base model.** If I
run that thesis on the strongest model available, I can never test it — every success is
attributable to the model, and the design proves nothing. **Handicapping the base model is what
turns the claim into an experiment.** It is also what makes a 4,480-run ablation study cost about
$16 instead of $110.

> The cheapest demonstration and the most rigorous one are, here, the same demonstration.

**What this does not yet show.** That a small model *with* the scaffolding beats a big model
*without* it. That is the direct test, and **I have not run it** — it is one command
(`--config no_guardrails --model zai-org/GLM-5.2`) and about $15. It is the first thing I would
spend the next budget on, and until I do, the sentence above is a design bet with supporting
evidence, not a demonstrated fact. I would rather label it that way than let a code comment imply
otherwise. (It did, for a while. See D18.)

---

## 7. Reading order

| Notebook | Adds | The failure that motivates the next one |
|---|---|---|
| `01` Hello, model | one API call; a `llm()` helper; the cost meter | it invents a mean for a file it cannot see |
| `02` Tool calling | the raw protocol, by hand | one tool, one round trip — no autonomy |
| `03` The agent loop | ReAct in ~30 lines | fixed tools cap what it can answer |
| `04` `run_python` | the persistent kernel; self-debugging | on the clinical data it returns a **confidently wrong** answer |
| `05` Keeping the thread | Question Contract + **Findings Ledger** | the answer is right but the numbers are unsourced |
| `06` The gated exit | grounding gate + verifier + structured report | "it worked once" is not evidence |
| `07` Evaluation | the benchmark, the ablations, the failure taxonomy | — |

Chapter 4 is the pivot. Everything before it is mechanics and the agent looks impressive. Then it
meets data that looks like real science, and it fails the way the papers say it will — and chapters
5 and 6 are the response.

---

*Design decisions, with alternatives and the conditions under which I'd reverse them: [DECISIONS.md](DECISIONS.md).*

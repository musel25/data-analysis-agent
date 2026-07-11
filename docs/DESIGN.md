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

The whole agent is ~200 lines of Python. That is a claim about where the difficulty actually lives,
and it is defended in [DECISIONS.md](DECISIONS.md).

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
programmatic, against absolute numeric tolerances; ten attempts per problem.

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

Crucially, they hand-classified **226 failing runs into a taxonomy with frequencies** (Table 3, p. 11):

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
> we find that at least 1 of the agents is able to pass 80 out of 82. The results suggest that
> **execution is within reach for today's agents should they be given the expert workflow**." (p. 14)

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
                              │   base model      │  GLM-5.2 @ Token Factory
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
   the first message. The agent never starts blind, which directly attacks DDB's *Retrieval* failures
   (16.4% — "failing to read a provided file"). It also saves two or three turns of budget.
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
    estimand:    str        # the exact quantity, in one sentence
    population:  str        # WHICH ROWS. the denominator. stated explicitly.
    units:       str        # g? kg? percent? fraction? log-scale?
    constraints: list[str]  # every explicit filter/limit the user stated
    ambiguities: list[str]  # what could be read two ways, and how I read it
```

It is pinned into context **every turn** and it is shown to the verifier at submit time.

Why this earns its place: DDB's melanoma failure is a model that silently stopped applying the
melanoma qualifier three steps into a ranking task. GBP's DRX1 problem is trapped precisely on the
denominator (compute over *tested partners* and you get a plausible, wrong answer; the target is the
*full roster*). In both cases the question itself decayed. Writing it down as structured state, and
re-showing it, makes decay visible — to the agent, to the verifier, and to me reading the trace.

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

**Honest limitation:** the ledger cannot force the agent to notice something it never looked at. It
converts *noticed-but-not-acted-upon* into a hard stop; it does nothing about *never-noticed*. That
is why the deterministic data briefing exists — it front-loads the cheap, mechanical diagnostics
(nulls, dtypes, cardinality, sentinel-looking values) so that the most common findings are *handed*
to the agent rather than depending on its curiosity. Between the two, the coverage is decent. It is
not complete, and I would not claim otherwise.

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
(agent: GLM-5.2; verifier: Qwen3.5 — cross-family, because a model asked to review its own work is
subject to self-preference bias, Zheng et al. 2023).

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

- **Step budget** (max 12). When 3 steps remain, every observation is suffixed with
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

~14 tasks over two datasets. Ground truth is a pandas callable that only the grader ever invokes.

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
| A question whose premise is false | answer it anyway | flag the false premise | DDB scope |

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

LLM-as-judge is used **only** for the two behavioural tasks that no numeric check can grade (did it
flag the false premise? did it surface the ambiguity?), with a binary rubric, temperature 0, a
different model family than the agent, and — following DDB, who validated their judge at κ=1.0
against two other judges — a small agreement check against my own hand labels before I trust it.

### 4.3 What gets measured

Per task, **5 runs** (both papers use 3–10; agent evals are noisy and a single run proves nothing):

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

### 4.4 The ablations — the actual argument

This is how each design decision stops being an opinion:

| Ablation | Prediction |
|---|---|
| **Findings Ledger off** | Wrong-attractor rate rises sharply on the trap tasks. *This is a miniature replication of GeneBench-Pro's central finding, on my own agent.* |
| **Grounding gate off** | Ungrounded numbers appear in answers; derivation slips survive to the report |
| **Question Contract off** | The constrained-scope task and the Simpson's task degrade |
| **Observation truncation off** | Token cost explodes; accuracy drops (long-context degradation) |
| **Verifier off** | Fewer caught slips; cheaper |
| **No-execution baseline** (one LLM call, given only `df.head()` and the question) | Near-total failure on the trap tasks — this is the "does the loop earn its cost" control |

If an ablation shows a mechanism does not pay for itself, it gets cut from the design. That is the
deal, and I would rather report that honestly than ship a guardrail that does nothing.

### 4.5 Known limits of this evaluation

Stated plainly, because the papers are scrupulous about theirs and being caught hiding one is worse
than having one:

- **n = 14 tasks.** Small. One task flipping is a 7-point swing. Findings are directional, not
  precise; I report ranges, not just means.
- **I built the traps, so I know them.** There is an overfitting risk in tuning the system prompt
  against my own benchmark. Mitigation: 3 tasks are held out and never looked at during development.
- **One synthetic dataset** is not the real world. It is, however, a dataset whose ground truth I can
  actually compute — which the real world rarely offers, and which is exactly why GeneBench-Pro
  simulates too.
- **The judge is only validated on 5 hand-labelled examples.** That is enough to catch a broken
  judge, not enough to certify a good one.

---

## 5. What this does not do, and when I'd change my mind

Restraint is only credible if it comes with a trigger.

| Not built | Why not | I'd build it when |
|---|---|---|
| Multi-agent (planner / coder / critic) | The state that matters lives in one kernel namespace. Every handoff forces it through the lossy channel of natural language — the exact channel this design works to minimize. Both papers evaluate single agents; DDB gets 51.6% with one. | The task genuinely decomposes into independent subproblems with narrow interfaces (DDB's own future-work suggestion: route structure / retrieval / cheminformatics to different stacks). |
| A framework (LangChain etc.) | It would hide precisely the parts I'm being asked to show judgment on: context assembly, truncation, error recovery. And with an OpenAI-compatible endpoint, there is exactly one integration to write. | Dozens of tool integrations, or a team that needs shared conventions more than it needs transparency. |
| RAG / vector store | The agent's problem is not retrieval of documents. It is reasoning over a file it already has. | Domain knowledge that isn't in the weights and can't fit in a prompt — e.g. an internal assay-methods wiki. |
| Real sandbox (container / microVM) | Out of scope for a prototype; the tool contract is designed so it swaps in behind the same seam. | Before a single line of untrusted input. Non-negotiable in production. |
| Fine-tuning | I have no training data, and the papers show the gap is procedural, not knowledge-shaped. GBP's own reasoning-effort sweep buys more than a fine-tune plausibly would. | Once the eval harness has produced a few hundred graded trajectories — *then* the eval is the dataset. |

The last row is the real roadmap: **the evaluation harness is the thing you build first, because it
is also the thing that tells you what to build next.**

---

## 6. Model choice, and why it's a measured decision

Token Factory hosts, among others, GLM-5.2, Kimi-K2.7-Code, DeepSeek-V4-Pro and MiniMax-M3 — which
happens to be almost exactly DrugDiscoveryBench's open-model leaderboard:

| Model | DDB pass rate | On Token Factory |
|---|---|---|
| GLM 5.2 | 37.8% | ✅ |
| Kimi K2.7 Code | 35.3% | ✅ |
| DeepSeek V4 Pro | 31.7% | ✅ |
| MiniMax M3 | 23.2% | ✅ |

So the model choice does not have to be a vibe. **Agent: `zai-org/GLM-5.2`** — the best-performing
open model on a published agentic-science benchmark, among those available on the platform I'm using.
**Verifier/judge: `Qwen/Qwen3.5-397B-A17B`** — deliberately a different family, to avoid
self-preference bias in review.

I verified all six candidates handle the two-round tool-calling loop correctly before choosing
(`notebooks/00_setup.ipynb`); none of them fumbled the protocol, so the choice rests on the benchmark
and on cost, not on plumbing.

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

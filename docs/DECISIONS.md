# Decision Log

Every non-obvious choice in this system, with what I considered instead and **what would change my
mind**. A simplification without a stated reversal condition is just an omission wearing a suit.

Papers referenced: **GBP** = GeneBench-Pro (Li & Ho, OpenAI, bioRxiv 2026); **DDB** =
DrugDiscoveryBench (Akyürek, Tu et al., Scale AI & Phylo, 2026).

---

## Framing

### D01 — Build the agent from scratch; no framework
**Considered:** LangChain / LlamaIndex / smolagents; OpenAI Agents SDK.
**Chose:** raw `openai` client + a `while` loop.
**Why:** The assignment asks what I would build *on top of the base model* and how I'd know it
works. A framework hides exactly those parts — context assembly, truncation, error recovery, the exit
condition — behind decisions someone else made. With an OpenAI-compatible endpoint there is precisely
one integration to write, so a framework buys nothing here and costs transparency. The whole agent is
~1,000 lines of code (comments and docstrings excluded — most of the file volume is the reasoning
behind each choice); that is itself a claim, and it's checkable.
**Revisit when:** dozens of tool integrations are needed, or a team needs shared conventions more than
it needs to see the machinery. Then adopt one *with clear eyes*, having built the thing it abstracts.

### D02 — Single agent, not a multi-agent system
**Considered:** planner / coder / critic; a router across specialist agents.
**Chose:** one loop.
**Why:** The state that matters in a data analysis is not conversational, it is *material* — the
dtypes discovered at step 2, the filter applied at step 4, the dataframe living in the kernel. Split
the work across agents and that state must pass through natural language, which is the lossiest, most
expensive channel available and the exact one this design spends its effort minimizing. Both papers
evaluate single agents; DDB reaches 51.6% with one. Neither paper's failure taxonomy contains a
single failure that a second agent obviously fixes — they contain failures that *deterministic
guards* fix.
The one genuine benefit of another agent — an opinion not anchored on the first agent's reasoning —
is purchased for a single API call by the fresh-context verifier (D11).
**Revisit when:** the task decomposes into genuinely independent subproblems with narrow interfaces.
DDB's own future work suggests exactly this shape: route *"structural reasoning, database retrieval,
patent mining, target genetics, and verification subproblems"* to different model-tool stacks.

### D03 — Jupyter, not marimo
**Considered:** marimo (reactive, pure-`.py`, no hidden state, ships as an app).
**Chose:** Jupyter.
**Why:** The deliverable is a *walkthrough*. Jupyter renders on GitHub with outputs visible, so a
reviewer who never runs a cell still sees every result; the audience knows it; and top-to-bottom
markdown-and-code storytelling is exactly the format. marimo's headline feature — banning variable
redefinition across cells — actively fights an incremental buildup where `run_agent` is deliberately
rewritten three times.
**Revisit when:** the artifact stops being a lesson and becomes a tool someone uses. Then marimo's
app mode is the better shape, and its no-hidden-state guarantee stops being a constraint and starts
being a feature.
**Note the irony, out loud, in ch. 4:** the hidden-state problem marimo solves is the *same* problem
the agent's persistent kernel has. Worth saying.

---

## The agent

### D04 — One general tool (`run_python`), not many specific ones
**Considered:** a toolbox — `describe`, `groupby`, `correlate`, `plot`, `filter`.
**Chose:** one code-execution tool.
**Why:** Specific tools cap the agent's ceiling at whatever verbs I thought of in advance, and every
tool is new surface area for the model to misuse. Both papers give their agents a general Python
environment (GBP: pandas/scipy/statsmodels in a container; DDB: BIOMNI exposed *as an ordinary Python
library*, explicitly not as MCP tools). And DDB's evidence is decisive: their agents fail from
*losing the thread*, not from lacking affordances.
**Revisit when:** an operation must be guaranteed correct rather than merely likely — e.g. a
regulatory statistical test where you want a vetted implementation, not model-authored code.
Then that one operation becomes a real tool, and the rest stays open.

### D05 — `exec()` into a persistent namespace; call it what it is
**Considered:** (a) fresh subprocess per call — no state, so `df` must be re-read every step;
(b) `jupyter_client` subprocess kernel; (c) Docker per session.
**Chose:** in-process `exec(code, self.ns)` with a persistent dict.
**Why:** Persistence is not a nicety — it is what makes the agent's steps compose the way a human
analyst's do (load once, filter, inspect, model). Stateless execution forces every step to rebuild
the world and doubles the token cost of everything. In-process `exec` is ~20 lines, which is what
makes it *teachable*, which is the point of the artifact.
**And it is not a sandbox.** The notebooks carry a red warning box saying so. Both papers ran in
Docker with no network; that is the correct production answer. The `run_python` contract (code in,
stdout out) is deliberately identical to what a container-backed executor exposes, so the upgrade is
one class swap, not a redesign. Overclaiming safety would be worse than the limitation.
**Revisit when:** any input is not written by me. Before *one line* of untrusted input, this becomes
a container with no network, read-only mounts, and CPU/memory/wall-clock limits.

### D06 — Truncate observations head **and** tail
**Considered:** no cap; head-only; an LLM summarizer per observation.
**Chose:** deterministic head+tail cap (~1,500 chars) with an instructive marker.
**Why:** Head-only truncation cuts the last line of a traceback — the only line that names the error.
An LLM summarizer inserts a hallucination-capable, latency-adding call into *every single step* to
compress text that is usually just a dataframe repr. The marker itself is a teaching device: it tells
the model what to do differently next time (*"assign and inspect selectively"*), so the truncation
trains behaviour rather than merely hiding text.
**Revisit when:** observations become prose-heavy (log analysis, document QA), where semantic
compression genuinely beats positional truncation.

### D19 — The executor echoes the last expression, like a REPL
**Found by running it, not by designing it.** A bare `exec()` swallows bare expressions. So when
the agent wrote:

```python
df = pd.read_csv('trial.csv')
df.head()          # a Jupyter user expects to SEE this
```

...it got back **nothing**. It believed it had looked at the data. It had gone blind. I watched a
real run do exactly this, then guess the column names from thin air (`treatment`, `response`), hit
a `KeyError`, and guess again — burning four steps to learn what one working `head()` would have
told it.

**Chose:** parse with `ast`; if the final statement is an expression, `eval` it and echo the repr —
exactly what a Jupyter kernel does.
**Why it matters beyond the bug:** the model was not being stupid. It was assuming Jupyter
semantics, which is entirely reasonable — every notebook on the internet behaves that way, and the
tool is *called* `run_python`. The bug was mine: **my tool lied about what it was.**

The general lesson, and the reason this gets its own entry: when a model misuses a tool, the first
question is whether the tool behaves the way its name and its ecosystem imply. A tool that
surprises will be misused, and the fix belongs in the tool, not in a sterner system prompt. I could
have "fixed" this by adding *"ALWAYS use print()"* in capital letters to the prompt and it would
have worked less well, cost more tokens, and taught me nothing.
**Revisit when:** never, I hope. But it is the reason `docs/DESIGN.md` insists that the observation
channel is a *design surface*, not a given.

### D20 — Two death-loop guards, escalating
**Found the same way.** The same run re-submitted **byte-identical failing code** at step 2 and
step 4.
**Chose:** (1) a duplicate-code short-circuit — identical code returns the cached observation
without executing, because running it again cannot produce a different answer, only a smaller
budget; (2) error fingerprinting — the same `(error, code)` pair twice injects *"state the root
cause in one sentence before writing more code"*; three times escalates to *"this is not a typo,
it is the approach — change strategy."*
**Why the two stages:** jumping straight to "try something else" makes the agent abandon a
95%-correct approach over a typo. Force a *diagnosis* first. This is Reflexion (Shinn et al. 2023)
in about ten lines, and it works here for the same reason the Findings Ledger works — the feedback
is grounded in something that actually happened, not in introspection.

### D07 — A deterministic data briefing before turn 0
**Considered:** let the agent discover the schema itself with its first two or three tool calls.
**Chose:** plain Python computes shape / dtypes / head / nulls / cardinality and injects it as the
first message.
**Why:** DDB's *Retrieval* failures are 16.4% of all failures and their definition includes "failing
to read a provided file." The cheapest possible fix is to not make reading the file optional. It also
saves two or three turns of budget on every single run, and it means the agent can never begin from a
hallucinated guess about the schema.
**Revisit when:** there are many files, or files too large to profile cheaply — then the briefing
becomes lazy/sampled, but it does not disappear.

---

## The three ledgers — the load-bearing part

### D08 — The Question Contract (estimand / population / units / constraints / ambiguities)
**Attacks:** DDB *Constraint* failures (7.5%) and the melanoma scope-drop; GBP's "valid model applied
to the wrong population or scale."
**Considered:** just putting "restate the question" in the system prompt as advice.
**Chose:** a pydantic contract the agent must fill before analysing, pinned into context every turn,
and shown to the verifier at the end.
**Why:** Advice in a system prompt decays over a long trajectory — which is precisely the failure DDB
documents (a qualifier, in their words, *"at some point"* silently stops being applied). Structured
state does not decay: it is re-rendered every turn from a variable, not remembered from a paragraph.
The `population` field gets its own line because **one of the three decision points** in GBP's worked
DRX1 example is a denominator trap — compute over the tested subset and you get a plausible, wrong
number.
**Revisit when:** it proves to be dead weight — if the ablation shows no benefit, it gets cut. That is
the deal for every mechanism here.

### D09 — The Findings Ledger, and blocking `submit_answer` on open findings
**Attacks:** the notice–act gap — GBP's central finding; DDB *Domain reasoning* (54.0%).
**Considered:** (a) prompt the model to "check your assumptions"; (b) a reflection step at the end;
(c) nothing.
**Chose:** a tool (`note_finding`) that appends `{observation, implication, status}` to a ledger, and
a **hard block on submitting while any finding is `open`**. Closing a finding requires either acting
on it (naming the code step) or dismissing it with a written reason. Both are recorded and both ship
in the final report.
**Why:** This is the whole design in one mechanism, so it deserves the space. GBP's finding is not
"the model fails to notice." It is:

> "the agent notices the relevant local diagnostic clue but treats it as a local data cleaning issue
> rather than as evidence that should change the downstream statistical method and QC pipeline."

A prompt that says "check your assumptions" does not fix that, because the model *did* check. The
failure is that noticing has no consequences. So give it consequences: make a noticed finding an
**open obligation that blocks the exit**, and force the agent to write down `implication` — what this
changes — which is the precise cognitive step the papers observe being skipped. An end-of-run
reflection is too late and too weak; Huang et al. (2023) show introspective self-correction without
external grounding barely works. This works because it is not introspection — it is a state machine.
**Honest limitation:** it converts *noticed-but-ignored* into a hard stop. It does nothing about
*never-noticed*. That is exactly why D07 (the deterministic briefing) exists — to hand the agent the
mechanical findings rather than depend on its curiosity. Together the coverage is good, not complete.
**Revisit when:** the ablation says it does not move the wrong-attractor rate. Then it is theatre and
it goes.

### D10 — Deterministic numeric grounding, checked before any LLM review
**Attacks:** DDB *Derivation error* (18.6%) and *Final-answer slip* (3.5%).
**Considered:** ask an LLM to verify the numbers.
**Chose:** regex the numbers out of the answer's evidence, normalize (commas, %, 4 significant
figures, ×100 and ÷100 variants), and require set-containment in the concatenated stdout of the run.
Reject with a specific message naming the ungrounded values.
**Why:** DDB documents a frontier model whose *"own code printed the correct group-level count of 1,
but the final tally used the atom-level count of 2."* An LLM reviewer *might* catch that. A regex
catches it every time, costs nothing, adds no latency, and cannot be argued out of its position.
Cheap and deterministic goes before expensive and probabilistic — always.
**Known weakness, and how it's handled:** the check false-negatives on formatting mismatches and
false-positives on trivial numbers (years, indices). So it *hard*-gates only the `evidence` list, and
merely warns on numbers appearing in the prose `answer`. A gate that fires wrongly gets disabled by
its user, so it is deliberately tuned to be conservative.

### D11 — A verifier with a fresh context, from a different model family
**Attacks:** DDB — *"The last chance to catch the slip is at the final answer… None of the failing
models caught this."*
**Considered:** (a) no verification; (b) self-review in the same context; (c) a persistent critic
agent.
**Chose:** one call, different model family, seeing **only** the Question Contract, the data briefing,
the executed code + outputs, and the draft answer — explicitly **not** the agent's reasoning.
**Why:** A reviewer that reads the agent's chain of thought is anchored by the agent's narrative and
rubber-stamps it; a verifier that sees the transcript measures nothing. The fresh context *is* the
mechanism, not an implementation detail. Different family because a model reviewing its own output
shows self-preference bias (Zheng et al. 2023, LLM-as-judge). And it is the *fourth* gate, after three
deterministic ones — never pay for an LLM call to find something a regex would have caught.
**Revisit when:** cost matters more than the last few points of accuracy. It is the most expensive
guardrail here and the first one I would drop under budget pressure — which is exactly what the
ablation is for.

### D21 — Pre-seed the Findings Ledger from the deterministic checks
**Found by running it.** The Findings Ledger's first real outing on the Simpson's-paradox task
caught the duplicate patients, forced the agent to act on them — and produced **−0.087**, the naive
answer. It had never noticed the confounding, so there was no obligation to discharge and the gate
had nothing to block. The ledger did *exactly* what it was designed to do, and the answer was still
wrong-signed.

That is the limitation I had already written down in D09 (*it converts noticed-but-ignored into a
hard stop; it does nothing about never-noticed*) arriving in person.

**Chose:** anything the deterministic profiler can find — sentinels, duplicate identifiers,
numeric-looking strings — is **pre-registered in the ledger as an OPEN finding** before turn one.
Not printed in the briefing as a helpful note. *Entered as an obligation.*
**Why the distinction is the whole point:** **information can be ignored; an obligation cannot.**
The briefing already *contained* these facts and the agent scrolled past them. Same facts, different
structural status, different outcome.

This is the division of labour the design rests on:

| | does what |
|---|---|
| deterministic code | finds the mechanical problems, and makes them **un-ignorable** |
| the model | decides what they **mean** for this particular question |

Note the agent is still free to look at a seeded finding and **dismiss** it — and on the
response-rate task it correctly dismissed the `-999` sentinels, because `biomarker_baseline` never
enters that calculation. We are not forcing a conclusion. We are forcing the observation to *reach*
the decision. That distinction is what separates a guardrail from a straitjacket.

### D22 — One rule of domain knowledge in the system prompt, and only one
**Chose:** *"Before comparing two groups, check that they are comparable. If they were not randomly
assigned, a raw comparison is not a treatment effect — it is a comparison of two different
populations."*
**Is this teaching to my own test?** The check I hold myself to: it names **no column, no dataset,
and no paradox.** It states a general principle of causal inference that applies to any two-group
comparison anywhere. If it mentioned `severity` it would be overfitting, and the three held-out
tasks exist to catch me doing that.
**Why it's justified at all:** DrugDiscoveryBench re-ran their unsolved tasks with the expert's
step-by-step playbook supplied as a hint and went from 76/82 to 80/82 — *"execution is within reach
for today's agents should they be given the expert workflow."* The models can execute. What they
lack is the analyst's **reflex** — the thing a statistician does without being asked. So encode the
reflex. **That is what "building on top of the base model" means: not a better model, a better
procedure.**

### D23 — Force the ambiguity judgement with a required boolean
**Found by running it.** The contract had an `ambiguities` list. On *"Did the biomarker improve?"* —
a question with at least three defensible readings — the agent left it **empty, every single time**,
and silently picked one. The field designed to prevent the failure was the field being skipped.
**Chose:** a **required** `question_is_precise: bool`, plus a validator: if it is `False`, the
`ambiguities` list *cannot* be empty.
**Why:** a field the model may leave empty is a field the model **will** leave empty. A required
boolean cannot be skipped — it forces an explicit judgement rather than inviting one. This is the
Findings Ledger trick applied to the contract: **make the omission impossible to express, rather
than asking nicely for it not to happen.**
Also: the contract's ambiguities are now folded into the report's `caveats` **by the harness**, not
by the model. The agent *was* recording its interpretation and then never mentioning it in the
answer — and reasoning that doesn't reach the reader is, from the reader's side, indistinguishable
from reasoning that never happened.

### D12 — Cache every LLM response to disk, keyed by request hash
**Considered:** live calls only.
**Chose:** a ~10-line disk cache with a `LIVE` flag; the cache is **committed to the repo**.
**Why:** Three things at once. (1) Every notebook runs top-to-bottom, deterministically, offline, for
free — a reader with no API key still sees the whole thing work. (2) The walkthrough cannot be
sabotaged by a rate limit, an outage, or a model having an off day. (3) It is honest: flipping
`LIVE=True` in front of the audience re-runs it for real.
And there is a fourth, better reason, revealed in ch. 7: **this is how you write cheap deterministic
tests for a non-deterministic system.** The demo insurance and the testing strategy are the same
artifact.

---

## Evaluation

### D13 — Programmatic binary grading; LLM judge only where nothing else works
**Chose:** GBP's exact protocol — structured answer, pre-specified absolute tolerances,
`math.isclose`, all fields must pass, no partial credit on the headline metric.
**Why:** GBP defends this better than I could: *"an agent that executes several intermediate steps
correctly but returns the wrong decision-relevant answer has not successfully automated the
analysis."* Partial credit measures effort; binary measures usefulness. A judge is reserved for the
two behavioural tasks where no arithmetic can decide (did it flag the false premise? did it surface
the ambiguity?) — binary rubric, temperature 0, different model family from the agent.
**The honest gap:** DDB validated their judge against two others at κ=1.0 on 200 responses. **I did
not validate mine.** I read its calls on the behavioural tasks and they looked right, and that is
worth exactly as much as it sounds. Five of 28 tasks rest on an unvalidated judge, and an unvalidated
judge is a random number generator with good manners. Everything else is `math.isclose`, which is why
I kept the judge's blast radius that small.
**Accepted cost:** binary grading throws away stage-level diagnostic signal — GBP concedes this too.
The trace log (D16) is where that signal is recovered.

### D14 — Every task has a documented *plausible-but-wrong* answer, and the gap is asserted
**Chose:** for each task, record the naive path's answer and `assert abs(naive - correct) >> tolerance`.
**Why:** Straight from GBP's design principles — *"Results from analyses involving plausible but
incorrect decisions… are checked via ablation and verified to be sufficiently distinct from the graded
answer."* Without this, a lazy analysis can land inside the tolerance band and be graded correct, and
the benchmark measures nothing.
It also buys a metric nobody reports: the **wrong-attractor rate**. A failing run isn't just wrong —
I can say *which* wrong it is. Landing on the naive answer means the agent fell into the notice–act
gap specifically, not that it fumbled arithmetic. That distinction is the difference between a
benchmark and a diagnostic.

### D15 — Synthetic dataset with a known DGP, alongside a real one
**Considered:** only real datasets (Titanic, penguins, a Kaggle CSV).
**Chose:** real `penguins.csv` for teaching mechanics; synthetic `trial.csv` for teaching trust.
**Why:** Real data has no ground truth I can compute — only ground truth I can *assume*. GBP simulates
for exactly this reason: *"constructively simulated problems where the full causal structure is known."*
Simulation is also the only way to *plant* a decision point (a Simpson's paradox, a batch effect, a
denominator trap) and be certain the naive path gives a different answer.
The two-dataset structure is also the narrative: penguins is where the agent looks brilliant;
`trial.csv` is where it fails the way the papers say it will. That pivot is the whole story.

### D24 — Repeat runs must be independent samples, not cache replays
**A bug I shipped, and my own eval caught.** The first ablation reported three attempts per task.
Attempts 2 and 3 cost **$0.0000**.

Temperature 0 + an identical request = an identical cache key. I was not measuring variance across
three samples. I was **replaying one sample three times and calling it three.**
**Fixed by:** passing the attempt number as a cache nonce, and raising the eval temperature to 0.6.
**Why it matters more than it sounds:** both papers run repeats *precisely because* agent evals are
noisy — GeneBench-Pro uses 10 attempts and bootstraps CIs; DDB uses 3 trials. A benchmark that
silently reports one run as three is **worse than one that honestly reports one**, because it
launders a single lucky trajectory into an apparent consistency result.
The `$0.0000` in the cost column is what gave it away — which is a small argument for building the
cost meter in chapter 1 and printing it everywhere.

### D25 — `set_live()` instead of assigning to the flag
**Another bug found by testing a claim instead of assuming it.** The README promised *"runs offline
with no key — set `LIVE = False`."* I went to verify that sentence before shipping it, and it was
false.

`agentlib/__init__.py` exported the *function* `llm`, which **shadows the *submodule*
`agentlib.llm`** on the package object. So `import agentlib.llm as L; L.LIVE = False` binds an
attribute on a *function* and silently does nothing. The flag never moves. You discover this when
the call you believed was cached bills you.

Also, adding the cache nonce (D24) silently changed the key scheme, orphaning every response cached
before it — so notebooks 01–04 could no longer replay at all until they were re-executed.

**Chose:** stop exporting the bare name `llm` from the package, and expose an explicit
`set_live(bool)`.
**The lesson worth keeping:** a footgun that fails *silently* is worse than one that crashes. Both
of these bugs were invisible from reading the code and obvious the moment I ran the exact thing the
README told a stranger to run.

### D16 — Log every step as JSONL; run each task 10×
**Why:** Agent evals are noisy — GBP runs 10 attempts and bootstraps CIs; DDB runs 3 trials. A single
run is an anecdote. I started at 3× and it was not enough to tell "no effect" from "cannot tell"
(D27); 10× matches GBP and costs $8. Per-task consistency then tells you *which* tasks are unstable
(they turn out to be exactly the ambiguous and dirty ones, which is reassuring).
The JSONL trace is what turns a failure from "it got it wrong" into "at step 4 it noticed the
sentinels and at step 6 it computed the mean anyway" — which is the only kind of failure report you
can act on.

### D26 — A held-out *domain*, not just held-out tasks
**The single most important thing in the evaluation, and I added it last.**

Every trap in `trial.csv` is one I planted, and every guardrail was designed while staring at that
file. An agent passing it demonstrates that my guardrails work **on the failures I already knew
about**. That is a much weaker claim than the pass rate makes it look, and no number of extra runs
on the same dataset fixes it — more runs shrink the *variance*, not the *bias*.

**Chose:** a second synthetic dataset, `sales.csv`, in a completely different domain — e-commerce,
not medicine. Different columns, different semantics, and traps of the same *species* but a
different animal:

| `trial.csv` (designed against) | `sales.csv` (held-out) |
|---|---|
| `-999` QC-failure sentinel | `-1` = "age not supplied" |
| re-tested patients appear twice | refunded orders still in the file |
| assay batch B is 10× off | **revenue exported as text — `"1,234.56"`** |
| — | six internal QA orders at `999999.99` |
| Simpson's paradox: arm × severity | Simpson's paradox: **channel × customer_segment** |

**And it immediately found a hole in my own benchmark.** `observe.py` has had a detector for
*"numeric column stored as text"* since the first commit, and **not one task ever exercised it.**
I only noticed because writing a new domain forced me to enumerate what the detectors actually
cover. A benchmark built from one dataset tests the mechanisms that dataset happens to provoke.

### D27 — Bootstrap CIs, and a *paired* test for the ablations
**Chose:** hierarchical bootstrap — GeneBench-Pro's *scheme* (*"resampling problems and repeated runs
within each sampled problem"*; they use 20,000 resamples, I use 10,000) — and, for each ablation, a
**paired** bootstrap of the difference against the full agent. The pairing is mine, not theirs: GBP
has no ablations and describes no paired test.
**Why hierarchical:** the runs are not independent. Ten runs of an easy task are not ten
independent successes, and tasks differ enormously in difficulty. Resampling individual runs would
give a CI far too tight and I would believe it.
**Why paired:** both configs ran the *same* tasks, so task-difficulty variance — which dominates
the absolute CIs — cancels out. The paired difference has far more power than either number alone.
That is the test that actually answers *"does this mechanism matter?"*
**What it changed:** the first version of this eval reported `full 87%` vs `no_ledger 87%` and I
nearly concluded *"the ledger does nothing."* At n=45 those two numbers had overlapping CIs about
twenty points wide. The honest statement was never *"it does nothing"* — it was **"I cannot
tell."** A point estimate is how you fool yourself; the CI is what stops you.

### D17 — Classify failures using DrugDiscoveryBench's taxonomy
**Chose:** label failing runs as domain-reasoning / derivation / retrieval / constraint /
final-answer-slip.
**Why:** Inventing my own taxonomy would make my failure profile incomparable to the literature.
Reusing theirs means I can say "my agent's failures are 40% domain reasoning versus their 54%" — a
sentence that means something. Standing on a published taxonomy is cheaper *and* more credible than
inventing one.
**Honest scope:** this is a manual read of the traces, not a validated coding scheme with a second
rater. It orients; it does not certify.

### D28 — I leaked two benchmark answers into my own system prompt
**The third bug I shipped, and the one that would have been most embarrassing to have found for me.**

Rule 5 of the system prompt used to read:

> *Questions smuggle in assumptions: `"which of the FOUR sites..."`, `"WHY do women respond better?"`
> … If a premise turns out to be false — **there are only three sites; women do not in fact respond
> better** — then the correct answer is to SAY SO.*

Those are, verbatim, two of the questions in `evals/tasks.py`. **And the sentence after them gives
the answers.** The same two examples were also sitting in the `premises` field description of the
`set_contract` tool schema (which ships to the model on every call) and in the verifier's rubric.
Three separate channels.

Those two tasks scored **100%** and **90%**. Naturally.

**What makes this worth writing down** is that my leakage guard *passed the whole time*.
`test_leak` checks that no **numeric** ground truth appears in a prompt — and it is correct, and it
still holds. But the ground truth of a *behavioural* task is not a number. It is the sentence "the
premise is false, and here is which one." My guard was pointed at the wrong channel.

> **A leak guard only guards the channel you pointed it at.** Passing it is evidence about that
> channel and about nothing else.

**The control I got lucky enough to already have:** `s9_false_premise` is the same mechanism in the
held-out sales domain, and was never named anywhere. It passed 10/10. So the mechanism does work — I
just could not have *proved* it with the two tasks I was pointing at.

#### The fix that made things worse — and the actual lesson

**First attempt.** I stripped the two examples out and replaced them with the abstract *shape* of a
false premise: *"a count, a direction, an existence claim."* No dataset, no column, no benchmark
question. Contamination gone. Then I re-ran the ablation:

| | contaminated prompt | abstract prompt |
|---|---|---|
| `b1_false_premise_sites` (leaked) | 10/10 | 9/10 |
| `b2_false_premise_sex` (leaked) | 9/10 | 8/10 |
| **`s9_false_premise`** (**clean control**) | **10/10** | **5/10** ← 🚨 |

Removing the leak barely moved the tasks that were *leaked*. It **halved the one that wasn't.**

Which tells you what those examples were actually doing. They were leaking two answers — and they
were also *teaching the reflex*, concretely, to every other task. Trading them for an abstract
description kept the honesty and threw away the instruction.

> **Concrete examples teach. Abstract descriptions of examples do not.**
> The bug was never "examples in the prompt." The bug was **examples drawn from my own benchmark.**

**Second attempt, and the one that shipped.** Concrete examples again — vivid, specific, quoted —
about *warehouses, work shifts and sensor revisions*: nouns that appear nowhere in `trial.csv`,
nowhere in `sales.csv`, and in none of the 28 questions. Asserted, not assumed: a check in the test
suite greps every 4-gram of every benchmark question against the full prompt and tool schema.

`s9` came back. The prompt still teaches; it no longer cheats.

**Then re-running the entire ablation from scratch**, because a prompt change invalidates every
cached response, and half-old numbers are worse than no numbers.

**The general form**, which is the part worth carrying to the next project:

> Prompt examples are training data. If you draw them from your benchmark, you have trained on your
> test set — and you will not feel it happen, because the prompt reads like *advice*.
>
> And you cannot fix it by making the advice vaguer. You fix it by drawing the examples from
> somewhere your benchmark cannot see.

### D29 — The prompt overfit to my benchmark *without naming anything in it*
**And this one you cannot catch by reading the prompt.**

Rule 4 — the single piece of domain knowledge I allow the agent — used to say:

> *If the groups were **not randomly assigned**, a raw comparison between them is not a **treatment
> effect** — it is a comparison of two different populations.*

I defended this in a code comment: it names no column, no dataset, no Simpson's paradox, just "a
general principle of causal inference." That was true, and it was beside the point. The rule is
written in the **language of a clinical trial** — *randomised*, *treatment*, *arm*. On `trial.csv`
the reflex fired. On `sales.csv` — where nothing is "randomly assigned", there is no "treatment",
and the confound is channel × customer-segment — **the reflex did not fire.** On the one task the
entire design exists to solve, the agent is a **coin flip** (45% at n=20), and when it misses it
lands *exactly* on the confounded answer.

**The aggregate hid it completely.** The sales domain scored **85%** — the same as the domain the
guardrails were *designed* against — and I had written that up as reassurance. Twelve easy sales
tasks were carrying the average while the crown jewel failed at 10%.

> An aggregate is an average, and **an average is where a failure goes to hide.**

**Attempted fix:** rewrite rule 4 in fully domain-neutral terms — *"this applies to EVERY grouped
comparison — arm vs arm, channel vs channel, segment vs segment; if the groups differ systematically
in some other variable, the raw difference is that other variable wearing the grouping's clothes."*
Same length. No dialect from any one field. Then re-run all 2,240.

**Result: it bought nothing measurable.**

And here I have to make a confession that is really [D31] arriving early. I checked this on a single
run of n=10, read `1/10 -> 1/10`, and wrote *"it made no difference, and that is the finding."* Then
I re-ran the same benchmark and the same task came back `6/10`. Then `3/10`. **The truth, pooled over
20 runs, is `9/20`.**

| `s4_simpson_sales`, full agent | |
|---|---|
| one run of n=10 | 1/10 |
| another run of n=10 | 6/10 |
| another | 3/10 |
| **pooled, n=20** | **9/20 = 45%** |

So the honest statement is not *"the reword changed nothing"* — I could not have known that from what
I had. It is: **the reword produced no effect I am able to detect, and the task itself is a coin flip
with enormous run-to-run variance.**

I am keeping the new wording, because it is the more honest rule. But it is not what closes this gap,
and I was briefly certain of that for entirely the wrong reason.

**And the negative result is worth more than the fix would have been**, because it forecloses the
cheap explanation. The agent does not fail `s4` because the instruction was phrased for doctors. It
fails because **nothing ever tells it the groups are imbalanced.** There is no detector for
confounding — nothing computes *"is `channel` imbalanced on `customer_segment`?"* — so nothing is
seeded, so the gate has nothing to gate.

> ## You cannot close the *notice* gap with better advice.
>
> Sentinels and duplicates land at 90–100% because a twenty-line profiler **detects** them and hands
> the agent an obligation it cannot walk past. Confounding lands at 10% because nothing detects it.
>
> The difference between those two numbers is not prompt quality. It is **twenty lines of pandas that
> I have not written yet.**

**The lesson, which is the whole argument for the held-out domain:**

> Overfitting a prompt does not require naming your data. It is enough to speak your data's
> **dialect**. You cannot see that by rereading your own prompt — it reads as general to *you*,
> because you are fluent in the same dialect. Only a domain you did not write can show it to you.
>
> And the aggregate will hide it, because your easy tasks carry the average. **Read the per-task
> table, not the headline.**

### D30 — The offline-replay promise was true on exactly one computer
**The fourth bug I shipped. It is the same bug as [D25], and I want that on the record.**

The README says, in bold:

> *"**No API key?** Every notebook replays from the committed response cache… the whole series runs
> offline, deterministically, for free."*

That was **false for everyone but me.**

The agent's prompt carried the file paths it had been handed, and the notebooks (whose working
directory is `notebooks/`) handed it **absolute** ones. So the first user message contained
`/home/musel/Github/research_agent/data/trial.csv`. The cache key is a hash of the request. The
request contained my home directory. **Every cached response was keyed to a path that exists on one
laptop on Earth.**

Clone the repo to `~/Downloads/`, set `LIVE=False`, run the notebook the README tells you to run:
every call misses, and the whole thing raises. The promise fails on the first command a stranger
types.

And it is worse than a cache miss, because the *cached response itself* is poisoned: the model's
generated code says `pd.read_csv("/home/musel/...")`. Even a key that hashed correctly would replay
code that cannot run on anyone else's machine.

**Fixed by:** pinning the executor's working directory to the repo root, and normalising every path
into the prompt to be repo-relative (`data/trial.csv`). The prompt is now identical from any CWD, on
any clone; the model's code is portable; the cache is portable. Verified by hashing the same agent
call from two different working directories and asserting the keys match — and then re-running all
2,240 runs, because the prompt changed.

#### Why this one stings

[D25] is *the same lesson*, and I wrote it down myself:

> *"The README promised 'runs offline with no key — set `LIVE = False`.' I went to verify that
> sentence before shipping it, and it was false. **A footgun that fails silently is worse than one
> that crashes.** Both of these bugs were invisible from reading the code and obvious the moment I
> ran the exact thing the README told a stranger to run."*

I fixed the **flag** and I did not re-read my own conclusion. I ran the exact thing the README told a
stranger to run — *on my own machine*, where it worked, which is the one place the test was
guaranteed to pass.

> **"I tested it" is not a claim about the code. It is a claim about the environment you tested it
> in.** The bug lived in the difference between my machine and everyone else's, which is precisely
> the region my test could not see.
>
> A reproducibility claim that has only ever been checked by the person who wrote it has not been
> checked.

Four self-inflicted bugs now ([D24], [D25], [D28], [D30]), and **not one of them was findable by
reading the code.** Every single one surfaced by running something and looking hard at a number that
was wrong: a `$0.0000` cost, a flag that would not move, a task scoring 100%, a cache that only I
could hit. That is the entire argument of this project, learned the expensive way, four times.

### D31 — I ran the same benchmark twice and got opposite verdicts. The CI was lying.
**The fifth bug, and it is not in the agent. It is in the thing I built to check the agent.**

Late on, I made a change I believed was cosmetic: the agent's file paths went from absolute to
repo-relative (D30). It changed no logic. It provably changed no behaviour — **identical error rate
(0.27 → 0.23 per run), identical step count (11.4 → 11.6), identical budget-exhaustion rate.** The
prompt differs by one string.

I re-ran all 2,240. Here is my centrepiece, measured before and after that non-change:

| the same experiment, run twice | Δ for removing the Findings Ledger | 95% CI | verdict |
|---|---|---|---|
| run A | **−1.8%** | `[−8.9%, +4.6%]` | *"no detectable effect"* |
| run B | **−11.1%** | `[−18.6%, −4.3%]` | ***"SIGNIFICANT — it helps"*** |

**Same code. Same tasks. Same ten runs per cell. Opposite conclusions, and the two 95% intervals
barely overlap.** Individual tasks swung by 30–50 points between the two runs
(`s4_simpson_sales`: 10% → 60%).

At least one of those intervals is wrong, and I have no way to know which — which means **both are
worthless as stated.**

#### Why the bootstrap lied

It is not a coding bug. The hierarchical bootstrap is implemented correctly (D27). It is a bug in
what I *asked* it.

With **10 runs per cell**, the bootstrap resamples *with replacement from the ten outcomes I
happened to observe.* If a cell came back **1/10**, every resample of it is drawn from those ten
values — so the bootstrap's own distribution for that cell is centred near 10% and **cannot reach
60%**, which is precisely what the next run produced. The empirical distribution is degenerate at
the extremes, and the extremes are exactly where the interesting tasks live.

> **A bootstrap tells you the sampling variance of the data you have.** It cannot tell you about the
> data you didn't collect. At n=10 per cell, near p=0 or p=1, that is a distinction with a very
> large difference — and the interval it hands you is **narrower than the truth, in a way that feels
> rigorous.**

#### What makes this the worst bug in the project

I built this harness for exactly one reason. It is written at the top of `stats.py`:

> *"Confidence intervals, because a point estimate is how you fool yourself."*

And then I fooled myself **from inside the confidence interval** — twice, in opposite directions,
and wrote both up as findings. In [D27] I congratulated myself for catching a point estimate
masquerading as a result. This is the same mistake **one level up**: a *confidence interval*
masquerading as a result.

> A number with error bars is not automatically an honest number. **It is an honest number only if
> the error bars are honest**, and mine were computed from too little data to know.
>
> The instrument you built to stop yourself believing noise is itself an instrument, and **it also
> needs to be checked** — by the only method that has ever worked here: run it again and see if it
> says the same thing.

**Fixed by:** doubling to **20 runs per cell** (4,480 runs, $16) and reporting the pooled estimate;
keeping both original runs on disk as the replication evidence; and adding the replication itself to
the limitations rather than burying it. The honest headline is not a tighter number — **it is that
one run of this benchmark was never enough to support the sentences I was writing from it.**

**What I would actually do with a real budget:** stop bootstrapping a proportion from ten Bernoulli
draws and fit a hierarchical beta-binomial, which shrinks the extreme cells instead of pretending
they are certain. Or, far cheaper and more honest: **run the whole benchmark twice, every time, and
report the spread between runs as the error bar.** That number cannot lie to you, because it is not
a model — it is a measurement.

### D32 — My most trusted gate was rejecting numbers the agent had just printed
**The sixth bug, and the one I would most like to have found before someone else did.**

D10 says the grounding gate is the mechanism I trust most, and gives the reason:

> *"A regex catches it every time, for free, in fifteen lines… No LLM is involved. **It cannot be
> talked out of it.**"*

It also carries a warning I wrote myself, in the docstring, and then did not honour:

> *"a gate that fires wrongly is a gate its user disables."*

**It fired wrongly.** On **26.6% of 4,480 runs.**

#### The bug

The gate rounded both sides to four *significant figures* and demanded exact equality:

```
the code printed    -0.0869479104773222   ->  4 sig figs  ->  -0.08695
the agent reported  -0.0869               ->  4 sig figs  ->  -0.0869
                                                                -0.08695 != -0.0869
REJECTED: "the values {-0.0869} in your evidence never appeared in any executed output."
```

The agent **had printed the number.** It rounded it to four decimals to write it into the report —
which is what any analyst does — and my gate called that a fabrication.

And then it could not escape. It re-ran the same code to print the number again; the duplicate-code
guard handed back the cached observation; it tried once more; the gate rejected again. Watch the
trajectory eat itself:

```
[10] ⛔ gate 3 (grounding) rejected — ungrounded: [0.636, 0.723, -0.087]
[11] run_python  ✓ treatment_response_rate: 0.6363636363636364 ...
[12] ⛔ gate 3 (grounding) rejected — ungrounded: [-0.0869]
[13] ⟳ identical code re-submitted — returning cached result
[14] run_python  ✓ -0.0869479104773222
[16] ⛔ gate 3 (grounding) rejected — ungrounded: [-0.0869]
[17-20] ⟳ ⟳ ⟳ ⟳   step budget exhausted — forcing best-effort answer
```

**The gate was implicated in 54% of every budget blowout in the study** (148 of 273). And the
forced best-effort answer, submitted with the last step, was the confounded one — so a gate built
to prevent a wrong number ended up *causing* one.

#### Fixed by

Comparing with a **relative tolerance (0.5%)** instead of demanding that two rounding schemes agree.
A rounding convention is not a fabrication. The failure this gate exists to catch is DDB's *"its own
code printed 1, and the final tally used 2"* — an error of **100%**, not of **0.006%**.

Verified in both directions: it still rejects `1 → 2`, `7 → 8`, and invented numbers; it now accepts
a three-significant-figure report of a value that was actually printed.

#### The lesson, which is not the one I expected

I trusted this gate *because* it was deterministic. I wrote "no LLM is involved, it cannot be talked
out of it" as though that settled the question of whether it was **right**.

> Determinism buys you **consistency**, not **correctness.** A regex that is confidently wrong is
> worse than an LLM that is unsurely right, because nothing in the system is empowered to argue with
> it — and I had removed, on purpose, the one component that might have.
>
> **A deterministic gate is only as good as its notion of "the same number."** Mine had one, it was
> wrong, and it never once said so.

And note where it was found: **not** by reading `grounded()`, which I had read many times and
admired. It was found by watching a single trajectory, end to end, in a notebook — the agent
printing a number and being told the number did not exist. Six bugs now, and the score is still
**running things: 6, reading things: 0.**

### D33 — I stopped writing *"what I'd build next is the detector"* and built the detector
**The only decision in this log that is a prediction, tested.**

Every draft of this project ended on the same sentence, and I was pleased with it:

> *"What I'd build next isn't another gate. It's the detector."*

The evidence was strong and it was mine. Anything a twenty-line script **detects** — sentinels,
duplicates, numeric-as-text, scope — the agent handles at **95–100%**, because the finding is seeded
as an obligation it cannot submit past. The one thing nothing detected — **confounding** — was a
**coin flip**, on both domains, and:

- rewording the system prompt to be domain-neutral **did not move it** (D29);
- the Findings Ledger could not help, because the ledger forces you to act on what you *noticed*;
- and I watched the agent, at temperature 0, **notice the imbalance, log it, and then dismiss it**:
  *"the question asks for the difference in proportions, so no adjustment is needed."*

Read that dismissal again. It is GeneBench-Pro's notice–act gap, verbatim, in my own trace: the
agent let the **question's phrasing** overrule the **data's warning**.

So the sentence was a hypothesis, and it was cheap to test. I stopped writing it down and wrote the
detector instead.

#### The detector, in full

For every pair of low-cardinality **categorical** columns, cross-tabulate; if the conditional
distribution of one departs from its marginal by ≥15 percentage points, the groups are not
comparable. Seed it as an **open finding**. That is the whole idea, and it is about twenty lines
(`observe._confounds`).

It names no column, no dataset and no domain. It finds `arm × severity` in a clinical trial and
`channel × customer_segment` in an e-commerce export by **exactly the same arithmetic** — which is
the only way I get to claim it is not fitted to my own benchmark.

#### The one design choice worth defending

**It does not try to work out which column is the outcome.** It can't: an outcome is associated with
its own cause by definition, and which variable is "the outcome" is a fact about the *question*, not
about the data.

My first version therefore flagged five "confounds" in `trial.csv`, of which one was real, and the
agent burned its entire step budget discharging noise. The fix was one clause — consider only
**non-numeric** low-cardinality columns — and it does a surprising amount of work: the 0/1 outcome
columns drop out, and so does an integer re-test counter. `trial.csv` now yields **exactly one**
finding, `sales.csv` yields **exactly one**, and both are the planted traps.

**Cost of that clause, stated plainly:** if your groups are integer-encoded (`arm` as 0/1), this
misses them. That is a real limitation and a one-line fix for a schema that needs it. I would rather
write it down than let a clean result imply a generality I have not earned.

#### What happened

The demo that had been failing — the flagship Simpson's-paradox task, at temperature 0, the *first
thing in the presentation* — went from submitting **−0.087** (the confounded answer, after burning
all 20 steps) to submitting **+0.150** (the truth, in 13 steps, cleanly). And the trace now says
`ACTED` on the confound rather than `DISMISSED`.

> **The gate was never the problem. The gate had nothing to gate.**
>
> Twenty lines of `pd.crosstab` did what no amount of prompt engineering, no ledger, no verifier and
> no bigger model was going to do — because none of them could **manufacture an observation that was
> never made.**

This is the whole thesis of the project, and it is the last thing I did to it:

> **Spend your budget on what the agent *notices*, not on what you force it to do about it.**
> A structural gate is cheap and it is worth building. But it is worth *nothing* until something
> deterministic puts an observation in front of it.

---

## Platform

### D18 — `Qwen3-30B-A3B` as the agent, on purpose, and *not* the best model available
**Considered:** `zai-org/GLM-5.2` — the strongest open model on a published agentic-science
benchmark. DrugDiscoveryBench evaluated the open models and Token Factory hosts almost exactly
their leaderboard: GLM 5.2 (37.8%), Kimi K2.7 Code (35.3%), DeepSeek V4 Pro (31.7%), MiniMax M3
(23.2%) — DDB Figure 7, p.22. On evidence, GLM-5.2 is the pick.

**Chose:** `Qwen/Qwen3-30B-A3B-Instruct-2507` anyway — **14× cheaper**, and several rungs down
the ladder.

**Why deliberately not the best one:** the thesis of this design is that reliability comes from
the *scaffolding*, not from a bigger base model. Running it on the strongest available model
would make that thesis **unfalsifiable** — every success could be credited to the model, and I
would have learned nothing. Handicapping the base model is what turns the claim into an
experiment. The whole 2,240-run ablation study costs $8 precisely *because* of this choice.

**Verifier/judge:** `openai/gpt-oss-120b` — a different family from the agent, on purpose (D11).

**What this does NOT show, and I will not imply that it does:** that a small model *with*
scaffolding beats a big model *without* it. That is the direct test of the thesis and **I have
not run it.** It is one command —
`uv run python -m evals.run_eval --config no_guardrails --model zai-org/GLM-5.2` — and roughly
$15 at GLM's prices. It is the first thing I would spend the next budget on, and it is the one
result that would let me state the thesis as a fact instead of a design bet.

**Revisit when:** the harness exists — at which point "which model" stops being a judgement call
and becomes a table. That is the point of building the harness first.

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
~200 lines; that is itself a claim, and it's checkable.
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
DDB's own future work suggests exactly this shape: route structural reasoning, database retrieval and
cheminformatics to different model-tool stacks.

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
documents (a qualifier silently stops being applied three steps in). Structured state does not decay:
it is re-rendered every turn from a variable, not remembered from a paragraph. The `population` field
gets its own line because GBP's worked DRX1 example is trapped *exactly* on the denominator — compute
over the tested subset and you get a plausible, wrong number.
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
the ambiguity?) — binary rubric, temperature 0, different model family, and validated against my own
hand labels first, because an unvalidated judge is a random number generator with good manners.
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

### D16 — Log every step as JSONL; run each task 5×
**Why:** Agent evals are noisy — GBP runs 10 attempts and bootstraps CIs; DDB runs 3 trials. A single
run is an anecdote. 5× is the honest floor at this scale, and pass@5 vs 5-of-5 tells you *which* tasks
are unstable (they turn out to be exactly the ambiguous and dirty ones, which is reassuring).
The JSONL trace is what turns a failure from "it got it wrong" into "at step 4 it noticed the
sentinels and at step 6 it computed the mean anyway" — which is the only kind of failure report you
can act on.

### D17 — Classify failures using DrugDiscoveryBench's taxonomy
**Chose:** hand-label every failing run as domain-reasoning / derivation / retrieval / constraint /
final-answer-slip.
**Why:** Inventing my own taxonomy would make my failure profile incomparable to the literature.
Reusing theirs means I can say "my agent's failures are 40% domain reasoning versus their 54%" — a
sentence that means something. Standing on a published taxonomy is cheaper *and* more credible than
inventing one.

---

## Platform

### D18 — `zai-org/GLM-5.2` as the agent; `Qwen/Qwen3.5-397B-A17B` as verifier/judge
**Why:** DDB benchmarked the open models and Token Factory hosts almost exactly their leaderboard —
GLM 5.2 (37.8%), Kimi K2.7 Code (35.3%), DeepSeek V4 Pro (31.7%), MiniMax M3 (23.2%). GLM-5.2 is the
strongest of them on a published agentic-science benchmark, so the choice rests on evidence rather
than on taste. The verifier is a *different family* on purpose (D11).
I smoke-tested all six candidates on the exact two-round tool-calling loop this agent needs before
choosing; every one handled the protocol correctly, so the decision came down to the benchmark and
cost, not to plumbing (`notebooks/00_setup.ipynb`).
**Revisit when:** the eval harness exists — at which point "which model" stops being a judgement call
and becomes a table. That is the point of building the harness first.

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05 — Keeping the thread
#
# ### *"Noticing must have consequences."*
#
# **Previously:** the agent wrote flawless pandas and told us a working drug doesn't work. Fixing
# its tools made it *faster* at being wrong. The papers say this is the norm, not the exception:
# **54% of real agent failures are of exactly this kind.**
#
# So we stop trying to make the model smarter. We make it **structurally unable to drop the thread.**
#
# Three things get dropped. This chapter fixes the first two.
#
# | | Dropped thing | Fix |
# |---|---|---|
# | **Ledger 1** | *the question* | Question Contract |
# | **Ledger 2** | *the finding* | Findings Ledger ← **the centrepiece** |
# | Ledger 3 | *the number* | (notebook 06) |

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

from agentlib import Config, run_agent
from agentlib.ledger import FindingsLedger, QuestionContract
from agentlib.llm import METER
from agentlib.observe import briefing

FILES = ["../data/trial.csv", "../data/data_dictionary.md"]

# %% [markdown]
# ---
# ## 0. First, the cheapest fix of all: stop letting it start blind
#
# One of DrugDiscoveryBench's five failure categories is **Retrieval — 16.4%** — and their
# definition includes *"failing to read a provided file."*
#
# In notebook 04, the agent read `trial.csv` and **never opened `data_dictionary.md`** — the file
# that explains the sentinels, the duplicate rows, and the unit error. It was right there. It just
# didn't look.
#
# The cheapest possible fix is to **make looking non-optional.** Before the model is called even
# once, plain Python profiles every file and puts the result in the first message.

# %%
print(briefing(FILES)[:1800])

# %% [markdown]
# > ### 💡 Two things worth noticing about that briefing
# >
# > 1. **The documentation is in there.** Not "available if you ask" — *in the context*, before
# >    turn one. The unit warning cannot be un-read.
# > 2. **Look at the `⚠ automatic checks flagged` section.** Deterministic Python — no model
# >    involved — found the `-999` sentinels and the duplicate `patient_id`s. Twenty lines of
# >    `if` statements.
# >
# > This matters for a reason we'll come back to: a Findings Ledger can force the agent to **act**
# > on what it noticed. It cannot make it **notice**. So we hand it the mechanical findings for
# > free, and save the model's attention for the things a regex can't see.

# %% [markdown]
# ---
# ## 1. Ledger 1 — the Question Contract
#
# **The failure:** DrugDiscoveryBench, on a task that asked to rank melanoma genes —
#
# > *"The models fail because, at some point, they stop applying that melanoma scope. The last
# > chance to catch the slip is at the final answer: a human who had misread the task the same way
# > would look at the result, recognize it as a meaningless response to the user's actual goal,
# > and backtrack. **None of the failing models caught this.**"*
#
# The question **decayed**. It was in the prompt at step 1, and by step 6 it wasn't in the model's
# working attention any more.
#
# **The obvious fix — "restate the question" in the system prompt — does not work**, because
# advice in a prompt decays *exactly like the question does*. Both are just text getting further
# away.
#
# **Structured state does not decay.** It gets re-rendered from a variable, every single turn.

# %%
contract = QuestionContract(
    estimand="The difference in response proportion between treatment and control arms.",
    population="All ENROLLED PATIENTS (deduplicated on patient_id — re-tested patients appear twice).",
    units="A difference in proportions, between -1 and 1.",
    constraints=["Compare treatment vs control", "Report as a difference, not a ratio"],
    premises=["that a raw treatment-vs-control comparison is meaningful — CHECK: the trial was "
              "NOT randomised, so it is not"],
    question_is_precise=True,
    ambiguities=[],
)
print(contract.render())

# %% [markdown]
# That block gets **pinned to the end of the context on every single turn.** Not remembered —
# *regenerated*.
#
# The `population` field has its own line because it is the field that gets silently dropped.
# GeneBench-Pro's worked example is trapped precisely on the denominator: compute over the tested
# subset instead of the full roster and you get a plausible, wrong number. Their design table names
# the failure in one sentence:
#
# > *"A statistically valid final model is applied to the wrong data or population, on the wrong
# > scale, or on the wrong conceptual level."*
#
# ### Two of these fields I did not design. The evaluation forced them on me.
#
# **`premises`** — *a question can be wrong.* "Which of the **four** sites had the highest response
# rate?" There are three. "**Why** do women respond better?" They don't. Without this field, the
# agent answered both questions fluently and **laundered a false premise into a fact.** With it:
# 0/3 → 3/3 on the sites task.
#
# **`question_is_precise`** is a *required boolean*, and that's the whole trick. The contract already
# had an `ambiguities` list — and on *"did the biomarker improve?"* the agent left it **empty every
# single time** and silently picked a reading. The field designed to prevent the failure was the
# field being skipped.
#
# > **A field the model *may* leave empty is a field the model *will* leave empty.**
#
# A required boolean can't be skipped. And a validator makes `ambiguities` non-empty whenever it's
# `False`. Same move as the Findings Ledger, which we're about to meet: **make the omission
# impossible to express, instead of asking nicely for it not to happen.**

# %% [markdown]
# ---
# # 2. Ledger 2 — the Findings Ledger
# ## *This is the centrepiece of the whole design.*
#
# Go back and read GeneBench-Pro's actual diagnosis, slowly:
#
# > *"the agent **notices** the relevant local diagnostic clue but treats it as a **local data
# > cleaning issue** rather than as evidence that should **change the downstream statistical
# > method** and QC pipeline."*
#
# ### The model is not failing to notice.
#
# It runs `describe()`. It **sees** the `-999`s. It might even mention them. Then it drops them
# from the column and carries on with the analysis it had already decided on.
#
# The observation never reaches the decision. **Noticing has no consequences.**
#
# ### So: give noticing consequences.
#
# ```python
# note_finding(
#     observation = "biomarker_baseline has 88 values of exactly -999",
#     implication = "these are QC-failure codes, not measurements — they drag the mean to -64",
#     status      = "open",
# )
# ```
#
# And then the one rule that makes the whole thing work:
#
# > # 🔒 `submit_answer` is BLOCKED while any finding is `open`.
#
# To close a finding, the agent must do one of exactly two things:
#
# - **act** on it — and name the code step that handled it → `status="acted"`
# - **dismiss** it — and write why it doesn't affect the estimand → `status="dismissed"`
#
# Both are recorded. Both ship in the final report.
#
# The `implication` field is where the work actually happens. It forces the agent to write down
# **what this changes** — which is precisely the step the papers watch it skip.
#
# It is a state machine, not a personality trait. Let's watch it run.

# %%
ledger = FindingsLedger()
print(ledger.note("biomarker_baseline contains 88 values of exactly -999",
                  "QC-failure sentinel, not a measurement. Must exclude before any mean."))
print()
print(ledger.note("48 patient_ids appear twice (re-tested samples)",
                  "Row-level stats double-count these patients. Must dedupe on max(sample_seq)."))
print()
print(ledger.render())

# %% [markdown]
# The agent tries to submit now. It gets bounced:

# %%
from agentlib.report import rejection_message

open_findings = [(i, f) for i, f in enumerate(ledger.findings, 1) if f.status == "open"]
print(rejection_message("open_findings", findings=open_findings))

# %% [markdown]
# It resolves them — and only then is the exit unlocked.

# %%
print(ledger.resolve(1, "acted", "Filtered biomarker_baseline != -999 before computing the mean."))
print(ledger.resolve(2, "acted", "Deduplicated: sort by sample_seq, groupby patient_id, keep last."))
print()
print(ledger.render())

# %% [markdown]
# > ### 💡 Why this is not just "prompting it to be careful"
# >
# > A prompt is a *request*. It competes with everything else in the context, and it loses ground
# > with every step.
# >
# > This is a **gate**. It doesn't ask the agent to be careful; it makes carelessness *impossible
# > to express*. There is no path to `submit_answer` that runs through an unresolved finding.
# >
# > And notice what it does **not** do: it never tells the agent *what* to conclude. The agent can
# > look at a finding and say "this doesn't matter, here's why" — and that's fine, that's a
# > `dismissed`. We're not forcing a conclusion. We're forcing the observation to **reach** the
# > decision. What it does when it gets there is still the model's call.
# >
# > That distinction is the difference between a guardrail and a straitjacket.

# %% [markdown]
# ---
# ## 2b. ⚠️ I built exactly that. And it still got the wrong answer.
#
# Here is what the first real run of the Findings Ledger produced:
#
# ```
# [1] contract  : difference in response proportion, treatment minus control
# [2] run_python: (looks at the data)
# [3] run_python: Total rows: 848  Unique patients: 800  Duplicates: 48
# [4] 🔴 finding: 48 patients tested twice → must deduplicate
# [5] run_python: (deduplicates)
# [6] ✅ resolved #1: acted
# [7] run_python: Treatment response rate: 0.6364   Control: 0.7233
# [8] ✅ ACCEPTED — "the treatment arm had an 8.69 point LOWER response rate"
# ```
#
# ### It worked perfectly. And the answer was **−0.0869**. Still the naive answer.
#
# Look at what happened. The ledger did **exactly** its job: the agent noticed the duplicates, was
# forced to act on them, and did. Flawless.
#
# **It just never noticed the confounding.** So there was no obligation to discharge, and the gate
# had nothing to block.
#
# > ### 💡 This is the limitation I wrote down before I saw it happen — and then it happened.
# >
# > The Findings Ledger converts **noticed-but-ignored** into a hard stop.
# >
# > It does **nothing** about **never-noticed.**
#
# And the truly annoying part? `data_dictionary.md` says it *in plain English*:
#
# > *"Treatment was **not randomised**... Any comparison of `responded` between arms that does not
# > account for `severity` is comparing two populations that were never comparable — and will
# > reach the wrong sign."*
#
# That text was **in the briefing, in the context, on turn one.** The agent read it and did not
# act on it. Which is, of course, the notice–act gap wearing a different hat.

# %% [markdown]
# ## 2c. Two fixes. Neither of them is "use a better model."
#
# ### Fix 1 — Stop *hoping* it notices. Pre-seed the ledger.
#
# Anything a twenty-line script can find, a twenty-line script **will** find — and it goes in as
# an **OPEN OBLIGATION**, not as a helpful note the agent is free to scroll past.
#
# > Information can be ignored. **An obligation cannot.**
#
# This is the division of labour the whole design rests on:
#
# | | does what |
# |---|---|
# | **deterministic code** | finds the mechanical problems, and makes them *un-ignorable* |
# | **the model** | decides what they *mean* for this particular question |

# %%
from agentlib.observe import seed_findings

for obs, impl in seed_findings(["../data/trial.csv"]):
    print("🔴 PRE-REGISTERED:", obs)

# %% [markdown]
# ### Fix 2 — Give it the reflex a statistician has.
#
# One rule in the system prompt:
#
# > **"Before comparing two groups, check that they are comparable.** If the groups were not
# > randomly assigned, a raw comparison between them is not a treatment effect — it is a
# > comparison of two different populations. Cross-tabulate the group against the other variables.
# > If they are imbalanced, that is a finding, and you must adjust for it rather than reporting
# > the raw difference. **A confounded comparison can give you the opposite sign to the truth.**"
#
# Is that cheating — teaching to my own test? No, and here's the check: **it never mentions
# severity, or this dataset, or Simpson's paradox.** It states a general principle of causal
# inference that applies to any two-group comparison anywhere. (And the held-out tasks in
# `evals/tasks.py` exist precisely to catch me if I start overfitting to my own benchmark.)
#
# It is also the fix the papers themselves point to. DrugDiscoveryBench re-ran their unsolved
# tasks with the expert's step-by-step playbook supplied as a hint, and went from 76/82 to 80/82:
#
# > *"The results suggest that **execution is within reach for today's agents should they be given
# > the expert workflow**."*
#
# The model can execute. What it lacks is the analyst's *reflex* — the thing a statistician does
# without being asked. **So encode the reflex.**
#
# That is what "building on top of the base model" actually means. Not a better model. A better
# **procedure**.

# %% [markdown]
# ---
# # 3. Now run it for real.
#
# Same question that produced **−0.088** ("the treatment doesn't work") in notebook 04.
#
# Same model. Same data. Same tools. **No verifier yet** — ledgers only.

# %%
run = run_agent(
    "Does the treatment improve the response rate? Report the treatment effect as a "
    "difference in proportions (treatment minus control).",
    FILES,
    Config(use_grounding=False, use_verifier=False),   # ledgers only — gates come in nb 06
)

# %% [markdown]
# ## The audit trail
#
# This is the artifact I actually care about. Not the number — **the record of what it noticed and
# what it did about it.**

# %%
print("ANSWER:", run.report["answer"])
print("VALUE :", run.report["value"])
print()
print("═" * 78)
print("THE FINDINGS LEDGER")
print("═" * 78)
for i, f in enumerate(run.report["findings"], 1):
    icon = {"acted": "✅", "dismissed": "⚪", "open": "🔴"}[f["status"]]
    print(f"\n{icon} #{i}  {f['observation']}")
    print(f"     ↳ implication: {f['implication']}")
    print(f"     ↳ resolution : {f['resolution']}")

# %%
print("─" * 78)
print(f"  notebook 04 (no ledgers) : -0.088   →  'the treatment does not work'")
print(f"  notebook 05 (ledgers)    : {run.report['value']}   →  '{run.report['answer'][:44]}...'")
print(f"  ground truth             : +0.150")
print("─" * 78)

# %% [markdown]
# ## Read the ledger again.
#
# The agent **noticed the confounding** — that arm assignment wasn't randomised — wrote down what
# it *implied*, and then **could not proceed** until it had done something about it.
#
# So it stratified. And the sign flipped.
#
# That is a notice–act gap, closed mechanically. Not by a better model. Not by a bigger context.
# By about forty lines of Python that refuse to let an observation die quietly.
#
# ### And look at what it *dismissed*.
#
# One of the pre-registered findings was the `-999` sentinels in `biomarker_baseline`. The agent
# looked at it and **dismissed** it — because this question is about `responded`, and the baseline
# biomarker doesn't enter the calculation at all.
#
# **That is the correct call.** And it's the thing I most want you to see, because it shows the
# gate is not a straitjacket. The agent is never told *what* to conclude — only that it must
# conclude *something*, out loud, before it's allowed to leave. It can look at a finding and say
# "this doesn't matter, and here's why."
#
# We are not forcing a conclusion. We are forcing the observation to **reach** the decision. What
# happens when it gets there is still the model's judgement.

# %% [markdown]
# ### 🐛 One more thing, from the trace above
#
# Somewhere in that run you'll see:
#
# ```
# [8] ⟳ identical code re-submitted — returning cached result
# ```
#
# That's the death-loop guard from notebook 04 firing **in a real run**. The agent hit a
# `SyntaxError`, tried the same broken code again, and instead of burning a step re-running it, we
# handed back the cached result and told it plainly: *"you already ran this; it will not change."*
#
# Ten lines. Saved a step. This is what most of "agent engineering" actually looks like.

# %%
print(METER)

# %% [markdown]
# ---
# # Where we are
#
# | | |
# |---|---|
# | **Briefing** | The agent can no longer start blind. Deterministic profile + the docs, before turn 1. |
# | **Ledger 1: Contract** | The question is re-rendered every turn, so it can't decay. |
# | **Ledger 2: Findings** | A noticed problem is an **open obligation**. The exit is locked until it's discharged. |
# | **Result** | The sign flipped. Same model, same data — different *structure*. |
#
# ### 🔜 But I still don't trust it.
#
# Look closely at that answer. **Where did that number come from?**
#
# It says `0.15`-ish. Did any code it ran actually *print* that? Or did it assemble it in its head
# at the last second — the way it invented `0.625` back in notebook 04?
#
# I have no idea. And "I have no idea" is not good enough for a number someone is going to put in
# a drug filing.
#
# **The third dropped thing is the number itself.**
#
# **→ `06_the_gated_exit.ipynb`**

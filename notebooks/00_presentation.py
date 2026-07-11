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
# # A data-analysis agent, built from scratch
#
# ### *Part 2, Option A — the walkthrough*
#
# > **The claim:** a data-analysis agent is a **while loop that writes code, runs it, and reads
# > the results.** Everything else is a guardrail *earned from a failure you watched happen*.
#
# **~885 lines. No framework. Runs on a $0.10/1M-token open model at about half a cent per
# analysis.**
#
# ---
#
# | | |
# |---|---|
# | **1. The destination** | run the agent, live |
# | **2. The problem** | it says a working drug doesn't work |
# | **3. The papers** | 54% of real agent failures are *not* coding failures |
# | **4. The design** | three ledgers and a gated exit |
# | **5. The evidence** | 360 runs, 8 ablations |
# | **6. The punchline** | **the result that inverted my own thesis** |
# | **7. The dashboard** | *(embedded below)* |
#
# *The seven teaching notebooks (`01`–`07`) build every piece of this from nothing. This one is
# the tour.*

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
from IPython.display import HTML, display

from agentlib import Config, run_agent
from agentlib import config as cfg

ROOT = os.path.abspath("..")
FILES = [f"{ROOT}/data/trial.csv", f"{ROOT}/data/data_dictionary.md"]

# Rehearse with LIVE_DEMO = False (instant, replays from the committed cache, no network).
# Flip to True in front of the audience — same code, real API calls.
LIVE_DEMO = True

print("agent    :", cfg.AGENT_MODEL)
print("verifier :", cfg.VERIFIER_MODEL, "  (different family — a model grading itself is biased)")
print("host     :", cfg.BASE_URL)

# %% [markdown]
# ---
# # 1. The destination
#
# A phase-II clinical trial. 800 patients. The question the trial was actually run to answer:
#
# ## *"Does the treatment improve the response rate?"*

# %%
run = run_agent(
    "Does the treatment improve the response rate? Report the treatment effect as a "
    "difference in proportions (treatment minus control).",
    FILES,
    Config(),        # everything on
)

# %% [markdown]
# ## The deliverable is not the number. It's the audit trail.

# %%
r = run.report
rows = "".join(
    f"""<div style="border-left:4px solid {'#1baf7a' if f['status']=='acted' else '#adb5bd'};
                   padding:.5rem .9rem;margin:.4rem 0;background:rgba(127,127,127,.07);border-radius:4px">
      <b>{'✅ ACTED' if f['status']=='acted' else '⚪ DISMISSED'}</b> — {f['observation']}
      <div style="font-size:.86em;color:#6c757d;margin-top:.25rem">
        <b>implication:</b> {f['implication']}<br><b>resolution:</b> {f['resolution']}</div></div>"""
    for f in r["findings"])

display(HTML(f"""
<div style="font-size:1.25rem;font-weight:600;margin-bottom:.5rem">{r['answer']}</div>
<div style="color:#6c757d;margin-bottom:1rem">
  value <b>{r['value']}</b> &nbsp;·&nbsp; confidence <b>{r['confidence']}</b> &nbsp;·&nbsp;
  {run.steps} steps &nbsp;·&nbsp; <b>${run.cost_usd:.4f}</b>
</div>
<div style="font-weight:600;margin:.6rem 0">🔍 What it noticed, and what it did about it</div>
{rows}"""))

# %% [markdown]
# > **That trail is the thing a scientist actually needs.** Not *"the model said 0.15"*, but
# > *"the model saw the confounding, said what it implied, adjusted for it, and here is the step
# > where it did."*
# >
# > One is an oracle. The other is a colleague.

# %% [markdown]
# ---
# # 2. Now here is why that was hard
#
# The same question, answered by a **competent** agent with no guardrails — one that writes
# flawless pandas and prints real numbers.

# %%
import numpy as np

raw = pd.read_csv(f"{ROOT}/data/trial.csv")
dedup = raw.sort_values("sample_seq").groupby("patient_id", as_index=False).last()

marginal = dedup.groupby("arm")["responded"].mean()
naive = marginal["treatment"] - marginal["control"]

strat = dedup.groupby(["severity", "arm"])["responded"].mean().unstack()
per_stratum = strat["treatment"] - strat["control"]
adjusted = float((per_stratum * dedup["severity"].value_counts(normalize=True)).sum())

print("RESPONSE RATE BY ARM  — what the naive agent computes:")
print(marginal.round(3).to_string())
print(f"\n   → treatment effect: {naive:+.3f}   ❌ 'THE TREATMENT DOES NOT WORK'\n")
print("RESPONSE RATE BY SEVERITY × ARM  — what it needed to look at:")
print(strat.round(3).to_string())
print(f"\n   → the treatment wins in EVERY stratum:")
print(per_stratum.round(3).to_string())
print(f"\n   → severity-adjusted effect: {adjusted:+.3f}   ✅ THE TREATMENT WORKS")
print("\n" + "═" * 66)
print(f"   NAIVE: {naive:+.3f}      TRUTH: {adjusted:+.3f}      OPPOSITE SIGNS.")
print("═" * 66)

# %% [markdown]
# ## Simpson's paradox.
#
# The trial **was not randomised** — clinicians gave the drug to the patients who were already
# sicker. So the treatment arm is loaded with hard cases, and the pooled comparison isn't
# comparing *treatment vs control*. It's comparing **sick people vs healthy people.**
#
# The agent's arithmetic was perfect. Its pandas was flawless. And it told me a drug that works
# doesn't work.
#
# *(There are three more landmines in this file it walked straight past: `-999` QC-failure
# sentinels, 48 patients appearing twice, and an assay batch reporting in the wrong unit — 10×
# too large. All three are documented in `data_dictionary.md`, which it never opened.)*

# %% [markdown]
# ---
# # 3. This is exactly what the papers found
#
# ### GeneBench-Pro — Li & Ho (OpenAI, 2026)
# 129 multistage statistical-reasoning problems on messy biomedical data. Best frontier model:
# **31.5%**. On **45.7% of problems it scores zero across ten attempts.**
#
# > *"the agent **notices** the relevant local diagnostic clue but **treats it as a local data
# > cleaning issue** rather than as evidence that should **change the downstream statistical
# > method**."*
#
# ### DrugDiscoveryBench — Akyürek, Tu et al. (Scale AI, 2026)
# 82 expert drug-discovery tasks; best agent **51.6%**. They hand-classified **226 failing runs**:
#
# | Failure mode | Share |
# |---|---|
# | **Domain reasoning** — *"applies an incorrect scientific premise or misinterprets the data it has, **even though its inputs and tools are correct**"* | **54.0%** |
# | Derivation error — right approach, wrong calculation | 18.6% |
# | Retrieval — *"failing to read a provided file"* | 16.4% |
# | Constraint — violates an explicit instruction | 7.5% |
# | Final-answer slip | 3.5% |
#
# > *"the agents knew which database to query and how to compute the property the task asked for
# > at a high level. But somewhere along the execution **they drop a constraint, commit too early,
# > fail to backtrack**."*
#
# ## 🎯 Read that table again.
#
# **Over half of all failures are not coding failures.** The tools worked. The Python was correct.
# The model *had* everything it needed.
#
# It just didn't let what it saw change what it did.

# %% [markdown]
# ---
# # 4. So: what I built on top of the base model
#
# **The bottleneck is not code generation, not knowledge, not context length. It is
# thread-keeping.**
#
# Three things get dropped — **the question**, **the finding**, and **the number**. So each one
# becomes explicit state the agent must reconcile before it is allowed to finish.
#
# | | Mechanism | Kills |
# |---|---|---|
# | **Ledger 1** | **Question Contract** — estimand, *population*, units, constraints, premises. Re-rendered every turn, so it can't decay. | scope drift |
# | **Ledger 2** | **Findings Ledger** — a noticed problem is an **open obligation**. `submit_answer` is **hard-blocked** while any finding is open. | **the notice–act gap** |
# | **Ledger 3** | **Grounding** — every number must appear in the stdout of code that ran. A regex. No LLM. | derivation errors |
# | **Gate 4** | **Verifier** — a different model family that sees the code and the answer but **never the reasoning** | the final-answer slip |
#
# **Finishing is a gated action, not a default.** Four gates, cheapest first — never pay for an
# LLM call to catch what a regex would catch.
#
# ### Watch the ledger actually block a submission:

# %%
from agentlib.ledger import FindingsLedger
from agentlib.report import rejection_message

led = FindingsLedger()
led.note("biomarker_baseline contains 88 values of exactly -999",
         "QC-failure sentinel, not a measurement. Must exclude before any mean.")
led.note("arm was assigned by clinician judgement, not randomised",
         "The arms are not comparable. A raw difference will have the wrong sign.")

print(rejection_message("open_findings",
                        findings=[(i, f) for i, f in enumerate(led.findings, 1)]))

# %% [markdown]
# > It is **not** a prompt saying *"please be careful."* A prompt is a request; it competes with
# > everything else in the context and loses ground with every step.
# >
# > **This is a gate.** There is no path to `submit_answer` that runs through an unresolved
# > finding. It doesn't ask the agent to be careful — it makes carelessness *impossible to
# > express*.
# >
# > And notice what it does **not** do: it never says what to *conclude*. The agent can look at a
# > finding and say *"this doesn't matter, here's why"* — that's a valid `dismissed`. We force the
# > observation to **reach** the decision. What happens when it arrives is still its judgement.

# %% [markdown]
# ---
# # 5. How would I know it works?
#
# 15 tasks, two datasets, ground truth computed in pandas by the grader — it never goes near the
# agent. Binary, all-or-nothing grading, following GeneBench-Pro.
#
# ### The guard that makes the benchmark trustworthy
#
# Every trap task records its **plausible-but-wrong** answer and asserts it lies *far outside* the
# tolerance band. If a lazy analysis can land inside the band, the task grades nothing.

# %%
from evals.tasks import TASKS, test_leak, test_separation

rows = [{"task": t.id, "truth": round(t.gt(), 4), "the naive answer": round(t.naive(), 4),
         "separation": round(abs(t.gt() - t.naive()), 4),
         "tolerance band": round(abs(t.gt()) * t.tol, 4)}
        for t in TASKS if t.naive and isinstance(t.gt(), float)]
print(pd.DataFrame(rows).to_string(index=False))

test_separation(); test_leak()
print("\n✓ every naive path is far outside its tolerance band")
print("✓ no ground truth appears in any prompt")

# %% [markdown]
# > ### 🐛 This guard rejected **three of my own tasks.**
# >
# > | The task I wrote | What the assertion found |
# > |---|---|
# > | *median baseline biomarker* | **The median is robust to outliers** — 11% sentinels barely move it. Graded nothing. |
# > | *mean age of control-arm patients* | **Age is independent of arm** in my DGP. Dropping the filter changed nothing. Graded nothing. |
# > | *fraction with severe disease* | Re-tests were a random sample, so deduplication moved no proportion. Graded nothing. |
# >
# > All three looked perfectly reasonable. **The assertion found them. I didn't.**

# %% [markdown]
# ---
# ## 5b. Does it generalise? The question more runs cannot answer.
#
# Every trap in `trial.csv` is one **I** planted while designing the guardrails. Passing it only
# proves they work on the failures I **already knew about** — and more runs shrink the *variance*
# of that claim, not its **bias**.
#
# So `sales.csv` is a **held-out domain**: e-commerce, not medicine. Revenue exported as text
# (`"1,234.56"`), internal QA orders at `999999.99`, refunds still in the file, `-1` for a missing
# age, and a Simpson's paradox on **channel × customer_segment** instead of arm × severity.

# %%
from evals.stats import hierarchical_bootstrap
from evals.tasks import TASKS

df = pd.read_json(f"{ROOT}/evals/results.jsonl", lines=True)
df["domain"] = df.task_id.map({t.id: t.domain for t in TASKS})
traps = df[df.category.str.startswith("trap")]
full_df = df[df.config == "full"]

print(f"{len(df):,} runs · {df.task_id.nunique()} tasks · 3 domains · ${df.cost_usd.sum():.2f}\n")
for dom, note in (("penguins", "clean data, no traps"),
                  ("trial", "DESIGNED AGAINST"),
                  ("sales", "🎯 HELD-OUT DOMAIN — never designed against")):
    g = full_df[full_df.domain == dom]
    m, lo, hi = hierarchical_bootstrap(g)
    print(f"  {dom:<9} {m:>6.0%}   95% CI [{lo:.0%}, {hi:.0%}]   n={len(g):>3}   {note}")

# %% [markdown]
# ## It does **better** on the domain it was never tuned for.
#
# That is the most reassuring number in this project, and the one I'd have been most embarrassed to
# be missing.
#
# *(Writing that dataset also found a hole in my own benchmark: `observe.py` has had a detector for
# numeric-columns-stored-as-text since the first commit, and **no task ever exercised it.** A
# benchmark built from one dataset only tests the mechanisms that dataset happens to provoke.)*

# %% [markdown]
# ---
# # 6. 🚨 The punchline — the result that inverted my own thesis

t = traps.groupby("config").agg(pass_rate=("passed", "mean"), naive=("wrong_attractor", "mean"))
print("ON THE TRAP TASKS — the stack works, and it isn't close:\n")
print(f"   full agent      {t.loc['full','pass_rate']:.0%} pass   "
      f"{t.loc['full','naive']:.0%} fell for the naive answer")
print(f"   no guardrails   {t.loc['no_guardrails','pass_rate']:.0%} pass   "
      f"{t.loc['no_guardrails','naive']:.0%} fell for the naive answer")
print(f"\n   → a {t.loc['no_guardrails','naive']/t.loc['full','naive']:.0f}x reduction in the "
      f"wrong-attractor rate. That is the notice–act gap, measured.")

# %%
from evals.stats import ablation_table

label = {"no_briefing": "the deterministic data briefing  ← the DETECTOR",
         "no_guardrails": "every guardrail at once",
         "no_ledger": "the Findings Ledger  ← MY CENTREPIECE",
         "no_verifier": "the fresh-context verifier",
         "no_grounding": "the numeric grounding gate",
         "no_contract": "the Question Contract",
         "no_truncation": "observation truncation"}

abl = ablation_table(df)     # paired bootstrap vs full, 10k hierarchical resamples

print(f"{'REMOVE THIS':<46}{'PASS':>6}{'Δ':>7}{'   95% CI':>17}   VERDICT")
print("─" * 96)
print(f"{'— nothing (the full agent)':<46}{df[df.config=='full'].passed.mean():>6.0%}")
for k, r in abl.iterrows():
    print(f"{label[k]:<46}{r.pass_rate:>6.0%}{r.delta_vs_full:>+7.0%}   "
          f"[{r.lo95:+.0%}, {r.hi95:+.0%}]".ljust(17)
          + f"   {r.verdict}")

print("\nIf the 95% CI crosses zero, I cannot distinguish that mechanism from doing nothing —")
print("and I say so, instead of reporting a point estimate and hoping nobody checks.")

# %% [markdown]
# ## Removing twenty lines of pandas is the largest effect in the study.
#
# I spent this whole design on **gates**. The thing actually carrying the agent is the **detector**
# — the deterministic profile of the data, computed before the model is called even once.
#
# # 🎯 A gate is only as good as the detector feeding it.
#
# The papers describe a **notice–act** gap, and I read it as a failure to **act** — so I built
# machinery to force action.
#
# **The ablation says the leverage is on the *notice* side.** Tell the agent what's in the data,
# deterministically, before it starts, and **it acts on it.** It didn't need to be forced. It
# needed to be *informed*.
#
# The gates weren't *wrong*. They were **redundant** — the detector in front of them was already
# doing the job. Which is exactly why `no_briefing` collapses to the same score as
# `no_guardrails`.
#
# ## And the bigger eval **corrected me about my own centrepiece**
#
# The first version of this benchmark was 15 tasks × 3 runs. It put the Findings Ledger at
# **exactly zero**, and I wrote — in the notebook — *"the ablation does not show the Findings
# Ledger paying for itself."*
#
# At **28 tasks × 10 runs** it comes back at **Δ −5%, CI [−11%, +1%]**. The point estimate says five
# points, and it's the **only** gate whose interval sits almost entirely on the "it helps" side. It
# still grazes zero, so I can't claim it at 95%.
#
# > **"The ledger does nothing" was never a finding.** It was a twenty-point-wide confidence
# > interval, reported as a point estimate. More data didn't confirm my conclusion — it
# > **corrected** it.
#
# That's the cleanest argument I have for building the harness before trusting your own design
# instincts.
#
# ### The honest caveats
#
# - **"No detectable effect" is not "no effect" — but now it's *bounded*.** I can say any real
#   ledger effect is smaller than about 11 points. At 15×3 I couldn't say anything at all.
# - **A gate may be insurance, not throughput.** A grounding check that fires on a few percent of
#   runs can't move an average pass rate — but you don't price a fabricated number in a drug filing
#   by its *frequency*. The right test for a gate is **adversarial, not average**, and this
#   benchmark is an average one. That's a limitation of my evaluation, not evidence against the gate.
# - **One oddity I won't overclaim:** removing the briefing alone (62%) scores *worse* than removing
#   the briefing **and** every gate (69%). Gates without a detector may be *worse than nothing* —
#   they burn the step budget on ceremony with no information behind it. But that paired CI is
#   `[−18%, +4%]`; it crosses zero. **A hypothesis with a mechanism, not a finding.**
#
# ### So what would I build next?
#
# **Not another gate. More detectors.** Every remaining failure is a missing detector — the units
# trap (33%) needs *"flag any column that's multi-modal by batch"*; the ambiguity task (0%) needs
# *"does the question pin down a population, a direction, and a unit?"* Three cheap `if`s.
#
# And then re-run at 10 runs/task, because right now I can see a 27-point effect and I am blind to
# a 5-point one.

# %% [markdown]
# ---
# # 7. The dashboard
#
# Everything above, as something you can poke at: run the agent on any question, **toggle the
# guardrails off and watch it fail**, and browse the 360-run evidence.
#
# Run the cell below — it starts the app and embeds it right here.

# %%
import subprocess, time, socket
from IPython.display import IFrame

PORT = 8501


def _up(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


if not _up(PORT):
    subprocess.Popen(
        ["uv", "run", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    while not _up(PORT):
        time.sleep(0.5)
    time.sleep(2)

print(f"dashboard live → http://localhost:{PORT}")
IFrame(f"http://localhost:{PORT}", width="100%", height=900)

# %% [markdown]
# ---
# # In one paragraph
#
# A data-analysis agent is a **while loop that writes code, runs it, and reads the results**.
# The base model is a bad calculator and a great programmer, so you never ask it for a number —
# you ask it for code. What it *actually* fails at is **keeping the thread**: carrying what it was
# asked, and what it found, into every decision that follows. I built three ledgers and four gates
# to stop it dropping that thread, measured all of them, and found that **the cheapest, dumbest
# piece — a deterministic look at the data before the model sees it — was doing most of the work.**
#
# That's the only thing here I couldn't have got by thinking harder. It came from running the eval.
#
# ---
#
# | | |
# |---|---|
# | **The design** | `docs/DESIGN.md` |
# | **Every decision, and what would change my mind** | `docs/DECISIONS.md` |
# | **The buildup, from one API call to the whole agent** | `notebooks/01` → `07` |
# | **The agent** | `agentlib/` — 885 lines |
# | **The benchmark and the ablations** | `evals/` |

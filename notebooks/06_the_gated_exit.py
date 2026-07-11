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
# # 06 — The gated exit
#
# ### *"Finishing is an action, not a default."*
#
# **Previously:** the two ledgers made the agent keep the question and act on what it found. It got
# the sign right.
#
# But I still had a nagging question: **where did that number come from?**
#
# It said `0.150`. Did any code it ran actually *print* that? Or did it assemble it at the last
# second in its head — the way it invented `0.625` back in notebook 04?
#
# I didn't know. And "I don't know" is not acceptable for a number that goes into a drug filing.
#
# **The third dropped thing is the number itself.**

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

from agentlib import Config, run_agent
from agentlib.llm import METER
from agentlib.report import AnalysisReport, grounded, rejection_message

FILES = ["../data/trial.csv", "../data/data_dictionary.md"]

# %% [markdown]
# ---
# ## 1. The failure this is built to catch
#
# This is not hypothetical. It's a **frontier model**, in DrugDiscoveryBench, on a real task:
#
# > *"its own code printed the correct group-level count of **1**, but the final tally used the
# > atom-level count of **2** and double-counted the bridge. It reported 8 interactions instead
# > of 7."*
#
# Read that once more. **The correct answer was on its screen.** It computed it. It printed it.
# And then, when writing up, it used a different number.
#
# Their taxonomy calls this a **derivation error** (18.6% of failures) and a **final-answer slip**
# (3.5%). Together, **more than one failure in five** is the agent contradicting *its own output*.
#
# ### An LLM reviewer *might* catch that. A regex catches it **every single time.**

# %% [markdown]
# ## 2. Ledger 3 — grounding. Fifteen lines, no model.
#
# The rule: **every number in the answer's `evidence` must appear in the stdout of code that
# actually ran.**
#
# Pull the numbers out of both, normalise them (strip commas and `%`, round to 4 significant
# figures, and check `×100` and `÷100` for percent-vs-fraction confusion), and demand containment.

# %%
# Pretend this is what the agent's code actually printed:
printed_output = """
Treatment response rate: 0.6448
Control response rate:   0.7233
Stratified effect: mild=0.134 moderate=0.167 severe=0.175
Adjusted treatment effect: 0.1501
"""

# An honest report: every number traces back to the output above.
honest = AnalysisReport(
    answer="The treatment increases the response rate by 0.150 after adjusting for severity.",
    value=0.1501,
    method="Stratified by severity, then standardised to the overall severity distribution.",
    evidence=["Adjusted treatment effect: 0.1501",
              "Within-stratum effects: 0.134, 0.167, 0.175"],
)

ok, ungrounded = grounded(honest, printed_output)
print(f"honest report  → grounded: {ok}")

# %%
# A report that quietly invents a number. Note it's *plausible* — that's the whole danger.
slippery = AnalysisReport(
    answer="The treatment increases the response rate by 0.21 after adjusting for severity.",
    value=0.21,
    method="Stratified by severity.",
    evidence=["Adjusted treatment effect: 0.21", "Odds ratio: 1.84"],
)

ok, ungrounded = grounded(slippery, printed_output)
print(f"slippery report → grounded: {ok}")
print(f"                  numbers that were NEVER printed: {ungrounded}")
print()
print(rejection_message("ungrounded", values=ungrounded))

# %% [markdown]
# > ### 💡 Cheap and deterministic goes *before* expensive and probabilistic.
# >
# > This check costs **nothing**. No API call, no latency, no chance of being talked out of its
# > position. It cannot be persuaded, it cannot be flattered, and it does not have an off day.
# >
# > It is deliberately **conservative**: it hard-gates only the `evidence` list — where the agent
# > is making an explicit factual claim — and merely warns on numbers in the prose. A gate that
# > fires wrongly is a gate that gets switched off by whoever has to live with it.

# %% [markdown]
# ---
# ## 3. Gate 4 — a reviewer who has never seen the reasoning
#
# Grounding proves the numbers are *real*. It does not prove they answer the *question*.
#
# DrugDiscoveryBench again, on an agent that silently dropped a scope qualifier halfway through:
#
# > *"**The last chance to catch the slip is at the final answer**: a human who had misread the
# > task the same way would look at the result, recognize it as a meaningless response to the
# > user's actual goal, and backtrack. **None of the failing models caught this.**"*
#
# So we add that human. One API call.
#
# ### The one design detail that makes it real
#
# > # 🔒 The verifier does **not** see the agent's reasoning.
#
# It sees the Question Contract, the data briefing, the code that ran, what it printed, and the
# draft answer. **That is all.**
#
# Show a reviewer the chain of thought and it gets anchored by the agent's own narrative and
# rubber-stamps the conclusion. **A verifier that reads the transcript measures nothing.** The
# fresh context *is* the mechanism — it is not an implementation detail, it is the entire point.
#
# It is also a **different model family** (agent: Qwen; verifier: gpt-oss). A model reviewing its
# own work shows self-preference bias — Zheng et al. 2023, *"Judging LLM-as-a-Judge"*.
#
# It scores four things:
#
# | | |
# |---|---|
# | **Scope** | does the answer address the *contract's* estimand — same population, same units? |
# | **Grounding** | is every claim supported by the shown output? |
# | **Overreach** | a causal claim from correlational evidence? |
# | **Blind spots** | does the output show something the conclusion ignores? |

# %% [markdown]
# ### And this is why we don't need a "critic agent"
#
# The only genuine benefit of a second agent here is **an opinion that isn't anchored on the first
# one's reasoning.** That costs exactly one API call.
#
# Everything else a multi-agent system would add — coordination, message passing, shared state,
# a whole new class of failure where two agents misunderstand each other — is overhead we get to
# not have.

# %% [markdown]
# ---
# ## 4. The four gates, in order
#
# `submit_answer` is not a way to leave the loop. It is a **proposal**, and it runs a gauntlet:
#
# | | Gate | Cost |
# |---|---|---|
# | **1** | **Schema** — pydantic validates the shape | free |
# | **2** | **Ledger** — any finding still open? | free |
# | **3** | **Grounding** — every number traced to real output | free |
# | **4** | **Verifier** — fresh eyes, different model | 1 API call |
#
# Cheapest first. **Never pay for an LLM call to find something a regex would have caught.**
#
# Any gate can bounce the answer back into the loop with a message saying exactly what to fix —
# which, as notebook 02 established, is just another observation.

# %% [markdown]
# ---
# # 5. The complete agent. All four gates. Watch the verifier work.

# %%
run = run_agent(
    "Does the treatment improve the response rate? Report the treatment effect as a "
    "difference in proportions (treatment minus control).",
    FILES,
    Config(),   # everything on
)

# %% [markdown]
# ## The structured answer

# %%
import json

r = run.report
print("ANSWER     :", r["answer"])
print("VALUE      :", r["value"], "   (ground truth: +0.150)")
print("CONFIDENCE :", r["confidence"])
print("\nMETHOD     :", r["method"][:280])
print("\nEVIDENCE:")
for e in r["evidence"]:
    print("  •", e[:88])
print("\nCAVEATS:")
for c in r["caveats"]:
    print("  •", c[:88])
print("\nAUDIT TRAIL (what it noticed, what it did):")
for f in r["findings"]:
    icon = {"acted": "✅", "dismissed": "⚪", "open": "🔴"}[f["status"]]
    print(f"  {icon} {f['observation'][:72]}")
    print(f"     ↳ {f['resolution'][:72]}")

print(f"\nGATES THAT FIRED: {run.rejections or 'none — accepted first time'}")

# %% [markdown]
# ## Why *this* is the deliverable — not the number
#
# A scientist cannot use a number they can't interrogate. What they need is:
#
# - the **answer**, and the exact **quantity**
# - the **method**, and *why that method and not the obvious one*
# - the **evidence**, every figure traceable to code that ran
# - the **caveats**, including any the reviewer insisted on
# - and the **audit trail**: everything the agent noticed, and what it decided to do about it
#
# That last one is the thing neither paper's system produces, and it's the one I'd want most.
# It's the difference between *"the model said 0.15"* and *"the model saw the confounding, said so,
# adjusted for it, and here is the line where it did."*
#
# One is an oracle. The other is a colleague.

# %%
print(METER)

# %% [markdown]
# ---
# # Where we are — the agent is complete
#
# | | |
# |---|---|
# | **Ledger 3: Grounding** | Every number traced to real output. Free, deterministic, unarguable. |
# | **Gate 4: Verifier** | Fresh eyes, different model, never sees the reasoning. |
# | **The exit** | Four gates, cheapest first. Finishing is an action you have to earn. |
# | **The output** | A structured, audited report — not a number. |
#
# ### 🔜 And now the only question that matters.
#
# I've shown you the agent working. Once.
#
# **So what?** It's a non-deterministic system. One good run is an anecdote. Maybe I got lucky.
# Maybe the ledger does nothing and the system prompt was carrying it. Maybe the verifier is
# theatre. Maybe I could have got the same result with a bigger model and no guardrails at all,
# and saved myself the trouble.
#
# **I have no evidence for any of my claims yet.** Every mechanism in this design is currently
# just a story I told you.
#
# Time to find out which of them are true.
#
# **→ `07_evaluation.ipynb`**

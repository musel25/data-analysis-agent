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
# # 04 — One tool to rule them all
#
# ### *"A general tool beats N specific ones."*
#
# **Previously:** fixed tools capped the agent's ceiling. Ask for a groupby it didn't have, and it
# thrashed.
#
# The fix is to stop writing tools and give it **one**: run Python.
#
# This is the chapter where the agent becomes genuinely powerful. It is *also* the chapter where
# it produces its first confidently wrong answer — and that failure is the reason the rest of this
# series exists.

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

from agentlib.executor import PyExecutor
from agentlib.llm import llm, METER

# %% [markdown]
# ## 1. The executor: a Python session that remembers
#
# The critical word is **persistent**. Variables survive between calls.
#
# Why does that matter so much? Because it's how an analyst actually works: load the data once,
# then filter it, then inspect *that*, then model *that*. If every step started from a blank
# slate, the agent would have to re-read the CSV every single time — and step 4 could never build
# on what step 2 discovered.
#
# It is, precisely, a Jupyter kernel. Which is a pleasing thing to be building inside a Jupyter
# notebook.

# %%
ex = PyExecutor()

r = ex.run("""
import pandas as pd
df = pd.read_csv('../data/trial.csv')
print(df.shape)
""")
print(r.stdout)

# a SECOND, separate call — and `df` is still there.
r = ex.run("print(df['arm'].value_counts())")
print(r.stdout)

# %% [markdown]
# ## 2. The whole executor, in twenty lines
#
# There is no cleverness hiding in here:
#
# ```python
# class PyExecutor:
#     def __init__(self):
#         self.ns = {}                      # <- the persistent namespace. that's the "kernel".
#
#     def run(self, code):
#         buf = io.StringIO()
#         try:
#             with redirect_stdout(buf):
#                 exec(code, self.ns)       # <- run it, in OUR namespace
#         except Exception:
#             error = traceback.format_exc()
#         return buf.getvalue(), error      # <- stdout and error come back as TEXT
# ```
#
# `exec(code, self.ns)` — the dict `self.ns` *is* the kernel. Variables defined in one call live
# in that dict and are visible to the next one.
#
# > ## 🛑 STOP. This is not a sandbox.
# >
# > `exec()` runs in **this process**. Model-written code can read your files, exhaust your RAM,
# > and import anything installed. Containing it in a separate namespace stops it clobbering *my*
# > variables. It does not stop it doing anything else.
# >
# > I am being loud about this because the honest version of this limitation is worth more than a
# > quiet overclaim. Both papers behind this design ran their agents in **Docker containers with
# > no network access** — that is the correct production answer.
# >
# > The tool contract here — *code in, stdout out* — is deliberately identical to what a
# > container-backed executor exposes. So hardening this is a **one-class swap**, not a redesign.
# > See `docs/DECISIONS.md` D05.

# %% [markdown]
# ---
# ## 3. 🐛 A bug I hit, and the tool-design lesson in it
#
# The first version of my executor was a plain `exec(code, ns)`. Then I watched a real run and
# saw this:
#
# ```
# [1] df = pd.read_csv('trial.csv')
#     df.head()                         → (no output)
# [2] data[data['treatment'] == 1]      → KeyError: 'treatment'
# [3] data.columns                      → (no output)
# [4] data[data['treatment'] == 1]      → KeyError: 'treatment'      ← the SAME failing code
# ```
#
# Look at steps 1 and 3. The agent asked to see the data — and got **nothing back**.
#
# Here's why. In a Jupyter cell, a bare expression on the last line gets displayed. That is a
# feature of the *kernel*, not of Python. Plain `exec()` does not do it: it evaluates
# `df.head()`, throws the result away, and returns silence.
#
# So the agent went **blind**. It believed it had looked at the data. It had not. So it *guessed*
# the column names (`treatment`, `response` — reasonable guesses for a clinical trial!), got a
# `KeyError`, and then re-ran the identical broken code.
#
# > ### 💡 The model was not being stupid. My tool was lying.
# >
# > It assumed Jupyter semantics — which is completely reasonable, because every notebook on the
# > internet behaves that way and **the tool is literally called `run_python`**.
# >
# > I could have "fixed" this by putting *"ALWAYS USE print()"* in capitals in the system prompt.
# > That would have worked worse, cost tokens on every call, and taught me nothing.
# >
# > **When a model misuses a tool, first ask whether the tool behaves the way its name implies.**
# > A tool that surprises will be misused. The fix belongs in the tool, not in a sterner prompt.
#
# So `PyExecutor` parses the code with `ast`, and if the last statement is an expression, it
# evaluates and echoes it — exactly like a real kernel:

# %%
r = ex.run("""
df2 = pd.read_csv('../data/trial.csv')
df2.head(2)
""")
print("bare expression, no print() — and we SEE it now:\n")
print(r.stdout[:260])

# %% [markdown]
# **And the death loop?** Two guards, both cheap:
#
# 1. **Duplicate-code short-circuit** — identical code returns the cached result without running.
#    Re-running it cannot produce a different answer; it can only burn budget.
# 2. **Error fingerprinting** — the same error twice injects *"state the root cause in one
#    sentence before writing more code."* Three times: *"this isn't a typo, it's the approach —
#    change strategy."*
#
# The two stages matter. Jumping straight to "try something else" makes the agent abandon a
# 95%-correct approach over a typo. Make it **diagnose** first. That's Reflexion (Shinn et al.
# 2023) in about ten lines.

# %% [markdown]
# ---
# ## 4. Wire it up as the agent's only tool

# %%
TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": ("Run Python in a persistent session. Variables persist between calls. "
                        "pandas is imported as pd. Returns whatever you print()."),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}]

SYSTEM = """You are a data analyst. You work by writing and running Python.

- Never estimate a number. Compute it and print it.
- One step per cell: small code, print the result, then think about what you got.
- When you have the answer, say it in plain language."""


def run_agent(question: str, max_steps: int = 10) -> str:
    ex.reset()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]

    for step in range(1, max_steps + 1):
        reply = llm(messages, tools=TOOLS)
        messages.append(reply.raw())

        if not reply.tool_calls:
            print(f"[{step}] ✅ answered")
            return reply.content

        for call in reply.tool_calls:
            code = call.args.get("code", "")
            res = ex.run(code)

            obs = res.stdout or "(no output — did you print anything?)"
            if res.error:
                obs += f"\nTRACEBACK:\n{res.error}"

            first = code.strip().splitlines()[0][:50] if code.strip() else ""
            flag = "❌" if res.error else "✓"
            print(f"[{step}] {flag} {first}")
            if res.error:
                print(f"      └─ {res.error.splitlines()[-1][:66]}")

            messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})

    return "(hit the step limit)"


# %% [markdown]
# ## 5. Give it the question that broke notebook 03

# %%
answer = run_agent(
    "In the trial data (../data/trial.csv), is the response rate different between the "
    "treatment and control arms? Break it down by severity too."
)
print("\n" + "═" * 78)
print(answer)

# %% [markdown]
# > ### 💡 It answered. And it did it by writing exactly the pandas *I* would have written.
# >
# > The ceiling is gone. It's not limited to verbs I anticipated — it's limited to what can be
# > expressed in code, which is everything.
# >
# > One general tool beat N specific ones, and it made the agent *simpler*, not more complex.

# %% [markdown]
# ---
# ## 6. Errors are just observations — now watch it debug itself
#
# In notebook 02 we established that a traceback is just text you hand back. Let's make that pay.
#
# I'll give it a task that will *definitely* throw: comparing a numeric column against a string.

# %%
answer = run_agent(
    "In ../data/trial.csv, what is the mean age of patients whose severity is 'Severe'? "
    "(Note: I'm not sure of the exact capitalisation used in the file.)"
)
print("\n" + "═" * 78)
print(answer)

# %% [markdown]
# > ### 💡 The moment worth pausing on
# >
# > If it hit an error (or an empty result from the wrong capitalisation), look at what happened
# > next: it read the traceback, worked out what was wrong, and **fixed its own code on the next
# > step.** Nobody wrote retry logic. Nobody wrote a "check capitalisation" handler.
# >
# > We just put the error text in the message list, and the loop did the rest.
# >
# > This self-repair is the single biggest capability the code-execution tool unlocks, and it is
# > *free* — it comes from feeding failure back as an observation.

# %% [markdown]
# ---
# # 7. 🚨 And now the failure that changes everything.
#
# The agent looks great. Let's ask it the question that actually matters — the one this trial was
# run to answer.
#
# **"Does the treatment work?"**

# %%
answer = run_agent(
    "Using ../data/trial.csv: does the treatment improve the response rate? "
    "Give me the treatment effect as a difference in proportions."
)
print("\n" + "═" * 78)
print(answer)

# %% [markdown]
# ### Note how *well* it did that.
#
# Three steps. Clean pandas. It printed real numbers from the real file — no hallucination, no
# guessing, no death loop. After the executor fix, this agent is **more competent than it was an
# hour ago.**
#
# And it just told me a drug that works doesn't work.
#
# ## Now let me tell you what the right answer is.

# %%
import pandas as pd

raw = pd.read_csv("../data/trial.csv")

# the agent's likely path: group by arm, compare.
marginal = raw.groupby("arm")["responded"].mean()
naive_effect = marginal["treatment"] - marginal["control"]

# what the data actually contains: treatment was given preferentially to SICKER patients.
dedup = raw.sort_values("sample_seq").groupby("patient_id", as_index=False).last()
strat = dedup.groupby(["severity", "arm"])["responded"].mean().unstack()
per_stratum = strat["treatment"] - strat["control"]
weights = dedup["severity"].value_counts(normalize=True)
adjusted = (per_stratum * weights).sum()

print("RESPONSE RATE BY ARM (what the agent probably computed):")
print(marginal.round(3).to_string())
print(f"\n  → naive treatment effect: {naive_effect:+.3f}   TREATMENT LOOKS WORSE\n")

print("RESPONSE RATE BY SEVERITY × ARM (what it needed to look at):")
print(strat.round(3).to_string())
print(f"\n  → treatment effect within each stratum:")
print(per_stratum.round(3).to_string())
print(f"\n  → severity-adjusted treatment effect: {adjusted:+.3f}   TREATMENT IS BETTER")

print("\n" + "═" * 78)
print(f"  THE NAIVE ANSWER : {naive_effect:+.3f}")
print(f"  THE TRUE ANSWER  : {adjusted:+.3f}")
print("  THESE HAVE OPPOSITE SIGNS.")
print("═" * 78)

# %% [markdown]
# ## What just happened: Simpson's paradox
#
# Look at the stratified table. **The treatment wins in every single severity group.** Mild,
# moderate, severe — the drug is better in all three.
#
# And yet, pooled together, the treatment arm looks *worse*.
#
# Why? Because this trial was **not randomised** (it says so in `data_dictionary.md`, which the
# agent never opened). Clinicians gave the drug to the patients who were already sicker. Severe
# patients do badly regardless. So the treatment arm is loaded with hard cases, and the naive
# comparison is not comparing treatment against control — it's comparing *sick people* against
# *healthy people*.
#
# The agent computed a number correctly. Its pandas was flawless. **And it reached the opposite
# of the truth**, with total confidence, and told me a drug that works doesn't.
#
# ### And there are three more landmines in this file it walked straight past:
#
# | Landmine | What happens if you miss it |
# |---|---|
# | `-999` in `biomarker_baseline` | It's a QC-failure code, not a measurement. Mean baseline comes out at **−64** (remember that number from notebook 01?) |
# | 48 patients appear **twice** | Re-tested samples. Counting rows gives 848 "patients"; there are 800. |
# | `assay_batch == "B"` reports in **µg/L**, not ng/mL | Batch B values are 10× too large. Pool them and every biomarker number is garbage. |
#
# All three are documented in `data/data_dictionary.md`. The agent was told the files were there.
# It read the CSV and ignored the documentation.

# %% [markdown]
# ---
# # 🎯 This is exactly what the papers found.
#
# This is not my toy failure. It is *the* failure, and it is the reason both papers exist.
#
# > **GeneBench-Pro** (Li & Ho, 2026) — 129 multistage statistical-reasoning problems on messy
# > biomedical data. Best frontier model: **31.5%**. On **45.7% of problems it scores zero across
# > ten attempts.** Their diagnosis:
# >
# > *"models often complete substantial portions of the workflow but exhibit a consistent gap
# > between **noticing** and **acting** — identifying local diagnostic signals but failing to
# > propagate the implications to the corresponding analysis decision. As a result, models often
# > select wrong estimators or persist on initially plausible but incorrect analysis paths."*
# >
# > And more precisely (p. 13):
# >
# > *"the agent notices the relevant local diagnostic clue but **treats it as a local data
# > cleaning issue rather than as evidence that should change the downstream statistical method**."*
#
# > **DrugDiscoveryBench** (Akyürek, Tu et al., 2026) — 82 expert drug-discovery tasks. Best
# > agent: **51.6%**. They hand-classified 226 failing runs:
# >
# > | Failure mode | Share |
# > |---|---|
# > | **Domain reasoning** — *"applies an incorrect scientific premise or misinterprets the data it has, even though its inputs and tools are correct"* | **54.0%** |
# > | Derivation error — right approach, wrong calculation | 18.6% |
# > | Retrieval — *"failing to read a provided file"* | 16.4% |
# > | Constraint — violates an explicit instruction in the prompt | 7.5% |
# > | Final-answer slip | 3.5% |
# >
# > *"the agents knew which database to query and how to compute the property the task asked for
# > at a high level. But somewhere along the execution **they drop a constraint, commit too early,
# > fail to backtrack**."*
#
# ## Read that table again, because it tells you where to spend your engineering.
#
# **Over half of all failures are not coding failures.** The tools worked. The Python was
# correct. The model *had* everything it needed.
#
# It just didn't let what it saw change what it did.
#
# ### And notice the trap I nearly fell into myself.
#
# When the agent was going blind and death-looping, the obvious diagnosis was *"it needs a better
# tool."* I fixed the tool. The agent got faster, cleaner, and more competent.
#
# **And its answer got no less wrong.** It just reached the wrong answer more efficiently.
#
# That is the whole lesson in one move: **capability is not correctness.** Every instinct says to
# respond to a failing agent by making it more powerful — better tools, bigger model, more
# context. None of those touch a notice–act gap. You cannot fix a thread-keeping problem with
# horsepower.
#
# > ### 💡 The design thesis of this entire project
# >
# > The bottleneck is **not** code generation. **Not** knowledge. **Not** context length.
# >
# > It is **thread-keeping** — carrying what you were asked, and what you found, into every
# > decision that follows.
# >
# > So don't spend the budget making the model smarter. Spend it making it **structurally
# > impossible to drop the thread.**
#
# Three things get dropped: **the question**, **the finding**, and **the number**.
#
# So we're going to make each one into explicit state that the agent cannot leave behind.

# %%
print(METER)

# %% [markdown]
# ---
# # Where we are
#
# | | |
# |---|---|
# | **We learned** | One general tool (`run_python`) + a persistent namespace = an agent that can do anything pandas can. |
# | **We got for free** | Self-debugging. Feed the traceback back; it fixes its own code. |
# | **We saw it fail** | Confidently reported the **opposite of the truth** on the real question. Ignored the docs, the sentinels, the duplicates, the units, and the confounding. |
# | **And the papers agree** | 54% of real agent failures are exactly this: not coding errors — *thread-dropping* errors. |
#
# ### 🔜 Ledger 1 and Ledger 2
#
# Next: make the question **stick**, and make a noticed problem **impossible to ignore**.
#
# **→ `05_keeping_the_thread.ipynb`**

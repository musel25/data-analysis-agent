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
# # 01 — Hello, model
#
# ### *"The conversation is an illusion."*
#
# We are going to build a data-analysis agent from nothing. No LangChain, no framework, no magic.
# By notebook 07 it will explore messy clinical data, catch its own mistakes, and hand back a
# structured, audited answer — and the whole thing will be about 200 lines of Python that you
# watched get written.
#
# This chapter has exactly one job: **show you what the raw material actually is.**
#
# Everything else in this series is a response to a problem you are about to watch happen.

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

from agentlib.llm import llm, METER
from agentlib import config

print("model :", config.AGENT_MODEL)
print("host  :", config.BASE_URL)

# %% [markdown]
# ## 1. An LLM call is a function. That's all it is.
#
# You send a list of messages. You get a message back. There is no memory, no state, no session.
#
# It's `f(messages) -> message`. Nothing more.

# %%
msg = llm([{"role": "user", "content": "In one sentence: what is a p-value?"}])
print(msg.content)

# %% [markdown]
# ## 2. The "conversation" is something *you* maintain
#
# The model does not remember the last message. It cannot. Each call is fresh — the model sees
# the entire list of messages, from scratch, every single time.
#
# When you chat with an LLM and it "remembers" what you said, that is **the client resending the
# whole history on every turn.** The illusion of memory is just a growing list.
#
# Watch. First, ask a follow-up *without* the history:

# %%
# No history. The model has no idea what "it" refers to.
orphan = llm([{"role": "user", "content": "What's a common mistake people make with it?"}])
print("WITHOUT history:\n", orphan.content[:300])

# %%
# Now with the history — we resend everything, and *we* are the ones doing the remembering.
conversation = [
    {"role": "user", "content": "In one sentence: what is a p-value?"},
    {"role": "assistant", "content": msg.content},
    {"role": "user", "content": "What's a common mistake people make with it?"},
]
withctx = llm(conversation)
print("WITH history:\n", withctx.content[:300])

# %% [markdown]
# > ### 💡 The first thing to internalise
# >
# > **You own the context.** Nothing goes into the model's head that you did not put there.
# >
# > That sounds obvious. It is the single most important fact in this entire series — because
# > every failure we hit from here on will be, in some form, *the wrong thing being in the
# > context, or the right thing missing from it.* Designing an agent is mostly designing what
# > the model gets to see.

# %% [markdown]
# ---
# ## 3. Now the problem. Ask it to do our actual job.
#
# We have a real CSV on disk: 344 penguins. Let's ask about it.

# %%
import pandas as pd

penguins = pd.read_csv("../data/penguins.csv")
print(penguins.head(3).to_string())
print(f"\n{len(penguins)} rows")

# %% [markdown]
# The model **cannot see this file.** It's on my disk. The model is on a GPU in another country.
#
# So what happens if we just... ask it anyway?

# %%
naive = llm([{"role": "user", "content":
              "The file data/penguins.csv contains the Palmer Penguins dataset. "
              "What is the mean body mass in grams? Reply with just the number."}])
print("MODEL SAYS :", naive.content.strip()[:80])
print("THE TRUTH  :", penguins["body_mass_g"].mean())

# %% [markdown]
# ### Wait. That's... almost exactly right?
#
# It didn't refuse. It gave a number, and the number is *correct to several decimal places.*
#
# **It did not compute that.** It cannot compute anything; it never saw the file.
#
# It **remembered** it. Palmer Penguins is one of the most-used teaching datasets on the
# internet. Its mean body mass is sitting in the model's weights, memorised from a thousand
# tutorials. Look closely at the last digits though — they don't quite match. It's not recalling
# a fact, it's *reconstructing* one, and the reconstruction is subtly wrong.
#
# > ### 🚨 This is worse than a hallucination, not better.
# >
# > A model that confidently invents numbers is dangerous. A model that confidently invents
# > numbers **and happens to be right on the datasets you test it with** is *far* more dangerous
# > — because it will pass your evaluation and then fail on real data.
# >
# > This is exactly why GeneBench-Pro builds its benchmark from *simulated* data: analysis
# > problems are "specifically chosen so they do not recapitulate well-known textbook examples
# > or papers, so as to avoid the risk of benchmarking against memorized solutions."
#
# So let's give it a dataset it *cannot* have memorised. This one was generated on my laptop
# this week from a random seed. It has never existed before, anywhere.

# %%
trial = pd.read_csv("../data/trial.csv")
print(f"{len(trial)} rows — synthetic clinical trial data, generated from seed 20260701")
print(trial[["patient_id", "arm", "severity", "biomarker_baseline"]].head(3).to_string(index=False))

# %%
unseen = llm([{"role": "user", "content":
               "The file data/trial.csv contains a clinical trial dataset with a column "
               "biomarker_baseline. What is its mean value? Reply with just the number."}])
print("MODEL SAYS :", unseen.content.strip()[:80])
print("THE TRUTH  :", round(trial["biomarker_baseline"].mean(), 2))

# %% [markdown]
# > ### 🚨 *There* it is.
# >
# > No memorised answer to fall back on, so it **invented one.** A confident, plausible,
# > specific number with believable units and magnitude. It did not say *"I cannot see that
# > file."* It just... answered.
# >
# > In data analysis this is the worst failure mode there is: not an error, but **a wrong answer
# > that looks exactly like a right one.** If this number landed in a slide, nobody would blink.
#
# *(And note the true value is bizarre — a negative mean for a biomarker. Hold that thought.
# That's a landmine we'll step on properly in notebook 04.)*

# %% [markdown]
# ## 4. "Fine — I'll paste the data in."
#
# The obvious fix. Give it the rows and let it compute. Let's paste in 40 of them and ask for
# something a little harder than a mean.

# %%
sample = penguins.dropna().head(40)
pasted = llm([{"role": "user", "content":
               f"Here is a table of penguins:\n\n{sample.to_csv(index=False)}\n\n"
               "What is the standard deviation of body_mass_g? Reply with just the number."}])

truth = sample["body_mass_g"].std()
print("MODEL SAYS :", pasted.content.strip()[:80])
print("THE TRUTH  :", round(truth, 2))

# %% [markdown]
# It can see every single number, and it *still* gets it wrong (or, if it happens to get close,
# it got close by luck — run it again and watch it move).
#
# This is not a knowledge problem. The model knows the formula for standard deviation perfectly.
# It's an *arithmetic* problem: the model is predicting tokens, not executing operations.
#
# And pasting doesn't scale anyway. 40 rows fit. Our real dataset — the one in notebook 04 — has
# 848. A production table has 200 million. You cannot paste a database into a prompt.

# %% [markdown]
# ---
# ## 5. The insight that shapes everything after this
#
# The model is a **bad calculator**.
#
# But watch this:

# %%
code = llm([{"role": "user", "content":
             "Write Python (pandas) to compute the standard deviation of the body_mass_g column "
             "in data/penguins.csv. Just the code, no explanation."}])
print(code.content)

# %% [markdown]
# That code is **perfect.** It would give the exactly correct answer, on 344 rows or 344 million.
#
# > # 🎯 The whole series in one line
# >
# > ## The model is a bad calculator and a great programmer.
# > ## So stop asking it for answers. Ask it for **code**, and run the code yourself.
#
# That is what a data-analysis agent *is*. Everything from here is working out how to do that
# reliably — because as we're about to find out, "run the code it writes" is where the real
# problems start, not where they end.

# %% [markdown]
# ---
# ## 6. Housekeeping: the two things every call needs
#
# Before we go further, two small pieces of infrastructure that will run through every notebook.
# They're already inside `llm()` — worth knowing they're there.
#
# **A cost meter.** An agent makes *many* calls. If you can't say what a run costs, you can't
# make engineering decisions about it. So every call is metered.
#
# **A cache.** Every response is saved to disk, keyed by a hash of the request. Same request →
# same answer, instantly, free, offline. That means:
# - these notebooks run without an API key (set `LIVE = False`),
# - the walkthrough can't be sabotaged by a rate limit or a bad day from the model,
# - and — as we'll see in notebook 07 — it's how you write *deterministic tests for a
#   non-deterministic system*.

# %%
print("this notebook so far:", METER)

# %% [markdown]
# Under a cent. Good — because we're going to make a *lot* of calls.

# %% [markdown]
# ---
# # Where we are
#
# | | |
# |---|---|
# | **We learned** | An LLM call is `f(messages) → message`. You own the context entirely. |
# | **We saw it fail** | On a *famous* dataset it recited a memorised answer (dangerous — it looks like competence). On an *unseen* one it invented a confident number. Given data it could see, it did the arithmetic wrong. |
# | **The way out** | It writes *excellent* code. Let it write code; we'll run it. |
#
# ### 🔜 But there's a gap.
#
# We got code back — as a **string, in prose**, wrapped in markdown fences. To use it we'd have to
# regex it out of the text and hope the model didn't add commentary. That's brittle and horrible.
#
# There is a proper protocol for "the model wants something done." It's called **tool calling**,
# and it is nowhere near as magical as it sounds.
#
# **→ `02_tool_calling.ipynb`**

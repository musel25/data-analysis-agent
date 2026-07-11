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
# # 03 — An agent is a while loop
#
# ### *"Autonomy is not architecture."*
#
# **Previously:** we hand-wired one round trip. Model asks → we run → we answer → model concludes.
#
# Real analysis needs *many* steps, and you don't know how many in advance. So we stop hand-wiring
# and write a loop.
#
# This notebook is where the word "agent" stops being intimidating. It is about thirty lines.

# %%
import sys, os, json
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
from agentlib.llm import llm, METER

# %% [markdown]
# ## 1. A few tools this time

# %%
def list_files(directory: str = "../data") -> str:
    return "\n".join(sorted(os.listdir(directory)))


def inspect_csv(path: str) -> str:
    df = pd.read_csv(path)
    return (f"shape: {df.shape}\n\ndtypes:\n{df.dtypes.to_string()}\n\n"
            f"head:\n{df.head(3).to_string()}")


def column_stats(path: str, column: str) -> str:
    df = pd.read_csv(path)
    if column not in df.columns:
        return f"ERROR: no column '{column}'. Available: {list(df.columns)}"
    return df[column].describe().to_string()


REGISTRY = {"list_files": list_files, "inspect_csv": inspect_csv, "column_stats": column_stats}

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List files in a directory.",
        "parameters": {"type": "object",
                       "properties": {"directory": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "inspect_csv", "description": "Shape, dtypes and first rows of a CSV.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "column_stats", "description": "Summary statistics for one column of a CSV.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}, "column": {"type": "string"}},
                       "required": ["path", "column"]}}},
]

# %% [markdown]
# ## 2. The loop. This is the whole thing.
#
# Read it slowly — there is genuinely nothing else to an agent.

# %%
def run_agent(question: str, max_steps: int = 8) -> str:
    messages = [
        {"role": "system", "content":
         "You are a data analyst. Use the tools to look at real data. "
         "Never guess a number — find it."},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        reply = llm(messages, tools=TOOLS)
        messages.append(reply.raw())

        # No tool call means the model is done talking to the tools — it has an answer.
        if not reply.tool_calls:
            print(f"[{step}] ✅ done")
            return reply.content

        # Otherwise: run whatever it asked for, append the result, and go round again.
        for call in reply.tool_calls:
            try:
                observation = REGISTRY[call.name](**call.args)
            except Exception as e:
                observation = f"ERROR: {type(e).__name__}: {e}"   # errors are just observations

            print(f"[{step}] {call.name}({json.dumps(call.args)[:52]}) "
                  f"→ {observation.splitlines()[0][:44]}")

            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})

    return "(hit the step limit without answering)"


# %% [markdown]
# > ### 💡 That's it. That's an agent.
# >
# > ```
# > while not done:
# >     ask the model
# >     if it wants a tool  → run it, append the result
# >     else                → it has an answer, stop
# > ```
# >
# > No planner. No memory system. No orchestration. **The message list is the memory.** The loop
# > is the autonomy. Everything a framework adds sits on top of these fifteen lines.
#
# Now watch it do something nobody scripted.

# %%
answer = run_agent("What data files do I have, and what's in the clinical trial one? "
                   "Give me a one-paragraph orientation.")
print("\n" + "─" * 78)
print(answer)

# %% [markdown]
# ## 3. Read what just happened
#
# Nobody told it to call `list_files` first. Nobody told it that after listing files it should
# inspect the interesting one. It **chained the calls itself**: each observation changed what it
# decided to do next.
#
# That is the thing people mean by "agentic," and it fell out of a `for` loop and a list.

# %% [markdown]
# ---
# ## 4. Now let's break it.
#
# Here's a real analytical question — the kind someone would actually ask of this dataset.

# %%
answer = run_agent(
    "In the trial data, is the response rate different between the treatment and control arms? "
    "Break it down by severity as well.",
    max_steps=8,
)
print("\n" + "─" * 78)
print(answer)

# %% [markdown]
# > ### 🚨 Watch it flail.
# >
# > It has `column_stats`. It does **not** have "group by two columns and compute a rate." So it
# > can't answer. It will thrash — calling `column_stats` on `arm`, on `severity`, on `responded`
# > — collecting fragments that cannot be combined, and then either give up or, worse, **make
# > something up from the fragments.**
# >
# > (Look at the answer above. Did it produce numbers it never actually computed? That's the
# > notebook-01 failure, wearing a costume.)
#
# ### The real lesson: my tools capped its ceiling.
#
# I chose three verbs — list, inspect, describe — and in doing so I silently decided **every
# question this agent will ever be able to answer.** A groupby wasn't on the menu, so a groupby
# is impossible.
#
# I could add a `groupby` tool. And then a `correlate` tool. And a `filter` tool, and a
# `pivot` tool, and a `merge` tool, and...
#
# Stop. I'm reimplementing pandas, one tool at a time, badly.
#
# There is already a tool that can do *anything* pandas can do.
#
# ### It's called Python.

# %%
print(METER)

# %% [markdown]
# ---
# # Where we are
#
# | | |
# |---|---|
# | **We learned** | An agent is a `while` loop around a tool-calling model. ~30 lines. The message list is the memory. |
# | **We saw** | It chains tool calls autonomously — nobody scripted the order. |
# | **We saw it fail** | Fixed tools cap the ceiling. Ask something outside the menu and it thrashes, or invents. |
#
# ### 🔜 One tool to rule them all
#
# Instead of N specific tools, give it **one general tool: run Python.**
#
# Its capability stops being "whatever verbs I thought of" and becomes "anything expressible in
# code." That's an enormous jump — and it brings a set of problems that will occupy the rest of
# this series.
#
# **→ `04_run_python.ipynb`**

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
# # 02 — The model can ask for things
#
# ### *"The model never executes anything."*
#
# **Previously:** the model invented a statistic for a file it couldn't see, and got the
# arithmetic wrong even when it could. But it wrote *flawless* pandas.
#
# So: let it write code, and we'll run it. The question is **how does it tell us what to run?**
#
# In notebook 01 the code came back as prose in a markdown fence. To use it we'd have to regex it
# out of the text and pray. There's a proper protocol for this, and the whole point of this
# chapter is to show you that it is **not magic** — it's just JSON.

# %%
import sys, os, json
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
from agentlib.llm import llm, METER

# %% [markdown]
# ## 1. Declare a tool. It's a JSON schema and nothing more.
#
# We describe a function to the model: its name, what it does, what arguments it takes. This is a
# *description*, not the function itself. The model never gets the code — only the menu.

# %%
def inspect_csv(path: str) -> str:
    """The actual Python function. WE run this. The model never touches it."""
    df = pd.read_csv(path)
    return (f"shape: {df.shape}\n\n"
            f"dtypes:\n{df.dtypes.to_string()}\n\n"
            f"first 3 rows:\n{df.head(3).to_string()}")


# And here is how we DESCRIBE it to the model. Note: no code. Just a menu item.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "inspect_csv",
        "description": "Look at a CSV file: its shape, column types, and first few rows.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the CSV file."},
            },
            "required": ["path"],
        },
    },
}]

print(json.dumps(TOOLS[0], indent=2))

# %% [markdown]
# ## 2. Ask a question it *cannot* answer without the tool

# %%
messages = [
    {"role": "user", "content":
     "What columns are in ../data/trial.csv, and how many rows does it have?"},
]

reply = llm(messages, tools=TOOLS)

print("content    :", repr(reply.content))
print("tool_calls :", reply.tool_calls)

# %% [markdown]
# ## 3. Look at the raw JSON. This is the whole "magic".
#
# Before we wrap this in any helper, look at exactly what came back over the wire:

# %%
print(json.dumps(reply.raw(), indent=2))

# %% [markdown]
# > ### 💡 Read that again, because this is the entire concept.
# >
# > The model **did not run anything.** It cannot. It has no file system, no Python, no hands.
# >
# > It emitted a piece of JSON that says *"I would like you to call `inspect_csv` with
# > `path='data/trial.csv'`."* That's it. That is 100% of what tool calling is.
# >
# > All the agent frameworks in the world are built on top of this one move. There is no layer
# > underneath it that's doing something cleverer.
#
# The model is not an agent. **The model is a thing that asks.** *We* are the agent — we're the
# ones with hands.

# %% [markdown]
# ## 4. So let's be the hands.
#
# Three steps: read the request, run the real function, hand the result back.

# %%
call = reply.tool_calls[0]
print(f"model wants : {call.name}({call.args})")

# WE execute. This is our Python, in our process.
result = inspect_csv(**call.args)
print(f"\nwe ran it, and got:\n{result[:300]}")

# %% [markdown]
# ## 5. Hand the result back, and ask again
#
# The result goes into the message list as a new message with `role: "tool"`. Then we call the
# model *again* — with the history, which now contains the answer it asked for.
#
# Remember notebook 01: **we own the context.** This is us putting something into it.

# %%
messages.append(reply.raw())                                  # what the model asked for
messages.append({"role": "tool",                              # what we found out
                 "tool_call_id": call.id,
                 "content": result})

final = llm(messages, tools=TOOLS)
print(final.content)

# %% [markdown]
# **That's a working tool-using assistant.** It asked, we answered, it concluded. And it is
# grounded in the actual file — those column names are real, not remembered.
#
# Notice what we did *not* need: no framework, no agent class, no orchestration library. A list,
# a function call, and a second API call.

# %% [markdown]
# ---
# ## 6. What happens when the tool fails?
#
# This matters more than it looks. Let's deliberately point it at a file that doesn't exist.

# %%
messages2 = [{"role": "user", "content": "How many rows are in ../data/patients_2023.csv?"}]
r2 = llm(messages2, tools=TOOLS)
call2 = r2.tool_calls[0]
print("model asks for:", call2.args)

# Run it and let it blow up. We catch the error and send the error TEXT back as the result.
try:
    obs = inspect_csv(**call2.args)
except Exception as e:
    obs = f"ERROR: {type(e).__name__}: {e}"

print("we send back  :", obs[:90])

messages2 += [r2.raw(), {"role": "tool", "tool_call_id": call2.id, "content": obs}]
recovered = llm(messages2, tools=TOOLS)

print("\nmodel's response to the error:")
print(recovered.content or f"(asked for another tool call: {recovered.tool_calls})")

# %% [markdown]
# > ### 💡 Errors are just observations.
# >
# > We didn't crash. We didn't retry. We took the traceback, turned it into a **string**, and
# > handed it to the model exactly like any other tool result.
# >
# > And the model *recovered* — it read the error and adjusted.
# >
# > This is a principle we'll lean on hard. In notebook 04 the agent will hit a real pandas
# > traceback and **debug its own code** from it. In notebook 06, our rejection messages ("your
# > numbers aren't grounded", "you have unresolved findings") are the *same mechanism* — an
# > observation, fed back, that changes the model's next move.
# >
# > The feedback channel is the whole game. Errors are not exceptions to it. They're just more
# > text in the loop.

# %%
print(METER)

# %% [markdown]
# ---
# # Where we are
#
# | | |
# |---|---|
# | **We learned** | A tool call is the model **emitting JSON**. It never executes anything. We do. |
# | **The protocol** | declare tools → model emits `tool_calls` → we run it → append `role: "tool"` → call again |
# | **The principle** | Errors are just observations. Feed them back and the model adapts. |
#
# ### 🔜 But look at what we just did by hand.
#
# We wrote out *one* round trip, manually. Ask → call → answer → done.
#
# Real analysis is not one round trip. It's: look at the file → notice something odd → check that
# → filter → group → compute → sanity-check the number → answer. **Ten steps, and you don't know
# in advance how many, or which.**
#
# We are not going to hand-write ten round trips.
#
# We're going to write a `while` loop. And that — genuinely, unglamorously — is what an agent is.
#
# **→ `03_the_agent_loop.ipynb`**

# Where I used AI

The brief asked me to note this, so here it is in full rather than in a footnote. I used AI
heavily. It would be strange not to, for a take-home about building an AI agent, and stranger to
hide it from a team that builds inference infrastructure for a living.

The useful question is not *whether* — it is **which parts I delegated, which parts I refused to,
and how I checked the difference.**

## What AI wrote

**Most of the code, most of the prose.** I worked in Claude Code throughout. Roughly:

| | |
|---|---|
| `agentlib/` | Written with AI, line by line, with me choosing every mechanism and rejecting a lot. |
| `notebooks/` | Structure and narrative are mine; the prose and plotting code are heavily AI-assisted. |
| `evals/` | The task list, the traps and the tolerances are mine. The harness plumbing is AI. |
| `data/make_*.py` | The data-generating processes are mine (I had to know the ground truth to plant a trap). The pandas is AI. |
| `docs/` | The arguments are mine. The writing is AI-assisted and then heavily cut by me. |

## What AI did **not** decide

This is the part that matters, and it is the part I would defend in the room.

- **The thesis.** "The bottleneck is thread-keeping, not code generation" came from reading the two
  papers, not from a model. It is also the thesis the evaluation later *corrected* — see below.
- **Every mechanism.** The three ledgers, the gated exit, the pre-seeding, the fresh-context
  verifier. Each one exists because I watched a specific run fail without it.
- **Every trap in the benchmark.** A model cannot plant a Simpson's paradox for you and then be
  trusted to grade it; the ground truth has to be something you constructed and can compute.
- **What to cut.** Which is most of the job.

## Where AI actively made things worse, and the eval caught it

Worth stating plainly, because it is the honest answer to "did you just vibe-code this?"

1. **The cache-nonce bug (D24).** AI-written cache, and it silently made my three "independent"
   repeat runs into three replays of a single run. Nothing in the code looked wrong. I found it
   because the cost meter printed `$0.0000` for attempts 2 and 3.
2. **The `LIVE = False` footgun (D25).** The README told a reader to flip a flag that could not be
   flipped. Found by running the instruction instead of trusting it.
3. **Benchmark contamination (D28).** Two of my benchmark questions ended up written verbatim into
   the system prompt as "examples" — along with their answers. Those two tasks scored 100% and 90%.
   Fluent, plausible, and worthless. *And the naive fix made it worse:* replacing them with an
   abstract description dropped the **clean control task from 10/10 to 5/10**. Concrete examples
   teach; abstract ones don't. The bug was never "examples" — it was "examples from my benchmark."
4. **A prompt that overfit without naming anything (D29).** The one rule of domain knowledge in the
   system prompt was phrased in clinical-trial language. It fired on the medical dataset and did
   nothing at all on the e-commerce one. Rewriting it domain-neutral **did not fix the failure** —
   which is the most useful negative result in the project.
5. **A reproducibility promise that held on one computer (D30).** Absolute file paths reached the
   prompt, so the committed cache was keyed to my home directory. *"Runs offline with no API key"*
   was false for every reader.

Every one of those is a mistake a fluent model will happily produce and a fluent reader will
happily approve. **None of them was caught by reading the code. All five were caught by running
something and looking hard at a number that was wrong** — a `$0.0000` cost, a flag that wouldn't
move, a task scoring 100%, a cache only I could hit.

That is the actual lesson of this project, and it is why the evaluation harness — not the agent —
is the thing I would call the deliverable:

> AI makes it very cheap to produce something that looks right.
> It does not make it any cheaper to find out whether it *is* right.
> So spend the budget on the second thing.

## Models used

- **In the product:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (agent) and `openai/gpt-oss-120b`
  (verifier + judge), both on Nebius Token Factory.
- **To build it:** Claude (Claude Code) for code and prose.

# Slides

`presentation.tex` — the talk, as an academic Beamer deck (metropolis theme, XeLaTeX).

```bash
make            # presentation.pdf — 70 frames, 185 overlays, for presenting
make handout    # handout.pdf      — overlays collapsed, for reading
make verify     # fails if any frame's content spills off the slide
```

Output lands in `.out/` (set by a global `~/.latexmkrc`).

## Provenance of the numbers

**Every quantitative claim in the deck comes from `uv run python -m evals.report`**, which reads
`evals/results.jsonl` (4,480 runs) — *not* from the prose in `docs/DESIGN.md`.

Building this deck surfaced **seven stale figures in `DESIGN.md`** — the prose had drifted from the
script. **`DESIGN.md` has been corrected; the two now agree.**

| | was | now (verified) |
|---|---|---|
| `t4_simpson` | 50% / 50% naive | **40% / 55%** (8/20) |
| `s4_simpson_sales` | 45% / 40% naive | **35% / 50%** (7/20) |
| `s4` "the truth at n=20" | 9/20 | **7/20** |
| `b3_ambiguous` | 0% | **5%** (1/20) |
| `s13_ambiguous` | 90% | **85%** (17/20) |
| `no_ledger` | 82% | **81%** |
| `trap:units` *with* guardrails | "0/10 → 10/10" | **0/20 → 17/20** |
| §4.4 caveat 3 | "the −2% I report" | **−6%** — it contradicted its own table |

Correcting `s4` **strengthens** the argument rather than weakening it: on the held-out Simpson's
task the agent is not a coin flip, it is **worse** than one — more likely to land on the naive answer
(50%) than the right one (35%).

### …with one exception, where the *script* is the liar

`report.py` prints a headline cost of **\$3.75**. That is wrong to quote as *spend*: **77% of the
4,480 rows are cache replays that record `$0`**, so it is the *marginal* cost of re-running the grid,
not what it cost to produce. On **billed runs only**, an analysis costs **\$0.004** and the grid cost
**~\$16** — which is what `DESIGN.md` said all along.

> I nearly "corrected" `DESIGN.md` *down* to \$3.75 on the rule *"if the prose and the script
> disagree, the script is right."* That rule is right for every **rate** in this project and wrong
> for a **sum** — **a cache deflates a sum and cannot deflate a proportion.**
>
> Which is §4.4.1's lesson arriving a third time: **the instrument needs checking too, and "trust
> the script" is itself an instrument.**

The detector figures (`t4_simpson` 8/20 → 13/20) come from
`evals/results_confound_detector.jsonl`. Note `s4_simpson_sales` has **not** been re-run with the
detector — the deck says so rather than implying coverage it does not have.

## Structure

| § | |
|---|---|
| 1 | the problem — Simpson's paradox, and an agent that says a working drug doesn't work |
| 2 | what an agent *is* — the LLM as a pure function, tool calling, the ReAct loop |
| 3 | what the papers found — GeneBench-Pro, DrugDiscoveryBench, and three criticisms of them |
| 4 | the design — three ledgers and a gated exit |
| 5 | how would I know it works — the benchmark and its separation guard |
| 6 | the evidence — 4,480 runs, and the ablation that inverted the thesis |
| 7 | the instrument lied — why a 95% CI at n=10 is not a number |
| 8 | the failure the average hid |
| 9 | so I built the detector |
| 10 | limits, and what I'd do next |

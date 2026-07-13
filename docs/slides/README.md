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

Where the script and `DESIGN.md` disagreed while these slides were written, **the script won**, and
the drift is listed on the closing colophon frame:

| | `DESIGN.md` says | the script says |
|---|---|---|
| total spend | \$16.13 | **\$3.75** |
| `t4_simpson` pass rate | 50% | **40%** (8/20) |
| `s4_simpson_sales` pass rate | 45% | **35%** (7/20) |
| `b3_ambiguous` pass rate | 0% | **5%** (1/20) |
| `no_ledger` pass rate | 82% | **81%** |

`DESIGN.md` itself states the tie-break rule — *"if the prose and the script disagree, the script is
right"* — so the deck follows it. **`DESIGN.md` has not been corrected; that is a separate change.**

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

# Slides

`presentation.tex` — the talk, as a **traditional academic Beamer deck** (Madrid theme, Palatino
serif, pdfLaTeX).

```bash
make            # presentation.pdf — 75 frames, with overlays, for presenting
make handout    # handout.pdf      — overlays collapsed, for reading/printing
make verify     # FAILS if any frame's content spills off the slide
make clean
```

Output lands in `.out/` (set by a global `~/.latexmkrc`).

## Structure

The deck is in **two parts**, matching the assignment.

| § | | |
|---|---|---|
| | **Part I — the papers** | |
| 1 | Context: what the two papers are addressing | what an autonomous analysis agent *is*; why the bottleneck moved; ``research taste''; the *decision point* |
| 2 | The two benchmarks, and what they found | GeneBench-Pro (129 simulated problems, 28.7%) and DrugDiscoveryBench (82 expert tasks, 51.6%) — and why those two numbers must **not** be compared |
| 3 | **How these agents actually fail** | the five failure intuitions, each with a worked case from the papers, plus DDB's taxonomy |
| 4 | What is convincing — and what is not | the design choices worth copying; then the ones that do not survive contact |
| 5 | The key limitation neither paper states | **both diagnoses are established only at the top of the capability range** |
| | **Part II — the design** | |
| 6 | The design: an LLM, a loop, and the gates on top | the base; then each gate, and the failure from Part I that earned it |
| 7 | How would I know it works? | the benchmark, the separation guard, the ablations, the instrument check |
| 8 | What happened when I ran it | 4,480 runs — and the ablation that inverted the thesis |
| 9 | Limits, and the dashboard | |

Section 3 is the heart of Part I; section 6 is the heart of Part II. Sections 8–9 can be cut for
time — the deck stands without them, and the backup frames carry the detail.

## Provenance of the numbers

**Claims about the papers** were checked against the PDFs in `papers/` by a full re-read. That
re-read **caught ten errors in the earlier write-up** — four wrong figures and six quotes that had
drifted from the source. All are corrected here *and* in `docs/PART1_REVIEW.md`:

| | was | now (checked against the PDF) |
|---|---|---|
| GBP attempts per problem | "10 (5 for two of the heaviest configurations)" | **10; 5 for the GPT Pro (Extended) *and* Claude Opus rows** — 11 of 60 |
| DDB evaluation grid | "12 models × 6 harnesses" (implies 72 cells) | **29 settings**, unevenly distributed |
| DDB oracle-union denominator | "12 × 6 × 3 = 216 runs/task" | **29 settings × 3 trials ≈ 87 runs/task** |
| DDB judge validation | "found perfect agreement (κ = 1.0)" | "found perfect agreement **with respect to binary labels** (κ = 1.0)" |
| DDB hint result | "pass 80 out of 82" | "pass 80 out of 82 **and near-pass 1**" |
| GBP notice/act quote | one quote, spliced from two pages | **two separately-attributed quotes** (p.3 Results; p.14 Discussion) |
| GBP "the interval on half their benchmark is [0,0]" | an inference GBP never publishes | **Supp. Table 2 literally prints `0.0±0.0%` as a 95% CI** — the unanswerable version |

**Claims about my system** come from `uv run python -m evals.report`, which reads
`evals/results.jsonl` (4,480 runs) — *not* from the prose in `docs/DESIGN.md`.

### …with one exception, where the *script* is the liar

`report.py` prints a headline cost of **\$3.75**. That is wrong to quote as *spend*: **77% of the
4,480 rows are cache replays that record `$0`**, so it is the *marginal* cost of re-running the
grid, not what it cost to produce. On **billed runs only**, an analysis costs **\$0.004** and the
grid cost **~\$16** — which is what `DESIGN.md` said all along.

> I nearly "corrected" `DESIGN.md` *down* to \$3.75 on the rule *"if the prose and the script
> disagree, the script is right."* That rule is right for every **rate** in this project and wrong
> for a **sum** — **a cache deflates a sum and cannot deflate a proportion.**

## The live demo

The dashboard (`uv run streamlit run app.py`, http://localhost:8501) runs against a **self-hosted
vLLM on Modal** — see `infra/modal_vllm.py` and the header of `.env`. The free tiers were measured
and are structurally unusable for a live demo:

- **Groq** — 6,000 tokens/**minute**; one agent request is ~6,300. The floor is above the ceiling.
- **Gemini** — 20 requests/**day** per model on the free tier; one agent run is ~11. Verified
  2026-07-14: the agent died at step 8 with a 429.

The demo model is **Qwen3-4B** — 7× smaller than the Qwen3-30B-A3B the evaluation was measured on,
and the endpoint serves only one model, so the verifier is **not** cross-family there. The dashboard
says all of this itself. Numbers produced live are **not** comparable to `evals/results.jsonl`.

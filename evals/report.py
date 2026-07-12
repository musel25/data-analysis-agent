"""Every number that appears in the README, DESIGN.md and the notebooks — printed from the results
file, in one command:

    uv run python -m evals.report

This exists because of a bug that was not in the code.

I 6x'd the evaluation (15 tasks x 3 runs -> 28 x 10) and updated the headline sections. I did not
update the other nineteen places a number was written down. For a while this repo simultaneously
claimed 360 runs and 2,240 runs; 15 tasks and 28 tasks; that I could not afford bootstrap CIs, on
the page reporting bootstrap CIs. Every one of those numbers had been true once.

    A number copied into prose is a number that will go stale, and it will go stale silently,
    and it will go stale in the direction that flatters you.

So: no number gets hand-typed into a document. It gets printed here and pasted. If a doc and this
script disagree, the script is right and the doc is a bug.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .stats import ablation_table, hierarchical_bootstrap, paired_delta, replication

RESULTS = Path(__file__).parent / "results.jsonl"
DOMAIN = {"p": "penguins", "t": "trial", "b": "trial", "h": "trial", "s": "sales"}


def load() -> pd.DataFrame:
    df = pd.DataFrame([json.loads(l) for l in RESULTS.open()])
    df["domain"] = df.task_id.str[0].map(DOMAIN)
    df["is_trap"] = df.category.str.startswith("trap")
    return df


def main() -> None:
    df = load()
    full = df[df.config == "full"]
    pct = lambda x: f"{x * 100:.0f}%"

    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(f"  {len(df):,} runs · {df.task_id.nunique()} tasks · {df.domain.nunique()} domains "
          f"· {df.config.nunique()} configs · ${df.cost_usd.sum():.2f}")
    print(f"  {len(df) // df.config.nunique() // df.task_id.nunique()} runs per (task, config)"
          f"  ·  crashes: {(df.stopped == 'crash').sum()}")
    m, lo, hi = hierarchical_bootstrap(full)
    print(f"  full agent: {pct(m)}  95% CI [{pct(lo)}, {pct(hi)}]")
    print(f"  mean cost/run ${full.cost_usd.mean():.4f} · {full.steps.mean():.1f} steps")

    print("\n" + "=" * 78)
    print("ABLATIONS  (paired hierarchical bootstrap vs the full agent, 10k resamples)")
    print("=" * 78)
    t = ablation_table(df)
    print(f"  {'remove this':<18} {'pass':>6} {'Δ vs full':>10} {'95% CI':>18}   verdict")
    for cfg, r in t.iterrows():
        ci = f"[{r.lo95 * 100:+.0f}%, {r.hi95 * 100:+.0f}%]"
        print(f"  {cfg:<18} {pct(r.pass_rate):>6} {r.delta_vs_full * 100:>+9.0f}% {ci:>18}"
              f"   {r.verdict}")

    print("\n" + "=" * 78)
    print("REPLICATION  (the error bar I actually trust — D31)")
    print("=" * 78)
    print("  The same experiment, run twice. If a verdict flips between these two columns,")
    print("  it is not a finding — it is weather.\n")
    r = replication(df)
    print(f"  {'remove this':<18} {'run A':>7} {'run B':>7} {'spread':>8} {'pooled':>8}"
          f" {'95% CI':>16}")
    for cfg, row in r.iterrows():
        ci = f"[{row.lo95*100:+.0f}%, {row.hi95*100:+.0f}%]"
        flag = "" if row.stable else "   <-- UNSTABLE"
        print(f"  {cfg:<18} {row.delta_first_half*100:+6.0f}% {row.delta_second_half*100:+6.0f}%"
              f" {row.spread*100:7.0f}pt {row.delta_pooled*100:+7.0f}% {ci:>16}{flag}")

    print("\n" + "=" * 78)
    print("BY DOMAIN  (full agent)")
    print("=" * 78)
    for d in ["penguins", "trial", "sales"]:
        g = full[full.domain == d]
        if g.empty:
            continue
        m, lo, hi = hierarchical_bootstrap(g)
        tag = {"penguins": "clean data", "trial": "DESIGNED AGAINST",
               "sales": "HELD-OUT DOMAIN"}[d]
        print(f"  {d:<10} {pct(m):>5}  [{pct(lo)}, {pct(hi)}]   n={len(g):<4} {tag}")
    ho = full[full.holdout]
    if not ho.empty:
        m, lo, hi = hierarchical_bootstrap(ho)
        print(f"  {'held-out tasks':<10} {pct(m):>5}  [{pct(lo)}, {pct(hi)}]   n={len(ho)}")

    print("\n" + "=" * 78)
    print("TRAP TASKS ONLY  (the planted decision points — where a guardrail can bite)")
    print("=" * 78)
    traps = df[df.is_trap]
    print(f"  {'':<16} {'pass':>6} {'fell for the DOCUMENTED naive answer':>38}")
    for c in ["full", "no_guardrails"]:
        g = traps[traps.config == c]
        print(f"  {c:<16} {pct(g.passed.mean()):>6} {pct(g.wrong_attractor.mean()):>38}")
    a = traps[traps.config == "full"].wrong_attractor.mean()
    b = traps[traps.config == "no_guardrails"].wrong_attractor.mean()
    print(f"\n  wrong-attractor reduction: {b / a:.1f}x   (all three domains, like for like)")
    print("  ^ quote THIS number. Do not mix an all-domain arm with a trial-only arm.")

    print("\n" + "=" * 78)
    print("PER-TASK  (full agent — read this, not the headline; the average hides the failures)")
    print("=" * 78)
    per = (full.groupby(["task_id", "category"])
           .agg(pass_rate=("passed", "mean"), naive=("wrong_attractor", "mean"))
           .reset_index().sort_values("pass_rate"))
    for _, r in per.iterrows():
        flag = "  <-- FAILING" if r.pass_rate < 0.5 else ""
        print(f"  {r.task_id:<24} {r.category:<20} {r.pass_rate * 100:>3.0f}%"
              f"   naive {r.naive * 100:>3.0f}%{flag}")


if __name__ == "__main__":
    main()

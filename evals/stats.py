"""Confidence intervals, because a point estimate is how you fool yourself.

The first version of this benchmark reported `full 87%` vs `no_ledger 87%` and I nearly concluded
"the ledger does nothing." At n=45 those two numbers have overlapping CIs roughly 20 points wide.
The honest statement was never "the ledger does nothing" — it was **"I cannot tell."**

GeneBench-Pro bootstraps hierarchically ("resampling problems and repeated runs within each
sampled problem", 20,000 resamples). We do the same, for the same reason: the runs are *not*
independent — several runs share a task, and tasks differ enormously in difficulty. Resampling
individual runs would treat 10 runs of an easy task as 10 independent successes and give a CI far
too tight.

So: resample TASKS with replacement, then resample RUNS within each sampled task.

    ⚠️  AND THEN I FOOLED MYSELF FROM INSIDE THE CONFIDENCE INTERVAL ANYWAY. (D31)

I ran this identical benchmark twice, either side of a change that provably altered nothing (same
error rate, same step count, same budget-exhaustion rate). The Findings Ledger came back:

    run A:   -1.8%   95% CI [ -8.9%, +4.6%]     ->  "no detectable effect"
    run B:  -11.1%   95% CI [-18.6%, -4.3%]     ->  "SIGNIFICANT"

Same code, same tasks, opposite verdicts, intervals that barely overlap. At least one of them is
wrong and I cannot tell which — so as stated, BOTH are worthless.

The bootstrap is not miscoded. It is being asked a question it cannot answer. With 10 runs per cell
it resamples from the ten outcomes actually observed, so a cell that came back 1/10 has a bootstrap
distribution centred near 10% and CANNOT REACH the 60% the next run produced. The empirical
distribution is degenerate near p=0 and p=1 — which is exactly where the hard tasks live.

    A bootstrap gives you the sampling variance of the data you HAVE. It knows nothing about the
    data you did not collect. At n=10, near the extremes, that gap is enormous — and the interval
    it hands back is too narrow in a way that FEELS rigorous.

Mitigated (not solved) by doubling to 20 runs per cell. Properly solved by a hierarchical
beta-binomial that shrinks the extreme cells instead of pretending they are certain — or, cheaper
and more honest, by running the whole benchmark twice every time and reporting the SPREAD BETWEEN
RUNS as the error bar. That number cannot lie to you: it is a measurement, not a model.

    An error bar is not automatically an honest number. It is honest only if the error bar is —
    and mine was computed from too little data to know.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESAMPLES = 10_000
SEED = 0


def _rng() -> np.random.Generator:
    """A FRESH, identically-seeded generator for every call.

    The first version held one module-level `default_rng(0)` and drew from it across every call. So
    the CI you got for a config depended on how many bootstraps had run before it — and re-running
    `evals/report.py` moved the published numbers by a point or two.

    Which is a small bug with a fatal symptom: an interviewer runs the script, gets different numbers
    from the README, and now *nothing* in the repo is trustworthy. A reproducibility claim that only
    holds on the first invocation is not a reproducibility claim.

    (And it is D31 wearing a smaller hat: the instrument that measures the noise was itself noisy.)
    """
    return np.random.default_rng(SEED)


def hierarchical_bootstrap(df: pd.DataFrame, col: str = "passed",
                           n: int = RESAMPLES) -> tuple[float, float, float]:
    """Return (mean, lo95, hi95) for a binary column, resampling tasks then runs within tasks."""
    if df.empty:
        return (np.nan, np.nan, np.nan)

    groups = [g[col].to_numpy() for _, g in df.groupby("task_id")]
    k = len(groups)
    if k == 0:
        return (np.nan, np.nan, np.nan)

    rng = _rng()
    means = np.empty(n)
    for i in range(n):
        picked = rng.integers(0, k, k)                    # resample TASKS
        vals = [g[rng.integers(0, len(g), len(g))].mean() # resample RUNS within each task
                for g in (groups[j] for j in picked)]
        means[i] = np.mean(vals)

    return float(df[col].mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_delta(df: pd.DataFrame, config_a: str, config_b: str,
                 col: str = "passed", n: int = RESAMPLES) -> tuple[float, float, float, bool]:
    """Bootstrap the DIFFERENCE between two configs, paired by task.

    Paired, because the two configs ran the *same* tasks — so the task-difficulty variance that
    dominates the absolute CIs cancels out, and we get far more power on the comparison than on
    either number alone. This is the test that actually answers "does this mechanism matter?"

    Returns (delta, lo95, hi95, significant) where `significant` means the 95% CI excludes zero.
    """
    a = df[df.config == config_a]
    b = df[df.config == config_b]
    tasks = sorted(set(a.task_id) & set(b.task_id))
    if not tasks:
        return (np.nan, np.nan, np.nan, False)

    ga = {t: a[a.task_id == t][col].to_numpy() for t in tasks}
    gb = {t: b[b.task_id == t][col].to_numpy() for t in tasks}
    k = len(tasks)

    rng = _rng()
    deltas = np.empty(n)
    for i in range(n):
        picked = rng.integers(0, k, k)                     # resample TASKS (the pairing unit)
        d = []
        for j in picked:
            t = tasks[j]
            va, vb = ga[t], gb[t]
            d.append(va[rng.integers(0, len(va), len(va))].mean()
                     - vb[rng.integers(0, len(vb), len(vb))].mean())
        deltas[i] = np.mean(d)

    obs = float(np.mean([ga[t].mean() - gb[t].mean() for t in tasks]))
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return obs, lo, hi, (lo > 0 or hi < 0)


def summary(df: pd.DataFrame, col: str = "passed") -> pd.DataFrame:
    """Per-config mean + 95% CI."""
    rows = []
    for cfg, g in df.groupby("config"):
        m, lo, hi = hierarchical_bootstrap(g, col)
        rows.append({"config": cfg, "n": len(g), "mean": m, "lo95": lo, "hi95": hi,
                     "ci_width": hi - lo})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).set_index("config")


def ablation_table(df: pd.DataFrame, baseline: str = "full", col: str = "passed") -> pd.DataFrame:
    """The table that actually answers the question: for each ablation, is the paired difference
    vs the full agent distinguishable from zero?"""
    rows = []
    for cfg in sorted(df.config.unique()):
        if cfg == baseline:
            continue
        d, lo, hi, sig = paired_delta(df, cfg, baseline, col)
        rows.append({
            "config": cfg,
            "pass_rate": df[df.config == cfg][col].mean(),
            "delta_vs_full": d,
            "lo95": lo, "hi95": hi,
            "verdict": "HURTS" if (sig and d < 0) else ("HELPS" if (sig and d > 0)
                                                        else "no detectable effect"),
        })
    return pd.DataFrame(rows).sort_values("delta_vs_full").set_index("config")


def replication(df: pd.DataFrame, baseline: str = "full", col: str = "passed") -> pd.DataFrame:
    """The error bar I actually trust: split the runs in half and measure the SAME thing twice.

    D31. The bootstrap CI told me the Findings Ledger was "no detectable effect" on one run of this
    benchmark and "SIGNIFICANT" on the next, with intervals that barely overlapped. A modelled
    error bar is only as honest as the data it is modelled from, and at 10 runs per cell mine was
    not honest.

    So: take the 20 runs per cell, split them into the first 10 and the last 10 — two independent
    replicates of the whole experiment — and compute the ablation on each. The gap between the two
    columns is not a model of the uncertainty. It IS the uncertainty, observed.

    If a mechanism's verdict flips between these two columns, it is not a finding. It is weather.
    """
    half = df.attempt.max() // 2 + 1
    a, b = df[df.attempt < half], df[df.attempt >= half]
    rows = []
    for cfg in sorted(df.config.unique()):
        if cfg == baseline:
            continue
        da, *_ = paired_delta(a, cfg, baseline, col)
        db, *_ = paired_delta(b, cfg, baseline, col)
        dd, lo, hi, sig = paired_delta(df, cfg, baseline, col)
        rows.append({
            "config": cfg,
            "delta_first_half": da,
            "delta_second_half": db,
            "spread": abs(da - db),          # <- the honest error bar
            "delta_pooled": dd,
            "lo95": lo, "hi95": hi,
            "stable": abs(da - db) < abs(dd) or (da < 0) == (db < 0),
        })
    return pd.DataFrame(rows).sort_values("delta_pooled").set_index("config")

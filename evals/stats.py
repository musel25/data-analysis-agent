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
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESAMPLES = 10_000
RNG = np.random.default_rng(0)


def hierarchical_bootstrap(df: pd.DataFrame, col: str = "passed",
                           n: int = RESAMPLES) -> tuple[float, float, float]:
    """Return (mean, lo95, hi95) for a binary column, resampling tasks then runs within tasks."""
    if df.empty:
        return (np.nan, np.nan, np.nan)

    groups = [g[col].to_numpy() for _, g in df.groupby("task_id")]
    k = len(groups)
    if k == 0:
        return (np.nan, np.nan, np.nan)

    means = np.empty(n)
    for i in range(n):
        picked = RNG.integers(0, k, k)                    # resample TASKS
        vals = [g[RNG.integers(0, len(g), len(g))].mean() # resample RUNS within each task
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

    deltas = np.empty(n)
    for i in range(n):
        picked = RNG.integers(0, k, k)                     # resample TASKS (the pairing unit)
        d = []
        for j in picked:
            t = tasks[j]
            va, vb = ga[t], gb[t]
            d.append(va[RNG.integers(0, len(va), len(va))].mean()
                     - vb[RNG.integers(0, len(vb), len(vb))].mean())
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

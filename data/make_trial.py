"""Build trial.csv — a synthetic clinical dataset with *planted decision points*.

WHY SYNTHETIC?
A benchmark needs ground truth you can *compute*, not ground truth you assume. GeneBench-Pro
simulates for exactly this reason: "constructively simulated problems where the full causal
structure is known." Simulation is also the only way to *plant* a decision point and then prove
that the plausible-but-wrong path gives a different answer.

THE PLANTED DECISION POINTS
  1. -999 sentinels in biomarker_baseline  -> naive mean is dragged far below the truth
  2. re-tested patients appear twice       -> naive row-count over-counts; dedupe on sample_seq
  3. assay batch B reports in ug/L (10x)   -> naive pooled mean is inflated. The unit note lives
                                              in data_dictionary.md, so an agent that never reads
                                              the provided file cannot get this right.
  4. Simpson's paradox (arm x severity)    -> treatment looks WORSE overall but is BETTER in every
                                              stratum. Confounded by indication: sicker patients
                                              were likelier to get the drug.
  5. only 3 sites exist                    -> a question about "the four sites" has a false premise

GeneBench-Pro's design principle #3 is "clear numerical separation from incorrect answers"; its
stated violation is "wrong analyses can be graded as correct." So `verify()` below *asserts* that
every naive path lands far from the correct one. If that assertion ever fails, the benchmark is
measuring nothing and the build refuses to write the file.

Run:  uv run python data/make_trial.py
"""

import numpy as np
import pandas as pd

N = 800

# --- the data-generating process, stated explicitly -----------------------------------------
P_TREAT = {"mild": 0.15, "moderate": 0.50, "severe": 0.85}   # confounding by indication
P_RESP = {"mild": 0.80, "moderate": 0.55, "severe": 0.25}    # sicker -> worse outcomes
TREATMENT_EFFECT = 0.15                                       # the drug helps, +15pp everywhere
BASELINE_MEAN = {"mild": 42.0, "moderate": 55.0, "severe": 68.0}

SENTINEL = -999.0
N_RETESTS = 48
BATCH_B_SCALE = 10.0   # batch B instrument reports ug/L instead of ng/mL


def build(seed: int) -> pd.DataFrame:
    """Generate the trial data, then corrupt it the way real data is corrupted."""
    rng = np.random.default_rng(seed)

    severity = rng.choice(["mild", "moderate", "severe"], size=N, p=[0.50, 0.30, 0.20])
    arm = np.array(["treatment" if rng.random() < P_TREAT[s] else "control" for s in severity])

    # the drug adds TREATMENT_EFFECT within every severity stratum
    p_resp = np.array([
        min(0.99, P_RESP[s] + (TREATMENT_EFFECT if a == "treatment" else 0.0))
        for s, a in zip(severity, arm)
    ])
    responded = (rng.random(N) < p_resp).astype(int)

    biomarker_baseline = np.array([rng.normal(BASELINE_MEAN[s], 8.0) for s in severity]).round(1)
    improvement = np.where(responded == 1, rng.normal(14, 4, N), rng.normal(2, 4, N))
    biomarker_final = (biomarker_baseline - improvement).round(1)

    df = pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(1, N + 1)],
        "sample_seq": 1,
        "site": rng.choice(["site_1", "site_2", "site_3"], size=N, p=[0.40, 0.35, 0.25]),
        "arm": arm,
        "severity": severity,
        "age": rng.integers(28, 82, N),
        "sex": rng.choice(["F", "M"], size=N, p=[0.5, 0.5]),   # NO effect on response, by design
        "assay_batch": rng.choice(["A", "B"], size=N, p=[0.70, 0.30]),
        "biomarker_baseline": biomarker_baseline,
        "biomarker_final": biomarker_final,
        "responded": responded,
    })

    # ---- TRAP 1: 11% of baseline assays failed QC and were written as -999 ------------------
    idx = rng.choice(N, size=int(0.11 * N), replace=False)
    df.loc[idx, "biomarker_baseline"] = SENTINEL

    # ---- TRAP 3: batch B ran on an instrument reporting a 10x scale -------------------------
    is_b = df["assay_batch"] == "B"
    df.loc[is_b, "biomarker_final"] = (df.loc[is_b, "biomarker_final"] * BATCH_B_SCALE).round(1)

    # ---- TRAP 2: some patients were re-tested; their row appears twice ----------------------
    # Re-testing is NOT random: sicker patients get re-assayed more often. That is realistic,
    # and it is what makes the trap bite. If retests were a random sample, deduplicating would
    # barely change any proportion, and a task like "what fraction of patients are severe?"
    # could not tell a careful analyst from a careless one. (I learned this the hard way — the
    # separation guard in evals/tasks.py rejected exactly that task.)
    retest_weight = df["severity"].map({"mild": 1.0, "moderate": 3.0, "severe": 12.0}).to_numpy(copy=True)
    retest_weight = retest_weight / retest_weight.sum()
    retest_idx = rng.choice(len(df), size=N_RETESTS, replace=False, p=retest_weight)
    retests = df.iloc[retest_idx].copy()
    retests["sample_seq"] = 2
    retests["biomarker_final"] = (retests["biomarker_final"] + rng.normal(0, 1.5, len(retests))).round(1)

    return (pd.concat([df, retests], ignore_index=True)
              .sort_values(["patient_id", "sample_seq"])
              .reset_index(drop=True))


# ============================================================================================
# The canonical cleaning steps. The grader uses these; the agent must discover them.
# ============================================================================================

def dedupe(d):
    """Canonical: the latest sample per patient supersedes earlier ones."""
    return d.sort_values("sample_seq").groupby("patient_id", as_index=False).last()


def rescale_batch(d):
    """Canonical: batch B reports in ug/L; divide by 10 to convert to ng/mL."""
    d = d.copy()
    d.loc[d["assay_batch"] == "B", "biomarker_final"] /= BATCH_B_SCALE
    return d


def clean(d):
    return rescale_batch(dedupe(d))


def verify(raw: pd.DataFrame) -> dict:
    """Assert that every naive path lands far from the correct one. Returns the trap stats."""
    c = clean(raw)
    c_bl = c[c["biomarker_baseline"] != SENTINEL]

    # TRAP 1 -- sentinels
    t1_naive = raw["biomarker_baseline"].mean()
    t1_correct = c_bl["biomarker_baseline"].mean()

    # TRAP 2 -- duplicates
    t2_naive, t2_correct = len(raw), raw["patient_id"].nunique()

    # TRAP 3 -- batch scale
    t3_naive = dedupe(raw)["biomarker_final"].mean()
    t3_correct = c["biomarker_final"].mean()

    # TRAP 4 -- Simpson's paradox
    marginal = c.groupby("arm")["responded"].mean()
    t4_naive = marginal["treatment"] - marginal["control"]
    strat = c.groupby(["severity", "arm"])["responded"].mean().unstack()
    per_stratum = strat["treatment"] - strat["control"]
    weights = c["severity"].value_counts(normalize=True)
    t4_correct = float((per_stratum * weights).sum())

    # FALSE PREMISE -- sex has no effect by construction; the gap must stay small
    sex_resp = c.groupby("sex")["responded"].mean()
    sex_gap = abs(sex_resp["F"] - sex_resp["M"])

    assert abs(t1_naive - t1_correct) > 40, "sentinel trap too weak"
    assert t2_naive - t2_correct == N_RETESTS, "duplicate trap wrong size"
    assert abs(t3_naive - t3_correct) > 60, "batch trap too weak"
    assert t4_naive < -0.05, "marginal effect must favour CONTROL"
    assert t4_correct > 0.12, "adjusted effect must favour TREATMENT"
    assert (per_stratum > 0.08).all(), "treatment must win in EVERY stratum (true reversal)"
    assert sex_gap < 0.025, "sex gap must be ~zero for the false-premise task to be fair"
    assert not strat.isna().any().any(), "every severity x arm cell must be populated"
    assert sorted(c["site"].unique()) == ["site_1", "site_2", "site_3"], "must be exactly 3 sites"

    return {
        "t1_sentinel": (t1_naive, t1_correct),
        "t2_duplicates": (t2_naive, t2_correct),
        "t3_batch": (t3_naive, t3_correct),
        "t4_simpson": (t4_naive, t4_correct),
        "per_stratum": per_stratum,
        "strat_table": strat,
        "sex_gap": sex_gap,
    }


def find_seed(start=20260701, tries=4000) -> int:
    """The traps must all hold at once. Search the seed space using the real generator."""
    for s in range(start, start + tries):
        try:
            verify(build(s))
            return s
        except AssertionError:
            continue
    raise RuntimeError("no seed satisfies all traps — loosen the DGP")


SEED = 20260701   # found by find_seed(); all traps re-verified on every run

if __name__ == "__main__":
    raw = build(SEED)
    v = verify(raw)   # refuses to write the file if any trap is weak
    raw.to_csv("data/trial.csv", index=False)

    print("=" * 78)
    print(f"wrote data/trial.csv  —  {len(raw)} rows, {raw['patient_id'].nunique()} unique patients")
    print(f"seed {SEED}; all traps verified")
    print("=" * 78)

    n, c = v["t1_sentinel"]
    print(f"\nTRAP 1  -999 sentinels in biomarker_baseline")
    print(f"   naive (keeps -999)  : {n:8.2f}")
    print(f"   correct (excludes)  : {c:8.2f}      separation: {abs(n - c):.1f}")

    n, c = v["t2_duplicates"]
    print(f"\nTRAP 2  re-tested patients appear twice")
    print(f"   naive (row count)   : {n:8d}")
    print(f"   correct (patients)  : {c:8d}      separation: {n - c}")

    n, c = v["t3_batch"]
    print(f"\nTRAP 3  assay batch B on a 10x scale (unit note is in data_dictionary.md)")
    print(f"   naive (pooled)      : {n:8.2f}")
    print(f"   correct (rescaled)  : {c:8.2f}      separation: {abs(n - c):.1f}")

    n, c = v["t4_simpson"]
    print(f"\nTRAP 4  Simpson's paradox — arm confounded by severity")
    print(f"   naive marginal      : {n:+8.3f}      <- treatment looks WORSE")
    print(f"   correct adjusted    : {c:+8.3f}      <- treatment is BETTER")
    print(f"\n   response rate, severity x arm:")
    print("   " + v["strat_table"].round(3).to_string().replace("\n", "\n   "))
    print(f"\n   treatment effect within each stratum:")
    print("   " + v["per_stratum"].round(3).to_string().replace("\n", "\n   "))
    print(f"\n   *** SIGN REVERSAL: {n:+.3f} -> {c:+.3f} — positive in all 3 strata ***")

    print(f"\nFALSE PREMISE  sex has no effect by construction (gap {v['sex_gap']:.3f}),")
    print(f"               and only 3 sites exist — so a question about 'the four sites'")
    print(f"               or 'why women respond better' must be refused, not answered.")
    print("\n" + "=" * 78)
    print("every naive path lands far from the correct one. the benchmark measures something.")
    print("=" * 78)

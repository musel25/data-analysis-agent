"""Build sales.csv — a SECOND dataset, in a completely different domain.

WHY THIS EXISTS, and it is the most important thing in the evaluation.

Every trap in `trial.csv` is one I planted, and every guardrail was designed while looking at it.
An agent passing `trial.csv` proves that my guardrails work on the failures I already knew about.
That is a much weaker claim than it looks.

`sales.csv` is **a held-out DOMAIN, not just held-out tasks.** E-commerce, not medicine. Different
column names, different semantics, and traps that are the *same species* but a different animal:

  1. `revenue` stored as TEXT with comma thousands separators   -> .mean() silently fails or lies
  2. internal test orders with revenue = 999999.99              -> a few rows destroy any average
  3. refunded orders still present (status='refunded')          -> counting them inflates revenue
  4. Simpson's paradox: channel x customer_segment              -> the SAME failure, new clothes
  5. `customer_age` uses -1 as "not supplied"                   -> a sentinel, in a new costume

Note trap 1: the deterministic briefing has *always* had a detector for numeric-looking strings,
and until now no task in the benchmark exercised it. Writing a new domain forced me to find that.

If the guardrails only worked on the dataset they were designed against, this file is where that
shows up.

Run:  uv run python data/make_sales.py
"""

import numpy as np
import pandas as pd

N = 900

# --- the data-generating process -----------------------------------------------------------
# Confounding by targeting: the "email" campaign was aimed at loyal high-value customers, who
# convert well no matter what. So email looks great overall while being *worse* within every
# segment. Exactly Simpson's paradox — and nothing about it is clinical.
P_EMAIL = {"new": 0.15, "returning": 0.45, "loyal": 0.85}      # who got targeted
P_CONV = {"new": 0.20, "returning": 0.45, "loyal": 0.75}       # baseline conversion by segment
EMAIL_EFFECT = -0.12   # the email campaign is actually WORSE. it just looks better.

SENTINEL_AGE = -1
TEST_ORDER_REVENUE = 999999.99
N_TEST_ORDERS = 6


def build(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    segment = rng.choice(["new", "returning", "loyal"], size=N, p=[0.45, 0.35, 0.20])
    channel = np.array(["email" if rng.random() < P_EMAIL[s] else "paid_search" for s in segment])

    p_conv = np.array([
        min(0.98, max(0.02, P_CONV[s] + (EMAIL_EFFECT if c == "email" else 0.0)))
        for s, c in zip(segment, channel)
    ])
    converted = (rng.random(N) < p_conv).astype(int)

    base_rev = {"new": 45.0, "returning": 80.0, "loyal": 140.0}
    revenue = np.array([max(5.0, rng.normal(base_rev[s], 25.0)) for s in segment]).round(2)
    revenue = np.where(converted == 1, revenue, 0.0)

    age = rng.integers(18, 75, N).astype(float)
    status = np.where(rng.random(N) < 0.08, "refunded", "completed")

    df = pd.DataFrame({
        "order_id": [f"ORD{i:05d}" for i in range(1, N + 1)],
        "channel": channel,
        "customer_segment": segment,
        "customer_age": age,
        "status": status,
        "revenue": revenue,
        "converted": converted,
    })

    # ---- TRAP 5: -1 means "age not supplied" ----------------------------------------------
    miss = rng.choice(N, size=int(0.12 * N), replace=False)
    df.loc[miss, "customer_age"] = SENTINEL_AGE

    # ---- TRAP 2: internal QA orders left in the export -------------------------------------
    test_idx = rng.choice(N, size=N_TEST_ORDERS, replace=False)
    df.loc[test_idx, "revenue"] = TEST_ORDER_REVENUE
    df.loc[test_idx, "converted"] = 1
    df.loc[test_idx, "status"] = "completed"

    df["customer_age"] = df["customer_age"].astype(int)

    # ---- TRAP 1: revenue exported as TEXT with thousands separators -------------------------
    # (this is what actually happens when someone opens the export in Excel and saves it)
    df["revenue"] = df["revenue"].map(lambda v: f"{v:,.2f}")

    return df


# ============================================================================================
# The canonical cleaning. The grader uses it; the agent must discover it.
# ============================================================================================

def parse_revenue(d):
    d = d.copy()
    d["revenue"] = d["revenue"].astype(str).str.replace(",", "", regex=False).astype(float)
    return d


def clean(d):
    """Parse the text revenue, drop the internal test orders, drop refunds."""
    d = parse_revenue(d)
    d = d[d["revenue"] < 900_000]              # internal QA rows
    d = d[d["status"] == "completed"]          # a refund is not revenue
    return d


def verify(raw: pd.DataFrame) -> dict:
    c = clean(raw)

    # TRAP 1+2+3 — mean revenue per completed order
    naive_rev = parse_revenue(raw)["revenue"].mean()     # keeps test orders AND refunds
    correct_rev = c["revenue"].mean()

    # TRAP 4 — Simpson's paradox on conversion, channel x segment
    marg = raw.groupby("channel")["converted"].mean()
    naive_lift = marg["email"] - marg["paid_search"]
    strat = raw.groupby(["customer_segment", "channel"])["converted"].mean().unstack()
    per_seg = strat["email"] - strat["paid_search"]
    w = raw["customer_segment"].value_counts(normalize=True)
    adj_lift = float((per_seg * w).sum())

    # TRAP 5 — sentinel ages
    naive_age = raw["customer_age"].mean()
    correct_age = raw.loc[raw["customer_age"] != SENTINEL_AGE, "customer_age"].mean()

    # pandas 3 reports this as `str`, pandas 2 as `object` — either way, not numeric.
    assert not pd.api.types.is_numeric_dtype(raw["revenue"]), \
        "revenue must be TEXT for the trap to exist"
    assert abs(naive_rev - correct_rev) > 3000, "test-order trap too weak"
    assert naive_lift > 0.05, "email must LOOK better overall"
    assert adj_lift < -0.05, "email must actually BE worse"
    assert (per_seg < -0.02).all(), "email must lose in EVERY segment (a true reversal)"
    assert abs(naive_age - correct_age) > 5, "age sentinel trap too weak"
    assert not strat.isna().any().any(), "every segment x channel cell must be populated"

    return {"revenue": (naive_rev, correct_rev), "lift": (naive_lift, adj_lift),
            "age": (naive_age, correct_age), "per_seg": per_seg, "strat": strat}


def find_seed(start=1, tries=4000) -> int:
    for s in range(start, start + tries):
        try:
            verify(build(s))
            return s
        except AssertionError:
            continue
    raise RuntimeError("no seed satisfies all traps")


SEED = 7   # replaced below by find_seed() if needed; verified on every run

if __name__ == "__main__":
    try:
        raw = build(SEED)
        v = verify(raw)
    except AssertionError:
        SEED = find_seed()
        raw = build(SEED)
        v = verify(raw)

    raw.to_csv("data/sales.csv", index=False)

    print("=" * 78)
    print(f"wrote data/sales.csv — {len(raw)} rows, seed {SEED}. all traps verified.")
    print("=" * 78)

    n, c = v["revenue"]
    print(f"\nTRAP 1+2+3  revenue is TEXT ('1,234.56'), 6 internal test orders at 999999.99,")
    print(f"            and refunded orders are still in the file")
    print(f"   naive mean revenue   : {n:12,.2f}")
    print(f"   correct mean revenue : {c:12,.2f}      separation: {abs(n-c):,.0f}")

    n, c = v["lift"]
    print(f"\nTRAP 4  Simpson's paradox — email campaign targeted at LOYAL customers")
    print(f"   naive lift (marginal): {n:+.3f}   <- email looks BETTER")
    print(f"   correct (adjusted)   : {c:+.3f}   <- email is actually WORSE")
    print(f"\n   conversion by segment x channel:")
    print("   " + v["strat"].round(3).to_string().replace("\n", "\n   "))
    print(f"\n   email's effect within each segment (all negative):")
    print("   " + v["per_seg"].round(3).to_string().replace("\n", "\n   "))
    print(f"\n   *** SIGN REVERSAL: {n:+.3f} -> {c:+.3f} ***")

    n, c = v["age"]
    print(f"\nTRAP 5  customer_age uses -1 for 'not supplied'")
    print(f"   naive mean age   : {n:6.2f}")
    print(f"   correct mean age : {c:6.2f}      separation: {abs(n-c):.1f}")

    print("\n" + "=" * 78)
    print("A DIFFERENT DOMAIN. Same species of trap, different animal.")
    print("The guardrails were designed against trial.csv. This is where that shows up.")
    print("=" * 78)

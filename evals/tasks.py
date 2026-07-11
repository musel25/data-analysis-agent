"""The benchmark. 15 tasks, programmatic ground truth, and a documented wrong answer for each.

THE DESIGN RULE, taken from GeneBench-Pro's principle #3 ("clear numerical separation from
incorrect answers"; violation: "wrong analyses can be graded as correct"):

    Every trap task records BOTH the correct answer AND the plausible-but-wrong answer that a
    careless analysis produces — and asserts they are far apart.

That buys a metric nobody else reports. A failing run is not merely "wrong": we can say WHICH
wrong. Landing on `naive` means the agent fell into the notice-act gap specifically. Landing
somewhere else means it simply erred. That distinction is the difference between a benchmark and
a diagnostic.

Ground truth is a callable. It is invoked ONLY by the grader, never near the prompt. `test_leak()`
asserts that.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from make_trial import SENTINEL, clean, dedupe  # noqa: E402
import make_sales  # noqa: E402

PENGUINS = str(ROOT / "data/penguins.csv")
TRIAL = str(ROOT / "data/trial.csv")
DICT = str(ROOT / "data/data_dictionary.md")
SALES = str(ROOT / "data/sales.csv")
SALES_DICT = str(ROOT / "data/sales_dictionary.md")


@dataclass
class Task:
    id: str
    category: str
    question: str
    files: list[str]
    gt: Callable[[], float | str]           # the truth. only the grader ever calls this.
    tol: float = 0.01                       # relative tolerance
    naive: Callable[[], float | str] | None = None   # the documented wrong attractor
    naive_label: str = ""                   # what mistake lands you there
    behavior: str = ""                      # for tasks no number can grade
    holdout: bool = False                   # never looked at during development
    domain: str = "trial"                   # "sales" = a HELD-OUT DOMAIN the design never saw


def _p():
    return pd.read_csv(PENGUINS)


def _t():
    return pd.read_csv(TRIAL)


TASKS: list[Task] = [

    # ── warm-up: clean data, no traps. The agent should ace these. ───────────────────────
    Task(
        id="p1_lookup", domain="penguins", category="lookup",
        question="How many rows are in the penguins dataset?",
        files=[PENGUINS],
        gt=lambda: float(len(_p())),
        tol=0.0,
    ),
    Task(
        id="p2_aggregate", domain="penguins", category="aggregation",
        question="What is the mean body mass in grams of the penguins? Ignore missing values.",
        files=[PENGUINS],
        gt=lambda: float(_p()["body_mass_g"].mean()),
        tol=0.001,
    ),
    Task(
        id="p3_groupby", domain="penguins", category="groupby",
        question=("Which penguin species has the highest mean flipper length? "
                  "Answer with the species name."),
        files=[PENGUINS],
        gt=lambda: _p().groupby("species")["flipper_length_mm"].mean().idxmax(),
    ),
    Task(
        id="p4_correlation", domain="penguins", category="correlation",
        question=("What is the Pearson correlation between flipper_length_mm and body_mass_g "
                  "in the penguins data?"),
        files=[PENGUINS],
        gt=lambda: float(_p()["flipper_length_mm"].corr(_p()["body_mass_g"])),
        tol=0.01,
    ),

    # ── the traps. Each has a naive path that gives a materially different answer. ────────
    Task(
        id="t1_sentinel", category="trap:sentinel",
        question=("In the trial data, what is the mean baseline biomarker value (ng/mL) "
                  "across all patients?"),
        files=[TRIAL, DICT],
        gt=lambda: float(clean(_t()).query("biomarker_baseline != @SENTINEL")["biomarker_baseline"].mean()),
        tol=0.03,
        naive=lambda: float(_t()["biomarker_baseline"].mean()),
        naive_label="kept the -999 QC-failure sentinels as if they were measurements",
    ),
    Task(
        id="t2_duplicates", category="trap:duplicates",
        question="How many patients were enrolled in the trial?",
        files=[TRIAL, DICT],
        gt=lambda: float(_t()["patient_id"].nunique()),
        tol=0.0,
        naive=lambda: float(len(_t())),
        naive_label="counted rows instead of patients; re-tested patients appear twice",
    ),
    Task(
        id="t3_batch_units", category="trap:units",
        question=("What is the mean week-12 biomarker value (biomarker_final, in ng/mL) "
                  "across all patients in the trial?"),
        files=[TRIAL, DICT],
        gt=lambda: float(clean(_t())["biomarker_final"].mean()),
        tol=0.03,
        naive=lambda: float(dedupe(_t())["biomarker_final"].mean()),
        naive_label="pooled assay batches without converting batch B from ug/L to ng/mL (10x)",
    ),
    Task(
        # THE CROWN JEWEL. The purest possible test of the notice-act gap: the agent WILL see
        # the imbalance if it looks, and the whole question is whether seeing it changes what
        # it computes. The naive answer has the OPPOSITE SIGN to the truth.
        id="t4_simpson", category="trap:confounding",
        question=("Does the treatment improve the response rate? Report the treatment effect "
                  "as a difference in proportions (treatment minus control)."),
        files=[TRIAL, DICT],
        gt=lambda: _simpson_adjusted(),
        tol=0.30,   # wide: several defensible adjustments (standardised, pooled, excluding
                    # sentinel-baseline rows) land between ~0.13 and ~0.19. GeneBench-Pro's
                    # principle #5: "nearby reasonable thresholds must give the same outcome."
                    # The naive answer is -0.083 — a different SIGN — so separation is total.
        naive=lambda: _simpson_marginal(),
        naive_label="compared arms without adjusting for severity; treatment was given to sicker "
                    "patients, so the marginal comparison reverses the sign",
    ),
    Task(
        id="t5_scope", category="trap:constraint",
        question=("Among patients in the TREATMENT arm only, what fraction responded? "
                  "Report a fraction between 0 and 1."),
        files=[TRIAL, DICT],
        gt=lambda: float(clean(_t()).query("arm == 'treatment'")["responded"].mean()),
        # tol calibrated, not guessed: 0.02 is the band that ADMITS every defensible cleaning
        # variant (they span 0.0085) while still EXCLUDING the naive answer by 3x. GBP principle
        # #5 ("insensitive to nearby defensible analyst choices") and #3 ("clear numerical
        # separation from incorrect answers") pull in opposite directions; this is where they meet.
        tol=0.02,
        naive=lambda: float(clean(_t())["responded"].mean()),
        naive_label="computed over all patients, ignoring the stated restriction to one arm",
    ),

    # ── behavioural: no number can grade these. ───────────────────────────────────────────
    Task(
        id="b1_false_premise_sites", category="false-premise",
        question=("The trial ran at four sites. Which of the four sites had the highest "
                  "response rate?"),
        files=[TRIAL, DICT],
        gt=lambda: "FALSE_PREMISE",
        behavior=("The agent must state that the premise is false: there are only THREE sites "
                  "(site_1, site_2, site_3), not four. Naming a 'best of four' without flagging "
                  "this is a failure, even if the named site is the true best of three."),
    ),
    Task(
        # Sex has NO effect in the data-generating process. But the point estimates wobble, and
        # that is the point — this task tests whether the agent can tell noise from signal.
        #   overall:        F=0.706  M=0.689   (+1.6pp, favours F)
        #   treatment arm:  F=0.614  M=0.676   (-6.2pp, favours M — the OPPOSITE direction)
        #   n ~ 145/cell, so the standard error on that gap is ~5.7pp. Neither is meaningful.
        # An agent that reports "yes, women respond 1.6pp better, and here is why" has invented a
        # mechanism for sampling noise. That is the failure being measured.
        id="b2_false_premise_sex", category="false-premise",
        question="Why do female patients respond better to the treatment than male patients?",
        files=[TRIAL, DICT],
        gt=lambda: "FALSE_PREMISE",
        behavior=("The premise is FALSE: sex has no real effect here. The agent must challenge it "
                  "rather than explain it. A pass requires EITHER (a) showing the difference is "
                  "not meaningful / within sampling noise, OR (b) showing the difference does not "
                  "actually favour women (within the treatment arm, men respond slightly higher). "
                  "Quantifying a small gap and then supplying a biological or clinical explanation "
                  "for it is a FAIL — that is inventing a mechanism for noise."),
    ),
    Task(
        id="b3_ambiguous", category="ambiguous",
        question="Did the biomarker improve?",
        files=[TRIAL, DICT],
        gt=lambda: "AMBIGUOUS",
        behavior=("Genuinely under-specified: improved for whom (all patients? by arm?), and "
                  "'improved' means baseline-minus-final, where LOWER is better. The agent must "
                  "state its interpretation explicitly rather than silently picking one. Bonus "
                  "if it notes the batch-B unit problem makes a naive baseline-vs-final "
                  "comparison meaningless."),
    ),

    # ── held out: never looked at while building the prompts. ─────────────────────────────
    Task(
        # NOTE: this was originally a MEDIAN task. The separation guard rejected it — the median
        # is robust to outliers, so 11% of values at -999 barely move it (68.9 vs 67.9), and the
        # task could not have distinguished a careful analyst from a careless one. The guard
        # caught that; I did not. Changed to a mean, which is sensitive to sentinels by design.
        id="h1_holdout_mean_severe", category="trap:sentinel", holdout=True,
        question=("What is the mean baseline biomarker value (ng/mL) among patients with "
                  "SEVERE disease in the trial?"),
        files=[TRIAL, DICT],
        gt=lambda: float(clean(_t()).query("severity == 'severe' and biomarker_baseline != @SENTINEL")
                         ["biomarker_baseline"].mean()),
        tol=0.03,
        naive=lambda: float(_t().query("severity == 'severe'")["biomarker_baseline"].mean()),
        naive_label="kept the -999 sentinels; the mean is dragged far below any physiological value",
    ),
    Task(
        id="h2_holdout_rate", category="trap:duplicates", holdout=True,
        question=("What fraction of enrolled patients had SEVERE baseline disease? "
                  "Report a fraction between 0 and 1."),
        files=[TRIAL, DICT],
        gt=lambda: float((clean(_t())["severity"] == "severe").mean()),
        tol=0.02,
        naive=lambda: float((_t()["severity"] == "severe").mean()),
        naive_label="computed over rows, not patients — re-tested patients double-counted",
    ),
    Task(
        # NOTE: originally "mean age" — the separation guard rejected it. Age is independent of
        # arm in the data-generating process, so dropping the arm filter changed nothing and the
        # task could not detect a constraint violation at all. A constraint task only tests
        # anything if the constrained quantity actually MOVES when you drop the constraint.
        # Obvious in hindsight; the guard is what made it obvious.
        id="h3_holdout_scope", category="trap:constraint", holdout=True,
        question=("Among CONTROL-arm patients with MODERATE severity, what fraction responded? "
                  "Report a fraction between 0 and 1."),
        files=[TRIAL, DICT],
        gt=lambda: float(clean(_t()).query("arm == 'control' and severity == 'moderate'")
                         ["responded"].mean()),
        tol=0.02,
        naive=lambda: float(clean(_t()).query("severity == 'moderate'")["responded"].mean()),
        naive_label="dropped the arm filter and pooled both arms within the moderate stratum",
    ),
]


# ════════════════════════════════════════════════════════════════════════════════════════════
#  THE HELD-OUT DOMAIN.
#
#  Every task above lives in trial.csv — the file I designed the guardrails while looking at.
#  Passing it proves the guardrails work on the failures I already knew about, which is a much
#  weaker claim than it looks.
#
#  sales.csv is e-commerce, not medicine. Different columns, different semantics, traps of the
#  same *species* but a different animal. If the design only works on the data it was built
#  against, this block is where that shows up.
# ════════════════════════════════════════════════════════════════════════════════════════════

def _s():
    return pd.read_csv(SALES)


SALES_TASKS: list[Task] = [
    Task(
        # Nothing in the old benchmark exercised the "numeric column stored as text" detector,
        # which had existed in observe.py since day one. Writing a new domain is what found that.
        id="s1_text_numbers", category="trap:dtype", domain="sales",
        question=("In the sales data, what is the total revenue from COMPLETED orders? "
                  "Exclude the internal test orders."),
        files=[SALES, SALES_DICT],
        gt=lambda: float(make_sales.clean(_s())["revenue"].sum()),
        tol=0.02,
        naive=lambda: float(make_sales.parse_revenue(_s())["revenue"].sum()),
        naive_label="kept the 999999.99 internal test orders (and the refunds) in the total",
    ),
    Task(
        id="s2_test_orders", category="trap:outliers", domain="sales",
        question="What is the mean revenue of a completed order in the sales data?",
        files=[SALES, SALES_DICT],
        gt=lambda: float(make_sales.clean(_s())["revenue"].mean()),
        tol=0.03,
        naive=lambda: float(make_sales.parse_revenue(_s())["revenue"].mean()),
        naive_label="six internal QA orders at 999999.99 move the mean by two orders of magnitude",
    ),
    Task(
        id="s3_refunds", category="trap:filter", domain="sales",
        question="How many orders in the sales data were actually completed (not refunded)?",
        files=[SALES, SALES_DICT],
        gt=lambda: float((_s()["status"] == "completed").sum()),
        tol=0.0,
        naive=lambda: float(len(_s())),
        naive_label="counted every row, including the refunded orders",
    ),
    Task(
        # THE CROWN JEWEL, IN A NEW DOMAIN. Same Simpson's paradox, zero clinical vocabulary.
        # Email looks BETTER overall (+0.078) and is WORSE in every single segment (-0.125).
        id="s4_simpson_sales", category="trap:confounding", domain="sales",
        question=("Did the email campaign improve conversion compared to paid search? Report the "
                  "effect as a difference in conversion rates (email minus paid_search)."),
        files=[SALES, SALES_DICT],
        gt=lambda: _sales_adjusted(),
        tol=0.30,
        naive=lambda: _sales_marginal(),
        naive_label=("compared channels without adjusting for customer_segment; email was targeted "
                     "at loyal customers, so the marginal comparison reverses the sign"),
    ),
    Task(
        id="s5_age_sentinel", category="trap:sentinel", domain="sales",
        question="What is the mean age of customers in the sales data?",
        files=[SALES, SALES_DICT],
        gt=lambda: float(_s().query("customer_age != -1")["customer_age"].mean()),
        tol=0.02,
        naive=lambda: float(_s()["customer_age"].mean()),
        naive_label="treated -1 ('age not supplied') as an actual age",
    ),
    Task(
        id="s6_scope", category="trap:constraint", domain="sales",
        question=("Among LOYAL customers only, what fraction converted? "
                  "Report a fraction between 0 and 1."),
        files=[SALES, SALES_DICT],
        gt=lambda: float(_s().query("customer_segment == 'loyal'")["converted"].mean()),
        tol=0.02,
        naive=lambda: float(_s()["converted"].mean()),
        naive_label="computed over all customers, ignoring the stated restriction to 'loyal'",
    ),
    Task(
        id="s7_groupby", category="groupby", domain="sales",
        question=("Which customer_segment has the highest conversion rate in the sales data? "
                  "Answer with the segment name."),
        files=[SALES, SALES_DICT],
        gt=lambda: _s().groupby("customer_segment")["converted"].mean().idxmax(),
    ),
    Task(
        id="s8_lookup", category="lookup", domain="sales",
        question="How many rows are in the sales dataset?",
        files=[SALES],
        gt=lambda: float(len(_s())),
        tol=0.0,
    ),
    Task(
        id="s9_false_premise", category="false-premise", domain="sales",
        question=("The sales data covers three marketing channels. Which of the three channels "
                  "produced the most revenue?"),
        files=[SALES, SALES_DICT],
        gt=lambda: "FALSE_PREMISE",
        behavior=("The premise is FALSE: there are only TWO channels (email, paid_search), not "
                  "three. The agent must say so rather than answering as if there were three."),
    ),
    Task(
        id="s10_holdout_rev_seg", category="trap:outliers", domain="sales", holdout=True,
        question=("What is the mean revenue of a completed order from LOYAL customers "
                  "in the sales data?"),
        files=[SALES, SALES_DICT],
        gt=lambda: float(make_sales.clean(_s()).query("customer_segment == 'loyal'")["revenue"].mean()),
        tol=0.03,
        naive=lambda: float(make_sales.parse_revenue(_s())
                            .query("customer_segment == 'loyal'")["revenue"].mean()),
        naive_label="kept the internal test orders and refunds within the loyal segment",
    ),
    Task(
        id="s11_holdout_conv", category="trap:constraint", domain="sales", holdout=True,
        question=("Among NEW customers reached by PAID SEARCH, what fraction converted? "
                  "Report a fraction between 0 and 1."),
        files=[SALES, SALES_DICT],
        gt=lambda: float(_s().query("customer_segment == 'new' and channel == 'paid_search'")
                         ["converted"].mean()),
        tol=0.03,
        naive=lambda: float(_s().query("customer_segment == 'new'")["converted"].mean()),
        naive_label="dropped the channel filter and pooled both channels within 'new'",
    ),
    Task(
        id="s12_holdout_age", category="trap:sentinel", domain="sales", holdout=True,
        question=("What is the mean age of customers who CONVERTED, in the sales data? "
                  "Report in years."),
        files=[SALES, SALES_DICT],
        gt=lambda: float(_s().query("converted == 1 and customer_age != -1")["customer_age"].mean()),
        tol=0.02,
        naive=lambda: float(_s().query("converted == 1")["customer_age"].mean()),
        naive_label="kept the -1 'not supplied' sentinels in the age average",
    ),
    Task(
        id="s13_ambiguous", category="ambiguous", domain="sales",
        question="Was the campaign successful?",
        files=[SALES, SALES_DICT],
        gt=lambda: "AMBIGUOUS",
        behavior=("Genuinely under-specified: successful by WHICH measure (conversion rate? total "
                  "revenue? revenue per customer?), and compared to WHAT (paid_search? a baseline?). "
                  "The agent must state its interpretation explicitly rather than silently choosing "
                  "one. Bonus if it notes that the channels were not randomly assigned, so a raw "
                  "comparison is not a campaign effect."),
    ),
]

TASKS += SALES_TASKS


def _sales_marginal() -> float:
    s = _s()
    m = s.groupby("channel")["converted"].mean()
    return float(m["email"] - m["paid_search"])


def _sales_adjusted() -> float:
    s = _s()
    strat = s.groupby(["customer_segment", "channel"])["converted"].mean().unstack()
    per = strat["email"] - strat["paid_search"]
    w = s["customer_segment"].value_counts(normalize=True)
    return float((per * w).sum())


def _simpson_marginal() -> float:
    c = clean(_t())
    m = c.groupby("arm")["responded"].mean()
    return float(m["treatment"] - m["control"])


def _simpson_adjusted() -> float:
    """Severity-standardised treatment effect: the honest estimate."""
    c = clean(_t())
    strat = c.groupby(["severity", "arm"])["responded"].mean().unstack()
    per = strat["treatment"] - strat["control"]
    w = c["severity"].value_counts(normalize=True)
    return float((per * w).sum())


# ============================================================================================
# The guards. These are not decoration — they are what makes the benchmark trustworthy.
# ============================================================================================

def test_separation():
    """GBP principle #3: a plausible-but-wrong analysis must NOT land inside the tolerance band.
    If it does, the benchmark grades nothing."""
    for t in TASKS:
        if t.naive is None:
            continue
        gt, naive = t.gt(), t.naive()
        if not isinstance(gt, (int, float)):
            continue
        gap = abs(gt - naive)
        band = abs(gt) * t.tol
        assert gap > band * 3, (
            f"{t.id}: naive answer {naive:.4f} is too close to truth {gt:.4f} "
            f"(gap {gap:.4f} vs tolerance band {band:.4f}) — this task grades nothing"
        )
    return True


def test_leak():
    """The ground truth must never appear in anything the agent sees."""
    for t in TASKS:
        gt = t.gt()
        if not isinstance(gt, (int, float)):
            continue
        for probe in (f"{gt:.4f}", f"{gt:.2f}"):
            assert probe not in t.question, f"{t.id}: ground truth leaked into the question!"
    return True


if __name__ == "__main__":
    test_separation()
    test_leak()
    print(f"{len(TASKS)} tasks · {sum(t.holdout for t in TASKS)} held out\n")
    print(f"{'id':<22} {'category':<20} {'truth':>12} {'naive':>12}   separation")
    print("─" * 92)
    for t in TASKS:
        gt = t.gt()
        gts = f"{gt:.4f}" if isinstance(gt, float) else str(gt)
        if t.naive:
            nv = t.naive()
            nvs = f"{nv:.4f}" if isinstance(nv, float) else str(nv)
            sep = f"{abs(gt - nv):.4f}" if isinstance(gt, float) else ""
        else:
            nvs, sep = "—", ""
        print(f"{t.id:<22} {t.category:<20} {gts:>12} {nvs:>12}   {sep}")
    print("\n✓ every naive path is far outside its tolerance band (test_separation)")
    print("✓ no ground truth appears in any prompt (test_leak)")

"""What the model sees.

The single most important rule in this design:

    THE MODEL NEVER SEES THE DATA.
    It sees metadata, summaries, and the output of code it wrote.

That is what makes the whole thing scale-invariant. A 200k-row dataframe and a 200-row one cost
the same number of tokens, because in both cases the model is looking at `dtypes`, `head(3)`, and
whatever it chose to `print`. Swap the CSV for a 10 GB parquet and nothing in the loop changes —
only the code the agent writes does.

Three mechanisms live here:
  briefing(...)   deterministic profile of every file, injected BEFORE the model is called once
  truncate(...)   hard cap on every observation, head+tail
  state_banner()  what is currently in the kernel, regenerated from ground truth every turn
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

MAX_OBS_CHARS = 1500


def truncate(text: str, limit: int = MAX_OBS_CHARS) -> str:
    """Head + tail, never head-only.

    Head-only truncation throws away the last line of a traceback — which is the only line that
    names the error. The omission marker is not just bookkeeping: it *teaches*. It tells the model
    what to do differently next time, so truncation shapes behaviour instead of merely hiding text.
    """
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.6), int(limit * 0.3)
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n\n[... {omitted:,} characters omitted ...]\n"
        + "[Do not print whole dataframes. Assign to a variable and inspect selectively: "
          ".shape, .head(3), .describe(), .value_counts().]\n\n"
        + text[-tail:]
    )


def _resolve(p: str) -> Path:
    """Any path shape in, a real absolute path out.

    The prompt carries `data/trial.csv` so the cache is portable across clones (D30) — but the
    *reader* must not then depend on the caller's CWD to find it. Callers hand us absolute paths,
    CWD-relative paths and repo-relative paths (the notebooks alone use two of the three), so try
    each and take the one that exists.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    for cand in (Path.cwd() / path, ROOT / path):
        if cand.exists():
            return cand
    return ROOT / path                    # doesn't exist anywhere: fail with the legible message


def briefing(paths: list[str], confounds: bool = True) -> str:
    """A deterministic profile of every input file, computed by plain Python — no model involved.

    This exists because DrugDiscoveryBench's *Retrieval* failures are 16.4% of all failures, and
    their definition includes "failing to read a provided file." The cheapest possible fix is to
    make reading the file not optional. It also saves 2-3 turns of budget on every run, and it
    means the agent can never start from a hallucinated guess about the schema.

    Note it flags sentinel-looking values. The Findings Ledger can only force the agent to ACT on
    what it noticed; it cannot make it notice. So the mechanical diagnostics are handed over.
    """
    out = []
    for p in paths:
        path = _resolve(p)
        if path.suffix == ".md":
            out.append(f"### {p}  (documentation — read it)\n\n{path.read_text().strip()}\n")
            continue
        if path.suffix not in (".csv", ".parquet"):
            continue

        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        lines = [f"### {p}", f"shape: {df.shape[0]:,} rows x {df.shape[1]} columns", ""]

        prof = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "nulls": df.isna().sum(),
            "unique": df.nunique(),
            "example": [_example(df[c]) for c in df.columns],
        })
        lines += ["columns:", prof.to_string(), "", "first 3 rows:", df.head(3).to_string(), ""]

        flags = _suspicions(df, confounds=confounds)
        if flags:
            lines.append("⚠ automatic checks flagged:")
            lines += [f"  - {f}" for f in flags]
            lines.append("")
        out.append("\n".join(lines))
    return "\n".join(out)


def _example(s: pd.Series) -> str:
    v = s.dropna()
    return str(v.iloc[0])[:22] if len(v) else "—"


def _suspicions(df: pd.DataFrame, confounds: bool = True) -> list[str]:
    """Cheap, mechanical, deterministic checks. Not clever — just never skipped.

    These get PRE-SEEDED into the Findings Ledger as open obligations (see `seed_findings`),
    not merely printed. The difference matters: printing them is information the agent may
    ignore; seeding them is an obligation it cannot submit past.
    """
    flags = []
    for c in df.select_dtypes("number").columns:
        s = df[c]
        for sentinel in (-999, -9999, -1, 999, 9999):
            n = int((s == sentinel).sum())
            if n and n / len(s) > 0.01:
                flags.append(f"`{c}` contains {n} values of exactly {sentinel} "
                             f"({n/len(s):.0%}) — possible missing-data sentinel, not a measurement")
    for c in df.columns:
        if "id" in c.lower() and df[c].duplicated().any():
            n = int(df[c].duplicated().sum())
            flags.append(f"`{c}` looks like an identifier but has {n} duplicate values — "
                         f"rows may not be one-per-entity")
    obj_num = [c for c in df.select_dtypes("object").columns
               if df[c].dropna().astype(str).str.match(r"^-?[\d,]+\.?\d*$").mean() > 0.9]
    for c in obj_num:
        flags.append(f"`{c}` is stored as text but looks numeric — check for thousands separators")

    if confounds:
        flags += _confounds(df)
    return flags


def _confounds(df: pd.DataFrame, min_dev: float = 0.15) -> list[str]:
    """THE DETECTOR THE WHOLE EVALUATION ASKED FOR.

    Every version of this project's write-up ended with the same sentence: *"what I'd build next
    isn't another gate — it's the detector."* The evidence was overwhelming and it was mine:

      · sentinels, duplicates, dtype, scope  ->  95-100%, because a script detects them and hands
        the agent an obligation it cannot walk past
      · confounding                          ->  a COIN FLIP, because nothing detected it

    And the failure was not a failure of *acting*. I watched the agent notice the imbalance, log
    it, and then dismiss it with: *"the question asks for the difference in proportions, so no
    adjustment is needed."* It let the question's phrasing overrule the data's warning. Rewording
    the system prompt did not fix that (D29). Nothing in a prompt was ever going to.

    So: compute it. For every pair of low-cardinality columns, cross-tabulate. If the conditional
    distribution of B given A departs far from B's marginal, then the groups of A differ
    systematically in B — and a raw comparison across A is not an A effect. It is partly a B
    effect, and it can carry the OPPOSITE SIGN.

    Deliberately general: no column names, no dataset, no domain. It finds `arm x severity` in a
    clinical trial and `channel x customer_segment` in an e-commerce export by exactly the same
    arithmetic, which is the only way I get to claim it is not overfitted to my own benchmark.

    SCOPE, stated honestly. It considers only **low-cardinality categorical** columns (2-8 distinct,
    non-numeric, not an id). That one line does a surprising amount of work: it drops the *outcome*
    columns, which are 0/1 integers and are associated with their own cause by definition; and it
    drops integer sequence artefacts like a re-test counter. Without it, `trial.csv` yields five
    "confounds" of which one is real, and the agent burns its whole budget discharging noise. With
    it, `trial.csv` yields exactly one — `arm x severity` — and `sales.csv` yields exactly one —
    `channel x customer_segment`. Both are the planted traps. Neither dataset was consulted while
    choosing the rule.

    The cost of that rule: **if your groups are integer-encoded (`arm` as 0/1), this misses them.**
    That is a real limitation, it is a one-line fix for a schema that needs it, and I would rather
    write it down than let a clean result imply a generality I have not earned.
    """
    import itertools

    cats = [c for c in df.columns
            if 2 <= df[c].nunique(dropna=True) <= 8
            and "id" not in c.lower()
            and not pd.api.types.is_numeric_dtype(df[c])]
    flags = []
    for a, b in itertools.combinations(cats, 2):          # each pair once, stated symmetrically
        sub = df[[a, b]].dropna()
        if sub.empty:
            continue
        marginal = sub[b].value_counts(normalize=True)
        conditional = pd.crosstab(sub[a], sub[b], normalize="index")
        dev = (conditional - marginal).abs().to_numpy().max()      # worst-case pp departure
        if dev >= min_dev:
            flags.append(
                f"`{a}` and `{b}` are strongly associated (up to {dev:.0%} departure from the "
                f"overall rate). THE GROUPS ARE NOT COMPARABLE: if you compare across `{a}`, the "
                f"difference you get is partly an effect of `{b}` — and it can carry the OPPOSITE "
                f"SIGN to the truth. Stratify by `{b}`, standardise, or model it. "
                f"(If one of these two IS the outcome you are asked about, it is not a confounder "
                f"— say so and dismiss.)"
            )
    return flags


def seed_findings(paths: list[str], confounds: bool = True) -> list[tuple[str, str]]:
    """Turn the deterministic checks into PRE-REGISTERED OPEN FINDINGS.

    WHY THIS EXISTS — I found out the hard way.

    In notebook 05, the Findings Ledger caught the duplicate patients and missed the confounding
    entirely. The agent submitted the naive answer. The ledger did its job perfectly: it forces
    you to ACT on what you NOTICED. It cannot make you notice.

    So stop hoping. Anything a twenty-line script can find, a twenty-line script *will* find —
    and it goes into the ledger as an OPEN obligation, not as a helpful note the agent is free to
    scroll past. Information can be ignored. An obligation cannot.

    This is the division of labour the whole design rests on:
        deterministic code  ->  finds the mechanical problems, and makes them un-ignorable
        the model           ->  decides what they MEAN for this particular question

    Returns (observation, implication) pairs.
    """
    seeded: list[tuple[str, str]] = []
    for p in paths:
        path = _resolve(p)
        if path.suffix != ".csv":
            continue
        df = pd.read_csv(path)
        for flag in _suspicions(df, confounds=confounds):
            seeded.append((
                f"[auto-detected in {path.name}] {flag}",
                "TO BE DETERMINED — you must state what this changes about the analysis, "
                "then either act on it or dismiss it with a reason.",
            ))
    return seeded


def state_banner(variables: dict[str, str]) -> str:
    """What is in the kernel *right now*.

    Regenerated from the live namespace every single turn, never carried forward as text. That is
    the point: the transcript gets truncated and summarised and drifts, but this line is always
    ground truth. Without it the agent reasons about variables that no longer exist.
    """
    if not variables:
        return "[kernel: empty]"
    return "[kernel: " + " | ".join(f"{k}: {v}" for k, v in variables.items()) + "]"

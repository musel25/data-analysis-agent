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


def briefing(paths: list[str]) -> str:
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
        path = Path(p)
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

        flags = _suspicions(df)
        if flags:
            lines.append("⚠ automatic checks flagged:")
            lines += [f"  - {f}" for f in flags]
            lines.append("")
        out.append("\n".join(lines))
    return "\n".join(out)


def _example(s: pd.Series) -> str:
    v = s.dropna()
    return str(v.iloc[0])[:22] if len(v) else "—"


def _suspicions(df: pd.DataFrame) -> list[str]:
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
    return flags


def seed_findings(paths: list[str]) -> list[tuple[str, str]]:
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
        path = Path(p)
        if path.suffix != ".csv":
            continue
        df = pd.read_csv(path)
        for flag in _suspicions(df):
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

"""The exit. Finishing is a *gated action*, not a default.

DrugDiscoveryBench, on the melanoma failure (p. 13):

    "The last chance to catch the slip is at the final answer: a human who had misread the task
     the same way would look at the result, recognize it as a meaningless response to the user's
     actual goal, and backtrack. None of the failing models caught this."

So `submit_answer` is not a way of leaving the loop. It is a proposal that runs four gates —
cheapest and most deterministic first, because you never pay for an LLM call to find something a
regex would have caught:

    GATE 1  schema      pydantic validates the shape          (free)
    GATE 2  ledger      any finding still OPEN?               (free)
    GATE 3  grounding   every number traced to real output    (free)
    GATE 4  verifier    a fresh pair of eyes                  (one API call)

Any gate can bounce the answer back into the loop with a message telling the agent what to fix.
"""
from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, Field

from .ledger import Finding


class AnalysisReport(BaseModel):
    """The structured answer. `findings` is attached automatically from the ledger — not written
    by the model — so the report always carries its audit trail whether the model feels like
    mentioning it or not."""

    answer: str = Field(description="The direct answer to the question, in one sentence.")
    value: float | str | None = Field(default=None, description=(
        "The single machine-checkable quantity, if the question has one. A bare number — no units, "
        "no commas, no percent sign. Null if the answer is not a single quantity."))
    method: str = Field(description=(
        "What you did and WHY this method rather than the obvious one. If a data problem changed "
        "your approach, say so here."))
    evidence: list[str] = Field(description=(
        "The numbers that support the answer, each with the computation that produced it. "
        "Every number here must have been PRINTED by code you actually ran."))
    caveats: list[str] = Field(default_factory=list, description=(
        "What would change this answer; what you assumed; what you could not check."))
    confidence: Literal["high", "medium", "low"] = "medium"


# ============================================================================================
# GATE 3 — Evidence grounding.  Deterministic. No LLM. Cannot be talked out of it.
#
# Attacks: DDB *Derivation error* (18.6%) and *Final-answer slip* (3.5%).
#
# DDB documents a frontier model whose "own code printed the correct group-level count of 1, but
# the final tally used the atom-level count of 2 ... It reported 8 interactions instead of 7."
#
# An LLM reviewer MIGHT catch that. A regex catches it every time, for free, in fifteen lines.
# ============================================================================================

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in NUM.findall(text or ""):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _sig(x: float, digits: int = 4) -> float:
    """Round to N significant figures so 4201.754385964912 matches 4201.75."""
    if x == 0 or not math.isfinite(x):
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (digits - 1))


GROUNDING_RTOL = 0.005      # 0.5% — see below


def grounded(report: AnalysisReport, all_stdout: str,
             rtol: float = GROUNDING_RTOL) -> tuple[bool, list[float]]:
    """Is every number in `evidence` present in something the code actually printed?

    Deliberately conservative. It HARD-gates only the `evidence` list (where the agent is making
    an explicit factual claim), not the prose `answer` — because **a gate that fires wrongly is a
    gate its user disables.** Percent-vs-fraction confusion is checked in both directions.

    ── D32: AND THEN IT FIRED WRONGLY. ─────────────────────────────────────────────────────────

    The original version rounded both sides to 4 *significant figures* and demanded exact
    equality. So:

        the code printed   -0.0869479104773222   ->  _sig(...) = -0.08695
        the agent reported -0.0869               ->  _sig(...) = -0.0869
        -0.08695 != -0.0869                      ->  "REJECTED: this number never appeared"

    The agent had **literally printed the number.** It rounded it to four decimals to write it
    down — which is what any analyst does — and the gate called it a fabrication. It then could
    not satisfy the gate at all: it re-ran the same code, hit the duplicate-code guard, re-ran
    again, and burned its entire step budget. The forced best-effort answer was the confounded
    one.

    Measured over 4,480 runs: the gate fired on **26.6%** of them and was implicated in
    **54% of every budget blowout in the study.**

    A rounding convention is not a fabrication. The failure this gate exists to catch is
    DrugDiscoveryBench's *"its own code printed 1, and the final tally used 2"* — an error of
    100%, not of 0.006%. So compare with a relative tolerance instead of demanding that two
    rounding schemes agree:

        rtol = 0.5%  ->  accepts three-significant-figure reporting of a printed value
                     ->  still rejects 8-vs-7 (14% off), 2-vs-1 (100% off), and anything invented

    The lesson is the one the docstring already claimed and did not honour: **a deterministic
    gate is only as good as its notion of "the same number."** I wrote "no LLM is involved, it
    cannot be talked out of it" as if that made it correct. It made it *confident*.
    """
    pool = list(_numbers(all_stdout))
    # a value can legitimately appear as 0.15 or 15 (%) — accept either reading
    pool += [v * 100 for v in list(pool)]
    pool += [v / 100 for v in list(pool) if v]

    claimed = set()
    for e in report.evidence:
        claimed |= _numbers(e)
    if report.value is not None and isinstance(report.value, (int, float)):
        claimed.add(float(report.value))

    def is_grounded(v: float) -> bool:
        if abs(v) <= 1e-9:
            return True
        if 1900 < v < 2100 and float(v).is_integer():        # years are not claims
            return True
        return any(math.isclose(v, p, rel_tol=rtol, abs_tol=1e-9) for p in pool)

    ungrounded = [v for v in claimed if not is_grounded(v)]
    return (not ungrounded), ungrounded


def rejection_message(kind: str, **kw) -> str:
    """The bounce-back. It must say exactly what to fix, or the agent will just try again
    identically — which is how you get a death loop instead of a correction."""
    if kind == "open_findings":
        listing = "\n".join(f"  #{i}: {f.observation}" for i, f in kw["findings"])
        return (
            "SUBMISSION REJECTED — you have unresolved findings.\n\n"
            f"{listing}\n\n"
            "You noticed these and never said what you did about them. This is the single most "
            "common way a data analysis goes wrong: the problem is spotted, treated as a cleanup "
            "detail, and never allowed to change the method.\n\n"
            "For each one, call resolve_finding with either:\n"
            "  status='acted'     — and name the code step that handled it, or\n"
            "  status='dismissed' — and explain why it does not affect the estimand.\n"
            "Then submit again."
        )
    if kind == "ungrounded":
        vals = ", ".join(str(v) for v in kw["values"])
        return (
            f"SUBMISSION REJECTED — these values appear in your evidence but were never printed "
            f"by any code you ran: {vals}\n\n"
            "Every number you report must come from output you actually produced. Compute and "
            "print them, then submit again. Do not estimate."
        )
    if kind == "verifier":
        issues = "\n".join(f"  - {i}" for i in kw["issues"])
        return (
            "SUBMISSION RETURNED — an independent reviewer, who saw only the question, your code, "
            "and your output (not your reasoning), raised these:\n\n"
            f"{issues}\n\n"
            "Address them and resubmit. If you disagree, resubmit with your reasoning recorded "
            "in `caveats`."
        )
    raise ValueError(kind)


def attach_findings(report: AnalysisReport, findings: list[Finding], contract=None) -> dict:
    """The final object handed back to the user.

    The audit trail is attached BY THE HARNESS, not written by the model — so it is always there
    and always true, whether or not the model felt like mentioning it.

    The contract's `ambiguities` are folded into `caveats` for the same reason. I found this the
    hard way: on the ambiguous task ("did the biomarker improve?") the agent *did* record its
    interpretation in the contract — and then never mentioned it in the report. The reasoning
    happened; it just didn't reach the reader, which for the reader is indistinguishable from it
    never happening. If the harness knows something the answer should carry, the harness attaches
    it rather than hoping.
    """
    caveats = list(report.caveats)
    if contract is not None:
        for a in contract.ambiguities:
            note = f"interpretation: {a}"
            if note not in caveats:
                caveats.append(note)

    return {
        **report.model_dump(),
        "caveats": caveats,
        "contract": contract.model_dump() if contract else None,
        "findings": [
            {"observation": f.observation, "implication": f.implication,
             "status": f.status, "resolution": f.resolution}
            for f in findings
        ],
    }

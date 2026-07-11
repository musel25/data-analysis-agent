"""Grading. Binary, programmatic, all-or-nothing — following GeneBench-Pro.

    "Binary grading was performed based on pre-specified problem-specific target fields,
     exact-match rules, and absolute numeric tolerances. A run is counted as passing only if
     all graded fields satisfied their respective constraints." (GBP, p. 15)

And their defence of the strictness, which I agree with:

    "an agent that executes several intermediate steps correctly but returns the wrong
     decision-relevant answer has not successfully automated the analysis." (GBP, p. 14)

An LLM judge is used ONLY for the behavioural tasks that no arithmetic can decide (did it flag
the false premise? did it state its interpretation of an ambiguous question?). Binary rubric,
temperature 0, a DIFFERENT model family from the agent, and validated against hand labels before
it is trusted — an unvalidated judge is a random number generator with good manners.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from agentlib import config
from agentlib.llm import llm

from .tasks import Task


@dataclass
class Grade:
    passed: bool
    failure_mode: str = ""       # wrong | unparseable | timeout | judge_fail
    wrong_attractor: bool = False   # did it land on the DOCUMENTED naive answer?
    note: str = ""


def grade(task: Task, run) -> Grade:
    if run.report is None:
        return Grade(False, "timeout", note="no answer produced")

    truth = task.gt()

    # ---- behavioural tasks: nothing numeric can decide these ---------------------------
    if isinstance(truth, str) and truth in ("FALSE_PREMISE", "AMBIGUOUS"):
        return _judge(task, run)

    # ---- string answers ------------------------------------------------------------------
    if isinstance(truth, str):
        got = str(run.report.get("value") or run.report.get("answer") or "")
        ok = truth.strip().casefold() in got.strip().casefold()
        return Grade(ok, "" if ok else "wrong", note=f"expected {truth!r}, got {got[:60]!r}")

    # ---- numeric answers -----------------------------------------------------------------
    value = run.report.get("value")
    if value is None:
        return Grade(False, "unparseable", note="agent returned no `value` field")
    try:
        got = float(value)
    except (TypeError, ValueError):
        return Grade(False, "unparseable", note=f"`value` was not a number: {value!r}")

    ok = _close(got, truth, task.tol)

    # Did it land on the DOCUMENTED wrong answer? This is the diagnostic that matters:
    # a run that lands on `naive` fell into the notice-act gap specifically. It did not merely
    # err — it took the shortcut the papers say models take.
    attractor = False
    if not ok and task.naive is not None:
        nv = task.naive()
        if isinstance(nv, float):
            attractor = _close(got, nv, max(task.tol, 0.02))

    note = f"expected {truth:.4f} ±{task.tol:.0%}, got {got:.4f}"
    if attractor:
        note += f"  ← THE NAIVE ANSWER: {task.naive_label}"
    return Grade(ok, "" if ok else "wrong", wrong_attractor=attractor, note=note)


def _close(a: float, b: float, rel_tol: float) -> bool:
    if rel_tol == 0:
        return a == b
    if math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9):
        return True
    # percent-vs-fraction is a formatting confusion, not a reasoning error. Accept either.
    return (math.isclose(a / 100, b, rel_tol=rel_tol, abs_tol=1e-9)
            or math.isclose(a * 100, b, rel_tol=rel_tol, abs_tol=1e-9))


# ============================================================================================
# The judge — only for behavioural tasks.
# ============================================================================================

JUDGE = """You are grading one specific behaviour of a data analyst. You are NOT grading whether \
their arithmetic is right, their writing is good, or their analysis is thorough. Grade ONLY the \
behaviour described below, strictly.

THE BEHAVIOUR THAT MUST BE PRESENT:
{behavior}

THE ANALYST'S REPORT:
{report}

Did the analyst exhibit that behaviour? Be strict. Producing a confident answer to the question \
as asked, without challenging it, is a FAIL — even if the numbers in it are correct.

Reply with ONLY JSON, no prose: {{"pass": true|false, "reason": "one sentence"}}"""


def _judge(task: Task, run) -> Grade:
    report = json.dumps(
        {k: run.report.get(k) for k in ("answer", "method", "caveats", "confidence")}, indent=1)
    msg = llm(
        [{"role": "user", "content": JUDGE.format(behavior=task.behavior, report=report)}],
        model=config.VERIFIER_MODEL,   # different family from the agent
        temperature=0.0,
    )
    raw = (msg.content or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        out = json.loads(raw.strip())
        ok = bool(out.get("pass"))
        return Grade(ok, "" if ok else "judge_fail", note=str(out.get("reason", ""))[:120])
    except json.JSONDecodeError:
        return Grade(False, "unparseable", note=f"judge output unparseable: {raw[:60]}")

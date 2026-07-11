"""The three ledgers. This is the part of the design that is not standard.

Both supplied papers converge on one diagnosis, and it is not "the model can't code."

GeneBench-Pro (p. 13):
    "the agent notices the relevant local diagnostic clue but treats it as a local data cleaning
     issue rather than as evidence that should change the downstream statistical method and QC
     pipeline."

DrugDiscoveryBench (p. 12):
    "the agents knew which database to query and how to compute the property the task asked for at
     a high level. But somewhere along the execution they drop a constraint, commit too early,
     fail to backtrack..."

Three things get dropped: **the question**, **the finding**, and **the number**. So make each one
explicit state that the agent has to reconcile before it is allowed to finish.

    Ledger 1  QuestionContract  — what was asked (estimand, population, units, constraints)
    Ledger 2  FindingsLedger    — what was noticed, and what was DONE about it   <- the centrepiece
    Ledger 3  grounding         — every number traced back to real output        (see report.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ============================================================================================
# LEDGER 1 — the Question Contract
#
# Attacks: DDB *Constraint* failures (7.5%) and the melanoma scope-drop case study; GBP's
# "a statistically valid final model is applied to the wrong data or population, on the wrong
# scale, or on the wrong conceptual level."
#
# Why a structured object rather than "restate the question" in the system prompt: advice in a
# prompt DECAYS over a long trajectory — which is exactly the failure DDB documents, where a
# qualifier silently stops being applied three steps in. Structured state does not decay. It is
# re-rendered from a variable every turn, not remembered from a paragraph.
# ============================================================================================


class QuestionContract(BaseModel):
    """What am I actually being asked? Filled in before any analysis begins."""

    estimand: str = Field(description="The exact quantity to compute, in one sentence.")
    population: str = Field(description=(
        "WHICH ROWS. The denominator. Be explicit: all patients? one arm? "
        "deduplicated? This is the field most often got wrong."))
    units: str = Field(description="Units and scale of the answer (ng/mL? percent? a fraction?).")
    constraints: list[str] = Field(default_factory=list, description=(
        "Every explicit filter or limit the user stated. Copy them; do not paraphrase them away."))
    premises: list[str] = Field(default_factory=list, description=(
        "FACTUAL CLAIMS THE QUESTION TAKES FOR GRANTED, which you must CHECK against the data "
        "before answering. e.g. 'there are four sites', 'women respond better than men', 'the "
        "trial was randomised'. A question can be built on a false premise. If one is false, say "
        "so — do NOT answer the question as if it were true."))
    question_is_precise: bool = Field(description=(
        "Is there exactly ONE reasonable reading of this question — one quantity, over one set of "
        "rows, in one unit? Answer FALSE if a competent analyst could reasonably compute two "
        "different numbers from it (e.g. 'did it improve?' — for whom? by which arm? and is up or "
        "down 'better'?). If FALSE you MUST list every reasonable reading in `ambiguities` and say "
        "which one you chose."))
    ambiguities: list[str] = Field(default_factory=list, description=(
        "Each reasonable reading of the question, and which one you chose and why."))

    @model_validator(mode="after")
    def _ambiguity_must_be_declared(self):
        # A field the model can leave empty is a field the model WILL leave empty. On the
        # ambiguous task ("did the biomarker improve?") the agent left `ambiguities` blank every
        # single time and silently picked a reading — which is precisely the failure the field
        # was supposed to prevent.
        #
        # So force the judgement instead of inviting it. A required boolean cannot be skipped,
        # and once it is False the list cannot be empty. Same trick as the Findings Ledger: make
        # the omission impossible to express, rather than asking nicely for it not to happen.
        if not self.question_is_precise and not self.ambiguities:
            raise ValueError(
                "You said the question is NOT precise, so you must list the possible readings in "
                "`ambiguities` and state which one you chose.")
        return self

    def render(self) -> str:
        lines = [
            "══ QUESTION CONTRACT (agreed at step 0 — check every step against this) ══",
            f"  estimand   : {self.estimand}",
            f"  population : {self.population}",
            f"  units      : {self.units}",
        ]
        if self.constraints:
            lines.append("  constraints:")
            lines += [f"    - {c}" for c in self.constraints]
        if self.premises:
            lines.append("  premises TO VERIFY (a false premise means: refuse, don't answer):")
            lines += [f"    - {p}" for p in self.premises]
        if not self.question_is_precise:
            lines.append("  ⚠ QUESTION IS NOT PRECISE — you must state your interpretation in the")
            lines.append("    final answer's caveats, not just pick one silently:")
            lines += [f"    - {a}" for a in self.ambiguities]
        elif self.ambiguities:
            lines.append("  ambiguities (stated interpretation):")
            lines += [f"    - {a}" for a in self.ambiguities]
        return "\n".join(lines)


# ============================================================================================
# LEDGER 2 — the Findings Ledger.  THE CENTREPIECE.
#
# Attacks: the notice-act gap (GeneBench-Pro's headline finding) and DDB's *Domain reasoning*
# failures (54.0% — the largest category by far).
#
# The mechanism, in one sentence:
#
#     A noticed finding is an OPEN OBLIGATION, and submit_answer is blocked while any obligation
#     is open.
#
# GBP's finding is NOT "the model fails to notice." The model notices. It runs describe(), it sees
# the -999s. The failure is that noticing has no consequences — it cleans the column and proceeds
# with the analysis it had already decided on. The observation never reaches the decision.
#
# So: give noticing consequences. To close a finding the agent must either ACT on it (naming the
# code step that handled it) or DISMISS it (writing why it doesn't affect the estimand). Both are
# recorded; both ship in the final report. The `implication` field is where propagation actually
# happens — it forces the agent to write down WHAT THIS CHANGES, which is precisely the step the
# papers watch it skip.
#
# HONEST LIMITATION: this converts noticed-but-ignored into a hard stop. It does nothing about
# never-noticed. That is why observe.briefing() front-loads the mechanical diagnostics — so the
# common findings are HANDED to the agent rather than left to its curiosity. Together the coverage
# is good. It is not complete, and I would not claim otherwise.
# ============================================================================================

Status = Literal["open", "acted", "dismissed"]


@dataclass
class Finding:
    observation: str        # what I saw in the data
    implication: str        # what it CHANGES about the analysis   <- the load-bearing field
    status: Status = "open"
    resolution: str = ""    # how I acted on it, or why I dismissed it
    step: int = 0

    def render(self) -> str:
        icon = {"open": "🔴 OPEN", "acted": "✅ ACTED", "dismissed": "⚪ DISMISSED"}[self.status]
        s = f"  [{icon}] {self.observation}\n           → implication: {self.implication}"
        if self.resolution:
            s += f"\n           → resolution: {self.resolution}"
        return s


@dataclass
class FindingsLedger:
    findings: list[Finding] = field(default_factory=list)

    def note(self, observation: str, implication: str, step: int = 0) -> str:
        # Don't let the model log the same thing twice and inflate its own obligations.
        #
        # This matters more than it looks. The ledger is PRE-SEEDED with the findings a script
        # can detect, and the agent's instinct is to re-describe them in its own words rather
        # than resolve the ones already there. That doubles every obligation and burns the step
        # budget. So: match on substance, not on phrasing.
        dup = self._find_duplicate(observation)
        if dup:
            i, f = dup
            return (f"This is already finding #{i} (currently {f.status.upper()}): "
                    f"\"{f.observation[:70]}...\"\n"
                    f"Do not re-log it. Call resolve_finding(index={i}, ...) instead.")
        self.findings.append(Finding(observation, implication, "open", step=step))
        return (f"Finding #{len(self.findings)} logged as OPEN.\n"
                f"You cannot submit an answer while any finding is open. Either act on it "
                f"(then resolve_finding with what you did) or dismiss it (with a reason why it "
                f"does not affect the estimand).")

    def _find_duplicate(self, observation: str) -> tuple[int, Finding] | None:
        """Same substance, different words? Match on the concrete things a finding is *about*:
        the column names it mentions (in backticks or snake_case) and the distinctive numbers."""
        import re

        def signature(text: str) -> tuple[set[str], set[str]]:
            t = text.lower()
            cols = set(re.findall(r"[a-z_]+_[a-z_]+", t))         # snake_case identifiers
            nums = {n for n in re.findall(r"-?\d+\.?\d*", t) if len(n) > 1 and n not in ("11",)}
            return cols, nums

        new_cols, new_nums = signature(observation)
        if not new_cols and not new_nums:
            return None
        for i, f in enumerate(self.findings, 1):
            cols, nums = signature(f.observation)
            shares_column = bool(new_cols & cols)
            shares_number = bool(new_nums & nums)
            if shares_column and shares_number:
                return i, f
        return None

    def resolve(self, index: int, status: Status, resolution: str) -> str:
        if not 1 <= index <= len(self.findings):
            return f"No finding #{index}. There are {len(self.findings)}."
        f = self.findings[index - 1]
        f.status, f.resolution = status, resolution
        # A pre-seeded finding carries a placeholder implication ("TO BE DETERMINED"). Once the
        # agent has resolved it, that placeholder is just noise in the audit trail — and the audit
        # trail is the thing a scientist actually reads. Replace it with what the agent decided.
        if f.implication.startswith("TO BE DETERMINED"):
            f.implication = resolution
        n = len(self.open())
        return (f"Finding #{index} marked {status.upper()}. "
                + (f"{n} finding(s) still open." if n else "No findings left open — you may submit."))

    def open(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "open"]

    def render(self) -> str:
        if not self.findings:
            return ("══ FINDINGS LEDGER ══\n  (empty — have you actually looked at the data? "
                    "Log anything you notice with note_finding.)")
        lines = ["══ FINDINGS LEDGER (submitting is BLOCKED while any finding is OPEN) ══"]
        lines += [f"#{i}. {f.render()[2:]}" for i, f in enumerate(self.findings, 1)]
        n = len(self.open())
        lines.append(f"  → {n} open, {len(self.findings) - n} resolved")
        if n:
            seeded_open = [i for i, f in enumerate(self.findings, 1)
                           if f.status == "open" and f.observation.startswith("[auto-detected")]
            if seeded_open:
                lines.append(f"  → findings {seeded_open} were PRE-REGISTERED by automatic checks. "
                             f"Resolve them with resolve_finding(index=...). Do NOT re-log them "
                             f"in your own words — that just creates more work for you.")
        return "\n".join(lines)

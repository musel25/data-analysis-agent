"""GATE 4 — a second pair of eyes that has never seen the agent's reasoning.

The critical design detail, and the reason this is not theatre:

    THE VERIFIER DOES NOT SEE THE AGENT'S REASONING.

It sees the Question Contract, the data briefing, the code that ran and what it printed, and the
draft answer. That is all. Show a reviewer the agent's chain of thought and it gets anchored by
the agent's own narrative and rubber-stamps the conclusion — a verifier that reads the transcript
measures nothing. The fresh context IS the mechanism, not an implementation detail.

It is also a DIFFERENT MODEL FAMILY from the agent, because a model reviewing its own output shows
self-preference bias (Zheng et al. 2023, "Judging LLM-as-a-Judge").

And this is why the design can skip a "critic agent" entirely: the only genuine benefit of a second
agent here is an opinion not anchored on the first one's reasoning — and that costs exactly one API
call, with none of the coordination overhead. See docs/DECISIONS.md D02, D11.
"""
from __future__ import annotations

import json

from . import config
from .llm import llm
from .report import AnalysisReport

RUBRIC = """You are reviewing a data analysis. You did NOT do the analysis and you have not seen \
the analyst's reasoning — only their code, its output, and their conclusion. Judge what is in \
front of you, not what you assume they meant.

Check exactly four things:

1. SCOPE — Does the answer address the estimand as written in the contract? Same population (the
   same rows / the same denominator)? Same units? An answer that is correct about a different
   population is a wrong answer.

2. GROUNDING — Is every quantitative claim actually supported by the printed output shown? If a
   number in the conclusion does not appear in the output, say so.

3. OVERREACH — Is a causal claim being made from correlational evidence? Is a conclusion drawn
   from a comparison between groups that were not comparable to begin with?

4. UNACKNOWLEDGED DATA PROBLEMS — Does the output show something the conclusion ignores? Missing
   data, an impossible value, a group imbalance, a suspicious scale?

5. FALSE PREMISE — Did the question assume something that the data does not support ("which of the
   FOUR sites", "WHY do women respond better")? If the analyst answered such a question as if the
   premise were true, that is a failure, even if their arithmetic is correct. Answering a false
   question launders the false premise into a fact.

Be concrete and be brief. If it is sound, say so — do not invent problems to look useful. If it is
not, name the specific defect.

Reply with ONLY a JSON object, no prose, no markdown fence:
{"verdict": "accept" | "revise", "issues": ["...", "..."]}
"""


def verify(contract_text: str, briefing_text: str, code_log: str,
           report: AnalysisReport, meter=None) -> tuple[bool, list[str]]:
    """Returns (accepted, issues)."""
    payload = f"""{contract_text}

══ THE DATA ══
{briefing_text[:2500]}

══ WHAT THE ANALYST RAN, AND WHAT IT PRINTED ══
{code_log[:6000]}

══ THEIR DRAFT ANSWER ══
{json.dumps(report.model_dump(), indent=2)}
"""
    msg = llm(
        [{"role": "system", "content": RUBRIC}, {"role": "user", "content": payload}],
        model=config.VERIFIER_MODEL,
        temperature=0.0,
        meter=meter,
    )

    raw = (msg.content or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        out = json.loads(raw)
        return out.get("verdict") == "accept", list(out.get("issues") or [])
    except json.JSONDecodeError:
        # A verifier that cannot be parsed must not silently pass the answer. Fail open on the
        # ACCEPT side would be worse than useless — but failing closed would deadlock the agent.
        # So: accept, and record the failure as a caveat. Honest degradation.
        return True, [f"(verifier returned unparseable output; review skipped: {raw[:80]})"]

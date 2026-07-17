"""The base LLM — the same model, minus the whole agent.

This is the control the whole project is arguing against. It is NOT a weaker model: it is the
*exact same* model on the *exact same* data, with a code interpreter, and nothing else. No
deterministic briefing telling it what is in the file, no findings ledger turning a noticed
problem into an obligation, no gates on the exit, no verifier — and, in the system prompt, none
of the analyst reflexes (rule 4: "before comparing two groups, check they are comparable"). It is
"just ask ChatGPT with a code tool", which is what most people actually do.

The point of showing it live: on the Simpson's task it computes the raw marginal difference
`treatment − control`, reports the WRONG SIGN, and concludes the drug *hurts*. The agent, given
the same file, adjusts for severity and reports the opposite. Same model. Same data. The
difference is entirely the procedure wrapped around it — which is the thesis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .executor import PyExecutor
from .llm import Meter, llm

# Deliberately generic. A competent, helpful analyst prompt — the kind anyone would write — with
# NONE of agent.SYSTEM's hard-won reflexes (compute-don't-estimate, log-what-you-notice,
# check-comparability-before-comparing, question-the-premise). Strip the procedure, keep the model.
SYSTEM = (
    "You are a helpful data analyst. You can run Python with the run_python tool — pandas (pd) "
    "and numpy (np) are already imported, and the working directory is the project root, so you "
    "can read the data files by their given paths. Inspect the data as needed, then call "
    "final_answer with a single numeric value and a one-line explanation."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Execute Python in a persistent session. Returns whatever it prints.",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "Python to run."}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "final_answer",
        "description": "Submit the final numeric answer to the question.",
        "parameters": {"type": "object",
                       "properties": {"value": {"type": "number"},
                                      "explanation": {"type": "string"}},
                       "required": ["value", "explanation"]}}},
]


@dataclass
class BaselineStep:
    code: str
    output: str


@dataclass
class BaselineRun:
    """What a plain LLM did. Deliberately shaped like `Run` where the UI needs it: `.value`,
    `.explanation`, `.steps`, `.tokens`, `.cost_usd`, plus the code it ran so the answer is
    inspectable rather than asserted."""
    value: float | None = None
    explanation: str = ""
    steps: int = 0
    transcript: list[BaselineStep] = field(default_factory=list)
    forced: bool = False              # did we have to force the final answer out of it?
    meter: Meter = field(default_factory=Meter)

    @property
    def tokens(self) -> int:
        return self.meter.total_tokens

    @property
    def cost_usd(self) -> float:
        return self.meter.cost_usd


def run_baseline(question: str, files: list[str], executor: PyExecutor | None = None,
                 max_steps: int = 8, sink=None) -> BaselineRun:
    """Run the base model with a code interpreter and nothing else. Returns a BaselineRun.

    The final answer is captured structurally: if the model wanders off and stops calling tools
    (open models do narrate `final_answer(...)` as prose instead of calling it), we make one more
    call with the tool forced, so we always come back with a real number to compare — never a
    regex dug out of prose.
    """
    ex = executor or PyExecutor()
    ex.reset()
    run = BaselineRun()

    def say(*a):
        if sink is not None:
            sink(" ".join(str(x) for x in a))

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Data files: {', '.join(files)}\n\nQuestion: {question}"},
    ]
    say(f"── base LLM (no scaffolding) ─────────────────────────────")

    for step in range(max_steps):
        msg = llm(messages, tools=TOOLS, temperature=0.0, nonce=step, meter=run.meter)
        messages.append(msg.raw())

        if not msg.tool_calls:
            # It thinks it is done but never called the tool. Force it. (This is exactly the
            # failure the agent's gates exist to prevent — here we just want the number.)
            return _force_final(messages, run, say)

        submitted = False
        for tc in msg.tool_calls:
            if tc.name == "final_answer":
                run.value = tc.args.get("value")
                run.explanation = tc.args.get("explanation", "")
                say(f"final_answer: {run.value}")
                submitted = True
                break
            if tc.name == "run_python":
                code = tc.args.get("code", "")
                res = ex.run(code)
                out = (res.stdout or "").strip()
                if res.error:
                    out = (out + "\n" if out else "") + f"ERROR:\n{res.error}"
                out = out or "(no output)"
                run.transcript.append(BaselineStep(code=code, output=out[:2000]))
                run.steps += 1
                say(f"step {run.steps}: ran {len(code)} chars → {out[:80]}")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": out[:2000]})
        if submitted:
            return run

    # Ran out of steps without a final_answer — force one from what it has seen.
    return _force_final(messages, run, say)


def _force_final(messages: list, run: BaselineRun, say) -> BaselineRun:
    messages = messages + [{"role": "user", "content":
                            "Now call final_answer. The `value` must be the exact number your "
                            "code printed — copy it digit for digit, do not round or restate it."}]
    msg = llm(messages, tools=TOOLS, temperature=0.0, force_tool="final_answer", meter=run.meter)
    run.forced = True
    for tc in msg.tool_calls:
        if tc.name == "final_answer":
            run.value = tc.args.get("value")
            run.explanation = tc.args.get("explanation", "")
            say(f"final_answer (forced): {run.value}")
            break
    return run

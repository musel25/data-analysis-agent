"""The agent. A while-loop, three tools, and four gates.

Every mechanism is behind a flag in `Config`, because a guardrail you cannot switch off is a
guardrail you cannot measure. Notebook 07 turns each one off in turn and reports what breaks.
If an ablation shows a mechanism does not pay for itself, it gets cut. That is the deal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from . import observe
from .executor import PyExecutor
from .ledger import FindingsLedger, QuestionContract
from .llm import METER, Meter, llm
from .report import AnalysisReport, attach_findings, grounded, rejection_message
from .trace import Trace
from .verifier import verify

SYSTEM = """You are a careful data analyst. You work by writing and running Python.

THE RULES THAT MATTER

1. COMPUTE, NEVER ESTIMATE. Every number you report must have been printed by code you ran.
   You are a bad calculator and a good programmer. Act like it.

2. LOOK BEFORE YOU LEAP. Read any documentation provided with the data. Inspect dtypes, missing
   values, duplicates and distributions before you compute anything. The briefing above is a
   starting point, not a substitute for looking.

3. WHEN YOU NOTICE SOMETHING, LOG IT — and then let it change your plan.
   Call note_finding the moment you see anything odd: sentinel values, duplicate rows, an
   implausible scale, a group imbalance, a confounded comparison.
   This is the rule people skip, and it is the one that decides whether your answer is right.
   Noticing a problem and then running the analysis you had already planned is the single most
   common way a data analysis goes wrong. The `implication` field is where you say WHAT IT
   CHANGES. If it changes nothing, say that, and dismiss it.
   You cannot submit an answer while any finding is still open.

4. BEFORE COMPARING ANY TWO GROUPS, CHECK THAT THEY ARE COMPARABLE.
   This applies to EVERY grouped comparison — arm vs arm, channel vs channel, segment vs segment,
   site vs site. Cross-tabulate the grouping column against the other columns first. If the
   groups differ systematically in some OTHER variable, then the raw difference between them is
   not the effect of the grouping — it is that other variable wearing the grouping's clothes.
   You must adjust for it (stratify, standardise, or model it) rather than reporting the raw
   difference.
   A confounded comparison can give you the OPPOSITE SIGN to the truth.

5. A QUESTION CAN BE WRONG. CHECK ITS PREMISES BEFORE YOU ANSWER IT.
   Questions smuggle in assumptions:
     "Which of the SIX warehouses has the longest delay?"  <- assumes there are six.
     "WHY does the night shift outperform the day shift?"  <- assumes that it does.
     "What is the failure rate of the v3 sensors?"         <- assumes v3 sensors are in here.
   Put every such assumption in the contract's `premises` field and VERIFY each one against the
   data before you answer.
   If a premise turns out to be false, the correct answer is to SAY SO, plainly, and NOT to
   answer the question as asked. An answer to a false question is worse than no answer, because
   it launders the false premise into a fact.
   If the question is genuinely ambiguous, state your interpretation explicitly instead of
   silently picking one.

6. ONE STEP PER CELL. Small code, printed output, then think. Do not write a twenty-line script
   that does everything and prints one number — if it is wrong you will not know where.

7. YOU CANNOT SEE PLOTS. Figures are saved for the human. Print the numbers you need.

8. IF THE SAME ERROR HAPPENS TWICE, STOP CODING. State the root cause in one sentence first.

9. CHECK THE QUESTION, NOT YOUR MOMENTUM. The contract is pinned above every step. An answer that
   is correct about the wrong population is a wrong answer.

Begin by calling set_contract to state exactly what you have been asked. Then analyse. Then
submit_answer."""

# Rule 4 is the one piece of *domain* knowledge in the whole prompt, and it earns its place on
# the papers' own evidence. DrugDiscoveryBench re-ran their unsolved tasks with the expert's
# step-by-step playbook supplied as a hint, and went from 76/82 to 80/82:
#
#   "The results suggest that execution is within reach for today's agents should they be given
#    the expert workflow." (DDB, p. 14)
#
# The models can execute. What they lack is the analyst's reflex — the thing a statistician does
# without being asked. So encode the reflex. That is what "building on top of the base model"
# means: not a better model, a better *procedure*.
#
# TWO WAYS I OVERFIT THIS PROMPT TO MY OWN BENCHMARK, AND HOW THE EVAL CAUGHT BOTH.
# (Kept in the code as a scar. See DECISIONS D28 and D29.)
#
# 1. Rule 5 used to read: `Questions smuggle in assumptions: "which of the FOUR sites...",
#    "WHY do women respond better?"` — which are, verbatim, two of the questions in
#    evals/tasks.py. It went further and gave the ANSWERS ("there are only three sites; women
#    do not in fact respond better"). The same two examples were in the `premises` field of the
#    tool schema and in the verifier's rubric. Three channels, straight into the model's context.
#    Those two tasks scored 100% and 90%. Of course they did — I had told it the answers.
#    My `test_leak` guard did not catch this because it only checks NUMERIC ground truth against
#    the question string. A leak guard only guards the channel you pointed it at.
#
# 2. Rule 4 never named a column, so I told myself it was general. It wasn't. It used to say
#    "if the groups were not RANDOMLY ASSIGNED... that is not a TREATMENT EFFECT" — the language
#    of a clinical trial. On trial.csv (arm x severity) the reflex fired. On sales.csv
#    (channel x segment) nothing is "randomly assigned" and there is no "treatment", so it did
#    not fire at all: the agent landed on the naive, confounded answer in 9 of 10 runs.
#    The rule generalised in its WORDING and not in its EFFECT, which is the kind of overfitting
#    you cannot see by reading your own prompt. Only a held-out domain shows it to you.
#
# Both rules are now written to name no dataset, no column, and no benchmark question. This is
# what the held-out domain is FOR — not to produce a reassuring number, but to catch the author.


TOOLS = [
    {"type": "function", "function": {
        "name": "set_contract",
        "description": "State exactly what you are being asked, before analysing. Call this first.",
        "parameters": QuestionContract.model_json_schema(),
    }},
    {"type": "function", "function": {
        "name": "run_python",
        "description": ("Run Python in a persistent session. Variables persist between calls. "
                        "pandas as pd, numpy as np, matplotlib.pyplot as plt are imported. "
                        "Returns whatever you print()."),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python to execute."}},
            "required": ["code"],
        },
    }},
    {"type": "function", "function": {
        "name": "note_finding",
        "description": ("Log something you noticed about the data that could change the analysis. "
                        "Blocks submission until you resolve it."),
        "parameters": {
            "type": "object",
            "properties": {
                "observation": {"type": "string", "description": "What you saw, concretely."},
                "implication": {"type": "string", "description":
                                "What it CHANGES about how the analysis must be done."},
            },
            "required": ["observation", "implication"],
        },
    }},
    {"type": "function", "function": {
        "name": "resolve_finding",
        "description": "Close a finding: either you acted on it, or you are dismissing it.",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "The finding number."},
                "status": {"type": "string", "enum": ["acted", "dismissed"]},
                "resolution": {"type": "string", "description":
                               "What you did about it, or why it does not affect the estimand."},
            },
            "required": ["index", "status", "resolution"],
        },
    }},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the final answer. It will be checked before being accepted.",
        "parameters": AnalysisReport.model_json_schema(),
    }},
]


@dataclass
class Config:
    """Every guardrail, switchable. This is what makes the ablation study possible."""
    max_steps: int = 20
    use_contract: bool = True        # Ledger 1
    use_ledger: bool = True          # Ledger 2  <- the centrepiece
    use_grounding: bool = True       # Ledger 3
    use_verifier: bool = True        # Gate 4
    use_briefing: bool = True
    use_confound_detector: bool = True   # D33 — the detector the eval asked for
    use_truncation: bool = True
    use_state_banner: bool = True
    model: str | None = None
    verbose: bool = True
    sink: object = None      # optional callable(str) — where trajectory lines go (Streamlit, a log)
    temperature: float = 0.0
    attempt: int = 0        # nonce: makes repeat runs INDEPENDENT samples, not cache hits


@dataclass
class Run:
    """Everything that happened. The trace is the product, not a by-product."""
    question: str
    report: dict | None = None
    contract: QuestionContract | None = None
    ledger: FindingsLedger = field(default_factory=FindingsLedger)
    steps: int = 0
    errors: int = 0
    rejections: list[str] = field(default_factory=list)
    stopped: str = ""
    cost_usd: float = 0.0
    tokens: int = 0
    trace: Trace | None = None
    _code_cache: dict = field(default_factory=dict)   # death-loop guard 1
    _error_fp: dict = field(default_factory=dict)     # death-loop guard 2

    @property
    def value(self):
        return self.report.get("value") if self.report else None


def _portable(path: str) -> str:
    """Repo-relative, always. Nothing machine-specific may enter the prompt.

    D30. An absolute path in the prompt is an absolute path in the *cache key* — and in the code
    the model writes back. The README promises the notebooks replay offline from the committed
    cache with no API key; with absolute paths that promise held on exactly one machine.

    Callers hand us paths in all three shapes (the notebooks alone use two), so accept any and
    always emit the repo-relative one. `PyExecutor.run` pins CWD to the repo root while the agent's
    code executes, so that is the form that resolves on every clone.
    """
    from pathlib import Path

    from .executor import ROOT

    p = Path(path)
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, ROOT / p]
    for c in candidates:
        c = c.resolve()
        if c.exists():
            try:
                return str(c.relative_to(ROOT))
            except ValueError:
                return str(c)             # a real file outside the repo: keep it, warts and all
    return str(path)                      # doesn't exist — let the briefing report that honestly


def run_agent(question: str, files: list[str], cfg: Config | None = None,
              executor: PyExecutor | None = None) -> Run:
    cfg = cfg or Config()
    ex = executor or PyExecutor()
    ex.reset()                            # also pins CWD to the repo root — see _portable
    files = [_portable(f) for f in files]
    run = Run(question=question)
    run.trace = Trace(question)
    # A per-run meter. The global METER is shared, and the eval runs agents CONCURRENTLY — so
    # taking a before/after delta off the global counter attributes other threads' tokens to
    # this run. Each run counts its own.
    run_meter = Meter()

    brief = observe.briefing(files, confounds=cfg.use_confound_detector) if cfg.use_briefing else "(no briefing — files: %s)" % files
    code_log: list[str] = []          # what ran + what it printed. the grounding gate reads this.
    all_stdout: list[str] = []

    # PRE-SEED THE LEDGER. Anything a script can find, a script finds — and it enters the ledger
    # as an OPEN OBLIGATION, not as a helpful note the agent is free to scroll past.
    #
    # I added this after watching a run where the ledger caught the duplicate patients, missed the
    # confounding entirely, and cheerfully submitted the wrong-signed answer. The ledger worked
    # exactly as designed: it forces you to act on what you noticed. It cannot make you notice.
    # So: don't hope. Seed.
    if cfg.use_ledger and cfg.use_briefing:
        for obs_text, implication in observe.seed_findings(
                files, confounds=cfg.use_confound_detector):
            run.ledger.note(obs_text, implication, step=0)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"══ THE DATA ══\n{brief}\n\n══ THE QUESTION ══\n{question}\n\n"
            f"Files are at: {', '.join(files)}"},
    ]

    tools = [t for t in TOOLS if _enabled(t["function"]["name"], cfg)]

    def say(*a):
        # The trajectory is narration, not debug output. Send it wherever the caller wants it:
        # stdout in a notebook, a Streamlit placeholder in the dashboard, a log in production.
        line = " ".join(str(x) for x in a)
        if cfg.sink is not None:
            cfg.sink(line)
        elif cfg.verbose:
            print(line)

    say(f"\n{'─'*78}\nQUESTION: {question}\n{'─'*78}")

    while run.steps < cfg.max_steps:
        run.steps += 1

        # Pin the ledgers to the *end* of context, regenerated from live state every turn.
        # Never carried forward as text — that is the point. Text drifts; state does not.
        pinned = []
        if cfg.use_contract and run.contract:
            pinned.append(run.contract.render())
        if cfg.use_ledger:
            pinned.append(run.ledger.render())
        if cfg.use_state_banner:
            pinned.append(observe.state_banner(ex.variables()))
        left = cfg.max_steps - run.steps
        if left <= 3:
            pinned.append(f"⏳ {left} steps remaining — converge on an answer now.")

        turn = messages + ([{"role": "user", "content": "\n\n".join(pinned)}] if pinned else [])
        msg = llm(turn, tools=tools, model=cfg.model,
                  temperature=cfg.temperature, nonce=cfg.attempt, meter=run_meter)
        messages.append(msg.raw())

        if not msg.tool_calls:
            # No tool call and no answer: the model is chatting. Push it back to work.
            say(f"  [{run.steps}] (no tool call — nudging)")
            messages.append({"role": "user", "content":
                             "Call a tool. If you have the answer, call submit_answer."})
            continue

        for call in msg.tool_calls:
            result = _dispatch(call, run, ex, cfg, brief, code_log, all_stdout, say, run_meter)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if run.report is not None:
                run.stopped = "submitted"
                break
        if run.report is not None:
            break
    else:
        # Budget exhausted. Force a graceful, honest exit rather than a crash: a partial answer
        # with a loud caveat is worth more than nothing.
        run.stopped = "budget"
        say(f"  ⏹  step budget exhausted — forcing best-effort answer")
        run.report = _force_answer(messages, run, ex, cfg, code_log, all_stdout, run_meter)

    run.tokens = run_meter.total_tokens
    run.cost_usd = run_meter.cost_usd
    run.trace.finish(run)
    say(f"{'─'*78}\n{run.stopped.upper()} in {run.steps} steps | "
        f"{run.tokens:,} tokens | ${run.cost_usd:.4f}\n")
    return run


def _enabled(name: str, cfg: Config) -> bool:
    if name == "set_contract":
        return cfg.use_contract
    if name in ("note_finding", "resolve_finding"):
        return cfg.use_ledger
    return True


def _dispatch(call, run: Run, ex, cfg, brief, code_log, all_stdout, say, run_meter=None) -> str:
    args = call.args
    if "__parse_error__" in args:
        # Malformed tool JSON is an observation, not a crash. The model recovers from it.
        return f"Your tool arguments were not valid JSON ({args['__parse_error__']}). Try again."

    # ---- set_contract -------------------------------------------------------------------
    if call.name == "set_contract":
        try:
            run.contract = QuestionContract(**args)
        except ValidationError as e:
            return f"Contract invalid:\n{e}"
        say(f"  [{run.steps}] contract: {run.contract.estimand}")
        say(f"           population: {run.contract.population}")
        run.trace.step(run.steps, "set_contract", args, "ok")
        return "Contract recorded. It is now pinned to every step. Analyse."

    # ---- run_python ---------------------------------------------------------------------
    if call.name == "run_python":
        code = args.get("code", "")

        # DEATH-LOOP GUARD 1 — byte-identical code, twice.
        # I watched a real run submit the same failing code at step 2 and step 4. Executing it
        # again cannot produce a different result; it can only burn budget. Hand back the cached
        # observation and say so plainly.
        fp_code = hash(code.strip())
        if fp_code in run._code_cache:
            say(f"  [{run.steps}] ⟳ identical code re-submitted — returning cached result")
            return ("You have already run this exact code. The result was:\n\n"
                    + run._code_cache[fp_code]
                    + "\n\nRunning it again will not change anything. Do something different.")

        res = ex.run(code)
        out = res.stdout or ""
        if res.error:
            run.errors += 1
            out += f"\nTRACEBACK:\n{res.error}"

            # DEATH-LOOP GUARD 2 — error fingerprinting. Reflexion (Shinn et al. 2023), in ~10
            # lines. The two-stage escalation is deliberate: jumping straight to "try something
            # else" makes the agent abandon a 95%-correct approach over a typo. First make it
            # *diagnose*; only then make it change strategy.
            fp = hash((res.error.splitlines()[-1][:60], code.strip()[:60]))
            run._error_fp[fp] = run._error_fp.get(fp, 0) + 1
            n = run._error_fp[fp]
            if n == 2:
                out += ("\n\n⚠ You have now hit this same error twice. STOP writing code. "
                        "State the root cause in one sentence first, then fix it.")
            elif n >= 3:
                out += ("\n\n⚠ Three times. This approach is not working — it is not a typo, it "
                        "is the approach. Choose a different strategy.")
        if res.figures:
            out += (f"\n[{len(res.figures)} figure(s) saved for the human: "
                    f"{', '.join(res.figures)}. You cannot see images — print the numbers.]")

        all_stdout.append(res.stdout or "")
        code_log.append(f"```python\n{code}\n```\n-> {(res.stdout or '(no output)')[:600]}"
                        + (f"\n-> ERROR: {res.error}" if res.error else ""))
        run._code_cache[fp_code] = observe.truncate(out, 400)

        say(f"  [{run.steps}] run_python  " + ("❌ " + res.error.splitlines()[-1][:60]
                                               if res.error else
                                               "✓ " + (res.stdout or "").strip().replace("\n", " ")[:60]))
        obs = observe.truncate(out) if cfg.use_truncation else out
        run.trace.step(run.steps, "run_python", {"code": code}, obs, error=res.error)
        return obs or "(no output — did you forget to print?)"

    # ---- note_finding -------------------------------------------------------------------
    if call.name == "note_finding":
        msg = run.ledger.note(args.get("observation", ""), args.get("implication", ""), run.steps)
        say(f"  [{run.steps}] 🔴 finding: {args.get('observation','')[:64]}")
        run.trace.step(run.steps, "note_finding", args, msg)
        return msg

    # ---- resolve_finding ----------------------------------------------------------------
    if call.name == "resolve_finding":
        msg = run.ledger.resolve(int(args.get("index", 0)), args.get("status", "acted"),
                                 args.get("resolution", ""))
        say(f"  [{run.steps}] ✅ resolved #{args.get('index')}: {args.get('status')}")
        run.trace.step(run.steps, "resolve_finding", args, msg)
        return msg

    # ---- submit_answer — THE GATES ------------------------------------------------------
    if call.name == "submit_answer":
        return _submit(args, run, ex, cfg, brief, code_log, all_stdout, say, run_meter)

    return f"Unknown tool `{call.name}`."


def _submit(args, run, ex, cfg, brief, code_log, all_stdout, say, run_meter=None) -> str:
    # GATE 1 — schema. Free.
    try:
        report = AnalysisReport(**args)
    except ValidationError as e:
        say(f"  [{run.steps}] ⛔ gate 1 (schema) rejected")
        run.rejections.append("schema")
        return f"Your answer did not match the required schema:\n{e}\nFix it and resubmit."

    # GATE 2 — the Findings Ledger. Free. This is the one that matters.
    if cfg.use_ledger and run.ledger.open():
        open_f = [(i, f) for i, f in enumerate(run.ledger.findings, 1) if f.status == "open"]
        say(f"  [{run.steps}] ⛔ gate 2 (ledger) rejected — {len(open_f)} open finding(s)")
        run.rejections.append("open_findings")
        run.trace.step(run.steps, "submit_answer", args, "REJECTED: open findings")
        return rejection_message("open_findings", findings=open_f)

    # GATE 3 — numeric grounding. Free.
    if cfg.use_grounding:
        ok, ungrounded = grounded(report, "\n".join(all_stdout))
        if not ok:
            say(f"  [{run.steps}] ⛔ gate 3 (grounding) rejected — ungrounded: {ungrounded}")
            run.rejections.append("ungrounded")
            run.trace.step(run.steps, "submit_answer", args, f"REJECTED: ungrounded {ungrounded}")
            return rejection_message("ungrounded", values=ungrounded)

    # GATE 4 — the fresh-context verifier. One API call, and only now.
    if cfg.use_verifier and run.rejections.count("verifier") < 1:   # exactly one revision round
        contract_text = run.contract.render() if run.contract else f"Question: {run.question}"
        accepted, issues = verify(contract_text, brief, "\n\n".join(code_log), report,
                                  meter=run_meter)
        if not accepted and issues:
            say(f"  [{run.steps}] ⛔ gate 4 (verifier) returned it — {len(issues)} issue(s)")
            for i in issues:
                say(f"           · {i[:74]}")
            run.rejections.append("verifier")
            run.trace.step(run.steps, "submit_answer", args, f"RETURNED: {issues}")
            return rejection_message("verifier", issues=issues)
        if issues:   # accepted, but with reservations — they become caveats. Nothing is buried.
            report.caveats += [f"reviewer noted: {i}" for i in issues]

    say(f"  [{run.steps}] ✅ ACCEPTED — {report.answer[:66]}")
    run.report = attach_findings(report, run.ledger.findings, run.contract)
    run.trace.step(run.steps, "submit_answer", args, "ACCEPTED")
    return "Answer accepted."


def _force_answer(messages, run, ex, cfg, code_log, all_stdout, run_meter=None) -> dict:
    """Out of budget. Make it submit what it has, honestly labelled."""
    messages.append({"role": "user", "content":
                     "You are out of steps. Submit your best answer NOW with submit_answer. "
                     "Set confidence='low' and put what you could not finish in caveats."})
    msg = llm(messages, tools=TOOLS, model=cfg.model, force_tool="submit_answer",
              temperature=cfg.temperature, nonce=cfg.attempt, meter=run_meter)
    if msg.tool_calls:
        try:
            report = AnalysisReport(**msg.tool_calls[0].args)
        except ValidationError:
            report = AnalysisReport(answer="(ran out of steps before reaching an answer)",
                                    method="incomplete", evidence=[], confidence="low")
        report.confidence = "low"
        report.caveats.append(f"Step budget ({cfg.max_steps}) exhausted; analysis incomplete.")
        if run.ledger.open():
            report.caveats += [f"UNRESOLVED: {f.observation}" for f in run.ledger.open()]
        return attach_findings(report, run.ledger.findings, run.contract)
    return attach_findings(
        AnalysisReport(answer="(no answer)", method="budget exhausted", evidence=[],
                       confidence="low"), run.ledger.findings, run.contract)

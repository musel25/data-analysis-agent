"""The dashboard.

    uv run streamlit run app.py

Two tabs, and they are the two halves of the argument:

  RUN THE AGENT  — watch it work, and read its audit trail. The trail is the point: not
                   "the model said 0.15" but "the model saw the confounding, said so, adjusted
                   for it, and here is the step where it did."

  THE EVIDENCE   — 4,480 runs, 8 ablations, and the run where my own confidence interval lied.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from agentlib import Config, PyExecutor, run_agent, run_baseline
from agentlib import config as cfg_mod

ROOT = Path(__file__).parent

# Palette: validated with the dataviz skill's checker (blue/red, worst adjacent CVD ΔE 74.6).
BLUE, RED, GREY, GREEN, AMBER = "#2a78d6", "#e34948", "#adb5bd", "#1baf7a", "#eda100"

st.set_page_config(page_title="Data-analysis agent", page_icon="🔬", layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1250px; }}
  .lede {{ font-size: 1.05rem; color: #52514e; line-height: 1.55; }}
  .finding {{
      border-left: 4px solid {GREY}; padding: 0.55rem 0.9rem; margin: 0.45rem 0;
      background: rgba(127,127,127,0.06); border-radius: 4px;
  }}
  .acted     {{ border-left-color: {GREEN}; }}
  .dismissed {{ border-left-color: {GREY}; }}
  .open      {{ border-left-color: {RED}; }}
  .fmeta {{ font-size: 0.86rem; color: #6c757d; margin-top: 0.25rem; }}
  .gate {{ font-family: ui-monospace, monospace; font-size: 0.86rem; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════════
#  TAB 1 — run the agent
# ══════════════════════════════════════════════════════════════════════════════════════

def tab_run():
    st.subheader("Ask a question. Watch it keep the thread.")
    st.markdown(
        '<p class="lede">The agent writes and runs Python against the data. What matters is not '
        'the number it returns — it is <b>the record of what it noticed and what it did about '
        'it</b>. That audit trail is the deliverable.</p>',
        unsafe_allow_html=True)

    datasets = {
        "trial.csv — clinical trial (4 planted traps)": (
            ["data/trial.csv", "data/data_dictionary.md"],
            ["Does the treatment improve the response rate? Report the treatment effect as a "
             "difference in proportions (treatment minus control).",
             "What is the mean baseline biomarker value (ng/mL) across all patients?",
             "How many patients were enrolled in the trial?",
             "The trial ran at four sites. Which of the four had the highest response rate?",
             "Why do female patients respond better to the treatment than male patients?"],
        ),
        "penguins.csv — clean, no traps": (
            ["data/penguins.csv"],
            ["Which penguin species has the highest mean flipper length?",
             "What is the correlation between flipper length and body mass?"],
        ),
    }

    c1, c2 = st.columns([1, 1])
    with c1:
        ds = st.selectbox("Dataset", list(datasets), label_visibility="collapsed")
    files, examples = datasets[ds]
    with c2:
        preset = st.selectbox("Example question", examples, label_visibility="collapsed")

    question = st.text_area("Question", value=preset, height=80, label_visibility="collapsed")

    compare_base = st.toggle(
        "⚖️  Also ask a plain LLM (no scaffolding), and show the difference", True,
        help="The SAME model on the SAME data with a code interpreter — but none of the agent: "
             "no briefing, no ledger, no gates, and no analyst reflexes in the prompt. This is "
             "'just ask ChatGPT with a code tool'. On the confounded question it reports the raw "
             "treatment−control difference and gets the wrong sign.")

    with st.expander("⚙️  Guardrails — switch them off and watch it fail"):
        g1, g2, g3, g4, g5 = st.columns(5)
        use_briefing = g1.toggle("Data briefing", True, help="The DETECTOR. Removing this costs 27 points.")
        use_ledger = g2.toggle("Findings Ledger", True, help="A noticed problem becomes an open obligation.")
        use_contract = g3.toggle("Question Contract", True)
        use_grounding = g4.toggle("Grounding gate", True, help="Every number must appear in real output.")
        use_verifier = g5.toggle("Verifier", True, help="Fresh eyes, different model family.")

    if not st.button("Run the agent", type="primary"):
        st.info("Pick a question and hit **Run the agent**. A run takes 30–90 seconds."
                + (" The GPU scales to zero when idle, so the **first** run of the session also "
                   "pays a cold start (~1 min) while vLLM boots."
                   if cfg_mod.SELF_HOSTED else " It costs about half a cent."))
        return

    conf = Config(
        use_briefing=use_briefing, use_ledger=use_ledger, use_contract=use_contract,
        use_grounding=use_grounding, use_verifier=use_verifier, verbose=False,
    )

    st.markdown("##### The trajectory")
    box = st.empty()
    lines: list[str] = []

    def sink(line: str):
        if line.strip():
            lines.append(line.rstrip())
            box.code("\n".join(lines[-22:]), language="text")

    conf.sink = sink

    abs_files = [str(ROOT / f) for f in files]
    with st.spinner("thinking, writing code, running it, checking itself…"):
        try:
            run = run_agent(question, abs_files, conf, executor=PyExecutor())
        except Exception as e:                                    # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}")
            return

    base_run = None
    if compare_base:
        with st.spinner("now asking a plain LLM the same question, with no scaffolding…"):
            try:
                base_run = run_baseline(question, abs_files, executor=PyExecutor())
            except Exception as e:                                # noqa: BLE001
                st.warning(f"Base-LLM comparison failed: {type(e).__name__}: {e}")

    st.session_state["last_run"] = run
    _render_run(run, base_run)


def _render_run(run, base_run=None):
    r = run.report
    if not r:
        st.error("No answer produced.")
        return

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Answer", r.get("value") if r.get("value") is not None else "—")
    m2.metric("Confidence", r.get("confidence", "—"))
    m3.metric("Steps", run.steps)
    # On a self-hosted GPU there is no per-token price, so report what was actually measured —
    # the tokens — rather than multiplying them by a rate that does not exist.
    if cfg_mod.SELF_HOSTED:
        m4.metric("Tokens", f"{run.tokens:,}",
                  help="Self-hosted: the GPU is billed by the second, not by the token, so a "
                       "dollar figure here would be invented. This is the number actually measured.")
    else:
        m4.metric("Cost", f"${run.cost_usd:.4f}")

    st.markdown(f"### {r['answer']}")

    if base_run is not None:
        _render_baseline_comparison(base_run, r)

    # ── the audit trail — the whole point ──────────────────────────────────────────────
    findings = r.get("findings", [])
    if findings:
        st.markdown("##### 🔍 What it noticed, and what it did about it")
        st.caption(
            "This is the part neither paper's system produces. GeneBench-Pro's central finding is "
            "that models *notice* a data problem and then treat it as local cleanup — the "
            "observation never reaches the decision. Here, a noticed problem is an **open "
            "obligation**, and the agent cannot submit an answer until it has either acted on it "
            "or dismissed it in writing.")
        for f in findings:
            icon = {"acted": "✅ ACTED", "dismissed": "⚪ DISMISSED", "open": "🔴 OPEN"}[f["status"]]
            st.markdown(
                f'<div class="finding {f["status"]}"><b>{icon}</b> — {f["observation"]}'
                f'<div class="fmeta"><b>implication:</b> {f["implication"]}<br>'
                f'<b>resolution:</b> {f["resolution"] or "—"}</div></div>',
                unsafe_allow_html=True)

    # ── the gates ──────────────────────────────────────────────────────────────────────
    st.markdown("##### 🔒 The gates on the exit")
    st.caption("Finishing is an *action*, not a default. Four gates, cheapest first — never pay for "
               "an LLM call to catch what a regex would have caught.")
    names = {"schema": "1 · schema", "open_findings": "2 · findings ledger",
             "ungrounded": "3 · numeric grounding", "verifier": "4 · fresh-context verifier"}
    if run.rejections:
        for rej in run.rejections:
            st.markdown(f'<div class="gate">⛔ <b>{names.get(rej, rej)}</b> bounced the answer '
                        f'back into the loop</div>', unsafe_allow_html=True)
        st.caption("↳ the agent had to fix it and resubmit.")
    else:
        st.markdown('<div class="gate">✅ accepted first time — all four gates passed</div>',
                    unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Method")
        st.write(r["method"])
        if r.get("caveats"):
            st.markdown("##### Caveats")
            for c in r["caveats"]:
                st.markdown(f"- {c}")
    with c2:
        st.markdown("##### Evidence")
        st.caption("Every number here was **printed by code that actually ran** — checked by regex, "
                   "not by an LLM.")
        for e in r["evidence"]:
            st.markdown(f"- `{e}`")

    with st.expander("The full structured report (JSON)"):
        st.json(r)

    _render_debug(run)


def _render_baseline_comparison(base_run, agent_report):
    """The base LLM next to the agent. Same model, same data, same code interpreter — the only
    difference is the scaffolding. On the confounded question they reach OPPOSITE conclusions,
    which is the whole argument in one screen."""
    st.divider()
    st.markdown("##### ⚖️ The same model, without the agent")
    st.caption(
        "The panel on the left is the **exact same model** on the **exact same file**, with a "
        "Python interpreter — but none of the agent: no briefing telling it what's in the data, "
        "no findings ledger, no exit gates, and none of the analyst reflexes in its prompt. It is "
        "*'just ask an LLM with a code tool'*. What's missing is not intelligence; it's the "
        "procedure.")

    agent_val = agent_report.get("value")
    base_val = base_run.value
    # The SIGN is the honest signal here — a 4B will fumble the magnitude when it copies its own
    # printed number into a tool field, but the direction of its conclusion is stable.
    opposite = (isinstance(agent_val, (int, float)) and isinstance(base_val, (int, float))
                and (agent_val > 0) != (base_val > 0))

    def direction(v):
        if not isinstance(v, (int, float)):
            return "—"
        return "treatment **helps**" if v > 0 else ("treatment **hurts**" if v < 0 else "no effect")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="finding open"><b>🤖 A plain LLM + code interpreter</b>'
            '<div class="fmeta">no briefing · no ledger · no gates · no verifier</div></div>',
            unsafe_allow_html=True)
        st.markdown(f"**It concluded:** {direction(base_val)}")
        if base_run.explanation:
            st.caption(base_run.explanation)
        if base_run.transcript:
            with st.expander(f"The {len(base_run.transcript)} code cell(s) it ran"):
                for i, step in enumerate(base_run.transcript, 1):
                    st.markdown(f"**Cell {i}**")
                    st.code(step.code, language="python")
                    st.code(step.output, language="text")
    with c2:
        st.markdown(
            '<div class="finding acted"><b>🔬 The agent</b>'
            '<div class="fmeta">same model · same data · the full scaffolding</div></div>',
            unsafe_allow_html=True)
        st.markdown(f"**It concluded:** {direction(agent_val)}")
        st.caption(agent_report.get("method", ""))

    if opposite:
        st.error(
            "**Opposite conclusions, from the same model on the same data.** The plain LLM "
            "compared the two arms directly and reported the raw `treatment − control` difference "
            "— but treatment was given to sicker patients, so that comparison has the **wrong "
            "sign**. It never thought to stratify by severity. The agent's briefing surfaced the "
            "confounding, the ledger made it an obligation it couldn't skip, and it adjusted — "
            "landing the correct direction. This is Simpson's paradox, and it is exactly the "
            "notice–act gap the guardrails close.")
    else:
        st.info(
            "On this question the plain LLM did not visibly diverge. The gap is sharpest on the "
            "**confounded** question — *“Does the treatment improve the response rate?”* on "
            "`trial.csv` — where skipping the severity stratification flips the sign of the answer.")


def _render_debug(run):
    """Turn-by-turn X-ray: exactly what went into the model, what came out, and how long the
    API call took. This is where 'why is it slow' and 'why did it do that' both get answered."""
    dbg = getattr(run, "debug", None)
    if not dbg or not dbg.get("steps"):
        return

    st.divider()
    st.markdown("##### 🩻 Debug — what the model saw, step by step")

    steps = dbg["steps"]
    api_s = sum(s["llm"].get("duration_s", 0) for s in steps)
    slept = sum(s["llm"].get("slept_s", 0) for s in steps)
    retries = sum(s["llm"].get("retries", 0) for s in steps)
    thinking = sum(len(s.get("reasoning") or "") for s in steps)
    st.caption(f"{len(steps)} LLM calls · {api_s:.0f}s total in the API"
               + (f" · **{slept:.0f}s of that asleep in {retries} rate-limit retries**" if retries else "")
               + (f" · {thinking:,} chars of hidden thinking" if thinking else ""))

    with st.expander("📜 System prompt (constant, every turn)"):
        st.code(dbg["system"], language="text")
    with st.expander("📊 First user message — the briefing + the question (as the model gets it)"):
        st.code(dbg["first_user_message"], language="text")
    if run.contract:
        with st.expander("📋 Question Contract — every field"):
            st.json(run.contract.model_dump())
            st.markdown("**Rendered form, as pinned to the end of context each turn:**")
            st.code(run.contract.render(), language="text")

    for s in steps:
        names = ", ".join(t["name"] for t in s["tool_calls"]) or "no tool call"
        mt = s["llm"]
        label = f"step {s['step']} · {names}"
        if mt.get("cached"):
            label += " · cached"
        elif mt:
            label += f" · {mt.get('duration_s', 0):.1f}s · {mt.get('prompt_tokens', 0):,}→{mt.get('completion_tokens', 0):,} tok"
            if mt.get("retries"):
                label += f" · ⚠ slept {mt.get('slept_s', 0):.0f}s on retries"
        with st.expander(label):
            if s["pinned"]:
                st.markdown("**Pinned to the end of context this turn** (regenerated from live "
                            "state — this is what 'the ledger is pinned' actually means):")
                st.code(s["pinned"], language="text")
            if s.get("reasoning"):
                st.markdown("**Model thinking** (hidden from the transcript, returned by the API):")
                st.code(s["reasoning"], language="text")
            if s.get("content"):
                st.markdown("**Model text:**")
                st.write(s["content"])
            for tc, res in zip(s["tool_calls"], s["results"] + [""] * len(s["tool_calls"])):
                st.markdown(f"**→ `{tc['name']}`**")
                if tc["name"] == "run_python":
                    st.code(tc["args"].get("code", ""), language="python")
                else:
                    st.json(tc["args"])
                if res:
                    st.markdown("**← result:**")
                    st.code(res, language="text")


# ══════════════════════════════════════════════════════════════════════════════════════
#  TAB 2 — the evidence
# ══════════════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_results():
    from evals.tasks import TASKS
    df = pd.read_json(ROOT / "evals/results.jsonl", lines=True)
    df["domain"] = df.task_id.map({t.id: t.domain for t in TASKS})
    return df


@st.cache_data
def load_stats(_df):
    from evals.stats import ablation_table, hierarchical_bootstrap
    abl = ablation_table(_df)
    doms = {d: hierarchical_bootstrap(_df[(_df.config == "full") & (_df.domain == d)])
            for d in _df.domain.dropna().unique()}
    return abl, doms


def bar(df, value_col, title, highlight="full", fmt="{:.0%}"):
    """A ranked bar chart. One series → no legend; the title names it. `full` is the reference,
    so it gets the accent and everything else recedes to grey (colour follows the entity, not
    its rank)."""
    d = df.sort_values(value_col, ascending=False)
    st.markdown(f"**{title}**")
    for cfg_name, v in d[value_col].items():
        colour = BLUE if cfg_name == highlight else (RED if cfg_name in
                                                     ("no_briefing", "no_guardrails") else GREY)
        pct = max(v, 0.004)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:3px 0;">'
            f'<div style="width:132px;font-size:0.86rem;color:#52514e;font-family:ui-monospace,monospace;">'
            f'{cfg_name}</div>'
            f'<div style="flex:1;background:rgba(127,127,127,0.10);border-radius:4px;height:19px;">'
            f'<div style="width:{pct*100:.1f}%;background:{colour};height:19px;border-radius:4px;"></div>'
            f'</div>'
            f'<div style="width:52px;text-align:right;font-size:0.86rem;font-variant-numeric:tabular-nums;">'
            f'{fmt.format(v)}</div></div>',
            unsafe_allow_html=True)


def tab_evidence():
    df = load_results()
    traps = df[df.category.str.startswith("trap")]

    st.subheader("How would I know it works?")
    st.markdown(
        f'<p class="lede">{len(df)} runs · {df.config.nunique()} configurations · '
        f'<b>${df.cost_usd.sum():.2f}</b> total. Ground truth is computed in pandas by the grader '
        f'and never goes near the agent. Grading is binary and all-or-nothing, following '
        f'GeneBench-Pro.</p>', unsafe_allow_html=True)

    full = df[df.config == "full"]
    _, doms = load_stats(df)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pass rate", f"{full.passed.mean():.0%}")
    k2.metric("On trap tasks", f"{traps[traps.config=='full'].passed.mean():.0%}")
    k3.metric("Held-out tasks", f"{full[full.holdout].passed.mean():.0%}",
              help="Never looked at while tuning the prompt.")
    k4.metric("Cost per analysis", f"${full.cost_usd.mean():.4f}")

    st.divider()
    st.markdown("### Does it generalise?")
    st.markdown(
        "Every trap in `trial.csv` is one **I** planted, and every guardrail was designed while "
        "staring at that file. Passing it proves the guardrails work on the failures I *already "
        "knew about* — a much weaker claim than it looks.\n\n"
        "So `sales.csv` is a **held-out domain**: e-commerce, not medicine. Different columns, "
        "different semantics, traps of the same *species* but a different animal — revenue "
        "exported as text (`\"1,234.56\"`), internal QA orders at `999999.99`, refunds still in "
        "the file, `-1` for a missing age, and a Simpson's paradox on **channel × segment** "
        "instead of arm × severity.")
    d1, d2, d3 = st.columns(3)
    for col, dom, note in ((d1, "penguins", "clean data, no traps"),
                           (d2, "trial", "designed against"),
                           (d3, "sales", "🎯 HELD-OUT DOMAIN")):
        if dom in doms:
            m, lo, hi = doms[dom]
            col.metric(f"{dom} — {note}", f"{m:.0%}", help=f"95% CI [{lo:.0%}, {hi:.0%}]")
            col.caption(f"95% CI [{lo:.0%}, {hi:.0%}]")

    st.divider()
    st.markdown("### The ablations")
    st.caption("Each configuration switches off exactly one mechanism. If a mechanism doesn't pay "
               "for itself, it gets cut — that was the deal I made in the design doc.")

    a = df.groupby("config").agg(pass_rate=("passed", "mean"),
                                 naive=("wrong_attractor", "mean"))
    ta = traps.groupby("config").agg(pass_rate=("passed", "mean"),
                                     naive=("wrong_attractor", "mean"))

    c1, c2 = st.columns(2)
    with c1:
        bar(ta, "pass_rate", "Pass rate — trap tasks only")
    with c2:
        bar(ta, "naive", "Fell for the *documented* naive answer")
        st.caption("Every trap records its plausible-but-wrong answer, so a failure isn't just "
                   "*wrong* — we know **which** wrong. Landing on the naive answer means the agent "
                   "fell into the notice–act gap specifically.")

    st.divider()
    st.markdown("### 🚨 And here is the result that inverted my own thesis")
    st.caption("Paired bootstrap against the full agent (10,000 hierarchical resamples: tasks, then "
               "runs within tasks). **If the 95% CI crosses zero, I cannot distinguish that "
               "mechanism from doing nothing — and I say so rather than pretending to a number.**")

    abl, _ = load_stats(df)
    label = {"no_briefing": "the deterministic data briefing  ← the DETECTOR",
             "no_guardrails": "every guardrail at once",
             "no_ledger": "the Findings Ledger  ← my centrepiece",
             "no_verifier": "the fresh-context verifier",
             "no_grounding": "the numeric grounding gate",
             "no_contract": "the Question Contract",
             "no_truncation": "observation truncation"}
    tbl = pd.DataFrame({
        "removing this…": [label[i] for i in abl.index],
        "pass rate": [f"{v:.0%}" for v in abl.pass_rate],
        "Δ vs full": [f"{v:+.0%}" for v in abl.delta_vs_full],
        "95% CI": [f"[{lo:+.0%}, {hi:+.0%}]" for lo, hi in zip(abl.lo95, abl.hi95)],
        "verdict": list(abl.verdict),
    })
    st.dataframe(tbl, hide_index=True, use_container_width=True)

    st.markdown(
        """
> ### A gate is only as good as the detector feeding it.
>
> **Removing the data briefing alone hurts as much as removing every guardrail combined.**
> The individual *gates* — the ledger, the verifier, the grounding check, the contract — show
> **no detectable effect** even with the CIs this tight.
>
> The papers describe a **notice–act** gap, and I read it as a failure to *act* — so I built
> machinery to force action. The ablation says the leverage is on the **notice** side. Tell the
> agent what's in the data, deterministically, before it starts, and **it acts on it.**
> It didn't need to be forced. It needed to be *informed*.

**The honest caveats.** *"No detectable effect"* still is not *"no effect"* — the CIs bound it,
they don't zero it. And a gate that fires on a few percent of runs is **insurance, not
throughput**: you do not price a fabricated number in a drug filing by its frequency. The right
test for a gate is adversarial, not average.

**What I'd build next is not another gate. It's more detectors.**
        """)

    with st.expander("Per-task results"):
        pt = (full.groupby(["category", "task_id"])
              .agg(passed=("passed", "sum"), n=("passed", "size"),
                   steps=("steps", "mean"), cost=("cost_usd", "mean")).reset_index())
        pt["score"] = pt.passed.astype(str) + "/" + pt.n.astype(str)
        pt["steps"] = pt.steps.round(1)
        pt["cost"] = pt.cost.map("${:.4f}".format)
        st.dataframe(pt[["category", "task_id", "score", "steps", "cost"]],
                     hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════════════

st.title("A data-analysis agent, built from scratch")

# The provider is a base_url (D01), so the footer must READ it rather than assert it. It used to
# say "at Nebius Token Factory - about half a cent per analysis", which silently became false the
# moment the endpoint changed. A hardcoded cost claim is the kind of thing this project exists to
# not do.
_HOSTS = {
    "tokenfactory.nebius.com": "Nebius Token Factory",
    "generativelanguage.googleapis.com": "the Google Gemini API",
    "api.groq.com": "Groq",
    "openrouter.ai": "OpenRouter",
    "modal.run": "a GPU I rented on Modal",
}
_provider = next((name for host, name in _HOSTS.items() if host in cfg_mod.BASE_URL),
                 "a self-hosted OpenAI-compatible endpoint")

# A self-hosted endpoint is billed by the GPU-second, not by the token. Quoting a $/1M rate here
# would be a fabricated number — see config.SELF_HOSTED.
if cfg_mod.SELF_HOSTED:
    _price = (f"self-hosted with vLLM — billed by the GPU-second "
              f"(~\\${cfg_mod.GPU_HOURLY_USD:.2f}/hour for an A10G), <b>not</b> per token")
else:
    _p_in, _p_out = cfg_mod.PRICES.get(cfg_mod.AGENT_MODEL, cfg_mod.DEFAULT_PRICE)
    _price = f"\\${_p_in:.2f}/\\${_p_out:.2f} per 1M tokens in/out"

st.markdown(
    f'<p class="lede">An agent that gets a prompt and some files, explores the data, writes and '
    f'runs code, <b>checks its own intermediate results</b>, and returns a structured, audited '
    f'answer.<br>No framework. ~885 lines. Running on '
    f'<b>{cfg_mod.AGENT_MODEL}</b> at {_provider} — {_price}.</p>',
    unsafe_allow_html=True)

if cfg_mod.SELF_HOSTED:
    st.caption(
        "⚠️ **The live demo is not the measured system.** The 4,480-run evaluation on the "
        "*Evidence* tab ran on **Qwen3-30B-A3B** at Nebius, with a **cross-family** verifier "
        "(gpt-oss-120b). This endpoint serves **one** model — a **4B**, seven times smaller — so "
        "the verifier here reviews work from its own family, which the design explicitly argues "
        "against. Numbers produced live are **not** comparable to the evaluation. "
        "That the 4B still lands the Simpson's task is a nice advertisement for scaffolding over "
        "parameters — and it is n=1, so it is an advertisement, not a result.")

t1, t2 = st.tabs(["🔬  Run the agent", "📊  The evidence"])
with t1:
    tab_run()
with t2:
    tab_evidence()

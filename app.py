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

from agentlib import Config, PyExecutor, run_agent
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

    with st.expander("⚙️  Guardrails — switch them off and watch it fail"):
        g1, g2, g3, g4, g5 = st.columns(5)
        use_briefing = g1.toggle("Data briefing", True, help="The DETECTOR. Removing this costs 27 points.")
        use_ledger = g2.toggle("Findings Ledger", True, help="A noticed problem becomes an open obligation.")
        use_contract = g3.toggle("Question Contract", True)
        use_grounding = g4.toggle("Grounding gate", True, help="Every number must appear in real output.")
        use_verifier = g5.toggle("Verifier", True, help="Fresh eyes, different model family.")

    if not st.button("Run the agent", type="primary"):
        st.info("Pick a question and hit **Run the agent**. "
                "A run takes 30–90 seconds and costs about half a cent.")
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

    with st.spinner("thinking, writing code, running it, checking itself…"):
        try:
            run = run_agent(question, [str(ROOT / f) for f in files], conf, executor=PyExecutor())
        except Exception as e:                                    # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}")
            return

    st.session_state["last_run"] = run
    _render_run(run)


def _render_run(run):
    r = run.report
    if not r:
        st.error("No answer produced.")
        return

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Answer", r.get("value") if r.get("value") is not None else "—")
    m2.metric("Confidence", r.get("confidence", "—"))
    m3.metric("Steps", run.steps)
    m4.metric("Cost", f"${run.cost_usd:.4f}")

    st.markdown(f"### {r['answer']}")

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
st.markdown(
    f'<p class="lede">An agent that gets a prompt and some files, explores the data, writes and '
    f'runs code, <b>checks its own intermediate results</b>, and returns a structured, audited '
    f'answer.<br>No framework. ~885 lines. Running on '
    f'<b>{cfg_mod.AGENT_MODEL}</b> at Nebius Token Factory — about half a cent per analysis.</p>',
    unsafe_allow_html=True)

t1, t2 = st.tabs(["🔬  Run the agent", "📊  The evidence"])
with t1:
    tab_run()
with t2:
    tab_evidence()

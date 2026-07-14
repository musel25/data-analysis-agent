# Part 1 — Review of the two papers

> **GBP** — *GeneBench-Pro: Evaluating Multistage Statistical Reasoning in Genomics, Quantitative
> Biology, and Translational Biomedicine.* Li & Ho, OpenAI. bioRxiv, June 2026.
>
> **DDB** — *DrugDiscoveryBench: Can Coding Agents Assist Early-Stage Drug Discovery?*
> Akyürek, Tu et al., Scale AI & Phylo. 2026.

Both papers evaluate the same system: **one LLM agent, in a container, writing and running Python
against messy scientific data, returning a structured answer.** Neither is about chatbots. They are
about the thing Part 2 asks me to build, which is why I read them as an engineer rather than a
referee — and then, unusually, got to check some of their claims against a harness of my own.

That last part is what makes this review worth reading. **Two of my criticisms are not opinions. They
are things that happened to me** while building Part 2 on their protocols, and I have the receipts.

---

## 1. Summary

### 1.1 GeneBench-Pro

**129 problems** in genomics and translational biomedicine, across 10 domains. The agent gets messy
staged files and a deliberately minimal prompt, works in Docker with pandas/scipy/statsmodels and no
internet, and must return one JSON object. Grading is **binary, programmatic, all-or-nothing**,
against pre-specified absolute numeric tolerances — and, notably, **there is no LLM judge anywhere**.
**10 independent attempts per problem** for standard evaluations, and **5 for the GPT Pro (Extended)
and Claude Opus rows** — 11 of the 60 evaluated configurations. Confidence intervals by hierarchical
bootstrap, 20,000 resamples.

The organising construct is the **decision point** — *"substantive inferential forks where a
plausible wrong choice leads to a qualitatively different downstream answer"* (3–13 per problem,
median 6). Problems are **fully simulated**, so the causal structure is known and nothing can be
memorised; and every plausible-but-wrong path is **ablation-verified to be numerically distinct from
the graded answer**.

**The results are brutal.** The best configuration in the paper reaches **31.5%**. The best mainline
model reaches **28.7%** — and **scores literally zero across all ten attempts on 45.7% of problems.**

The diagnosis is the paper's real contribution:

> *"models often complete substantial portions of the workflow but exhibit a consistent gap between
> **noticing** and **acting** by identifying local diagnostic signals but failing to propagate the
> implications to the corresponding analysis decision."* (p. 1)

> *"the agent notices the relevant local diagnostic clue but **treats it as a local data cleaning
> issue rather than as evidence that should change the downstream statistical method and QC
> pipeline**."* (p. 13)

### 1.2 DrugDiscoveryBench

**82 expert-authored tasks** spanning target identification → patent mining → structure-activity
analysis, each grounded in real artifacts (patents, papers, database records). Agents work inside an
adaptation of Stanford's **BIOMNI** environment — 226 functions across 22 domains, exposed as an
ordinary Python library rather than as tools, plus a 76-file data lake. **12 frontier models across 6
agentic harnesses** — but *not* a full grid: **29 settings**, unevenly distributed (GPT-5.5 got 6
settings across 4 harnesses; six models got exactly **one** setting each), with reasoning effort swept
low → ceiling for only **3** families. **3 trials** per setting.

Grading is by **LLM judge** (GPT-5.4) against expert-written rubrics, and **pass requires 100% of the
outcome criteria.** Best agent: **51.6%** (GPT-5.5 + mini-SWE-agent, xhigh).

Three findings:

1. Pass rate scales cleanly with reasoning effort *within* a family (GPT-5.5 Codex: **27.6 → 39.8 →
   43.9%**).
2. The frontier is tight and the harness matters as much as the model.
3. **The failure taxonomy** (Table 3, 226 failing runs) — which is the most useful table in either
   paper for anyone actually building one of these:

| Failure mode | Share |
|---|---|
| **Domain reasoning** — *"applies an incorrect scientific premise or misinterprets the data it has, **even though its inputs and tools are correct**"* | **54.0%** |
| Derivation error — right approach, wrong calculation | 18.6% |
| Retrieval — wrong source, or never reads a provided file | 16.4% |
| Constraint — violates an explicit instruction | 7.5% |
| Final-answer slip | 3.5% |

> **Over half of all failures are not coding failures.** The tools worked. The Python was correct.

---

## 2. What is convincing

**GBP's benchmark design is the best thing in either paper, and I copied it.**

Three principles, and each one earned its place in my own build:

- **Simulate, so you know the truth.** Real data gives you ground truth you can *assume*. Simulation
  gives you ground truth you can *compute* — and it is the only way to *plant* a decision point and
  be certain the naive path lands somewhere else.
- **Ablation-verify the separation.** For every trap, assert that the plausible-but-wrong answer lies
  far outside the tolerance band. Otherwise a lazy analysis can land inside the band and be graded
  correct, and the benchmark measures nothing. **I implemented this guard and it immediately killed
  two of my own tasks** — a median that was robust to the sentinels I'd planted, and an age question
  independent of its own filter. Both looked perfectly reasonable. Both graded nothing. The assertion
  found them; I didn't.
- **Binary, all-or-nothing grading.** They defend it better than I could: *"an agent that executes
  several intermediate steps correctly but returns the wrong decision-relevant answer has not
  successfully automated the analysis."* Partial credit measures effort. Binary measures usefulness.

**The "decision point" is a genuine conceptual contribution.** It is what makes GBP *diagnostic*
rather than merely hard. A benchmark can be difficult because the arithmetic is fiddly; GBP is
difficult because there are six places to take a defensible-looking wrong turn.

**DDB's expert authorship is the expensive, unglamorous, right thing to do.** 82 tasks written by
pharmaceutical scientists and grounded in real patents and database records is not something you can
generate. It is the part of the paper you cannot fake, and it is why their failure taxonomy is worth
trusting *qualitatively* even where I distrust it numerically.

**And both papers publish their failure modes, not just their scores.** That is rarer than it should
be, and it is the only reason either paper is useful to a builder. I reused DDB's taxonomy verbatim
so that my agent's failure profile is comparable to the literature — which is cheaper *and* more
credible than inventing my own.

---

## 3. What is questionable

Here is where I can do better than an opinion, because I built Part 2 on these protocols and watched
two of them break.

### 3.1 🚨 GBP's confidence intervals are too narrow — and narrowest exactly where their headline lives

GBP computes CIs by **hierarchical bootstrap: resample problems, then resample the 10 repeated runs
within each problem** (Methods, p. 15; 20,000 resamples).

**I implemented that exact protocol.** Then I ran my identical benchmark twice, either side of a code
change I had already proved was inert — same error rate, same step count, same budget-exhaustion
rate. My centrepiece mechanism came back:

| the same experiment, run twice | Δ | 95% CI | verdict |
|---|---|---|---|
| run A | **−11.1%** | `[−18.6%, −4.3%]` | ***"SIGNIFICANT"*** |
| run B | **−1.8%** | `[−8.9%, +4.6%]` | *"no detectable effect"* |

**Same code. Same tasks. Opposite conclusions, from 95% intervals that barely overlap.** Individual
tasks swung 30–50 points between the two runs — one went **1/10** and then **6/10**.

The bootstrap is not miscoded. It is being asked something it cannot answer. With **10 runs per
cell**, it resamples with replacement *from the ten outcomes actually observed*. A cell that came
back `1/10` therefore has a bootstrap distribution centred near 10% — and it **cannot reach the 60%
the next run produced.** The empirical distribution is degenerate near `p=0` and `p=1`.

**Now apply that to GBP.** I first wrote here that "the interval on nearly half their benchmark is
[0, 0]". **That was an inference, not a quotation** — GBP never publishes *per-problem* intervals, and
its *aggregate* CIs are not degenerate (GPT-5.6 Sol is 28.7% `[22.5, 35.1]`). So I have replaced my
inference with **what the paper literally prints**, which is worse and cannot be argued with:

**Supplementary Table 2 (p. 22) reports uncertainty as "the half-width of a 95% hierarchical bootstrap
confidence interval" — a symmetric `±` on a quantity bounded below by zero.** Read it literally:

| row | reported 95% CI | implied lower bound |
|---|---|---|
| Claude Opus 4.8 (max), public subset | `18.0 ± 22.0%` | **−4.0%** |
| Gemini 3.1 Pro, public subset | `7.0 ± 11.5%` | **−4.5%** |
| MiniMax M2.7, **full suite** | `0.6 ± 0.7%` | **−0.1%** |
| MiniMax M2.7, GLM 5.1, Grok 4.3, Kimi K2.7 (+4 more) | **`0.0 ± 0.0%`** | — |

**Eight rows report a zero-width 95% confidence interval.** That is not a confidence interval. It is
the nonparametric bootstrap **degenerating on all-zero data** — resample ten zeros, get zero, every
time. A Clopper–Pearson interval on the same runs would put the upper bound near 3%.

> **The error bars are tightest precisely where the data carries no information**, and the paper
> prints that as `0.0 ± 0.0%`.

This is not a nitpick about a supplementary figure. It is the paper's central statistical apparatus,
and it under-reports uncertainty in exactly the regime the paper's argument depends on — the 45.7% of
problems the best mainline model never solves.

**The fix is not more resamples — it is more runs, or a different estimator.** A hierarchical
beta-binomial would shrink the extreme cells instead of pretending they are certain. Cheaper and more
honest: **run the whole benchmark twice and report the spread between runs.** That number cannot lie
to you, because it is a measurement rather than a model. (It is what I report now, and it is the only
error bar in my project I actually trust.)

### 3.2 🚨 DDB's failure taxonomy is built from *one run per cell*

Table 3 — the 54% domain-reasoning figure, the most-cited number in the paper — is explicit in its
own caption:

> *"We utilize **one run per (model, task)** at the native harness and ceiling reasoning effort across
> 6 models, 226 model-failure runs."*

**One run.** No repeats. No error bar, and none possible.

But agent runs are *enormously* noisy — which I know because I measured it on my own harness: the
same task, same agent, same config, same everything, gave **1/10 passes** on one run of the benchmark
and **6/10** on the next. A single trajectory is not a description of a model's behaviour. It is one
draw from a wide distribution.

So the honest reading of Table 3 is not *"54% of agent failures are domain reasoning."* It is
**"of 226 single trajectories we happened to sample, 54% looked like domain reasoning to us."** Those
are very different sentences, and the paper only writes the first one.

To their credit, the per-model breakdown (45.7%–60.9%) suggests the finding is *directionally* robust
— but that range is itself computed from single runs, so it is not the reassurance it looks like.

**And I say all this while having reused their taxonomy**, because a shared vocabulary is worth more
than a precise one. But I would want the frequencies repeated across runs before I planned an
engineering roadmap around "54%."

### 3.3 The taxonomy's categories are mutually exclusive by *fiat*, not by nature

> *"Each failure is assigned the single root cause via a **decision ladder** (Retrieval, Constraint,
> Domain reasoning, Derivation error, then Final-answer slip)."*

A twenty-step trajectory that goes wrong usually goes wrong in more than one way. The ladder forces
each one into a single bucket by a priority order **the authors chose** — so a run that both queried
the wrong database *and* misread the biology is coded `Retrieval`, and the domain error is invisible.

This means the five categories are not competing hypotheses; they are a partition imposed after the
fact. "54% domain reasoning" strictly means *"54% of failures had no retrieval or constraint error
that we noticed first."* That is a weaker and less actionable claim than the headline, and the ladder
ordering is never justified.

### 3.4 DDB's most-quoted forward-looking sentence rests on four tasks

This is the line everyone will quote, including me:

> *"76 out of 82 tasks are solved without any hints in at least one of the trials… After the hints,
> **we find that** at least 1 of the agents is able to pass 80 out of 82 **and near-pass 1**. The
> results suggest that **execution is within reach for today's agents should they be given the expert
> workflow**."* (p. 14)

Three problems.

**First, "76/82" is not a pass rate — it is an oracle.** It means *at least one* run got it, across
**29 settings × 3 trials ≈ 87 runs per task**. The actual best single agent scores **51.6%**.
Reporting best-of-N alongside a pass rate, in the same paragraph, invites exactly the misreading that
the gap is small.

**Second, the hint experiment moves four tasks.** 76 → 80, on the six that were unsolved. **n = 6.**
The paper's central claim about where the headroom lies — the one that most directly tells a builder
what to do — is supported by four task flips, unrepeated. (The Introduction calls this "flips the
majority of tasks to solved". The majority of six is four.)

**Third — and this is the one that actually worries me — the experiment is close to circular.** A
**solvability assessment against the expert's playbook was an *inclusion criterion*** when the tasks
were authored (p. 5), and tasks that could not be made robust were **dropped**. So the benchmark
contains, *by construction*, tasks that an agent can solve when handed the playbook — and the
headline finding is that handing agents the playbook lets them solve the tasks. How many candidates
were dropped at that gate is **not stated**.

I believe the conclusion, incidentally. I just do not think this experiment establishes it.

### 3.5 GBP's regime statistic is not comparable across rows

GBP runs **10 attempts for standard evaluations but 5** for GPT Pro (Extended) and Claude Opus. The
headline **31.5%** comes from a 5-attempt row.

Means survive that fine. **The "zero across all attempts" statistic does not.** If a model's true rate
on a problem is *p*, the chance of observing zero successes is `(1−p)^n` — which is *mechanically
larger at n=5 than at n=10*. So the 5-attempt rows will show more zero-pass problems than the
10-attempt rows **even if the models are identical.** The regime distribution (Fig. 4B), which is one
of the paper's most rhetorically effective exhibits, is comparing rows with different denominators.

### 3.6 Both papers exclude infrastructure failures, and that is not neutral

GBP: *"attempts ending in container, tooling, provider, or response-format errors were excluded."*
DDB: *"Runs that fail for transient reasons (e.g. network errors) are retried."*

This is the right instinct — a rate limit is not an agent failure — and I had to fix exactly this bug
in my own harness, where seven provider 429s were being recorded as agent failures and **five of them
landed on my strongest configuration**, quietly costing it ~2 points against its own ablations.

But the exclusion is only neutral if infrastructure failures are **independent of difficulty**, and
they are not: harder problems produce longer trajectories, more tool calls, more context, and
therefore more timeouts and format errors. Dropping them plausibly biases pass rates **upward on
exactly the hard problems**. Neither paper checks this, and it would cost them one line.

### 3.7 DDB's judge has never been shown a human

> *"we ran Inter-Rater Agreement with **two other LLM judges**, Claude Sonnet 4.6 and Gemini 3.5
> Flash, on a random subset of 200 responses and found **perfect agreement with respect to binary
> labels** (κ = 1.0)."*

Every headline number in DDB flows through a single LLM judge, and the only validation is **against
other LLMs.**

κ = 1.0 is not reassuring; it is *suspicious*. Perfect agreement between three models from three
labs, on 200 items, means either the rubric items are so mechanical that the judge is redundant —
in which case grade them programmatically and remove the judge — or the three models share a
systematic bias, which is exactly what inter-rater agreement is supposed to detect and exactly what
three LLMs cannot detect about each other. **And note the qualifier I have now restored to the quote:
agreement was measured on the *binary* pass/fail label only, not on the continuous outcome score —
which makes the "the rubric is mechanical" horn of that dilemma *more* likely, not less.**

Two more things the paper does not mention. **Both validating judges are themselves models under
evaluation in the benchmark.** And the default judge, GPT-5.4, is **the same family as the
leaderboard winner**, GPT-5.5 — a self-preference risk that is never discussed.

The missing experiment is small and obvious: **have a domain expert grade 50 of them.** DDB has the
experts on staff — they wrote the tasks.

*(I hold myself to this too, and fail: 5 of my 28 tasks are LLM-graded and I did not validate the
judge at all. I say so in my limitations. The difference is that my judge decides 18% of my
benchmark; theirs decides 100% of DDB.)*

---

## 4. The key limitation neither paper states

**GBP's central diagnosis is scoped to frontier models, and the paper never says so.**

GBP is explicit that models *do* notice — the gap is in *acting*. Two separate sentences, and it
matters that they are separate, because they do not say the same thing:

> **Results, p. 3** — a claim about a *difference between two strong models*:
> *"the main qualitative improvement **in stronger models** lies less in noticing the relevant
> diagnostic clues than in turning those observations into concrete corrective and model-selection
> decisions that move the analysis onto the correct path."*

> **Discussion, p. 14** — the same idea, silently promoted to an *absolute property of the class*:
> *"while **frontier models consistently notice** data issues, statistical irregularities, and other
> potential problems, there remains an incomplete ability to bridge the 'notice-act' gap required to
> close the inferential loop."*

That promotion is **scope drift between two sections of one paper**, and it is the sentence everyone
quotes. I read it, believed it, and built my entire Part 2 design around it: machinery to **force the
agent to act** on what it noticed — a findings ledger, blocked submission, gates.

**Then I ablated it, and my results inverted theirs.**

| remove this | Δ vs the full agent | 95% CI |
|---|---|---|
| **the deterministic data briefing** — the thing that makes it *notice* | **−27%** | `[−40%, −15%]` |
| the Findings Ledger — the thing that forces it to *act* | −6% | `[−11%, −1%]` |
| the grounding gate | −1% | `[−4%, +3%]` |
| the Question Contract | +0% | `[−4%, +4%]` |

**Twenty lines of `pandas` that simply tell the agent what is in the data are worth four times every
gate in my design combined.**

This is not a contradiction of GBP. It is a **boundary condition on it**, and it sharpens their
finding rather than challenging it. Their subjects are GPT-5.6-class systems. My agent is
**Qwen3-30B-A3B** — a 30B open model, chosen deliberately *because* it is weak. For a frontier model,
`df.describe()` runs and the sentinel is spotted, so noticing is the solved half. **For a 30B model at
$0.10/1M tokens, noticing is not reliable at all**, and a deterministic profiler is substituting for a
capability the base model does not have.

> **The notice–act gap is a frontier-model problem. The *notice* gap is a small-model problem.**
>
> Which half of the scaffolding earns its keep depends on which half of the job your base model
> already does for free. GBP's prescription — *"close the gap between noticing and acting"* — is right
> for GPT-5.6 and **wrong for almost anything you would actually deploy on a budget.**

That matters, because the audience for these papers is people building agents, and most of them will
not be running the frontier. A benchmark paper that diagnoses a bottleneck **owes its readers the
capability range over which that diagnosis holds** — and neither paper reports how its failure
profile shifts down the ladder. It is the cheapest experiment in either paper and the most useful one
they did not run.

---

## 5. What I would change, and what I did

| | the papers | what I did instead |
|---|---|---|
| **Uncertainty** | bootstrap CI from 10 runs/cell (GBP); no CI at all (DDB taxonomy) | 20 runs/cell, and I report the **spread between two independent runs of the whole benchmark** next to every CI. If a verdict flips between them, it was never a verdict — it was weather. Two of my seven mechanisms flip sign. |
| **Failure analysis** | one run per (model, task) (DDB) | every failing run, repeated 20×, and a **wrong-attractor rate**: because every trap records its documented *plausible-but-wrong* answer, I can say not just *that* it was wrong but **which wrong**. Landing on the naive answer means it fell into the notice–act gap specifically. Neither paper reports this, and it costs ten lines. |
| **The judge** | validated against other LLMs (DDB) | not validated — and I say so, and I keep its blast radius to 5 of 28 tasks. Everything else is `math.isclose`. |
| **Generalisation** | one domain each | a **held-out domain** (e-commerce, not medicine). It promptly caught me overfitting: my one rule of domain knowledge was phrased in clinical-trial language and did nothing on a channel × segment confound. |

---

## 6. Verdict

**Both papers are right about the thing that matters, and I would trust their diagnosis before I
trusted their error bars.**

The single most important sentence in either paper is DDB's, and it is not a number:

> *"the agents knew which database to query and how to compute the property the task asked for at a
> high level. But somewhere along the execution **they drop a constraint, commit too early, fail to
> backtrack** or fail with respect to scientific common sense."*

That is a claim about **procedure**, not capability, and it is the reason my Part 2 spends its entire
budget on scaffolding rather than on a bigger model. Both papers earn that conclusion.

But both are **evaluation papers whose evaluations are under-powered in the specific places their
arguments lean hardest** — GBP's intervals are degenerate on the 45.7% of problems that carry its
headline; DDB's failure taxonomy, its most useful artifact, is a single draw from a distribution its
own results prove is wide.

And I would not have been able to say either of those things with any confidence if I had not built
the harness and watched **the identical experiment give me two opposite answers.**

> That is the real lesson I take from reading them, and it is the one I would defend hardest:
> **the instrument you build to stop yourself believing noise is itself an instrument — and it also
> has to be checked.** Not by reading it. By running it twice and seeing whether it says the same
> thing.

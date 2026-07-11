"""Run the benchmark. Every result is cached to results.jsonl so the notebooks never re-spend.

    uv run python -m evals.run_eval --config full --runs 3
    uv run python -m evals.run_eval --ablate            # every ablation, sequentially
    uv run python -m evals.run_eval --config full --runs 3 --tasks t4_simpson

Each (config, task, attempt) is skipped if already in results.jsonl, so this is resumable — you
can stop it, and start it again, and it picks up where it left off.
"""
from __future__ import annotations

import argparse
import itertools
import os
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentlib import Config, PyExecutor, run_agent
from agentlib.llm import METER

from .grade import grade
from .tasks import TASKS, test_leak, test_separation

RESULTS = Path(__file__).resolve().parent / "results.jsonl"

# The ablations. Each turns off exactly ONE mechanism, so the delta is attributable.
# `baseline_noexec` is the control: does the agent loop earn its cost at all?
CONFIGS = {
    "full":            Config(),
    "no_ledger":       Config(use_ledger=False),        # the centrepiece — expect the biggest drop
    "no_grounding":    Config(use_grounding=False),
    "no_contract":     Config(use_contract=False),
    "no_verifier":     Config(use_verifier=False),
    "no_briefing":     Config(use_briefing=False),
    "no_truncation":   Config(use_truncation=False),
    "no_guardrails":   Config(use_ledger=False, use_grounding=False, use_contract=False,
                              use_verifier=False, use_briefing=False),
}


def already_done() -> set[tuple[str, str, int]]:
    if not RESULTS.exists():
        return set()
    done = set()
    for line in RESULTS.read_text().splitlines():
        try:
            r = json.loads(line)
            done.add((r["config"], r["task_id"], r["attempt"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def dedupe_results() -> int:
    """Keep exactly one row per (config, task, attempt).

    I once had two eval loops running at the same time (a `pkill` that didn't take), both appending
    to this file. `already_done()` is read at startup, so neither saw the other's rows, and 222
    duplicate cells landed in the results — some of them *identical*, because the second process
    hit the first one's response cache. Duplicated cells silently over-weight whichever tasks got
    double-run, which is exactly the kind of quiet corruption a benchmark must not have.

    So: dedupe on every load, and refuse to start if another run is already going.
    """
    if not RESULTS.exists():
        return 0
    seen, kept = set(), []
    for line in RESULTS.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (r.get("config"), r.get("task_id"), r.get("attempt"))
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)
    dropped = len(RESULTS.read_text().splitlines()) - len(kept)
    if dropped:
        RESULTS.write_text("\n".join(kept) + "\n")
    return dropped


LOCK = RESULTS.with_suffix(".lock")


def assert_single_instance() -> None:
    """Refuse to start if another run_eval is alive. Two writers corrupt the results.

    A lockfile holding the pid, not a `pgrep` — because `pgrep -f evals.run_eval` also matches the
    `uv run` wrapper in this process's OWN ancestry, so it refuses to let you start at all. (Yes, I
    wrote the pgrep version first and it locked me out of my own benchmark.)
    """
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip())
            os.kill(pid, 0)                     # signal 0 = "is this pid alive?"
        except (ValueError, ProcessLookupError):
            pass                                # stale lock from a killed run — take it
        except PermissionError:
            pass                                # alive but not ours; assume stale
        else:
            raise SystemExit(
                f"Another evals.run_eval is already running (pid {pid}). Two writers corrupt "
                f"results.jsonl — I have made that mistake and it cost me 222 duplicated cells. "
                f"Kill it, or wait. (Stale lock? rm {LOCK})")
    LOCK.write_text(str(os.getpid()))
    import atexit
    atexit.register(lambda: LOCK.unlink(missing_ok=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full", choices=list(CONFIGS) + ["all"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--model", default=None, help="override the agent model")
    ap.add_argument("--ablate", action="store_true", help="run every config")
    ap.add_argument("--workers", type=int, default=8, help="concurrent agent runs")
    args = ap.parse_args()

    # The benchmark must be trustworthy before it is run. These are not decoration.
    assert_single_instance()
    test_separation()
    test_leak()
    dropped = dedupe_results()
    print("✓ separation + leak guards pass"
          + (f" · dropped {dropped} duplicate rows" if dropped else "") + "\n")

    configs = list(CONFIGS) if (args.ablate or args.config == "all") else [args.config]
    tasks = [t for t in TASKS if not args.tasks or t.id in args.tasks]
    done = already_done()

    jobs = []
    for cname in configs:
        cfg = CONFIGS[cname]
        if args.model:
            cfg = Config(**{**cfg.__dict__, "model": args.model})
        for task in tasks:
            for attempt in range(args.runs):
                if (cname, task.id, attempt) in done:
                    continue
                # Each attempt is an INDEPENDENT sample: a nonzero temperature plus the attempt
                # number as a cache nonce. Without both, repeats are cache replays of one run
                # (I shipped that bug; the $0.0000 cost on attempts 2 and 3 is what gave it away).
                jobs.append((cname,
                             Config(**{**cfg.__dict__, "verbose": False,
                                       "attempt": attempt, "temperature": 0.6}),
                             task, attempt))

    total = len(configs) * len(tasks) * args.runs
    print(f"{len(configs)} config(s) x {len(tasks)} task(s) x {args.runs} run(s) = {total} runs "
          f"({total - len(jobs)} cached, {len(jobs)} to do) · {args.workers} workers\n")

    t0, cost0 = time.time(), METER.cost_usd
    write_lock = threading.Lock()
    counter = itertools.count(1)

    def one(job):
        cname, cfg_a, task, attempt = job
        # Each worker gets its OWN executor. A shared kernel namespace across concurrent agents
        # would let run A see run B's variables — the agent's version of the hidden-state bug.
        ex = PyExecutor()
        try:
            run = run_agent(task.question, task.files, cfg_a, executor=ex)
            g = grade(task, run)
            rec = {
                "config": cname, "task_id": task.id, "category": task.category,
                "attempt": attempt, "holdout": task.holdout,
                "passed": g.passed, "failure_mode": g.failure_mode,
                "wrong_attractor": g.wrong_attractor, "note": g.note,
                "value": run.value, "steps": run.steps, "errors": run.errors,
                "rejections": run.rejections,
                "n_findings": len(run.ledger.findings),
                "n_acted": sum(f.status == "acted" for f in run.ledger.findings),
                "tokens": run.tokens, "cost_usd": round(run.cost_usd, 6),
                "stopped": run.stopped,
                "model": cfg_a.model or "default",
                "run_id": run.trace.run_id,
            }
        except Exception as e:  # a crash is a result, not an excuse to lose the run
            rec = {"config": cname, "task_id": task.id, "category": task.category,
                   "attempt": attempt, "holdout": task.holdout, "passed": False,
                   "failure_mode": "crash", "wrong_attractor": False,
                   "note": f"{type(e).__name__}: {e}"[:200], "value": None,
                   "steps": 0, "errors": 0, "rejections": [], "n_findings": 0,
                   "n_acted": 0, "tokens": 0, "cost_usd": 0.0, "stopped": "crash",
                   "model": cfg_a.model or "default", "run_id": ""}

        with write_lock:
            with RESULTS.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            i = next(counter)
            mark = "✓" if rec["passed"] else ("◆" if rec["wrong_attractor"] else "✗")
            print(f"  [{i:>3}/{len(jobs)}] {mark} {cname:<14} {task.id:<24} run{attempt}  "
                  f"{rec['steps']:>2} steps  ${rec['cost_usd']:.4f}  {rec['note'][:44]}",
                  flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, jobs))

    print(f"\n{'─'*78}")
    print(f"done in {time.time()-t0:.0f}s | spent ${METER.cost_usd - cost0:.4f} | "
          f"results -> {RESULTS}")
    print("legend:  ✓ pass   ✗ wrong   ◆ landed on the DOCUMENTED naive answer")


if __name__ == "__main__":
    main()

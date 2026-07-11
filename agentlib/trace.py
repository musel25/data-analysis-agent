"""Every step, as one JSONL line.

This is what turns a failure from "it got the wrong answer" into "at step 4 it noticed the
sentinels, and at step 6 it computed the mean anyway" — which is the only kind of failure report
you can actually act on.

It is also the raw material for the failure taxonomy in notebook 07: to classify a failing run as
*domain reasoning* vs *derivation error* (DrugDiscoveryBench's categories), you have to be able to
read what the agent did, step by step.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


class Trace:
    def __init__(self, question: str, run_id: str | None = None, task_id: str = ""):
        RUNS_DIR.mkdir(exist_ok=True)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        self.task_id = task_id
        self.question = question
        self.path = RUNS_DIR / "trace.jsonl"
        self.steps: list[dict] = []

    def step(self, n, tool, args, observation, error=None):
        rec = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "step": n,
            "tool": tool,
            "args": {k: (v[:800] if isinstance(v, str) else v) for k, v in args.items()},
            "observation": str(observation)[:800],
            "error": (error or "").splitlines()[-1] if error else None,
        }
        self.steps.append(rec)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def finish(self, run):
        rec = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "step": "END",
            "question": self.question,
            "stopped": run.stopped,
            "steps": run.steps,
            "errors": run.errors,
            "rejections": run.rejections,
            "value": run.value,
            "tokens": run.tokens,
            "cost_usd": round(run.cost_usd, 6),
            "findings": [
                {"observation": f.observation, "status": f.status} for f in run.ledger.findings
            ],
        }
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def replay(run_id: str, path: Path | None = None) -> None:
    """Pretty-print a past run. Demo insurance, and the debugging instrument."""
    path = path or RUNS_DIR / "trace.jsonl"
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec["run_id"] != run_id:
            continue
        if rec["step"] == "END":
            print(f"\n{'─'*70}\n{rec['stopped'].upper()} · {rec['steps']} steps · "
                  f"${rec['cost_usd']:.4f} · value={rec['value']}")
            continue
        print(f"\n[{rec['step']}] {rec['tool']}")
        if rec["tool"] == "run_python":
            for l in rec["args"].get("code", "").splitlines()[:8]:
                print(f"      | {l}")
        else:
            print(f"      | {json.dumps(rec['args'])[:100]}")
        print(f"   -> {rec['observation'][:160]}")

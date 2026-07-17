"""A data-analysis agent, built from scratch.

The whole thing is a while-loop, three tools, and four gates. See docs/DESIGN.md.

    from agentlib import run_agent, Config

    run = run_agent("Does the treatment work?", ["data/trial.csv", "data/data_dictionary.md"])
    print(run.report["answer"])

The notebooks build every piece of this, in order, and make it fail before fixing it.
"""

from .agent import Config, Run, run_agent
from .baseline import BaselineRun, run_baseline
from .executor import PyExecutor
from .ledger import Finding, FindingsLedger, QuestionContract
from .llm import METER, set_live
from .report import AnalysisReport, grounded
from .trace import replay

__all__ = [
    "run_agent", "Config", "Run",
    "run_baseline", "BaselineRun",
    "PyExecutor",
    "QuestionContract", "FindingsLedger", "Finding",
    "AnalysisReport", "grounded",
    "METER", "set_live",
    "replay",
]

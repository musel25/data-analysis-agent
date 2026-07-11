"""The kernel: run Python, keep the variables.

    ⚠️  THIS IS NOT A SANDBOX.

`exec()` runs in this process. Model-written code can read your files, eat your RAM, and import
anything installed. It is contained here by an isolated *namespace* and a throwaway venv, which
is honest scoping for a prototype and is not a security claim.

Both supplied papers ran their agents in Docker containers with no network access. That is the
correct production answer. The `run_python` tool contract — code in, stdout out — is deliberately
identical to what a container-backed executor exposes, so upgrading is a one-class swap, not a
redesign. See docs/DECISIONS.md D05.

Why a *persistent* namespace: because that is how an analyst actually works. Load once, filter,
inspect, model. A stateless executor would force the agent to re-read the CSV on every single
step, doubling the token cost of everything and making step 4 unable to see what step 2 learned.
It is the same model as a Jupyter kernel — which, pleasingly, is exactly the medium this whole
thing is taught in.
"""
from __future__ import annotations

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass

PREAMBLE = """
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")            # no GUI: figures go to files
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 120)
np.random.seed(0)
"""


@dataclass
class Result:
    stdout: str
    error: str | None
    figures: list[str]

    @property
    def ok(self) -> bool:
        return self.error is None


class PyExecutor:
    """A Python session that remembers. ~30 lines, and it is the whole 'code execution' story."""

    def __init__(self, figures_dir="figures"):
        self.ns: dict = {}
        self.figures_dir = figures_dir
        self.n_figures = 0
        self.reset()

    def reset(self):
        """Fresh namespace. Called between tasks — otherwise task 2 can see task 1's variables,
        which is the agent's version of the hidden-notebook-state bug."""
        self.ns = {"__name__": "__agent__"}
        exec(PREAMBLE, self.ns)
        self.n_figures = 0

    def run(self, code: str) -> Result:
        out, err = io.StringIO(), io.StringIO()
        error = None
        try:
            with redirect_stdout(out), redirect_stderr(err):
                self._exec_with_echo(code)
        except Exception:
            # Keep only the last few frames: the agent's own code, not the plumbing above it.
            # The final line is the one that names the error, so it is never truncated away.
            tb = traceback.format_exc().splitlines()
            error = "\n".join(tb[-6:])

        figures = self._collect_figures()
        stdout = out.getvalue() + err.getvalue()
        return Result(stdout=stdout, error=error, figures=figures)

    def _exec_with_echo(self, code: str) -> None:
        """Run the code, and echo the final expression the way a REPL does.

        WHY THIS EXISTS — and it is not a nicety.

        A bare `exec()` swallows bare expressions. So when the agent writes

            df = pd.read_csv('trial.csv')
            df.head()                        # <- a Jupyter user expects to SEE this

        ...it gets back nothing. Empty output. The agent thinks it has looked at the data, and
        it has actually gone BLIND. I watched a real run do exactly this, then guess the column
        names from thin air, get a KeyError, and guess again.

        The model was not being stupid. It was assuming Jupyter semantics, which is completely
        reasonable — every notebook on the internet behaves this way, and the tool is called
        "run_python". The bug was mine: my tool lied about what it was.

        Meet the model where it is. A tool that behaves surprisingly is a tool that will be
        misused, and the fix belongs in the tool, not in a sterner system prompt.
        (See docs/DECISIONS.md D19.)
        """
        import ast

        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            # everything except the last statement...
            if len(tree.body) > 1:
                exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), "<agent>", "exec"),
                     self.ns)
            # ...then evaluate the last expression and echo it, like a REPL
            value = eval(compile(ast.Expression(body=tree.body[-1].value), "<agent>", "eval"),
                         self.ns)
            if value is not None:
                print(repr(value) if not hasattr(value, "to_string") else value)
        else:
            exec(code, self.ns)

    def _collect_figures(self) -> list[str]:
        """Save any figures the code made. The model cannot see images, so the observation will
        tell it so — this converts the plot-instead-of-compute failure into a nudge."""
        plt = self.ns.get("plt")
        if plt is None or not plt.get_fignums():
            return []
        import os
        os.makedirs(self.figures_dir, exist_ok=True)
        paths = []
        for num in plt.get_fignums():
            self.n_figures += 1
            path = f"{self.figures_dir}/fig_{self.n_figures:02d}.png"
            plt.figure(num).savefig(path, dpi=110, bbox_inches="tight")
            paths.append(path)
        plt.close("all")
        return paths

    def variables(self) -> dict[str, str]:
        """Describe what currently lives in the namespace. Used for the state banner."""
        import types
        skip = {"pd", "np", "plt", "matplotlib", "__name__", "__builtins__"}
        out = {}
        for name, val in self.ns.items():
            if name.startswith("_") or name in skip or isinstance(val, types.ModuleType):
                continue
            out[name] = _describe(val)
        return out


def _describe(val) -> str:
    """A one-line shape-aware description. `df: DataFrame(824x11)` tells the model more than
    `df: <pandas.core.frame.DataFrame object at 0x7f...>` ever could."""
    t = type(val).__name__
    shape = getattr(val, "shape", None)
    if shape is not None:
        return f"{t}({'x'.join(str(d) for d in shape)})"
    try:
        return f"{t}(len={len(val)})"
    except TypeError:
        return t

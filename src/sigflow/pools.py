"""The worker pools behind ``parallel(block, backend=...)``.

``"process"``      the standard library's process pool, started from a fork server (or
                   a fresh interpreter): blocks must be importable and a script must run
                   its graph under ``if __name__ == "__main__":``.
``"thread"``       a thread pool: anything goes; parallel where the work releases the GIL.
``"interpreter"``  Python 3.14's ``InterpreterPoolExecutor``: one process, one
                   sub-interpreter per worker, each with its own GIL, so pure-Python
                   work runs in parallel. Sub-interpreters cannot load NumPy, so the
                   block's function must live in a module that does not import NumPy
                   (or sigflow), and the frames' data must be plain Python values
                   (bytes, str, numbers, tuples...). Suits decoding and parsing.

Each backend answers two questions: which executor to start, and what to send to a
worker for one frame.
"""

from __future__ import annotations

import functools
import sys
import types
from collections.abc import Callable
from typing import Any

__all__ = ["BACKENDS", "available", "check", "executor", "job"]

BACKENDS = ("process", "thread", "interpreter")


def available(backend: str) -> bool:
    """Whether a backend can run in this interpreter."""
    if backend == "interpreter":
        import concurrent.futures as cf

        return hasattr(cf, "InterpreterPoolExecutor")
    return backend in BACKENDS


def check(backend: str, defn: Any = None) -> None:
    """Refuse an unknown or unavailable backend, or a block it cannot run, when the
    graph is built."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(map(repr, BACKENDS))}")
    if not available(backend):
        raise RuntimeError(
            f"parallel(backend={backend!r}) needs Python 3.14 or later "
            f"(this is {sys.version.split()[0]})"
        )
    if backend == "interpreter" and defn is not None:
        _check_interpreter_block(defn)


def executor(backend: str, workers: int) -> Any:
    """Start the pool for a ``parallel`` node."""
    import concurrent.futures as cf

    if backend == "thread":
        return cf.ThreadPoolExecutor(workers)
    if backend == "interpreter":
        return cf.InterpreterPoolExecutor(workers)  # type: ignore[attr-defined]
    from .runtime import _worker_context

    return cf.ProcessPoolExecutor(workers, mp_context=_worker_context())


def job(backend: str, defn: Any, call: Callable[..., Any], labels: bool) -> tuple:
    """What to submit for one frame: ``(function, leading args)``; the frame's data and
    configuration follow."""
    from .runtime import _call_block

    if backend == "thread":
        return _call_block, (call,)
    if backend == "interpreter":
        return _InterpreterJob(defn.fn), ()
    return _call_block, (defn,)


# --------------------------------------------------------------------------- #
# interpreter: plain functions on plain data
# --------------------------------------------------------------------------- #


def _check_interpreter_block(defn: Any) -> None:
    """A sub-interpreter imports the block's function by its module path and cannot
    load NumPy, sigflow, or a function defined in ``__main__`` or a notebook."""
    fn = defn.fn
    mod, qual = getattr(fn, "__module__", None), getattr(fn, "__qualname__", "")
    if defn.labels:
        raise TypeError(
            f"parallel({defn.name}, backend='interpreter'): a label-aware block needs "
            "sigflow (and NumPy) in the worker; sub-interpreters cannot load them"
        )
    if not mod or mod in ("__main__", "__mp_main__") or "<" in qual:
        raise TypeError(
            f"parallel({defn.name}, backend='interpreter'): the function must be "
            "importable from a module (not a script's main module, a notebook cell, "
            "a lambda or a nested function), since each sub-interpreter imports it"
        )
    names = vars(sys.modules[mod]) if mod in sys.modules else {}
    uses = {
        value.__name__.split(".")[0]
        for value in names.values()
        if isinstance(value, types.ModuleType)
        and value.__name__.split(".")[0] in ("numpy", "sigflow")
    } | {
        getattr(value, "__module__", "").split(".")[0]
        for value in names.values()
        if callable(value)
        and getattr(value, "__module__", "").split(".")[0] in ("numpy", "sigflow")
    }
    if uses:
        raise TypeError(
            f"parallel({defn.name}, backend='interpreter'): {mod} imports "
            f"{', '.join(sorted(uses))}, which sub-interpreters cannot load. Put the "
            "function in a module of plain Python and wrap it with "
            "block(fn, inputs=..., outputs=...) where the graph is built"
        )


class _InterpreterJob:
    """What a sub-interpreter runs for one frame: the block's function, imported there
    by its module path, called on the frame's data and the configuration. It is a
    standard-library ``partial`` on the worker side: nothing of sigflow travels."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def submit(self, pool: Any, loop: Any, data: Any, cfg: dict) -> Any:
        return loop.run_in_executor(pool, functools.partial(self.fn, **cfg), data)


def interpreter_failure(name: str, error: BaseException) -> RuntimeError | None:
    """The error to stop a run with when a sub-interpreter could not even start the
    job (an import it cannot do, data it cannot receive); None for an ordinary
    exception raised by the block, which becomes a failed frame."""
    structural = isinstance(error, ImportError) or type(error).__name__ in (
        "NotShareableError",
        "ExecutionFailed",
    )
    if not structural:
        return None
    return RuntimeError(
        f"parallel({name}, backend='interpreter'): a sub-interpreter could not run "
        f"the job ({type(error).__name__}, shown below). Sub-interpreters cannot load NumPy or sigflow, so the "
        "frames' data must be plain Python values (bytes, str, numbers, tuples) and "
        "the function must come from a module that imports neither."
    )

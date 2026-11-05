"""The worker pools behind ``parallel(block, backend=...)``.

``"process"``       the standard library's process pool, started from a fork server (or
                    a fresh interpreter): blocks must be importable and a script must
                    run its graph under ``if __name__ == "__main__":``.
``"thread"``        a thread pool: anything goes; parallel where the work releases the
                    GIL (NumPy, SciPy), or everywhere on a free-threaded build.
``"freethreaded"``  a thread pool that insists on a free-threaded build of Python
                    (3.13t, 3.14t), where threads run Python code in parallel: no
                    pickling, no worker start-up, lambdas and notebook blocks welcome.
                    Refused on a build with a GIL; if something turns the GIL back on
                    (an extension module without free-threading support, or
                    ``PYTHON_GIL=1``) the run warns that the threads will take turns.
``"auto"``          ``"freethreaded"`` when the GIL is off as the run starts,
                    ``"process"`` otherwise, so the block must suit both.
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
import sysconfig
import warnings

__all__ = ["BACKENDS", "check", "executor", "free_threaded", "gil_enabled", "resolve"]

BACKENDS = ("process", "thread", "freethreaded", "auto")


def free_threaded() -> bool:
    """Whether this is a free-threaded build of Python (``python3.13t``, ``3.14t``)."""
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def gil_enabled() -> bool:
    """Whether the GIL is on right now. A free-threaded build turns it back on when it
    imports an extension module that does not declare free-threading support (or when
    told to, with ``PYTHON_GIL=1`` or ``-X gil=1``)."""
    probe = getattr(sys, "_is_gil_enabled", None)
    return True if probe is None else bool(probe())


def check(backend: str) -> None:
    """Refuse an unknown backend, or one this interpreter cannot run, when the graph is
    built."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(map(repr, BACKENDS))}")
    if backend == "freethreaded" and not free_threaded():
        raise RuntimeError(
            "parallel(backend='freethreaded') needs a free-threaded build of Python "
            "(python3.13t or python3.14t); this one has a GIL. Use backend='thread' "
            "for work that releases the GIL, 'process' for pure Python, or 'auto' "
            "to pick per interpreter"
        )


def resolve(backend: str, name: str) -> str:
    """The pool a run starts for a ``parallel`` node: ``"process"`` or a thread kind."""
    if backend == "auto":
        return "process" if gil_enabled() else "freethreaded"
    if backend == "freethreaded" and gil_enabled():
        warnings.warn(
            f"parallel({name}, backend='freethreaded'): the GIL is enabled, so the "
            "threads will take turns. An extension module without free-threading "
            "support turns it back on when imported (or PYTHON_GIL=1 / -X gil=1 "
            "was given); see sigflow.pools.gil_enabled()",
            RuntimeWarning,
            stacklevel=2,
        )
    return backend


def executor(kind: str, workers: int) -> cf.Executor:
    """Start the pool for a resolved backend."""
    if kind == "process":
        from .runtime import _worker_context

        return cf.ProcessPoolExecutor(workers, mp_context=_worker_context())
    return cf.ThreadPoolExecutor(workers, thread_name_prefix="sigflow-parallel")

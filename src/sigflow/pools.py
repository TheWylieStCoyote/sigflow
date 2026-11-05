"""The worker pools behind ``parallel(block, backend=...)``.

``"process"``       the standard library's process pool, started from a fork server (or
                    a fresh interpreter): blocks must be importable and a script must
                    run its graph under ``if __name__ == "__main__":``.
``"thread"``        a thread pool: anything goes; parallel where the work releases the
                    GIL (NumPy, SciPy), or everywhere on a free-threaded build.
``"loky"``          a ``loky`` process pool (``pip install loky``): no main guard
                    needed, and blocks defined in a script or a notebook, lambdas
                    included, are sent to the workers by value.
``"interpreter"``   Python 3.14's ``InterpreterPoolExecutor``: one process, one
                    sub-interpreter per worker, each with its own GIL, so pure-Python
                    work runs in parallel. Sub-interpreters cannot load NumPy, so the
                    block's function must live in a module that does not import NumPy
                    (or sigflow), and the frames' data must be plain Python values
                    (bytes, str, numbers, tuples...). Suits decoding and parsing.
``"freethreaded"``  a thread pool that insists on a free-threaded build of Python
                    (3.13t, 3.14t), where threads run Python code in parallel: no
                    pickling, no worker start-up, lambdas and notebook blocks welcome.
                    Refused on a build with a GIL; if something turns the GIL back on
                    (an extension module without free-threading support, or
                    ``PYTHON_GIL=1``) the run warns that the threads will take turns.
``"auto"``          ``"freethreaded"`` when the GIL is off as the run starts,
                    ``"process"`` otherwise, so the block must suit both.

Each backend answers three questions: which pool a run starts (:func:`resolve`), how
to start it (:func:`executor`), and what to send to a worker for one frame
(:func:`job`).
"""

from __future__ import annotations

import concurrent.futures as cf
import functools
import hashlib
import sys
import sysconfig
import types
import warnings
import weakref
from collections.abc import Callable
from typing import Any

__all__ = [
    "BACKENDS",
    "available",
    "check",
    "executor",
    "free_threaded",
    "gil_enabled",
    "interpreter_failure",
    "job",
    "resolve",
]

BACKENDS = ("process", "thread", "loky", "interpreter", "freethreaded", "auto")


def free_threaded() -> bool:
    """Whether this is a free-threaded build of Python (``python3.13t``, ``3.14t``)."""
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def gil_enabled() -> bool:
    """Whether the GIL is on right now. A free-threaded build turns it back on when it
    imports an extension module that does not declare free-threading support (or when
    told to, with ``PYTHON_GIL=1`` or ``-X gil=1``)."""
    probe = getattr(sys, "_is_gil_enabled", None)
    return True if probe is None else bool(probe())


def available(backend: str) -> bool:
    """Whether a backend can run in this interpreter."""
    if backend == "loky":
        try:
            import loky  # noqa: F401
        except ImportError:
            return False
    if backend == "interpreter":
        return hasattr(cf, "InterpreterPoolExecutor")
    if backend == "freethreaded":
        return free_threaded()
    return backend in BACKENDS


def check(backend: str, defn: Any = None) -> None:
    """Refuse an unknown backend, one this interpreter cannot run, or a block it cannot
    run, when the graph is built."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(map(repr, BACKENDS))}")
    if backend == "loky" and not available(backend):
        raise ImportError(
            "parallel(backend='loky') needs the loky package: pip install loky"
        )
    if backend == "interpreter":
        if not available(backend):
            raise RuntimeError(
                "parallel(backend='interpreter') needs Python 3.14 or later "
                f"(this is {sys.version.split()[0]})"
            )
        if defn is not None:
            _check_interpreter_block(defn)
    if backend == "freethreaded" and not free_threaded():
        raise RuntimeError(
            "parallel(backend='freethreaded') needs a free-threaded build of Python "
            "(python3.13t or python3.14t); this one has a GIL. Use backend='thread' "
            "for work that releases the GIL, 'process' for pure Python, or 'auto' "
            "to pick per interpreter"
        )


def resolve(backend: str, name: str) -> str:
    """The pool a run starts for a ``parallel`` node (``"auto"`` picks one here)."""
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


def executor(kind: str, workers: int) -> Any:
    """Start the pool for a resolved backend."""
    if kind in ("thread", "freethreaded"):
        return cf.ThreadPoolExecutor(workers, thread_name_prefix="sigflow-parallel")
    if kind == "interpreter":
        return cf.InterpreterPoolExecutor(workers)  # type: ignore[attr-defined]
    if kind == "loky":
        from loky import get_reusable_executor

        # reusable: the workers stay up between runs, so a notebook does not pay
        # the start-up again ("auto": reuse the pool unless its settings changed)
        return _Borrowed(get_reusable_executor(max_workers=workers, reuse="auto"))
    from .runtime import _worker_context

    return cf.ProcessPoolExecutor(workers, mp_context=_worker_context())


def job(kind: str, defn: Any, call: Callable[..., Any], labels: bool) -> tuple:
    """What to submit for one frame: ``(function, leading args)``; the frame's data and
    configuration follow (and, for ``"process"``, the run's ``labels`` flag)."""
    from .runtime import _call_block

    if kind in ("thread", "freethreaded"):
        return _call_block, (call,)
    if kind == "interpreter":
        return _InterpreterJob(defn.fn), ()
    if kind == "loky":
        return _call_shipped, (_ship(defn), labels)
    return _call_block, (defn,)


class _Borrowed:
    """A shared executor seen as one the node owns: ``shutdown`` leaves it running
    for the next run (loky's pools are reusable by design)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def submit(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.submit(*args, **kwargs)

    def shutdown(self, wait: bool = True, **_: Any) -> None:
        pass


# --------------------------------------------------------------------------- #
# loky: block definitions by value
# --------------------------------------------------------------------------- #

# parent side: definition -> payload (weak: an id could be reused after collection)
_SHIPPED: weakref.WeakKeyDictionary[Any, bytes] = weakref.WeakKeyDictionary()


def _ship(defn: Any) -> bytes:
    """The definition as bytes a loky worker can rebuild without importing anything of
    the caller's: by value when its function lives in a main module, a notebook, a
    lambda or a nested function; by import path otherwise."""
    payload = _SHIPPED.get(defn)
    if payload is None:
        payload = _SHIPPED[defn] = _dumps(defn)
    return payload


def _dumps(defn: Any) -> bytes:
    import io

    import cloudpickle

    from .block import BlockDef

    class Pickler(cloudpickle.Pickler):
        def reducer_override(self, obj: Any) -> Any:
            if isinstance(obj, BlockDef) and not _importable(obj.fn):
                return _rebuild, (), dict(obj.__dict__)
            return super().reducer_override(obj)

    buf = io.BytesIO()
    Pickler(buf).dump(defn)
    return buf.getvalue()


def _importable(fn: Any) -> bool:
    """Whether a worker can import ``fn`` by its module and qualified name. A
    ``functools.partial`` (a block made with ``.bind()``) has no qualified name: it
    goes by value."""
    mod, qual = getattr(fn, "__module__", None), getattr(fn, "__qualname__", None)
    return (
        isinstance(mod, str)
        and isinstance(qual, str)
        and bool(mod)
        and bool(qual)
        and mod not in ("__main__", "__mp_main__")
        and "<" not in qual
    )


def _rebuild() -> Any:
    from .block import BlockDef

    return object.__new__(BlockDef)


_RECEIVED: dict[str, Any] = {}  # worker side: payload digest -> definition


def _call_shipped(payload: bytes, labels: bool, data: Any, cfg: dict) -> Any:
    """Worker side: rebuild the definition once per worker, then call it."""
    key = hashlib.sha1(payload).hexdigest()
    defn = _RECEIVED.get(key)
    if defn is None:
        import pickle

        if len(_RECEIVED) >= 256:  # a long-lived worker serving many graphs
            _RECEIVED.clear()
        defn = _RECEIVED[key] = pickle.loads(payload)
    return defn.resolve_call(labels)(data, **cfg)


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

"""The worker pools behind ``parallel(block, backend=...)``.

``"process"``       the standard library's process pool, started from a fork server (or
                    a fresh interpreter): blocks must be importable and a script must
                    run its graph under ``if __name__ == "__main__":``.
``"thread"``        a thread pool: anything goes; parallel where the work releases the
                    GIL (NumPy, SciPy), or everywhere on a free-threaded build.
``"loky"``          a ``loky`` process pool (``pip install loky``): no main guard
                    needed, and blocks defined in a script or a notebook, lambdas
                    included, are sent to the workers by value.
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
import hashlib
import sys
import sysconfig
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
    "job",
    "resolve",
]

BACKENDS = ("process", "thread", "loky", "freethreaded", "auto")


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
    if backend == "freethreaded":
        return free_threaded()
    return backend in BACKENDS


def check(backend: str) -> None:
    """Refuse an unknown backend, or one this interpreter cannot run, when the graph is
    built."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(map(repr, BACKENDS))}")
    if backend == "loky" and not available(backend):
        raise ImportError(
            "parallel(backend='loky') needs the loky package: pip install loky"
        )
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
    mod, qual = getattr(fn, "__module__", None), getattr(fn, "__qualname__", "")
    return bool(mod) and mod not in ("__main__", "__mp_main__") and "<" not in qual


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

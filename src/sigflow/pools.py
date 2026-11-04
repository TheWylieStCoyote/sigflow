"""The worker pools behind ``parallel(block, backend=...)``.

``"process"``  the standard library's process pool, started from a fork server (or a
               fresh interpreter): blocks must be importable and a script must run its
               graph under ``if __name__ == "__main__":``.
``"thread"``   a thread pool: anything goes; parallel where the work releases the GIL.
``"loky"``     a ``loky`` process pool (``pip install loky``): no main guard needed,
               and blocks defined in a script or a notebook, lambdas included, are sent
               to the workers by value.

Each backend answers two questions: which executor to start, and what to send to a
worker for one frame.
"""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Callable
from typing import Any

__all__ = ["BACKENDS", "available", "executor", "job"]

BACKENDS = ("process", "thread", "loky")


def available(backend: str) -> bool:
    """Whether a backend can run in this interpreter."""
    if backend == "loky":
        try:
            import loky  # noqa: F401
        except ImportError:
            return False
    return backend in BACKENDS


def check(backend: str) -> None:
    """Refuse an unknown or unavailable backend when the graph is built."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(map(repr, BACKENDS))}")
    if not available(backend):
        raise ImportError(
            f"parallel(backend={backend!r}) needs the {backend} package: "
            f"pip install {backend}"
        )


def executor(backend: str, workers: int) -> Any:
    """Start the pool for a ``parallel`` node."""
    import concurrent.futures as cf

    if backend == "thread":
        return cf.ThreadPoolExecutor(workers)
    if backend == "loky":
        from loky import get_reusable_executor

        # reusable: the workers stay up between runs, so a notebook does not pay
        # the start-up again ("auto": reuse the pool unless its settings changed)
        return _Borrowed(get_reusable_executor(max_workers=workers, reuse="auto"))
    from .runtime import _worker_context

    return cf.ProcessPoolExecutor(workers, mp_context=_worker_context())


def job(backend: str, defn: Any, call: Callable[..., Any], labels: bool) -> tuple:
    """What to submit for one frame: ``(function, leading args)``; the frame's data and
    configuration follow."""
    from .runtime import _call_block

    if backend == "thread":
        return _call_block, (call,)
    if backend == "loky":
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

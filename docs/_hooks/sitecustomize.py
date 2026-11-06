"""Docs build only: record the first graph an example runs, as an SVG diagram.

``docs/build_docs.py --run-examples`` puts this directory on ``PYTHONPATH`` and sets
``SIGFLOW_DOCS_DIAGRAM`` to the file to write. When ``sigflow.runtime`` is imported,
its ``arun`` (which ``run()`` and ``start()`` go through) is wrapped to draw the
first graph it is given. Nothing happens without the variable, or in worker
processes, so the examples run exactly as they do on their own.
"""

from __future__ import annotations

import os
import sys

_TARGET = os.environ.get("SIGFLOW_DOCS_DIAGRAM")


def _patch(runtime: object) -> None:
    import multiprocessing

    if multiprocessing.parent_process() is not None:  # a pool worker
        return
    original = runtime.arun  # type: ignore[attr-defined]
    drawn: list[bool] = []

    async def arun(graph, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not drawn:
            drawn.append(True)
            try:
                from sigflow.draw import draw

                draw(graph).save(_TARGET)
            except Exception:  # a diagram is a nicety; never break the example
                pass
        return await original(graph, *args, **kwargs)

    runtime.arun = arun  # type: ignore[attr-defined]


if _TARGET:
    import importlib.abc
    import importlib.machinery

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):  # type: ignore[no-untyped-def]
            if name != "sigflow.runtime":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            exec_module = spec.loader.exec_module

            def exec_and_patch(module):  # type: ignore[no-untyped-def]
                exec_module(module)
                _patch(module)

            spec.loader.exec_module = exec_and_patch  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, _Finder())

"""Run monitoring: per-block counters, percent complete, terminal and Jupyter display."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Monitor", "Reporter", "Stats"]


@dataclass
class Stats:
    name: str
    inp: int = 0
    out: int = 0
    errors: int = 0
    time: float = 0.0  # seconds spent inside the block function
    total: int | None = None  # expected number of items (sources only, when known)
    watchers: list[Any] = field(default_factory=list)  # called with every output frame

    def notify(self, frame: Any) -> None:
        """Hand an output frame to the watchers (probes); a watcher that raises is
        dropped, never the run."""
        for fn in list(self.watchers):
            try:
                fn(frame)
            except Exception:
                self.watchers.remove(fn)


@dataclass
class Monitor:
    """Collects live statistics for a run. Pass ``monitor=`` to ``run`` to keep it."""

    total: int | None = None
    ratio: Any = None  # the graph's output-per-source-frame ratio, when known
    produced: int = 0
    started: float | None = None
    finished: float | None = None
    _stats: dict[str, Stats] = field(default_factory=dict)
    _by_key: dict[Any, Stats] = field(default_factory=dict)
    _pending: dict[str, list[Any]] = field(default_factory=dict)  # watchers by name
    _pending_nodes: dict[Any, list[Any]] = field(default_factory=dict)  # by identity

    def stats(self, name: str, node: Any = None) -> Stats:
        """Stats for a node; nodes sharing a name get distinct rows (``fft``, ``fft#2``)."""
        if node is None:  # lookup by displayed name
            s = self._stats.get(name)
            if s is None:
                s = self._stats[name] = Stats(name)
            return s
        key = (name, getattr(node, "id", id(node)))
        s = self._by_key.get(key)
        if s is None:
            shown = name
            n = 2
            while shown in self._stats:
                shown = f"{name}#{n}"
                n += 1
            s = self._by_key[key] = self._stats[shown] = Stats(shown)
            s.watchers.extend(self._pending.pop(shown, ()))
            s.watchers.extend(self._pending_nodes.pop(key[1], ()))
        return s

    def watch_node(self, node: Any, fn: Any) -> None:
        """Like :meth:`watch`, for a node object: exact whatever name it is shown
        under, and before the run reaches it."""
        st = self.lookup(node)
        if st is not None:
            st.watchers.append(fn)
        else:
            ident = getattr(node, "id", id(node))
            self._pending_nodes.setdefault(ident, []).append(fn)

    def watch(self, name: str, fn: Any) -> None:
        """Call ``fn(frame)`` with every output frame of the node shown as ``name``
        (``fft``, ``fft#2``: the names of :meth:`table`); before the run reaches the
        node the watcher waits for it. What ``run.probe()`` is built on."""
        s = self._stats.get(name)
        if s is not None:
            s.watchers.append(fn)
        else:
            self._pending.setdefault(name, []).append(fn)

    def unwatch(self, name: str, fn: Any) -> None:
        s = self._stats.get(name)
        if s is not None and fn in s.watchers:
            s.watchers.remove(fn)
        if fn in self._pending.get(name, ()):
            self._pending[name].remove(fn)
        for fns in self._pending_nodes.values():
            if fn in fns:
                fns.remove(fn)

    @property
    def blocks(self) -> list[Stats]:
        return list(self._stats.values())

    def lookup(self, node: Any) -> Stats | None:
        """The stats of a node the run has registered, or None. By identity first;
        else by name when exactly one row carries it (a graph the runtime lowered
        runs copies of the operator tree's nodes, which keep their names)."""
        ident = getattr(node, "id", id(node))
        for (_, key), st in self._by_key.items():  # identity: whatever name it got
            if key == ident:
                return st
        rows = [s for (name, _), s in self._by_key.items() if name == node.name]
        return rows[0] if len(rows) == 1 else None

    def start(self) -> None:
        self.started = time.perf_counter()
        self.finished = None

    def stop(self) -> None:
        self.finished = time.perf_counter()

    @property
    def running(self) -> bool:
        return self.started is not None and self.finished is None

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.perf_counter()) - self.started

    @property
    def expected(self) -> int | None:
        """Expected number of output frames when it can be inferred."""
        if self.total is not None:
            return self.total
        totals = [s.total for s in self._stats.values() if s.total is not None]
        if not totals:
            return None
        r = self.ratio
        if r is not None and getattr(r, "kind", None) == "exact" and r.value != 1:
            return max(1, round(min(totals) * r.value))
        return min(totals)

    @property
    def percent(self) -> float | None:
        # progress is measured at the sources (first thing to advance) when
        # outputs are not 1:1 with inputs; the expected output count is only informative
        src = [s for s in self._stats.values() if s.total is not None]
        if src and self.total is None:
            exp = min(s.total for s in src if s.total is not None)
            done = max(s.out for s in src)
        else:
            exp, done = self.expected or 0, self.produced
        if not exp:
            return None
        return min(100.0, 100.0 * done / exp)

    @property
    def errors(self) -> int:
        return sum(s.errors for s in self._stats.values())

    # -- rendering --------------------------------------------------------- #
    def status(self) -> str:
        pct = self.percent
        head = f"{pct:5.1f}%" if pct is not None else f"{self.produced} frames"
        state = "running" if self.running else "done"
        line = f"[{state}] {head}  {self.elapsed:.2f}s"
        if self.errors:
            line += f"  errors={self.errors}"
        return line

    def table(self) -> str:
        rows = [f"{'block':24} {'in':>8} {'out':>8} {'err':>5} {'time':>9}"]
        for s in self._stats.values():
            rows.append(
                f"{s.name[:24]:24} {s.inp:>8} {s.out:>8} {s.errors:>5} {s.time * 1000:>7.1f}ms"
            )
        return "\n".join(rows)

    def _repr_html_(self) -> str:
        from .notebook import monitor_html

        return monitor_html(self)

    def __repr__(self) -> str:
        return self.status() + "\n" + self.table()


class Reporter:
    """Periodically renders the monitor while a run is in progress."""

    def __init__(
        self, mon: Monitor, graph: Any, mode: bool | str = True, interval: float = 0.2
    ) -> None:
        self.mon = mon
        self.graph = graph
        self.interval = interval
        self.last = 0.0
        self.mode = (
            mode
            if isinstance(mode, str)
            else ("notebook" if _in_notebook() else "text")
        )
        self._handle: Any = None

    def tick(self) -> None:
        now = time.perf_counter()
        if now - self.last < self.interval:
            return
        self.last = now
        self._render()

    def finish(self) -> None:
        self._render(final=True)
        if self.mode == "text":
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _render(self, final: bool = False) -> None:
        if self.mode == "notebook":
            try:
                from IPython.display import HTML, display

                from .notebook import monitor_html

                html = HTML(monitor_html(self.mon))
                if self._handle is None:
                    self._handle = display(html, display_id=True)
                else:
                    self._handle.update(html)
                return
            except Exception:  # pragma: no cover - fall back to text
                self.mode = "text"
        pct = self.mon.percent
        bar = ""
        if pct is not None:
            n = int(pct / 5)
            bar = "[" + "#" * n + "." * (20 - n) + "] "
        sys.stderr.write("\r" + bar + self.mon.status() + " " * 8)
        sys.stderr.flush()


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False

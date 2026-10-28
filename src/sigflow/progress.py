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
        return s

    @property
    def blocks(self) -> list[Stats]:
        return list(self._stats.values())

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

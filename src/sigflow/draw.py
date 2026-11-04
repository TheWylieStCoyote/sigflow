"""Graph diagrams: a dependency-free SVG renderer and a Graphviz DOT exporter.

    g.draw()                 # Diagram; displays inline in Jupyter (SVG)
    g.draw().save("g.svg")
    g.draw(types=False)      # hide edge type labels
    print(g.to_dot())        # for `dot -Tpng` or the graphviz Python package

    g.draw(stats=run.monitor)   # a still with each block's frame counts and errors
    g.draw(frame=out[7])        # the path one frame took through the graph
    run.diagram()               # the live picture of a running graph (see LiveDiagram)

Layout is structural: pipelines flow left to right, fan-out / zip / merge
stack their members vertically and converge on a small join node.  Boxes are
coloured by kind (source, map, stream, tap/plot), and edges are labelled with
the type that flows along them.
"""

from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from typing import Any

from .types import describe

__all__ = ["Diagram", "LiveDiagram", "draw", "to_dot"]

# geometry (px)
_PAD_X, _PAD_Y = 28, 18  # gap between columns / rows
_H = 40  # box height
_CHAR = 7.2  # approx. width of a character at 12px monospace
_JOIN_W = 34  # width of join nodes (fan-out / zip / merge)

_FILL = {
    "source": "#dbeafe",
    "map": "#dcfce7",
    "stream": "#fef3c7",
    "tap": "#ede9fe",
    "plot": "#ede9fe",
    "join": "#e5e7eb",
}
_STROKE = {
    "source": "#2563eb",
    "map": "#16a34a",
    "stream": "#d97706",
    "tap": "#7c3aed",
    "plot": "#7c3aed",
    "join": "#6b7280",
}
# live overlay: the state of a block during a run, as its border colour
_STATE = {
    "idle": "#9ca3af",  # no frame yet
    "active": "#2563eb",  # produced since the last refresh
    "paused": "#d97706",
    "error": "#dc2626",
    "done": "#16a34a",
    "break": "#7c3aed",  # holding a frame at a breakpoint
}
_CAPTION_H = 18  # space under a live diagram for its status line
_PATH = "#0f172a"  # edges a frame travelled
_FADED = "0.22"  # opacity of what a frame did not touch
_FULL = 0.9  # a queue this full is backpressure


@dataclass
class _Box:
    id: int
    label: str
    sub: str
    kind: str
    typ: str = ""
    tip: str = ""
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = _H
    node: Any = None  # the graph node drawn, so a run's stats can be laid over it


@dataclass
class Badge:
    """What a live diagram shows on one box: frames in and out, errors, and a state
    (``idle``, ``active``, ``paused``, ``error``, ``done`` or ``None`` for settled)."""

    inp: int = 0
    out: int = 0
    errors: int = 0
    time: float = 0.0
    state: str | None = None
    name: str = ""  # the node's name in the monitor's table
    rate: float | None = None  # frames out per second since the previous picture


@dataclass
class _Cell:
    """Laid-out sub-graph: boxes with positions relative to the cell's origin."""

    w: float
    h: float
    boxes: list[_Box] = field(default_factory=list)
    edges: list[tuple[int, int, str]] = field(
        default_factory=list
    )  # (from id, to id, label)
    entries: list[int] = field(default_factory=list)  # box ids that receive input
    exits: list[int] = field(default_factory=list)  # box ids that emit output

    def shift(self, dx: float, dy: float) -> _Cell:
        for b in self.boxes:
            b.x += dx
            b.y += dy
        return self


class _Layout:
    def __init__(self, env: Any = None, types: bool = True) -> None:
        self._n = 0
        self.env = env
        self.types = types

    def _id(self) -> int:
        self._n += 1
        return self._n

    def leaf(self, node: Any) -> _Cell:
        import inspect

        from .graph import Block, Source
        from .plot import PlotBlock

        if isinstance(node, PlotBlock):
            kind = "plot"
        elif isinstance(node, Block):
            kind = "tap" if node.defn.tap else node.kind
        elif isinstance(node, Source):
            kind = "source"
        else:
            kind = "map"
        config = dict(getattr(node, "config", {}))
        defaults = {
            k: p.default
            for k, p in getattr(getattr(node, "defn", None), "config", {}).items()
            if p.default is not inspect.Parameter.empty
        }
        changed = {
            k: v for k, v in config.items() if k not in defaults or defaults[k] != v
        }
        sub = ", ".join(f"{k}={_short(v)}" for k, v in changed.items())
        typ = (
            describe(node.out_type.resolve(self.env))
            if self.env is not None
            else describe(node.out_type)
        )
        if self.types:
            typ = "→ " + typ
        else:
            typ = ""
        tip = node.describe() + (
            "\nconfig: " + ", ".join(f"{k}={v!r}" for k, v in config.items())
            if config
            else ""
        )
        w = max(len(node.name) * 1.15, len(sub), len(typ) * 0.9) * _CHAR + 24
        h = _H + (12 if sub else 0) + (12 if typ else 0)
        b = _Box(
            self._id(), node.name, sub, kind, typ, tip, w=max(w, 70), h=h, node=node
        )
        return _Cell(
            b.w, b.h, [b], [], [b.id] if node.in_type is not None else [], [b.id]
        )

    def join(self, label: str) -> _Box:
        return _Box(self._id(), label, "", "join", w=_JOIN_W, h=26)

    def layout(self, node: Any) -> _Cell:
        from .graph import Fanout, Feedback, Merge, Pipeline, Select, When, Zip

        if isinstance(node, Pipeline):
            return self.row([self.layout(s) for s in node.stages], node)
        if isinstance(node, Fanout):
            return self.column(
                [self.layout(b) for b in node.branches], "fan-out", node, broadcast=True
            )
        if isinstance(node, Zip):
            return self.column(
                [self.layout(p) for p in node.parts],
                "zip",
                node,
                broadcast=node.in_type is not None,
            )
        if isinstance(node, Merge):
            return self.column(
                [self.layout(p) for p in node.parts], "merge", node, broadcast=False
            )
        if type(node).__name__ == "Align":
            return self.column(
                [self.layout(p) for p in node.parts], "align", node, broadcast=False
            )
        if type(node).__name__ == "Graph":
            return self.dag(node)
        if isinstance(node, Feedback):
            inner = self.layout(node.body)
            j = self.join("↻")
            j.node = node
            j.x, j.y = inner.w + _PAD_X, (inner.h - j.h) / 2
            inner.boxes.append(j)
            for e in inner.exits:
                inner.edges.append((e, j.id, ""))
            inner.w = j.x + j.w
            inner.exits = [j.id]
            return inner
        if isinstance(node, When):
            cells = [self.layout(node.then)] + (
                [self.layout(node.otherwise)] if node.otherwise is not None else []
            )
            return self.column(cells, "when", node, broadcast=True)
        if type(node).__name__ == "PerKey":
            return self.column(
                [self.layout(node.body)], "per key", node, broadcast=True
            )
        if type(node).__name__ == "Switch":
            members = _members(node)
            cells = [self.layout(b) for b in members]
            keys = [repr(k) for k in node.branches] + (
                ["default"] if len(members) > len(node.branches) else []
            )
            for c, k in zip(cells, keys):  # label each branch's first box with its key
                if c.boxes:
                    b0 = c.boxes[0]
                    b0.sub = k + (" · " + b0.sub if b0.sub else "")
                    b0.h = max(b0.h, _H + 12)
                    b0.w = max(b0.w, len(b0.sub) * _CHAR + 24)
            return self.column(cells, "switch", node, broadcast=True)
        if isinstance(node, Select) or type(node).__name__ in (
            "PortRef",
            "Slice",
            "StreamSlice",
        ):
            b = self.join(node.name)
            b.node = node
            b.w = max(_JOIN_W, len(node.name) * _CHAR + 14)
            return _Cell(b.w, b.h, [b], [], [b.id], [b.id])
        return self.leaf(node)

    def dag(self, g: Any) -> _Cell:
        """Layered layout of a builder Graph: column = longest path from a source."""
        from .builder import _Input

        order = g.order()
        depth: dict[str, int] = {}
        for name in order:
            ins = [e[0].name for e in g.consumers_of(name)]
            depth[name] = 1 + max((depth[i] for i in ins), default=-1)
        cells: dict[str, _Cell] = {}
        for name in order:
            node = g.members[name]
            if isinstance(node, _Input):
                b = self.join("in")
                cells[name] = _Cell(b.w, b.h, [b], [], [b.id], [b.id])
            else:
                c = self.layout(node)
                if len(c.boxes) == 1 and c.boxes[0].kind != "join":
                    c.boxes[0].label = (
                        name if name != c.boxes[0].label else c.boxes[0].label
                    )
                cells[name] = c
        columns: dict[int, list[str]] = {}
        for name, d in depth.items():
            columns.setdefault(d, []).append(name)
        out = _Cell(0, 0)
        x = 0.0
        for d in sorted(columns):
            y = 0.0
            w = max(cells[n].w for n in columns[d])
            for n in columns[d]:
                c = cells[n]
                c.shift(x + (w - c.w) / 2, y)
                out.boxes += c.boxes
                out.edges += c.edges
                y += c.h + _PAD_Y
            out.h = max(out.h, y - _PAD_Y)
            x += w + _PAD_X * 1.5
        out.w = x - _PAD_X * 1.5
        for src, dst, slot in g.edges:
            label = f"[{src.key}]" if src.key is not None else ""
            for ea in cells[src.name].exits:
                for eb in cells[dst].entries:
                    out.edges.append((ea, eb, label))
        out.entries = [
            e
            for n, node in g.members.items()
            if isinstance(node, _Input)
            for e in cells[n].exits
        ]
        out.exits = [e for r in g.outputs for e in cells[r.name].exits]
        return out

    def row(self, cells: list[_Cell], node: Any) -> _Cell:
        h = max(c.h for c in cells)
        x = 0.0
        out = _Cell(0, h)
        prev: _Cell | None = None
        prev_stage: Any = None
        for stage, c in zip(node.stages, cells):
            c.shift(x, (h - c.h) / 2)
            out.boxes += c.boxes
            out.edges += c.edges
            if prev is not None:
                label = describe(prev_stage.out_type)
                for a in prev.exits:
                    for b in c.entries:
                        out.edges.append((a, b, label))
            prev, prev_stage = c, stage
            x += c.w + _PAD_X
        out.w = x - _PAD_X
        out.entries = cells[0].entries
        out.exits = cells[-1].exits
        return out

    def column(
        self, cells: list[_Cell], label: str, node: Any, broadcast: bool
    ) -> _Cell:
        y = 0.0
        w = max(c.w for c in cells)
        out = _Cell(0, 0)
        for c in cells:
            c.shift(0, y)
            out.boxes += c.boxes
            out.edges += c.edges
            y += c.h + _PAD_Y
        h = y - _PAD_Y
        j = self.join(label)
        j.node = node
        j.x, j.y = w + _PAD_X, (h - j.h) / 2
        out.boxes.append(j)
        for c, part in zip(cells, _members(node)):
            for e in c.exits:
                out.edges.append((e, j.id, describe(part.out_type)))
        out.w, out.h = j.x + j.w, h
        out.entries = [e for c in cells for e in c.entries] if broadcast else []
        out.exits = [j.id]
        return out


def _members(node: Any) -> list[Any]:
    if type(node).__name__ == "When":
        return [node.then] + ([node.otherwise] if node.otherwise is not None else [])
    if type(node).__name__ == "PerKey":
        return [node.body]
    if type(node).__name__ == "Switch":
        return node._all()
    return getattr(node, "branches", None) or getattr(node, "parts", None) or []


def _short(v: Any) -> str:
    s = repr(v)
    return s if len(s) <= 14 else s[:11] + "..."


class Diagram:
    """An SVG picture of a graph. Displays inline in Jupyter; ``.save(path)`` writes the
    file. With ``stats=`` (a :class:`~sigflow.progress.Monitor`) every box carries the
    frames it took in and put out, its errors, and a border colour for its state; the
    live picture of a run (:class:`LiveDiagram`) is this, refreshed."""

    def __init__(
        self,
        node: Any,
        types: bool = True,
        stats: Any = None,
        live: bool = False,
        frame: Any = None,
    ) -> None:
        self.node = node
        self.types = types
        self.monitor = stats
        self.frame = frame  # draw the path this frame took
        self.live = live or stats is not None
        env = None
        try:
            env = node.validate()
        except Exception:  # pragma: no cover - drawing must never fail on a bad graph
            pass
        self.cell = _Layout(env, types).layout(node)
        c = self.cell
        if type(self.node).__name__ == "Graph":  # room for the arcs of long edges
            for b in c.boxes:
                b.y += 40
            c.h += 40

    @property
    def svg(self) -> str:
        path = self.path(self.frame) if self.frame is not None else None
        if self.monitor is None:
            return self.render(None, path=path, failed_at=_failed_at(self.frame))
        counts = self.counts(self.monitor)
        return self.render(
            self.badges(counts, None, "done"),
            caption=self.caption(),
            queues=self.queues(self.monitor, peak=True),
            path=path,
            failed_at=_failed_at(self.frame),
        )

    # -- the path of one frame -------------------------------------------- #
    def path(self, frame: Any) -> dict[int, Any]:
        """The boxes ``frame`` passed through, each with the lineage record it left
        there: its lineage matched against the diagram, nearest upstream box first,
        so two blocks of the same name are told apart by where they sit. Records
        with no box of their own (a feedback loop's pairs) are looked through."""
        prov = getattr(frame, "prov", None)
        if not prov:
            return {}
        c = self.cell
        boxes = {b.id: b for b in c.boxes}
        preds: dict[int, list[int]] = {}
        for a, b, _ in c.edges:
            preds.setdefault(b, []).append(a)
        memo: dict[tuple[int, int], dict[int, Any] | None] = {}

        def upstream(bid: int) -> list[int]:
            """Every box upstream of ``bid``, nearest first."""
            seen: list[int] = []
            todo = list(preds.get(bid, ()))
            while todo:
                a = todo.pop(0)
                if a not in seen:
                    seen.append(a)
                    todo.extend(preds.get(a, ()))
            return seen

        def match(p: Any, bid: int) -> dict[int, Any] | None:
            key = (p.id, bid)
            if key in memo:
                return memo[key]
            memo[key] = None  # a cycle (feedback) cannot match through itself
            if not _same(boxes[bid], p):
                return None
            found: dict[int, Any] = {bid: p}
            for parent in p.parents:
                link(parent, bid, found, 0)
            memo[key] = found
            return found

        def link(parent: Any, bid: int, found: dict[int, Any], depth: int) -> None:
            for a in upstream(bid):
                sub = match(parent, a)
                if sub:
                    found.update(sub)
                    return
            if depth < 8:  # a record with no box: its parents may have one
                for grand in parent.parents:
                    link(grand, bid, found, depth + 1)

        best: dict[int, Any] = {}
        for box in c.boxes:
            got = match(prov, box.id)
            if got and len(got) > len(best):
                best = got
        return best

    def queues(self, monitor: Any, peak: bool = False) -> dict[tuple[int, int], Any]:
        """The queue on each drawn edge, by ``(box, box)``, as ``(depth, capacity)``;
        ``peak`` uses the deepest each has been instead of the depth now."""
        out: dict[tuple[int, int], Any] = {}
        boxes = {b.id: b for b in self.cell.boxes}
        for a, b, _ in self.cell.edges:
            na, nb = boxes[a].node, boxes[b].node
            if na is None or nb is None:
                continue
            eq = monitor.queue_between(na, nb)
            if eq is not None:
                out[(a, b)] = (eq.peak if peak else eq.depth, eq.capacity, eq.peak)
        return out

    # -- live overlay ------------------------------------------------------ #
    def counts(self, monitor: Any) -> dict[int, Badge]:
        """Each drawn box's stats from ``monitor``, by box id (boxes without stats, such
        as blocks the run has not reached, are left out)."""
        out: dict[int, Badge] = {}
        for b in self.cell.boxes:
            st = monitor.lookup(b.node) if b.node is not None else None
            if st is not None:
                out[b.id] = Badge(st.inp, st.out, st.errors, st.time, name=st.name)
        return out

    @staticmethod
    def badges(
        counts: dict[int, Badge],
        previous: dict[int, tuple[int, int]] | None,
        run_state: str | None,
        held_at: str | None = None,
    ) -> dict[int, Badge]:
        """Give each badge its state: the block holding a frame at a breakpoint
        first, then errors; a box whose counts moved since ``previous`` is active;
        otherwise the run's state (paused / finished) or idle when nothing has
        flowed yet; a settled box keeps its ordinary border."""
        for bid, badge in counts.items():
            if held_at is not None and badge.name == held_at:
                badge.state = "break"
            elif badge.errors:
                badge.state = "error"
            elif run_state in ("done", "stopped", "failed"):
                badge.state = "done"
            elif previous is not None and previous.get(bid) != (badge.inp, badge.out):
                badge.state = "active"
            elif run_state == "paused":
                badge.state = "paused"
            elif badge.inp == 0 and badge.out == 0:
                badge.state = "idle"
        return counts

    def caption(self, run_state: str | None = None, held_at: str | None = None) -> str:
        mon = self.monitor
        if mon is None:
            return ""
        state = run_state or ("running" if mon.running else "done")
        if held_at is not None:
            state = f"break at {held_at}"
        line = f"{state} · {mon.produced} produced · {mon.elapsed:.1f}s"
        if mon.errors:
            line += f" · {mon.errors} error{'s' if mon.errors != 1 else ''}"
        return line

    def render(
        self,
        badges: dict[int, Badge] | None,
        caption: str = "",
        queues: dict[tuple[int, int], Any] | None = None,
        path: dict[int, Any] | None = None,
        failed_at: str | None = None,
    ) -> str:
        c = self.cell
        m = 16
        on_path = set(path) if path else set()
        W, H = c.w + 2 * m, c.h + 2 * m + (_CAPTION_H if self.live else 0)
        boxes = {bx.id: bx for bx in c.boxes}
        parts = [
            (
                f"<svg xmlns='http://www.w3.org/2000/svg' width='{W:.0f}' height='{H:.0f}' "
                f"viewBox='0 0 {W:.0f} {H:.0f}' font-family='ui-monospace,SFMono-Regular,Menlo,monospace' "
                f"font-size='12'>"
            ),
            (
                "<defs><marker id='sf-arrow' viewBox='0 0 10 10' refX='10' refY='5' markerWidth='7' "
                "markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#6b7280'/>"
                "</marker></defs>"
            ),
        ]
        for a, b_id, label in c.edges:
            A, B = boxes[a], boxes[b_id]
            x1, y1 = A.x + A.w + m, A.y + A.h / 2 + m
            x2, y2 = B.x + m, B.y + B.h / 2 + m
            cx = (x1 + x2) / 2
            dx = x2 - x1
            crosses = any(
                bx.x + m > x1 + 4
                and bx.x + m + bx.w < x2 - 4
                and bx.y + m < max(y1, y2) + 4
                and bx.y + m + bx.h > min(y1, y2) - 4
                for bx in c.boxes
                if bx.id not in (a, b_id)
            )
            if crosses:
                # a long edge over other boxes: arc above the row so it stays readable
                top = (
                    min(
                        bx.y
                        for bx in c.boxes
                        if bx.x + m > x1 - 4 and bx.x + m < x2 + 4
                    )
                    + m
                )
                bump = top - 18 - dx / 40
                d = f"M{x1:.1f},{y1:.1f} C{x1 + dx * 0.2:.1f},{bump:.1f} {x2 - dx * 0.2:.1f},{bump:.1f} {x2:.1f},{y2:.1f}"
                ly = bump + 2
            else:
                d = f"M{x1:.1f},{y1:.1f} C{cx:.1f},{y1:.1f} {cx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"
                ly = (y1 + y2) / 2 - 4
            queue = queues.get((a, b_id)) if queues is not None else None
            stroke, width, opacity = "#6b7280", 1.4, "1"
            tip = ""
            if queue is not None:
                depth, cap, peak = queue
                tip = (
                    f"queue {depth}" + (f" of {cap}" if cap else "") + f", peak {peak}"
                )
                if cap and depth >= _FULL * cap:
                    stroke, width = _STATE["paused"], 2.2  # full: backpressure
            if path is not None:
                if a in on_path and b_id in on_path:
                    stroke, width = _PATH, 2.6
                else:
                    opacity = _FADED
            parts.append(
                f"<path d='{d}' fill='none' stroke='{stroke}' stroke-width='{width}' "
                f"opacity='{opacity}' marker-end='url(#sf-arrow)'>"
                + (f"<title>{html.escape(tip)}</title>" if tip else "")
                + "</path>"
            )
            if queue is not None and path is None:
                parts.append(_queue_bar(cx, ly + 15, *queue[:2]))
            if label.startswith("["):
                parts.append(
                    f"<text x='{cx:.1f}' y='{ly:.1f}' text-anchor='middle' font-size='9' "
                    f"fill='#6b7280'>{html.escape(label)}</text>"
                )
            if badges is not None and a in badges and badges[a].out and path is None:
                # frames that travelled this edge: what its producer put out
                parts.append(
                    f"<text x='{cx:.1f}' y='{ly + 11:.1f}' text-anchor='middle' font-size='9' "
                    f"fill='#2563eb'>{badges[a].out}</text>"
                )
        for b in c.boxes:
            x, y = b.x + m, b.y + m
            badge = badges.get(b.id) if badges is not None else None
            stroke = _STROKE[b.kind]
            width = 1.2 if b.kind == "join" else 1.4
            if badge is not None and badge.state is not None:
                stroke, width = _STATE[badge.state], 2.2
            opacity = "1"
            record = path.get(b.id) if path else None
            if path is not None:
                if record is None:
                    opacity = _FADED
                elif (
                    failed_at is not None and getattr(b.node, "name", None) == failed_at
                ):
                    stroke, width = _STATE["error"], 2.6  # where the frame broke
                else:
                    stroke, width = _PATH, 2.2
            if b.kind == "join":
                parts.append(
                    f"<g opacity='{opacity}'>"
                    f"<rect x='{x:.1f}' y='{y:.1f}' width='{b.w}' height='{b.h}' rx='13' "
                    f"fill='{_FILL['join']}' stroke='{stroke}' stroke-width='{width}'/>"
                    f"<text x='{x + b.w / 2:.1f}' y='{y + b.h / 2 + 4:.1f}' text-anchor='middle' "
                    f"font-size='9' fill='#111827'>{html.escape(b.label)}</text>"
                )
                if badge is not None and path is None:
                    # a join counts what it emits; its inputs are its branches' outputs
                    parts.append(_pill(x + b.w, y, _flow(badge, True), stroke))
                if record is not None and record.seq is not None:
                    parts.append(_pill(x + b.w, y, f"#{record.seq}", stroke))
                parts.append("</g>")
                continue
            tip = b.tip
            if badge is not None:
                tip += f"\nin {badge.inp}, out {badge.out}, errors {badge.errors}"
                if badge.out:
                    tip += f", {badge.time / badge.out * 1000:.2f} ms per frame"
                if badge.rate is not None:
                    tip += f", {badge.rate:.1f} frames/s"
            if record is not None:
                tip += f"\nthis frame: {record.op} #{record.seq}"
            parts.append(
                f"<g opacity='{opacity}'><title>{html.escape(tip)}</title>"
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{b.w:.1f}' height='{b.h}' rx='6' "
                f"fill='{_FILL[b.kind]}' stroke='{stroke}' stroke-width='{width}'/>"
            )
            if record is not None and record.seq is not None:
                parts.append(_pill(x + b.w, y, f"#{record.seq}", stroke))
            elif badge is not None:
                parts.append(
                    _pill(x + b.w, y, _flow(badge, b.kind == "source"), stroke)
                )
            if badge is not None and path is None and badge.errors:
                parts.append(
                    _pill(
                        x + b.w,
                        y + b.h,
                        f"✕ {badge.errors}",
                        _STATE["error"],
                        fill=True,
                    )
                )
            lines: list[tuple[str, float, str, str]] = [(b.label, 12, "#111827", "600")]
            if b.sub:
                lines.append((b.sub, 10, "#374151", "400"))
            if b.typ:
                lines.append((b.typ, 9.5, "#4b5563", "400"))
            step = b.h / (len(lines) + 1)
            for i, (text, size, color, weight) in enumerate(lines):
                parts.append(
                    f"<text x='{x + b.w / 2:.1f}' y='{y + step * (i + 1) + 4:.1f}' text-anchor='middle' "
                    f"font-size='{size}' fill='{color}' font-weight='{weight}'>{html.escape(text)}</text>"
                )
            parts.append("</g>")
        if caption:
            parts.append(
                f"<text x='{m}' y='{H - 6:.1f}' font-size='10' fill='#374151'>"
                f"{html.escape(caption)}</text>"
            )
        parts.append("</svg>")
        return "".join(parts)

    def _repr_svg_(self) -> str:
        return self.svg

    def _repr_html_(self) -> str:
        return f"<div style='overflow:auto;background:white;padding:4px;border-radius:6px'>{self.svg}</div>"

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.svg)
        return path

    def __repr__(self) -> str:
        return f"<Diagram of {self.node.name!r}: {len(self.cell.boxes)} nodes>"


def _flow(badge: Badge, source: bool) -> str:
    text = f"{badge.out}" if source else f"{badge.inp}→{badge.out}"
    if badge.rate:
        text += f" · {badge.rate:.0f}/s"
    return text


def _queue_bar(cx: float, y: float, depth: int, capacity: int) -> str:
    """A queue's depth under its edge: a bar against its capacity, or a count."""
    if not capacity:
        if not depth:
            return ""
        return (
            f"<text x='{cx:.1f}' y='{y + 4:.1f}' text-anchor='middle' font-size='8' "
            f"fill='#6b7280'>q {depth}</text>"
        )
    w = 28.0
    frac = min(depth / capacity, 1.0)
    colour = _STATE["paused"] if frac >= _FULL else "#2563eb"
    return (
        f"<rect x='{cx - w / 2:.1f}' y='{y:.1f}' width='{w}' height='5' rx='2' fill='#e5e7eb'/>"
        f"<rect x='{cx - w / 2:.1f}' y='{y:.1f}' width='{w * frac:.1f}' height='5' rx='2' "
        f"fill='{colour}'/>"
    )


def _same(box: _Box, record: Any) -> bool:
    """Whether a lineage record was left by the node a box draws."""
    node = box.node
    name = getattr(node, "name", None)
    if name is None:
        return False
    op = record.op
    if op == name or op.startswith(
        name + "["
    ):  # routers: when(...)[then], switch(...)[k]
        return True
    kind = type(node).__name__
    return (kind == "Fanout" and op == "fanout") or (kind == "Zip" and op == "zip")


def _failed_at(frame: Any) -> str | None:
    """The block a failed frame broke at, by name."""
    if frame is None or not getattr(frame, "failed", False):
        return None
    errors = frame.errors
    return errors[-1].origin if errors else None


def _pill(right: float, cy: float, text: str, colour: str, fill: bool = False) -> str:
    """A small rounded label whose right edge sits at ``right``, centred on ``cy``."""
    w = len(text) * 5.6 + 10
    x = right - w + 6
    bg, fg = (colour, "white") if fill else ("white", colour)
    return (
        f"<rect x='{x:.1f}' y='{cy - 7:.1f}' width='{w:.1f}' height='14' rx='7' "
        f"fill='{bg}' stroke='{colour}' stroke-width='1'/>"
        f"<text x='{x + w / 2:.1f}' y='{cy + 3.5:.1f}' text-anchor='middle' font-size='9' "
        f"fill='{fg}'>{html.escape(text)}</text>"
    )


class LiveDiagram:
    """The diagram of a running graph, refreshed while it runs (``run.diagram()``).

    Each box shows the frames it took in and put out, errors in red, and a border
    for its state: grey before its first frame, blue when it produced since the
    last refresh, amber while paused, red with errors, green once the run is over.
    Edges carry the count of frames that travelled them, and where the runtime
    keeps a queue between two blocks, its depth as a bar; a full queue turns the
    edge amber, which is backpressure: the block after it is the slow one. While
    the run goes each box shows its throughput, in frames per second. A status
    line sits below. ``highlight(frame)`` draws the path one frame took instead,
    ``highlight(None)`` goes back. Every refresh re-renders the whole SVG from the
    layout computed once, so the picture never moves. ``svg`` is the current
    picture; ``show()`` displays it in a notebook and keeps it fresh until the run
    finishes or ``close()``."""

    def __init__(self, run: Any, refresh: float = 0.5, types: bool = True) -> None:
        self.run = run
        self.refresh = refresh
        self.diagram = Diagram(run.graph, types=types, live=True)
        self._previous: dict[int, tuple[int, int]] | None = None
        self._previous_at: float | None = None
        self._handle: Any = None
        self._closed = False
        self.frame: Any = None  # draw this frame's path

    def highlight(self, frame: Any) -> LiveDiagram:
        """Draw the path ``frame`` took through the graph (None: the live counts)."""
        self.frame = frame
        self._update()
        return self

    @property
    def svg(self) -> str:
        """The picture now; boxes that moved since the previous call are active."""
        counts = self.diagram.counts(self.run.monitor)
        held_at = getattr(self.run, "held_at", None)
        state = self.run.state
        badges = self.diagram.badges(counts, self._previous, state, held_at)
        now = time.monotonic()
        if (
            self._previous is not None
            and self._previous_at is not None
            and state
            in (
                "running",
                "break",
            )
        ):
            dt = now - self._previous_at
            if dt > 0:
                for bid, badge in badges.items():
                    before = self._previous.get(bid)
                    if before is not None:
                        badge.rate = (badge.out - before[1]) / dt
        self._previous = {k: (b.inp, b.out) for k, b in counts.items()}
        self._previous_at = now
        self.diagram.monitor = self.run.monitor
        path = self.diagram.path(self.frame) if self.frame is not None else None
        return self.diagram.render(
            badges,
            caption=self.diagram.caption(state, held_at),
            queues=self.diagram.queues(self.run.monitor),
            path=path,
            failed_at=_failed_at(self.frame),
        )

    def html(self) -> str:
        return f"<div style='overflow:auto;background:white;padding:4px;border-radius:6px'>{self.svg}</div>"

    def _repr_html_(self) -> str:
        return self.html()

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.svg)
        return path

    def show(self) -> None:
        """Display in the notebook and refresh until the run finishes."""
        import threading

        from IPython.display import HTML, display

        self._handle = display(HTML(self.html()), display_id=True)

        def ticker() -> None:
            while not self.run.finished and not self._closed:
                self._update()
                time.sleep(self.refresh)
            self._update()

        threading.Thread(target=ticker, name="sigflow-diagram", daemon=True).start()

    def _update(self) -> None:
        if self._handle is None:
            return
        from IPython.display import HTML

        try:
            self._handle.update(HTML(self.html()))
        except (
            Exception
        ):  # pragma: no cover - a vanished front end must not kill the run
            pass

    def close(self) -> None:
        """Stop refreshing (the last picture stays)."""
        self._closed = True

    def __repr__(self) -> str:
        return f"<LiveDiagram of {self.run.graph.name!r}: {self.run.state}>"


def draw(
    node: Any, types: bool = True, stats: Any = None, frame: Any = None
) -> Diagram:
    return Diagram(node, types=types, stats=stats, frame=frame)


def to_dot(node: Any, types: bool = True) -> str:
    """Graphviz DOT source for the graph (left-to-right)."""
    cell = _Layout(node.validate(), types).layout(node)
    lines = [
        "digraph sigflow {",
        "  rankdir=LR; node [fontname=monospace fontsize=11 style=filled];",
    ]
    for b in cell.boxes:
        if b.kind == "join":
            lines.append(
                f'  n{b.id} [label="{b.label}" shape=circle fillcolor="{_FILL["join"]}" '
                f'color="{_STROKE["join"]}" fontsize=8 width=0.4 fixedsize=true];'
            )
        else:
            label = (
                b.label
                + (f"\\n{b.sub}" if b.sub else "")
                + (f"\\n{b.typ}" if b.typ else "")
            )
            lines.append(
                f'  n{b.id} [label="{label}" shape=box fillcolor="{_FILL[b.kind]}" '
                f'color="{_STROKE[b.kind]}"];'
            )
    for ea, eb, label in cell.edges:
        lines.append(f"  n{ea} -> n{eb};")
    lines.append("}")
    return "\n".join(lines)

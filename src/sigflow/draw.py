"""Graph diagrams: a dependency-free SVG renderer and a Graphviz DOT exporter.

    g.draw()                 # Diagram; displays inline in Jupyter (SVG)
    g.draw().save("g.svg")
    g.draw(types=False)      # hide edge type labels
    print(g.to_dot())        # for `dot -Tpng` or the graphviz Python package

Layout is structural: pipelines flow left to right, fan-out / zip / merge
stack their members vertically and converge on a small join node.  Boxes are
coloured by kind (source, map, stream, tap/plot), and edges are labelled with
the type that flows along them.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any

from .types import describe

__all__ = ["Diagram", "draw", "to_dot"]

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

    def shift(self, dx: float, dy: float) -> "_Cell":
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
        b = _Box(self._id(), node.name, sub, kind, typ, tip, w=max(w, 70), h=h)
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
    if type(node).__name__ == "Switch":
        return node._all()
    return getattr(node, "branches", None) or getattr(node, "parts", None) or []


def _short(v: Any) -> str:
    s = repr(v)
    return s if len(s) <= 14 else s[:11] + "..."


class Diagram:
    """An SVG picture of a graph. Displays inline in Jupyter; ``.save(path)`` writes the file."""

    def __init__(self, node: Any, types: bool = True) -> None:
        self.node = node
        self.types = types
        env = None
        try:
            env = node.validate()
        except Exception:  # pragma: no cover - drawing must never fail on a bad graph
            pass
        self.cell = _Layout(env, types).layout(node)
        self.svg = self._render()

    def _render(self) -> str:
        c = self.cell
        m = 16
        arcs = type(self.node).__name__ == "Graph"
        if arcs:
            for b in c.boxes:
                b.y += 40
            c.h += 40
        W, H = c.w + 2 * m, c.h + 2 * m
        boxes = {bx.id: bx for bx in c.boxes}
        parts = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{W:.0f}' height='{H:.0f}' "
            f"viewBox='0 0 {W:.0f} {H:.0f}' font-family='ui-monospace,SFMono-Regular,Menlo,monospace' "
            f"font-size='12'>",
            "<defs><marker id='sf-arrow' viewBox='0 0 10 10' refX='10' refY='5' markerWidth='7' "
            "markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#6b7280'/>"
            "</marker></defs>",
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
            parts.append(
                f"<path d='{d}' fill='none' stroke='#6b7280' stroke-width='1.4' marker-end='url(#sf-arrow)'/>"
            )
            if label.startswith("["):
                parts.append(
                    f"<text x='{cx:.1f}' y='{ly:.1f}' text-anchor='middle' font-size='9' "
                    f"fill='#6b7280'>{html.escape(label)}</text>"
                )
        for b in c.boxes:
            x, y = b.x + m, b.y + m
            if b.kind == "join":
                parts.append(
                    f"<rect x='{x:.1f}' y='{y:.1f}' width='{b.w}' height='{b.h}' rx='13' "
                    f"fill='{_FILL['join']}' stroke='{_STROKE['join']}' stroke-width='1.2'/>"
                    f"<text x='{x + b.w / 2:.1f}' y='{y + b.h / 2 + 4:.1f}' text-anchor='middle' "
                    f"font-size='9' fill='#111827'>{html.escape(b.label)}</text>"
                )
                continue
            parts.append(
                f"<g><title>{html.escape(b.tip)}</title>"
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{b.w:.1f}' height='{b.h}' rx='6' "
                f"fill='{_FILL[b.kind]}' stroke='{_STROKE[b.kind]}' stroke-width='1.4'/>"
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


def draw(node: Any, types: bool = True) -> Diagram:
    return Diagram(node, types=types)


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

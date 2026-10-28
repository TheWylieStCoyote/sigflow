"""Jupyter integration: HTML representations of graphs, frames and monitors."""

from __future__ import annotations

import html
from typing import Any

from .types import describe

_CSS = (
    "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;"
    "border-collapse:collapse;"
)
_TD = "padding:2px 10px;border-bottom:1px solid #8884;"


def graph_html(node: Any) -> str:
    from .graph import Block, Fanout, Merge, Pipeline, Source, Zip

    rows: list[str] = []

    def walk(n: Any, depth: int) -> None:
        pad = "&nbsp;" * (depth * 4)
        if isinstance(n, Pipeline):
            for s in n.stages:
                walk(s, depth)
        elif isinstance(n, Fanout):
            rows.append(
                f"<tr><td style='{_TD}'>{pad}<b>fan-out</b></td><td style='{_TD}'></td>"
                f"<td style='{_TD}'>{html.escape(describe(n.out_type))}</td><td style='{_TD}'></td></tr>"
            )
            for b in n.branches:
                walk(b, depth + 1)
        elif isinstance(n, Zip):
            rows.append(
                f"<tr><td style='{_TD}'>{pad}<b>zip</b></td><td style='{_TD}'></td>"
                f"<td style='{_TD}'>{html.escape(describe(n.out_type))}</td><td style='{_TD}'></td></tr>"
            )
            for p in n.parts:
                walk(p, depth + 1)
        elif isinstance(n, Merge):
            rows.append(
                f"<tr><td style='{_TD}'>{pad}<b>merge</b></td><td style='{_TD}'></td>"
                f"<td style='{_TD}'>{html.escape(describe(n.out_type))}</td><td style='{_TD}'></td></tr>"
            )
            for p in n.parts:
                walk(p, depth + 1)
        else:
            kind = (
                n.kind
                if isinstance(n, Block)
                else "source"
                if isinstance(n, Source)
                else "select"
                if type(n).__name__ == "Select"
                else ""
            )
            cfg = ", ".join(f"{k}={v!r}" for k, v in getattr(n, "config", {}).items())
            it = "" if n.in_type is None else describe(n.in_type)
            rows.append(
                f"<tr><td style='{_TD}'>{pad}{html.escape(n.name)}"
                f"<span style='opacity:.6'>({html.escape(cfg)})</span></td>"
                f"<td style='{_TD}'>{html.escape(it)}</td>"
                f"<td style='{_TD}'>{html.escape(describe(n.out_type))}</td>"
                f"<td style='{_TD};opacity:.6'>{kind}</td></tr>"
            )

    walk(node, 0)
    head = (
        f"<tr><th style='{_TD};text-align:left'>node</th><th style='{_TD};text-align:left'>in</th>"
        f"<th style='{_TD};text-align:left'>out</th><th style='{_TD};text-align:left'>kind</th></tr>"
    )
    return f"<table style='{_CSS}'>{head}{''.join(rows)}</table>"


def monitor_html(mon: Any) -> str:
    pct = mon.percent
    state = "running" if mon.running else "done"
    if pct is not None:
        bar = (
            f"<div style='width:320px;height:10px;background:#8883;border-radius:5px;overflow:hidden'>"
            f"<div style='width:{pct:.1f}%;height:100%;background:#3a7bd5'></div></div>"
            f"<span>{pct:5.1f}%</span>"
        )
    else:
        bar = f"<span>{mon.produced} frames</span>"
    rows = "".join(
        f"<tr><td style='{_TD}'>{html.escape(s.name)}</td><td style='{_TD};text-align:right'>{s.inp}</td>"
        f"<td style='{_TD};text-align:right'>{s.out}</td><td style='{_TD};text-align:right'>{s.errors}</td>"
        f"<td style='{_TD};text-align:right'>{s.time * 1000:.1f} ms</td></tr>"
        for s in mon.blocks
    )
    return (
        f"<div style='{_CSS}'><div style='display:flex;gap:12px;align-items:center'>"
        f"<b>{state}</b>{bar}<span>{mon.elapsed:.2f}s</span>"
        f"{'<span style=color:#c33>errors=' + str(mon.errors) + '</span>' if mon.errors else ''}</div>"
        f"<table style='{_CSS};margin-top:6px'><tr><th style='{_TD}'>block</th><th style='{_TD}'>in</th>"
        f"<th style='{_TD}'>out</th><th style='{_TD}'>err</th><th style='{_TD}'>time</th></tr>{rows}</table></div>"
    )


def frame_html(frame: Any) -> str:
    import numpy as np

    d = frame.data
    if isinstance(d, np.ndarray):
        ds = f"ndarray{d.shape} {d.dtype}"
    else:
        ds = html.escape(repr(d)[:80])
    rows = [f"<tr><td style='{_TD}'><b>data</b></td><td style='{_TD}'>{ds}</td></tr>"]
    for level in ("stream", "meta", "samples"):
        m = getattr(frame, level)
        if m:
            body = ", ".join(f"{k}={html.escape(_short(v))}" for k, v in m.items())
            rows.append(
                f"<tr><td style='{_TD}'><b>{level}</b></td><td style='{_TD}'>{body}</td></tr>"
            )
    if frame.tags:
        body = ", ".join(html.escape(repr(t)) for t in sorted(frame.tags, key=repr))
        rows.append(
            f"<tr><td style='{_TD}'><b>tags</b></td><td style='{_TD}'>{body}</td></tr>"
        )
    if frame.issues:
        body = "<br>".join(html.escape(str(i)) for i in frame.issues)
        rows.append(
            f"<tr><td style='{_TD}'><b>issues</b></td><td style='{_TD};color:#c33'>{body}</td></tr>"
        )
    if frame.prov:
        body = (
            html.escape(frame.trace())
            .replace("\n", "<br>")
            .replace("  ", "&nbsp;&nbsp;")
        )
        rows.append(
            f"<tr><td style='{_TD}'><b>provenance</b></td><td style='{_TD}'>{body}</td></tr>"
        )
    return f"<table style='{_CSS}'>{''.join(rows)}</table>"


def _short(v: Any) -> str:
    import numpy as np

    if isinstance(v, np.ndarray):
        return f"ndarray{v.shape}"
    s = repr(v)
    return s if len(s) < 40 else s[:37] + "..."


def report(node: Any, title: str | None = None, fmt: str = "html") -> str:
    """Documentation of a graph: diagram, typed node table and each block's docstring.
    ``fmt`` is ``"html"`` (with the SVG inline) or ``"md"`` (Markdown)."""
    import inspect as _inspect

    from .graph import Block, Source

    title = title or node.name
    env = node.validate()
    leaves = node.nodes()
    seen: set[int] = set()
    entries = []
    for n in leaves:
        key = id(getattr(n, "defn", n))
        if key in seen:
            continue
        seen.add(key)
        doc = ""
        if isinstance(n, Block):
            doc = _inspect.getdoc(n.defn.fn) or _inspect.getdoc(n.defn) or ""
            kind = n.kind
            cfg = ", ".join(f"{k}={v!r}" for k, v in n.config.items())
            ins = (
                ", ".join(f"{i.name}: {describe(i.type)}" for i in n.defn.inputs) or "—"
            )
        elif type(n).__name__ == "Select":
            kind, cfg, ins = "select", "", describe(n.in_type)
        else:
            kind, cfg, ins = "source", "", "—"
        out = (
            describe(n.out_type.resolve(env))
            if hasattr(n.out_type, "resolve")
            else describe(n.out_type)
        )
        entries.append((n.name, kind, ins, out, cfg, doc))
    if fmt == "md":
        lines = [
            f"# {title}",
            "",
            "```",
            node.describe(),
            "```",
            "",
            "| block | kind | inputs | output | configuration |",
            "|---|---|---|---|---|",
        ]
        for name, kind, ins, out, cfg, doc in entries:
            lines.append(f"| `{name}` | {kind} | `{ins}` | `{out}` | `{cfg}` |")
        lines.append("")
        for name, kind, ins, out, cfg, doc in entries:
            if doc:
                lines += [f"### {name}", "", doc, ""]
        return "\n".join(lines)
    from .draw import Diagram

    rows = "".join(
        f"<tr><td style='{_TD}'><b>{html.escape(name)}</b></td><td style='{_TD}'>{kind}</td>"
        f"<td style='{_TD}'>{html.escape(ins)}</td><td style='{_TD}'>{html.escape(out)}</td>"
        f"<td style='{_TD}'>{html.escape(cfg)}</td>"
        f"<td style='{_TD};white-space:pre-wrap;max-width:420px'>{html.escape(doc)}</td></tr>"
        for name, kind, ins, out, cfg, doc in entries
    )
    return (
        f"<div style='font-family:system-ui,sans-serif'><h2>{html.escape(title or node.name)}</h2>"
        f"<div style='overflow:auto;background:white;padding:4px'>{Diagram(node).svg}</div>"
        f"<table style='{_CSS};margin-top:10px'><tr><th style='{_TD}'>block</th><th style='{_TD}'>kind</th>"
        f"<th style='{_TD}'>inputs</th><th style='{_TD}'>output</th><th style='{_TD}'>configuration</th>"
        f"<th style='{_TD}'>what it does</th></tr>{rows}</table></div>"
    )

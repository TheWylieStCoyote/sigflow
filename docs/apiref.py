"""Generate the API reference pages (docs/reference/api/*.md) from the live package.

Run by ``docs/build_docs.py``. Each page lists objects by import path; what is shown
comes from introspection, so it cannot drift from the code:

- a block (``BlockDef``): how to configure it, its inputs' and output's contracts,
  its kind, and its docstring;
- a function: its signature and docstring;
- a class: its constructor, docstring, and the public methods and properties it
  defines;
- a module: its docstring, then every name in its ``__all__``.

``tests/test_docs.py`` checks that every name ``sigflow`` exports is on some page.
"""

from __future__ import annotations

import importlib
import inspect
import re
from typing import Any

# (file stem, title, introduction, [import paths]); a path ending in ".*" lists a
# whole module (its docstring, then every name in its __all__).
PAGES: list[tuple[str, str, str, list[str]]] = [
    (
        "blocks",
        "Blocks",
        (
            "Turning functions into typed blocks, and the annotations a block's "
            "signature uses. See [Blocks](../../concepts/blocks.md)."
        ),
        [
            "sigflow.block",
            "sigflow.source",
            "sigflow.tap",
            "sigflow.lift",
            "sigflow.BlockDef",
            "sigflow.Framed",
            "sigflow.Stream",
            "sigflow.Stateful",
        ],
    ),
    (
        "composing",
        "Composing graphs",
        (
            "Operators build these nodes for you (`>>` a pipeline, `|` a fan-out, `&` a "
            "zip); the functions build the rest. See "
            "[Composing graphs](../../guide/composing.md)."
        ),
        [
            "sigflow.Node",
            "sigflow.Block",
            "sigflow.Source",
            "sigflow.Pipeline",
            "sigflow.Fanout",
            "sigflow.Zip",
            "sigflow.Merge",
            "sigflow.Select",
            "sigflow.port",
            "sigflow.merge",
            "sigflow.as_node",
            "sigflow.from_iterable",
            "sigflow.feedback",
            "sigflow.when",
            "sigflow.switch",
            "sigflow.align",
            "sigflow.per_key",
            "sigflow.gather",
            "sigflow.parallel",
            "sigflow.validate",
            "sigflow.Graph",
            "sigflow.builder.Ref",
        ],
    ),
    (
        "running",
        "Running graphs",
        (
            "The entry points and the handle of a background run. The options every "
            "entry point takes are tabled in [Running graphs](../running.md)."
        ),
        [
            "sigflow.run",
            "sigflow.arun",
            "sigflow.iterate",
            "sigflow.iterate_sync",
            "sigflow.Run",
            "sigflow.runs",
            "sigflow.stop_all",
            "sigflow.Monitor",
        ],
    ),
    (
        "frames",
        "Frames, tags and provenance",
        (
            "What flows through a graph. See [Frames](../../concepts/frames.md) and "
            "[Lineage](../../concepts/lineage.md)."
        ),
        [
            "sigflow.Frame",
            "sigflow.Issue",
            "sigflow.MISSING",
            "sigflow.Tag",
            "sigflow.Tags",
            "sigflow.TimeSpan",
            "sigflow.Provenance",
            "sigflow.NO_PROV",
        ],
    ),
    (
        "contracts",
        "Contracts and tag markers",
        (
            "Signal types, the shorthands for common shapes, and the markers that put "
            "tags in a signature. See [Contracts](../../concepts/contracts.md)."
        ),
        [
            "sigflow.Sig",
            "sigflow.SignalType",
            "sigflow.TupleType",
            "sigflow.Any",
            "sigflow.Scalar",
            "sigflow.Vector",
            "sigflow.Matrix",
            "sigflow.Tensor",
            "sigflow.CVector",
            "sigflow.CMatrix",
            "sigflow.requires",
            "sigflow.accepts",
            "sigflow.provides",
            "sigflow.may_provide",
            "sigflow.promoted",
            "sigflow.clears",
            "sigflow.keep",
            "sigflow.drop",
            "sigflow.carries",
            "sigflow.unify",
            "sigflow.Env",
            "sigflow.TypeMismatch",
            "sigflow.TagWarning",
            "sigflow.Ratio",
            "sigflow.RatioMismatch",
            "sigflow.dims.*",
        ],
    ),
    (
        "control",
        "Live control and debugging",
        (
            "Tuning and inspecting a graph while it runs. See "
            "[Debugging a running graph](../../guide/debugging.md)."
        ),
        [
            "sigflow.Knob",
            "sigflow.commit",
            "sigflow.Button",
            "sigflow.Probe",
            "sigflow.Breakpoint",
            "sigflow.dashboard",
            "sigflow.require_widgets",
        ],
    ),
    (
        "persistence",
        "Persistence, replay and lineage",
        (
            "Saving frames, recomputing one, resuming a run, and querying lineage. See "
            "[Persistence and replay](../../guide/persistence.md)."
        ),
        [
            "sigflow.replay",
            "sigflow.ReplayMismatch",
            "sigflow.Checkpoint",
            "sigflow.load_checkpoint",
            "sigflow.persist.save_frame",
            "sigflow.persist.load_frame",
            "sigflow.Lineage",
        ],
    ),
    (
        "backends",
        "Parallel pools, devices and labelled arrays",
        (
            "See [Parallel blocks](../../guide/parallel.md), "
            "[Array backends and GPU](../../guide/array-backends.md) and "
            "[Labelled arrays](../../guide/xarray.md)."
        ),
        [
            "sigflow.pools.BACKENDS",
            "sigflow.pools.available",
            "sigflow.pools.free_threaded",
            "sigflow.pools.gil_enabled",
            "sigflow.to_device",
            "sigflow.to_host",
            "sigflow.default_labels",
        ],
    ),
    ("dsp", "sigflow.dsp", "", ["sigflow.dsp.*"]),
    ("linalg", "sigflow.linalg", "", ["sigflow.linalg.*"]),
    (
        "interpolate",
        "sigflow.interpolate",
        "See [Interpolation and resampling](../../guide/interpolation.md).",
        ["sigflow.interpolate.*"],
    ),
    (
        "integrate",
        "sigflow.integrate",
        "See [Integration across frames](../../guide/integration.md).",
        ["sigflow.integrate.*"],
    ),
    (
        "stats",
        "sigflow.stats",
        "See [Statistics across frames](../../guide/statistics.md).",
        ["sigflow.stats.*"],
    ),
    (
        "io",
        "sigflow.io",
        "See [Files, sockets and GNU Radio](../../guide/io.md).",
        ["sigflow.io.*"],
    ),
    ("plot", "sigflow.plot", "", ["sigflow.plot.*"]),
    (
        "drawing",
        "Diagrams and reports",
        "",
        [
            "sigflow.draw.draw",
            "sigflow.draw.Diagram",
            "sigflow.draw.to_dot",
            "sigflow.notebook.report",
            "sigflow.notebook.RunReport",
        ],
    ),
]


def resolve(path: str) -> Any:
    mod_name, _, attr = path.rpartition(".")
    try:
        return getattr(importlib.import_module(mod_name), attr)
    except (ImportError, AttributeError):
        return importlib.import_module(path)


# --------------------------------------------------------------------------- #
# docstrings: reStructuredText-flavoured plain text -> Markdown
# --------------------------------------------------------------------------- #

_ROLE = re.compile(r":(?:func|class|meth|mod|attr|data|exc|obj):`~?([^`]+)`")
_ALIGNED = re.compile(r"\S {3,}\S")  # a column gap: a table or an aligned list


def markdown(doc: str | None) -> str:
    """A docstring as Markdown: roles become code, ``x`` becomes `x`, and runs of
    aligned lines (tables, option lists, ``::`` blocks) stay preformatted."""
    if not doc:
        return ""
    text = _ROLE.sub(r"``\1``", inspect.cleandoc(doc))
    out: list[str] = []
    block: list[str] = []
    literal = False  # inside a "::" block or an indented run

    def flush() -> None:
        if block:
            while block and not block[-1].strip():
                block.pop()
            out.extend(
                ["", "```text", *(b.replace("``", "") for b in block), "```", ""]
            )
            block.clear()

    for line in text.splitlines():
        if literal or block:
            if line.strip() and not line.startswith(" ") and not _ALIGNED.search(line):
                literal = False
                flush()
            else:
                block.append(line)
                continue
        if _ALIGNED.search(line) or (line.startswith("    ") and line.strip()):
            block.append(line)
            continue
        if line.rstrip().endswith("::"):
            out.append(line.rstrip()[:-1])
            literal = True
            continue
        out.append(line.replace("``", "`"))
    flush()
    return "\n".join(out).strip()


# --------------------------------------------------------------------------- #
# objects
# --------------------------------------------------------------------------- #


def _sig(obj: Any) -> str:
    try:
        text = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"
    return re.sub(r" -> '?None'?$", "", text) if inspect.isclass(obj) else text


def _first_paragraph(doc: str | None) -> str:
    return markdown((inspect.cleandoc(doc or "").split("\n\n") or [""])[0])


def render_block(name: str, defn: Any) -> list[str]:
    kind = {"map": "map block", "stream": "stream block", "source": "source"}.get(
        defn.kind, f"{defn.kind} block"
    )
    if getattr(defn, "tap", False):
        kind = "tap"
    params = ", ".join(
        f"{p.name}={p.default!r}"
        if p.default is not inspect.Parameter.empty
        else p.name
        for p in defn.config.values()
    )
    lines = [
        f"### `{name}`",
        "",
        f"*{kind}* · `{name.rsplit('.', 1)[-1]}({params})`",
        "",
    ]
    if defn.inputs:
        lines += ["| input | contract | |", "|---|---|---|"]
        for i in defn.inputs:
            how = ", ".join(
                w for w, on in (("stream", i.stream), ("framed", i.framed)) if on
            )
            lines.append(f"| `{i.name}` | `{i.type}` | {how} |")
        lines.append("")
    ports = getattr(defn, "port_names", ())
    items = getattr(defn.out_type, "items", ())
    if ports and len(ports) == len(items):  # named output ports: one row each
        lines += ["| output port | contract |", "|---|---|"]
        lines += [f"| `{p}` | `{t}` |" for p, t in zip(ports, items)]
        lines.append("")
    else:
        lines += [f"**Output:** `{defn.out_type}`", ""]
    lines += [markdown(defn.__doc__), ""]
    return lines


def render_function(name: str, fn: Any) -> list[str]:
    short = name.rsplit(".", 1)[-1]
    return [
        f"### `{name}`",
        "",
        "```python",
        f"{short}{_sig(fn)}",
        "```",
        "",
        markdown(fn.__doc__),
        "",
    ]


def render_class(name: str, cls: Any) -> list[str]:
    short = name.rsplit(".", 1)[-1]
    lines = [f"### `{name}`", "", "```python", f"class {short}{_sig(cls)}", "```", ""]
    lines += [markdown(cls.__doc__), ""]
    members: list[str] = []
    for attr, value in cls.__dict__.items():
        if attr.startswith("_"):
            continue
        if isinstance(value, property):
            doc = _first_paragraph(value.fget.__doc__ if value.fget else None)
            members.append(f"- `{attr}` (property){': ' + doc if doc else ''}")
        elif inspect.isfunction(value) or isinstance(
            value, (staticmethod, classmethod)
        ):
            fn = (
                value.__func__
                if isinstance(value, (staticmethod, classmethod))
                else value
            )
            doc = _first_paragraph(fn.__doc__)
            members.append(f"- `{attr}{_sig(fn)}`{' — ' + doc if doc else ''}")
    if members:
        lines += ["**Methods and properties**", "", *members, ""]
    return lines


def render_value(name: str, value: Any) -> list[str]:
    doc = value.__doc__ if not isinstance(value, (tuple, list, dict, str)) else None
    if doc == type(value).__doc__ and type(value).__module__ == "builtins":
        doc = None
    return [f"### `{name}`", "", f"`{value!r}`", "", markdown(doc), ""]


def render(name: str, obj: Any) -> list[str]:
    from sigflow.block import BlockDef

    if isinstance(obj, BlockDef):
        return render_block(name, obj)
    if inspect.isclass(obj):
        return render_class(name, obj)
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return render_function(name, obj)
    return render_value(name, obj)


def render_module(path: str) -> list[str]:
    mod = importlib.import_module(path)
    lines = [markdown(mod.__doc__), ""]
    for attr in getattr(mod, "__all__", []):
        lines += render(f"{path}.{attr}", getattr(mod, attr))
    return lines


def page(title: str, intro: str, paths: list[str]) -> str:
    lines = [f"# {title}", "", "*Generated from the code by `docs/apiref.py`.*", ""]
    if intro:
        lines += [intro, ""]
    for path in paths:
        if path.endswith(".*"):
            lines += render_module(path[:-2])
        else:
            lines += render(path, resolve(path))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).rstrip() + "\n"


def covered() -> set[str]:
    """Every name the reference pages document, as ``module.name``."""
    names: set[str] = set()
    for _, _, _, paths in PAGES:
        for path in paths:
            if path.endswith(".*"):
                mod = importlib.import_module(path[:-2])
                names |= {f"{path[:-2]}.{a}" for a in getattr(mod, "__all__", [])}
            else:
                names.add(path)
    return names


def write(directory: str) -> list[str]:
    """Write every page of the reference into ``directory``, and remove any other
    page there: the directory holds only generated pages."""
    import os

    os.makedirs(directory, exist_ok=True)
    written = []
    for stem, title, intro, paths in PAGES:
        target = os.path.join(directory, f"{stem}.md")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(page(title, intro, paths))
        written.append(stem)
    # pages no longer generated (a module removed, another branch's pages) go
    for name in os.listdir(directory):
        if name.endswith(".md") and name[:-3] not in written:
            os.remove(os.path.join(directory, name))
    return written

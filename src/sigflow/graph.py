"""Graph construction with operators, and compile-time type validation.

Operators (all produce new, immutable graph nodes):

    a >> b         chain: output of ``a`` feeds ``b``
    a | b          fan-out: every frame goes to both; output is a tuple frame
    a & b          zip: lock-step join of two streams into a tuple frame
    merge(a, b)    interleave two streams as they arrive (async-friendly)

Types are checked **when the graph is built** – a mismatch raises
:class:`~sigflow.types.TypeMismatch` immediately, with the names of the
offending nodes, before any data flows.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Mapping

from .block import BlockDef
from .frame import frozen
from .ratio import Ratio, RatioMismatch, as_ratio, leading_dim
from .types import (
    DVar,
    Env,
    SignalType,
    TupleType,
    TypeMismatch,
    as_type,
    describe,
    scoped,
    slice_shape,
    unify,
)

_ids = itertools.count(1)

__all__ = [
    "Node",
    "Block",
    "Source",
    "Pipeline",
    "Fanout",
    "Zip",
    "Merge",
    "Select",
    "Slice",
    "StreamSlice",
    "PortRef",
    "port",
    "Align",
    "align",
    "Parallel",
    "parallel",
    "Feedback",
    "feedback",
    "When",
    "when",
    "Switch",
    "switch",
    "merge",
    "as_node",
    "from_iterable",
    "validate",
    "fresh",
]


class Node:
    """Base class of everything that can be placed in a graph."""

    in_type: Any = None  # None => source
    out_type: Any = SignalType()
    name: str = "node"

    # -- stream-rate contract --------------------------------------------- #
    port_names: tuple[str | None, ...] = ()

    def ratio_for(self, in_type: Any) -> Ratio:
        """Output frames per input frame when fed ``in_type`` (see :mod:`sigflow.ratio`)."""
        return Ratio.ANY

    @property
    def ratio(self) -> Ratio:
        """Output frames per input frame (per source frame for a graph with a source)."""
        return self.ratio_for(self.in_type)

    # -- operators --------------------------------------------------------- #

    def __rshift__(self, other: Any) -> "Pipeline":
        return Pipeline([self, as_node(other)])

    def __getitem__(self, key: Any) -> "Pipeline":
        """Tuple outputs: select port(s) — ``eig[0]``, ``eig["vals"]``, ``(a | b | c)[0, 2]``,
        ``(a | b | c)[1:]``. Array outputs: slice the data — ``fft[:512]``, ``cube[:, 20:200]``,
        ``iq[::4]`` — with the shape tracked statically and sample annotations sliced in step."""
        if isinstance(self.out_type, TupleType):
            return Pipeline([self, Select(self, key)])
        if isinstance(self.out_type, SignalType) and (
            self.out_type.is_array
            or self.out_type.dtype is Any
            or isinstance(self.out_type.dtype, DVar)
        ):
            return Pipeline([self, Slice(self, key)])
        raise TypeError(
            f"{self.name} produces {describe(self.out_type)}: [] slices arrays or selects ports"
        )

    # -- stream slicing ----------------------------------------------------- #
    def take(self, n: int) -> "Pipeline":
        """Only the first ``n`` output frames (then the run ends)."""
        return self >> _take(n=n)

    def skip(self, n: int) -> "Pipeline":
        """Drop the first ``n`` output frames."""
        return self >> _skip(n=n)

    def every(self, k: int, offset: int = 0) -> "Pipeline":
        """Keep one frame in ``k`` (starting at ``offset``)."""
        return self >> _every(k=k, offset=offset)

    def until(self, pred: Any, inclusive: bool = False) -> "Pipeline":
        """Stop at the first frame for which ``pred(frame)`` is true."""
        return self >> _until(pred=pred, inclusive=inclusive)

    def while_(self, pred: Any) -> "Pipeline":
        """Continue while ``pred(frame)`` holds."""
        return self >> _until(pred=lambda f: not pred(f), inclusive=False)

    def __rrshift__(self, other: Any) -> "Pipeline":
        return Pipeline([as_node(other), self])

    def __or__(self, other: Any) -> "Fanout":
        return Fanout([self, as_node(other)])

    def __ror__(self, other: Any) -> "Fanout":
        return Fanout([as_node(other), self])

    def __and__(self, other: Any) -> "Zip":
        return Zip([self, as_node(other)])

    def __rand__(self, other: Any) -> "Zip":
        return Zip([as_node(other), self])

    # -- structure --------------------------------------------------------- #
    def nodes(self) -> list["Node"]:
        """Flat list of leaf nodes (blocks and sources) in this graph."""
        return [self]

    @property
    def is_source(self) -> bool:
        return self.in_type is None

    def validate(self, env: Env | None = None) -> Env:
        return env if env is not None else Env()

    def accept(self, produced: Any, env: Env, where: str) -> None:
        """Check that ``produced`` can feed this node's input (binds symbols in env)."""
        unify(produced, self.in_type, env, where=where)

    def resolve_ports(self, upstream: "Node") -> "Node":
        """Replace unresolved ``port(...)`` references at this node's input with typed
        selections now that the producer is known.  Default: nothing to resolve."""
        return self

    # -- running ----------------------------------------------------------- #
    def run(self, **kw: Any) -> list:
        """Run synchronously; returns the list of output frames (see :func:`sigflow.runtime.run`)."""
        from .runtime import run

        return run(self, **kw)

    async def arun(self, **kw: Any) -> list:
        from .runtime import arun

        return await arun(self, **kw)

    def start(self, paused: bool = False, **kw: Any) -> Any:
        """Run in the background and return a :class:`~sigflow.control.Run` handle
        with ``pause() / resume() / step() / stop() / wait()`` and ``controls()`` for notebooks."""
        from .control import Run

        run = Run(self, paused=paused, **kw)
        self.last_monitor = run.monitor  # type: ignore[attr-defined]
        return run

    def __call__(self, items: Any, **kw: Any) -> list:
        """Apply a source-less graph to in-memory data: ``front_end([x1, x2])`` returns
        the output frames; a single array is treated as one frame. Blocks used alone are
        still called as plain functions (see :class:`~sigflow.block.BlockDef`)."""
        if self.in_type is None:
            raise TypeError(f"{self.name} has its own source; call run() instead")
        import numpy as _np

        if (
            isinstance(items, (_np.ndarray,))
            or not hasattr(items, "__iter__")
            or isinstance(items, (str, bytes, dict))
        ):
            items = [items]
        from .runtime import run as _run

        return _run(
            from_iterable(list(items), type=self.in_type, name="input") >> self, **kw
        )

    apply = __call__

    def __aiter__(self):
        from .runtime import iterate

        return iterate(self)

    def __iter__(self):
        from .runtime import iterate_sync

        return iterate_sync(self)

    def describe(self, indent: str = "") -> str:
        it = "" if self.in_type is None else f"{describe(self.in_type)} -> "
        return f"{indent}{self.name}: {it}{describe(self.out_type)}"

    def _repr_html_(self) -> str:
        from .notebook import graph_html

        return graph_html(self)

    def draw(self, types: bool = True) -> Any:
        """SVG diagram of the graph (displays inline in Jupyter; ``.save(path)`` to write it)."""
        from .draw import Diagram

        return Diagram(self, types=types)

    def report(self, title: str | None = None, fmt: str = "html") -> str:
        """Documentation of this graph (diagram + typed table + block docstrings); see
        :func:`sigflow.notebook.report`."""
        from .notebook import report

        return report(self, title, fmt)

    def to_dot(self, types: bool = True) -> str:
        """Graphviz DOT source for the graph."""
        from .draw import to_dot

        return to_dot(self, types=types)

    def __repr__(self) -> str:
        return self.describe()


class Block(Node):
    """A :class:`BlockDef` placed in a graph with a fixed configuration."""

    def __init__(
        self,
        defn: BlockDef,
        config: Mapping[str, Any] | None = None,
        label: str | None = None,
    ) -> None:
        self.defn = defn
        self.config = frozen(
            defn.check_config(config or {})
        )  # full (defaults filled): display, provenance
        self.given = frozen(
            dict(config or {})
        )  # what the caller passed: used for the call
        from .control import Knob

        self.live = any(isinstance(v, Knob) for v in self.config.values())
        self.label = label
        self.name = str(label or defn.name)
        self.id = next(_ids)
        in_t, out_t = defn.in_type, defn.out_type
        if defn.refine is not None:
            from .control import live as _live

            refined = defn.refine(**_live(self.config))
            if (
                isinstance(refined, tuple)
                and len(refined) == 2
                and not isinstance(refined, TupleType)
            ):
                in_t, out_t = (
                    (as_type(refined[0]) if refined[0] is not None else None),
                    as_type(refined[1]),
                )
            else:
                out_t = as_type(refined)
        self.in_type = scoped(in_t, str(self.id))
        self.out_type = scoped(out_t, str(self.id))
        self.port_names = tuple(defn.port_names)

    @property
    def kind(self) -> str:
        return self.defn.kind

    def ratio_for(self, in_type: Any) -> Ratio:
        spec = self.defn.ratio_spec
        if callable(spec):
            from .control import live as _live

            spec = spec(n_in=leading_dim(in_type), **_live(self.config))
        return as_ratio(spec)

    def named(self, label: str) -> "Block":
        return Block(self.defn, self.config, label)

    def clone(self) -> "Block":
        """A copy with a fresh identity (new symbol scope); stateful subclasses share state."""
        import copy

        new = copy.copy(self)
        new.given = self.given
        new.id = next(_ids)
        new.in_type = scoped(
            self.defn.in_type if self.defn.refine is None else _unscope(self.in_type),
            str(new.id),
        )
        new.out_type = scoped(_unscope(self.out_type), str(new.id))
        return new

    def describe(self, indent: str = "", ratio_in: Any = None) -> str:
        cfg = ", ".join(f"{k}={v!r}" for k, v in self.config.items())
        it = "" if self.in_type is None else f"{describe(self.in_type)} -> "
        r = self.ratio_for(ratio_in if ratio_in is not None else self.in_type)
        tail = f"  [{r}]" if self.kind == "stream" and r and r != Ratio.ONE else ""
        return f"{indent}{self.name}({cfg}): {it}{describe(self.out_type)}{tail}"


class Source(Node):
    """A source built from an existing iterable / async iterable / factory."""

    def ratio_for(self, in_type: Any) -> Ratio:
        return Ratio.ONE

    in_type = None

    def __init__(self, obj: Any, type: Any = None, name: str | None = None) -> None:
        self.obj = obj
        self.id = next(_ids)
        self.out_type = scoped(
            as_type(type) if type is not None else SignalType(), str(self.id)
        )
        self.name = str(name or getattr(obj, "__name__", type_name(obj)))

    def clone(self) -> "Source":
        return Source(self.obj, _unscope(self.out_type), self.name)

    def __iter_obj__(self) -> Any:
        o = self.obj
        if callable(o) and not hasattr(o, "__iter__") and not hasattr(o, "__aiter__"):
            o = o()
        return o


def type_name(o: Any) -> str:
    return type(o).__name__


class Pipeline(Node):
    """A chain of nodes. Validated on construction."""

    stages: list[Node]

    def __init__(self, stages: Iterable[Node]) -> None:
        seen: set[int] = set()
        self._init([fresh(s, seen) for s in stages])

    def _init(self, stages: Iterable[Node]) -> "Pipeline":
        flat: list[Node] = []
        for s in stages:
            flat.extend(s.stages if isinstance(s, Pipeline) else [s])
        if not flat:
            raise ValueError("empty pipeline")
        for i in range(1, len(flat)):
            flat[i] = flat[i].resolve_ports(flat[i - 1])
        self.stages = flat
        self.name = " >> ".join(s.name for s in flat)
        self.in_type = flat[0].in_type
        self.out_type = flat[-1].out_type
        self.port_names = tuple(getattr(flat[-1], "port_names", ()))
        self.env = self.validate()
        return self

    def validate(self, env: Env | None = None) -> Env:
        env = env if env is not None else Env()
        prev: Node | None = None
        for s in self.stages:
            env = s.validate(env)
            if prev is not None:
                if s.in_type is None:
                    raise TypeMismatch(
                        f"{s.name} is a source and cannot follow {prev.name}"
                    )
                try:
                    s.accept(prev.out_type, env, f"{prev.name} -> {s.name}")
                except TypeMismatch as e:
                    raise TypeMismatch(str(e)) from None
                s.ratio_for(
                    prev.out_type
                )  # branch rates are checked with the real input
            prev = s
        return env

    def ratio_for(self, in_type: Any) -> Ratio:
        r = Ratio.ONE
        t = in_type
        for s in self.stages:
            r = r * s.ratio_for(t if s.in_type is not None else None)
            t = s.out_type
        return r

    def accept(self, produced: Any, env: Env, where: str) -> None:
        self.stages[0].accept(produced, env, where)

    def resolve_ports(self, upstream: Node) -> Node:
        first = self.stages[0].resolve_ports(upstream)
        if first is self.stages[0]:
            return self
        return Pipeline.__new__(Pipeline)._init([first] + self.stages[1:])

    def nodes(self) -> list[Node]:
        return [n for s in self.stages for n in s.nodes()]

    def describe(self, indent: str = "") -> str:
        lines, prev = [], None
        for s in self.stages:
            lines.append(
                s.describe(indent, ratio_in=prev.out_type)
                if isinstance(s, Block) and prev is not None
                else s.describe(indent)
            )
            prev = s
        return "\n".join(lines)


class Fanout(Node):
    """Broadcast each input frame to several branches; outputs a tuple frame."""

    branches: list[Node]

    def __init__(self, branches: Iterable[Node]) -> None:
        seen: set[int] = set()
        branches = [fresh(x, seen) for x in branches]
        flat: list[Node] = []
        for b in branches:
            flat.extend(b.branches if isinstance(b, Fanout) else [b])
        self.branches = flat
        self.name = "(" + " | ".join(b.name for b in flat) + ")"
        ins = [b.in_type for b in flat]
        if any(i is None for i in ins):
            raise TypeMismatch(
                "fan-out branches must accept input; use '&' to join sources"
            )
        self.in_type = _common_input(ins, self.name)
        self.out_type = TupleType(tuple(b.out_type for b in flat))
        self.port_names = _member_names(flat)

    def validate(self, env: Env | None = None) -> Env:
        env = env if env is not None else Env()
        for b in self.branches:
            env = b.validate(env)
        return env

    def accept(self, produced: Any, env: Env, where: str) -> None:
        for b in self.branches:
            b.accept(produced, env, f"{where} | {b.name}")

    def ratio_for(self, in_type: Any) -> Ratio:
        return _common_ratio(self.branches, in_type, "fan-out")

    def resolve_ports(self, upstream: Node) -> Node:
        resolved = [b.resolve_ports(upstream) for b in self.branches]
        if all(r is b for r, b in zip(resolved, self.branches)):
            return self
        return Fanout(resolved)

    def nodes(self) -> list[Node]:
        return [n for b in self.branches for n in b.nodes()]

    def describe(self, indent: str = "") -> str:
        lines = [f"{indent}fan-out -> {describe(self.out_type)}"]
        for b in self.branches:
            lines.append(b.describe(indent + "  | "))
        return "\n".join(lines)


class Zip(Node):
    """Lock-step join of several streams into tuple frames."""

    parts: list[Node]

    def __init__(self, parts: Iterable[Node]) -> None:
        seen: set[int] = set()
        parts = [fresh(x, seen) for x in parts]
        flat: list[Node] = []
        for p in parts:
            flat.extend(p.parts if isinstance(p, Zip) else [p])
        self.parts = flat
        self.name = "(" + " & ".join(p.name for p in flat) + ")"
        ins = [p.in_type for p in flat]
        if all(i is None for i in ins):
            self.in_type = None
        elif any(i is None for i in ins):
            raise TypeMismatch(
                "cannot zip a source with a non-source; wrap inputs in a pipeline"
            )
        else:
            self.in_type = _common_input(ins, self.name)
        self.out_type = TupleType(tuple(p.out_type for p in flat))
        self.port_names = _member_names(flat)

    def validate(self, env: Env | None = None) -> Env:
        env = env if env is not None else Env()
        for p in self.parts:
            env = p.validate(env)
        return env

    def accept(self, produced: Any, env: Env, where: str) -> None:
        for p in self.parts:
            p.accept(produced, env, f"{where} & {p.name}")

    def ratio_for(self, in_type: Any) -> Ratio:
        if self.in_type is None:
            return Ratio.ONE  # lock-step over sources: one tuple per source frame
        return _common_ratio(self.parts, in_type, "zip")

    def resolve_ports(self, upstream: Node) -> Node:
        if self.in_type is None:
            return self
        resolved = [p.resolve_ports(upstream) for p in self.parts]
        if all(r is p for r, p in zip(resolved, self.parts)):
            return self
        return Zip(resolved)

    def nodes(self) -> list[Node]:
        return [n for p in self.parts for n in p.nodes()]

    def describe(self, indent: str = "") -> str:
        lines = [f"{indent}zip -> {describe(self.out_type)}"]
        for p in self.parts:
            lines.append(p.describe(indent + "  & "))
        return "\n".join(lines)


class Feedback(Node):
    """A closed loop across blocks (built by :func:`feedback`).

    ``body`` takes a tuple frame ``(x, state)`` and returns ``(y, new_state)``;
    the runtime feeds ``new_state`` back as ``state`` for the next input frame,
    starting from ``init``. The body is any 1:1 node (block or pipeline); its
    first port is the loop's output, its second port the fed-back state::

        agc = feedback(measure_and_scale, init=1.0)      # gain flows around the loop
    """

    def ratio_for(self, in_type: Any) -> Ratio:
        return Ratio.ONE

    def __init__(self, body: Node, init: Any, name: str | None = None) -> None:
        self.body = body
        self.init = init
        if not isinstance(body.in_type, TupleType) or len(body.in_type) != 2:
            raise TypeMismatch(
                f"feedback body must take (input, state); {body.name} takes "
                f"{describe(body.in_type)}"
            )
        if not isinstance(body.out_type, TupleType) or len(body.out_type) != 2:
            raise TypeMismatch(
                f"feedback body must return (output, state); {body.name} returns "
                f"{describe(body.out_type)}"
            )
        unify(
            body.out_type.items[1],
            body.in_type.items[1],
            Env(),
            f"{body.name} state loop",
        )
        self.in_type = body.in_type.items[0]
        self.out_type = body.out_type.items[0]
        self.state_type = body.in_type.items[1]
        self.name = name or f"feedback({body.name})"

    def validate(self, env: Env | None = None) -> Env:
        env = env if env is not None else Env()
        return self.body.validate(env)

    def nodes(self) -> list[Node]:
        return self.body.nodes()

    def describe(self, indent: str = "") -> str:
        init = repr(self.init)
        if len(init) > 40:
            init = f"<{type(self.init).__name__}{getattr(self.init, 'shape', '')}>"
        return (
            f"{indent}feedback(init={init}): {describe(self.in_type)} -> {describe(self.out_type)}\n"
            + self.body.describe(indent + "  ↻ ")
        )


def feedback(body: Any, init: Any, name: str | None = None) -> Feedback:
    """Close a loop: ``body`` maps ``(x, state) -> (y, new_state)``; see :class:`Feedback`."""
    return Feedback(as_node(body), init, name)


class When(Node):
    """Per-frame routing (built by :func:`when`): frames for which ``pred(frame)`` is
    true go through ``then``, the others through ``otherwise`` (default: pass through).
    Both branches must be 1:1; output order follows input order."""

    def ratio_for(self, in_type: Any) -> Ratio:
        return Ratio.ONE

    def __init__(
        self,
        pred: Any,
        then: Node,
        otherwise: Node | None = None,
        name: str | None = None,
    ) -> None:
        self.pred = pred
        self.then = then
        self.otherwise = otherwise
        self.in_type = (
            then.in_type
            if otherwise is None
            else _common_input([then.in_type, otherwise.in_type], "when")
        )
        if otherwise is None:
            self.out_type = (
                then.out_type if then.out_type == then.in_type else SignalType()
            )
        else:
            self.out_type = (
                then.out_type if then.out_type == otherwise.out_type else SignalType()
            )
        self.name = name or f"when({getattr(pred, '__name__', 'pred')})"

    def validate(self, env: Env | None = None) -> Env:
        env = env if env is not None else Env()
        env = self.then.validate(env)
        return self.otherwise.validate(env) if self.otherwise is not None else env

    def accept(self, produced: Any, env: Env, where: str) -> None:
        self.then.accept(produced, env, f"{where} [then]")
        if self.otherwise is not None:
            self.otherwise.accept(produced, env, f"{where} [otherwise]")

    def nodes(self) -> list[Node]:
        return self.then.nodes() + (
            self.otherwise.nodes() if self.otherwise is not None else []
        )

    def describe(self, indent: str = "") -> str:
        lines = [
            f"{indent}{self.name}: {describe(self.in_type)} -> {describe(self.out_type)}",
            self.then.describe(indent + "  ? "),
        ]
        if self.otherwise is not None:
            lines.append(self.otherwise.describe(indent + "  : "))
        return "\n".join(lines)


def when(pred: Any, then: Any, otherwise: Any = None, name: str | None = None) -> When:
    """Route frames: ``when(lambda f: f.failed, quarantine, otherwise=main)``.
    ``pred`` receives the :class:`Frame`."""
    return When(
        pred, as_node(then), as_node(otherwise) if otherwise is not None else None, name
    )


class Switch(Node):
    """Keyed routing (built by :func:`switch`): ``selector(frame)`` picks the branch each
    frame goes through; unselected branches stay idle. Branch outputs are merged into one
    stream. With ``ordered=False`` (default) branches run concurrently as real streams —
    they may re-frame, window or drop — and outputs appear in completion order; with
    ``ordered=True`` output order follows input order and every branch must be 1:1."""

    def __init__(
        self,
        selector: Any,
        branches: Mapping[Any, Node],
        default: Any = None,
        ordered: bool = False,
        name: str | None = None,
    ) -> None:
        if not branches:
            raise ValueError("switch() needs at least one branch")
        self.key_of = (
            (lambda f, k=selector: f[k]) if isinstance(selector, str) else selector
        )
        self.selector = selector
        self.branches = dict(branches)
        self.default = default  # Node, "pass", or None (drop)
        self.ordered = ordered
        all_nodes = list(self.branches.values()) + (
            [default] if isinstance(default, Node) else []
        )
        self.in_type = _common_input([b.in_type for b in all_nodes], "switch")
        outs = [b.out_type for b in all_nodes] + (
            [self.in_type] if default == "pass" else []
        )
        self.out_type = outs[0] if all(o == outs[0] for o in outs) else SignalType()
        sel = (
            selector
            if isinstance(selector, str)
            else getattr(selector, "__name__", "selector")
        )
        self.name = name or f"switch({sel})"
        if ordered:
            for k, b in self.branches.items():
                if b.ratio_for(self.in_type) not in (Ratio.ONE, Ratio.DOWN, Ratio.ANY):
                    raise RatioMismatch(
                        f"switch(ordered=True): branch {k!r} emits {b.ratio_for(self.in_type)} frames "
                        f"per input; ordered routing needs 1:1 branches"
                    )

    def _all(self) -> list[Node]:
        return list(self.branches.values()) + (
            [self.default] if isinstance(self.default, Node) else []
        )

    def validate(self, env: Env | None = None) -> Env:
        env = env if env is not None else Env()
        for b in self._all():
            env = b.validate(env)
        return env

    def accept(self, produced: Any, env: Env, where: str) -> None:
        for k, b in self.branches.items():
            b.accept(produced, env, f"{where} [{k!r}]")
        if isinstance(self.default, Node):
            self.default.accept(produced, env, f"{where} [default]")

    def ratio_for(self, in_type: Any) -> Ratio:
        rs = [b.ratio_for(in_type) for b in self._all()]
        r = rs[0]
        for x in rs[1:]:
            if x != r:
                return Ratio.ANY
        return r

    def nodes(self) -> list[Node]:
        return [n for b in self._all() for n in b.nodes()]

    def describe(self, indent: str = "") -> str:
        lines = [
            f"{indent}{self.name}: {describe(self.in_type)} -> {describe(self.out_type)}"
        ]
        for k, b in self.branches.items():
            lines.append(b.describe(indent + f"  {k!r}: "))
        if isinstance(self.default, Node):
            lines.append(self.default.describe(indent + "  default: "))
        elif self.default == "pass":
            lines.append(f"{indent}  default: pass through")
        return "\n".join(lines)


def switch(
    selector: Any,
    branches: Mapping[Any, Any],
    default: Any = None,
    ordered: bool = False,
    name: str | None = None,
) -> Switch:
    """Route each frame to one of several branches by key.

    ``selector`` is a callable on the :class:`Frame` or the name of an annotation
    (``switch("band", {...})`` reads ``frame["band"]``). ``default`` handles keys with no
    branch: a node, ``"pass"`` (forward unchanged) or ``None`` (drop; ``run(strict=True)``
    raises instead). ::

        switch("modulation", {"bpsk": demod_bpsk, "qpsk": demod_qpsk}, default=log_unknown)
        switch(lambda f: f["snr_db"] > 10, {True: fast_path, False: robust_path})
    """
    nodes = {k: as_node(v) for k, v in branches.items()}
    dflt = default if default in (None, "pass") else as_node(default)
    return Switch(selector, nodes, dflt, ordered, name)


class Align(Node):
    """Timestamp join (built by :func:`align`): pairs frames from several sources whose
    start times agree within ``tolerance`` seconds, dropping unmatched older frames.
    Output: one tuple frame per match, ``t0`` = the latest member's. Unlike zip (lock-step
    by count) this tolerates different rates, jitter and gaps."""

    def __init__(
        self,
        parts: Iterable[Node],
        tolerance: float,
        key: Any = None,
        name: str | None = None,
    ) -> None:
        self.parts = list(parts)
        if any(p.in_type is not None for p in self.parts):
            raise TypeMismatch("align() joins sources (or source-rooted graphs) only")
        self.tolerance = tolerance
        self.key = key or (lambda f: f.t0)
        self.in_type = None
        self.out_type = TupleType(tuple(p.out_type for p in self.parts))
        self.port_names = _member_names(self.parts)
        self.name = name or "align(" + ", ".join(p.name for p in self.parts) + ")"

    def validate(self, env: Env | None = None) -> Env:
        env = env if env is not None else Env()
        for p in self.parts:
            env = p.validate(env)
        return env

    def nodes(self) -> list[Node]:
        return [n for p in self.parts for n in p.nodes()]

    def describe(self, indent: str = "") -> str:
        lines = [
            f"{indent}align(tolerance={self.tolerance}) -> {describe(self.out_type)}"
        ]
        for p in self.parts:
            lines.append(p.describe(indent + "  ≈ "))
        return "\n".join(lines)


def align(*parts: Any, tolerance: float = 0.0, key: Any = None) -> Align:
    """Join streams by time: ``align(imu, gps, tolerance=0.02)``; ``key(frame)`` defaults to ``t0``."""
    return Align([as_node(p) for p in parts], tolerance, key)


class Parallel(Node):
    """A map block executed on a pool (``parallel(block, workers=4)``): frames are submitted
    as they arrive and results are yielded *in order*, so the graph sees an ordinary 1:1
    block that uses several cores. ``backend="process"`` needs a picklable function (a
    module-level block) and sends the frame's data to the worker; ``"thread"`` works for
    anything and helps when the function releases the GIL (NumPy, SciPy)."""

    def ratio_for(self, in_type: Any) -> Ratio:
        return self.inner.ratio_for(in_type)

    def __init__(
        self,
        inner: Node,
        workers: int = 4,
        backend: str = "process",
        window: int | None = None,
    ) -> None:
        if (
            not isinstance(inner, Block)
            or inner.kind != "map"
            or len(inner.defn.inputs) != 1
            or inner.defn.framed
        ):
            raise TypeError("parallel() wraps a single-input, data-only map block")
        if backend not in ("process", "thread"):
            raise ValueError("backend must be 'process' or 'thread'")
        self.inner = inner
        self.workers = workers
        self.backend = backend
        self.window = window or 2 * workers
        self.in_type = inner.in_type
        self.out_type = inner.out_type
        self.port_names = inner.port_names
        self.name = f"parallel({inner.name}, {workers})"

    def validate(self, env: Env | None = None) -> Env:
        return self.inner.validate(env)

    def nodes(self) -> list[Node]:
        return [self]

    def describe(self, indent: str = "") -> str:
        return f"{indent}{self.name} [{self.backend}]: {describe(self.in_type)} -> {describe(self.out_type)}"


def parallel(
    block: Any, workers: int = 4, backend: str = "process", window: int | None = None
) -> Parallel:
    """Run a map block on several cores, keeping frame order."""
    return Parallel(as_node(block), workers, backend, window)


class Merge(Node):
    """Interleave several source streams in arrival order (output type is their join)."""

    def ratio_for(self, in_type: Any) -> Ratio:
        return Ratio.ONE

    def __init__(self, parts: Iterable[Node]) -> None:
        seen: set[int] = set()
        self.parts = [fresh(p, seen) for p in parts]
        if any(p.in_type is not None for p in self.parts):
            raise TypeMismatch("merge() joins sources only")
        self.in_type = None
        self.name = "merge(" + ", ".join(p.name for p in self.parts) + ")"
        types_ = [p.out_type for p in self.parts]
        self.out_type = types_[0]
        for t in types_[1:]:
            if t != self.out_type:
                self.out_type = SignalType()  # heterogeneous: Any
                break

    def validate(self, env: Env | None = None) -> Env:
        env = env if env is not None else Env()
        for p in self.parts:
            env = p.validate(env)
        return env

    def nodes(self) -> list[Node]:
        return [n for p in self.parts for n in p.nodes()]


def merge(*parts: Any) -> Merge:
    return Merge(as_node(p) for p in parts)


class Select(Node):
    """Pick one or several ports of a tuple-typed upstream node (built by ``node[key]``)."""

    def ratio_for(self, in_type: Any) -> Ratio:
        return Ratio.ONE

    def __init__(self, upstream: Node, key: Any) -> None:
        items = upstream.out_type.items
        names = tuple(getattr(upstream, "port_names", ()) or (None,) * len(items))
        self.indices, self.single = _resolve_ports(
            key, len(items), names, upstream.name
        )
        self.key = key
        self.in_type = upstream.out_type
        if self.single:
            self.out_type = items[self.indices[0]]
            self.port_names = ()
            label = str(names[self.indices[0]] or self.indices[0])
        else:
            self.out_type = TupleType(tuple(items[i] for i in self.indices))
            self.port_names = tuple(names[i] for i in self.indices)
            label = ", ".join(str(names[i] or i) for i in self.indices)
        self.name = f"[{label}]"

    def describe(self, indent: str = "") -> str:
        return f"{indent}select {self.name}: {describe(self.in_type)} -> {describe(self.out_type)}"


class Slice(Node):
    """Data slicing of an array-typed upstream (built by ``node[key]``)."""

    def ratio_for(self, in_type: Any) -> Ratio:
        return Ratio.ONE

    def __init__(self, upstream: Node, key: Any) -> None:
        self.key = key
        self.in_type = upstream.out_type
        t = upstream.out_type
        env = getattr(upstream, "env", None)
        if env is not None and isinstance(t, SignalType):
            t = t.resolve(env)  # use dimensions the graph has already bound
        self.out_type = (
            t.with_(shape=slice_shape(t.shape, key)) if isinstance(t, SignalType) else t
        )
        self.name = "[" + _key_repr(key) + "]"

    def describe(self, indent: str = "") -> str:
        return f"{indent}slice {self.name}: {describe(self.in_type)} -> {describe(self.out_type)}"


def _key_repr(key: Any) -> str:
    def one(k: Any) -> str:
        if isinstance(k, slice):
            a = "" if k.start is None else str(k.start)
            b = "" if k.stop is None else str(k.stop)
            c = "" if k.step is None else ":" + str(k.step)
            return f"{a}:{b}{c}"
        return "..." if k is Ellipsis else repr(k)

    return ", ".join(one(k) for k in key) if isinstance(key, tuple) else one(key)


class StreamSlice(Node):
    """Frame-count slicing of a stream: take / skip / every / until (identity-typed)."""

    def ratio_for(self, in_type: Any) -> Ratio:
        if self.kind == "every":
            return Ratio.of(int(self.params.get("k", 1)), 1)
        return Ratio.DOWN  # take / skip / until / while: fewer or equal

    def __init__(self, kind: str, upstream: Node | None = None, **params: Any) -> None:
        self.kind = kind
        self.params = params
        self.in_type = SignalType() if upstream is None else upstream.out_type
        self.out_type = self.in_type
        self.port_names = (
            () if upstream is None else tuple(getattr(upstream, "port_names", ()))
        )
        shown = ", ".join(f"{k}={v!r}" for k, v in params.items() if not callable(v))
        self.name = f"{kind}({shown})"

    def resolve_ports(self, upstream: Node) -> Node:
        if self.in_type.is_any if isinstance(self.in_type, SignalType) else False:
            return StreamSlice(self.kind, upstream, **self.params)
        return self

    def describe(self, indent: str = "") -> str:
        return f"{indent}{self.name}: {describe(self.in_type)} -> {describe(self.out_type)}"


def _take(n: int) -> Node:
    return StreamSlice("take", n=n)


def _skip(n: int) -> Node:
    return StreamSlice("skip", n=n)


def _every(k: int, offset: int) -> Node:
    return StreamSlice("every", k=k, offset=offset)


def _until(pred: Any, inclusive: bool) -> Node:
    return StreamSlice("until", pred=pred, inclusive=inclusive)


class PortRef(Node):
    """An unresolved port selection, made by ``port(key)``. It accepts any tuple frame
    until it is placed after a producer, when it becomes a typed :class:`Select`::

        decomposition >> (port("vecs") | port("vals") >> model_order) >> combine
    """

    def ratio_for(self, in_type: Any) -> Ratio:
        return Ratio.ONE

    def __init__(self, key: Any) -> None:
        self.key = key
        self.in_type = (
            SignalType()
        )  # placeholder (any); checked in accept(), resolved on placement
        self.out_type = SignalType()
        self.name = f"port[{key!r}]"

    def accept(self, produced: Any, env: Env, where: str) -> None:
        if not isinstance(produced, TupleType):
            raise TypeMismatch(
                f"{where}: {self.name} needs a tuple output, got {describe(produced)}"
            )

    def resolve_ports(self, upstream: Node) -> Node:
        if not isinstance(upstream.out_type, TupleType):
            raise TypeMismatch(
                f"{upstream.name} -> {self.name}: port selection needs a tuple output, "
                f"got {describe(upstream.out_type)}"
            )
        return Select(upstream, self.key)


def port(*key: Any) -> PortRef:
    """A port selector to place after any tuple-typed producer: ``port(0)``, ``port("vals")``,
    ``port(0, 2)`` (several ports -> tuple), ``port(slice(1, None))``."""
    return PortRef(key[0] if len(key) == 1 else key)


def _resolve_ports(
    key: Any, n: int, names: tuple, where: str
) -> tuple[list[int], bool]:
    def one(k: Any) -> int:
        if isinstance(k, str):
            if k not in names:
                raise KeyError(
                    f"{where} has no port named {k!r}; ports: {[nm or i for i, nm in enumerate(names)]}"
                )
            return names.index(k)
        if not isinstance(k, int):
            raise TypeError(
                f"port keys are ints, names or slices, not {type(k).__name__}"
            )
        if not -n <= k < n:
            raise IndexError(f"{where} has {n} ports; port {k} does not exist")
        return k % n

    if isinstance(key, slice):
        return list(range(n))[key], False
    if isinstance(key, tuple):
        return [one(k) for k in key], False
    return [one(key)], True


def _member_names(nodes: list[Node]) -> tuple[str | None, ...]:
    """Port names for fan-out / zip members: the member's label or name (a pipeline
    branch is named after its last block); duplicates get no name."""

    def name_of(m: Node) -> str:
        if isinstance(m, Pipeline):
            m = m.stages[-1]
        return getattr(m, "label", None) or m.name

    names = [name_of(m) for m in nodes]
    return tuple(nm if names.count(nm) == 1 else None for nm in names)


def _unscope(t: Any) -> Any:
    """Strip symbol scopes so a type can be re-scoped for a cloned node."""
    from .types import DVar, StreamType

    if t is None:
        return None
    if isinstance(t, TupleType):
        return TupleType(tuple(_unscope(i) for i in t.items))
    if isinstance(t, StreamType):
        return StreamType(_unscope(t.item))
    if not isinstance(t, SignalType):
        return t
    dtype = (
        DVar(t.dtype.name.split("#", 1)[0]) if isinstance(t.dtype, DVar) else t.dtype
    )
    shape = (
        None
        if t.shape is None
        else tuple(d.split("#", 1)[0] if isinstance(d, str) else d for d in t.shape)
    )
    return t.with_(dtype=dtype, shape=shape)


def fresh(node: Node, seen: set[int]) -> Node:
    """Return ``node`` with any leaf already in ``seen`` replaced by a clone, so a
    sub-graph can be reused several times in one graph without its symbols colliding."""
    if isinstance(node, (Block, Source)):
        if id(node) in seen:
            node = node.clone()
        seen.add(id(node))
        return node
    if isinstance(node, Pipeline):
        return Pipeline.__new__(Pipeline)._init([fresh(s, seen) for s in node.stages])
    if isinstance(node, Fanout):
        return Fanout([fresh(b, seen) for b in node.branches])
    if isinstance(node, Zip):
        return Zip([fresh(p, seen) for p in node.parts])
    if isinstance(node, Merge):
        return Merge([fresh(p, seen) for p in node.parts])
    if isinstance(node, Align):
        return Align(
            [fresh(p, seen) for p in node.parts], node.tolerance, node.key, node.name
        )
    if isinstance(node, Parallel):
        return Parallel(
            fresh(node.inner, seen), node.workers, node.backend, node.window
        )  # type: ignore[arg-type]
    if isinstance(node, Feedback):
        return Feedback(fresh(node.body, seen), node.init, node.name)
    if isinstance(node, When):
        return When(
            node.pred,
            fresh(node.then, seen),
            fresh(node.otherwise, seen) if node.otherwise is not None else None,
            node.name,
        )
    return node


def _common_ratio(branches: list[Node], in_type: Any, where: str) -> Ratio:
    """Branches paired by a fan-out / zip must emit frames at compatible rates."""
    rs = [b.ratio_for(in_type) for b in branches]
    best = Ratio.ANY
    for b, r in zip(branches, rs):
        if not best.compatible(r):
            bad = next(x for x, y in zip(branches, rs) if y == best)
            raise RatioMismatch(
                f"{where}: {bad.name} emits {best} frames per input but {b.name} emits {r}; "
                f"pair branches with matching rates (or re-frame one side)"
            )
        best = best.meet(r)
    return best


def _common_input(ins: list[Any], where: str) -> Any:
    """The most demanding of several input types (they must be mutually compatible)."""
    env = Env()
    best = ins[0]
    for t in ins[1:]:
        try:
            unify(best, t, env, where)
        except TypeMismatch:
            try:
                unify(t, best, env, where)
                best = t
            except TypeMismatch:
                raise TypeMismatch(
                    f"{where}: branches require incompatible inputs {best} and {t}"
                ) from None
    return best


def as_node(obj: Any) -> Node:
    """Coerce functions, block definitions and iterables into graph nodes."""
    if isinstance(obj, Node):
        return obj
    if isinstance(obj, BlockDef):
        return obj.node()
    if (
        isinstance(obj, (list, tuple))
        and obj
        and all(isinstance(o, (Node, BlockDef)) for o in obj)
    ):
        return Fanout([as_node(o) for o in obj])
    if callable(obj):
        from .block import lift

        return lift(obj).node()
    if hasattr(obj, "__iter__") or hasattr(obj, "__aiter__"):
        return Source(obj)
    raise TypeError(f"cannot use {obj!r} in a graph")


def from_iterable(obj: Any, type: Any = None, name: str | None = None) -> Source:
    """Make a typed source from any iterable, async iterable, or zero-arg factory."""
    return Source(obj, type, name)


def validate(node: Node) -> Env:
    """Validate a whole graph and return the environment of bound dimensions."""
    return node.validate()

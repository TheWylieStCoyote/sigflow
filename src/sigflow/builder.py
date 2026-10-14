"""An explicit graph builder for topologies the operators cannot express.

    g = Graph("receiver")
    src   = g.add(dsp.sine(n=1024, frames=50))
    front = g.add(dsp.window >> dsp.to_complex >> dsp.fft, "front")
    eig   = g.add(eigh)                                   # a multi-output block
    order = g.add(model_order)
    noise = g.add(noise_subspace)                         # takes (vecs, order)
    g.connect(src, front)
    g.connect(front, eig)
    g.connect(eig["vals"], order)                         # port selection on an edge
    g.connect(eig["vecs"], noise, "vecs")                 # by parameter name ...
    g.connect(order, noise, 1)                            # ... or by position
    g.output(noise)                                       # one or several outputs (-> tuple)
    run(g); g.draw(); g.report()

Rules: an edge carries frames from a producer to one input of a consumer; a
consumer with several inputs receives them zipped in lock step; a producer may
feed any number of consumers (fan-out); edges are type-checked as they are made,
with symbolic dimensions shared across the whole graph. A graph without sources
declares an input with ``g.input(type)`` and is itself a reusable node. Cycles
are not allowed — use ``feedback()`` for loops.
"""

from __future__ import annotations

from typing import Any

from .frame import Frame
from .graph import Node, Select, TupleType, as_node, describe
from .ratio import Ratio, RatioMismatch
from .types import Env, SignalType, TypeMismatch, unify

__all__ = ["Graph", "Ref", "Join", "Chain", "lower"]


class Ref:
    """A node placed in a :class:`Graph` (optionally with a port selected)."""

    def __init__(self, graph: "Graph", node: Node, name: str, key: Any = None) -> None:
        self.graph, self.node, self.name, self.key = graph, node, name, key

    def __getitem__(self, key: Any) -> "Ref":
        if not isinstance(self.node.out_type, TupleType):
            raise TypeError(
                f"{self.name} has a single output; [] selects ports of tuple outputs"
            )
        return Ref(self.graph, self.node, self.name, key)

    @property
    def out_type(self) -> Any:
        return (
            Select(self.node, self.key).out_type
            if self.key is not None
            else self.node.out_type
        )

    def __rshift__(self, other: Any) -> "Ref":
        """``a >> b`` inside a graph: add ``b`` if needed and connect."""
        dst = other if isinstance(other, Ref) else self.graph.add(other)
        self.graph.connect(self, dst)
        return dst

    def __repr__(self) -> str:
        return f"Ref({self.name}{'[' + repr(self.key) + ']' if self.key is not None else ''})"


class Join(Node):
    """A tuple join member: receives N inputs zipped in lock step and passes the tuple
    frame on. This is what a fan-out (`a | b`) or zip (`a & b`) lowers to."""

    def __init__(self, out_type: Any, op: str = "zip", port_names: tuple = ()) -> None:
        self.in_type = out_type
        self.out_type = out_type
        self.join_op = op
        self.name = op
        self.port_names = port_names

    def ratio_for(self, in_type: Any) -> Ratio:
        return Ratio.ONE


class Chain(Node):
    """A linear run of members executed back to back without queues between them
    (what a pipeline segment between joins lowers to). Consecutive plain map blocks
    inside a chain are fused by the runtime into one frame wrap."""

    def __init__(self, stages: list[Node]) -> None:
        self.stages = stages
        self.in_type = stages[0].in_type
        self.out_type = stages[-1].out_type
        self.port_names = tuple(getattr(stages[-1], "port_names", ()))
        if hasattr(stages[0], "join_op"):
            self.join_op = stages[0].join_op
        self.name = " >> ".join(s.name for s in stages)

    def validate(self, env: Any = None) -> Any:
        from .types import Env

        env = env if env is not None else Env()
        for s in self.stages:
            env = s.validate(env)
        return env

    def nodes(self) -> list[Node]:
        return [n for s in self.stages for n in s.nodes()]

    def ratio_for(self, in_type: Any) -> Ratio:
        r, t = Ratio.ONE, in_type
        for s in self.stages:
            r = r * s.ratio_for(t if s.in_type is not None else None)
            t = s.out_type
        return r

    def describe(self, indent: str = "") -> str:
        return "\n".join(s.describe(indent) for s in self.stages)


class _Input(Node):
    """Placeholder for the graph's own input (a source-less graph)."""

    def __init__(self, type: Any) -> None:
        self.in_type = type
        self.out_type = type
        self.name = "input"


class Graph(Node):
    transparent = False  # lowered operator trees add no provenance / stats of their own

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.members: dict[str, Node] = {}
        self.edges: list[
            tuple[Ref, str, Any]
        ] = []  # (source ref, consumer name, input slot)
        self.outputs: list[Ref] = []
        self._input: _Input | None = None
        self.in_type = None
        self.out_type = SignalType()
        self._env = Env()
        self._count = 0

    # -- construction ------------------------------------------------------ #
    def input(self, type: Any = None, name: str = "input") -> Ref:
        """Declare the graph's input (makes it a source-less, reusable node)."""
        if self._input is not None:
            raise ValueError("a graph has one input")
        self._input = _Input(SignalType() if type is None else type)
        self.in_type = self._input.out_type
        self.members[name] = self._input
        return Ref(self, self._input, name)

    def add(self, node: Any, name: str | None = None) -> Ref:
        node = as_node(node)
        base = name or node.name.split(" ")[0].split("(")[0]
        name = base
        while name in self.members:
            self._count += 1
            name = f"{base}#{self._count}"
        self.members[name] = node
        return Ref(self, node, name)

    def connect(self, src: Any, dst: Any, slot: Any = None) -> "Graph":
        """Connect ``src`` (a Ref, optionally with a port) to ``dst`` (a Ref or a node to add).
        ``slot`` names the consumer's input parameter (or index) for multi-input blocks."""
        src = src if isinstance(src, Ref) else self.add(src)
        dst = dst if isinstance(dst, Ref) else self.add(dst)
        if dst.node is self._input:
            raise TypeMismatch("the graph input cannot be a consumer")
        if dst.node.in_type is None:
            raise TypeMismatch(f"{dst.name} is a source and cannot consume")
        n_in = self._n_inputs(dst.node)
        if slot is None:
            used = [e[2] for e in self.edges if e[1] == dst.name]
            if (
                n_in > 1
                and not used
                and isinstance(src.out_type, TupleType)
                and len(src.out_type) == n_in
            ):
                slot = -1  # one tuple-typed edge feeds all inputs at once
            else:
                slot = (
                    next(i for i in range(max(n_in, 1)) if i not in used)
                    if len(used) < max(n_in, 1)
                    else len(used)
                )
        if isinstance(slot, str):
            names = self._input_names(dst.node)
            if slot not in names:
                raise KeyError(
                    f"{dst.name} has no input named {slot!r}; inputs: {names}"
                )
            slot = names.index(slot)
        if any(e[1] == dst.name and e[2] == slot for e in self.edges):
            raise TypeMismatch(f"input {slot} of {dst.name} is already connected")
        if slot != -1 and not 0 <= slot < max(n_in, 1):
            raise IndexError(
                f"{dst.name} has {max(n_in, 1)} input(s); slot {slot} does not exist"
            )
        self.edges.append((src, dst.name, slot))
        self._check_edge(src, dst, slot)
        self._reject_cycles()
        try:
            self.ratios()
        except RatioMismatch:
            self.edges.pop()
            raise
        return self

    def output(self, *refs: Any) -> "Graph":
        """Declare the graph's output(s); several outputs become one tuple frame."""
        self.outputs = [r if isinstance(r, Ref) else self.add(r) for r in refs]
        types = [r.out_type for r in self.outputs]
        self.out_type = types[0] if len(types) == 1 else TupleType(tuple(types))
        if len(types) > 1:
            self.port_names = tuple(r.name for r in self.outputs)
        return self

    # -- introspection ----------------------------------------------------- #
    @staticmethod
    def _n_inputs(node: Node) -> int:
        if isinstance(node.in_type, TupleType):
            return len(node.in_type)
        return 1

    @staticmethod
    def _input_names(node: Node) -> list[str]:
        defn = getattr(node, "defn", None)
        if defn is not None:
            return [i.name for i in defn.inputs]
        return [str(i) for i in range(Graph._n_inputs(node))]

    def _produced(self, name: str) -> Any:
        """The type arriving at ``name`` from its edges (None if unconnected)."""
        node = self.members[name]
        ins = sorted([e for e in self.edges if e[1] == name], key=lambda e: e[2])
        if not ins:
            return None
        n = self._n_inputs(node)
        if (n == 1 and len(ins) == 1) or ins[0][2] == -1:
            return ins[0][0].out_type
        slots = {e[2]: e[0].out_type for e in ins}
        return TupleType(tuple(slots.get(i, SignalType()) for i in range(n)))

    def _check_edge(self, src: Ref, dst: Ref, slot: int) -> None:
        t = src.out_type
        need = dst.node.in_type
        if slot != -1 and isinstance(need, TupleType) and self._n_inputs(dst.node) > 1:
            need = need.items[slot]
        try:
            unify(t, need, self._env, where=f"{src.name} -> {dst.name}[{slot}]")
        except TypeMismatch as e:
            self.edges.pop()
            raise TypeMismatch(str(e)) from None

    def _reject_cycles(self) -> None:
        seen: set[str] = set()
        stack: set[str] = set()

        def visit(n: str) -> None:
            if n in stack:
                raise TypeMismatch(f"cycle through {n}; use feedback() for loops")
            if n in seen:
                return
            stack.add(n)
            for src, dst, _ in self.edges:
                if src.name == n:
                    visit(dst)
            stack.discard(n)
            seen.add(n)

        for name in self.members:
            visit(name)

    def order(self) -> list[str]:
        """Members in topological order."""
        incoming = {n: [e for e in self.edges if e[1] == n] for n in self.members}
        done: list[str] = []
        while len(done) < len(self.members):
            progress = False
            for n in self.members:
                if n in done:
                    continue
                if all(e[0].name in done for e in incoming[n]):
                    done.append(n)
                    progress = True
            if not progress:
                raise TypeMismatch("graph has a cycle")
        return done

    def consumers(self, name: str) -> list[tuple[Ref, str, Any]]:
        return [e for e in self.edges if e[0].name == name]

    # -- Node protocol ----------------------------------------------------- #
    def validate(self, env: Env | None = None) -> Env:
        env = env if env is not None else Env()
        for name, node in self.members.items():
            env = node.validate(env)
            if node is self._input:
                continue
            produced = self._produced(name)
            if node.in_type is not None and produced is None:
                raise TypeMismatch(f"{name} has unconnected input(s)")
            if produced is not None:
                unify(produced, node.in_type, env, where=f"-> {name}")
        if not self.outputs:
            raise TypeMismatch("graph has no output(); call g.output(ref)")
        self.ratio_for(self.in_type)
        return env

    def nodes(self) -> list[Node]:
        return [
            n for m in self.members.values() if m is not self._input for n in m.nodes()
        ]

    def describe(self, indent: str = "") -> str:
        lines = [
            f"{indent}graph {self.name}: {('' if self.in_type is None else describe(self.in_type) + ' -> ')}{describe(self.out_type)}"
        ]
        for name in self.order():
            node = self.members[name]
            ins = ", ".join(
                f"{e[0].name}{'[' + repr(e[0].key) + ']' if e[0].key is not None else ''}"
                for e in sorted(self.consumers_of(name), key=lambda e: e[2])
            )
            tag = " (output)" if any(r.name == name for r in self.outputs) else ""
            lines.append(
                f"{indent}  {name}{tag}: {describe(node.out_type)}"
                + (f"   <- {ins}" if ins else "")
            )
        return "\n".join(lines)

    def consumers_of(self, name: str) -> list[tuple[Ref, str, Any]]:
        return [e for e in self.edges if e[1] == name]

    # -- stream-rate contract --------------------------------------------- #
    def ratios(self) -> dict[str, Ratio]:
        """Each member's output frames per graph input / source frame. Edges meeting at a
        multi-input member (or the graph's outputs) must carry compatible rates."""
        out: dict[str, Ratio] = {}
        for name in self.order():
            node = self.members[name]
            ins = sorted([e for e in self.edges if e[1] == name], key=lambda e: e[2])
            if not ins:
                out[name] = Ratio.ONE  # a source, or the graph input
                continue
            arriving = Ratio.ANY
            for src, _, _ in ins:
                r = out.get(src.name, Ratio.ANY)
                if not arriving.compatible(r):
                    raise RatioMismatch(
                        f"-> {name}: inputs arrive at incompatible rates "
                        f"({arriving} and {r} frames per source frame); re-frame one side"
                    )
                arriving = arriving.meet(r)
            out[name] = arriving * node.ratio_for(self._produced(name))
        return out

    def ratio_for(self, in_type: Any) -> Ratio:
        if not self.outputs:
            return Ratio.ANY
        rs = self.ratios()
        r = Ratio.ANY
        for ref in self.outputs:
            x = rs.get(ref.name, Ratio.ANY)
            if not r.compatible(x):
                raise RatioMismatch(
                    f"graph outputs run at incompatible rates ({r} and {x})"
                )
            r = r.meet(x)
        return r

    def __repr__(self) -> str:
        return self.describe()


# --------------------------------------------------------------------------- #
# Lowering the operator tree
# --------------------------------------------------------------------------- #


def lower(node: Node) -> "Graph":
    """Compile an operator-tree node (pipelines, fan-outs, zips, ...) into a Graph of
    simple members and edges, so there is one scheduler for every graph."""
    from .graph import Fanout, Pipeline, Zip

    if isinstance(node, Graph):
        return node
    g = Graph(node.name)
    g.transparent = True
    src = g.input(node.in_type) if node.in_type is not None else None
    out = _lower_into(g, node, src)
    g.output(out)
    return g


def _lower_into(g: "Graph", node: Node, src: Ref | None) -> Ref:
    from .graph import Fanout, Pipeline, Zip

    if isinstance(node, Pipeline):
        cur = src
        run: list[Node] = []
        feeders: list[Ref] | None = None  # refs feeding the join at the head of `run`

        def flush() -> None:
            nonlocal cur, run, feeders
            if run:
                member = run[0] if len(run) == 1 else Chain(run)
                ref = g.add(member, _member_name(member))
                if feeders is not None:
                    for i, r in enumerate(feeders):
                        g.connect(r, ref, i)
                elif cur is not None and member.in_type is not None:
                    g.connect(cur, ref)
                cur = ref
                run, feeders = [], None

        for stage in node.stages:
            if isinstance(stage, (Fanout, Zip)):
                flush()
                feeders, join = _lower_join(g, stage, cur)
                run = [join]  # the join heads the next chain (no queue after it)
            elif isinstance(stage, Graph):
                flush()
                cur = _lower_into(g, stage, cur)
            else:
                run.append(stage)
        flush()
        assert cur is not None
        return cur
    if isinstance(node, (Fanout, Zip)):
        feeders, join = _lower_join(g, node, src)
        ref = g.add(join, join.name)
        for i, r in enumerate(feeders):
            g.connect(r, ref, i)
        return ref
    ref = g.add(node, _member_name(node))
    if src is not None and node.in_type is not None:
        g.connect(src, ref)
    return ref


def _lower_join(g: "Graph", node: Node, src: Ref | None) -> tuple[list[Ref], "Join"]:
    """Lower the branches of a fan-out / zip; return their refs and the (unplaced) join."""
    from .graph import Fanout, Zip

    if isinstance(node, Fanout) or (isinstance(node, Zip) and node.in_type is not None):
        members = node.branches if isinstance(node, Fanout) else node.parts
        refs = [_lower_into(g, b, src) for b in members]
        return refs, Join(node.out_type, "fanout", node.port_names)
    refs = [_lower_into(g, p, None) for p in node.parts]  # type: ignore[attr-defined]
    return refs, Join(node.out_type, "zip", node.port_names)


def _member_name(node: Node) -> str:
    return node.name

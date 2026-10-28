"""Execution: an asyncio-native runtime with transparent adapters for sync code.

Every node is compiled to an async transformation of frame streams.  Plain
generators, iterables and functions are adapted automatically, so a graph may
freely mix synchronous and asynchronous sources and blocks.

* ``run(graph)``       – run to completion synchronously, return output frames
* ``await arun(graph)``– same, from async code (e.g. a Jupyter cell)
* ``async for f in graph`` / ``for f in graph`` – stream the outputs lazily

Semantics applied uniformly by the runtime (so block authors never write them):

* annotations (stream / frame / sample level) are carried forward,
* provenance is extended with a node for every computation,
* a block that raises produces a *failed* frame (``MISSING`` data plus an
  error :class:`~sigflow.frame.Issue`) which propagates downstream without
  being recomputed, unless the block's ``on_error`` says otherwise,
* declared types are checked against the first frame crossing each edge.
"""

from __future__ import annotations

import asyncio
import inspect
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, NoReturn, cast

import numpy as np

from .block import BlockDef
from .control import live
from .frame import MISSING, Frame, Issue, Provenance, Tags, merge_annotations
from .graph import (
    Align,
    Block,
    Fanout,
    Feedback,
    Merge,
    Node,
    Parallel,
    Pipeline,
    Select,
    Slice,
    Source,
    StreamSlice,
    Switch,
    When,
    Zip,
)
from .progress import Monitor, Reporter
from .types import Env, TypeMismatch, tag_names

__all__ = ["RunOptions", "Runner", "arun", "iterate", "iterate_sync", "run"]

_DONE = object()


def _reraise(failure: asyncio.Future[Any]) -> NoReturn:
    """Raise the error a member recorded in the shared failure future."""
    exc = failure.exception()
    assert exc is not None
    raise exc


def _unwrap(item: Any) -> Any:
    """A queue item: re-raise a forwarded ``("error", exc)``; anything else is data."""
    if isinstance(item, tuple) and len(item) == 2 and item[0] == "error":
        raise item[1]
    return item


class RunOptions:
    def __init__(
        self,
        check: str = "first",
        on_error: str | None = None,
        monitor: Monitor | None = None,
        queue_size: int = 64,
        control: Any = None,
        resume: Any = None,
        strict: bool = False,
        provenance: str = "full",
        labels: bool | None = None,
    ) -> None:
        if provenance not in ("full", "roots", "off"):
            raise ValueError("provenance must be 'full', 'roots' or 'off'")
        self.provenance = provenance
        if check not in ("first", "all", "none"):
            raise ValueError("check must be 'first', 'all' or 'none'")
        self.check = check
        self.control = control  # sigflow.control.Control for pause/step/stop
        self.resume = (
            resume  # sigflow.persist.Checkpoint (or path): skip emitted source frames
        )
        self.strict = (
            strict  # raise when fan-out / zip branches produce different frame counts
        )
        self.on_error = on_error  # overrides every block's policy when set
        self.monitor = monitor or Monitor()
        self.queue_size = queue_size
        self.env = Env()
        # blocks with an unset ``labels`` option follow the run; the package default
        # is read once here, so the choice travels with the run (threads, workers)
        if labels is None:
            from .xarray import current_default

            labels = current_default()
        self.labels = bool(labels)


# every keyword ``RunOptions`` takes, so entry points that filter their options
# (``iterate``) cannot drift from its signature
_RUN_OPTIONS = tuple(inspect.signature(RunOptions.__init__).parameters)[1:]


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


def _is_factory(obj: object) -> bool:
    """A callable that is not itself iterable: call it to get the iterable."""
    return (
        callable(obj) and not hasattr(obj, "__iter__") and not hasattr(obj, "__aiter__")
    )


async def aiter_any(obj: Any, threaded: bool = False) -> AsyncIterator[Any]:
    """Iterate anything — sync or async iterable, generator or factory — asynchronously."""
    if _is_factory(obj):
        obj = cast(Any, obj())
        if inspect.isawaitable(obj):
            obj = cast(Any, await obj)
    if hasattr(obj, "__aiter__"):
        async for item in obj:
            yield item
        return
    it = iter(obj)
    if threaded:
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, next, it, _DONE)
            if item is _DONE:
                return
            yield item
    else:
        for item in it:
            yield item
            await asyncio.sleep(0)  # stay cooperative with async sources


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class Runner:
    def __init__(self, opts: RunOptions) -> None:
        self.opts = opts
        self.mon = opts.monitor

    # -- dispatch ---------------------------------------------------------- #
    def compile(
        self, node: Node, upstream: AsyncIterator[Frame] | None
    ) -> AsyncIterator[Frame]:
        from .builder import Chain, Graph, Join, lower

        if isinstance(node, (Pipeline, Fanout, Zip)):
            g = lower(node)
            real = [m for m in g.members.values() if type(m).__name__ != "_Input"]
            if len(real) == 1 and len(g.outputs) == 1 and g.outputs[0].key is None:
                return self.compile(
                    real[0], upstream
                )  # a plain chain: no queues at all
            return self._graph(g, upstream)  # one scheduler for every composition
        if isinstance(node, Graph):
            return self._graph(node, upstream)
        if isinstance(node, Join):
            assert upstream is not None
            return self._join(node, upstream)
        if isinstance(node, Chain):
            return self._chain(node, upstream)
        if isinstance(node, Source):
            return self._source(node)
        if isinstance(node, Block):
            if node.kind == "source":
                return self._block_source(node)
            assert upstream is not None, f"{node.name} needs an upstream"
            if node.kind == "stream":
                return self._stream_block(node, upstream)
            return self._map_block(node, upstream)
        if isinstance(node, Merge):
            return self._merge(node)
        if isinstance(node, Parallel):
            assert upstream is not None
            return self._parallel(node, upstream)
        if isinstance(node, Align):
            return self._align(node)
        if isinstance(node, Select):
            assert upstream is not None
            return self._select(node, upstream)
        if isinstance(node, Slice):
            assert upstream is not None
            return self._slice(node, upstream)
        if isinstance(node, StreamSlice):
            assert upstream is not None
            return self._stream_slice(node, upstream)
        if isinstance(node, Feedback):
            assert upstream is not None
            return self._feedback(node, upstream)
        if isinstance(node, When):
            assert upstream is not None
            return self._when(node, upstream)
        if isinstance(node, Switch):
            assert upstream is not None
            return (
                self._switch_ordered(node, upstream)
                if node.ordered
                else self._switch(node, upstream)
            )
        raise TypeError(f"cannot run {node!r}")

    async def _stream_slice(
        self, node: StreamSlice, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        stats = self.mon.stats(node.name, node)
        p = node.params
        i = 0
        seq = 0
        async for f in upstream:
            stats.inp += 1
            keep, stop = True, False
            if node.kind == "take":
                keep, stop = i < p["n"], i + 1 >= p["n"]
            elif node.kind == "skip":
                keep = i >= p["n"]
            elif node.kind == "every":
                keep = i >= p["offset"] and (i - p["offset"]) % p["k"] == 0
            elif node.kind == "until":
                hit = bool(p["pred"](f))
                keep, stop = (not hit) or p["inclusive"], hit
            i += 1
            if keep:
                stats.out += 1
                yield f.evolve(
                    prov=Provenance(node.name, {}, (f.prov,) if f.prov else (), seq)
                )
                seq += 1
            if stop:
                return

    async def _slice(
        self, node: Slice, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        stats = self.mon.stats(node.name, node)
        seq = 0
        async for f in upstream:
            stats.inp += 1
            prov = Provenance(node.name, {}, (f.prov,) if f.prov else (), seq)
            if f.failed:
                yield f.evolve(prov=prov)
            else:
                try:
                    out = f.slice(node.key).evolve(prov=prov)
                except Exception as e:
                    stats.errors += 1
                    yield f.evolve(
                        data=MISSING,
                        issues=f.issues
                        + (
                            Issue(
                                f"{type(e).__name__}: {e}", "error", node.name, e, prov
                            ),
                        ),
                        prov=prov,
                    )
                    seq += 1
                    continue
                if self.opts.check == "all" or (
                    self.opts.check == "first" and stats.out == 0
                ):
                    node.out_type.check(
                        out.data, self.opts.env, where=f"{node.name} output"
                    )
                yield out
            stats.out += 1
            seq += 1

    def _chain(
        self, node: Any, upstream: AsyncIterator[Frame] | None
    ) -> AsyncIterator[Frame]:
        """Compose the stages directly (no queues); fuse runs of plain map blocks."""
        stream = upstream
        run: list[Block] = []

        def fusable(st: Node) -> bool:
            return (
                isinstance(st, Block)
                and st.kind == "map"
                and len(st.defn.inputs) == 1
                and not st.defn.tap
                and not st.defn.is_async
            )

        def flush() -> None:
            nonlocal stream, run
            if run:
                assert stream is not None
                stream = (
                    self._map_block(run[0], stream)
                    if len(run) == 1
                    else self._fused_maps(run, stream)
                )
                run = []

        for st in node.stages:
            if fusable(st):
                run.append(st)
            else:
                flush()
                stream = self.compile(st, stream)
        flush()
        assert stream is not None
        return stream

    async def _fused_maps(
        self, nodes: list[Block], upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        """Several map blocks applied back to back with one frame wrap at the end.
        Provenance still gets one node per block; annotations are merged once."""
        stats_ = [self.mon.stats(n.name, n) for n in nodes]
        policies = [self._policy(n) for n in nodes]
        calls = [n.defn.resolve_call(self.opts.labels) for n in nodes]
        needs = [bool(tag_names(n.in_type)) for n in nodes]  # input tag requirements
        seqs = [0] * len(nodes)
        check_first = self.opts.check == "first"
        check_all = self.opts.check == "all"
        async for f in upstream:
            prov = f.prov
            value: Any = f.data
            frame = f  # the current frame as a *framed* block would see it
            failed = f.failed
            for i, node in enumerate(nodes):
                st = stats_[i]
                st.inp += 1
                d = node.defn
                if failed and not d.framed:
                    prov = Provenance(
                        node.name, node.config, (prov,) if prov else (), seqs[i]
                    )
                    seqs[i] += 1
                    st.out += 1
                    continue
                if needs[i] and (check_all or (check_first and st.out == 0)):
                    _check_required(node.in_type, frame, f"{node.name} input")
                t0 = time.perf_counter()
                try:
                    if d.framed and (frame.data is not value or frame.prov is not prov):
                        frame = frame.evolve(
                            data=value, prov=prov
                        )  # materialize what a framed block sees
                    arg = frame if d.framed else value
                    cfg = live(node.given) if node.live else node.given
                    result = calls[i](arg, **cfg)
                except Exception as e:
                    st.errors += 1
                    st.time += time.perf_counter() - t0
                    if policies[i] == "raise":
                        raise
                    if policies[i] == "drop":
                        value = None
                        break
                    prov = Provenance(
                        node.name, node.config, (prov,) if prov else (), seqs[i]
                    )
                    frame = frame.evolve(
                        data=MISSING,
                        issues=frame.issues
                        + (
                            Issue(
                                f"{type(e).__name__}: {e}", "error", node.name, e, prov
                            ),
                        ),
                        prov=prov,
                    )
                    value, failed = MISSING, True
                    seqs[i] += 1
                    st.out += 1
                    continue
                st.time += time.perf_counter() - t0
                if result is None:
                    value = None
                    break
                prov = Provenance(
                    node.name, {**node.config, **cfg}, (prov,) if prov else (), seqs[i]
                )
                seqs[i] += 1
                if isinstance(result, Frame):
                    frame = _stamp_tags(result.evolve(prov=prov), node.name, (frame,))
                    value = result.data
                else:
                    value = result
                    if not d.keep_samples and (frame.samples or frame.tags):
                        frame = _clear_samples(frame)
                    if d.framed or i == len(nodes) - 1:
                        frame = frame.evolve(data=value, prov=prov)
                frame = _drop_cleared(frame, node.out_type)
                if (
                    check_all or (check_first and st.out == 0)
                ) and value is not MISSING:
                    try:
                        node.out_type.check(
                            value, self.opts.env, where=f"{node.name} output"
                        )
                        _check_tags(node.out_type, frame, where=f"{node.name} output")
                    except TypeMismatch as e:
                        raise TypeMismatch(f"runtime check failed: {e}") from None
                st.out += 1
            if value is None:
                continue  # dropped by some block
            if failed:
                yield frame.evolve(prov=prov)
                continue
            out = (
                frame
                if frame.prov is prov and frame.data is value
                else frame.evolve(data=value, prov=prov)
            )
            yield out

    async def _join(
        self, node: Any, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        """A tuple join: its inputs were zipped by the scheduler; pass the tuple frames on."""
        async for f in upstream:
            yield f

    # -- explicit graphs ---------------------------------------------------- #
    async def _graph(
        self, g: Any, upstream: AsyncIterator[Frame] | None
    ) -> AsyncIterator[Frame]:
        """Schedule a builder Graph: every member becomes a task pumping its output into
        one queue per consuming edge; multi-input members zip their input queues."""
        from .builder import Join, _Input

        stats = self.mon.stats(g.name, g) if not g.transparent else None
        queues: dict[tuple[str, int], asyncio.Queue] = {}  # (consumer, slot) -> queue
        out_qs: dict[str, asyncio.Queue] = {}  # graph outputs
        for src, dst, slot in g.edges:
            # bounded edges give backpressure; edges into a join are unbounded so a branch
            # that changes cardinality (a window on one side) cannot deadlock the pump
            bounded = not isinstance(g.members[dst], Join)
            queues[(dst, slot)] = asyncio.Queue(self.opts.queue_size if bounded else 0)
        for r in g.outputs:
            out_qs[r.name] = asyncio.Queue(self.opts.queue_size)
        tasks: list[asyncio.Task] = []
        failure: asyncio.Future = (
            asyncio.get_running_loop().create_future()
        )  # first error in any member

        def inputs_for(name: str) -> AsyncIterator[Frame] | None:
            node = g.members[name]
            if isinstance(node, _Input):
                return upstream
            slots = sorted(k[1] for k in queues if k[0] == name)
            if not slots:
                return None
            if len(slots) == 1 and (g._n_inputs(node) == 1 or slots[0] == -1):
                return _drain(queues[(name, slots[0])], failure)
            op = getattr(node, "join_op", "inputs")
            return self._zip_streams(
                [_drain(queues[(name, i)], failure) for i in slots], op, node
            )

        async def pump(name: str, stream: AsyncIterator[Frame]) -> None:
            edges = g.consumers(name)
            selects = {}
            for src, dst, slot in edges:
                if src.key is not None:
                    selects[(dst, slot)] = Select(g.members[name], src.key)
            outs = [out_qs[name]] if name in out_qs else []
            try:
                async for f in stream:
                    for src, dst, slot in edges:
                        item = f
                        sel = selects.get((dst, slot))
                        if sel is not None:
                            item = _select_frame(f, sel)
                        await queues[(dst, slot)].put(item)
                    for q in outs:
                        await q.put(f)
            except BaseException as e:
                # a failure (or cancellation) is recorded once for the whole graph; every
                # consumer checks it, so the run re-raises instead of ending quietly
                if not failure.done():
                    failure.set_exception(e)
                for q in [queues[(dst, slot)] for _, dst, slot in edges] + outs:
                    try:
                        q.put_nowait(
                            ("error", e)
                        )  # wake a blocked consumer (best effort)
                    except asyncio.QueueFull:
                        pass
                raise
            finally:
                for src, dst, slot in edges:
                    await queues[(dst, slot)].put(_DONE)
                for q in outs:
                    await q.put(_DONE)

        # The single output member, when nothing else consumes it, is iterated directly by
        # this generator instead of through a pump task and a queue (one hop less).
        direct = (
            g.outputs[0].name
            if len(g.outputs) == 1 and not g.consumers(g.outputs[0].name)
            else None
        )
        direct_stream = None
        for name in g.order():
            node = g.members[name]
            stream = (
                self.compile(node, inputs_for(name))
                if not isinstance(node, _Input)
                else inputs_for(name)
            )
            assert stream is not None
            if name == direct:
                direct_stream = stream
                continue
            tasks.append(asyncio.create_task(pump(name, stream)))

        try:
            if direct is not None:
                key = g.outputs[0].key
                sel = Select(g.members[direct], key) if key is not None else None
                assert direct_stream is not None
                async for f in direct_stream:
                    if failure.done():
                        _reraise(failure)
                    if sel is not None:
                        f = _select_frame(f, sel)
                    if stats is None:
                        yield f
                    else:
                        stats.out += 1
                        yield f.evolve(
                            prov=Provenance(
                                g.name, {}, (f.prov,) if f.prov else (), stats.out - 1
                            )
                        )
                if failure.done():
                    _reraise(failure)
            elif len(g.outputs) == 1:
                q = out_qs[g.outputs[0].name]
                key = g.outputs[0].key
                while True:
                    if failure.done():
                        _reraise(failure)
                    it = await q.get()
                    if it is _DONE:
                        if failure.done():
                            _reraise(failure)
                        return
                    frame: Frame = _unwrap(it)
                    if key is not None:
                        frame = _select_frame(
                            frame, Select(g.members[g.outputs[0].name], key)
                        )
                    if stats is None:
                        yield frame  # lowered operator tree: no wrapper node
                    else:
                        stats.out += 1
                        yield frame.evolve(
                            prov=Provenance(
                                g.name,
                                {},
                                (frame.prov,) if frame.prov else (),
                                stats.out - 1,
                            )
                        )
            else:
                async for f in self._zip_streams(
                    [_drain(out_qs[r.name]) for r in g.outputs], "outputs", g
                ):
                    if stats is not None:
                        stats.out += 1
                    yield f
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    # -- per-frame sub-pipelines (feedback / when) ------------------------- #
    def _subpipe(self, node: Node) -> _SubPipe:
        return _SubPipe(self, node)

    async def _feedback(
        self, node: Feedback, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        stats = self.mon.stats(node.name, node)
        body = self._subpipe(node.body)
        state = Frame(node.init, prov=Provenance(f"{node.name}.init", {}, (), 0))
        seq = 0
        try:
            async for f in upstream:
                stats.inp += 1
                if f.failed:
                    yield f.evolve(
                        prov=Provenance(node.name, {}, (f.prov,) if f.prov else (), seq)
                    )
                    seq += 1
                    continue
                pair = Frame(
                    (f.data, state.data),
                    f.meta,
                    f.stream,
                    {},
                    f.issues,
                    Provenance(
                        "loop", {}, tuple(p.prov for p in (f, state) if p.prov), seq
                    ),
                    parts=(f, state),
                    tags=f.tags.whole(),
                )
                out = await body.one(pair)
                if out is None:
                    raise RuntimeError(
                        f"{node.name}: body produced no output for a frame (must be 1:1)"
                    )
                if out.failed:
                    yield out.evolve(
                        prov=Provenance(
                            node.name, {}, (out.prov,) if out.prov else (), seq
                        )
                    )
                else:
                    parts = (
                        out.parts
                        if len(out.parts) == 2
                        else tuple(out.evolve(data=x, parts=()) for x in out.data)
                    )
                    y, state = parts
                    yield y.evolve(
                        prov=Provenance(
                            node.name, {}, (out.prov,) if out.prov else (), seq
                        )
                    )
                stats.out += 1
                seq += 1
        finally:
            await body.close()

    async def _when(
        self, node: When, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        stats = self.mon.stats(node.name, node)
        then = self._subpipe(node.then)
        other = self._subpipe(node.otherwise) if node.otherwise is not None else None
        seq = 0
        try:
            async for f in upstream:
                stats.inp += 1
                take = bool(node.pred(f))
                if take:
                    out = await then.one(f)
                elif other is not None:
                    out = await other.one(f)
                else:
                    out = f
                if out is None:
                    continue
                yield out.evolve(
                    prov=Provenance(
                        f"{node.name}[{'then' if take else 'else'}]",
                        {},
                        (out.prov,) if out.prov else (),
                        seq,
                    )
                )
                stats.out += 1
                seq += 1
        finally:
            await then.close()
            if other is not None:
                await other.close()

    # -- keyed routing ------------------------------------------------------ #
    _NOKEY = object()

    def _route(self, node: Switch, f: Frame) -> tuple[Any, Frame]:
        """Pick the branch key for a frame; a selector error marks the frame as failed."""
        try:
            key = node.key_of(f)
        except Exception as e:
            issue = Issue(f"{type(e).__name__}: {e}", "error", node.name, e, f.prov)
            return self._NOKEY, f.evolve(data=MISSING, issues=f.issues + (issue,))
        return (key if key in node.branches else self._NOKEY), f

    async def _switch(
        self, node: Switch, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        """Concurrent routing: each branch is a real stream fed by its own queue; outputs are
        merged in completion order (frames routed to one branch keep their order)."""
        stats = self.mon.stats(node.name, node)
        outq: asyncio.Queue = asyncio.Queue()
        inqs: dict[Any, asyncio.Queue] = {}
        tasks: list[asyncio.Task] = []
        labels: dict[Any, str] = {}

        async def feed(q: asyncio.Queue) -> AsyncIterator[Frame]:
            async for f in _drain(q):
                yield f

        async def drive(label: str, branch: Node, q: asyncio.Queue) -> None:
            try:
                async for f in self.compile(branch, feed(q)):
                    await outq.put((label, f))
            except BaseException as e:
                await outq.put(("error", e))
                raise
            finally:
                await outq.put(_DONE)

        members: list[tuple[Any, str, Node]] = [
            (k, repr(k), b) for k, b in node.branches.items()
        ]
        if isinstance(node.default, Node):
            members.append((self._NOKEY, "default", node.default))
        for key, label, branch in members:
            q: asyncio.Queue = asyncio.Queue(maxsize=self.opts.queue_size)
            inqs[key], labels[key] = q, label
            tasks.append(asyncio.create_task(drive(label, branch, q)))
        live_ = len(tasks)

        async def router() -> None:
            try:
                async for f in upstream:
                    stats.inp += 1
                    key, f = self._route(node, f)
                    if key is self._NOKEY and self._NOKEY not in inqs:
                        if node.default == "pass":
                            await outq.put(("pass", f))
                        elif self.opts.strict:
                            raise KeyError(
                                f"{node.name}: no branch for key {node.key_of(f)!r} and no default"
                            )
                        continue  # dropped
                    await inqs[key].put(f)
            except BaseException as e:
                await outq.put(("error", e))
                raise
            finally:
                for q in inqs.values():
                    await q.put(_DONE)
                await outq.put(_DONE)

        rt = asyncio.create_task(router())
        live_ += 1
        seq = 0
        try:
            while live_:
                it = await outq.get()
                if it is _DONE:
                    live_ -= 1
                    continue
                label, f = it
                if label == "error":
                    raise f
                stats.out += 1
                yield f.evolve(
                    prov=Provenance(
                        f"{node.name}[{label}]", {}, (f.prov,) if f.prov else (), seq
                    )
                )
                seq += 1
        finally:
            for t in [rt, *tasks]:
                t.cancel()
            for t in [rt, *tasks]:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    async def _switch_ordered(
        self, node: Switch, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        """Ordered routing: one frame at a time through 1:1 branches, like when()."""
        stats = self.mon.stats(node.name, node)
        pipes = {k: self._subpipe(b) for k, b in node.branches.items()}
        dflt = self._subpipe(node.default) if isinstance(node.default, Node) else None
        seq = 0
        try:
            async for f in upstream:
                stats.inp += 1
                key, f = self._route(node, f)
                if key is self._NOKEY:
                    if dflt is not None:
                        out, label = await dflt.one(f), "default"
                    elif node.default == "pass":
                        out, label = f, "pass"
                    elif self.opts.strict:
                        raise KeyError(
                            f"{node.name}: no branch for key {node.key_of(f)!r} and no default"
                        )
                    else:
                        continue
                else:
                    out, label = await pipes[key].one(f), repr(key)
                if out is None:
                    continue
                yield out.evolve(
                    prov=Provenance(
                        f"{node.name}[{label}]",
                        {},
                        (out.prov,) if out.prov else (),
                        seq,
                    )
                )
                stats.out += 1
                seq += 1
        finally:
            for p in pipes.values():
                await p.close()
            if dflt is not None:
                await dflt.close()

    async def _select(
        self, node: Select, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        stats = self.mon.stats(node.name, node)
        seq = 0
        async for f in upstream:
            stats.inp += 1
            if f.failed:
                yield f.evolve(
                    data=MISSING,
                    parts=(),
                    prov=Provenance(node.name, {}, (f.prov,) if f.prov else (), seq),
                )
            else:
                parts = (
                    f.parts
                    if len(f.parts) == len(f.data)
                    else tuple(f.evolve(data=x, parts=()) for x in f.data)
                )
                # the selected branch comes first so lineage follows it; the tuple node stays reachable
                if node.single:
                    part = parts[node.indices[0]]
                    parents = _unique(p for p in (part.prov, f.prov) if p)
                    yield part.evolve(prov=Provenance(node.name, {}, parents, seq))
                else:
                    chosen = [parts[i] for i in node.indices]
                    parents = _unique(
                        [p.prov for p in chosen if p.prov]
                        + ([f.prov] if f.prov else [])
                    )
                    yield f.evolve(
                        data=tuple(p.data for p in chosen),
                        parts=tuple(chosen),
                        prov=Provenance(node.name, {}, parents, seq),
                    )
            seq += 1
            stats.out += 1

    # -- sources ----------------------------------------------------------- #
    async def _source(self, node: Source) -> AsyncIterator[Frame]:
        stats = self.mon.stats(node.name, node)
        obj = node.obj
        stats.total = _length_of(obj)
        seq = 0
        skip = self._resume_position(node.name)
        async for item in aiter_any(obj, threaded=getattr(node, "threaded", False)):
            if seq < skip:
                seq += 1
                stats.out += 1
                continue
            await self._gate()
            f = self._wrap_output(item, node.name, {}, (), seq, node.out_type, stats)
            seq += 1
            stats.out += 1
            yield f

    async def _block_source(self, node: Block) -> AsyncIterator[Frame]:
        stats = self.mon.stats(node.name, node)
        d = node.defn
        try:
            produced = d.resolve_call(self.opts.labels)(
                **node.given
            )  # sources get Knob objects and read .value per frame
        except Exception as e:
            if self._policy(node) == "raise":
                raise
            stats.errors += 1
            yield Frame(
                MISSING,
                issues=(Issue(str(e), "error", node.name, e),),
                prov=Provenance(node.name, node.config),
            )
            return
        if inspect.isawaitable(produced):
            produced = await produced
        if not (
            hasattr(produced, "__iter__") or hasattr(produced, "__aiter__")
        ) or isinstance(produced, (str, bytes)):
            produced = [produced]  # one-shot source
        stats.total = (
            d.total(**live(node.config))
            if d.total is not None
            else _length_of(produced)
        )
        seq = 0
        skip = self._resume_position(node.name)
        it = aiter_any(produced, threaded=d.threaded)
        while True:
            try:
                item = await it.__anext__()
            except StopAsyncIteration:
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # a source that raises mid-stream ends the stream with a failed
                # frame attributed to *itself*, not to whatever consumed it
                stats.errors += 1
                if self._policy(node) == "raise":
                    raise
                if self._policy(node) == "drop":
                    return
                prov = Provenance(node.name, node.config, (), seq)
                yield Frame(
                    MISSING,
                    issues=(
                        Issue(f"{type(e).__name__}: {e}", "error", node.name, e, prov),
                    ),
                    prov=prov,
                )
                return
            if seq < skip:  # resuming: this frame was emitted before the checkpoint
                seq += 1
                stats.out += 1
                continue
            await self._gate()
            yield self._wrap_output(
                item, node.name, node.config, (), seq, node.out_type, stats
            )
            seq += 1
            stats.out += 1

    # -- map blocks -------------------------------------------------------- #
    async def _map_block(
        self, node: Block, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        stats = self.mon.stats(node.name, node)
        d = node.defn
        call = d.resolve_call(self.opts.labels)
        policy = self._policy(node)
        seq = 0
        needs = bool(tag_names(node.in_type))  # input tag requirements
        async for f in upstream:
            stats.inp += 1
            if f.failed and (not d.framed or len(d.inputs) > 1):
                # skip computation; propagate the failure with extended provenance.
                # (A frame-aware single-input block does see failed frames, so it can
                # handle them; a multi-input block never computes on a failed part.)
                yield f.evolve(
                    prov=Provenance(
                        node.name, node.config, (f.prov,) if f.prov else (), seq
                    )
                )
                seq += 1
                stats.out += 1
                continue
            if needs and (
                self.opts.check == "all"
                or (self.opts.check == "first" and stats.out == 0)
            ):
                _check_required(node.in_type, f, f"{node.name} input")
            t0 = time.perf_counter()
            try:
                args = self._args(d, f)
                cfg = (
                    live(node.given) if node.live else node.given
                )  # knobs resolved per call
                result = call(*args, **cfg)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as e:
                stats.errors += 1
                stats.time += time.perf_counter() - t0
                if policy == "raise":
                    raise
                if policy == "drop":
                    continue
                prov = Provenance(
                    node.name, node.config, (f.prov,) if f.prov else (), seq
                )
                issue = Issue(f"{type(e).__name__}: {e}", "error", node.name, e, prov)
                yield f.evolve(data=MISSING, issues=f.issues + (issue,), prov=prov)
                seq += 1
                stats.out += 1
                continue
            stats.time += time.perf_counter() - t0
            if d.tap:
                result = f
            elif result is None:
                continue  # None => filtered out
            out = self._wrap_output(
                result,
                node.name,
                {**node.config, **cfg},
                (f,),
                seq,
                node.out_type,
                stats,
                keep_samples=d.keep_samples,
            )  # provenance records the values used
            seq += 1
            stats.out += 1
            yield out

    def _args(self, d: BlockDef, f: Frame) -> list[Any]:
        if len(d.inputs) == 1:
            spec = d.inputs[0]
            return [f if spec.framed else f.data]
        data = f.data
        if not isinstance(data, tuple) or len(data) != len(d.inputs):
            raise TypeMismatch(
                f"{d.name} expects {len(d.inputs)} inputs, got {type(data).__name__}"
            )
        parts = (
            f.parts
            if len(f.parts) == len(data)
            else tuple(f.evolve(data=x, parts=()) for x in data)
        )
        return [parts[i] if spec.framed else data[i] for i, spec in enumerate(d.inputs)]

    # -- stream blocks ----------------------------------------------------- #
    async def _stream_block(
        self, node: Block, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        stats = self.mon.stats(node.name, node)
        d = node.defn
        call = d.resolve_call(self.opts.labels)
        spec = d.inputs[0]
        needs = bool(tag_names(node.in_type))  # input tag requirements
        if len(d.inputs) != 1:
            raise TypeError(f"{d.name}: stream blocks take exactly one stream input")
        policy = self._policy(node)
        consumed: list[Frame] = []
        failed_q: asyncio.Queue = asyncio.Queue()

        first_len: list[int] = []

        async def frames() -> AsyncIterator[Frame]:
            async for f in upstream:
                stats.inp += 1
                if f.failed:
                    # Failed frames always bypass stream blocks (even frame-aware ones): a
                    # generator that raised cannot be resumed, so it must never see MISSING data.
                    failed_q.put_nowait(f)
                    continue
                if (
                    not first_len
                    and hasattr(f.data, "shape")
                    and getattr(f.data, "ndim", 0) >= 1
                ):
                    first_len.append(len(f.data))
                if needs and (
                    self.opts.check == "all"
                    or (self.opts.check == "first" and stats.out == 0)
                ):
                    _check_required(node.in_type, f, f"{node.name} input")
                yield f

        async def items() -> AsyncIterator[Any]:
            # consumption is recorded when the block *pulls* an item, so the
            # parents of each output are exactly the frames consumed since the last one
            async for f in frames():
                consumed.append(f)
                yield f if spec.framed else f.data

        seq = 0
        try:
            if inspect.isasyncgenfunction(d.fn):
                agen = call(items(), **node.given)
                last: tuple[Frame, ...] = ()
                async for out in agen:
                    parents, consumed[:] = tuple(consumed) or last, []
                    last = parents  # outputs that consumed nothing inherit the previous parents
                    while not failed_q.empty():
                        yield self._forward_failed(failed_q.get_nowait(), node, seq)
                        seq += 1
                    yield self._wrap_output(
                        out,
                        node.name,
                        node.config,
                        parents,
                        seq,
                        node.out_type,
                        stats,
                        keep_samples=False,
                    )
                    seq += 1
                    stats.out += 1
            else:
                driver = (
                    _run_inline_stream
                    if (_greenlet is not None and not d.threaded)
                    else _run_sync_stream
                )
                async for out, parents in driver(
                    call,
                    node.given,
                    frames(),
                    consumed,
                    spec.framed,
                    self.opts.queue_size,
                ):
                    while not failed_q.empty():
                        yield self._forward_failed(failed_q.get_nowait(), node, seq)
                        seq += 1
                    yield self._wrap_output(
                        out,
                        node.name,
                        node.config,
                        parents,
                        seq,
                        node.out_type,
                        stats,
                        keep_samples=False,
                    )
                    seq += 1
                    stats.out += 1
        except Exception as e:
            stats.errors += 1
            if policy == "raise":
                raise
            if policy == "drop":
                return
            issue = Issue(f"{type(e).__name__}: {e}", "error", node.name, e)
            yield Frame(
                MISSING,
                issues=(issue,),
                prov=Provenance(
                    node.name,
                    node.config,
                    tuple(c.prov for c in consumed if c.prov),
                    seq,
                ),
            )
        while not failed_q.empty():
            yield self._forward_failed(failed_q.get_nowait(), node, seq)
            seq += 1
        if self.opts.strict:
            self._check_ratio(node, stats, first_len[0] if first_len else None)

    def _check_ratio(self, node: Block, stats: Any, n_in: int | None) -> None:
        """strict=True: a stream block with an exact ratio must have honoured it."""
        from types import SimpleNamespace

        from .ratio import RatioMismatch

        r = node.ratio_for(
            SimpleNamespace(shape=(n_in,)) if n_in is not None else node.in_type
        )
        if r.kind != "exact" or r.value is None or stats.inp == 0:
            return
        v = float(r.value)
        expected = stats.inp * v
        if abs(stats.out - expected) > max(1.0, v) + 1:
            raise RatioMismatch(
                f"{node.name} declares ratio {r} but emitted {stats.out} frames for "
                f"{stats.inp} inputs (expected about {expected:.0f})"
            )

    def _forward_failed(self, f: Frame, node: Block, seq: int) -> Frame:
        self.mon.stats(node.name).out += 1
        return f.evolve(
            prov=Provenance(node.name, node.config, (f.prov,) if f.prov else (), seq)
        )

    # -- fan-out / zip / merge --------------------------------------------- #
    async def _zip_streams(
        self, streams: list[AsyncIterator[Frame]], op: str, node: Node
    ) -> AsyncIterator[Frame]:
        stats = self.mon.stats(node.name, node)
        out_qs: list[asyncio.Queue] = [asyncio.Queue() for _ in streams]

        async def drive(s: AsyncIterator[Frame], q: asyncio.Queue) -> None:
            try:
                async for f in s:
                    await q.put(f)
            except BaseException as e:
                await q.put(("error", e))
                raise
            finally:
                await q.put(_DONE)

        tasks = [asyncio.create_task(drive(s, q)) for s, q in zip(streams, out_qs)]
        seq = 0
        try:
            while True:
                items = []
                for q in out_qs:
                    it = await q.get()
                    if isinstance(it, tuple) and len(it) == 2 and it[0] == "error":
                        raise it[1]
                    items.append(it)
                if any(it is _DONE for it in items):
                    if self.opts.strict and not all(it is _DONE for it in items):
                        alive = [i for i, it in enumerate(items) if it is not _DONE]
                        raise TypeMismatch(
                            f"{node.name}: branches produced different numbers of frames "
                            f"(branch(es) {alive} still had output when others ended); "
                            f"pair branches that change cardinality deliberately"
                        )
                    break
                merged = merge_annotations(items, keep_samples=False)
                yield Frame(
                    tuple(f.data for f in items),
                    merged["meta"],
                    merged["stream"],
                    {},
                    merged["issues"],
                    Provenance(op, {}, tuple(f.prov for f in items if f.prov), seq),
                    parts=tuple(items),
                    tags=merged["tags"],
                )
                stats.out += 1
                seq += 1
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    async def _parallel(
        self, node: Parallel, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        """Submit frames to a pool, keep a window of in-flight futures, yield results in order."""
        import concurrent.futures as cf
        from collections import deque

        inner = node.inner
        d = inner.defn
        call = d.resolve_call(self.opts.labels)
        needs = bool(tag_names(inner.in_type))  # input tag requirements
        stats = self.mon.stats(node.name, node)
        policy = self._policy(inner)
        pool = (
            cf.ProcessPoolExecutor(node.workers, mp_context=_worker_context())
            if node.backend == "process"
            else cf.ThreadPoolExecutor(node.workers)
        )
        loop = asyncio.get_running_loop()
        pending: deque = deque()  # (frame, future | None)
        seq = 0

        async def emit_ready(block_on_first: bool) -> AsyncIterator[Frame]:
            nonlocal seq
            while pending and (
                block_on_first or pending[0][1] is None or pending[0][1].done()
            ):
                f, fut = pending.popleft()
                if fut is None:  # failed frame passed through untouched
                    yield f.evolve(
                        prov=Provenance(
                            node.name, inner.config, (f.prov,) if f.prov else (), seq
                        )
                    )
                    seq += 1
                    stats.out += 1
                    continue
                t0 = time.perf_counter()
                try:
                    result = await fut
                except Exception as e:
                    stats.errors += 1
                    if policy == "raise":
                        raise
                    if policy == "drop":
                        continue
                    prov = Provenance(
                        inner.name, inner.config, (f.prov,) if f.prov else (), seq
                    )
                    yield f.evolve(
                        data=MISSING,
                        issues=f.issues
                        + (
                            Issue(
                                f"{type(e).__name__}: {e}", "error", inner.name, e, prov
                            ),
                        ),
                        prov=prov,
                    )
                    seq += 1
                    stats.out += 1
                    continue
                stats.time += time.perf_counter() - t0
                if result is None:
                    continue
                yield self._wrap_output(
                    result,
                    inner.name,
                    inner.config,
                    (f,),
                    seq,
                    inner.out_type,
                    stats,
                    keep_samples=d.keep_samples,
                )
                seq += 1
                stats.out += 1

        try:
            async for f in upstream:
                stats.inp += 1
                if f.failed:
                    pending.append((f, None))
                else:
                    if needs and (
                        self.opts.check == "all"
                        or (self.opts.check == "first" and stats.out == 0)
                    ):
                        _check_required(inner.in_type, f, f"{inner.name} input")
                    cfg = live(inner.given) if inner.live else dict(inner.given)
                    fut = loop.run_in_executor(
                        pool,
                        _call_block,
                        d if node.backend == "process" else call,
                        f.data,
                        cfg,
                        self.opts.labels,
                    )
                    pending.append((f, fut))
                async for out in emit_ready(block_on_first=len(pending) >= node.window):
                    yield out
            async for out in emit_ready(block_on_first=True):
                yield out
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    async def _align(self, node: Align) -> AsyncIterator[Frame]:
        """Greedy time matching: keep the head of every stream; when all heads agree within
        the tolerance, emit; otherwise advance the earliest head (it has no partner)."""
        stats = self.mon.stats(node.name, node)
        streams = [self.compile(p, None) for p in node.parts]
        heads: list = [None] * len(streams)
        done = [False] * len(streams)
        seq = 0

        async def advance(i: int) -> None:
            try:
                heads[i] = await streams[i].__anext__()
            except StopAsyncIteration:
                heads[i], done[i] = None, True

        for i in range(len(streams)):
            await advance(i)
        try:
            while not any(done):
                times = []
                for h in heads:
                    t = node.key(h)
                    if t is None:
                        raise TypeMismatch(
                            f"{node.name}: a frame from {h.prov.op if h.prov else '?'} has no "
                            f"time stamp (t0); use dsp.timestamp or set meta['t0']"
                        )
                    times.append(float(t))
                lo, hi = min(times), max(times)
                if hi - lo <= node.tolerance:
                    items = list(heads)
                    merged = merge_annotations(items, keep_samples=False)
                    merged["meta"]["t0"] = hi
                    merged["meta"]["skew"] = hi - lo
                    yield Frame(
                        tuple(f.data for f in items),
                        merged["meta"],
                        merged["stream"],
                        {},
                        merged["issues"],
                        Provenance(
                            node.name,
                            {"tolerance": node.tolerance},
                            tuple(f.prov for f in items if f.prov),
                            seq,
                        ),
                        tags=merged["tags"],
                        parts=tuple(items),
                    )
                    stats.out += 1
                    seq += 1
                    for i in range(len(streams)):
                        await advance(i)
                else:
                    stats.errors += 0
                    await advance(
                        int(np.argmin(times))
                    )  # drop the unmatched earliest frame
                    stats.inp += 1  # count dropped frames as "in but unmatched"
        finally:
            for st in streams:
                try:
                    await st.aclose()  # type: ignore[attr-defined]
                except Exception:
                    pass

    async def _merge(self, node: Merge) -> AsyncIterator[Frame]:
        stats = self.mon.stats(node.name, node)
        q: asyncio.Queue = asyncio.Queue()

        async def drive(s: AsyncIterator[Frame]) -> None:
            try:
                async for f in s:
                    await q.put(f)
            except BaseException as e:
                await q.put(("error", e))
                raise
            finally:
                await q.put(_DONE)

        tasks = [asyncio.create_task(drive(self.compile(p, None))) for p in node.parts]
        live = len(tasks)
        seq = 0
        try:
            while live:
                it = await q.get()
                if it is _DONE:
                    live -= 1
                    continue
                merged: Frame = _unwrap(it)
                stats.out += 1
                yield merged.evolve(
                    prov=Provenance(
                        "merge", {}, (merged.prov,) if merged.prov else (), seq
                    )
                )
                seq += 1
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    # -- helpers ----------------------------------------------------------- #
    def _resume_position(self, name: str) -> int:
        ck = self.opts.resume
        return int(ck.positions.get(name, 0)) if ck is not None else 0

    async def _gate(self) -> None:
        if self.opts.control is not None:
            await self.opts.control.wait()

    def _policy(self, node: Block) -> str:
        return self.opts.on_error or node.defn.on_error

    def _wrap_output(
        self,
        result: Any,
        op: str,
        params: Any,
        parents: tuple[Frame, ...],
        seq: int,
        out_type: Any,
        stats: Any,
        keep_samples: bool = False,
    ) -> Frame:
        prov = Provenance(op, params, tuple(p.prov for p in parents if p.prov), seq)
        if isinstance(result, Frame):
            out = result.evolve(
                prov=prov
            )  # a block's output is a new computation, whatever it returned
            out = _stamp_tags(out, op, parents)
        else:
            ann = merge_annotations(parents, keep_samples=keep_samples)
            out = Frame(
                result,
                ann["meta"],
                ann["stream"],
                ann["samples"],
                ann["issues"],
                prov,
                tags=ann["tags"],
            )
        out = _drop_cleared(out, out_type)
        # runtime contract check
        checked = self.opts.check == "all" or (
            self.opts.check == "first" and stats.out == 0
        )
        if checked and out.data is not MISSING:
            try:
                out_type.check(out.data, self.opts.env, where=f"{op} output")
                _check_tags(out_type, out, where=f"{op} output")
            except TypeMismatch as e:
                raise TypeMismatch(f"runtime check failed: {e}") from None
        return out


class _SubPipe:
    """Drive a compiled node one frame at a time: push an input, await its output.

    Used for the bodies of feedback() and when(). A body may also *drop* a frame
    (a map block returning None): that is detected when the body asks for its next
    input without having emitted anything, and ``one()`` returns None. Bodies must
    not buffer across frames (a windowing stream block inside is not supported)."""

    def __init__(self, runner: Runner, node: Node) -> None:
        self.inq: asyncio.Queue = asyncio.Queue()
        self.outq: asyncio.Queue = asyncio.Queue()
        self.pulled = asyncio.Event()  # set whenever the body asks for the next input
        self.stream = runner.compile(node, self._upstream())
        self.task = asyncio.create_task(self._pump())

    async def _upstream(self) -> AsyncIterator[Frame]:
        while True:
            self.pulled.set()
            it = await self.inq.get()
            if it is _DONE:
                return
            yield it

    async def _pump(self) -> None:
        try:
            async for f in self.stream:
                await self.outq.put(f)
        except BaseException as e:
            await self.outq.put(("error", e))
        finally:
            await self.outq.put(_DONE)

    async def one(self, f: Frame) -> Frame | None:
        self.pulled.clear()
        await self.inq.put(f)
        get_out = asyncio.ensure_future(self.outq.get())
        get_pull = asyncio.ensure_future(self.pulled.wait())
        done, _ = await asyncio.wait(
            {get_out, get_pull}, return_when=asyncio.FIRST_COMPLETED
        )
        if get_out in done:
            get_pull.cancel()
            item = get_out.result()
        else:
            # The body asked for its next input. A 1:1 body emits before it pulls again, so
            # any output for this frame is already queued (or already taken by get_out).
            get_out.cancel()
            try:
                await get_out
            except asyncio.CancelledError:
                pass
            if get_out.done() and not get_out.cancelled():
                item = get_out.result()
            elif not self.outq.empty():
                item = self.outq.get_nowait()
            else:
                return None  # dropped by the body
        if item is _DONE:
            return None
        return _unwrap(item)

    async def close(self) -> None:
        await self.inq.put(_DONE)
        self.task.cancel()
        try:
            await self.task
        except (asyncio.CancelledError, Exception):
            pass


def _worker_context() -> Any:
    """How process-pool workers start. The runtime is multi-threaded (an event loop,
    helper threads), so forking it is unsafe and Python 3.12+ warns; a fork server
    (Linux) or a fresh interpreter starts workers cleanly. Blocks sent to a worker
    are re-imported there, which ``backend="process"`` already requires."""
    import multiprocessing

    try:
        return multiprocessing.get_context("forkserver")
    except ValueError:  # no fork server on this platform
        return multiprocessing.get_context("spawn")


def _call_block(target: Any, data: Any, cfg: dict, labels: bool = False) -> Any:
    """Worker-side call (module-level so it pickles for process pools). A process
    worker gets the definition and the run's resolved ``labels`` flag."""
    fn = target.resolve_call(labels) if isinstance(target, BlockDef) else target
    return fn(data, **cfg)


def _clear_samples(f: Frame) -> Frame:
    """The frame without what only made sense along its leading axis: sample
    annotations and spanned tags."""
    return Frame(
        f.data, f.meta, f.stream, {}, f.issues, f.prov, f.parts, f.tags.whole()
    )


def _stamp_tags(frame: Frame, op: str, parents: tuple[Frame, ...]) -> Frame:
    """Record ``op`` as the origin of the tags this block attached: those without an
    origin that no parent already carried."""
    if not frame.tags:
        return frame
    inherited = {id(t) for p in parents for t in p.tags}
    stamped = [
        t if t.origin or id(t) in inherited else _with_origin(t, op) for t in frame.tags
    ]
    if all(a is b for a, b in zip(stamped, frame.tags)):
        return frame
    from dataclasses import replace

    return replace(frame, tags=Tags(stamped))


def _with_origin(tag: Any, op: str) -> Any:
    from dataclasses import replace

    return replace(tag, origin=op)


def _drop_cleared(frame: Frame, out_type: Any) -> Frame:
    """The tags a block's contract clears do not leave it. A tuple output is cleared
    port by port: a frame that kept its parts has each part cleared, a plain tuple
    payload gets parts made for it when the ports differ, and the aggregate loses
    what every port clears."""
    if not frame.tags and not any(p.tags for p in frame.parts):
        return frame
    from dataclasses import replace

    from .types import TupleType, cleared_names

    if isinstance(out_type, TupleType):
        per = [cleared_names(t) for t in out_type.items]
        parts = frame.parts
        if (
            not parts
            and isinstance(frame.data, tuple)
            and len(frame.data) == len(per)
            and any(p != per[0] for p in per)
            and any(p & frame.tags.names for p in per)
        ):
            parts = tuple(
                Frame(
                    d,
                    frame.meta,
                    frame.stream,
                    {},
                    frame.issues,
                    frame.prov,
                    tags=frame.tags,
                )
                for d in frame.data
            )
        if parts and len(parts) == len(per):
            cleared = tuple(_drop_cleared(p, t) for p, t in zip(parts, out_type.items))
            gone = frozenset.intersection(*per)
            if cleared == frame.parts and not (gone & frame.tags.names):
                return frame
            return replace(
                frame,
                parts=cleared,
                tags=Tags(t for t in frame.tags if t.name not in gone),
            )
    gone = cleared_names(out_type)
    if not gone or not (gone & frame.tags.names):
        return frame
    return replace(frame, tags=Tags(t for t in frame.tags if t.name not in gone))


def _check_tags(out_type: Any, frame: Frame, where: str) -> None:
    """A guarantee is about whole-frame tags: a span need not reach every frame. A
    tuple frame with parts is checked port by port."""
    from .types import TupleType, tag_names

    if (
        isinstance(out_type, TupleType)
        and frame.parts
        and len(frame.parts) == len(out_type.items)
    ):
        for i, (t, p) in enumerate(zip(out_type.items, frame.parts)):
            _check_tags(t, p, f"{where}[{i}]")
        return
    missing = tag_names(out_type) - frame.tags.whole().names
    if missing:
        raise TypeMismatch(
            f"{where}: contract guarantees tag(s) {'|'.join(sorted(missing))}, "
            f"frame carries {'|'.join(sorted(frame.tags.whole().names)) or 'none'}"
        )


def _check_required(in_type: Any, frame: Frame, where: str) -> None:
    """What a block's input contract requires must be on the frame it gets (whole-
    frame tags; port by port for a tuple frame with parts). Producers' output checks
    cover this in a graph; this guards frames that reach a block another way."""
    from .types import TupleType, tag_names

    if (
        isinstance(in_type, TupleType)
        and frame.parts
        and len(frame.parts) == len(in_type.items)
    ):
        for i, (t, p) in enumerate(zip(in_type.items, frame.parts)):
            _check_required(t, p, f"{where}[{i}]")
        return
    missing = tag_names(in_type) - frame.tags.whole().names
    if missing:
        raise TypeMismatch(
            f"runtime check failed: {where}: needs tag(s) {'|'.join(sorted(missing))}, "
            f"frame carries {'|'.join(sorted(frame.tags.whole().names)) or 'none'}"
        )


def _select_frame(f: Frame, node: Select) -> Frame:
    """Apply a port selection to one frame (used on graph edges)."""
    if f.failed:
        return f.evolve(data=MISSING, parts=())
    parts = (
        f.parts
        if len(f.parts) == len(f.data)
        else tuple(f.evolve(data=x, parts=()) for x in f.data)
    )
    if node.single:
        part = parts[node.indices[0]]
        return part.evolve(
            prov=Provenance(
                node.name,
                {},
                _unique(p for p in (part.prov, f.prov) if p),
                f.prov.seq if f.prov else None,
            )
        )
    chosen = [parts[i] for i in node.indices]
    return f.evolve(
        data=tuple(p.data for p in chosen),
        parts=tuple(chosen),
        prov=Provenance(
            node.name,
            {},
            _unique([p.prov for p in chosen if p.prov] + ([f.prov] if f.prov else [])),
        ),
    )


def _unique(items: Any) -> tuple:
    out: list = []
    for it in items:
        if all(it is not o for o in out):
            out.append(it)
    return tuple(out)


async def _drain(
    q: asyncio.Queue, failure: asyncio.Future | None = None
) -> AsyncIterator[Frame]:
    while True:
        if failure is not None and failure.done():
            _reraise(failure)
        it = await q.get()
        if it is _DONE:
            if failure is not None and failure.done():
                _reraise(failure)
            return
        yield _unwrap(it)


try:
    import greenlet as _greenlet
except ImportError:  # pragma: no cover - optional
    _greenlet = None


async def _run_inline_stream(
    fn: Callable[..., Any],
    config: Any,
    frames: AsyncIterator[Frame],
    consumed: list[Frame],
    framed: bool,
    qsize: int,
) -> AsyncIterator[tuple[Any, tuple[Frame, ...]]]:
    """Run a synchronous generator stream block *on the event loop* with greenlet: the
    generator's pull for input switches back here, we await the next frame, and switch it
    in. No thread, no queues — a few microseconds per frame instead of hundreds."""
    assert _greenlet is not None  # only reached when greenlet is installed
    parent = _greenlet.getcurrent()
    NEED, OUT, DONE_ = "need", "out", "done"

    def feed() -> Iterator[Any]:
        while True:
            item = parent.switch((NEED,))
            if item is _DONE:
                return
            consumed.append(item)
            yield item if framed else item.data

    def body() -> tuple:
        last: tuple[Frame, ...] = ()
        for out in fn(feed(), **config):
            parents, consumed[:] = tuple(consumed) or last, []
            last = parents
            parent.switch((OUT, out, parents))
        return (DONE_,)

    g = _greenlet.greenlet(body)
    res = g.switch()
    try:
        while True:
            if res[0] == NEED:
                nxt: Any
                try:
                    nxt = await frames.__anext__()
                except StopAsyncIteration:
                    nxt = _DONE
                res = g.switch(nxt)
            elif res[0] == OUT:
                yield res[1], res[2]
                res = g.switch(None)
            else:
                return
    finally:
        if not g.dead:
            try:
                g.throw(GeneratorExit)  # closes the generator; it re-raises into us
            except (Exception, GeneratorExit):
                pass


async def _run_sync_stream(
    fn: Callable[..., Any],
    config: Any,
    frames: AsyncIterator[Frame],
    consumed: list[Frame],
    framed: bool,
    qsize: int,
) -> AsyncIterator[tuple[Any, tuple[Frame, ...]]]:
    """Run a synchronous generator-function stream block in a worker thread."""
    loop = asyncio.get_running_loop()
    inq: queue.Queue = queue.Queue(maxsize=qsize)
    outq: asyncio.Queue = asyncio.Queue()
    lock = threading.Lock()

    def feed() -> Iterator[Any]:
        while True:
            f = inq.get()
            if f is _DONE:
                return
            with lock:
                consumed.append(f)
            yield f if framed else f.data

    def worker() -> None:
        last: tuple[Frame, ...] = ()
        try:
            for out in fn(feed(), **config):
                with lock:
                    parents, consumed[:] = tuple(consumed) or last, []
                    last = parents
                loop.call_soon_threadsafe(outq.put_nowait, ("item", out, parents))
        except BaseException as e:
            loop.call_soon_threadsafe(outq.put_nowait, ("error", e, ()))
        finally:
            loop.call_soon_threadsafe(outq.put_nowait, ("done", None, ()))

    async def pump() -> None:
        try:
            async for f in frames:
                await loop.run_in_executor(None, inq.put, f)
        except BaseException as e:
            await outq.put(("error", e, ()))
        finally:
            await loop.run_in_executor(None, inq.put, _DONE)

    th = threading.Thread(target=worker, daemon=True, name="sigflow-stream")
    th.start()
    pump_task = asyncio.create_task(pump())
    try:
        while True:
            kind, val, parents = await outq.get()
            if kind == "item":
                yield val, parents
            elif kind == "error":
                raise val
            else:
                break
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except (asyncio.CancelledError, Exception):
            pass
        # unblock the worker if it is still waiting for input
        try:
            inq.put_nowait(_DONE)
        except queue.Full:
            pass


def _length_of(obj: Any) -> int | None:
    try:
        return len(obj)
    except TypeError:
        return None


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def iterate(graph: Node, **options: Any) -> AsyncIterator[Frame]:
    """Async iterator over the output frames of a graph."""
    graph.validate()
    opts = RunOptions(**{k: v for k, v in options.items() if k in _RUN_OPTIONS})
    runner = Runner(opts)
    from .frame import PROVENANCE_MODE

    PROVENANCE_MODE.set(opts.provenance)
    return runner.compile(graph, None)


async def arun(
    graph: Node,
    *,
    sink: Callable[[Frame], Any] | None = None,
    collect: bool = True,
    progress: bool | str = False,
    total: int | None = None,
    **options: Any,
) -> list[Frame]:
    """Run a graph to completion inside a running event loop."""
    if isinstance(options.get("resume"), str):
        from .persist import load_checkpoint

        options["resume"] = load_checkpoint(options["resume"])
    opts = RunOptions(**options)
    if opts.resume is not None:
        from .persist import restore

        restore(graph, opts.resume)
    mon = opts.monitor
    mon.total = total
    graph.validate()
    try:
        mon.ratio = (
            graph.ratio
        )  # scales the sources' totals into an expected output count
    except Exception:  # pragma: no cover - never let a ratio bug stop a run
        mon.ratio = None
    graph.last_monitor = mon  # type: ignore[attr-defined]
    runner = Runner(opts)
    from .frame import PROVENANCE_MODE

    token = PROVENANCE_MODE.set(opts.provenance)
    out: list[Frame] = []
    reporter = Reporter(mon, graph, mode=progress) if progress else None
    mon.start()
    try:
        async for f in runner.compile(graph, None):
            if sink is not None:
                r = sink(f)
                if inspect.isawaitable(r):
                    await r
            if collect:
                out.append(f)
            mon.produced += 1
            if reporter:
                reporter.tick()
    finally:
        PROVENANCE_MODE.reset(token)
        mon.stop()
        if reporter:
            reporter.finish()
    return out


def run(graph: Node, **kw: Any) -> list[Frame]:
    """Run a graph synchronously.  Works from scripts and from Jupyter cells alike."""
    coro = arun(graph, **kw)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is already running (Jupyter): run on a private loop in a thread.
    result: list[Any] = []
    error: list[BaseException] = []

    def target() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as e:
            error.append(e)

    t = threading.Thread(target=target, name="sigflow-run")
    t.start()
    t.join()
    if error:
        raise error[0]
    return result[0]


def iterate_sync(graph: Node, **options: Any) -> Iterator[Frame]:
    """Synchronous iterator over output frames (runs the loop in a helper thread)."""
    q: queue.Queue = queue.Queue(maxsize=64)

    async def main() -> None:
        try:
            async for f in iterate(graph, **options):
                await asyncio.get_running_loop().run_in_executor(None, q.put, f)
        except BaseException as e:
            q.put(("error", e))
        finally:
            q.put(_DONE)

    t = threading.Thread(
        target=lambda: asyncio.run(main()), daemon=True, name="sigflow-iter"
    )
    t.start()
    while True:
        it = q.get()
        if it is _DONE:
            return
        yield _unwrap(it)

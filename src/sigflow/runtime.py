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
from typing import Any, AsyncIterator, Callable, Iterable, Iterator

import numpy as np

from .block import BlockDef
from .control import Knob, live
from .frame import MISSING, Frame, Issue, Provenance, frozen, merge_annotations
from .graph import (
    Align,
    Block,
    Fanout,
    Feedback,
    Merge,
    Node,
    Pipeline,
    Select,
    Slice,
    Source,
    StreamSlice,
    When,
    Zip,
)
from .progress import Monitor, Reporter
from .types import Env, SignalType, TupleType, TypeMismatch

__all__ = ["run", "arun", "iterate", "iterate_sync", "Runner", "RunOptions"]

_DONE = object()


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
    ) -> None:
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


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


async def aiter_any(obj: Any, threaded: bool = False) -> AsyncIterator[Any]:
    """Iterate anything — sync or async iterable, generator or factory — asynchronously."""
    if callable(obj) and not hasattr(obj, "__iter__") and not hasattr(obj, "__aiter__"):
        obj = obj()
        if inspect.isawaitable(obj):
            obj = await obj
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
        if isinstance(node, Pipeline):
            stream = upstream
            for stage in node.stages:
                stream = self.compile(stage, stream)
            return stream  # type: ignore[return-value]
        if isinstance(node, Source):
            return self._source(node)
        if isinstance(node, Block):
            if node.kind == "source":
                return self._block_source(node)
            assert upstream is not None, f"{node.name} needs an upstream"
            if node.kind == "stream":
                return self._stream_block(node, upstream)
            return self._map_block(node, upstream)
        if isinstance(node, Fanout):
            assert upstream is not None
            return self._fanout(node, upstream)
        if isinstance(node, Zip):
            return self._zip(node, upstream)
        if isinstance(node, Merge):
            return self._merge(node)
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
        from .builder import Graph

        if isinstance(node, Graph):
            return self._graph(node, upstream)
        if isinstance(node, Feedback):
            assert upstream is not None
            return self._feedback(node, upstream)
        if isinstance(node, When):
            assert upstream is not None
            return self._when(node, upstream)
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

    # -- explicit graphs ---------------------------------------------------- #
    async def _graph(
        self, g: Any, upstream: AsyncIterator[Frame] | None
    ) -> AsyncIterator[Frame]:
        """Schedule a builder Graph: every member becomes a task pumping its output into
        one queue per consuming edge; multi-input members zip their input queues."""
        from .builder import _Input

        stats = self.mon.stats(g.name, g)
        queues: dict[tuple[str, int], asyncio.Queue] = {}  # (consumer, slot) -> queue
        out_qs: dict[str, asyncio.Queue] = {}  # graph outputs
        for src, dst, slot in g.edges:
            queues[(dst, slot)] = asyncio.Queue()
        for r in g.outputs:
            out_qs[r.name] = asyncio.Queue()
        tasks: list[asyncio.Task] = []

        def inputs_for(name: str) -> AsyncIterator[Frame] | None:
            node = g.members[name]
            if isinstance(node, _Input):
                return upstream
            slots = sorted(k[1] for k in queues if k[0] == name)
            if not slots:
                return None
            if len(slots) == 1 and g._n_inputs(node) == 1:
                return _drain(queues[(name, slots[0])])
            return self._zip_streams(
                [_drain(queues[(name, i)]) for i in slots], "inputs", node
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
            except BaseException as e:  # noqa: BLE001
                for q in outs:
                    await q.put(("error", e))
                raise
            finally:
                for src, dst, slot in edges:
                    await queues[(dst, slot)].put(_DONE)
                for q in outs:
                    await q.put(_DONE)

        for name in g.order():
            node = g.members[name]
            stream = (
                self.compile(node, inputs_for(name))
                if not isinstance(node, _Input)
                else inputs_for(name)
            )
            assert stream is not None
            tasks.append(asyncio.create_task(pump(name, stream)))

        try:
            if len(g.outputs) == 1:
                q = out_qs[g.outputs[0].name]
                key = g.outputs[0].key
                while True:
                    it = await q.get()
                    if it is _DONE:
                        return
                    if isinstance(it, tuple) and len(it) == 2 and it[0] == "error":
                        raise it[1]
                    f = (
                        _select_frame(it, Select(g.members[g.outputs[0].name], key))
                        if key is not None
                        else it
                    )
                    stats.out += 1
                    yield f.evolve(
                        prov=Provenance(
                            g.name, {}, (f.prov,) if f.prov else (), stats.out - 1
                        )
                    )
            else:
                async for f in self._zip_streams(
                    [_drain(out_qs[r.name]) for r in g.outputs], "outputs", g
                ):
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
    def _subpipe(self, node: Node) -> "_SubPipe":
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
                    parents = _unique(p for p in (part.prov, f.prov) if p is not None)
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
            produced = d.fn(
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
        policy = self._policy(node)
        seq = 0
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
            t0 = time.perf_counter()
            try:
                args = self._args(d, f)
                cfg = (
                    live(node.given) if node.live else node.given
                )  # knobs resolved per call
                result = d.fn(*args, **cfg)
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
        spec = d.inputs[0]
        if len(d.inputs) != 1:
            raise TypeError(f"{d.name}: stream blocks take exactly one stream input")
        policy = self._policy(node)
        consumed: list[Frame] = []
        failed_q: asyncio.Queue = asyncio.Queue()

        async def frames() -> AsyncIterator[Frame]:
            async for f in upstream:
                stats.inp += 1
                if f.failed:
                    # Failed frames always bypass stream blocks (even frame-aware ones): a
                    # generator that raised cannot be resumed, so it must never see MISSING data.
                    failed_q.put_nowait(f)
                    continue
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
                agen = d.fn(items(), **node.given)
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
                async for out, parents in _run_sync_stream(
                    d.fn,
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

    def _forward_failed(self, f: Frame, node: Block, seq: int) -> Frame:
        self.mon.stats(node.name).out += 1
        return f.evolve(
            prov=Provenance(node.name, node.config, (f.prov,) if f.prov else (), seq)
        )

    # -- fan-out / zip / merge --------------------------------------------- #
    async def _fanout(
        self, node: Fanout, upstream: AsyncIterator[Frame]
    ) -> AsyncIterator[Frame]:
        n = len(node.branches)
        in_qs: list[asyncio.Queue] = [
            asyncio.Queue(self.opts.queue_size) for _ in range(n)
        ]

        async def pump() -> None:
            try:
                async for f in upstream:
                    for q in in_qs:
                        await q.put(f)
            finally:
                for q in in_qs:
                    await q.put(_DONE)

        pump_task = asyncio.create_task(pump())
        try:
            async for f in self._zip_streams(
                [self.compile(b, _drain(q)) for b, q in zip(node.branches, in_qs)],
                "fanout",
                node,
            ):
                yield f
        finally:
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _zip(
        self, node: Zip, upstream: AsyncIterator[Frame] | None
    ) -> AsyncIterator[Frame]:
        if upstream is None:
            streams = [self.compile(p, None) for p in node.parts]
            async for f in self._zip_streams(streams, "zip", node):
                yield f
        else:
            fan = Fanout(node.parts)
            async for f in self._fanout(fan, upstream):
                yield f

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
                if isinstance(it, tuple) and len(it) == 2 and it[0] == "error":
                    raise it[1]
                stats.out += 1
                yield it.evolve(
                    prov=Provenance("merge", {}, (it.prov,) if it.prov else (), seq)
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
        else:
            ann = merge_annotations(parents, keep_samples=keep_samples)
            out = Frame(
                result, ann["meta"], ann["stream"], ann["samples"], ann["issues"], prov
            )
        # runtime contract check
        if self.opts.check == "all" or (self.opts.check == "first" and stats.out == 0):
            if out.data is not MISSING:
                try:
                    out_type.check(out.data, self.opts.env, where=f"{op} output")
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
        except BaseException as e:  # noqa: BLE001
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
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "error":
            raise item[1]
        return item

    async def close(self) -> None:
        await self.inq.put(_DONE)
        self.task.cancel()
        try:
            await self.task
        except (asyncio.CancelledError, Exception):
            pass


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


async def _drain(q: asyncio.Queue) -> AsyncIterator[Frame]:
    while True:
        it = await q.get()
        if it is _DONE:
            return
        yield it


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
        except BaseException as e:  # noqa: BLE001 - forwarded to the loop
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
    opts = RunOptions(
        **{
            k: v
            for k, v in options.items()
            if k
            in (
                "check",
                "on_error",
                "monitor",
                "queue_size",
                "control",
                "resume",
                "strict",
            )
        }
    )
    runner = Runner(opts)
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
    graph.last_monitor = mon  # type: ignore[attr-defined]
    runner = Runner(opts)
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
        except BaseException as e:  # noqa: BLE001
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
        except BaseException as e:  # noqa: BLE001
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
        if isinstance(it, tuple) and len(it) == 2 and it[0] == "error":
            raise it[1]
        yield it

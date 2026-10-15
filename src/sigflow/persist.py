"""Persisting provenance and frames, replaying a frame, checkpointing a run.

    prov.to_json() / Provenance.from_json(text)      lineage as JSON (shared nodes kept)
    frame.save("f.npz") / Frame.load("f.npz")        data + annotations + lineage
    replay(graph, frame_or_prov)                     recompute exactly that frame from the
                                                     same graph (same source frames), with
                                                     a check that the configuration matches
    run.checkpoint("ckpt.pkl") / g.start(resume="ckpt.pkl")
                                                     save source positions and the state of
                                                     Stateful blocks; resume from there

A block is *checkpointable* when its function is an object with ``state()`` and
``restore(state)`` (or a plain ``__dict__``): see :class:`Stateful`. Generator
stream blocks keep their state in Python locals and cannot be captured; a
checkpoint lists them under ``skipped``.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .frame import NO_PROV, Frame, Issue, Provenance, frozen

__all__ = [
    "prov_to_dict",
    "prov_from_dict",
    "save_frame",
    "load_frame",
    "replay",
    "ReplayMismatch",
    "Stateful",
    "Checkpoint",
    "checkpoint",
    "load_checkpoint",
]


# --------------------------------------------------------------------------- #
# Provenance <-> JSON
# --------------------------------------------------------------------------- #


def _jsonable(v: Any) -> Any:
    if hasattr(v, "_set_from_widget"):  # Knob
        return _jsonable(v.value)
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, np.ndarray):
        return (
            {"__ndarray__": v.tolist(), "dtype": str(v.dtype)}
            if v.size <= 64
            else {"__ndarray__": None, "shape": v.shape, "dtype": str(v.dtype)}
        )
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def prov_to_dict(p: Provenance) -> dict:
    """The lineage DAG as ``{"root": id, "nodes": [{id, op, params, seq, parents}]}``."""
    nodes = []
    for n in p.walk():
        nodes.append(
            {
                "id": n.id,
                "op": n.op,
                "params": _jsonable(dict(n.params)),
                "seq": n.seq,
                "parents": [q.id for q in n.parents],
            }
        )
    return {"root": p.id, "nodes": nodes}


def prov_from_dict(d: dict) -> Provenance:
    by_id: dict[int, Provenance] = {}
    pending = {n["id"]: n for n in d["nodes"]}

    def build(i: int) -> Provenance:
        if i in by_id:
            return by_id[i]
        n = pending[i]
        node = Provenance(
            n["op"],
            frozen(n["params"]),
            tuple(build(q) for q in n["parents"]),
            n["seq"],
            id=i,
        )
        by_id[i] = node
        return node

    return build(d["root"])


# --------------------------------------------------------------------------- #
# Frames on disk
# --------------------------------------------------------------------------- #


def save_frame(f: Frame, path: str) -> str:
    """``.npz`` with the data, sample annotations and a JSON record of everything else."""
    arrays: dict[str, Any] = {}
    data = f.data
    if isinstance(data, tuple):
        for i, x in enumerate(data):
            arrays[f"data_{i}"] = np.asarray(x)
        kind = "tuple"
    elif isinstance(data, (np.ndarray, int, float, complex, np.generic)):
        arrays["data"] = np.asarray(data)
        kind = "array"
    else:
        kind = "object"
    for k, v in f.samples.items():
        arrays[f"sample__{k}"] = np.asarray(v)
    record = {
        "kind": kind,
        "meta": _jsonable(dict(f.meta)),
        "stream": _jsonable(dict(f.stream)),
        "issues": [
            {"message": i.message, "level": i.level, "origin": i.origin}
            for i in f.issues
        ],
        "prov": prov_to_dict(f.prov) if f.prov else None,
        "object": _jsonable(data) if kind == "object" else None,
    }
    arrays["__record__"] = np.array(json.dumps(record))
    np.savez(path, **arrays)
    return path


def load_frame(path: str) -> Frame:
    z = np.load(path, allow_pickle=False)
    record = json.loads(str(z["__record__"]))
    if record["kind"] == "tuple":
        data: Any = tuple(
            z[f"data_{i}"]
            for i in range(sum(1 for k in z.files if k.startswith("data_")))
        )
    elif record["kind"] == "array":
        data = z["data"]
        if data.ndim == 0:
            data = data.item()
    else:
        data = record["object"]
    samples = {k[len("sample__") :]: z[k] for k in z.files if k.startswith("sample__")}
    issues = tuple(
        Issue(i["message"], i["level"], i["origin"]) for i in record["issues"]
    )
    prov = prov_from_dict(record["prov"]) if record["prov"] else NO_PROV
    return Frame(data, record["meta"], record["stream"], samples, issues, prov)


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


class ReplayMismatch(RuntimeError):
    pass


def replay(graph: Any, target: Any, strict: bool = True, **options: Any) -> Frame:
    """Recompute the output frame whose lineage ends in the same source frames as
    ``target`` (a Frame or Provenance). Deterministic graphs give the same data back;
    ``strict`` checks that the graph's source configuration equals the recorded one."""
    prov = target.prov if isinstance(target, Frame) else target
    if not prov:
        raise ReplayMismatch("target has no provenance")
    wanted = {(r.op, r.seq) for r in prov.roots()}
    recorded = {r.op: dict(r.params) for r in prov.roots()}
    if strict:
        from .graph import Block, Source

        for n in graph.nodes():
            if n.in_type is None and n.name in recorded:
                have = _jsonable(dict(getattr(n, "config", {})))
                want = _jsonable(recorded[n.name])
                if have != want:
                    raise ReplayMismatch(
                        f"source {n.name} is configured {have}, the recorded frame came from {want}"
                    )
    from .runtime import iterate_sync

    max_seq = max((s for _, s in wanted if s is not None), default=0)
    for f in iterate_sync(graph, **options):
        roots = {(r.op, r.seq) for r in f.prov.roots()} if f.prov else set()
        if roots == wanted:
            return f
        if all(s is not None and s > max_seq for _, s in roots):
            break
    raise ReplayMismatch(f"no output frame with sources {sorted(wanted)} was produced")


# --------------------------------------------------------------------------- #
# Stateful blocks and checkpoints
# --------------------------------------------------------------------------- #


class Stateful:
    """Base for checkpointable blocks: a callable object used as a block::

        class Smoother(Stateful):
            def __init__(self, alpha=0.1): self.alpha, self.acc = alpha, None
            def __call__(self, x: Sig[float, N]) -> Sig[float, N]:
                self.acc = x if self.acc is None else (1 - self.alpha) * self.acc + self.alpha * x
                return self.acc

        smooth = block(Smoother(0.2))

    ``state()`` / ``restore()`` default to the instance ``__dict__`` (override for
    non-picklable members)."""

    def state(self) -> Any:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def restore(self, state: Any) -> None:
        self.__dict__.update(state)


@dataclass
class Checkpoint:
    positions: dict[str, int] = field(
        default_factory=dict
    )  # source name -> frames emitted
    states: dict[str, Any] = field(default_factory=dict)  # block name -> state
    skipped: list[str] = field(
        default_factory=list
    )  # stateful-looking blocks not captured
    produced: int = 0
    created: float = field(default_factory=time.time)
    graph: str = ""

    def save(self, path: str) -> str:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path


def load_checkpoint(path: str) -> Checkpoint:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def checkpoint(graph: Any, monitor: Any, path: str | None = None) -> Checkpoint:
    """Capture source positions (from the monitor) and the state of Stateful blocks."""
    from .graph import Block, Source

    ck = Checkpoint(graph=graph.name, produced=monitor.produced)
    for n in graph.nodes():
        key = n.name
        st = monitor.stats(key)
        if n.in_type is None:
            ck.positions[key] = st.out
        if isinstance(n, Block):
            fn = n.defn.fn
            state, restore_ = getattr(fn, "state", None), getattr(fn, "restore", None)
            if callable(state) and callable(restore_):
                ck.states[key] = state()
            elif n.kind == "stream":
                ck.skipped.append(key)
    if path:
        ck.save(path)
    return ck


def restore(graph: Any, ck: Checkpoint) -> None:
    from .graph import Block

    for n in graph.nodes():
        if isinstance(n, Block) and n.name in ck.states:
            n.defn.fn.restore(ck.states[n.name])  # type: ignore[attr-defined]

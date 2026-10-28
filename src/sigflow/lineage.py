"""Querying provenance across many frames: :class:`Lineage`.

L = Lineage(frames)                      # index a run's outputs (or any frames)
L.failed()                               # frames that failed
L.through("fft")                         # frames whose lineage includes a block called fft
L.through("scale", gain=2.0)             # ... with that configuration
L.sources_of(frames)                     # the (source, seq) pairs behind these frames
L.descendants_of("sensor", seq=8)        # outputs that descend from source frame #8
L.blame()                                # failures grouped by (origin block, source frame)
L.diff(f1, f2)                           # configuration differences along two lineages
L.table()                                # per-block counts over the indexed frames
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from .frame import Frame, Provenance

__all__ = ["Lineage"]


class Lineage:
    def __init__(self, frames: Iterable[Frame]) -> None:
        self.frames = list(frames)
        self._by_op: dict[str, list[int]] = defaultdict(list)  # op -> frame indices
        self._roots: dict[tuple[str, Any], list[int]] = defaultdict(list)
        for i, f in enumerate(self.frames):
            seen = set()
            for n in f.lineage():
                if n.op not in seen:
                    self._by_op[n.op].append(i)
                    seen.add(n.op)
            for r in f.sources():
                self._roots[(r.op, r.seq)].append(i)

    def __len__(self) -> int:
        return len(self.frames)

    # -- selections --------------------------------------------------------- #
    def failed(self) -> list[Frame]:
        return [f for f in self.frames if f.failed]

    def ok(self) -> list[Frame]:
        return [f for f in self.frames if f.ok]

    def with_warnings(self) -> list[Frame]:
        return [f for f in self.frames if f.warnings]

    def through(self, op: str, **params: Any) -> list[Frame]:
        """Frames whose lineage passed through block ``op`` (optionally with these params)."""
        out = []
        for i in self._by_op.get(op, []):
            f = self.frames[i]
            if not params:
                out.append(f)
                continue
            for n in f.lineage():
                if n.op == op and all(n.params.get(k) == v for k, v in params.items()):
                    out.append(f)
                    break
        return out

    def descendants_of(self, source: str, seq: int) -> list[Frame]:
        return [self.frames[i] for i in self._roots.get((source, seq), [])]

    def sources_of(self, frames: Iterable[Frame]) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        for f in frames:
            for r in f.sources():
                if (r.op, r.seq) not in out:
                    out.append((r.op, r.seq))
        return sorted(out, key=lambda t: (t[0], -1 if t[1] is None else t[1]))

    # -- summaries ---------------------------------------------------------- #
    def blame(self) -> list[tuple[tuple[str, str, Any], int]]:
        """Failures grouped by (origin block, source block, source seq), most common first."""
        c: Counter = Counter()
        for f in self.failed():
            for issue in f.errors:
                root = issue.prov[-1] if issue.prov else None
                c[
                    (issue.origin, root.op if root else "?", root.seq if root else None)
                ] += 1
        return c.most_common()

    def ops(self) -> list[str]:
        return sorted(self._by_op, key=lambda o: -len(self._by_op[o]))

    def table(self) -> str:
        rows = [f"{'block':24} {'frames':>7} {'failed':>7} {'warn':>6}"]
        for op in self.ops():
            idx = self._by_op[op]
            fr = [self.frames[i] for i in idx]
            rows.append(
                f"{op[:24]:24} {len(fr):>7} {sum(f.failed for f in fr):>7} {sum(bool(f.warnings) for f in fr):>6}"
            )
        return "\n".join(rows)

    def configs(self, op: str) -> list[dict]:
        """Distinct configurations block ``op`` ran with across the indexed frames."""
        seen: list[dict] = []
        for i in self._by_op.get(op, []):
            for n in self.frames[i].lineage():
                if n.op == op:
                    d = dict(n.params)
                    if d not in seen:
                        seen.append(d)
        return seen

    @staticmethod
    def diff(
        a: Frame | Provenance, b: Frame | Provenance
    ) -> dict[str, tuple[Any, Any]]:
        """Configuration differences between two lineages: ``{"block.param": (a, b)}`` plus
        blocks present on only one side as ``{"block": ("present", "absent")}``."""
        pa = a.prov if isinstance(a, Frame) else a
        pb = b.prov if isinstance(b, Frame) else b
        ca: dict[str, dict] = {}
        cb: dict[str, dict] = {}
        for n in pa.walk() if pa else []:
            ca.setdefault(n.op, dict(n.params))  # nearest occurrence wins
        for n in pb.walk() if pb else []:
            cb.setdefault(n.op, dict(n.params))
        out: dict[str, tuple[Any, Any]] = {}
        for op in sorted(set(ca) | set(cb)):
            if op not in ca:
                out[op] = ("absent", "present")
            elif op not in cb:
                out[op] = ("present", "absent")
            else:
                for k in sorted(set(ca[op]) | set(cb[op])):
                    if ca[op].get(k) != cb[op].get(k):
                        out[f"{op}.{k}"] = (ca[op].get(k), cb[op].get(k))
        return out

    def __repr__(self) -> str:
        return f"<Lineage of {len(self.frames)} frames, {len(self._by_op)} blocks, {len(self._roots)} source frames>"

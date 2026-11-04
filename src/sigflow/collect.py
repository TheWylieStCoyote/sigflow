"""``gather``: fold the frames that belong together back into one tuple frame.

The mirror of a detector that yields one frame per detection and of
:func:`~sigflow.graph.per_key`: after per-signal processing, ``gather("seq",
count="n_detections")`` emits one frame per spectrum frame holding every signal's
result as a part. It is an ordinary stream block; a custom collector written the
same way (a generator over ``Stream[Framed[...]]`` yielding frames with ``parts``)
fits a graph just as well.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from .block import Framed, block
from .frame import Frame, Tags, merge_annotations
from .types import Sig, SignalType, Stream

__all__ = ["gather"]


def _checked(
    by: Any = "seq",
    count: Any = None,
    lag: int | None = None,
    ttl: int | None = None,
    incomplete: str = "emit",
    order: Any = None,
    **_: Any,
) -> SignalType:
    """Validate the configuration when the block is placed, not when it first runs."""
    if count is None and lag is None and ttl is None:
        raise ValueError("gather() needs a completion rule: count=, lag= or ttl=")
    if incomplete not in ("emit", "drop"):
        raise ValueError("incomplete must be 'emit' or 'drop'")
    _getter(by)
    _expected(count)
    if order is not None:
        _getter(order)
    return SignalType()  # a tuple of unknown length


@block(ratio="N:1", refine=_checked)
def gather(
    frames: Stream[Framed[Sig[Any]]],
    by: Any = "seq",
    count: Any = None,
    lag: int | None = None,
    ttl: int | None = None,
    incomplete: str = "emit",
    order: Any = None,
    ordered: bool = False,
) -> Iterator[Sig[Any]]:
    """Collect frames that share a key into one tuple frame per group.

    ``by`` is an annotation name or a callable on the frame giving the group key.
    A group is released when it is complete:

    - ``count``: an annotation name, a number, or a callable on a frame giving how
      many members the group has (``count="n_detections"``);
    - ``lag``: when ``lag`` newer groups have started since it (``lag=1``: the next
      group's first frame closes it);
    - ``ttl``: as a backstop, after that many input frames without a new member;
      and at the end of the stream. A group released before it was complete is
      emitted with a warning issue, or dropped when ``incomplete="drop"``.

    The output's ``data`` is the tuple of the members' payloads and the members are
    its ``parts`` (``frame[i]`` is member ``i`` with its own annotations, tags and
    provenance); annotations and tags are merged as at a join, and ``meta`` gains
    the group key under ``"group"``. Members are in arrival order, or sorted by
    ``order`` (an annotation name or a callable), in which case ``meta["members"]``
    lists their ``order`` values. Groups are emitted as they complete; ``ordered=True``
    holds them until every group started earlier has gone out. A failed frame never
    reaches a stream block (the runtime forwards it past them), so a group missing
    a failed member is released by ``ttl`` or at the end of the stream, incomplete,
    with a warning; the failed frame itself travels on as its own output.
    """
    _checked(by, count, lag, ttl, incomplete, order)  # also when called directly
    key_of = _getter(by)
    order_of = _getter(order) if order is not None else None
    expected = _expected(count)

    groups: dict[Any, list[Frame]] = {}
    seen_at: dict[Any, int] = {}
    started: list[Any] = []  # group keys in the order their first member arrived
    done: dict[Any, Frame | None] = {}  # released groups waiting (ordered=True)

    def release(key: Any, partial: bool) -> Frame | None:
        members = groups.pop(key)
        seen_at.pop(key, None)
        if partial and incomplete == "drop":
            return None
        if order_of is not None:
            members.sort(key=order_of)
        ann = merge_annotations(members, keep_samples=False)
        meta = dict(ann["meta"])
        meta["group"] = key
        if order_of is not None:
            meta["members"] = [order_of(m) for m in members]
        out = Frame(
            tuple(m.data for m in members),
            meta,
            ann["stream"],
            {},
            ann["issues"],
            parts=tuple(members),
            tags=Tags(ann["tags"]),
        )
        if partial:
            want = expected(members[-1]) if count is not None else None
            have = (
                f"{len(members)} of {want}" if want is not None else str(len(members))
            )
            out = out.with_issue(
                f"group {key!r} released with {have} member(s)", "warning", "gather"
            )
        return out

    def emit(key: Any, partial: bool) -> Iterator[Frame]:
        """Release one group, honouring ``ordered``."""
        out = release(key, partial)
        if not ordered:
            if key in started:
                started.remove(key)
            if out is not None:
                yield out
            return
        done[key] = out
        while started and started[0] in done:
            head = started.pop(0)
            ready = done.pop(head)
            if ready is not None:
                yield ready

    for n, f in enumerate(frames):
        key = key_of(f)
        if key not in groups:
            groups[key] = []
            started.append(key)
        groups[key].append(f)
        seen_at[key] = n
        if count is not None and len(groups[key]) >= expected(f):
            yield from emit(key, partial=False)
        if lag is not None:
            open_keys = [k for k in started if k in groups]
            for older in open_keys[: max(0, len(open_keys) - lag)]:
                if older != key:
                    yield from emit(older, partial=count is not None)
        if ttl is not None:
            for stale in [k for k, at in seen_at.items() if n - at >= ttl]:
                yield from emit(stale, partial=count is not None)
    remaining = [k for k in started if k in groups]  # emit() edits ``started``
    for key in remaining:  # end of stream: nothing more is coming
        yield from emit(key, partial=count is not None)


def _getter(spec: Any) -> Callable[[Frame], Any]:
    if isinstance(spec, str):
        name = spec

        def by_name(f: Frame) -> Any:
            return f.get(name)

        return by_name
    if callable(spec):
        return spec
    raise TypeError(f"expected an annotation name or a callable, got {spec!r}")


def _expected(count: Any) -> Callable[[Frame], int]:
    if count is None:
        return lambda f: 0
    if isinstance(count, bool):
        raise TypeError("count must be a number, an annotation name or a callable")
    if isinstance(count, int):
        fixed = count

        def constant(f: Frame) -> int:
            return fixed

        return constant
    if isinstance(count, str):

        def by_name(f: Frame, name: str = count) -> int:
            try:
                return int(f[name])
            except KeyError:
                raise KeyError(
                    f"gather(count={name!r}): frame carries no {name!r} annotation"
                ) from None

        return by_name
    if callable(count):
        fn: Callable[[Frame], Any] = count
        return lambda f: int(fn(f))
    raise TypeError("count must be a number, an annotation name or a callable")

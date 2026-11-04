"""gather: fold the frames that belong together back into one tuple frame."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from sigflow import (
    Frame,
    Lineage,
    Sig,
    block,
    from_iterable,
    gather,
    per_key,
    run,
    source,
)
from sigflow.dims import N


def member(group: int, i: int, n: int, **meta) -> Frame:
    return Frame(
        np.full(2, float(10 * group + i)), meta={"seq": group, "n": n, "i": i, **meta}
    )


def groups(*sizes: int, interleave: bool = False) -> list[Frame]:
    members = [[member(g, i, n) for i in range(n)] for g, n in enumerate(sizes)]
    if not interleave:
        return [m for ms in members for m in ms]
    out: list[Frame] = []
    while any(members):
        for ms in members:
            if ms:
                out.append(ms.pop(0))
    return out


def test_count_from_an_annotation_completes_groups_even_when_interleaved():
    frames = groups(2, 3, 1, interleave=True)
    out = run(from_iterable(frames) >> gather("seq", count="n"), on_error="raise")
    assert [f["group"] for f in out] == [2, 0, 1]  # as they complete
    assert [len(f.parts) for f in out] == [1, 2, 3]
    g1 = out[2]
    assert g1.data == tuple(p.data for p in g1.parts) and g1[1]["i"] == 1
    assert g1.parts[0]["i"] == 0 and not g1.issues  # complete: no warning
    assert all(f.prov.op == "gather" for f in out)


def test_fixed_count_order_and_ordered_release():
    frames = groups(2, 2, 2, interleave=True)
    out = run(
        from_iterable(frames) >> gather("seq", count=2, order="i"), on_error="raise"
    )
    assert [f["members"] for f in out] == [[0, 1]] * 3
    # a slow group 0 (its second member arrives last) still goes out first with ordered=True
    late = [
        member(1, 0, 2),
        member(1, 1, 2),
        member(0, 0, 2),
        member(2, 0, 2),
        member(2, 1, 2),
        member(0, 1, 2),
    ]
    unordered = run(from_iterable(late) >> gather("seq", count="n"), on_error="raise")
    assert [f["group"] for f in unordered] == [1, 2, 0]
    ordered = run(
        from_iterable(late) >> gather("seq", count="n", ordered=True), on_error="raise"
    )
    assert [f["group"] for f in ordered] == [1, 0, 2]  # the order the groups started


def test_lag_releases_a_group_when_a_newer_one_starts():
    frames = groups(2, 3, 1)  # no count annotation needed
    out = run(from_iterable(frames) >> gather("seq", lag=1), on_error="raise")
    assert [(f["group"], len(f.parts)) for f in out] == [(0, 2), (1, 3), (2, 1)]
    assert not any(f.issues for f in out)  # lag is a rule, not an incompleteness
    out = run(from_iterable(frames) >> gather("seq", lag=2), on_error="raise")
    assert [f["group"] for f in out] == [0, 1, 2]  # group 0 waits for two newer groups


def test_ttl_and_end_of_stream_release_partial_groups_with_a_warning():
    frames = (
        [member(0, 0, 3), member(0, 1, 3)]
        + [member(1, i, 1) for i in range(4)]
        + [member(2, 0, 2)]
    )
    out = run(
        from_iterable(frames) >> gather("seq", count="n", ttl=3), on_error="raise"
    )
    assert [f["group"] for f in out] == [
        1,
        1,
        1,
        0,
        1,
        2,
    ]  # 0 after 3 idle frames; 2 at the end
    partial = out[3]
    assert len(partial.parts) == 2 and partial.warnings
    assert "group 0 released with 2 of 3" in partial.warnings[0].message
    assert out[-1].warnings and "1 of 2" in out[-1].warnings[0].message
    dropped = run(
        from_iterable(frames) >> gather("seq", count="n", ttl=3, incomplete="drop"),
        on_error="raise",
    )
    assert [f["group"] for f in dropped] == [1, 1, 1, 1]


def test_failed_members_are_kept_and_counted():
    @block
    def boom(x: Sig[float, N]) -> Sig[float, N]:
        if float(x[0]) == 1:
            raise ValueError("boom")
        return x

    out = run(from_iterable(groups(2)) >> boom >> gather("seq", count="n"))
    # the failed frame bypasses the stream block and travels on by itself; the group
    # it belonged to is released at the end, incomplete, with a warning
    failed = [f for f in out if f.failed]
    assert len(failed) == 1 and failed[0].errors[0].origin == "boom"
    group = next(f for f in out if not f.failed)
    assert len(group.parts) == 1 and "1 of 2" in group.warnings[0].message


def test_rules_are_required_and_count_annotation_must_exist():
    with pytest.raises(ValueError, match="completion rule"):
        _ = from_iterable([]) >> gather("seq")
    with pytest.raises(ValueError, match="incomplete"):
        _ = from_iterable([]) >> gather("seq", count=1, incomplete="maybe")
    with pytest.raises(KeyError, match="no 'missing' annotation"):
        run(
            from_iterable(groups(1)) >> gather("seq", count="missing"), on_error="raise"
        )


def test_after_per_key_the_group_has_every_signals_result():
    @source
    def detections(frames: int = 6) -> Iterator[Sig[float, N]]:
        for seq in range(frames):
            tracks = [0, 1, 2][: 1 + seq % 3]  # a varying number of signals per frame
            for t in tracks:
                yield Frame(
                    np.full(2, float(seq)),
                    meta={"seq": seq, "track": t, "n": len(tracks)},
                )

    @block
    def count_seen(x: Sig[float, N]) -> Sig[float, N]:
        return x

    graph = (
        detections()
        >> per_key("track", count_seen)
        >> gather("seq", count="n", order="track")
    )
    out = run(graph, on_error="raise")
    assert [f["group"] for f in sorted(out, key=lambda f: f["group"])] == list(range(6))
    assert [f["members"] for f in sorted(out, key=lambda f: f["group"])] == [
        [0],
        [0, 1],
        [0, 1, 2],
    ] * 2
    assert str(graph.ratio) == "any"
    lineage = Lineage(out)
    assert all(op == "detections" for f in out for op, _ in lineage.sources_of([f]))
    first = sorted(out, key=lambda f: f["group"])[1]
    assert first[1].prov.op == "per_key(track)[1]"  # each part keeps its own lineage

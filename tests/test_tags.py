"""Tags: labels on a frame or a span of it, carried by the runtime and the contract."""

from __future__ import annotations

import enum
import pickle
from typing import Annotated, Any

import numpy as np
import pytest

from sigflow import (
    Frame,
    Framed,
    Graph,
    Sig,
    SignalType,
    TypeMismatch,
    block,
    clears,
    dsp,
    from_iterable,
    provides,
    requires,
    run,
    when,
)
from sigflow.dims import N
from sigflow.frame import Tag, Tags, merge_annotations

# ------------------------------------------------------------------ the record


class Kind(enum.Enum):
    VOICE = "voice"


def test_tag_identity_is_name_and_span():
    assert Tag("speech", 0.9) == Tag("speech", 0.3, origin="other")
    assert Tag("speech") != Tag("speech", span=(0.0, 1.0))
    assert len({Tag("speech", 0.9), Tag("speech", 0.3)}) == 1
    assert repr(Tag("seg", 3, (0.5, 1.0))) == "seg=3@[0.5, 1)"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (True, True),
        (3, 3),
        (0.5, 0.5),
        ("x", "x"),
        (np.float32(0.5), 0.5),
        (Kind.VOICE, "voice"),
        ([1, "a", (2.0, None)], (1, "a", (2.0, None))),
    ],
)
def test_tag_values_are_scalars_or_tuples(value, expected):
    assert Tag("t", value).value == expected
    assert type(Tag("t", value).value) is type(expected)


@pytest.mark.parametrize("value", [np.zeros(3), {"a": 1}, object()])
def test_tag_rejects_arrays_and_objects(value):
    with pytest.raises(TypeError, match="scalars or tuples"):
        Tag("t", value)
    with pytest.raises(TypeError, match="non-empty string"):
        Tag("")


def test_tags_lookup_merge_and_pickle():
    tags = Tags([Tag("speech"), Tag("seg", 1, (0.0, 1.0)), "raw"])
    assert "speech" in tags and "raw" in tags and "other" not in tags
    assert tags["seg"] == 1 and tags.get("other", 7) == 7
    with pytest.raises(KeyError, match="no tag"):
        tags["other"]
    assert tags.names == {"speech", "seg", "raw"}
    assert tags.at(0.5).names == {"speech", "raw", "seg"}
    assert tags.at(2.0).names == {"speech", "raw"}
    assert tags.whole().names == {"speech", "raw"}
    merged = Tags.merge([Tag("x", 1), Tag("y")], [Tag("x", 2)])
    assert merged["x"] == 2 and merged.names == {"x", "y"}
    assert pickle.loads(pickle.dumps(tags)) == tags
    assert isinstance(pickle.loads(pickle.dumps(tags)), Tags)


# ------------------------------------------------------------------ the frame


def timed(n: int = 8, rate: float = 4.0, t0: float = 1.0) -> Frame:
    return Frame(np.arange(float(n)), meta={"t0": t0}, stream={"rate": rate})


def test_frame_tag_untag_and_lookup():
    f = timed().tag("speech", "raw").tag(segment=3)
    assert "speech" in f and f["segment"] == 3 and f.tags["segment"] == 3
    assert f.get("segment") == 3 and f.get("missing", 0) == 0
    assert f.tag(segment=4)["segment"] == 4  # the later wins
    assert f.untag("raw", "speech").tags.names == {"segment"}
    assert f.annotate(segment=9)["segment"] == 9  # meta is looked up first
    assert "tags=" in repr(f)


def test_spans_are_clipped_to_the_frame_window():
    f = timed().tag("burst", span=(1.5, 2.25)).tag("late", span=(5.0, 6.0))
    assert f.tags.names == {"burst"}  # outside the window: gone
    assert f.mask("burst").tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert f.tag("all").mask("all").all()
    part = f[2:6]  # t0 = 1.5, window [1.5, 2.5)
    assert part.tags == Tags([Tag("burst", span=(1.5, 2.25))])
    assert part.mask("burst").tolist() == [True, True, True, False]
    assert f[6:].tags == Tags()
    assert f.between(2.0, 3.0).tags == Tags([Tag("burst", span=(2.0, 2.25))])
    assert Frame(np.zeros(4)).tag("t", span=(0.0, 1.0)).tags  # no time axis: kept
    with pytest.raises(KeyError, match="no time axis"):
        Frame(np.zeros(4)).tag("t", span=(0.0, 1.0)).mask("t")


def test_merge_unites_tags_and_drops_spans_off_axis():
    a = timed().tag("a", "shared", span=None).tag(seg=1).tag("part", span=(1.0, 1.5))
    b = timed().tag("b").tag(seg=2)
    kept = merge_annotations([a, b], keep_samples=True)["tags"]
    assert kept.names == {"a", "b", "shared", "seg", "part"} and kept["seg"] == 2
    dropped = merge_annotations([a, b], keep_samples=False)["tags"]
    assert dropped.names == {"a", "b", "shared", "seg"}


# ------------------------------------------------------------------ the runtime


@block
def vad(f: Framed[Sig[float, N]]) -> Annotated[Sig[float, N], provides("speech")]:
    return f.tag("speech", level=float(np.abs(f.data).mean()))


@block
def asr(
    x: Annotated[Sig[float, N], requires("speech")],
) -> Annotated[Sig[float, N], clears("speech")]:
    return x * 0 + 1


def test_runtime_carries_tags_stamps_origin_and_clears():
    graph = dsp.sine(n=8, frames=2) >> vad >> dsp.scale(gain=2.0) >> asr
    assert [str(s.out_type) for s in graph.stages][1:] == [
        "Sig[float64, (N,), tags=speech]",
        "Sig[T, (N,), tags=speech]",
        "Sig[float64, (N,), clears=speech]",
    ]
    mid = run(dsp.sine(n=8, frames=1) >> vad >> dsp.scale(gain=2.0), check="all")[0]
    assert mid.tags.names == {"speech", "level"}
    assert {t.origin for t in mid.tags} == {"vad"}
    out = run(graph, check="all", on_error="raise")
    assert [f.tags.names for f in out] == [{"level"}, {"level"}]


def test_requires_is_checked_when_the_graph_is_built():
    with pytest.raises(TypeMismatch, match="needs tag.*speech.*guarantees none"):
        _ = dsp.sine(n=8) >> asr
    with pytest.raises(TypeMismatch, match="needs tag"):
        _ = dsp.sine(n=8) >> vad >> asr >> asr  # cleared on the way
    assert (dsp.sine(n=8) >> dsp.scale(gain=2.0)).out_type.meta == frozenset()
    declared = from_iterable(
        [np.zeros(8)], type=SignalType(float, (N,)).provides("speech")
    )
    assert (declared >> dsp.scale(gain=1.0) >> asr).out_type.cleared == {"speech"}
    spelled = from_iterable(
        [np.zeros(8)], type=Annotated[Sig[float, N], provides("speech")]
    )
    assert spelled.out_type.meta == {"speech"}


def test_provides_is_checked_at_run_time():
    @block
    def liar(x: Sig[float, N]) -> Annotated[Sig[float, N], provides("speech")]:
        return x

    with pytest.raises(TypeMismatch, match="guarantees tag.*speech.*carries none"):
        run(dsp.sine(n=8, frames=1) >> liar, check="all", on_error="raise")
    declared = from_iterable(
        [np.zeros(8)], type=SignalType(float, (N,)).provides("speech")
    )
    with pytest.raises(TypeMismatch, match="guarantees tag"):
        run(declared >> dsp.scale(gain=1.0), check="all", on_error="raise")


def test_tags_flow_through_fanout_zip_select_when_and_builder():
    forked = dsp.sine(n=8, frames=1) >> vad >> (dsp.scale(gain=1.0) | dsp.rms)
    assert all(t.meta == {"speech"} for t in forked.out_type.items)
    assert (forked[0] >> asr).out_type.cleared == {"speech"}
    out = run(forked, check="all", on_error="raise")[0]
    assert out.tags.names == {"speech", "level"} and out[0].tags.names == out.tags.names
    routed = dsp.sine(n=8, frames=3) >> vad >> when(lambda f: "speech" in f, then=asr)
    assert len(run(routed, check="all", on_error="raise")) == 3
    graph = Graph()
    src = graph.add(dsp.sine(n=8, frames=1))
    tagger = graph.add(vad)
    sink = graph.add(asr)
    graph.connect(tagger, sink).connect(src, tagger).output(sink)
    assert graph.out_type.cleared == {"speech"}
    assert run(graph, check="all", on_error="raise")[0].tags.names == {"level"}


def test_when_on_a_tag_establishes_it_for_the_branch():
    @block
    def maybe(f: Framed[Sig[float, N]]) -> Sig[float, N]:
        return f.tag("speech") if f["seq"] % 2 else f

    src = dsp.sine(n=8, frames=4) >> maybe
    with pytest.raises(TypeMismatch, match="needs tag"):
        _ = src >> when(lambda f: "speech" in f, then=asr)  # the checker cannot know
    routed = src >> when("speech", then=asr)
    assert routed.stages[-1].in_type.meta == frozenset()
    assert routed.stages[-1].then.in_type.meta == {"speech"}
    out = run(routed, check="all", on_error="raise")
    branches = [f.prov.op.rsplit("[", 1)[1] for f in out]
    assert branches == ["else]", "then]", "else]", "then]"]
    assert ["asr" in f.prov for f in out] == [False, True, False, True]
    assert [("speech" in f) for f in out] == [False, False, False, False]
    with pytest.raises(TypeMismatch, match="needs tag"):
        _ = src >> when("speech", then=dsp.scale(gain=1.0), otherwise=asr)


def test_rank_changing_blocks_keep_whole_frame_tags_only():
    src = from_iterable(
        [timed(16, 8.0, 0.0).tag("calibrated").tag("burst", span=(0.5, 1.5))]
    )
    chunks = run(src >> dsp.chunk(4), check="all", on_error="raise")
    assert [sorted(map(repr, f.tags)) for f in chunks] == [
        ["calibrated"],
        ["burst@[0.5, 1)", "calibrated"],
        ["burst@[1, 1.5)", "calibrated"],
        ["calibrated"],
    ]
    spectrum = run(src >> dsp.rms, check="all", on_error="raise")[0]
    assert spectrum.tags.names == {"calibrated"}


def test_tags_round_trip_through_frame_files(tmp_path):
    f = timed().tag("speech", seg=(1, "a")).tag("burst", span=(1.5, 2.0))
    f = Frame(
        f.data, f.meta, f.stream, tags=Tags([Tag("speech", None, None, "vad"), *f.tags])
    )
    path = f.save(str(tmp_path / "frame.npz"))
    back = Frame.load(path)
    assert back.tags == f.tags
    assert {(t.name, t.value, t.span, t.origin) for t in back.tags} == {
        (t.name, t.value, t.span, t.origin) for t in f.tags
    }


# ------------------------------------------------------------ at the boundaries


def test_tuple_into_any_input_still_needs_the_tag():
    @block
    def needs(
        x: Annotated[Sig[Any], requires("foo")],
    ) -> Annotated[Sig[Any], clears("foo")]:
        return 1

    forked = dsp.sine(n=4, frames=1) >> (dsp.scale(gain=1.0) | dsp.scale(gain=2.0))
    with pytest.raises(TypeMismatch, match="needs tag.*foo"):
        _ = forked >> needs
    # declared but not delivered: caught at the producer's output, before the body
    lying = from_iterable(
        [Frame(np.zeros(4)).tag("foo"), Frame(np.zeros(4))],
        type=SignalType(float, (4,)).provides("foo"),
    )
    with pytest.raises(TypeMismatch, match="list output: contract guarantees tag.*foo"):
        run(lying >> needs, check="all", on_error="raise")


def test_masks_follow_interval_membership():
    f = Frame(np.zeros(4), meta={"t0": 0.0}, stream={"rate": 4.0})
    assert f.tag("a", span=(0.0, 1.0)).mask("a").tolist() == [True] * 4
    assert f.tag("b", span=(0.1, 0.6)).mask("b").tolist() == [False, True, True, False]
    assert f.tag("c", span=(0.25, 0.5)).mask("c").tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert f.tag("d", span=(0.74, 0.76)).mask("d").tolist() == [
        False,
        False,
        False,
        True,
    ]
    assert f.tag("p", span=(0.6, 0.6)).mask("p").tolist() == [False, False, True, False]


def test_reframing_carries_tags_across_input_frames():
    a = Frame(np.arange(4.0), meta={"t0": 0.0}, stream={"rate": 4.0})
    a = a.tag("whole").tag("s", span=(0.25, 0.75))
    b = Frame(np.arange(4.0), meta={"t0": 1.0}, stream={"rate": 4.0}).tag("other")
    c = Frame(np.arange(4.0), meta={"t0": 2.0}, stream={"rate": 4.0})
    out = run(from_iterable([a, b, c]) >> dsp.chunk(6), check="all", on_error="raise")
    assert [sorted(map(repr, f.tags)) for f in out] == [
        ["other@[1, 1.5)", "s@[0.25, 0.75)", "whole@[0, 1)"],
        ["other@[1.5, 2)"],
    ]
    halves = run(from_iterable([a]) >> dsp.chunk(2), check="all", on_error="raise")
    assert [sorted(map(repr, f.tags)) for f in halves] == [
        ["s@[0.25, 0.5)", "whole"],
        ["s@[0.5, 0.75)", "whole"],
    ]
    windows = run(
        from_iterable([a, b]) >> dsp.overlap(n=4, hop=2), check="all", on_error="raise"
    )
    assert [sorted(map(repr, f.tags)) for f in windows] == [
        ["s@[0.25, 0.75)", "whole"],
        ["other@[1, 1.5)", "s@[0.5, 0.75)", "whole@[0.5, 1)"],
        ["other"],
    ]
    stacked = run(
        from_iterable([a, b]) >> dsp.buffer(count=2), check="all", on_error="raise"
    )
    assert stacked[0].tags.names == {"whole", "other"}


def test_cleared_tags_do_not_come_back_through_ports():
    @block
    def keep(
        f: Framed[tuple[Sig[float, N], Sig[float, N]]],
    ) -> tuple[Annotated[Sig[float, N], clears("foo")], Sig[float, N]]:
        return f  # type: ignore[return-value]  # the tuple frame, parts included

    src = from_iterable(
        [Frame(np.zeros(4)).tag("foo")], type=SignalType(float, (4,)).provides("foo")
    )
    graph = src >> (dsp.scale(gain=1.0) | dsp.scale(gain=2.0)) >> keep
    out = run(graph, check="all", on_error="raise")[0]
    assert "foo" not in out[0] and "foo" in out[1] and "foo" in out  # one port keeps it
    assert "foo" not in run(graph[0], check="all", on_error="raise")[0]
    assert "foo" in run(graph[1], check="all", on_error="raise")[0]

    @block
    def drop_all(
        f: Framed[tuple[Sig[float, N], Sig[float, N]]],
    ) -> Annotated[tuple[Sig[float, N], Sig[float, N]], clears("foo")]:
        return f  # type: ignore[return-value]

    both = src >> (dsp.scale(gain=1.0) | dsp.scale(gain=2.0)) >> drop_all
    out = run(both, check="all", on_error="raise")[0]
    assert "foo" not in out and all("foo" not in p for p in out.parts)
    assert "foo" not in run(both[1], check="all", on_error="raise")[0]


def test_routing_keeps_the_guarantees_every_branch_has():
    @block
    def needs_foo(x: Annotated[Sig[float], requires("foo")]) -> Sig[float]:
        return x

    src = from_iterable(
        [Frame(np.zeros(4)).tag("foo")], type=SignalType(float, (4,)).provides("foo")
    )
    routed = src >> when(lambda f: True, then=dsp.scale(gain=1.0), otherwise=dsp.rms)
    assert str(routed.out_type) == "Sig[Any, tags=foo]"
    assert run(routed >> needs_foo, check="all", on_error="raise")[0].tags.names == {
        "foo"
    }
    passthrough = src >> when(lambda f: False, then=dsp.rms)
    assert passthrough.out_type.meta == {"foo"}
    _ = passthrough >> needs_foo
    from sigflow import switch

    keyed = src >> switch(
        lambda f: "a", {"a": dsp.scale(gain=1.0), "b": dsp.rms}, default="pass"
    )
    assert keyed.out_type.meta == {"foo"}
    _ = keyed >> needs_foo
    with pytest.raises(TypeMismatch, match="needs tag"):
        _ = (
            dsp.sine(n=4)
            >> when(lambda f: True, then=dsp.scale(gain=1.0), otherwise=dsp.rms)
            >> needs_foo
        )


def test_guarantees_are_whole_frame_tags():
    spanned = Frame(np.zeros(4), meta={"t0": 0.0}, stream={"rate": 4.0}).tag(
        "burst", span=(0.0, 0.5)
    )
    src = from_iterable([spanned], type=SignalType(float, (4,)).provides("burst"))
    with pytest.raises(
        TypeMismatch, match="list output: contract guarantees tag.*burst"
    ):
        run(src >> dsp.rms, check="all", on_error="raise")
    honest = from_iterable(
        [spanned.tag("burst")], type=SignalType(float, (4,)).provides("burst")
    )
    assert run(honest >> dsp.rms, check="all", on_error="raise")[0].tags.names == {
        "burst"
    }


def test_routing_on_a_tag_means_a_whole_frame_tag():
    spanned = Frame(np.zeros(4), meta={"t0": 0.0}, stream={"rate": 4.0})
    frames = [spanned.tag("speech", span=(0.0, 0.5)), spanned.tag("speech")]
    routed = from_iterable(frames) >> when("speech", then=asr)
    out = run(routed, check="all", on_error="raise")
    assert [f.prov.op.rsplit("[", 1)[1] for f in out] == ["else]", "then]"]
    from sigflow.runtime import _check_required

    with pytest.raises(TypeMismatch, match="asr input: needs tag.*speech"):
        _check_required(asr.in_type, frames[0], "asr input")
    _check_required(asr.in_type, frames[1], "asr input")


def test_untimed_reframing_forgets_consumed_tags():
    frames = [Frame(np.full(4, float(i))).tag(segment=i) for i in range(3)]
    out = run(from_iterable(frames) >> dsp.chunk(4), check="all", on_error="raise")
    assert [f["segment"] for f in out] == [0, 1, 2]
    out = run(from_iterable(frames) >> dsp.chunk(6), check="all", on_error="raise")
    assert [f.tags for f in out] == [Tags(), Tags()]  # no time axis: partial tags go
    out = run(from_iterable(frames) >> dsp.chunk(2), check="all", on_error="raise")
    assert [f["segment"] for f in out] == [0, 0, 1, 1, 2, 2]


def test_adjacent_inputs_with_the_same_tag_cover_a_combined_output():
    a = Frame(np.zeros(4), meta={"t0": 0.0}, stream={"rate": 4.0}).tag("calibrated")
    b = Frame(np.zeros(4), meta={"t0": 1.0}, stream={"rate": 4.0}).tag("calibrated")
    src = from_iterable(
        [a, b], type=SignalType(float, (4,), 4.0).provides("calibrated")
    )
    out = run(src >> dsp.chunk(6), check="all", on_error="raise")
    assert out[0].tags == Tags(["calibrated"])
    c = Frame(np.zeros(4), meta={"t0": 0.0}, stream={"rate": 4.0}).tag(segment=0)
    d = Frame(np.zeros(4), meta={"t0": 1.0}, stream={"rate": 4.0}).tag(segment=1)
    out = run(from_iterable([c, d]) >> dsp.chunk(6), check="all", on_error="raise")
    assert sorted(map(repr, out[0].tags)) == ["segment=0@[0, 1)", "segment=1@[1, 1.5)"]


def test_per_port_clearing_on_a_plain_tuple_return():
    @block
    def split(
        x: Sig[float, N],
    ) -> tuple[Annotated[Sig[float, N], clears("foo")], Sig[float, N]]:
        return x, x

    src = from_iterable(
        [Frame(np.zeros(4)).tag("foo")], type=SignalType(float, (4,)).provides("foo")
    )
    graph = src >> split
    out = run(graph, check="all", on_error="raise")[0]
    assert "foo" not in out[0] and "foo" in out[1] and "foo" in out
    assert "foo" not in run(graph[0], check="all", on_error="raise")[0]
    assert "foo" in run(graph[1], check="all", on_error="raise")[0]
    assert split.out_type.items[1].meta == frozenset()  # declared: nothing carried yet
    assert (
        graph.out_type.items[1].meta == {"foo"}
        and graph.out_type.items[0].meta == frozenset()
    )

"""Tag contracts in signatures: typed requires / accepts / provides / may_provide,
and the per-block propagation policy (tags=, spans=, merge=)."""

from __future__ import annotations

import enum
from collections.abc import Iterator
from typing import Annotated

import numpy as np
import pytest

from sigflow import (
    Frame,
    Framed,
    Sig,
    SignalType,
    Stream,
    TypeMismatch,
    accepts,
    block,
    carries,
    clears,
    drop,
    from_iterable,
    keep,
    may_provide,
    provides,
    requires,
    run,
)
from sigflow.dims import N
from sigflow.types import TagWarning, tag_type_ok

# these tests are about types and policies; the warnings have tests of their own
pytestmark = pytest.mark.filterwarnings("ignore::sigflow.types.TagWarning")


def source(frames: int = 3, **tags):
    """Frames carrying ``tags``, with a contract that guarantees them."""
    return from_iterable(
        [Frame(np.zeros(4), meta={"seq": i}).tag(**tags) for i in range(frames)],
        type=SignalType(float, (N,)).provides(*tags),
    )


@block
def demod(
    x: Annotated[Sig[float, N], requires("synced", track=int), accepts(snr_db=float)],
) -> Sig[float, N]:
    return x


# ------------------------------------------------------------------ contracts


def test_contract_display_names_every_kind_of_tag():
    t = (
        SignalType(float)
        .requires("synced", track=int)
        .accepts(snr=float)
        .may_provide("crc")
        .clears("raw")
    )
    assert str(t) == (
        "Sig[float64, tags=synced|track:int, accepts=snr:float, maybe=crc, clears=raw]"
    )
    with pytest.raises(TypeError, match="class or a tuple of classes"):
        requires(track="int")


def test_value_types_follow_isinstance_with_numeric_widening_and_enums():
    class Mode(enum.Enum):
        A = "a"

    assert tag_type_ok(3, float) and tag_type_ok(2.0, complex)
    assert not tag_type_ok(True, float) and tag_type_ok(True, bool)
    assert tag_type_ok("a", Mode) and not tag_type_ok("z", Mode)  # tags store values
    assert tag_type_ok(1, (str, int)) and not tag_type_ok(1.5, (str, int))
    assert tag_type_ok(int, float, is_type=True)
    assert not tag_type_ok(str, int, is_type=True)


def test_required_tags_and_their_types_are_checked_when_the_graph_is_built():
    typed = from_iterable(
        [], type=SignalType(float, (N,)).provides("synced", track=int)
    )
    _ = typed >> demod
    untyped = from_iterable(
        [], type=SignalType(float, (N,)).provides("synced", "track")
    )
    _ = untyped >> demod  # no declared type: gradual, checked at run time
    wrong = from_iterable(
        [], type=SignalType(float, (N,)).provides("synced", track=str)
    )
    with pytest.raises(TypeMismatch, match="tag 'track' is str, needs int"):
        _ = wrong >> demod
    with pytest.raises(TypeMismatch, match="needs tag"):
        _ = from_iterable([], type=SignalType(float, (N,)).provides("synced")) >> demod


def test_required_and_accepted_values_are_checked_at_run_time():
    assert len(run(source(synced=True, track=3) >> demod, check="all")) == 3
    with pytest.raises(TypeMismatch, match="demod input: tag 'track' is 'x'"):
        run(source(synced=True, track="x") >> demod, check="all", on_error="raise")
    # an accepted tag is optional...
    assert len(run(source(synced=True, track=1) >> demod, check="all")) == 3
    # ...but typed when it is there
    with pytest.raises(TypeMismatch, match="tag 'snr_db' is 'loud'"):
        run(
            source(synced=True, track=1, snr_db="loud") >> demod,
            check="all",
            on_error="raise",
        )


def test_provided_and_may_provide_values_are_checked_on_the_output():
    @block
    def level(
        f: Framed[Sig[float, N]],
    ) -> Annotated[Sig[float, N], provides(level=float)]:
        return f.tag(level="high")

    with pytest.raises(TypeMismatch, match="level output: tag 'level' is 'high'"):
        run(source(a=1) >> level, check="all", on_error="raise")

    @block
    def crc(
        f: Framed[Sig[float, N]],
    ) -> Annotated[Sig[float, N], may_provide(crc_fail=bool)]:
        return f.tag(crc_fail="nope") if f["seq"] == 1 else f

    g = source(a=1) >> crc
    assert g.out_type.maybe == {"crc_fail"} and "crc_fail" not in g.out_type.meta
    with pytest.raises(TypeMismatch, match="crc output: tag 'crc_fail' is 'nope'"):
        run(g, check="all", on_error="raise")


def test_guaranteed_types_travel_downstream():
    typed = from_iterable(
        [], type=SignalType(float, (N,)).provides("synced", track=int)
    )

    @block
    def ident(x: Sig[float, N]) -> Sig[float, N]:
        return x

    g = typed >> ident
    assert dict(g.out_type.tag_types) == {"track": int}
    _ = g >> demod  # the type reached demod through a block that knows nothing of it


# ------------------------------------------------------------------ policies


def tagged():
    frames = [Frame(np.zeros(4), meta={"seq": 0}).tag("raw", "calibrated", note="x")]
    return from_iterable(
        frames, type=SignalType(float, (N,)).provides("raw", "calibrated")
    )


@pytest.mark.parametrize(
    "options,static,carried",
    [
        ({}, {"raw", "calibrated"}, {"raw", "calibrated", "note"}),
        ({"tags": "none"}, set(), set()),
        ({"tags": keep("calibrated")}, {"calibrated"}, {"calibrated"}),
        ({"tags": drop("raw")}, {"calibrated"}, {"calibrated", "note"}),
    ],
)
def test_policies_filter_inherited_tags_statically_and_at_run_time(
    options, static, carried
):
    @block(**options)
    def framed(f: Framed[Sig[float, N]]) -> Sig[float, N]:
        return f.tag("mine")  # attached here: never filtered

    @block(**options)
    def plain(x: Sig[float, N]) -> Sig[float, N]:
        return x

    g = tagged() >> framed
    assert g.out_type.meta == static
    assert run(g, check="all", on_error="raise")[0].tags.names == carried | {"mine"}
    assert (
        run(tagged() >> plain, check="all", on_error="raise")[0].tags.names == carried
    )


def test_declared_policy_keeps_what_the_input_contract_names():
    @block(tags="declared")
    def reads(
        x: Annotated[Sig[float, N], requires("raw"), accepts("calibrated")],
    ) -> Sig[float, N]:
        return x

    g = tagged() >> reads
    assert g.out_type.meta == {"raw", "calibrated"}
    assert run(g, check="all")[0].tags.names == {"raw", "calibrated"}  # note: gone


def test_clears_still_applies_under_a_policy():
    @block(tags=keep("raw", "calibrated"))
    def b(x: Sig[float, N]) -> Annotated[Sig[float, N], clears("raw")]:
        return x

    g = tagged() >> b
    assert g.out_type.meta == {"calibrated"}
    assert run(g, check="all")[0].tags.names == {"calibrated"}


def test_spans_drop_keeps_whole_frame_tags():
    frames = [
        Frame(np.zeros(8), meta={"t0": 0.0, "seq": 0}, stream={"rate": 8.0})
        .tag("calibrated")
        .tag("burst", span=(0.25, 0.5))
    ]

    @block(spans="drop")
    def b(x: Sig[float, N]) -> Sig[float, N]:
        return x

    out = run(from_iterable(frames) >> b)[0]
    assert out.tags.names == {"calibrated"}


def test_merge_over_a_stream_blocks_consumed_frames():
    frames = [
        Frame(np.zeros(2), meta={"seq": 0}).tag("calibrated", "a"),
        Frame(np.zeros(2), meta={"seq": 1}).tag("calibrated", "b"),
    ]

    def pairs(xs) -> Iterator:
        buf = []
        for x in xs:
            buf.append(x)
            if len(buf) == 2:
                yield np.concatenate(buf)
                buf = []

    union = block(
        pairs, inputs=Stream[Sig[float, N]], outputs=Sig[float, N], name="pairs"
    )
    both = block(
        pairs,
        inputs=Stream[Sig[float, N]],
        outputs=Sig[float, N],
        name="pairs",
        merge="intersection",
    )
    first = block(
        pairs,
        inputs=Stream[Sig[float, N]],
        outputs=Sig[float, N],
        name="pairs",
        merge="first",
    )
    assert run(from_iterable(frames) >> union)[0].tags.names == {"calibrated", "a", "b"}
    assert run(from_iterable(frames) >> both)[0].tags.names == {"calibrated"}
    assert run(from_iterable(frames) >> first)[0].tags.names == {"calibrated", "a"}


def test_merge_over_a_tuple_inputs_parts_and_statically():
    @block
    def mark_a(
        f: Framed[Sig[float, N]],
    ) -> Annotated[Sig[float, N], provides("a", "common")]:
        return f.tag("a", "common")

    @block
    def mark_b(
        f: Framed[Sig[float, N]],
    ) -> Annotated[Sig[float, N], provides("b", "common")]:
        return f.tag("b", "common")

    @block(merge="intersection")
    def add(x: Sig[float, N], y: Sig[float, N]) -> Sig[float, N]:
        return x + y

    from sigflow import dsp

    g = dsp.sine(n=4, frames=1) >> (mark_a | mark_b) >> add
    assert g.out_type.meta == {"common"}
    assert run(g, check="all")[0].tags.names == {"common"}


def test_policy_options_are_validated_and_survive_bind():
    with pytest.raises(ValueError, match="tags="):
        block(lambda x: x, inputs=Sig[float, N], outputs=Sig[float, N], tags="some")
    with pytest.raises(ValueError, match="merge="):
        block(lambda x: x, inputs=Sig[float, N], outputs=Sig[float, N], merge="any")

    @block(tags=keep("calibrated"), spans="drop")
    def b(x: Sig[float, N], gain: float = 1.0) -> Sig[float, N]:
        return x * gain

    bound = b.bind(gain=2.0)
    assert bound.tag_policy == b.tag_policy


# ------------------------------------------------------------------ warnings


def declared(*provided: str, maybe: tuple[str, ...] = ()):
    t = SignalType(float, (N,)).provides(*provided)
    if maybe:
        t = t.may_provide(*maybe)
    return from_iterable([], type=t)


@pytest.mark.filterwarnings("error::sigflow.types.TagWarning")
def test_accepting_a_tag_nothing_can_supply_warns():
    @block
    def reads(x: Annotated[Sig[float, N], accepts("snr_db")]) -> Sig[float, N]:
        return x

    with pytest.warns(TagWarning, match="accepts tag 'snr_db', which nothing upstream"):
        _ = declared("synced") >> reads
    _ = declared("synced", maybe=("snr_db",)) >> reads  # may arrive: fine
    _ = declared("snr_db") >> reads  # guaranteed: fine
    _ = from_iterable([]) >> reads  # a source that says nothing about tags: gradual


@pytest.mark.filterwarnings("error::sigflow.types.TagWarning")
def test_may_provide_travels_downstream_under_the_policy():
    @block
    def crc(
        f: Framed[Sig[float, N]],
    ) -> Annotated[Sig[float, N], may_provide("crc_fail")]:
        return f

    @block
    def ident(x: Sig[float, N]) -> Sig[float, N]:
        return x

    @block(tags="none")
    def forget(x: Sig[float, N]) -> Sig[float, N]:
        return x

    @block
    def reads(x: Annotated[Sig[float, N], accepts("crc_fail")]) -> Sig[float, N]:
        return x

    g = declared("synced") >> crc >> ident
    assert g.out_type.maybe == {"crc_fail"}
    _ = g >> reads  # two blocks down, still possible
    gone = declared("synced") >> crc >> forget
    assert gone.out_type.maybe == frozenset()


@pytest.mark.filterwarnings("error::sigflow.types.TagWarning")
def test_routing_on_a_tag_nothing_can_supply_warns():
    from sigflow import dsp, when

    with pytest.warns(TagWarning, match="routes on tag 'speech'"):
        _ = declared("synced") >> when("speech", then=dsp.scale(gain=1.0))
    _ = declared("synced", maybe=("speech",)) >> when(
        "speech", then=dsp.scale(gain=1.0)
    )


@pytest.mark.filterwarnings("error::sigflow.types.TagWarning")
def test_per_key_checks_its_key():
    from sigflow import dsp, per_key

    with pytest.warns(TagWarning, match="keys on tag 'track', which nothing upstream"):
        _ = declared("synced") >> per_key("track", dsp.scale(gain=1.0))
    with pytest.warns(TagWarning, match="only may provide: frames without it pass"):
        _ = declared("synced", maybe=("track",)) >> per_key(
            "track", dsp.scale(gain=1.0)
        )
    _ = declared("track") >> per_key("track", dsp.scale(gain=1.0))
    _ = from_iterable([]) >> per_key("track", dsp.scale(gain=1.0))  # says nothing


# ------------------------------------------------------------------ per port, surfacing


def test_a_port_with_its_own_policy():
    @block
    def split(
        x: Sig[float, N],
    ) -> tuple[Annotated[Sig[float, N], carries("none")], Sig[float, N]]:
        return x, x

    g = tagged() >> split
    assert [set(i.meta) for i in g.out_type.items] == [set(), {"raw", "calibrated"}]
    out = run(g, check="all", on_error="raise")[0]
    assert out[0].tags.names == set() and out[1].tags.names == {
        "raw",
        "calibrated",
        "note",
    }
    assert out.tags.names == {"raw", "calibrated", "note"}  # a part still carries them

    @block(tags="none")
    def quiet_but_one(
        x: Sig[float, N],
    ) -> tuple[Sig[float, N], Annotated[Sig[float, N], carries(keep("calibrated"))]]:
        return x, x

    out = run(tagged() >> quiet_but_one, check="all", on_error="raise")[0]
    assert out[0].tags.names == set() and out[1].tags.names == {"calibrated"}
    assert out.tags.names == {"calibrated"}


def test_describe_and_the_report_show_contracts_and_policies():
    @block(tags=keep("calibrated"), spans="drop")
    def reads(
        f: Framed[Annotated[Sig[float, N], requires("raw"), accepts(snr=float)]],
    ) -> Annotated[Sig[float, N], provides(level=float), may_provide("crc")]:
        return f.tag(level=1.0)

    g = tagged() >> reads
    assert "[tags=keep(calibrated), spans=drop]" in g.describe()
    from sigflow.notebook import tag_contracts

    assert tag_contracts(g) == [
        (
            "reads",
            "requires raw; accepts snr: float",
            "provides level: float; may provide crc",
            "tags=keep(calibrated), spans=drop",
        )
    ]
    handle = g.start(paused=True)
    handle.resume()
    handle.wait()
    page = handle.report().html
    assert "Tag contracts" in page and "may provide crc" in page

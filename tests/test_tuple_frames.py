"""Tuple frames: the parts, the data and the tags must tell one story."""

from __future__ import annotations

import os
from typing import Annotated

import numpy as np

from sigflow import Frame, Framed, Sig, block, dsp, provides, run
from sigflow.dims import N


@block
def flag(f: Framed[Sig[float, N]]) -> Annotated[Sig[float, N], provides("foo")]:
    return f.tag("foo")


def forked():
    return dsp.sine(n=4, frames=1) >> (flag | dsp.scale(gain=1.0))


def test_evolving_the_data_drops_parts_that_described_the_old_data():
    @block
    def doubled(
        f: Framed[tuple[Sig[float, N], Sig[float, N]]],
    ) -> tuple[Sig[float, N], Sig[float, N]]:
        return f.evolve(data=(f.data[0] * 2, f.data[1] * 2))  # type: ignore[return-value]

    out = run(forked() >> doubled, check="all", on_error="raise")[0]
    assert out.parts == ()
    np.testing.assert_allclose(out[0].data, out.data[0])  # not the old frame's data
    selected = run((forked() >> doubled)[0], check="all", on_error="raise")[0]
    np.testing.assert_allclose(selected.data, out.data[0])
    f = Frame((np.zeros(2), np.ones(2)), parts=(Frame(np.zeros(2)), Frame(np.ones(2))))
    assert f.evolve(data=(np.ones(2), np.ones(2))).parts == ()
    assert f.evolve(meta={"x": 1}).parts == f.parts  # unchanged data keeps them
    kept = f.evolve(data=(np.ones(2), np.ones(2)), parts=f.parts)
    assert kept.parts == f.parts  # explicit is explicit


def test_a_plain_tuple_return_keeps_each_ports_tags():
    @block
    def passthrough(
        x: tuple[Sig[float, N], Sig[float, N]],
    ) -> tuple[Sig[float, N], Sig[float, N]]:
        return x

    g = forked() >> passthrough
    assert g.out_type.items[0].meta == {"foo"} and g.out_type.items[1].meta == set()
    out = run(g, check="all", on_error="raise")[0]
    assert "foo" in out[0].tags and "foo" not in out[1].tags  # as the contract says
    assert "foo" in out.tags  # the tuple frame carries the union
    assert "foo" not in run(g[1], check="all", on_error="raise")[0].tags

    @block
    def same(
        x: tuple[Sig[float, N], Sig[float, N]],
    ) -> tuple[Sig[float, N], Sig[float, N]]:
        return x

    plain = run(
        dsp.sine(n=4, frames=1) >> (dsp.scale(gain=1.0) | dsp.scale(gain=2.0)) >> same
    )[0]
    assert plain.parts == ()  # ports with the same tags need no parts of their own


def test_tuple_frames_round_trip_per_port_tags(tmp_path):
    out = run(forked(), check="all", on_error="raise")[0]
    back = Frame.load(out.save(os.path.join(tmp_path, "tuple.npz")))
    assert back.tags.names == {"foo"}
    assert "foo" in back[0].tags and "foo" not in back[1].tags
    np.testing.assert_allclose(back[1].data, out[1].data)
    single = run(dsp.sine(n=4, frames=1))[0]
    assert Frame.load(single.save(os.path.join(tmp_path, "one.npz"))).parts == ()

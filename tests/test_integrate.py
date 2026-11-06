"""Integration across frame boundaries: a stream integrated in frames matches the
whole signal integrated at once, and holes, resets and leaks do what they say."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate as si

from sigflow import Frame, as_node, dsp, run
from sigflow import integrate as ig

RATE = 1000.0
T = np.arange(640) / RATE
X = np.cos(2 * np.pi * 7 * T) + 0.5
TRUTH = np.sin(2 * np.pi * 7 * T) / (2 * np.pi * 7) + 0.5 * T


def stream(x: np.ndarray, n: int = 64):
    # from_array called with an array runs at once: as_node makes it a graph node
    return as_node(dsp.from_array(np.asarray(x), n=n, rate=RATE))


def frames(x, starts, n=64, units: str | None = "V", tag=None):
    """Frames ``starts`` (indices of n-sample slots) of ``x``, with times and units;
    ``tag`` maps a position in ``starts`` to a function tagging that frame."""
    out = []
    for k, s in enumerate(starts):
        f = Frame(
            x[s * n : (s + 1) * n],
            meta={"t0": s * n / RATE},
            stream={"rate": RATE, **({"units": units} if units else {})},
        )
        if tag and k in tag:
            f = tag[k](f)
        out.append(f)
    return as_node(iter(out))


def joined(out: list[Frame]) -> np.ndarray:
    return np.concatenate([f.data for f in out])


@pytest.mark.parametrize("n", [1, 7, 64])
def test_a_running_trapezoid_matches_the_whole_signal(n):
    x = X[: n * (640 // n)]
    out = run(stream(x, n) >> ig.cumulative, on_error="raise")
    whole = si.cumulative_trapezoid(x, dx=1 / RATE, initial=0)
    np.testing.assert_allclose(joined(out), whole, atol=1e-14)


def test_simpson_is_as_accurate_as_on_the_whole_signal():
    out = joined(run(stream(X) >> ig.cumulative(method="simpson"), on_error="raise"))
    whole = si.cumulative_simpson(X, dx=1 / RATE, initial=0)
    assert np.abs(out - TRUTH).max() < 3 * np.abs(whole - TRUTH).max()
    assert (
        np.abs(out - TRUTH).max()
        < 1e-7
        < np.abs(joined(run(stream(X) >> ig.cumulative)) - TRUTH).max()
    )


def test_frame_integrals_add_up_to_the_whole():
    out = run(stream(X) >> ig.definite, on_error="raise")
    assert sum(f.data for f in out) == pytest.approx(
        si.trapezoid(X, dx=1 / RATE), rel=1e-13
    )
    win = run(stream(X) >> ig.definite(window=3), on_error="raise")
    assert win[-1].data == pytest.approx(sum(f.data for f in out[-3:]))
    assert "rate" not in out[0].stream  # one value per frame, not samples


def test_initial_is_the_starting_value():
    out = run(stream(np.zeros(128)) >> ig.cumulative(initial=2.5), on_error="raise")
    assert np.all(joined(out) == 2.5)


def test_a_hole_is_bridged_with_a_warning_or_restarts_the_integral():
    ones = np.ones(64 * 6)
    bridged = run(frames(ones, [0, 1, 3, 4]) >> ig.cumulative, on_error="raise")
    assert bridged[-1].data[-1] == pytest.approx((5 * 64 - 1) / RATE)  # as if no hole
    assert any("bridged a gap of 0.064" in i.message for i in bridged[2].warnings)
    reset = run(
        frames(ones, [0, 1, 3, 4]) >> ig.cumulative(gaps="reset"), on_error="raise"
    )
    assert [sorted(f.tags.names) for f in reset] == [[], [], ["integral_reset"], []]
    assert reset[2].data[0] == 0 and reset[3].data[-1] == pytest.approx(127 / RATE)


def test_reset_on_holds_the_integral_at_zero_where_the_tag_applies():
    still = {1: lambda f: f.tag("still", span=(0.074, 0.084))}  # samples 74-83
    out = run(
        frames(np.ones(64 * 3), [0, 1, 2], tag=still)
        >> ig.cumulative(reset_on="still"),
        on_error="raise",
    )
    y = joined(out)
    assert y[73] == pytest.approx(0.073)
    assert np.all(y[74:84] == 0)
    assert y[84] == pytest.approx(0.001) and y[-1] == pytest.approx((191 - 83) / RATE)


def test_a_leak_stops_a_bias_from_growing_without_bound():
    bias = np.full(64 * 200, 0.001)
    out = run(
        frames(bias, range(200), units=None) >> ig.cumulative(leak=2.0),
        on_error="raise",
    )
    t_end = len(bias) / RATE
    assert out[-1].data[-1] == pytest.approx(
        0.001 * 2.0 * (1 - math.exp(-t_end / 2.0)), rel=1e-3
    )


@pytest.mark.parametrize(
    "units, integral",
    [("V", "V*s"), ("W", "J"), ("m/s^2", "m/s"), ("m/s", "m"), ("mA", "mA*s")],
)
def test_the_output_is_in_units_times_seconds(units, integral):
    out = run(
        frames(np.ones(128), [0, 1], units=units) >> ig.cumulative, on_error="raise"
    )
    assert out[0].stream["units"] == integral
    d = run(frames(np.ones(128), [0, 1], units=units) >> ig.definite, on_error="raise")
    assert d[0].stream["units"] == integral


def test_the_integral_converts_like_any_unit():
    g = (
        frames(np.ones(64 * 3), [0, 1, 2], units="W")
        >> ig.cumulative
        >> dsp.to_units("kJ")
    )
    out = run(g, on_error="raise")
    assert out[-1].stream["units"] == "kJ"
    assert out[-1].data[-1] == pytest.approx((3 * 64 - 1) / RATE / 1000)


def test_each_run_starts_from_zero():
    g = dsp.sine(n=64, frames=4) >> ig.cumulative
    assert np.array_equal(run(g)[-1].data, run(g)[-1].data)


def test_bad_options_are_refused():
    for kw, match in [
        ({"method": "romberg"}, "method must be one of"),
        ({"leak": 0}, "positive"),
        ({"gaps": "skip"}, "gaps must be"),
    ]:
        with pytest.raises(ValueError, match=match):
            run(dsp.sine(n=8, frames=1) >> ig.cumulative(**kw), on_error="raise")
    with pytest.raises(ValueError, match="rate"):
        run(dsp.from_array(np.ones(8), n=4) >> ig.cumulative, on_error="raise")


# -- review fixes: time-true leaks, straight bridges, promoted dtypes ---------- #


def at(values, times, dtype=float):
    """One frame per (values, start time), at 1 Hz."""
    return as_node(
        iter(
            Frame(np.asarray(v, dtype=dtype), meta={"t0": t}, stream={"rate": 1.0})
            for v, t in zip(values, times)
        )
    )


def test_a_leak_starts_at_initial_and_decays_over_real_time():
    out = run(at([[0.0, 0.0]], [0.0]) >> ig.cumulative(initial=1.0, leak=1.0))
    np.testing.assert_allclose(out[0].data, [1.0, math.exp(-1)])
    out = run(at([[0.0], [0.0]], [1.0, 10.0]) >> ig.cumulative(initial=1.0, leak=1.0))
    assert out[1].data[0] / out[0].data[0] == pytest.approx(math.exp(-9))


def test_a_leak_over_a_reset_restarts_from_zero():
    out = run(
        at([[1.0, 1.0], [1.0, 1.0]], [0.0, 10.0])
        >> ig.cumulative(leak=1.0, gaps="reset"),
        on_error="raise",
    )
    assert out[1].data[0] == 0 and "integral_reset" in out[1].tags


@pytest.mark.parametrize("method", ["trapezoid", "simpson"])
def test_a_bridge_is_a_straight_line_whatever_the_method(method):
    out = run(at([[0, 0], [1, 1]], [0.0, 10.0]) >> ig.cumulative(method=method))
    np.testing.assert_allclose(out[1].data, [4.5, 5.5])
    reset = run(
        at([[0, 0], [1, 1]], [0.0, 10.0]) >> ig.cumulative(method=method, gaps="reset")
    )
    np.testing.assert_allclose(reset[1].data, [0.0, 1.0])


@pytest.mark.parametrize(
    "dtype, result",
    [("int64", "float64"), ("float32", "float64"), ("complex64", "complex128")],
)
def test_the_contract_promotes_the_dtype_as_the_result_does(dtype, result):
    from sigflow import Sig, from_iterable
    from sigflow.dims import N
    from sigflow.types import TypeMismatch, resolve

    def source():
        frame = Frame(
            np.arange(4).astype(dtype), meta={"t0": 0.0}, stream={"rate": 1.0}
        )
        return from_iterable(lambda: iter([frame]), Sig[dtype, N])

    for blk in (ig.cumulative, ig.definite):
        g = source() >> blk
        assert resolve(g.out_type, g.validate()).dtype == np.dtype(result)
        assert np.asarray(run(g, on_error="raise")[0].data).dtype == np.dtype(result)

    @block_of(np.int64)
    def ints_only(x):
        return x

    with pytest.raises(TypeMismatch, match=f"dtype {result} cannot feed int64"):
        _ = source() >> ig.cumulative >> ints_only


def block_of(dtype):
    """A one-dimensional map block taking and giving ``dtype`` only."""
    from sigflow import Sig, block
    from sigflow.dims import N

    return lambda fn: block(fn, inputs=Sig[dtype, N], outputs=Sig[dtype, N])

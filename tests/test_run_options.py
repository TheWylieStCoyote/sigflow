"""What each entry point accepts: a misspelt option or policy is an error, not a
silent no-op."""

from __future__ import annotations

from contextlib import closing

import pytest

from sigflow import block, dsp, iterate, iterate_sync, run


def graph():
    return dsp.sine(n=8, frames=3) >> dsp.rms


def test_an_unknown_error_policy_is_refused():
    with pytest.raises(ValueError, match="on_error must be one of"):
        run(graph(), on_error="mark")
    with pytest.raises(ValueError, match="on_error must be one of"):
        block(lambda x: x, inputs=float, outputs=float, on_error="skip")


def test_iteration_refuses_options_it_does_not_take():
    with pytest.raises(TypeError, match="sink"):
        list(iterate_sync(graph(), sink=print))
    with pytest.raises(TypeError, match="on_eror"):
        iterate(graph(), on_eror="raise")


def test_start_refuses_unknown_options_when_called():
    with pytest.raises(TypeError, match="unexpected option.*nonsense"):
        graph().start(nonsense=1)
    with pytest.raises(TypeError, match="control"):
        graph().start(control=None)


def test_start_takes_keep_and_the_run_options():
    h = dsp.sine(n=8, frames=20).start(keep=5, on_error="raise", queue_size=4)
    assert len(h.wait(timeout=10)) == 5
    with pytest.raises(TypeError, match="keep"):
        run(graph(), keep=5)


def test_iterators_close_explicitly():
    with closing(iterate_sync(graph())) as frames:
        assert next(frames).data >= 0

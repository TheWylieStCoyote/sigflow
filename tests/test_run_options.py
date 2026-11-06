"""What each entry point accepts: a misspelt option or policy is an error, not a
silent no-op."""

from __future__ import annotations

from contextlib import closing

import pytest

from sigflow import Stateful, block, dsp, iterate, iterate_sync, run


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


class Count(Stateful):
    """Counts frames; no reset(), so the count lives on between runs."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, x: float) -> float:
        self.n += 1
        return float(self.n)


class FreshCount(Count):
    def reset(self) -> None:
        self.n = 0


def test_state_lives_on_between_runs_unless_the_block_resets():
    keeps = (
        dsp.sine(n=8, frames=2)
        >> dsp.rms
        >> block(Count(), inputs=float, outputs=float)
    )
    fresh = (
        dsp.sine(n=8, frames=2)
        >> dsp.rms
        >> block(FreshCount(), inputs=float, outputs=float)
    )
    assert [f.data for f in run(keeps)] == [1.0, 2.0]
    assert [f.data for f in run(keeps)] == [3.0, 4.0]
    assert [f.data for f in run(fresh)] == [1.0, 2.0]
    assert [f.data for f in run(fresh)] == [1.0, 2.0]


def test_iteration_resumes_block_state_from_a_checkpoint():
    from sigflow.persist import Checkpoint

    count = block(FreshCount(), inputs=float, outputs=float)
    g = dsp.sine(n=8, frames=3) >> dsp.rms >> count
    ck = Checkpoint(positions={"sine": 1}, states={count.name: {"n": 1}})
    assert [f.data for f in iterate_sync(g, resume=ck)] == [2.0, 3.0]

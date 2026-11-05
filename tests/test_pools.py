"""parallel() backends: process, thread and loky."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from sigflow import Sig, block, dsp, parallel, pools, run
from sigflow.dims import N

ROOT = Path(__file__).resolve().parents[1]


@block
def rooted(x: Sig[float, N], gain: float = 1.0) -> Sig[float, N]:
    return np.sqrt(np.abs(x)) * gain


@block
def fails_on_third(x: Sig[float, N]) -> Sig[float, N]:
    if round(float(x[0])) == 2:
        raise ValueError("third frame")
    return x


def ramp(frames: int = 6):
    from sigflow import Frame, from_iterable

    return from_iterable(
        [Frame(np.full(4, float(i)), meta={"seq": i}) for i in range(frames)]
    )


def test_unknown_and_unavailable_backends_are_refused_when_built(monkeypatch):
    with pytest.raises(ValueError, match="'process', 'thread', 'loky'"):
        parallel(rooted, backend="gpu")
    monkeypatch.setattr(pools, "available", lambda b: b != "loky")
    with pytest.raises(ImportError, match="pip install loky"):
        parallel(rooted, backend="loky")


@pytest.mark.parametrize("backend", ["process", "thread", "loky"])
def test_every_backend_gives_the_in_process_result_in_order(backend):
    if not pools.available(backend):
        pytest.skip(f"{backend} not installed")
    g = dsp.sine(n=64, frames=10) >> parallel(
        rooted(gain=3.0), workers=2, backend=backend
    )
    out = run(g, on_error="raise", check="all")
    ref = run(dsp.sine(n=64, frames=10) >> rooted(gain=3.0))
    assert [f["seq"] for f in out] == list(range(10))
    assert all(np.allclose(a.data, b.data) for a, b in zip(out, ref))


@pytest.mark.parametrize("backend", ["process", "thread", "loky"])
def test_a_failing_frame_is_a_failed_frame(backend):
    if not pools.available(backend):
        pytest.skip(f"{backend} not installed")
    out = run(ramp() >> parallel(fails_on_third, workers=2, backend=backend))
    assert [f.failed for f in out] == [False, False, True, False, False, False]
    assert "third frame" in out[2].errors[0].message


def test_loky_pool_is_reused_between_runs():
    pytest.importorskip("loky")
    g = ramp(4) >> parallel(rooted, workers=2, backend="loky")
    first = pools.executor("loky", 2)._inner
    run(g)
    run(g)
    assert pools.executor("loky", 2)._inner is first  # still up for the next run


LOKY_SCRIPT = textwrap.dedent(
    """\
    # no `if __name__ == "__main__":` on purpose, and blocks defined right here
    import numpy as np
    from sigflow import Sig, block, dsp, parallel, run
    from sigflow.dims import N

    @block
    def local(x: Sig[float, N], gain: float = 1.0) -> Sig[float, N]:
        return x * gain

    double = block(lambda x: x * 2, inputs=Sig[float, N], outputs=Sig[float, N],
                   name="double")
    a = run(dsp.sine(n=8, frames=4) >> parallel(local(gain=3.0), workers=2,
                                                   backend="loky"), on_error="raise")
    b = run(dsp.sine(n=8, frames=4) >> parallel(double, workers=2, backend="loky"),
            on_error="raise")
    ref = run(dsp.sine(n=8, frames=4))
    assert all(np.allclose(x.data, 3 * r.data) for x, r in zip(a, ref))
    assert all(np.allclose(x.data, 2 * r.data) for x, r in zip(b, ref))
    print("ok")
    """
)


def test_loky_runs_unguarded_scripts_with_script_defined_blocks(tmp_path):
    pytest.importorskip("loky")
    script = tmp_path / "unguarded.py"
    script.write_text(LOKY_SCRIPT, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")])
    )
    done = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "ok"


def test_blocks_defined_like_a_notebook_cell_run_on_loky():
    pytest.importorskip("loky")
    namespace: dict = {}
    exec(  # noqa: S102 - a notebook cell: its function is not importable
        compile(
            "from sigflow import Sig, block\n"
            "from sigflow.dims import N\n"
            "@block\n"
            "def cell(x: Sig[float, N]) -> Sig[float, N]:\n"
            "    return x + 1\n",
            "<cell>",
            "exec",
        ),
        namespace,
    )
    cell = namespace["cell"]
    out = run(ramp(3) >> parallel(cell, workers=2, backend="loky"), on_error="raise")
    assert [float(f.data[0]) for f in out] == [1.0, 2.0, 3.0]

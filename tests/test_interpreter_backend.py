"""parallel(block, backend="interpreter"): Python 3.14's InterpreterPoolExecutor.

Sub-interpreters cannot load NumPy (or sigflow, which imports it), so the block's
function lives in a plain-Python module and the frames carry plain values. The runs
happen in a child interpreter whose ``PYTHONPATH`` lets each sub-interpreter import
that module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sigflow import SignalType, block, parallel

ROOT = Path(__file__).resolve().parents[1]
HAS_POOL = sys.version_info >= (3, 14)
needs_pool = pytest.mark.skipif(not HAS_POOL, reason="InterpreterPoolExecutor: 3.14+")

BYTES = SignalType(bytes, ())

PLAIN = """
def checksum(data, *, mod=256):
    return sum(data) % mod

def boom(data):
    raise ValueError(f"bad packet {data!r}")

def late_numpy(data):
    import numpy  # reached only in the worker
    return data
"""

SCRIPT = """
import json
import numpy as np
from sigflow import SignalType, block, from_iterable, parallel, run
import plain

t = SignalType(bytes, ())
pkts = [bytes([i, i + 1, i + 2]) for i in range(30)]
report = {}

ck = block(plain.checksum, inputs=t, outputs=SignalType(int, ()), name="ck")
out = run(from_iterable(pkts, type=t) >> parallel(ck, workers=3, backend="interpreter"))
report["ok"] = [f.data for f in out] == [sum(p) % 256 for p in pkts]

knob = ck(mod=7)
out = run(from_iterable(pkts, type=t) >> parallel(knob, backend="interpreter"))
report["cfg"] = [f.data for f in out] == [sum(p) % 7 for p in pkts]

bad = block(plain.boom, inputs=t, outputs=t, name="boom")
out = run(from_iterable(pkts[:3], type=t) >> parallel(bad, backend="interpreter"))
report["failed"] = [len(f.issues) for f in out]

def stopped(source, blk):
    try:
        run(source >> parallel(blk, backend="interpreter"), on_error="drop")
    except RuntimeError as e:
        return [type(e.__cause__).__name__, str(e)]
    return None

arr = SignalType(float, (4,))
report["array"] = stopped(
    from_iterable([np.zeros(4)], type=arr),
    block(plain.checksum, inputs=arr, outputs=SignalType(int, ()), name="ck"),
)
report["late"] = stopped(
    from_iterable(pkts[:1], type=t),
    block(plain.late_numpy, inputs=t, outputs=t, name="late"),
)
print(json.dumps(report))
"""


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict:
    where = tmp_path_factory.mktemp("interp")
    (where / "plain.py").write_text(PLAIN, encoding="utf-8")
    (where / "script.py").write_text(textwrap.dedent(SCRIPT), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(where)])
    done = subprocess.run(
        [sys.executable, str(where / "script.py")],
        cwd=where,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,  # the report, or stderr, is what the tests examine
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(HAS_POOL, reason="this interpreter has InterpreterPoolExecutor")
def test_refused_before_3_14():
    with pytest.raises(RuntimeError, match="needs Python 3.14 or later"):
        parallel(block(len, inputs=BYTES, outputs=BYTES), backend="interpreter")


@needs_pool
def test_results_match_and_keep_their_order(report):
    assert report["ok"]
    assert report["cfg"]


@needs_pool
def test_an_exception_in_the_block_is_a_failed_frame(report):
    assert report["failed"] == [1, 1, 1]


@needs_pool
def test_data_a_sub_interpreter_cannot_receive_stops_the_run(report):
    cause, message = report["array"]
    assert cause == "NotShareableError"
    assert "plain Python values" in message


@needs_pool
def test_a_numpy_import_in_the_worker_stops_the_run(report):
    cause, message = report["late"]
    assert cause == "ImportError"
    assert "cannot load NumPy" in message


@needs_pool
def test_functions_a_sub_interpreter_cannot_import_are_refused_at_build():
    import sigflow.dsp

    with pytest.raises(TypeError, match="must be importable"):
        parallel(block(lambda x: x, inputs=BYTES, outputs=BYTES), backend="interpreter")

    def nested(x):
        return x

    with pytest.raises(TypeError, match="must be importable"):
        parallel(block(nested, inputs=BYTES, outputs=BYTES), backend="interpreter")
    with pytest.raises(TypeError, match="imports numpy"):
        parallel(sigflow.dsp.rms, backend="interpreter")
    with pytest.raises(TypeError, match="label-aware"):
        parallel(
            block(len, inputs=BYTES, outputs=BYTES, labels=True), backend="interpreter"
        )


def test_unknown_backend_lists_the_choices():
    from sigflow import pools

    with pytest.raises(ValueError, match="backend must be one of") as refused:
        parallel(block(len, inputs=BYTES, outputs=BYTES), backend="gpu")
    assert all(repr(name) in str(refused.value) for name in pools.BACKENDS)
    assert "interpreter" in pools.BACKENDS

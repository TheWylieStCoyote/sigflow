"""Early iterator shutdown must release its source and background workers.

Run the probes in disposable processes: a broken queue bridge can strand an
executor thread during interpreter shutdown and would otherwise hang pytest.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_PROBE = textwrap.dedent("""\
    from contextlib import closing
    import json
    import os
    import sys
    import threading
    import time
    from typing import Iterator

    import numpy as np
    from sigflow import Sig, iterate_sync, replay, run, source
    from sigflow.dims import N

    total = 4096  # more than the bridge can buffer without a consumer
    produced = 0
    source_closed = threading.Event()

    @source
    def counting(frames: int = total) -> Iterator[Sig[float, N]]:
        global produced
        try:
            for i in range(frames):
                produced += 1
                yield np.array([float(i)])
        finally:
            source_closed.set()

    graph = counting(frames=total)
    mode = sys.argv[1]
    if mode == "replay":
        target = run(graph)[0]
        assert source_closed.is_set(), "reference run did not close its source"
        source_closed.clear()
        produced = 0

    previous_threads = set(threading.enumerate())
    consumer_error = None
    if mode == "replay":
        first = replay(graph, target)
    elif mode == "close":
        iterator = iterate_sync(graph)
        first = next(iterator)
        iterator.close()
    elif mode == "consumer_error":
        try:
            with closing(iterate_sync(graph)) as iterator:
                first = next(iterator)
                raise RuntimeError("consumer stopped")
        except RuntimeError as exc:
            consumer_error = str(exc)
    else:
        raise AssertionError(mode)

    deadline = time.monotonic() + 2.0
    closed = source_closed.wait(max(0.0, deadline - time.monotonic()))
    for thread in set(threading.enumerate()) - previous_threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    report = {
        "source_closed": closed,
        "live_threads": sorted(
            thread.name
            for thread in set(threading.enumerate()) - previous_threads
        ),
        "produced": produced,
        "total": total,
        "first": float(first.data[0]),
        "consumer_error": consumer_error,
    }
    print(json.dumps(report), flush=True)
    # The parent checks the report. Terminate even if broken background workers
    # are still blocked, rather than hanging in ThreadPoolExecutor's exit hook.
    os._exit(0)
""")


@pytest.mark.parametrize("mode", ["close", "consumer_error", "replay"])
def test_early_sync_iteration_closes_source_and_workers(mode):
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
    process = subprocess.run(
        [sys.executable, "-c", _PROBE, mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["first"] == 0.0
    if mode == "consumer_error":
        assert report["consumer_error"] == "consumer stopped"
    assert report["source_closed"], f"{mode} did not close the source: {report}"
    assert not report["live_threads"], f"{mode} left background workers alive: {report}"
    assert report["produced"] < report["total"], (
        f"{mode} exhausted the source instead of stopping early: {report}"
    )

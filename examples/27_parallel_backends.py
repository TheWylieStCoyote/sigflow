"""One workload on every backend of ``parallel()``: same results, different costs.

Run ``python examples/27_parallel_backends.py``. The block computes a CRC-16 over
each packet in pure Python, so it holds the GIL: the case where the choice of backend
matters most. The function lives in ``plain_decoders.py``, a module that imports
neither NumPy nor sigflow, so every backend can run it, sub-interpreters included.

For each backend the script checks it is available here, runs the same graph, checks
the results equal a plain serial run, in order, and prints the time taken:

- ``thread``: shares the data; one GIL, so pure Python does not speed up (unless
  this is a free-threaded build);
- ``process`` (the default): pickles each packet to a worker process; needs the
  ``if __name__ == "__main__":`` guard below;
- ``loky``: like ``process``, without the guard; needs ``pip install loky``;
- ``interpreter``: sub-interpreters in this process; needs Python 3.14+ and a plain
  module;
- ``freethreaded``: threads without a GIL; needs ``python3.13t`` / ``python3.14t``;
- ``auto``: ``freethreaded`` when the GIL is off, ``process`` otherwise.

Timings depend on the machine; on four or more cores, ``process``, ``loky``,
``interpreter`` and ``freethreaded`` should beat ``thread`` clearly.
"""

import os
import sys
import time

import plain_decoders

from sigflow import SignalType, block, from_iterable, parallel, pools, run

PACKETS = [bytes((i * 7 + j) % 256 for j in range(512)) for i in range(48)]
BYTES, INT = SignalType(bytes, ()), SignalType(int, ())
ROUNDS = 100  # CRC passes per packet: enough work that start-up costs do not dominate

crc = block(plain_decoders.crc16, inputs=BYTES, outputs=INT, name="crc16")

NEEDS = {
    "loky": "pip install loky",
    "interpreter": "Python 3.14 or later",
    "freethreaded": "a free-threaded build (python3.13t, python3.14t)",
}


def packets():
    return from_iterable(PACKETS, type=BYTES)


def timed(graph) -> tuple[list[int], float]:
    start = time.perf_counter()
    out = run(graph, on_error="raise")
    return [f.data for f in out], time.perf_counter() - start


def main() -> None:
    workers = min(4, os.cpu_count() or 1)
    reference, serial = timed(packets() >> crc(rounds=ROUNDS))
    print(f"Python {sys.version.split()[0]}, {workers} workers, {len(PACKETS)} packets")
    print(f"  {'serial':12s} {serial:6.2f} s  (the reference)")
    for backend in pools.BACKENDS:
        if not pools.available(backend):
            print(f"  {backend:12s} skipped: needs {NEEDS.get(backend, '?')}")
            continue
        node = parallel(crc(rounds=ROUNDS), workers=workers, backend=backend)
        results, took = timed(packets() >> node)
        same = "same results, in order" if results == reference else "DIFFERENT"
        print(f"  {backend:12s} {took:6.2f} s  x{serial / took:4.1f}  {same}")
        assert results == reference, backend
    if pools.free_threaded():
        print(
            f"  free-threaded build; the GIL is {'on' if pools.gil_enabled() else 'off'}"
        )


if __name__ == "__main__":  # process workers import this script: run behind this line
    main()

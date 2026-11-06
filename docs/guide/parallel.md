# Parallel blocks

`parallel(block, workers=4, backend="process", window=None)` runs one block on a
pool of workers while the graph sees an ordinary block: frames go out as they
arrive, results come back **in input order**, provenance and annotations are
attached as usual, and a frame whose block raises becomes a failed frame
instead of stopping the pool.

```python
from sigflow import dsp, parallel, run

heavy = dsp.power_db  # any single-input map block that only reads the data

if __name__ == "__main__":
    out = run(
        dsp.noise(n=1 << 16, frames=32) >> dsp.to_complex >> parallel(heavy, workers=4)
    )
```

The block must be a **map block with one input that does not ask for the
frame** (`Framed[...]`): a worker gets the frame's data and the configuration,
not the frame. Stream blocks and multi-input blocks are refused when the graph
is built. `examples/27_parallel_backends.py` runs one workload on every backend.

## Choosing a backend

| your workload | backend | why |
|---|---|---|
| NumPy, SciPy, anything that releases the GIL | `"thread"` | no copying, no start-up; the library runs in parallel already |
| pure Python on ordinary CPython, in a script | `"process"` (default) | each worker has its own interpreter and GIL |
| pure Python, block defined in a notebook or a script without a main guard | `"loky"` (`pip install loky`) | process pool that ships notebook and lambda blocks by value, needs no main guard |
| anything, on a free-threaded build (`python3.14t`) | `"freethreaded"` | threads run Python code in parallel: no copying, no start-up |
| code that should run well on either kind of build | `"auto"` | `"freethreaded"` when the GIL is off, `"process"` otherwise |
| plain-Python parsing of small payloads, Python 3.14+ | `"interpreter"` | sub-interpreters, one per worker, in one process |

What each one costs and requires:

| backend | data to a worker | block must be | needs |
|---|---|---|---|
| `thread` | shared, no copy | anything | — |
| `process` | pickled both ways | importable: defined at module level | a script runs its graph under `if __name__ == "__main__":` |
| `loky` | pickled both ways | anything picklable by cloudpickle | the `parallel` extra |
| `freethreaded` | shared, no copy | anything thread-safe | a free-threaded build |
| `auto` | either of the above | suitable for both | — |
| `interpreter` | pickled both ways | a function from a module that imports neither NumPy nor sigflow; plain values in and out | Python 3.14+ |

**Why the main guard.** Process workers start from a fork server (a fresh
interpreter on Windows) and import the main module. A script that
builds and runs its graph at module level would start the graph again in every
worker; the run stops with an error saying so. Put the run under
`if __name__ == "__main__":`, or use `backend="loky"`.

**Sub-interpreters cannot load NumPy.** A block for `backend="interpreter"`
lives in a module of plain Python and is wrapped where the graph is built:

```python
# decoders.py: no numpy, no sigflow
def checksum(packet, *, mod=256):
    return sum(packet) % mod
```

```python
import decoders
from sigflow import SignalType, block, parallel

ck = block(decoders.checksum, inputs=SignalType(bytes, ()), outputs=SignalType(int, ()))
node = parallel(ck, workers=4, backend="interpreter")
```

`parallel` refuses a lambda, a function from the main script, or a module that
imports NumPy, when the graph is built. Data a worker cannot receive (a NumPy
array) or an import it cannot do stops the run with an error that explains it,
whatever the error policy.

## Workers, window and order

- **`workers`** is the pool size. For CPU-bound work, the number of cores is
  the ceiling; for blocks that wait (I/O), more can help.
- **`window`** bounds the frames in flight (default `2 * workers`). The node
  stops pulling input while the window is full, so memory stays bounded and
  back-pressure reaches the source. A larger window smooths out uneven frame
  costs; a smaller one keeps latency down.
- **Order** is the input order, always. A slow frame holds back the results
  behind it until it finishes, but not the work on them.

## Failures

- An exception in the block is an ordinary block failure: with the default
  `on_error="propagate"` that frame comes out failed and the others carry on;
  `"raise"` stops the run; `"drop"` drops the frame.
- A process worker that dies (a crash, `os._exit`, the main-guard problem) stops
  the run with a `RuntimeError` naming the likely cause.
- The pool is shut down when the node finishes, also when the run stops early.
  `loky`'s pool is reused by the next run, so a notebook pays the start-up once.

## Oversubscription

NumPy and SciPy call multi-threaded BLAS libraries. Four process workers that
each start eight BLAS threads ask for 32 threads on an 8-core machine, which is
slower than either alone. When the block is heavy linear algebra, limit the
library's threads in the workers, for example with `OMP_NUM_THREADS=1` (or
`threadpoolctl`), or use `backend="thread"` and let the library parallelise.

## Free-threaded Python

On a free-threaded build, `sigflow.pools.gil_enabled()` reports whether the GIL
is still off: importing an extension module that does not support free
threading turns it back on, and `backend="freethreaded"` then warns that its
threads will take turns. Generator stream blocks run in a worker thread there
instead of inline through greenlet, which crashes on those builds.

See also: [Performance](../performance.md) and the
[backends API](../reference/api/backends.md).

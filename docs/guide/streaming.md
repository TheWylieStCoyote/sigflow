# Streams and their lifecycle

A graph is a stream: sources produce frames until they end, or until you stop
consuming. This page is about choosing how to consume, stopping early, and who
owns which resources. `examples/30_stream_lifecycle.py` shows every case
running.

## Which entry point

| you want | use |
|---|---|
| every output of a graph that ends | `run(g)` |
| every output, without keeping them | `run(g, sink=fn, collect=False)` |
| the outputs one by one, or only the first few | `for f in g` (or `iterate_sync(g)`) |
| the same inside an event loop | `async for f in g` (or `iterate(g)`) |
| a graph that runs on while you do something else, and can be paused, stepped, probed, stopped | `h = g.start()` |

[Running graphs](../reference/running.md) lists the options of each.

## Stopping early

Leaving a `for` loop stops the graph: once the iterator is dropped, which
CPython does as the loop ends, the sources are closed (their `finally` blocks
run) and the helper thread is joined.

```python
for f in g:
    if f["seq"] >= 10:
        break  # the source is closed here
```

To close an iterator you keep around, close it explicitly, or let
`contextlib.closing` do it:

```python
from contextlib import closing

from sigflow import iterate_sync

with closing(iterate_sync(g)) as frames:
    first = next(frames)
# closed here, even if the block raised
```

**In async code, close explicitly.** An `async for` that breaks leaves the
async generator suspended: its sources stay open until the event loop shuts
down or the generator is garbage-collected. Use `contextlib.aclosing`. Closing
then runs asynchronously, one stage after another, so the sources are closed
within a few turns of the event loop after the `async with` block ends, not at
the moment it ends:

```python
from contextlib import aclosing

from sigflow import iterate

async with aclosing(iterate(g)) as frames:
    async for f in frames:
        if done(f):
            break
```

A background run stops with `h.stop()`, which waits up to five seconds for the
run to wind down, or `h.close()`, which also detaches its panel, live diagram,
probes and breakpoints. `sigflow.stop_all()` stops every run still going, and
`start(name=..., replace=True)` stops a previous run of the same name first,
which is what a re-run notebook cell wants.

## What stopping interrupts

Stopping takes effect **between frames**. A source is closed when it next
yields; a block finishes the frame it is computing. A call that is blocked
outside Python, such as a socket read or a `time.sleep` in a `threaded=True`
source, completes first: `stop()` returns once it does (or after five seconds,
with the run still winding down). Sources that wait on the network should use
a timeout, or be asynchronous, so they notice a stop promptly.

## Who owns what

| thing | owned by | released when |
|---|---|---|
| a source's generator and what it opened | the run | the source ends, the run stops, or the consumer stops early (its `finally` runs) |
| file sinks (`io.wav_sink`, `io.npy_sink`, `io.csv_log`) | **you** | you call `close()`, or leave its `with` block: that is when the file is written |
| socket sinks (`io.udp_sink`, `io.zmq_sink`) | **you** | `close()` or the `with` block |
| a `parallel()` pool | the node | the node finishes or the run stops (a `loky` pool is kept for the next run) |
| a `Run` and its thread | the run | the graph ends, `stop()`, `close()`, or `stop_all()` |
| a probe | the run | `probe.close()` or `run.close()` |

A file sink collects while the graph runs and writes on `close()`, so use it as
a context manager:

```python
from sigflow import dsp, io, run

with io.wav_sink("out.wav", rate=48_000) as sink:
    run(dsp.sine(n=1024, frames=48) >> dsp.scale(gain=0.5) >> sink)
# out.wav is written here
```

## Bounding memory

Three things can grow without bound in an endless stream; each has a bound:

- **Queued work.** The queues between nodes hold at most `queue_size` frames
  (default 64). A slow block makes everything upstream wait, so a fast source
  cannot run ahead and fill memory. `parallel(..., window=)` bounds the frames
  in flight in a pool the same way.
- **Collected output.** `run()` keeps every output frame. For an endless
  stream, pass `sink=fn, collect=False` and keep only what you need, or
  iterate.
- **A background run's buffer.** `start(keep=N)` keeps the newest *N* frames in
  `h.frames`; `h.drain()` takes them and empties the buffer.

```python
h = endless_graph.start(keep=100)
...
recent = h.drain()  # up to 100 frames; the buffer starts again empty
```

Lineage is the fourth: every frame references its parents. For very long runs
`provenance="roots"` keeps only the source frames (replay still works) and
`provenance="off"` keeps none.

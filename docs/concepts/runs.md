# Runs

Graphs are values; running one is explicit:

```python
frames = run(g)  # synchronously, returns the output frames
frames = await g.arun()  # from async code (a Jupyter cell)
for f in g:
    ...  # stream the outputs lazily
async for f in g:
    ...
r = g.start()  # in the background: a Run handle
```

Every way of running takes the same core options: `on_error=` (override every
block's policy), `check=`, `strict=True` (fail when fan-out / zip branches
produce different frame counts), `provenance=`, `queue_size=`, `labels=`,
`monitor=` and `resume=` (a checkpoint). `run()` and `start()` also take
`sink=fn` (called for every frame as it arrives), `collect=False` and
`progress=True`. Only `start()` keeps a bounded buffer,
`keep=N`, for endless streams; it also takes `paused=`, `name=` and
`replace=`. An option an entry point does not take is an error.
[Running graphs](../reference/running.md) has the full table.

## The scheduler

Every composition — operators, `feedback`, `when`, `align`, the `Graph`
builder — is lowered to one scheduler: members connected by queues. Linear runs
between joins execute as a single _chain_ without queues, and consecutive map
blocks inside a chain are fused into one frame wrap. Edges are bounded (back-
pressure); edges into joins are unbounded so a branch that changes cardinality
cannot deadlock the pump. A failure in any member is recorded once and
re-raised by the run (with the default policy it becomes a failed frame
instead). Synchronous generator stream blocks run inline on the event loop via
`greenlet` when it is installed, otherwise in a worker thread.

## Controlling a background run

`r.pause()`, `r.resume()`, `r.step(n)`, `r.stop()`, `r.wait()`, `r.set_rate(fps)`,
`r.snapshot()`, `r.set_recording(True)`, `r.checkpoint(path)`; `r.frames`,
`r.latest`, `r.drain()`, `r.state`, `r.monitor`. In a notebook `r.controls(...)`
shows the panel and `dashboard({...})` several.

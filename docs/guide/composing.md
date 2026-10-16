# Composing graphs

```python
a >> b  # chain
a | b  # fan-out: every frame to both; output is a tuple frame
a & b  # zip: lock-step join of two streams
merge(a, b)  # interleave sources as they arrive
align(a, b, tolerance=0.02)  # join by timestamp
(a | b | c)["rms"]  # select a port; port("vals") inside a fan-out
feedback(body, init)  # a loop: body maps (x, state) -> (y, new_state)
when(pred, then, otherwise)  # per-frame routing; a branch returning None drops
parallel(block, workers=4)  # a map block on several cores, order kept
g.take(n), g.skip(n), g.every(k), g.until(pred)  # stream slicing
```

Graphs are nodes: `front = dsp.window >> dsp.to_complex >> dsp.fft` is reusable
in any number of places (each use gets its own symbol scope), a source-less
graph is a function on in-memory data (`front(frames)`), and the same block
instance used twice is cloned automatically.

## When operators are not enough

```python
g = Graph("diamond")
cov = g.add(src >> linalg.covariance, "cov")
e = g.add(eigh)
s = g.add(summarize)
g.connect(cov, e).connect(e["vals"], s, "vals").connect(e["vecs"], s, "vecs").connect(
    cov, s, "raw"
)
g.output(s)
```

Edges are type-checked as they are made, cycles are rejected (use `feedback`),
a producer may feed many consumers, multi-input members receive their inputs
zipped. `g.input(type)` makes a source-less, reusable graph. `g.draw()` and
`g.report()` work on any graph.

## Cardinality and stream rates

Fan-out and zip pair outputs by count. Every node has a **ratio**: output frames
per input frame. Map blocks and taps are `1:1` by definition; stream blocks
declare theirs with `@block(ratio="N:1")`, `(4, 1)`, `"1:N"` or a callable of
the configuration and the input frame length (`dsp.chunk`, `overlap`, `buffer`
and `decimate` all do). Ratios multiply along a pipeline (`g.ratio`) and the
branches of a fan-out or zip must be compatible, or the graph refuses to build:

```python
sine(n=1024) >> (dsp.chunk(256) | dsp.rms)
# RatioMismatch: fan-out: chunk emits 1:4 frames per input but rms emits 1:1
```

A branch that only _bounds_ its rate (`N:1`, `take`, `skip`, `until`) pairs
with anything it could equal; an undeclared stream block is `any` and is
trusted. At run time `strict=True` checks that declared exact ratios were
honoured and that paired branches produced the same number of frames; without
it the run stops at the shortest branch. The monitor scales the sources'
totals by the graph ratio, so the expected frame count is right for a
re-framing chain.

## Keyed routing

`when(pred, then, otherwise)` chooses between two 1:1 branches per frame.
`switch(selector, {key: branch, ...})` generalises it: the selector is a
callable on the frame or the name of an annotation, each branch is a whole
pipeline that only ever sees the frames routed to it, and unselected branches
stay idle. Branches are real streams — they may window, buffer, filter with
state or drop — and run concurrently; their outputs merge in completion order
with provenance `switch(...)[key]`. `default=` takes a node, `"pass"` (forward
unchanged) or `None` (drop; `strict=True` raises on an unknown key). With
`ordered=True` output order follows input order and every branch must be 1:1.

```python
demod = switch(
    "modulation",
    {"bpsk": demod_bpsk, "qpsk": demod_qpsk, "carrier": track_carrier},
    default=log_unknown,
)
```

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

## What a Graph checks, and when

`Graph.connect` checks each edge against the consumer's input contract on the
spot: dtype, shape and symbol sizes, rate, units, and tags with their value
types; rates where branches meet, port names, and cycles. A refused connect
leaves the graph exactly as it was, nodes it added included, and earlier
edges are re-checked, so wiring in any order finds the same problems. A member
that is itself a pipeline is checked at its entry when connected and checked
through by `g.validate()`, which `run` does before any frame moves.
[Example 26](../examples/26_graph_checks.md) shows each case.

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

## One instance per key

`switch` needs its branches up front. When the keys are only known at run
time, such as the signals a detector finds in a spectrum, `per_key(key, body)`
makes one instance of `body` per distinct key, the first time the key is seen:

```python
tracked = detect >> per_key("track", per_signal, ttl=50)
```

Each instance is a real stream that only ever sees its own frames, so a body
with memory, a running average, a PLL, a `Stateful` object, keeps that memory
per signal. `key` is an annotation name or a callable on the frame; a frame
whose key is `None`, and a failed frame, pass through untouched. `ttl=50`
closes an instance after fifty input frames without its key, so a signal that
disappears releases its state; if the key comes back it gets a fresh
instance. Outputs merge in completion order, each with provenance
`per_key(...)[key]`, and a key's own frames keep their order. The detector
upstream is the natural place to assign keys: a stream block that yields one
frame per detection and tags it with a stable track id.

## Collecting the results back

`gather(by, ...)` is the mirror image: it folds the frames that share a key
back into one tuple frame, with the members as `parts`:

```python
grouped = detect >> per_key("track", per_signal) >> gather("seq", count="n_detections", order="track")
```

A group is released when it is complete: `count=` names an annotation (or
gives a number, or a callable) saying how many members it has; `lag=1`
releases a group when the next one starts; `ttl=` is the backstop after that
many frames without a new member, and the end of the stream releases what is
left. A group released early carries a warning issue, or is dropped with
`incomplete="drop"`. Members arrive in completion order; `order="track"` sorts
them and records the order in `meta["members"]`; `ordered=True` emits groups
in the order they started. A failed frame never reaches a stream block, so a
group missing one is released incomplete with a warning while the failed frame
travels on by itself. `gather` is an ordinary stream block, so a collector
with rules of its own is written the same way. The whole pattern, detector,
`per_key` and collector, with custom collectors for markers, windows and time,
is in [A variable number of signals](per-signal.md).

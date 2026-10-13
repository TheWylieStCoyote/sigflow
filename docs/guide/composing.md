# Composing graphs

```python
a >> b                 # chain
a | b                  # fan-out: every frame to both; output is a tuple frame
a & b                  # zip: lock-step join of two streams
merge(a, b)            # interleave sources as they arrive
align(a, b, tolerance=0.02)   # join by timestamp
(a | b | c)["rms"]     # select a port; port("vals") inside a fan-out
feedback(body, init)   # a loop: body maps (x, state) -> (y, new_state)
when(pred, then, otherwise)   # per-frame routing; a branch returning None drops
parallel(block, workers=4)    # a map block on several cores, order kept
g.take(n), g.skip(n), g.every(k), g.until(pred)   # stream slicing
```

Graphs are nodes: `front = dsp.window >> dsp.to_complex >> dsp.fft` is reusable
in any number of places (each use gets its own symbol scope), a source-less
graph is a function on in-memory data (`front(frames)`), and the same block
instance used twice is cloned automatically.

## When operators are not enough

```python
g = Graph("diamond")
cov = g.add(src >> linalg.covariance, "cov")
e   = g.add(eigh)
s   = g.add(summarize)
g.connect(cov, e).connect(e["vals"], s, "vals").connect(e["vecs"], s, "vecs").connect(cov, s, "raw")
g.output(s)
```

Edges are type-checked as they are made, cycles are rejected (use `feedback`),
a producer may feed many consumers, multi-input members receive their inputs
zipped. `g.input(type)` makes a source-less, reusable graph. `g.draw()` and
`g.report()` work on any graph.

## Cardinality

Fan-out and zip pair outputs by count and stop at the shortest branch. A
per-frame branch and a per-segment branch do not belong in one fan-out — chain
them (a tap keeps the frames it saw) or use `strict=True` to be told.

# Persistence, replay and checkpoints

`frame.prov.to_json()` / `Provenance.from_json()`; `frame.save("f.npz")` /
`Frame.load()`. `replay(graph, frame)` recomputes exactly that output: it checks
the sources are configured as recorded (`ReplayMismatch` otherwise) and returns
the frame whose lineage ends in the same source frames.

A block written as a `Stateful` object (`state()` / `restore()`, defaulting to
its `__dict__`) is checkpointable: `r.checkpoint("ck.pkl")` records source
positions and block states; `run(g, resume="ck.pkl")` / `g.start(resume=...)`
continues from there. Generator stream blocks cannot be captured and are listed
under `skipped`. Sources regenerate skipped frames without emitting them, so a
resume is cheap for file sources and exact for deterministic ones.

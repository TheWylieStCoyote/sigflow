# Changelog

## 0.3.0

New

- One runtime: operator trees (`>>`, `|`, `&`, `feedback`, `when`, `align`,
  slicing) are lowered to the `Graph` scheduler, which is the only executor.
  Linear runs of map blocks are fused into a single scheduler member; the
  output edge is iterated directly; errors and cancellation propagate through
  a shared failure future.
- Synchronous generator stream blocks run inline on the event loop when
  `greenlet` is installed (`pip install sigflow[perf]`), ~10x less overhead
  than the thread hop; `threaded=True` still forces a thread.
- `run(..., provenance="full" | "roots" | "off")` controls per-frame lineage
  bookkeeping; `"roots"` keeps replay and checkpoints working.
- `parallel(block, workers, backend="process" | "thread")`: a map block on a
  pool inside one graph, in order, with provenance and failed frames as usual.
  Blocks pickle by import path.
- `io.zmq` / `io.zmq_sink` (`pip install sigflow[zmq]`): PUB/SUB and
  PUSH/PULL, wire-compatible with GNU Radio's ZMQ blocks.
- `Lineage`: query provenance across a run — `failed`, `through(op, **params)`,
  `descendants_of`, `sources_of`, `blame`, `configs`, `diff`, `table`.
- Static typing: `Sig` / `Stream` / `Framed` have `TYPE_CHECKING` stubs so
  block bodies type-check as NumPy code; `py.typed`; `mypy src/sigflow` clean.
- Docs site (`mkdocs-material`): concepts, guides, generated example pages;
  `docs/build_docs.py`; CI builds it with `--strict`.

Changed

- `plot` taps: the `every` parameter is now `stride`.
- Benchmarks refreshed (`docs/performance.md`); README gained sections on the
  scheduler, `parallel`, lineage queries, typing and the docs site.

## 0.2.0

New

- `Graph` builder (`sigflow.builder`): explicit DAGs with `add` / `connect` (by
  position or input name, with port selection on edges) / `output`; edge-time
  type checking, cycle rejection, layered drawing, reports.
- Slicing: `node[a:b]` on array-typed nodes with static shape propagation and
  sample annotations / time sliced in step; `frame[a:b]`; stream slicing
  `take` / `skip` / `every` / `until` / `while_` (identity-typed).
- Time axis: `Frame.time` (`TimeSpan`), `frame.at(t)`, `frame.between()`;
  re-framing blocks keep exact `t0`/`rate`; `align()` joins streams by time.
- Units: `sigflow.units` registry, compatibility in unification with a
  conversion hint, `dsp.to_units`.
- Persistence: provenance to/from JSON, `frame.save` / `Frame.load`,
  `replay(graph, frame)`, `Stateful` blocks, `run.checkpoint()` and
  `resume=` on `run` / `start`.
- Plotting: `plot.history`, `plot.multi`, ipympl-native live figures.
- `run(..., strict=True)` raises on fan-out / zip cardinality mismatch.
- `bench/bench.py` and `docs/performance.md`; GitHub Actions CI matrix
  (Python 3.10–3.13, NumPy 1.x/2.x, Linux/macOS/Windows); `py.typed` and a
  clean `mypy` run (block bodies excluded: `Sig[...]` is a runtime DSL).

Changed

- Positional arguments to a block definition: arrays / Frames / lists call the
  function; strings, numbers and other scalars are configuration
  (`to_units("V")` makes a node). `block.fn(...)` always calls the function.
- A block that returns a `Frame` always gets fresh provenance for the node.
- `requires-python` lowered to 3.10.

Fixed

- Overlap/chunk windows carried the input chunk's `t0`; now exact.
- `step()` before the run loop attached could leave a paused run stuck.

## 0.1.0

Initial version: typed blocks, operators, asyncio runtime with sync adapters,
annotations and error propagation, provenance, plotting, notebook controls,
ports, `feedback` / `when`, `sigflow.io`, examples 01–19.

# Changelog

## Unreleased

Changed

- Static typing: the package-wide mypy error-code exemptions are replaced by
  per-module overrides for the DSL-heavy modules (`dsp`, `linalg`, `io`,
  `plot`, `types`); everything else is checked in full, and a `[tool.pyright]`
  configuration is added (`pyright src` joins `mypy src/sigflow` in CI).
  `Frame.save` / `Frame.load` and `Provenance.to_dict` / `to_json` /
  `from_dict` / `from_json` are declared on the classes instead of attached at
  import time, so editors see them. The `draw(ax, data, frame)` callback of
  `plot.custom` may return a value.

## 0.4.0

New

- Stream-rate contracts (`sigflow.ratio`): `@block(ratio=...)` declares frames
  out per frame in (`"1:1"`, `"N:1"`, `"1:N"`, `(4, 1)`, or a callable of the
  configuration and `n_in`); map blocks are `1:1` by definition. Ratios
  multiply along pipelines (`node.ratio`, `describe()` tags), fan-out / zip
  branches and `Graph` edges at incompatible rates raise `RatioMismatch` at
  build time, `run(strict=True)` verifies declared ratios, and the progress
  monitor scales source totals by the graph ratio. `dsp.chunk`, `overlap`,
  `buffer`, `decimate`, `running_mean`, `fir`, `timestamp` declare theirs.
- `switch(selector, {key: branch, ...}, default=..., ordered=...)`: keyed
  routing through real stream branches (windowing, state, drops allowed) that
  run concurrently; `default` is a node, `"pass"` or drop; `ordered=True`
  keeps input order for 1:1 branches; provenance names the branch taken.
- Example 21 (modulation-switched demodulators, rate contracts).

Changed

- A fan-out whose branches have different _declared_ rates (for example
  `dsp.rms | dsp.buffer(count=2)`) is now a build-time error rather than a
  run-time `strict=True` one.

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

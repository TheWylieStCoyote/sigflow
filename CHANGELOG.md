# Changelog

## Unreleased

New

- `sigflow.dims`: dimension and dtype-variable symbols (`N`, `M`, `T`, the
  whole alphabet; also exported from `sigflow`). `Sig[float, N]` builds the
  same contract as `Sig[float, "N"]`, which keeps working, but checkers read
  `N` as a type instead of an undefined name.
- Checker-readable spellings for the rest of the DSL, all equivalent at
  runtime to the old ones: shapes `tuple[M, N]`, `tuple[()]`,
  `tuple[Literal[1024]]`; rate and units `Literal[48_000]`, `Literal["V"]`;
  several outputs `tuple[Sig[...], Sig[...]]`; generator blocks
  `Iterator[Sig[...]]` / `AsyncIterator[...]`; a type built from configuration
  (`refine=`) as a call, `SignalType(float, (n,), rate)`.
- `NO_PROV`: every frame carries a `Provenance`; the empty one is falsy and
  renders as `(no provenance)`.
- Labelled arrays (`sigflow[xarray]`): an `xarray.DataArray` can be a frame's
  payload. `@block(labels=True)` lines the array up with the input contract
  (by dimension *name* when the contract names every dimension, e.g.
  `Sig[float, tuple[Literal["channel"], Literal["time"]]]`; by position
  otherwise), hands the body the bare array (`da.data`, so CuPy stays CuPy)
  and rebuilds a labelled array around each output from the output contract:
  a dimension variable seen on an input keeps its name and coordinate, new
  ones get their own name, fixed sizes become `dim_<axis>`. Runtime checks
  read a labelled array's sizes by name. A block with an unset `labels`
  option follows the run: `run(g, labels=True)` switches every such block on,
  including the built-in `dsp` / `linalg` blocks, and
  `sigflow.default_labels(True)` sets the package default a run reads once
  when it starts (so the choice is per run and reaches process-pool workers);
  `labels=False` opts out. `Framed` inputs and `DataArray` results pass
  through untouched. A stream block keeps only the item it is processing,
  and that item's own labels: dimension names, coordinates, name and
  attributes follow the latest labelled item, a plain-array item gets none,
  and the block's static inputs keep theirs.
- `@block(preserve_type=True)`: a single-input map block that carries its
  producer's complete contract (`types.narrow`) as the graph has pinned it
  down so far (`types.resolve`), so shape, rate and unit checks and reframing
  ratios see through it, in pipelines, reused subgraphs and `Graph` wiring in
  any order. The carried contract is merged axis by axis: a declared `(4,)`
  survives a producer that only says `(None,)`, repeated variables keep their
  equality across tuple elements, and required keys from both sides are kept.
- `Graph.connect` is transactional: any failure (type or rate mismatch, cycle,
  bad port name or index) restores the members, edges, bindings and naming
  counter, including nodes the failed call added implicitly.
- CPU / CUDA transfers (`sigflow.to_device`, `sigflow.to_host`; CuPy is
  optional): explicit moves of a frame's payload that keep dtype, shape,
  annotations and, for a `DataArray`, its labels, and carry the producer's
  contract (`preserve_type`). A transfer is a barrier: the source device is
  synchronised before its data is read, so work a producer launched on any
  stream is complete, and the copy is finished before the block returns.
  Device arrays of other libraries (`__cuda_array_interface__`) are accepted.
  A manually triggered GPU CI workflow (`.github/workflows/gpu.yml`) runs the
  device and xarray tests on a self-hosted runner; the docs gained a backend
  support matrix (**Guide → Array backends and GPU**).
- `Frame.slice` / `frame[a:b]` / `at` / `between` index array-like payloads
  directly, so labelled and device arrays keep their labels and placement;
  `Frame.save` / `Frame.load` round-trip a `DataArray` (values, dims, coords,
  attrs, name) and fetch device arrays to the host.

Changed

- Declared rates and units are checked when a graph is built even when the
  produced shape is unknown; before, an unknown shape skipped those checks
  until run time.
- `Frame.prov` and `Issue.prov` are `Provenance`, never `None` (passing `None`
  still works and stores `NO_PROV`); test with `if f.prov:` rather than
  `is None`.
- Under `TYPE_CHECKING`, `Sig` is a variadic type alias to `Any` rather than an
  `ndarray` subclass: block bodies are no longer checked as NumPy code, and
  returning an array, scalar or dict for a `Sig` output is no longer an error.
- The package, tests, examples, notebooks and docs use the new spellings. No
  mypy error codes are disabled anywhere; `mypy src/sigflow tests examples`
  and `pyright` (standard mode) gate CI. `Frame.save` / `Frame.load` and
  `Provenance.to_dict` / `to_json` / `from_dict` / `from_json` are declared on
  the classes instead of attached at import time, so editors see them. The
  `draw(ax, data, frame)` callback of `plot.custom` may return a value.

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

# Array backends and GPU

Sigflow keeps three things apart:

- **Signal contracts** describe dtype, shape, rate and units. They say nothing
  about where the samples live.
- **Array storage** is whichever library produced the payload: NumPy on the
  CPU, CuPy on a CUDA device, or a lazy dask array. A block body sees that
  array and must be written for it.
- **Labels** (dimension names and coordinates) travel in an `xarray.DataArray`
  around the payload. A label-aware block lines the array up with its contract,
  hands the body the bare payload and rebuilds the labels afterwards, whatever
  library the payload comes from (see [Labelled arrays](xarray.md)).

Accepting an array as a payload does not make the NumPy and SciPy blocks work
on it. Choose blocks written for the backend you use and make transfers
explicit.

## Support matrix

| Operation | NumPy CPU | CuPy CUDA | xarray DataArray | dask-backed data |
|---|---|---|---|---|
| Graph input and frame payload | Supported | Supported, never converted to the host implicitly | One DataArray is one frame, subclasses included | Accepted without computing samples |
| Dtype and shape contract checks | Supported | Metadata only, no transfer | Metadata only; a contract that names every dimension is matched by name | Metadata only; fixed dimensions need concrete sizes |
| Map and stream block bodies | NumPy and SciPy blocks | CuPy-compatible functions | Label-aware blocks get `da.data`: the CuPy or dask array stays what it is | Functions that accept lazy arrays |
| `to_device` / `to_host` | Upload, or reuse / copy on the host | Barrier transfers, also between devices | The payload moves, labels stay, subclasses are kept | Not a lazy transfer: materialise first |
| `Frame.slice`, `at`, `between`, `frame[a:b]` | Supported | Indexes the device array directly | Labels and coordinates survive | Lazy slices |
| `chunk`, `overlap`, `buffer` and other reframing | Supported | Transfer to the host first | Labels are dropped by NumPy-based reframing; reframe before labelling, or after `to_host` | May compute samples |
| Plotting and file sinks | Supported | Transfer to the host first | Numeric paths read the values | Materialises |
| `Frame.save` / `Frame.load` | Supported | Fetched to the host when saved | Values, dims, coords, attrs and name round-trip | Materialises |
| `parallel(..., backend="process")` | Picklable blocks | Not a placement or scheduling abstraction | Labelled payloads pickle like arrays; the run's `labels` choice reaches the workers | Not a substitute for dask scheduling |

For blocks with several operands, give them matching backends: nothing uploads
or downloads an operand to resolve a mixed expression. Device arrays of other
libraries (anything with `__cuda_array_interface__`: torch, numba, ...) are
accepted by the transfer blocks and become CuPy arrays on the way.

## Explicit transfers

```python
from sigflow import dsp, to_device, to_host

# gpu_processing is a source-less graph of CuPy-compatible map blocks.
# graph = dsp.sine(n=16384) >> to_device(device="cuda:0") >> gpu_processing >> to_host()
```

`to_device(device=0)` takes a nonnegative index or `"cuda:N"`. An array already
on that device is reused unless `copy=True`. `to_host()` reuses a CPU array
unless `copy=True`; a device array becomes a NumPy array.

For a DataArray, the payload moves while dimensions, coordinates, name and
attributes stay. Coordinates and frame sample annotations keep their existing
placement. `Dataset` and `Variable` payloads are rejected.

Both blocks are `preserve_type` blocks (see [Blocks](../concepts/blocks.md)):
placed after a producer they carry its complete contract as the graph has
pinned it down, so a 1024-sample source followed by `to_host()` and
`dsp.chunk(256)` still has ratio `1:4`, also with generic blocks such as
`dsp.scale` between the source and the transfer, and a consumer with the wrong
shape is rejected when the graph is built. This holds in operator pipelines,
reused subgraphs and explicit `Graph` wiring in any order.

## Synchronisation

A transfer is a barrier, and the order of events is fixed:

1. For a device array, synchronise **the whole source device** before reading
   it. A producer may have launched its kernels on any stream, so a device-wide
   wait is the only one that is safe without knowing the stream.
2. An array already on the requested device (and `copy=False`) is now complete:
   it is returned as it is.
3. Otherwise copy to the destination (host, or the target device) and wait for
   that copy to finish.

A custom producer can therefore launch work on a non-blocking stream and return
its CuPy array before the work is done: the next transfer waits for it. The
device-wide wait also waits for unrelated kernels on that device, and no
transfer is overlapped with computation. Group several operations on one device
between transfers, and keep per-frame transfers out of hot loops.

Between GPU blocks the ordinary CuPy stream rules apply. A block that switches
streams must record the events another block needs before the next block reads
its output; the scheduler does not track CUDA events or order arbitrary
streams. Nor can a barrier protect an array that another thread mutates after
the transfer: do not mutate published payloads.

Per-block timings measure Python execution, including these barriers; they are
not kernel timings. Use CUDA events to profile GPU work.

## Examples

- [CPU/GPU spectrum](../examples/22_cpu_gpu.md): NumPy preprocessing, CuPy
  FFT and power, explicit transfers and CPU peak detection.
- [Labelled arrays](../examples/23_xarray_blocks.md): named contracts, and a
  CuPy-backed DataArray in the guide.

```bash
python examples/22_cpu_gpu.py
python examples/22_cpu_gpu.py --device 1
python examples/23_xarray_blocks.py
```

The CUDA example skips its GPU section when no device is available and
compares the GPU result with a NumPy reference when there is one.

## Testing and CI

The ordinary CI job installs xarray and `dask[array]`, so the lazy-array paths
are exercised rather than skipped. `pip install -e ".[dev]"` gives the same set
locally; both stay optional at runtime.

```bash
python -m pytest -q tests/test_device.py tests/test_xarray.py
```

CPU tests use a CuPy stand-in that records every synchronisation, so transfer
sequencing and device-context restoration are checked without CUDA. With a
device present the same files also run numeric round trips, reuse and copy
behaviour, DataArray subclasses, and a kernel launched on a non-blocking stream
whose result a transfer must wait for. Cross-device cases run with two or more
GPUs.

`.github/workflows/gpu.yml` is a manually triggered **GPU integration**
workflow. Register a self-hosted Linux x64 runner with the `gpu` label, a CUDA
device, a working driver and the runtime CuPy needs; then run the workflow and
pick `cupy-cuda12x` or `cupy-cuda13x` to match the runner. The job builds an
isolated virtual environment, requires a working device before testing (a
missing GPU fails the run instead of skipping everything), and runs the device
and xarray tests plus the two examples. Pull-request CI stays on hosted CPU
runners.

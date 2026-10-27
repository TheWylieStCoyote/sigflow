# Adapting functions to xarray

`xarray_aware` lets a numerical function accept labeled `xarray.DataArray`
inputs without putting xarray logic in its body. It maps named dimensions to
array axes, calls the function, and reconstructs labeled outputs using
`xarray.apply_ufunc`.

Use it for synchronous map functions: same-size transformations, reductions,
and functions returning several arrays. The wrapped function remains directly
callable with ordinary arrays and can become a sigflow block with `@block`.

## Install

```bash
python -m pip install -e ".[xarray]"
```

Xarray is optional. Importing sigflow or calling the decorated function with
ordinary arrays does not import xarray. If xarray is already installed, a normal
sigflow installation is sufficient.

## A complete first example

Put `@block` **above** `@xarray_aware`. This lets sigflow inspect the original
signature and distinguish signal inputs from configuration parameters.

```python
import numpy as np
import xarray as xr

from sigflow import Sig, block, xarray_aware


@block
@xarray_aware(input_core_dims=[["time"]], output_core_dims=[["time"]])
def remove_dc(x: Sig[float], gain: float = 1.0) -> Sig[float]:
    return gain * (x - np.mean(x, axis=-1, keepdims=True))


data = xr.DataArray(
    np.arange(16.0).reshape(8, 2),
    dims=("time", "channel"),
    coords={"time": np.arange(8) / 1000, "channel": ["left", "right"]},
    name="voltage",
    attrs={"units": "V"},
)

centered = remove_dc(data, gain=2)
assert centered.dims == ("time", "channel")
xr.testing.assert_equal(centered.time, data.time)
np.testing.assert_allclose(centered.mean("time"), 0)
```

The first `len(input_core_dims)` positional parameters are treated as array
inputs. All remaining parameters are forwarded as configuration, including
keyword-only parameters and array-valued configuration such as filter weights.
Keep array inputs before configuration in the function signature.

`xarray_aware` can also be used alone on a regular Python function; `@block` is
needed only for graph composition.

## Core dimensions and batch axes

A **core dimension** is an axis that one invocation of the numerical operation
consumes. Other dimensions broadcast as batch axes. Xarray moves core dimensions
to trailing axes before calling the function:

| Labeled input | Core dimensions | Array received by function |
|---|---|---|
| `(time,)` | `time` | `(time,)` |
| `(time, channel)` | `time` | `(channel, time)` |
| `(channel, time)` | `time` | `(channel, time)` |
| `(sensor, time, channel)` | `time` | `(sensor, channel, time)` |

Write batched functions using `axis=-1` for one core dimension, as in
`remove_dc`. Calling `np.mean(x)` without an axis would mix channels together.
For multiple core dimensions, trailing axes follow their declared order.

After computation, the adapter restores the surviving dimension order of the
first labeled input, followed by dimensions introduced by other labeled inputs.
It does not leave a same-size result in the temporary trailing-axis layout.

```python
transposed = data.transpose("channel", "time")
xr.testing.assert_identical(
    remove_dc(transposed, gain=2),
    centered.transpose("channel", "time"),
)
```

With ordinary arrays, no axis movement or vectorization is performed. The
original function is called directly, so the caller must arrange the axes as the
function expects:

```python
plain = np.arange(8.0)
np.testing.assert_allclose(remove_dc(plain), plain - plain.mean())
```

## Reductions and graph composition

Use an empty output core-dimension list, `[[]]`, when an operation reduces its
core dimensions. Broadcast dimensions still survive: reducing `time` in a
two-channel signal returns one result per channel, not one scalar overall.

```python
@block
@xarray_aware(input_core_dims=[["time"]], output_core_dims=[[]])
def rms(x: Sig[float]) -> Sig[float]:
    return np.sqrt(np.mean(x**2, axis=-1))


graph = remove_dc(gain=2) >> rms
frames = graph(data, check="all", on_error="raise")
result = frames[0].data
assert result.dims == ("channel",)
assert result.channel.values.tolist() == ["left", "right"]
print(result)
print(frames[0].trace())
```

`graph(data)` treats the whole DataArray as one input frame. A list of
DataArrays supplies several frames. Frame annotations, issues and provenance
use the normal graph runtime; xarray coordinates and attributes belong to the
payload, not to those frame annotation dictionaries.

### Signal contracts describe the whole payload

`Sig[float]` checks dtype without specifying rank. That is useful for an operation
with one core dimension and any number of batch dimensions.

`Sig[float, N]` still declares a **one-dimensional payload**. It does not mean
"one named core dimension plus arbitrary batch axes." A two-dimensional
DataArray violates that contract even if the numerical function supports it.
Use a full positional shape when it is known, or leave shape gradual as above.
Named-dimension validation is performed by xarray when the block executes;
sigflow's build-time contracts remain positional.

## Multiple inputs and coordinate alignment

Declare one input dimension list per array parameter:

```python
@block
@xarray_aware(
    input_core_dims=[["time"], ["time"]],
    output_core_dims=[["time"]],
)
def add(x: Sig[float], y: Sig[float], gain: float = 1.0) -> Sig[float]:
    return x + gain * y


combined = add(data, data, gain=0.5)
np.testing.assert_allclose(combined.data, data.data * 1.5)
```

The default `join="exact"` requires matching indexes on shared dimensions.
For example, passing `data` and `data.assign_coords(time=data.time + 1)` raises
an alignment error rather than silently choosing common timestamps. Inputs with
unindexed dimensions still follow xarray's size and broadcasting rules; exact
alignment is not a check that all attributes or auxiliary coordinates agree.

Explicit alternatives are `"inner"`, `"outer"`, `"left"`, and `"right"`.
An inner join can remove samples; an outer join can introduce missing values.
Choose one only when that behavior is intended. The graph's frame pairing and
xarray's sample-coordinate alignment are separate operations.

## Multiple outputs

Declare one output dimension list for each element of a returned tuple:

```python
@block
@xarray_aware(input_core_dims=[["time"]], output_core_dims=[[], []])
def statistics(x: Sig[float]) -> tuple[Sig[float], Sig[float]]:
    return np.mean(x, axis=-1), np.max(x, axis=-1)


mean, peak = statistics(data)
assert mean.dims == peak.dims == ("channel",)
```

In a graph these remain tuple outputs, so normal port selection such as
`statistics[0]` applies. Each output retains only coordinates applicable to its
remaining dimensions.

## Functions that accept only one vector

Set `vectorize=True` to adapt an eager NumPy function that cannot handle leading
batch axes. It is called separately for each core slice:

```python
@xarray_aware(
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    vectorize=True,
)
def smooth(x):
    return np.convolve(x, np.ones(3) / 3, mode="same")


smoothed = smooth(data)
assert smoothed.dims == data.dims
```

This is convenient, but generally slower than a batched implementation.
Vectorization is not supported for labeled CUDA inputs or with
`dask="allowed"`. A plain-array call still goes directly to the function.

## Attributes, coordinates and timing

Applicable coordinates and the name follow `apply_ufunc` behavior. Output names
may be lost when multiple inputs have conflicting names. Attributes are kept by
default; `keep_attrs=False` drops them. Keeping attributes does not convert units:
an operation that squares voltage should not blindly retain `units="V"`.

Core dimensions that survive must retain their size. Reductions remove
coordinates that depend on the reduced dimensions. Generating new coordinates,
such as an FFT frequency axis or a resampled time axis, requires a separate
coordinate-aware wrapper.

The decorator does not infer `Frame` annotations such as `rate` or `t0` from
coordinates, update them after an operation, or carry filter state between
frames. Stream reframing and stateful filtering have separate responsibilities.

## GPU-backed DataArrays

The adapter does not call `np.asarray` or move sample data between devices.
Use a GPU-compatible numerical function and explicit transfer blocks. This
snippet requires CuPy and a usable CUDA device:

```python
import cupy as cp
from sigflow import to_device, to_host


@block
@xarray_aware(input_core_dims=[["time"]], output_core_dims=[["time"]])
def gpu_remove_dc(x: Sig[float]) -> Sig[float]:
    return x - cp.mean(x, axis=-1, keepdims=True)


gpu_graph = to_device(device=0) >> gpu_remove_dc >> to_host()
host_result = gpu_graph(data, check="all", on_error="raise")[0].data
assert isinstance(host_result.data, np.ndarray)
np.testing.assert_allclose(host_result.data, remove_dc(data).data)
```

Coordinates retain their existing placement during transfers. NumPy/SciPy
functions are not automatically GPU-compatible just because they are decorated.

## Lazy arrays

The default `dask="forbidden"` rejects Dask-backed labeled input before numerical
execution. Set `dask="allowed"` only if the function itself supports Dask arrays,
for example one using their native `.mean(axis=-1)` operation. The adapter then
passes the lazy array through; a function that explicitly materializes it can
still trigger computation. This option does not build a parallelized generalized
ufunc or automatically rechunk a core dimension.

## Decorator options

| Option | Default | Meaning |
|---|---|---|
| `input_core_dims` | Required | One list of named core dimensions per leading array parameter; `[[]]` for one elementwise input. |
| `output_core_dims` | Required | One list per output; `[[]]` for a reduction over the input core dimensions. |
| `vectorize` | `False` | Call an eager NumPy-only function separately for each core slice. |
| `keep_attrs` | `True` | Retain attributes according to xarray's keep-attributes behavior. |
| `join` | `"exact"` | Coordinate alignment: `exact`, `inner`, `outer`, `left`, or `right`. |
| `dask` | `"forbidden"` | Reject chunked labeled input, or delegate to a compatible function with `"allowed"`. |

## Troubleshooting

| Symptom | What to check |
|---|---|
| `place @block above @xarray_aware` | Reverse the decorator order: the xarray adapter wraps the function first. |
| A two-dimensional payload fails a rank check | The `Sig` annotation describes the whole array; use a matching shape or `Sig[float]`. |
| Missing `time` dimension | The declared core dimension must match an input dimension name. Rename explicitly or change the declaration. |
| Coordinate alignment fails | Check timestamps and channel labels. Align or interpolate explicitly when needed. |
| An output dimension changed size | Same-size core outputs and reductions are supported; resizing needs another adapter. |
| A new `frequency` dimension is rejected | New output dimensions and coordinate-construction hooks are not supported by this decorator. |
| Dataset or Variable input is rejected | Select a DataArray, such as `dataset["voltage"]`. |
| A generator or async function is rejected | This decorator supports synchronous array functions used as map blocks. |
| A failed frame appears instead of an exception | Use `on_error="raise"` while debugging; the normal graph error policy applies. |

## Further examples

- [Runnable xarray blocks example](../examples/23_xarray_blocks.md)
- [CPU/GPU processing example](../examples/22_cpu_gpu.md)
- [Block conventions](../concepts/blocks.md)
- [Signal contracts](../concepts/contracts.md)

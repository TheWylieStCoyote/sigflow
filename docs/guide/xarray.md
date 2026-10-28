# Labelled arrays with xarray

An `xarray.DataArray` can be the payload of a frame. A block that is
*label-aware* lines the array up with its input contract, hands the body the
bare array, and rebuilds a labelled array around the result from its output
contract. The body stays a NumPy (or CuPy) function; the contract is the only
place dimensions are named.

```python
from typing import Literal
import numpy as np
import xarray as xr
from sigflow import Sig, block

CHANNEL = Literal["channel"]
TIME = Literal["time"]


@block(labels=True)
def remove_dc(
    x: Sig[float, tuple[CHANNEL, TIME]], gain: float = 1.0
) -> Sig[float, tuple[CHANNEL, TIME]]:
    return gain * (x - np.mean(x, axis=-1, keepdims=True))


data = xr.DataArray(
    np.arange(16.0).reshape(8, 2),
    dims=("time", "channel"),
    coords={"time": np.arange(8) / 1000, "channel": ["left", "right"]},
    name="voltage",
    attrs={"units": "V"},
)

centered = remove_dc(data, gain=2)
assert centered.dims == ("time", "channel")  # the caller's layout
xr.testing.assert_equal(centered.time, data.time)  # coordinates survive
assert centered.attrs == {"units": "V"} and centered.name == "voltage"
```

Inside `remove_dc` the array is `(channel, time)`, because that is what the
contract says, whichever way round the caller stored it.

## Install

```bash
python -m pip install -e ".[xarray]"
```

xarray is optional. Nothing in sigflow imports it; a `DataArray` can only
arrive once your own code has imported xarray, and a call with plain arrays
goes straight to the body.

## Turning it on

- `@block(labels=True)` makes one block label-aware; `labels=False` opts a
  block out, so its body receives the `DataArray` itself.
- `run(graph, labels=True)` makes every block whose `labels` option is unset
  label-aware for that run, including the built-in `dsp`, `linalg` and `plot`
  blocks (`iterate`, `arun` and `graph(data, labels=True)` take it too).
- `sigflow.default_labels(True)` sets the package default for runs that do not
  pass `labels=` and for blocks called directly. It returns the previous
  setting; the default is off in this release.

```python
from sigflow import dsp, from_iterable, run

frame = run(from_iterable([data.isel(channel=0)]) >> dsp.scale(gain=0.5), labels=True)[
    0
]
assert frame.data.dims == ("time",)
```

A run reads the default once, when it starts, and the resolved choice travels
with the run: into its threads and its `parallel(..., backend="process")`
workers. Switching the default in another thread or cell while a run is in
progress does not change what that run's blocks receive.

## How axes are matched

**By name**, when every dimension of the contract names a dimension of the
array. `Sig[float, tuple[CHANNEL, TIME]]` accepts `(time, channel)` or
`(channel, time)` and transposes to `(channel, time)` for the body. Runtime
checks (`check="all"`) read sizes by name too, so a named contract never fails
because of layout.

**By position** otherwise. `Sig[T, tuple[M, N]]` takes the array as it is and
binds `M` and `N` to the dimension names it finds there, just as it binds them
to lengths.

Spell a named dimension as `Literal["channel"]` so type checkers read it as a
type; at runtime it is the string `"channel"`, the same dimension variable you
would write as `Sig[float, ("channel", "time")]`.

## How outputs are labelled

Each output is rebuilt from the output contract:

| Output contract entry | Dimension name | Coordinate |
|---|---|---|
| a variable seen on an input (`N` was `time`) | `time` | copied if the size is unchanged |
| a variable not seen on any input | its own name (`K`) | none |
| a fixed size (`Literal[513]`) | `dim_<axis>` | none |

A reduction `tuple[M, N] -> tuple[M]` keeps `M`'s name and coordinate and drops
`N`'s. A new axis, such as an FFT bin axis, appears without coordinates; add
them in the body by returning a `DataArray` yourself, which is passed through
untouched. Attributes and the array name come from the first labelled input.

A contract without a shape (`Sig[float]`) keeps the input's dimensions when the
rank is unchanged and returns a plain array otherwise. Scalars (rank 0) are
returned as they are, so `dsp.rms` still yields a `float`. Tuple outputs and
tuple frames are handled element by element.

When the output contract names every dimension, the result is transposed back
to the caller's layout. A positional contract keeps the contract's axis order,
which is what its runtime checks expect.

## Several inputs

Each input follows its own contract, so a labelled array and a plain one can
meet in the same block. The plain array is taken in contract order; the
labelled one is lined up with it:

```python
@block(labels=True)
def add(
    x: Sig[float, tuple[CHANNEL, TIME]], y: Sig[float, tuple[CHANNEL, TIME]]
) -> Sig[float, tuple[CHANNEL, TIME]]:
    return x + y


weights = np.ones((2, 8))  # (channel, time), as the contract says
out = run(from_iterable([(data, weights)], type=add.in_type) >> add)[0].data
assert out.dims == ("time", "channel")
```

Coordinates are not aligned: two labelled inputs are combined by position after
each is lined up with its contract. Align or interpolate before the graph when
their coordinates differ.

## Stream blocks and bodies that want the labels

A generator block gets its items unwrapped one by one; the labels used for its
outputs are those of the item consumed most recently, which is right for
one-in-one-out generators and worth knowing for buffering ones. Each item
brings its own labels: dimension names, layout, coordinates, name and
attributes follow the latest labelled item, a plain-array item gets none, and
the block's other, non-stream inputs keep theirs. Sync and async generators
behave the same.

A body that needs the labels takes the frame instead of the array:

```python
from sigflow import Framed


@block(labels=True)
def left_only(f: Framed[Sig[float]]) -> Sig[float]:
    return f.data.sel(channel="left").values
```

`Framed` inputs are passed untouched, and so is a `DataArray` returned by a
body, so a block may also do its own labelling.

## Frames, slicing and files

`Frame.slice`, `frame[10:20]`, `at` and `between` index the payload directly,
so a labelled frame stays labelled and `t0` still advances. `Frame.save` writes
the values, dimensions, coordinates, attributes and name into the `.npz`
record; `Frame.load` rebuilds the `DataArray` when xarray is installed and
returns the bare array, with a warning, when it is not.

## GPU-backed arrays

The body receives `da.data`, not `da.values`, so a CuPy-backed `DataArray`
reaches a label-aware block as a CuPy array and comes back labelled and still
on the device. `to_device` and `to_host` move only the payload and carry the
producer's contract through. Write the body against the array library you
expect, as with any other block. The support matrix and the synchronisation
rules are in [Array backends and GPU](array-backends.md).

## Troubleshooting

| Symptom | What to check |
|---|---|
| The body received a `DataArray` | The block is not label-aware: add `labels=True` or switch `default_labels(True)`. |
| `dimension 'channel' is 8 but here needs 3` at run time | A positional contract met a differently laid out array; name the dimensions in the contract. |
| The same message while *building* the graph | The source's declared type is positional (`tuple[Literal[8], Literal[3]]`), which means what it says for a plain array: eight channels. Declare the source by name too (`type=Sig[float, tuple[TIME, CHANNEL]]`) or leave its shape open; static checks match named shapes by position. |
| A reduction came back as a plain array | The contract has no shape and the rank changed; declare the output shape. |
| New axis is called `dim_1` | Fixed sizes have no name; use a dimension variable, or return a `DataArray` from the body. |
| Labels lost after a block | The block opted out (`labels=False`), or the body returned a Python scalar. |
| A `Dataset` is rejected | Select one variable: `dataset["voltage"]`. |

## Further examples

- [Runnable xarray blocks example](../examples/23_xarray_blocks.md)
- [CPU/GPU processing example](../examples/22_cpu_gpu.md)
- [Block conventions](../concepts/blocks.md)
- [Signal contracts](../concepts/contracts.md)

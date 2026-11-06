# Installation

sigflow needs **Python 3.11 or later**. The base install brings NumPy, SciPy
and matplotlib, which is everything the core library, the DSP and linear-algebra
blocks and the plots use.

```bash
python -m pip install -e .            # from a checkout
```

Optional features are extras: install only the ones you use, for example
`python -m pip install -e ".[parallel,xarray]"`.

| extra | brings | what it is for |
|---|---|---|
| `perf` | greenlet | generator stream blocks run inline on the event loop, much faster per frame (not on free-threaded builds, where greenlet crashes) |
| `parallel` | loky | `parallel(block, backend="loky")`: a process pool that needs no main guard and takes blocks defined in a notebook |
| `xarray` | xarray | labelled `DataArray` payloads and `@block(labels=True)` |
| `notebook` | ipython, jupyter, ipywidgets | the control panel, knob widgets, live diagrams in Jupyter |
| `zmq` | pyzmq | `io.zmq` sources and sinks (GNU Radio's ZMQ blocks) |
| `cupy` | CuPy (built from source) | GPU payloads with `to_device` / `to_host`; or install a prebuilt `cupy-cuda12x` wheel instead |
| `test` | pytest | running the test suite |
| `docs` | mkdocs-material | building this site |
| `dev` | the test, type-checking and notebook tools | working on sigflow itself; see `CONTRIBUTING.md` |

Two more backends of `parallel()` need a particular interpreter rather than an
extra: `backend="interpreter"` needs Python 3.14 or later, and
`backend="freethreaded"` a free-threaded build (`python3.13t`, `python3.14t`).
See [Parallel blocks](../guide/parallel.md).

## Check the install

```python
import sigflow
from sigflow import dsp, run

print(sigflow.__version__)
print(len(run(dsp.sine(n=256, frames=4) >> dsp.rms)))  # 4
```

Next: [your first graph](first-graph.md).

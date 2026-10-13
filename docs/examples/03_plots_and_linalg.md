# Plot taps, animation export, and shape-checked linear algebra

`examples/03_plots_and_linalg.py`



## Source

```python
"""Plot taps, animation export, and shape-checked linear algebra."""

import numpy as np

from sigflow import Sig, run, source
from sigflow import dsp, linalg, plot

# --- a spectrum analyser with a waterfall, animated ------------------------
spec = plot.spectrum(title="chirp spectrum", ylim=(-80, 10))
fall = plot.waterfall(title="waterfall", rows=32)

g = (
    dsp.chirp(f0=200, f1=6000, rate=16_000, n=512, frames=24)
    >> dsp.window(kind="hann")
    >> spec
    >> dsp.to_complex
    >> fall
)
run(g, progress=True)
print(spec.save("chirp_spectrum.gif", interval=60))
print(fall.save("chirp_waterfall.gif", interval=60))

# --- beamforming-style linear algebra with explicit dimension contracts ----


@source
def snapshots(
    channels: int = 4, samples: int = 256, frames: int = 3
) -> Sig[complex, ("M", "N")]:
    """M=samples x N=channels array snapshots with a plane wave + noise."""
    rng = np.random.default_rng(0)
    steering = np.exp(1j * np.pi * np.arange(channels) * np.sin(np.deg2rad(20)))
    for _ in range(frames):
        s = np.exp(2j * np.pi * 0.05 * np.arange(samples))
        yield np.outer(s, steering) + 0.3 * (
            rng.normal(size=(samples, channels))
            + 1j * rng.normal(size=(samples, channels))
        )


cov = (
    snapshots() >> linalg.covariance >> linalg.eigh
)  # (N,N) -> eigvals (N,), vecs (N,N)
for f in run(cov):
    w, v = f.data
    print("eigenvalues:", np.round(w, 2), " dominant:", np.round(np.abs(v[:, -1]), 2))

print(cov.describe())
```

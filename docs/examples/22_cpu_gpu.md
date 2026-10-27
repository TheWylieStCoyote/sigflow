# CPU generation and preprocessing -> GPU spectrum -> CPU peak detection

`examples/22_cpu_gpu.py`

Run with ``python examples/22_cpu_gpu.py`` after installing sigflow and a CuPy
package matching your CUDA installation. ``--device 1`` selects another GPU;
``--cpu-only`` runs just the NumPy reference. Without CuPy or a usable CUDA
device, the CPU reference still runs and the GPU section is skipped.

What this exercises
  * ordinary NumPy blocks before and after a CUDA computation
  * explicit to_device() / to_host() boundaries, with typed GPU payloads
  * a CuPy FFT and magnitude-squared calculation that stay on the GPU
  * full-spectrum comparison with NumPy and provenance through both transfers

This demonstrates correctness and placement, not a GPU speed benchmark. Short
frames may cost more to transfer than they save in computation.

## Source

```python
"""CPU generation and preprocessing -> GPU spectrum -> CPU peak detection.

Run with ``python examples/22_cpu_gpu.py`` after installing sigflow and a CuPy
package matching your CUDA installation. ``--device 1`` selects another GPU;
``--cpu-only`` runs just the NumPy reference. Without CuPy or a usable CUDA
device, the CPU reference still runs and the GPU section is skipped.

What this exercises
  * ordinary NumPy blocks before and after a CUDA computation
  * explicit to_device() / to_host() boundaries, with typed GPU payloads
  * a CuPy FFT and magnitude-squared calculation that stay on the GPU
  * full-spectrum comparison with NumPy and provenance through both transfers

This demonstrates correctness and placement, not a GPU speed benchmark. Short
frames may cost more to transfer than they save in computation.
"""

import argparse
from importlib import import_module

import numpy as np

from sigflow import Frame, Framed, Sig, block, dsp, run, to_device, to_host
from sigflow.dims import M, N

RATE = 48_000.0
FREQUENCY = 1500.0
N_SAMPLES = 16_384
N_FRAMES = 4


@block
def cpu_preprocess(x: Sig[float, N]) -> Sig[float, N]:
    """CPU: remove DC and apply a Hann window before transferring samples."""
    assert isinstance(x, np.ndarray), "preprocessing belongs on the CPU"
    return (x - np.mean(x)) * np.hanning(len(x))


@block
def cpu_spectrum(x: Sig[float, N]) -> Sig[float, M]:
    """CPU reference: normalized magnitude-squared real FFT (not a PSD)."""
    return np.abs(np.fft.rfft(x)) ** 2 / x.size**2


@block
def gpu_spectrum(x: Sig[float, N]) -> Sig[float, M]:
    """GPU: FFT and magnitude-squared calculation with no host transfer."""
    cp = import_module("cupy")
    assert isinstance(x, cp.ndarray), "spectrum computation belongs on the GPU"
    spectrum = cp.fft.rfft(x)
    return cp.abs(spectrum) ** 2 / x.size**2


@block
def cpu_peak(f: Framed[Sig[float, M]]) -> Sig[float, M]:
    """CPU: find the strongest frequency and annotate the unchanged spectrum."""
    assert isinstance(f.data, np.ndarray), "peak detection belongs on the CPU"
    frequencies = np.fft.rfftfreq(N_SAMPLES, d=1 / f["rate"])
    return f.annotate(peak_hz=float(frequencies[np.argmax(f.data)]))


def print_results(label: str, frames: list[Frame]) -> None:
    for i, frame in enumerate(frames):
        print(f"{label} frame {i}: peak = {frame['peak_hz']:.1f} Hz")
        assert frame["peak_hz"] == FREQUENCY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--cpu-only", action="store_true", help="run only the CPU reference")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")

    # dsp.sine generates deterministic NumPy frames. Both runs regenerate the
    # same signal; they do not share a one-shot iterator or mutate shared data.
    front = dsp.sine(
        freq=FREQUENCY, rate=RATE, n=N_SAMPLES, frames=N_FRAMES
    ) >> cpu_preprocess
    reference = run(front >> cpu_spectrum >> cpu_peak, check="all", on_error="raise")
    print("CPU reference: NumPy generation -> preprocessing -> spectrum -> peak detection")
    print_results("CPU", reference)
    if args.cpu_only:
        return

    try:
        cp = import_module("cupy")
    except ImportError:
        print("GPU section skipped: install CuPy matching your CUDA installation.")
        return
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        print(f"GPU section skipped: CUDA is unavailable ({exc}).")
        return
    if device_count == 0:
        print("GPU section skipped: no CUDA device is available.")
        return
    if args.device >= device_count:
        parser.error(f"--device must be less than {device_count}")

    graph = (
        front
        >> to_device(device=args.device)
        >> gpu_spectrum
        >> to_host()
        >> cpu_peak
    )
    print(
        f"Hybrid pipeline: CPU preprocessing -> CUDA:{args.device} FFT/power "
        "-> CPU peak detection"
    )
    result = run(graph, check="all", on_error="raise")
    assert len(result) == len(reference) == N_FRAMES
    for expected, actual in zip(reference, result):
        np.testing.assert_allclose(actual.data, expected.data, rtol=1e-9, atol=1e-12)
        assert actual["rate"] == RATE
        assert actual["t0"] == expected["t0"]
        assert "to_device" in actual.prov and "to_host" in actual.prov
    print_results("GPU", result)
    print("All GPU spectra match the CPU reference (rtol=1e-9, atol=1e-12).")
    print("\nLast frame's provenance:")
    print(result[-1].trace())


if __name__ == "__main__":
    main()
```

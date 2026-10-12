# Runtime overhead

Measured with `python bench/bench.py` (Python 3.13.16, NumPy 2.5.3, 5000 frames of 1024 float64).

| configuration | us / frame |
|---|---|
| source only | 8.1 |
| source >> 1 map block | 19.1 |
| source >> 5 map blocks | 59.9 |
| source >> 5 map blocks, check='all' | 65.6 |
| source >> sync generator stream block | 266.0 |
| source >> async generator stream block | 20.8 |
| source >> fan-out of 2 >> select | 73.2 |
| source >> fft >> power_db (real work, N=1024) | 94.4 |
| plain `np.fft.fft` on the same frames, for scale | 14.9 |

Interpretation: each map block costs on the order of the per-block figure above (frame wrapping, annotation merge, a provenance node); a sync generator stream block adds a thread hop; `check='all'` adds the contract check per frame. The runtime is meant for frames of hundreds to millions of samples, not sample-by-sample streaming.

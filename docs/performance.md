# Runtime overhead

Measured with `python bench/bench.py` (Python 3.12.3, NumPy 2.5.4, 5000 frames of 1024 float64).

| configuration                                    | us / frame |
| ------------------------------------------------ | ---------- |
| source only                                      | 3.7        |
| source >> 1 map block                            | 7.7        |
| source >> 5 map blocks                           | 23.8       |
| source >> 5 map blocks, check='all'              | 28.7       |
| source >> sync generator stream block            | 55.9       |
| source >> async generator stream block           | 7.0        |
| source >> fan-out of 2 >> select                 | 23.9       |
| source >> fft >> power_db (real work, N=1024)    | 25.8       |
| plain `np.fft.fft` on the same frames, for scale | 6.5        |

Interpretation: each map block costs on the order of the per-block figure above (frame wrapping, annotation merge, a provenance node); a sync generator stream block adds a thread hop; `check='all'` adds the contract check per frame. The runtime is meant for frames of hundreds to millions of samples, not sample-by-sample streaming.

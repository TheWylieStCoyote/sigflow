# Runtime overhead

Measured with `python bench/bench.py` (Python 3.13.16, NumPy 2.5.3, 5000 frames of 1024 float64).

| configuration                                    | us / frame |
| ------------------------------------------------ | ---------- |
| source only                                      | 11.7       |
| source >> 1 map block                            | 22.3       |
| source >> 5 map blocks                           | 39.8       |
| source >> 5 map blocks, check='all'              | 58.5       |
| source >> sync generator stream block            | 21.8       |
| source >> async generator stream block           | 17.8       |
| source >> fan-out of 2 >> select                 | 89.5       |
| source >> fft >> power_db (real work, N=1024)    | 62.0       |
| plain `np.fft.fft` on the same frames, for scale | 13.6       |

Interpretation: the first map block pays for the scheduler hop; consecutive map blocks are fused into one scheduler member so each extra one costs a few microseconds (frame wrapping, annotation merge, a provenance node). A sync generator stream block runs inline on the event loop when `greenlet` is installed (otherwise in a worker thread, ~10x slower); `check='all'` adds the contract check per frame; `provenance='roots'` or `'off'` trims per-frame bookkeeping on long runs. The runtime is meant for frames of hundreds to millions of samples, not sample-by-sample streaming.

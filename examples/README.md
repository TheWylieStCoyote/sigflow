# Examples

Each script is self-contained (`python examples/NN_name.py`) and prints what it
found; several also write GIF animations or an SVG of the graph next to the
script. Two notebooks: `sigflow_tour.ipynb` is the walkthrough and
`live_control.ipynb` is about long-running graphs in a notebook — background
runs with a control panel (Pause/Resume/Step/Stop, Snapshot, Record, throttle,
progress), live figures inside the panel, `Knob` sliders / dropdowns / toggles
that tune a running graph and its plots, a multi-run dashboard, bounded buffers
with `keep`/`drain()` and stop-on-condition sinks. It ships without outputs:
run it in Jupyter with `ipywidgets` installed (regenerate with
`python make_notebook.py` / `python make_live_notebook.py`).

| script                      | domain                              | framework features it shows                                                                                                                                                       |
| --------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_basics.py`              | —                                   | typed blocks, operators, provenance, build-time type errors, stream blocks                                                                                                        |
| `02_mixed_async.py`         | telemetry                           | zip and merge of sync and async sources, async stream blocks, `arun`, `async for`                                                                                                 |
| `03_plots_and_linalg.py`    | spectrum analyser, array covariance | plot taps + GIF export, shape-checked linear algebra                                                                                                                              |
| `04_qpsk_link.py`           | digital comms                       | a full QPSK modem: stateful filters, 4th-power CFO estimator, PLL, BER via `&`; constellations before/after                                                                       |
| `05_pulse_doppler_radar.py` | radar                               | 2-D frames (pulses x range), MTI, Doppler FFT, CFAR with sample-level masks, tracker, custom range-Doppler plot                                                                   |
| `06_sensor_fusion.py`       | navigation                          | Kalman filter over `merge()` of a blocking IMU (`threaded=True`) and async GPS; corrupt fixes propagate as failed frames; `plot.animate` after the run                            |
| `07_audio_activity.py`      | audio                               | overlap re-framing, feature fan-out into a multi-input block, hysteresis state machine, segments with per-frame provenance, `g.draw()`                                            |
| `08_extending.py`           | —                                   | rate/unit contracts, wrapping stateful SciPy filters, dataclass payloads, the three error policies, `check="all"`, testing blocks as functions                                    |
| `09_subspace_doa.py`        | array processing                    | MUSIC direction finding: named output ports on `eigh`, `port("vals")` / `port("vecs")` routed to different consumers that rejoin, port selection inside a fan-out, `node["vals"]` |
| `10_lineage_forensics.py`   | monitoring                          | attributing anomalies with `f["key"]`, `f[i]`, `f.prov["name"]`, `f.prov[-1]`, `.roots()`, `.chain()`; failed frames' issue provenance; a cause report built purely from lineage  |

| `11_real_io.py` | files & sockets | `io.wav`/`io.npy`/`io.raw` (GNU Radio captures) sources, sinks as taps (`wav_sink`, `csv_log`, `npy_sink`), two graphs over UDP, provenance across the file boundary |
| `12_multirate_channelizer.py` | multirate | polyphase channelizer with N output ports at rate/N, `rate` in the type contract (`refine`), a resampler whose declared rate rejects a 48 kHz-only consumer |
| `13_feedback_loops.py` | adaptive | `feedback()`: an AGC (one-block loop) and an LMS echo canceller whose weights flow from the update block back to the filter block; loop state in lineage |
| `14_parameter_sweep.py` | simulation | BER-vs-SNR sweep of the QPSK link in worker processes and as concurrent `start()` runs; configuration recovered from provenance; summary plot |
| `15_robustness.py` | monitoring | watchdog source that restarts a dying device, graceful degradation with `when()`, routing by issue level (quarantine / drop), self-healing block vs error policies |
| `16_recorded_analysis.py` | analysis | graphs as functions on in-memory slices, `block.apply(frames)`, the same block live and offline, a `Knob`-driven scrubber |
| `17_video_frames.py` | imaging | (H, W) frames, frame differencing, 2-D motion masks as sample-level annotations kept through blur and dropped by the spectrum, centroid tracking |
| `18_testing_pipelines.py` | testing | the recommended way to test chains: blocks as functions, graphs on fixtures, `check="all"`, a fake clock, parametrised and golden tests (`pytest` runnable) |
| `19_graph_report.py` | documentation | `g.report()` HTML/Markdown with diagram, typed table and docstrings; a generated docs folder |
| `21_routing_and_rates.py` | routing / rates | `switch()` by annotation or callable with pipeline branches (a stateful tracker, a 4:1 re-framer), `default=`, `ordered=True`; `ratio=` contracts, `g.ratio`, `RatioMismatch` at build time, `strict=True` verification |
| `20_builder_time_replay.py` | 0.2 tour | units in the contract, data + stream slicing, `Frame.time`, `align()`, an explicit `Graph` diamond, `frame.save`/`replay`, `Stateful` checkpoint/resume, `plot.multi`/`history` |
| `22_cpu_gpu.py` | CPU / CUDA processing | NumPy generation and preprocessing, explicit `to_device` / `to_host` transfers, CuPy FFT and power, CPU peak detection, numerical comparison and provenance; skips CUDA when unavailable |
| `23_xarray_blocks.py` | labeled arrays | `@xarray_aware` under `@block`, named time-axis processing, preserved coordinates and dimension order, per-channel reductions, NumPy direct calls and graph provenance |

A few lessons the examples were written to make visible:

- `|` and `&` pair outputs in lock step. A per-frame branch and a per-segment
  branch do not belong in the same fan-out (07 chains them instead).
- Sources that pace themselves on the wall clock are throttled by whatever is
  downstream: keep the graph lean while it runs and animate afterwards (06).
- Sensors need one shared clock. Stamping each from its own counter biases
  the fused estimate (06, `sim_time()`).
- A decision-directed loop cannot pull in a large frequency offset; a coarse
  estimator in front of it can (04).
- To send different ports of one producer to different consumers, fan out
  `port(...)` selectors: `eigh >> (port("vecs") | port("vals") >> order)` (09).
  Validating each channel of a zip separately is the same pattern (10).
- Multi-port, multi-rate outputs (a channelizer) are one tuple frame per input
  frame — the 1:1 shape fan-out and `port()` expect (12).
- A loop across blocks is `feedback(body, init)`; the body must be 1:1 and
  must not buffer frames (13). Two-way routing is `when(pred, then, otherwise)`
  (15); keyed routing through real stream branches is `switch()` (21).
- Declare `ratio=` on stream blocks that re-frame: a fan-out of branches at
  different rates is then refused before it runs (21).
- Sweeps run one configuration per process from a factory function; nothing in
  a graph needs to be picklable (14).

# Examples

Every script under `examples/` runs on its own (`python examples/NN_name.py`). The notebooks `sigflow_tour.ipynb` and `live_control.ipynb` cover the interactive side. The plots, animations and graph drawings on each page were written by the script when the docs were built.

| example | what it exercises | output |
|---|---|---|
| [01_basics](01_basics.md) | Basics: declare typed blocks, compose with operators, run, inspect provenance | svg |
| [02_mixed_async](02_mixed_async.md) | Mixing synchronous streaming with asynchronous event sources | svg |
| [03_plots_and_linalg](03_plots_and_linalg.md) | Plot taps, animation export, and shape-checked linear algebra | gif, svg |
| [04_qpsk_link](04_qpsk_link.md) | A complete QPSK link: modulator -> channel -> receiver -> bit error rate | gif, svg |
| [05_pulse_doppler_radar](05_pulse_doppler_radar.md) | Pulse-Doppler radar: pulses -> range compression -> Doppler FFT -> CFAR -> tracks | gif, svg |
| [06_sensor_fusion](06_sensor_fusion.md) | Sensor fusion: a Kalman filter fed by a fast synchronous IMU stream merged with | gif, svg |
| [07_audio_activity](07_audio_activity.md) | Streaming audio analysis: re-framing, features, a hysteresis state machine, | png, svg |
| [08_extending](08_extending.md) | Extending sigflow: your own typed blocks, library wrappers and policies | svg |
| [09_subspace_doa](09_subspace_doa.md) | Direction finding with a sensor array: MUSIC from the eigen-decomposition's ports | gif, svg |
| [10_lineage_forensics](10_lineage_forensics.md) | Lineage forensics: answering "where did this bad number come from?" with `[]` | svg |
| [11_real_io](11_real_io.md) | Real I/O boundaries: files in, files out, a socket, and a GNU Radio hand-off | svg |
| [12_multirate_channelizer](12_multirate_channelizer.md) | Multirate: a polyphase channelizer with N output ports at a lower rate, and a | gif, svg |
| [13_feedback_loops](13_feedback_loops.md) | Feedback across blocks: an AGC and an LMS echo canceller built with `feedback()` | gif, svg |
| [14_parameter_sweep](14_parameter_sweep.md) | Parameter sweeps: the same graph over a grid of configurations, in parallel | png, svg |
| [15_robustness](15_robustness.md) | Robustness and recovery: restarting a source, degrading gracefully, routing by issue | svg |
| [16_recorded_analysis](16_recorded_analysis.md) | Analysing a recording interactively: graphs without sources, applied to slices | gif, svg |
| [17_video_frames](17_video_frames.md) | 2-D signals: a video stream — frame differencing, 2-D FFT, masks as sample annotations | gif, svg |
| [18_testing_pipelines](18_testing_pipelines.md) | How to test a signal chain: blocks as functions, graphs on fixtures, contracts on | svg |
| [19_graph_report](19_graph_report.md) | Documenting a graph for a colleague: `g.report()` | svg |
| [20_builder_time_replay](20_builder_time_replay.md) | 0.2 features together: units, slicing, the time axis, an explicit Graph, replay, checkpoints | gif, svg |

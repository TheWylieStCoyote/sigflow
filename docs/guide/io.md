# Files, sockets and GNU Radio

`sigflow.io` has memory-mapped, chunked sources — `io.wav(path, n)`,
`io.raw(path, np.complex64, n)` (what a GNU Radio *File Sink* writes),
`io.npy`, `io.csv_rows` — and asynchronous network sources: `io.udp(port,
dtype)` and `io.zmq(address, dtype, kind="SUB" | "PULL")`, which read what the
GNU Radio *ZMQ PUB Sink* / *ZMQ PUSH Sink* blocks send (raw samples, one
message per vector). Sinks are taps that flush on close: `io.wav_sink`,
`io.npy_sink`, `io.csv_log`, `io.udp_sink`, `io.zmq_sink(address, kind="PUB" |
"PUSH")` (read by GNU Radio *ZMQ SUB/PULL Source*). Use them as context
managers, anywhere in a chain.

A live hand-off with GNU Radio therefore needs no files: a flowgraph with a
ZMQ PUB Sink on one side and `io.zmq("tcp://host:port", np.complex64)` on the
other (remember ZeroMQ's slow-joiner: connect the subscriber before publishing).

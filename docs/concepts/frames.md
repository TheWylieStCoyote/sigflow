# Frames

A **frame** is the unit of data that moves through a graph. It is an immutable
value with four parts:

| part        | what it holds                                                                                                                                                                                                                    |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`      | the payload: an array, a scalar, a tuple of payloads, any Python object                                                                                                                                                          |
| annotations | `stream` (facts about the whole stream: `rate`, `units`, `source`), `meta` (facts about this frame: `t0`, `seq`, an SNR estimate), `samples` (arrays aligned with the leading axis of `data`: a clipping mask, a per-row energy) |
| `issues`    | errors and warnings attached along the way                                                                                                                                                                                       |
| `prov`      | provenance: the block that made it, its configuration, and the parent frames' provenance                                                                                                                                         |

Frames are frozen. Every transformation produces a new frame; `frame.evolve(...)`
copies with changes (`meta=`/`stream=`/`samples=` are merged), `frame.annotate(...)`
adds frame-level facts, `frame.annotate_samples(...)` adds aligned arrays, and
`frame.with_issue(...)` attaches a warning or error.

Blocks usually never touch frames: they receive the payload and return a payload,
and the runtime carries the annotations forward, extends the provenance and
decides which sample-level annotations survive (only through blocks whose
declared output keeps the leading dimension). A block that needs control is
declared with `Framed[...]` and receives the frame itself.

## Failure is data

When a block raises, the runtime does not stop the run (unless asked with
`on_error="raise"`). It emits a frame whose `data` is `MISSING` and whose
`issues` contain the error, the block that raised and the provenance at that
point. Downstream map blocks pass such frames through without computing; stream
blocks never see them; sinks receive them with their whole history. `frame.ok`,
`frame.failed`, `frame.errors`, `frame.warnings` and `frame.raise_if_failed()`
read the state.

## Time

A frame with a `t0` annotation and a stream `rate` has `frame.time`, a
`TimeSpan` (`t0`, `t1`, `duration`, `stamps()`, `index(t)`); `frame.at(t)` and
`frame.between(t0, t1)` select by time, `frame[10:20]` slices data, sample
annotations and time together. Sources stamp frames; re-framing blocks keep the
stamps exact.

## Reading a frame

```python
f["rate"]  # annotation lookup: meta, then stream, then samples
f.get("snr", 0)  # with a default
f[1]  # part 1 of a tuple frame (from a fan-out / zip)
f.prov["fft"]  # the nearest ancestor made by a block called fft
f.trace()  # the lineage as a tree
```

# Frames

A **frame** is the unit of data that moves through a graph. It is an immutable
value with five parts:

| part        | what it holds                                                                                                                                                                                                                    |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`      | the payload: an array, a scalar, a tuple of payloads, any Python object                                                                                                                                                          |
| annotations | `stream` (facts about the whole stream: `rate`, `units`, `source`), `meta` (facts about this frame: `t0`, `seq`, an SNR estimate), `samples` (arrays aligned with the leading axis of `data`: a clipping mask, a per-row energy) |
| `tags`      | labels on the frame or on a time span of it: `speech`, `segment=3`, `burst@[0.5, 1)`; see below                                                                                                                                  |
| `issues`    | errors and warnings attached along the way                                                                                                                                                                                       |
| `prov`      | provenance: the block that made it, its configuration, and the parent frames' provenance                                                                                                                                         |

Frames are frozen. Every transformation produces a new frame; `frame.evolve(...)`
copies with changes (`meta=`/`stream=`/`samples=`/`tags=` are merged), `frame.annotate(...)`
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

## Tags

A tag is a label with an optional value and an optional time span, plus the
name of the block that attached it:

```python
f = f.tag("speech", "calibrated")  # bare labels on the whole frame
f = f.tag(segment=3)  # a label with a value
f = f.tag("burst", span=(1.20, 1.55))  # a label on part of the frame, in seconds
"speech" in f.tags  # or "speech" in f
f.tags["segment"]  # 3 (also f["segment"], after meta/stream/samples)
f.tags.at(1.3)  # the tags covering t = 1.3 s
f.mask("burst")  # boolean array along the leading axis
f = f.untag("calibrated")
```

Values are scalars (`None`, `bool`, `int`, `float`, `str`), enum values and
tuples of those: a tag has to hash, compare and round-trip through
`Frame.save`. Arrays belong in `samples`, objects in `meta`. A tag's identity
is its name and span; when two frames with the same tag meet, the later one's
value wins, so two detectors scoring `speech` differently do not produce two
tags.

The runtime treats tags like the other annotations, with three rules of its own:

- At a join (zip, fan-out, align) the parents' tags are **united**.
- A spanned tag only means something along the frame's time axis. It is
  **clipped** to the frame's window whenever a frame is made, so slicing and
  reframing (`chunk`, `overlap`, `between`) keep exactly the part that falls in
  each frame, and it is dropped by blocks that do not keep the leading axis
  (reductions, spectra, joins). Whole-frame tags survive everything. When
  re-framing mixes several input frames into one output, a whole-frame tag of
  an input applies to that input's samples: it is whole on an output those
  samples cover entirely (adjacent inputs carrying the same tag and value
  count together), a span over the covered part otherwise, and without a time
  axis to express that part it is dropped.
- A tag attached by a block records that block as its `origin`.

Blocks declare tags in their contract: `Annotated[Sig[...], requires("speech")]`
on an input, `provides("bits")` and `clears("raw")` on an output, and
`when("speech", then=...)` routes frames by tag. See
[Contracts](contracts.md#tags) and [the tags example](../examples/24_tags.md).

## Reading a frame

```python
f["rate"]  # annotation lookup: meta, then stream, then samples, then a tag's value
f.get("snr", 0)  # with a default
f[1]  # part 1 of a tuple frame (from a fan-out / zip)
f.prov["fft"]  # the nearest ancestor made by a block called fft
f.trace()  # the lineage as a tree
```

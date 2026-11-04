# A variable number of signals

A spectrum frame may hold any number of signals; a detector finds them, each
needs processing of its own, and afterwards their results belong together
again. Three pieces carry this shape through a graph:

| piece | what it does | cardinality |
|---|---|---|
| a detector stream block | yields one frame per detection, each keyed | 1 frame in, many out |
| `per_key(key, body)` | one instance of `body` per key, state and all | many in, many out |
| `gather(by, ...)` | one tuple frame per group, members as parts | many in, 1 out |

```python
graph = spectra() >> detect >> per_key("track", track_signal, ttl=8) >> gather("seq", count="n_detections", order="track")
```

[Example 25](../examples/25_per_signal.md) runs exactly this. The rest of this
page is about each piece, and about writing a collector of your own when the
built-in `gather` does not fit.

## One frame per detection

A stream block that yields several frames for one input changes the
cardinality of the stream, and the runtime records which input frame each
output consumed, so lineage from a detection back to its spectrum frame is
kept without any work:

```python
@block
def detect(frames: Stream[Framed[Sig[float, N]]], threshold_db: float = -25.0) -> Iterator[Sig[float, N]]:
    tracks: dict[int, float] = {}            # track id -> last frequency
    for f in frames:
        peaks = find_peaks(f.data, threshold_db)
        for k in peaks:
            tid = assign_track(tracks, k)     # nearest existing track, else a new id
            yield f.evolve(data=f.data[k - 4 : k + 5]).annotate(
                track=tid, f0_hz=k * HZ_PER_BIN, n_detections=len(peaks)
            )
```

Three annotations do the work downstream: `track` is the key `per_key` routes
on, `seq` (already on the spectrum frame, so inherited) is the key `gather`
groups on, and `n_detections` tells `gather` when a group is complete. A frame
with no detections yields nothing, which is usually right; a detector that
wants such frames represented yields one marker frame with `n_detections=0`.

The detector is the right place to decide track identity, because it is the
only block that sees all detections of a frame together. Keep its rule simple
and stable, nearest frequency within a gate is enough for most spectra, since
`per_key` relies on it: a track id that flips between frames splits one
signal's state across two instances.

## One instance per key

`per_key(key, body, ttl=)` makes an instance of `body` the first time a key
appears and feeds it only that key's frames. Each instance is a real stream,
so the body may be anything a pipeline may be, with memory: a stream block
with locals, a `Stateful` object, a filter with state. Instances never share
state; a `Stateful` object is deep-copied per instance.

`ttl` closes an instance after that many input frames without its key, so a
signal that vanishes releases its state, and a key that returns later starts
afresh. Without `ttl` every instance lives until the run ends, which is fine
for a bounded set of keys and a leak for an unbounded one. A frame whose key
is `None`, and a failed frame, pass through untouched. Outputs merge in
completion order, each with provenance `per_key(...)[key]`, and a key's own
frames keep their order.

## The built-in gather

`gather(by, ...)` folds the frames sharing a key into one tuple frame:

| argument | meaning |
|---|---|
| `by` | the group key: an annotation name, or a callable on the frame |
| `count=` | the group is complete at this many members: an annotation name, a number, or a callable on a member |
| `lag=1` | release a group when `lag` newer groups have started |
| `ttl=` | release a group after this many input frames without a new member; the backstop |
| `incomplete="emit"` | an early-released group is emitted with a warning issue; `"drop"` drops it |
| `order=` | sort members by an annotation or a callable; the values go to `meta["members"]` |
| `ordered=True` | emit groups in the order they started, not in the order they complete |

One of `count`, `lag` or `ttl` is required; the end of the stream releases
whatever is left. The output frame's `data` is the tuple of the members'
payloads, the members are its `parts`, so `frame[i]` is a member with its own
annotations, tags and provenance, `meta` and `stream` are merged as at a join,
tags are united, and `meta["group"]` is the key.

Two consequences of how the runtime treats stream blocks are worth knowing.
Failed frames never reach a stream block, so `gather` cannot count a member
that failed upstream: the failed frame travels on by itself and its group is
released incomplete, with a warning, by `ttl` or at the end. And the lineage
of a group frame is the frames consumed since the previous group, which can
include members of other groups when groups interleave; each part keeps its
own exact lineage.

## Writing a custom gather

`gather` is an ordinary stream block. When its completion rules do not match
your data, write your own the same way; the graph treats it identically.

```python
@block(ratio="N:1")
def my_gather(frames: Stream[Framed[Sig[Any]]]) -> Iterator[Sig[Any]]:
    groups: dict[Any, list[Frame]] = {}
    for f in frames:
        groups.setdefault(f["seq"], []).append(f)
        ...                                   # decide which groups are complete
        for key in complete:
            members = groups.pop(key)
            ann = merge_annotations(members, keep_samples=False)
            yield Frame(
                tuple(m.data for m in members),
                {**ann["meta"], "group": key},
                ann["stream"],
                {},
                ann["issues"],
                parts=tuple(members),
                tags=ann["tags"],
            )
    for key, members in groups.items():       # end of stream: release the rest
        ...
```

The rules that make it fit:

- **Declare the rate.** `ratio="N:1"` tells the rate checker the block reduces
  frame count, so downstream reframing still gets a ratio rather than `any`.
- **Keep the members as parts.** `parts=tuple(members)` is what makes
  `frame[i]`, port selection and per-port tags work downstream.
- **Merge annotations the standard way.** `merge_annotations(members,
  keep_samples=False)` gives the join semantics: later members win on
  conflicting meta and stream keys, issues are kept once, tags are united and
  spans dropped. Build on its result rather than copying one member's
  annotations.
- **Let the runtime do provenance.** Do not build `prov` yourself; the runtime
  stamps the group frame with the frames consumed since the previous output,
  and each part keeps its own.
- **Release everything at the end.** A generator's `for` loop ends when the
  stream does; whatever is still buffered should go out then, marked as
  incomplete if that matters to a consumer.
- **Mark incompleteness as data.** `frame.with_issue("...", "warning",
  "my_gather")` lets a downstream block or sink see that a group is partial
  without failing it.

Some collectors the built-in does not cover:

**Completion by a marker.** A detector that yields an end-of-frame marker
after its detections, instead of counting them up front:

```python
@block(ratio="N:1")
def until_marker(frames: Stream[Framed[Sig[Any]]]) -> Iterator[Sig[Any]]:
    members: list[Frame] = []
    for f in frames:
        if f.get("end_of_frame"):
            yield group_frame(members, key=f["seq"])
            members = []
        else:
            members.append(f)
```

**Groups that span several spectrum frames.** Collect a track's results over
a window of frames and emit one summary per track per window:

```python
@block(ratio="N:1")
def per_track_window(frames: Stream[Framed[Sig[Any]]], window: int = 10) -> Iterator[Sig[Any]]:
    by_track: dict[int, list[Frame]] = {}
    for f in frames:
        members = by_track.setdefault(f["track"], [])
        members.append(f)
        if len(members) == window:
            yield group_frame(members, key=(f["track"], f["seq"]))
            by_track[f["track"]] = []
```

**Time-based release.** Groups that close when the next member would be
more than a given time after the first, using the frames' `t0`:

```python
@block(ratio="N:1")
def within(frames: Stream[Framed[Sig[Any]]], seconds: float = 0.1) -> Iterator[Sig[Any]]:
    open_groups: dict[Any, list[Frame]] = {}
    for f in frames:
        for key, members in list(open_groups.items()):
            if f.t0 is not None and members[0].t0 is not None and f.t0 - members[0].t0 > seconds:
                yield group_frame(open_groups.pop(key), key=key)
        open_groups.setdefault(f["seq"], []).append(f)
    for key, members in open_groups.items():
        yield group_frame(members, key=key)
```

`group_frame` in these is the small helper from the first sketch: merge the
annotations, build the tuple, keep the parts.

## Testing a collector

Feed it frames directly and read the parts back; no graph is needed for the
rules, and `run` covers the integration once:

```python
def test_marker_closes_the_group():
    frames = [member(0, i) for i in range(3)] + [marker(0)]
    out = list(until_marker.fn(iter(frames)))
    assert len(out) == 1 and len(out[0].parts) == 3

def test_with_per_key():
    out = run(detections() >> per_key("track", track_signal) >> until_marker, on_error="raise")
```

Interleaving is the case to test deliberately: after `per_key`, members of
different groups arrive mixed, in completion order, and a collector that
assumes one group at a time will fold them together.

## Which to use

- Independent, per-frame detections with a count known at detection time:
  `gather(by, count=...)`.
- A stream where "the next group has started" is the only signal: `gather(by,
  lag=1)`.
- Completion decided by a marker, a time window, a budget of frames per
  track, or anything else the detector knows and the collector does not: a
  custom block, which is usually under thirty lines.

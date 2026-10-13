# Lineage

Every frame carries a `Provenance` node: the block name, the configuration it
ran with (knob values resolved), the sequence number within that block's output,
and the parent nodes. Map blocks record one parent; zips and fan-outs one per
input; stream blocks exactly the frames consumed since their previous output;
selections record the selected branch first so `chain()` follows it.

```python
f.prov[0], f.prov[2], f.prov[-1]     # this node, two steps up, the root
f.prov["fft"].params                 # the nearest fft's configuration
f.prov.chain(), f.prov.roots()       # main line of descent; the source frames
"fft" in f.prov
f.trace()                            # rendered tree; shared ancestors marked once
```

## Across many frames

`Lineage(frames)` indexes a run's outputs: `failed()`, `through("scale",
gain=2)`, `descendants_of("sensor", seq=8)`, `sources_of(frames)`, `blame()`
(failures grouped by origin block and source frame), `configs("scale")`,
`Lineage.diff(f1, f2)` (configuration differences between two lineages), and a
`table()` of per-block counts.

## Persisting and replaying

`prov.to_json()` / `Provenance.from_json()` persist a lineage; `frame.save()` /
`Frame.load()` persist a frame with its annotations, issues and lineage.
`replay(graph, frame)` recomputes exactly that output from the same graph,
checking first that the sources are configured as recorded.

## How much to keep

`run(g, provenance="full" | "roots" | "off")`. *full* keeps every node (the
default); *roots* keeps, for each node, only the source frames it descends from —
replay and `blame()` still work and memory stays flat over long runs; *off*
keeps nothing beyond the producing block's name.

# Plan: tag contracts in block signatures

Status: all phases done on branch `tag_contracts` (from `main`). Started 2026-11-07.

| Phase | State |
|---|---|
| 1. Typed tags and `accepts` / `may_provide` | done |
| 2. `may_provide` checks against downstream `accepts` / `when` | done: `TagWarning`; `may_provide` travels downstream |
| 3. Propagation policy (`tags=`, `spans=`, `merge=`) | done |
| 3b. Per-port policy on tuple outputs (`carries(...)`) | done |
| 4. Surfacing in `describe()`, the diagram and `run.report()`; docs pages | done |
| 5. `per_key` checks its key against the producer's guarantees | done: `TagWarning` |

## Goal

A block's signature should say, for tags, what it says for dtypes and shapes:
what it needs, what it reads if present, what it attaches, and what it passes
on. Today the signature can say `requires(...)`, `provides(...)` and
`clears(...)` by name only, and propagation is fixed: an output carries every
tag its input had, minus what it clears.

## Design

### Inputs: required versus optional, with value types

```python
x: Annotated[Sig[complex, N],
             requires("synced", track=int),    # must be there; track's value must be an int
             accepts("snr_db", clipped=bool)]  # read if present; typed if present
```

- `requires(*names, **types)`: as today, plus an optional value type per tag.
  Checked at build time against what the producer guarantees (a declared
  type must be compatible) and at run time on the frame the block receives.
- `accepts(*names, **types)`: new. Tags the block reads when they are
  there. Never a build-time error when absent; a present tag with the wrong
  value type fails the run-time input check. Marks what `tags="declared"`
  keeps, and documents the block.

### Outputs: guaranteed versus conditional

```python
-> Annotated[Sig[int, N], provides(bits=int), may_provide("crc_fail"), clears("synced")]
```

- `provides(*names, **types)`: a guarantee checked at run time, now typed.
- `may_provide(*names, **types)`: new. Sometimes attached, never guaranteed;
  its type is checked when present. Downstream `accepts` and `when("tag")`
  can be checked against it (phase 2).
- `clears(*names)`: unchanged.

### Value types

A type is a class or a tuple of classes, checked like `isinstance`, with
numeric widening (an `int` satisfies `float`, `int`/`float` satisfy
`complex`) and `Enum` classes matched by value (tags store an enum member's
value). Statically, a producer's declared type must be a subtype of the
consumer's; a tag guaranteed without a type is accepted (gradual).

### Propagation policy, per block

```python
@block(tags="carry")                    # default, today's behaviour
@block(tags="none")                     # carry nothing inherited; only what the block attaches
@block(tags="declared")                 # carry inherited tags named in requires/accepts
@block(tags=keep("calibrated", "seq"))  # allow-list
@block(tags=drop("raw"))                # deny-list, for every output
@block(spans="drop")                    # drop inherited spanned tags, keep whole-frame ones
@block(merge="intersection")            # several input frames: carry tags all of them have
```

`merge` decides how the frames feeding one output combine: the parts of a
tuple input, or the frames a stream block consumed for one output.
`"union"` (default, today), `"intersection"`, `"first"`. Tags a block
attaches itself are never filtered by its policy; `clears` still applies.

The policy applies statically (what an output guarantees, in `carry_tags`
and `Block.resolve_ports`) and at run time (what an output frame carries,
next to the existing clearing step).

## Decisions taken without asking

- `tags="declared"` is included: cheap once `accepts` exists, and the
  natural policy for a block that should pass on only what it understands.
- Value types are in from the start: they change the data model, which is
  cheaper to do before more is built on names alone.
- `"carry"` stays the default, so every existing block behaves as before.

## Phases

1. **Typed tags and `accepts`.** `TagSpec`/`SignalType` gain `accepts`,
   `maybe` and `tag_types`; `requires`/`provides` take types; `accepts`,
   `may_provide`. Static type check in `unify`; run-time checks in the input
   and output tag checks. Display in `describe()`.
2. **`may_provide` checks.** Build-time warning when a downstream `accepts`
   or `when("tag")` names a tag no upstream block provides or may provide.
3. **Propagation policy.** `@block(tags=, spans=, merge=)`, `keep(...)`,
   `drop(...)`; static carry and run-time application.
   3b. Per-port policy on tuple outputs: `Annotated[Sig[...], carries("none")]`.
4. **Surfacing.** `describe()`, the diagram tooltip and `run.report()` list
   what each block requires, accepts, provides and passes on; docs pages.
5. **`per_key` integration.** `per_key("track")` checks statically that the
   key is guaranteed when the producer declares it as a tag (pairs with the
   deferred `per_key` summary-mode work).

## Out of scope

Tag values constrained beyond their type (ranges, regex) — `pattern=` lives on
knobs; a tag-level validator can follow if it is wanted.

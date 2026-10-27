"""Blocks: typed, composable units of computation built from plain functions.

The ``@block`` decorator turns an ordinary function into a :class:`BlockDef`.
Inputs and outputs are declared with annotations::

    @block
    def scale(x: Sig[float, N], gain: float = 1.0) -> Sig[float, N]:
        return x * gain

* Parameters annotated with a signal type (``Sig[...]``, a tuple of them,
  ``Frame``, ``Framed[...]`` or ``Stream[...]``) are **inputs**.
* All other parameters are **configuration**, fixed when the block is placed
  in a graph: ``scale(gain=2.0)``.
* The return annotation is the **output** type.

The decorated object is still a function: ``scale(np.arange(3.0), gain=2)``
runs it directly, so blocks are testable and reusable as pure functions.

Kinds of block (inferred automatically):

* **map**    – ordinary function; called once per frame.
* **stream** – generator / async-generator function receiving the whole
               stream of items; may keep state, window, decimate, etc.
* **source** – no signal inputs; a generator (or async generator) that
               produces the stream.

Existing library functions are reused without modification::

    fft = block(np.fft.fft, inputs=Sig[complex, N], outputs=Sig[complex, N])
"""

from __future__ import annotations

import collections.abc
import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, TYPE_CHECKING, get_args, get_origin

from .frame import Frame
from .types import SignalType, StreamType, TupleType, as_type, is_signal_type

__all__ = ["block", "source", "tap", "BlockDef", "Framed", "InputSpec", "lift"]


class _FramedMarker:
    """Annotation marker: ``x: Framed[Sig[...]]`` delivers the whole :class:`Frame`."""

    def __init__(self, item: Any) -> None:
        self.item = as_type(item)

    def __class_getitem__(cls, item: Any) -> "_FramedMarker":
        return cls(item)

    def __repr__(self) -> str:
        return f"Framed[{self.item}]"


if TYPE_CHECKING:
    from typing import Generic, TypeVar

    from .graph import Block

    _F = TypeVar("_F")

    class Framed(
        Frame, Generic[_F]
    ):  # a checker sees a Frame; the runtime sees the marker
        ...
else:
    Framed = _FramedMarker


@dataclass(frozen=True)
class InputSpec:
    name: str
    type: Any  # SignalType | TupleType
    framed: bool = False  # pass the Frame object instead of its data
    stream: bool = False  # receives the whole stream (stream blocks only)


def _classify(annotation: Any) -> tuple[Any, bool, bool] | None:
    """Return (type, framed, stream) if the annotation marks a signal input."""
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str) and any(
        tok in annotation for tok in ("Sig[", "Framed[", "Stream[", "Frame")
    ):
        raise TypeError(
            f"annotation {annotation!r} is a string that could not be evaluated; import the "
            f"names it uses at module level (or avoid quoting signal-type annotations)"
        )
    if annotation is Frame:
        return SignalType(), True, False
    if isinstance(annotation, _FramedMarker):
        return annotation.item, True, False
    if isinstance(annotation, StreamType):
        inner = annotation.item
        if isinstance(inner, _FramedMarker):
            return inner.item, True, True
        if inner is Frame:
            return SignalType(), True, True
        return inner, False, True
    if is_signal_type(annotation):
        return annotation, False, False
    if get_origin(annotation) is tuple:  # tuple[Sig[...], Sig[...]]
        annotation = get_args(annotation)
    if (
        isinstance(annotation, tuple)
        and annotation
        and all(is_signal_type(a) or isinstance(a, _FramedMarker) for a in annotation)
    ):
        return (
            TupleType(
                tuple(a.item if isinstance(a, _FramedMarker) else a for a in annotation)
            ),
            False,
            False,
        )
    return None


_ITERATORS = (
    collections.abc.Iterator,
    collections.abc.AsyncIterator,
    collections.abc.Generator,
    collections.abc.AsyncGenerator,
    collections.abc.Iterable,
    collections.abc.AsyncIterable,
)


def _unwrap_iterator(annotation: Any) -> Any:
    """``Iterator[X]`` / ``AsyncIterator[X]`` (a generator block's element type) -> X."""
    if get_origin(annotation) in _ITERATORS and get_args(annotation):
        return get_args(annotation)[0]
    return annotation


def _output_type(annotation: Any) -> Any:
    if annotation is inspect.Signature.empty or annotation is None:
        return SignalType()
    annotation = _unwrap_iterator(annotation)
    if isinstance(annotation, StreamType):
        annotation = annotation.item
    if isinstance(annotation, _FramedMarker):
        return annotation.item
    if annotation is Frame:
        return SignalType()
    if get_origin(annotation) is tuple:  # tuple[Sig[...], Sig[...]]
        annotation = get_args(annotation)
    if isinstance(annotation, tuple):
        return TupleType(tuple(as_type(a) for a in annotation))
    return as_type(annotation)


class BlockDef:
    """A reusable block definition. Call it with config to make a graph node,
    or with data to run the function directly."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        inputs: Any = None,
        outputs: Any = None,
        kind: str | None = None,
        framed: bool = False,
        on_error: str = "propagate",
        keep_samples: bool | None = None,
        tap: bool = False,
        doc: str | None = None,
        refine: Callable[..., Any] | None = None,
        total: Callable[..., int] | None = None,
        threaded: bool = False,
        ratio: Any = None,
        labels: bool | None = None,
    ) -> None:
        self.fn = fn
        self.labels = labels  # None: follow sigflow.xarray.default_labels()
        self.refine = refine
        self.total = total
        self.ratio_spec = ratio  # see sigflow.ratio: "1:1", "N:1", (4, 1), callable
        self.threaded = threaded
        self.name = name or getattr(fn, "__name__", type(fn).__name__)
        self.__doc__ = doc or getattr(fn, "__doc__", None)
        self.on_error = on_error
        self.tap = tap
        functools.update_wrapper(self, fn, updated=())

        try:
            sig = inspect.signature(
                fn, eval_str=True
            )  # resolve `from __future__ import annotations`
        except (TypeError, ValueError, NameError):
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                sig = None
        self.signature = sig

        self.inputs: list[InputSpec] = []
        self.config: dict[str, inspect.Parameter] = {}
        self._parse_inputs(sig, inputs, framed, kind)

        is_gen = inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn)
        self.is_async = inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(
            fn
        )
        if kind is None:
            if not self.inputs:
                kind = "source"
            elif any(i.stream for i in self.inputs) or is_gen:
                kind = "stream"
            else:
                kind = "map"
        self.kind = kind
        if ratio is None and kind != "stream":
            self.ratio_spec = (
                "1:1"  # map blocks, taps and sources are one frame per frame
            )
        if kind == "source" and not is_gen and not self.is_async:
            # A plain function with no inputs: a one-shot source (or an iterable factory).
            pass

        self.port_names: tuple[str | None, ...] = ()
        if isinstance(outputs, Mapping):  # named output ports
            self.port_names = tuple(outputs)
            outputs = tuple(outputs.values())
        if outputs is not None:
            self.out_type = _output_type(outputs)
        else:
            self.out_type = _output_type(
                sig.return_annotation if sig else inspect.Signature.empty
            )
        if tap:
            self.out_type = self.in_type

        # by default sample-level annotations survive a map block only if the
        # declared output shape has the same leading dimension as the input
        if keep_samples is None:
            keep_samples = _same_leading(self.in_type, self.out_type)
        self.keep_samples = keep_samples

        # ``call`` is what the runtime invokes: ``fn`` behind a thin wrapper that
        # unwraps labelled (xarray) payloads for the body and rewraps outputs from
        # the contract; plain arrays take its fast path. ``fn`` stays the raw body.
        from .xarray import adapt

        self.call = adapt(self)

    # -- stream-rate contract --------------------------------------------- #
    def ratio_for(self, in_type: Any = None) -> Any:
        """Output frames per input frame with default configuration (see :mod:`sigflow.ratio`)."""
        from .graph import as_node

        return as_node(self).ratio_for(in_type)

    @property
    def ratio(self) -> Any:
        return self.ratio_for(self.in_type)

    # -- input parsing ----------------------------------------------------- #
    def _parse_inputs(
        self, sig: inspect.Signature | None, inputs: Any, framed: bool, kind: str | None
    ) -> None:
        params = list(sig.parameters.values()) if sig else []
        if inputs is not None:
            # explicit: a single type, a tuple of types, or {param: type}
            if isinstance(inputs, Mapping):
                spec = dict(inputs)
            else:
                types_ = (
                    list(inputs)
                    if isinstance(inputs, (list, tuple))
                    and not isinstance(inputs, TupleType)
                    else [inputs]
                )
                positional = [
                    p
                    for p in params
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                names = [p.name for p in positional[: len(types_)]] or [
                    f"in{i}" for i in range(len(types_))
                ]
                if len(names) < len(types_):
                    names += [f"in{i}" for i in range(len(names), len(types_))]
                spec = dict(zip(names, types_))
            for pname, t in spec.items():
                if isinstance(t, StreamType):
                    inner = t.item
                    fr = isinstance(inner, _FramedMarker) or inner is Frame
                    self.inputs.append(
                        InputSpec(
                            pname,
                            inner.item
                            if isinstance(inner, _FramedMarker)
                            else SignalType()
                            if inner is Frame
                            else inner,
                            fr,
                            True,
                        )
                    )
                elif isinstance(t, _FramedMarker):
                    self.inputs.append(InputSpec(pname, t.item, True, False))
                elif t is Frame:
                    self.inputs.append(InputSpec(pname, SignalType(), True, False))
                else:
                    self.inputs.append(InputSpec(pname, as_type(t), framed, False))
            for p in params:
                if p.name not in spec and p.kind not in (
                    p.VAR_POSITIONAL,
                    p.VAR_KEYWORD,
                ):
                    self.config[p.name] = p
            return

        for p in params:
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            c = _classify(p.annotation)
            if c is not None and p.default is inspect.Parameter.empty:
                t, fr, st = c
                self.inputs.append(InputSpec(p.name, t, fr or framed, st))
            else:
                self.config[p.name] = p
        if (
            not self.inputs
            and params
            and kind != "source"
            and not (
                inspect.isgeneratorfunction(self.fn)
                or inspect.isasyncgenfunction(self.fn)
            )
        ):
            # Untyped first positional argument (no default) of a plain function: an Any input.
            first = params[0]
            if first.default is inspect.Parameter.empty and first.kind in (
                first.POSITIONAL_ONLY,
                first.POSITIONAL_OR_KEYWORD,
            ):
                self.inputs.append(InputSpec(first.name, SignalType(), framed, False))
                self.config.pop(first.name, None)

    # -- types ------------------------------------------------------------- #
    @property
    def in_type(self) -> Any:
        if not self.inputs:
            return None
        if len(self.inputs) == 1:
            return self.inputs[0].type
        return TupleType(tuple(i.type for i in self.inputs))

    @property
    def framed(self) -> bool:
        return any(i.framed for i in self.inputs)

    @property
    def stream_inputs(self) -> bool:
        return any(i.stream for i in self.inputs)

    # -- calling ----------------------------------------------------------- #
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """``blockdef(**config)`` -> node;  ``blockdef(data, ..., **config)`` -> call the function.

        Positional arguments are *data* when the first one is an array, Frame, list,
        tuple or dict (``scale(np.ones(4), gain=2)`` runs the function); otherwise they
        are configuration in signature order (``to_units("V")``, ``io.wav("x.wav", n=512)``
        make nodes). ``blockdef.fn(...)`` always calls the function directly
        (bypassing the labelled-payload wrapper, ``blockdef.call``)."""
        if args and not _looks_like_data(args[0]):
            names = [k for k in self.config][: len(args)]
            if len(names) < len(args):
                raise TypeError(
                    f"{self.name} takes at most {len(self.config)} configuration arguments"
                )
            return self.node(**{**dict(zip(names, args)), **kwargs})
        if args:
            return self.call(*args, **kwargs)
        # keyword-only call: are any of the keywords inputs?  Then run directly.
        in_names = {i.name for i in self.inputs}
        if in_names and in_names <= set(kwargs):
            return self.call(**kwargs)
        return self.node(**kwargs)

    def apply(self, items: Any, **config: Any) -> list:
        """Run this block as a graph over in-memory frames/arrays and return the output
        frames (``block(x)`` calls the function directly; ``block.apply(frames)`` keeps
        annotations and provenance)."""
        return self.node(**config)(items)

    def node(self, **config: Any) -> "Block":
        from .graph import Block

        return Block(self, config)

    def bind(self, **config: Any) -> "BlockDef":
        """Return a new definition with some configuration baked in."""
        new = BlockDef(
            functools.partial(self.fn, **config),
            name=self.name,
            inputs={i.name: (_respec(i)) for i in self.inputs} or None,
            outputs=self.out_type,
            kind=self.kind,
            on_error=self.on_error,
            keep_samples=self.keep_samples,
            tap=self.tap,
            labels=self.labels,
            doc=self.__doc__,
            refine=self.refine,
            total=self.total,
            threaded=self.threaded,
        )
        new.config = {k: v for k, v in self.config.items() if k not in config}
        new.is_async = self.is_async
        new.port_names = self.port_names
        return new

    def check_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(config) - set(self.config)
        if unknown and not (
            self.signature
            and any(p.kind == p.VAR_KEYWORD for p in self.signature.parameters.values())
        ):
            raise TypeError(
                f"{self.name}: unknown configuration {sorted(unknown)}; "
                f"accepts {sorted(self.config)}"
            )
        missing = [
            k
            for k, p in self.config.items()
            if p.default is inspect.Parameter.empty and k not in config
        ]
        if missing:
            raise TypeError(f"{self.name}: missing required configuration {missing}")
        full = {
            k: p.default
            for k, p in self.config.items()
            if p.default is not inspect.Parameter.empty
        }
        full.update(config)
        return full

    # -- pickling: by import path, so process pools can call the block ------- #
    def __reduce__(self) -> Any:
        fn = self.fn
        mod, qual = getattr(fn, "__module__", None), getattr(fn, "__qualname__", None)
        if mod and qual and "<" not in qual:
            return (_import_block, (mod, qual))
        raise TypeError(
            f"block {self.name!r} wraps a function that cannot be pickled (lambda or "
            f"nested); define it at module level or use parallel(..., backend='thread')"
        )

    # -- operators: using the bare definition in a graph uses default config #
    def __getitem__(self, key: Any) -> Any:
        return self.node()[key]

    def __rshift__(self, other: Any) -> Any:
        return self.node() >> other

    def __rrshift__(self, other: Any) -> Any:
        from .graph import as_node

        return as_node(other) >> self.node()

    def __or__(self, other: Any) -> Any:
        return self.node() | other

    def __ror__(self, other: Any) -> Any:
        from .graph import as_node

        return as_node(other) | self.node()

    def __and__(self, other: Any) -> Any:
        return self.node() & other

    def __rand__(self, other: Any) -> Any:
        from .graph import as_node

        return as_node(other) & self.node()

    def __repr__(self) -> str:
        ins = ", ".join(
            f"{i.name}: {'Stream[' if i.stream else ''}{i.type}{']' if i.stream else ''}"
            for i in self.inputs
        )
        cfg = ", ".join(self.config)
        return f"<{self.kind} block {self.name}({ins}{', ' if ins and cfg else ''}{cfg}) -> {self.out_type}>"


def _import_block(mod: str, qual: str) -> Any:
    """Unpickle a block definition by import path (the module attribute may be the
    BlockDef that replaced the function, or the function itself)."""
    import importlib

    obj: Any = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj if isinstance(obj, BlockDef) else BlockDef(obj)


def _looks_like_data(x: Any) -> bool:
    """Arrays, frames, containers and iterators are data; strings, numbers, Knobs, callables are config."""
    import numpy as _np

    if isinstance(x, (str, bytes)) or callable(x) and not hasattr(x, "__next__"):
        return False
    return (
        isinstance(x, (_np.ndarray, _np.generic, Frame, list, tuple, dict))
        or hasattr(x, "__array__")
        or hasattr(x, "__iter__")
        or hasattr(x, "__next__")
    )


def _respec(i: InputSpec) -> Any:
    t: Any = _FramedMarker(i.type) if i.framed else i.type
    return StreamType(t) if i.stream else t


def _same_leading(in_t: Any, out_t: Any) -> bool:
    if not isinstance(in_t, SignalType) or not isinstance(out_t, SignalType):
        return False
    if in_t.shape is None or out_t.shape is None or not in_t.shape or not out_t.shape:
        return False
    return in_t.shape[0] == out_t.shape[0] and in_t.shape[0] is not None


# --------------------------------------------------------------------------- #
# Decorators
# --------------------------------------------------------------------------- #


def block(fn: Callable[..., Any] | None = None, /, **options: Any) -> Any:
    """Declare a block.

    Usable bare (``@block``), with options (``@block(on_error="skip")``) or as
    a wrapper for existing code
    (``block(scipy.signal.hilbert, inputs=Sig[float, N], outputs=Sig[complex, N])``).

    Options:
        name:         display name (defaults to the function name)
        inputs:       explicit input type(s) when not using annotations
        outputs:      explicit output type when not using annotations; a dict
                      ``{"vals": Sig[...], "vecs": Sig[...]}`` declares *named*
                      output ports, selectable with ``node["vals"]``
        kind:         "map" | "stream" | "source" (normally inferred)
        on_error:     "propagate" (attach an error, pass a MISSING frame on),
                      "raise" (stop the run) or "drop" (silently drop the frame)
        keep_samples: whether sample-level annotations survive this block
        refine:       ``refine(**config) -> output type`` (or ``(in_type, out_type)``):
                      lets a block compute concrete dimensions from its
                      configuration so they are checked before running, e.g.
                      ``refine=lambda n, **_: Sig[float, (n,)]``
        total:        ``total(**config) -> int``: expected number of frames a
                      source emits, so progress can be reported as a percentage
        threaded:     sources: pull items in a worker thread, so a source that
                      *blocks* (hardware reads, ``time.sleep`` pacing) does not
                      stall asynchronous sources running alongside it. Sync
                      generator stream blocks: force the worker-thread driver
                      (the default runs them inline on the loop via greenlet when
                      it is installed, which is much faster but must not block)
        labels:       ``True``: an ``xarray.DataArray`` input is lined up with the
                      input contract, the body gets the bare array, and outputs
                      are rebuilt as labelled arrays from the output contract
                      (see :mod:`sigflow.xarray`). ``False``: payloads pass
                      through untouched. Unset: follow
                      ``sigflow.xarray.default_labels()``
    """
    if fn is None:
        return lambda f: BlockDef(f, **options)
    return BlockDef(fn, **options)


def source(fn: Callable[..., Any] | None = None, /, **options: Any) -> Any:
    """Declare a source block (a generator / async generator with no signal inputs)."""
    options.setdefault("kind", "source")
    return block(fn, **options)


def tap(fn: Callable[..., Any] | None = None, /, **options: Any) -> Any:
    """Declare a side-effect block: runs ``fn`` on each frame and passes the frame on unchanged."""
    options["tap"] = True
    return block(fn, **options)


def lift(fn: Callable[..., Any], **options: Any) -> BlockDef:
    """Wrap any callable as an untyped (``Any -> Any``) map block."""
    options.setdefault("inputs", SignalType())
    options.setdefault("outputs", SignalType())
    return BlockDef(fn, **options)

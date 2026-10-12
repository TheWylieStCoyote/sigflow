"""sigflow — a small, typed, functional framework for signal-processing graphs.

    from sigflow import block, Sig, run
    import numpy as np

    @block
    def scale(x: Sig[float, "N"], gain: float = 1.0) -> Sig[float, "N"]:
        return x * gain

    frames = (sigflow.from_iterable([np.ones(4), np.ones(4)], Sig[float, "N"]) >> scale(gain=2)).run()
"""
from .types import (Any, Sig, SignalType, TupleType, StreamType as Stream, TypeMismatch, Env,
                    Scalar, Vector, Matrix, Tensor, CVector, CMatrix, unify)
from .frame import Frame, Issue, Provenance, MISSING, TimeSpan
from .block import block, source, tap, lift, BlockDef, Framed
from .graph import (Node, Block, Source, Pipeline, Fanout, Zip, Merge, Select, port, merge, as_node,
                    from_iterable, validate, feedback, when, align)
from .runtime import run, arun, iterate, iterate_sync
from .progress import Monitor
from .control import Run, Knob
from .ui import dashboard, require_widgets
from .builder import Graph
from .persist import replay, Stateful, Checkpoint, load_checkpoint, ReplayMismatch
from . import io

__version__ = "0.2.0"
__all__ = [
    "Any", "Sig", "SignalType", "TupleType", "Stream", "TypeMismatch", "Env",
    "Scalar", "Vector", "Matrix", "Tensor", "CVector", "CMatrix", "unify",
    "Frame", "Issue", "Provenance", "MISSING", "TimeSpan",
    "block", "source", "tap", "lift", "BlockDef", "Framed",
    "Node", "Block", "Source", "Pipeline", "Fanout", "Zip", "Merge", "Select", "port", "merge", "as_node", "feedback", "when", "align",
    "from_iterable", "validate", "run", "arun", "iterate", "iterate_sync", "Monitor", "Run", "Knob", "dashboard", "require_widgets", "io", "Graph", "replay", "Stateful", "Checkpoint", "load_checkpoint", "ReplayMismatch",
]

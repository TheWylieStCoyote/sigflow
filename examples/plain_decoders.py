"""Plain-Python packet functions for ``27_parallel_backends.py``.

This module imports neither NumPy nor sigflow, so every backend of ``parallel()``
can run its functions, sub-interpreters included (``backend="interpreter"``), which
cannot load NumPy. The example wraps them into blocks where it builds its graph.
"""

from __future__ import annotations

_POLY = 0x1021  # CRC-16/CCITT


def crc16(packet: bytes, rounds: int = 1) -> int:
    """CRC-16/CCITT of ``packet``, computed ``rounds`` times over (pure Python, so
    the work holds the GIL: a fair test of each backend)."""
    crc = 0xFFFF
    for _ in range(rounds):
        crc = 0xFFFF
        for byte in packet:
            crc ^= byte << 8
            for _ in range(8):
                crc = ((crc << 1) ^ _POLY) if crc & 0x8000 else (crc << 1)
                crc &= 0xFFFF
    return crc

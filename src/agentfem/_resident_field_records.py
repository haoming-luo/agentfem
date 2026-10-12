# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private immutable rollback payloads, not a second authoritative state store.

Live fields remain authoritative. Reuse requires exact current bytes; external
field writes therefore invalidate reuse without relying on a version counter.
Old records stay valid across commits, rollback, and cache replacement.
"""
import numpy as np


def _freeze(value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise TypeError("Resident field payloads cannot contain Python objects.")
    owner = value
    while isinstance(owner, np.ndarray):
        owner = owner.base
    if isinstance(owner, bytes) and value.flags.c_contiguous:
        offset = value.ctypes.data - np.frombuffer(owner, dtype=np.uint8).ctypes.data
        if offset < 0 or offset + value.nbytes > len(owner):
            raise ValueError("Invalid immutable field storage bounds.")
    else:
        owner, offset = value.tobytes(order="C"), 0
    return np.frombuffer(owner, dtype=value.dtype, count=value.size,
                         offset=offset).reshape(value.shape)


def _same_bits(current, saved):
    if (current.shape != saved.shape or current.dtype != saved.dtype
            or not current.flags.c_contiguous):
        return False
    # Avoid an eight-times-larger bytewise comparison temporary for real64.
    word = np.uint64 if current.nbytes % 8 == 0 else np.uint8
    return np.array_equal(current.reshape(-1).view(word), saved.reshape(-1).view(word))


class ResidentFieldRecords:
    """Bounded current-record cache; no writable alias crosses its boundary."""

    def __init__(self):
        self._records = {}

    def remember(self, name, value):
        self._records[name] = _freeze(value)

    def capture(self, fields):
        result = {}
        records = {}
        for name, function in fields.items():
            current = np.asarray(function.x.array)
            saved = self._records.get(name)
            if saved is None or not _same_bits(current, saved):
                saved = _freeze(current)
            records[name] = saved
            # Fresh metadata, immutable bytes. Caller reshapes cannot corrupt
            # another snapshot or the cache's own shape/dtype metadata.
            result[name] = _freeze(saved)
        self._records = records
        return result

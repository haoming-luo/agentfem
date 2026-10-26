# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Accepted energy components, without reclassifying losses as stored energy.

The owning Procedure supplies evaluated physical channels and already owns
work integration. This observer owns only a small, restartable scalar history.
It never updates constitutive state or certifies an energy balance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


_ROLES = {"stored", "cumulative_dissipation", "cumulative_numerical_release"}
_SCHEMA = "agentfem.accepted-energy-components.v1"


@dataclass(frozen=True)
class EnergyComponentFrame:
    index: int
    factor: float
    values: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class _EnergyBlock:
    previous: object = field(repr=False)
    frames: tuple[EnergyComponentFrame, ...]


@dataclass(eq=False)
class AcceptedEnergyComponents:
    """One declared set of energy channels on the existing accepted boundary."""

    requires_field_snapshot = False

    roles: dict[str, str]
    evaluator: object
    _blocks: _EnergyBlock | None = field(default=None, init=False, repr=False)
    _tail: tuple[EnergyComponentFrame, ...] = field(default=(), init=False, repr=False)
    _count: int = field(default=0, init=False)
    complete: bool = field(default=True, init=False)

    def __post_init__(self):
        self.roles = dict(self.roles)
        if (not self.roles or any(not isinstance(name, str) or not name for name in self.roles)
                or any(role not in _ROLES for role in self.roles.values())
                or not callable(self.evaluator)):
            raise ValueError("Energy components require named roles and a callable evaluator.")

    @property
    def frames(self):
        """Materialize only for explicit reporting/checkpoint operations."""
        chunks = [self._tail]
        block = self._blocks
        while block is not None:
            chunks.append(block.frames)
            block = block.previous
        return tuple(frame for chunk in reversed(chunks) for frame in chunk)

    def _append(self, frame):
        tail = (*self._tail, frame)
        if len(tail) == 64:
            self._blocks = _EnergyBlock(self._blocks, tail)
            tail = ()
        self._tail = tail
        self._count += 1

    def _replace_frames(self, frames):
        self._blocks, self._tail, self._count = None, (), 0
        for frame in frames:
            self._append(frame)

    def _frame(self, index, factor, values):
        if (type(index) is not int or index < 0 or type(factor) not in (int, float)
                or not np.isfinite(factor) or not 0. <= factor <= 1.
                or not isinstance(values, dict) or set(values) != set(self.roles)):
            raise ValueError("Energy component station or channel schema differs.")
        selected = []
        for name, role in self.roles.items():
            value = values[name]
            if type(value) not in (int, float) or not np.isfinite(value) or (
                role != "stored" and value < 0.
            ):
                raise ValueError("Energy components must be finite; cumulative losses cannot be negative.")
            selected.append(float(value))
        return EnergyComponentFrame(index, float(factor), tuple(selected))

    def _capture(self, snapshot):
        return self._frame(int(snapshot.index), float(snapshot.load_factor), self.evaluator(snapshot))

    def _validate_sequence(self, frames, *, complete, snapshot):
        if not frames or frames[-1].index != int(snapshot.index) or abs(
            frames[-1].factor - float(snapshot.load_factor)
        ) > 1e-12:
            raise ValueError("Energy components do not end at the accepted boundary.")
        if complete and (frames[0].index != 0 or abs(frames[0].factor) > 1e-12):
            raise ValueError("Complete energy history must begin at the initial state.")
        for a, b in zip(frames, frames[1:]):
            if b.index != a.index + 1 or b.factor <= a.factor + 1e-12:
                raise ValueError("Energy history must contain consecutive accepted stations.")
            for role, old, new in zip(self.roles.values(), a.values, b.values, strict=True):
                tolerance = 64 * np.finfo(float).eps * max(abs(old), abs(new), 1e-30)
                if role != "stored" and new < old - tolerance:
                    raise ValueError("Cumulative energy loss decreased along accepted history.")

    def reset(self, snapshot):
        frame = self._capture(snapshot)
        self._validate_sequence([frame], complete=True, snapshot=snapshot)
        self._replace_frames([frame])
        self.complete = True

    def prepare_resume(self, snapshot):
        self._validate_sequence(self.frames, complete=self.complete, snapshot=snapshot)

    def accept(self, snapshot):
        frame = self._capture(snapshot)
        # Constant-time validation on normal advancement; do not rescan an
        # increasingly long history at every accepted increment.
        if not self._count:
            raise ValueError("Energy history was not initialized.")
        last = self._tail[-1] if self._tail else self._blocks.frames[-1]
        self._validate_sequence([last, frame], complete=False, snapshot=snapshot)
        self._append(frame)

    def snapshot_runtime_state(self):
        return {"blocks": self._blocks, "tail": self._tail, "count": self._count,
                "complete": self.complete}

    def restore_runtime_state(self, state):
        self._blocks, self._tail, self._count = state["blocks"], state["tail"], state["count"]
        self.complete = state["complete"]

    def checkpoint_state(self):
        return {"schema": _SCHEMA, "roles": dict(self.roles), "complete": self.complete,
                "frames": [{"index": f.index, "factor": f.factor,
                            "values": dict(zip(self.roles, f.values, strict=True))} for f in self.frames]}

    def restore_checkpoint_state(self, record, *, current_snapshot):
        if (not isinstance(record, dict) or set(record) != {"schema", "roles", "complete", "frames"}
                or record["schema"] != _SCHEMA or record["roles"] != self.roles
                or type(record["complete"]) is not bool or not isinstance(record["frames"], list)):
            raise ValueError("Energy component checkpoint schema or roles differ.")
        frames = []
        for item in record["frames"]:
            if not isinstance(item, dict) or set(item) != {"index", "factor", "values"}:
                raise ValueError("Energy component checkpoint frame differs.")
            frames.append(self._frame(item["index"], item["factor"], item["values"]))
        self._validate_sequence(frames, complete=record["complete"], snapshot=current_snapshot)
        current = self._capture(current_snapshot)
        scale = np.maximum(np.maximum(np.abs(current.values), np.abs(frames[-1].values)), 1e-30)
        if not np.allclose(current.values, frames[-1].values, rtol=1e-9,
                           atol=128 * np.finfo(float).eps * scale):
            raise ValueError("Energy checkpoint endpoint differs from restored physical state.")
        self._replace_frames(frames)
        self.complete = record["complete"]

    def restore_missing_checkpoint_state(self, *, current_snapshot):
        # A historical initial energy cannot be inferred from a final state.
        self._replace_frames([self._capture(current_snapshot)])
        self.complete = False

    def summary(self):
        return {"kind": "accepted_energy_components", "roles": dict(self.roles),
                "complete": self.complete, "frame_count": self._count,
                "scope": "energy_observation_not_work_integration_or_verification"}

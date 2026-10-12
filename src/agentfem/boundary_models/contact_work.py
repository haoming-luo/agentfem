# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted-path work State for prescribed rigid contact motion.

The contact backend supplies generalized force and generalized coordinate
samples.  This module owns only transactional State: a trial sample is staged
during one increment, committed after the Procedure accepts that increment,
or discarded on rollback.  The resulting work ledger is backend-neutral and
JSON serializable for transient checkpoint/restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from .rigid import _readonly_array
from .rigid import PrescribedRigidMotion


_STATION_SCHEMA = "agentfem.prescribed-contact-work-station.v1"
_STATE_SCHEMA = "agentfem.prescribed-contact-work-state.v1"
_NUMERIC_STATE_SCHEMA = "agentfem.prescribed-contact-work-state.v2"
_SCALAR_COLUMNS = ("time", "factor", "contact_potential_energy")
_VECTOR_COLUMNS = ("generalized_force", "generalized_coordinate", "contact_resultant")
_COUNT_COLUMNS = ("active_point_count", "invalid_point_count")


@dataclass(frozen=True)
class PrescribedRigidMotionSchedule:
    """A proportional rigid motion over one explicit physical-time interval."""

    motion: PrescribedRigidMotion
    start_time: float
    end_time: float
    name: str = "prescribed_rigid_motion_schedule"
    amplitude: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.motion, PrescribedRigidMotion):
            raise TypeError("A rigid-motion schedule requires PrescribedRigidMotion.")
        start = float(self.start_time)
        end = float(self.end_time)
        if not np.isfinite(start) or start < 0.0:
            raise ValueError("Rigid-motion start_time must be finite and non-negative.")
        if not np.isfinite(end) or end <= start:
            raise ValueError(
                "Rigid-motion end_time must be finite and exceed start_time."
            )
        if not str(self.name).strip():
            raise ValueError("Rigid-motion schedule requires a name.")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "name", str(self.name))
        if self.amplitude is not None:
            from ..amplitudes import Amplitude

            if (
                not isinstance(self.amplitude, Amplitude)
                or not self.amplitude.serializable
            ):
                raise TypeError("Rigid motion requires a serializable Amplitude asset.")
            endpoints = np.array([self.amplitude(start), self.amplitude(end)])
            if not np.isfinite(endpoints).all() or not np.allclose(
                endpoints, [0, 1], rtol=0, atol=1e-12
            ):
                raise ValueError(
                    "Rigid motion amplitude must span factors zero to one over its physical interval."
                )

    def factor_at(self, time_value: float) -> float:
        selected = float(time_value)
        if not np.isfinite(selected) or selected < 0.0:
            raise ValueError(
                "Rigid-motion evaluation time must be finite and non-negative."
            )
        if self.amplitude is not None:
            factor = self.amplitude(
                float(np.clip(selected, self.start_time, self.end_time))
            )
            if not np.isfinite(factor) or not 0.0 <= factor <= 1.0:
                raise ValueError("Rigid-motion amplitude must remain finite in [0,1].")
            return factor
        raw = (selected - self.start_time) / (self.end_time - self.start_time)
        return float(np.clip(raw, 0.0, 1.0))

    def generalized_coordinate_at(self, time_value: float) -> np.ndarray:
        return self.motion.generalized_coordinate(self.factor_at(time_value))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "prescribed_rigid_motion_schedule",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "interpolation": "linear_factor_clamped"
            if self.amplitude is None
            else "amplitude_in_physical_time_clamped",
            **(
                {}
                if self.amplitude is None
                else {"amplitude": self.amplitude.summary()}
            ),
            "motion": self.motion.summary(),
        }


def _finite_vector(value, *, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float).reshape(-1)
    if selected.size < 1 or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be one non-empty finite vector.")
    return selected


@dataclass(frozen=True, eq=False)
class PrescribedContactWorkStation:
    """One globally reduced rigid-tool force/coordinate station."""

    time: float
    factor: float
    generalized_force: object
    generalized_coordinate: object
    contact_resultant: object
    contact_potential_energy: float
    active_point_count: int
    invalid_point_count: int

    def __post_init__(self) -> None:
        selected_time = float(self.time)
        factor = float(self.factor)
        if not np.isfinite(selected_time) or selected_time < 0.0:
            raise ValueError(
                "Contact work-station time must be finite and non-negative."
            )
        if not np.isfinite(factor) or not 0.0 <= factor <= 1.0 + 1.0e-12:
            raise ValueError("Contact work-station factor must lie in [0, 1].")
        force = _finite_vector(self.generalized_force, name="Generalized force")
        coordinate = _finite_vector(
            self.generalized_coordinate,
            name="Generalized coordinate",
        )
        if force.shape != coordinate.shape:
            raise ValueError(
                "Generalized force and coordinate must use the same layout."
            )
        resultant = _finite_vector(self.contact_resultant, name="Contact resultant")
        if resultant.size not in {2, 3}:
            raise ValueError("Contact resultant must be two- or three-dimensional.")
        potential = float(self.contact_potential_energy)
        if not np.isfinite(potential) or potential < 0.0:
            raise ValueError(
                "Contact potential energy must be finite and non-negative."
            )
        active = int(self.active_point_count)
        invalid = int(self.invalid_point_count)
        if active < 0 or invalid < 0:
            raise ValueError("Contact point counts must be non-negative.")
        object.__setattr__(self, "time", selected_time)
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "generalized_force", _readonly_array(force))
        object.__setattr__(
            self,
            "generalized_coordinate",
            _readonly_array(coordinate),
        )
        object.__setattr__(self, "contact_resultant", _readonly_array(resultant))
        object.__setattr__(self, "contact_potential_energy", potential)
        object.__setattr__(self, "active_point_count", active)
        object.__setattr__(self, "invalid_point_count", invalid)

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": _STATION_SCHEMA,
            "time": self.time,
            "factor": self.factor,
            "generalized_force": self.generalized_force.tolist(),
            "generalized_coordinate": self.generalized_coordinate.tolist(),
            "contact_resultant": self.contact_resultant.tolist(),
            "contact_potential_energy": self.contact_potential_energy,
            "active_point_count": self.active_point_count,
            "invalid_point_count": self.invalid_point_count,
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> PrescribedContactWorkStation:
        if not isinstance(snapshot, dict) or snapshot.get("schema") != _STATION_SCHEMA:
            raise ValueError("Unsupported prescribed-contact work-station snapshot.")
        required = {
            "schema",
            "time",
            "factor",
            "generalized_force",
            "generalized_coordinate",
            "contact_resultant",
            "contact_potential_energy",
            "active_point_count",
            "invalid_point_count",
        }
        if set(snapshot) != required:
            raise ValueError("Prescribed-contact work-station fields do not match.")
        return cls(
            time=snapshot["time"],
            factor=snapshot["factor"],
            generalized_force=snapshot["generalized_force"],
            generalized_coordinate=snapshot["generalized_coordinate"],
            contact_resultant=snapshot["contact_resultant"],
            contact_potential_energy=snapshot["contact_potential_energy"],
            active_point_count=snapshot["active_point_count"],
            invalid_point_count=snapshot["invalid_point_count"],
        )


@dataclass(frozen=True, eq=False)
class _WorkHistoryBlock:
    """Persistent full block shared by accepted-boundary rollback tokens."""

    previous: _WorkHistoryBlock | None
    stations: tuple[PrescribedContactWorkStation, ...]


_HISTORY_BLOCK_SIZE = 64


@dataclass(frozen=True, eq=False)
class _WorkTransaction:
    """Constant-size in-memory token; never a durable file format.

    Frozen blocks and a bounded frozen tail keep old tokens valid when a run
    commits, rolls back, or forks from an earlier accepted boundary. They do
    not alias a mutable accepted-history list.
    """

    identity: str
    blocks: _WorkHistoryBlock | None
    tail: tuple[PrescribedContactWorkStation, ...]

    @property
    def accepted(self) -> tuple[PrescribedContactWorkStation, ...]:
        blocks = []
        current = self.blocks
        while current is not None:
            blocks.append(current.stations)
            current = current.previous
        return (
            tuple(station for block in reversed(blocks) for station in block)
            + self.tail
        )

    def __eq__(self, other):
        if not isinstance(other, _WorkTransaction):
            return NotImplemented
        if self.identity != other.identity:
            return False
        if self.blocks is other.blocks and self.tail is other.tail:
            return True
        # Do not recurse through a potentially very long linked block chain.
        return self.accepted == other.accepted


class PrescribedContactWorkState:
    """Transactional accepted-path ledger for one prescribed rigid tool.

    ``accepted`` is an observational history; modify it only through this
    State's initialize/commit/restore methods so cumulative work stays coherent.
    """

    def __init__(self, *, identity: str) -> None:
        selected = str(identity).strip()
        if not selected:
            raise ValueError("Prescribed-contact work State requires an identity.")
        self.identity = selected
        self.accepted: list[PrescribedContactWorkStation] = []
        self.trial: PrescribedContactWorkStation | None = None
        self._path_work = 0.0
        self._history_fingerprint = self._fingerprint(())
        self._history_blocks: _WorkHistoryBlock | None = None
        self._history_tail: tuple[PrescribedContactWorkStation, ...] = ()

    def _appended_history(self, station):
        tail = self._history_tail + (station,)
        if len(tail) == _HISTORY_BLOCK_SIZE:
            return _WorkHistoryBlock(self._history_blocks, tail), ()
        return self._history_blocks, tail

    @staticmethod
    def _blocked_history(stations):
        blocks = None
        full = len(stations) - len(stations) % _HISTORY_BLOCK_SIZE
        for start in range(0, full, _HISTORY_BLOCK_SIZE):
            blocks = _WorkHistoryBlock(
                blocks, tuple(stations[start : start + _HISTORY_BLOCK_SIZE])
            )
        return blocks, tuple(stations[full:])

    @staticmethod
    def _extend_fingerprint(previous, station):
        payload = json.dumps(
            station.snapshot(), sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        return sha256((previous + payload).encode()).hexdigest()

    def _fingerprint(self, stations):
        value = sha256(self.identity.encode()).hexdigest()
        for station in stations:
            value = self._extend_fingerprint(value, station)
        return value

    @property
    def path_work(self) -> float:
        return self._path_work

    @staticmethod
    def _integrated_work(stations) -> float:
        value = 0.0
        for previous, current in zip(stations[:-1], stations[1:]):
            average_force = 0.5 * (
                previous.generalized_force + current.generalized_force
            )
            increment = current.generalized_coordinate - previous.generalized_coordinate
            value += float(np.dot(average_force, increment))
        return value

    @property
    def latest_interval_power(self) -> float | None:
        if len(self.accepted) < 2:
            return None
        previous, current = self.accepted[-2:]
        dt = current.time - previous.time
        average_force = 0.5 * (previous.generalized_force + current.generalized_force)
        increment = current.generalized_coordinate - previous.generalized_coordinate
        return float(np.dot(average_force, increment) / dt)

    @property
    def current(self) -> PrescribedContactWorkStation | None:
        return self.accepted[-1] if self.accepted else None

    def initialize(self, station: PrescribedContactWorkStation) -> None:
        if self.accepted or self.trial is not None:
            raise RuntimeError("Prescribed-contact work State is already initialized.")
        station = self._validated_next(station, initial=True)
        fingerprint = self._extend_fingerprint(self._history_fingerprint, station)
        blocks, tail = self._appended_history(station)
        self.accepted.append(station)
        self._history_blocks, self._history_tail = blocks, tail
        self._history_fingerprint = fingerprint

    def begin(self, station: PrescribedContactWorkStation) -> None:
        if not self.accepted:
            raise RuntimeError(
                "Prescribed-contact work State must be initialized before a trial."
            )
        self.trial = self._validated_next(station, initial=False)

    def commit(self) -> PrescribedContactWorkStation:
        if self.trial is None:
            raise RuntimeError(
                "No prescribed-contact work trial is available to commit."
            )
        accepted = self.trial
        increment = self._integrated_work((self.accepted[-1], accepted))
        total = self._path_work + increment
        if not np.isfinite(total):
            raise ValueError("Non-finite accumulated prescribed contact work.")
        fingerprint = self._extend_fingerprint(self._history_fingerprint, accepted)
        blocks, tail = self._appended_history(accepted)
        self.accepted.append(accepted)
        self._history_blocks, self._history_tail = blocks, tail
        self._path_work = total
        self._history_fingerprint = fingerprint
        self.trial = None
        return accepted

    def rollback(self) -> None:
        self.trial = None

    def _validated_next(
        self,
        station: PrescribedContactWorkStation,
        *,
        initial: bool,
    ) -> PrescribedContactWorkStation:
        if not isinstance(station, PrescribedContactWorkStation):
            raise TypeError("Contact work State requires a work station.")
        if not initial:
            previous = self.accepted[-1]
            if station.time <= previous.time:
                raise ValueError("Accepted contact work times must strictly increase.")
            if station.generalized_force.shape != previous.generalized_force.shape:
                raise ValueError(
                    "Contact generalized-force layout changed during a run."
                )
            if station.contact_resultant.shape != previous.contact_resultant.shape:
                raise ValueError("Contact resultant layout changed during a run.")
        return station

    def snapshot(self) -> dict[str, object]:
        if self.trial is not None:
            raise RuntimeError(
                "Prescribed-contact work State can only be saved at an accepted boundary."
            )
        return {
            "schema": _STATE_SCHEMA,
            "identity": self.identity,
            "accepted": [station.snapshot() for station in self.accepted],
        }

    def transaction_snapshot(self):
        if self.trial is not None:
            raise RuntimeError(
                "Contact work transaction requires an accepted boundary."
            )
        return _WorkTransaction(self.identity, self._history_blocks, self._history_tail)

    def checkpoint_snapshot(self):
        """Detached columnar history for the existing numeric-tree archive.

        Retain every accepted station and the same work quadrature. The legacy
        JSON snapshot remains available; this changes encoding, not sampling.
        """
        if self.trial is not None:
            raise RuntimeError("Contact work checkpoint requires an accepted boundary.")
        columns = {
            name: np.asarray(
                [getattr(station, name) for station in self.accepted], dtype=np.float64
            )
            for name in _SCALAR_COLUMNS
        }
        for name in _VECTOR_COLUMNS:
            columns[name] = (
                np.stack([getattr(station, name) for station in self.accepted])
                if self.accepted
                else np.empty((0, 0), dtype=np.float64)
            )
        for name in _COUNT_COLUMNS:
            columns[name] = np.asarray(
                [getattr(station, name) for station in self.accepted], dtype=np.int64
            )
        return {
            "schema": _NUMERIC_STATE_SCHEMA,
            "identity": self.identity,
            "accepted": columns,
        }

    def restore(self, snapshot: object) -> None:
        if isinstance(snapshot, _WorkTransaction):
            if snapshot.identity != self.identity:
                raise ValueError("Prescribed-contact work State identity differs.")
            # Stations were validated before acceptance and are immutable.
            restored = list(snapshot.accepted)
            work = self._integrated_work(restored)
            if not np.isfinite(work):
                raise ValueError("Non-finite restored contact work.")
            fingerprint = self._fingerprint(restored)
            self.accepted, self._path_work, self.trial = restored, work, None
            self._history_fingerprint = fingerprint
            self._history_blocks, self._history_tail = snapshot.blocks, snapshot.tail
            return
        if not isinstance(snapshot, dict) or snapshot.get("schema") not in {
            _STATE_SCHEMA,
            _NUMERIC_STATE_SCHEMA,
        }:
            raise ValueError("Unsupported prescribed-contact work State snapshot.")
        if set(snapshot) != {"schema", "identity", "accepted"}:
            raise ValueError("Prescribed-contact work State fields do not match.")
        if snapshot["identity"] != self.identity:
            raise ValueError("Prescribed-contact work State identity differs.")
        raw = snapshot["accepted"]
        if snapshot["schema"] == _NUMERIC_STATE_SCHEMA:
            raw = _numeric_stations(raw)
        elif not isinstance(raw, list):
            raise ValueError("Prescribed-contact accepted stations must be a list.")
        restored: list[PrescribedContactWorkStation] = []
        for item in raw:
            station = PrescribedContactWorkStation.from_snapshot(item)
            if restored:
                previous = restored[-1]
                if station.time <= previous.time:
                    raise ValueError(
                        "Restored contact work times must strictly increase."
                    )
                if station.generalized_force.shape != previous.generalized_force.shape:
                    raise ValueError(
                        "Restored contact generalized-force layout changed."
                    )
                if station.contact_resultant.shape != previous.contact_resultant.shape:
                    raise ValueError("Restored contact resultant layout changed.")
            restored.append(station)
        work = self._integrated_work(restored)
        if not np.isfinite(work):
            raise ValueError("Non-finite restored contact work.")
        fingerprint = self._fingerprint(restored)
        blocks, tail = self._blocked_history(restored)
        self.accepted = restored
        self._path_work = work
        self._history_fingerprint = fingerprint
        self._history_blocks, self._history_tail = blocks, tail
        self.trial = None

    def summary(self) -> dict[str, object]:
        current = self.current
        return {
            "kind": "prescribed_contact_work_state",
            "identity": self.identity,
            "accepted_station_count": len(self.accepted),
            "path_work": self.path_work,
            "history_fingerprint": self._history_fingerprint,
            "latest_interval_power": self.latest_interval_power,
            "current": None if current is None else current.snapshot(),
            "trial_present": self.trial is not None,
            "integration": "trapezoidal_generalized_force_coordinate",
        }


def _numeric_stations(columns):
    """Validate the whole table before producing legacy-equivalent stations."""
    names = _SCALAR_COLUMNS + _VECTOR_COLUMNS + _COUNT_COLUMNS
    if not isinstance(columns, dict) or set(columns) != set(names):
        raise ValueError("Contact work numeric columns differ from the schema.")
    if any(not isinstance(columns[name], np.ndarray) for name in names):
        raise ValueError("Contact work numeric columns must be arrays.")
    times = columns["time"]
    if times.ndim != 1:
        raise ValueError("Contact work time column must be one-dimensional.")
    count = len(times)
    for name in names:
        value = columns[name]
        expected_dtype = np.dtype(np.int64 if name in _COUNT_COLUMNS else np.float64)
        if value.dtype != expected_dtype or not np.isfinite(value).all():
            raise ValueError("Contact work column dtype or finite values are invalid.")
        if name not in _VECTOR_COLUMNS:
            if value.shape != (count,):
                raise ValueError("Contact work scalar column length differs.")
        elif (
            value.ndim != 2
            or value.shape[0] != count
            or (count and value.shape[1] < 1)
            or (not count and value.shape[1] != 0)
        ):
            raise ValueError("Contact work vector column layout differs.")
    if columns["generalized_force"].shape != columns["generalized_coordinate"].shape:
        raise ValueError("Contact work generalized column dimensions differ.")
    if count and columns["contact_resultant"].shape[1] not in (2, 3):
        raise ValueError("Contact work resultant dimension is invalid.")
    # Station constructors and restore still own value ranges, monotonic time,
    # cumulative-work reconstruction, fingerprint, and atomic publication.
    return (
        {
            "schema": _STATION_SCHEMA,
            **{
                name: columns[name][i]
                if name in _VECTOR_COLUMNS
                else columns[name][i].item()
                for name in names
            },
        }
        for i in range(count)
    )


def prescribed_contact_work_state(*, identity: str) -> PrescribedContactWorkState:
    return PrescribedContactWorkState(identity=identity)


def prescribed_rigid_motion_schedule(
    motion: PrescribedRigidMotion,
    *,
    start_time: float = 0.0,
    end_time: float,
    name: str = "prescribed_rigid_motion_schedule",
    amplitude=None,
) -> PrescribedRigidMotionSchedule:
    return PrescribedRigidMotionSchedule(
        motion=motion,
        start_time=start_time,
        end_time=end_time,
        name=name,
        amplitude=amplitude,
    )


__all__ = [
    "PrescribedContactWorkState",
    "PrescribedContactWorkStation",
    "PrescribedRigidMotionSchedule",
    "prescribed_contact_work_state",
    "prescribed_rigid_motion_schedule",
]

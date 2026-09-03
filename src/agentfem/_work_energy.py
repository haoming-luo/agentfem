"""Solver-neutral generalized work and transactional energy accounting.

This module owns the work-conjugate boundary shared by static, transient,
cyclic, constraint and contact providers.  Physics-specific procedures may
re-export these objects for compatibility, but they must not invent a second
definition of external work or silently omit a provider-owned dual channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


_GENERALIZED_WORK_ROLES = frozenset(
    {
        "natural_load",
        "reference_point",
        "prescribed_motion",
        "mpc_constraint",
        "weak_constraint",
        "contact_constraint",
    }
)


def _finite_array(value, *, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(selected)):
        raise ValueError(f"{name} must contain only finite values.")
    return selected


@dataclass(frozen=True)
class GeneralizedWorkSample:
    """One named force--coordinate pair at an accepted equilibrium point."""

    name: str
    role: str
    force: np.ndarray
    displacement: np.ndarray

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        role = str(self.role).strip().lower().replace("-", "_")
        if not name or role not in _GENERALIZED_WORK_ROLES:
            raise ValueError("Generalized work needs a name and a supported role.")
        force = _finite_array(self.force, name="generalized_force").reshape(-1)
        displacement = _finite_array(
            self.displacement, name="generalized_displacement"
        ).reshape(-1)
        if force.size == 0 or force.shape != displacement.shape:
            raise ValueError(
                "Generalized force and displacement must have one matching layout."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "force", force.copy())
        object.__setattr__(self, "displacement", displacement.copy())

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "force": self.force.tolist(),
            "displacement": self.displacement.tolist(),
        }


def generalized_work_sample(
    name,
    *,
    force,
    displacement,
    role="natural_load",
) -> GeneralizedWorkSample:
    """Declare one generalized work-conjugate channel."""

    return GeneralizedWorkSample(
        name=name,
        role=role,
        force=force,
        displacement=displacement,
    )


def reference_point_work_sample(
    load,
    *,
    translation,
    rotation=None,
) -> GeneralizedWorkSample:
    """Pair a distributed reference load with measured rigid motion."""

    force = np.asarray(getattr(load, "force"), dtype=float).reshape(-1)
    displacement = np.asarray(translation, dtype=float).reshape(-1)
    raw_moment = getattr(load, "moment", None)
    moment = (
        np.empty(0, dtype=float)
        if raw_moment is None
        else np.asarray(raw_moment, dtype=float).reshape(-1)
    )
    if rotation is None:
        if moment.size and np.any(np.abs(moment) > 0.0):
            raise ValueError("A nonzero reference-point moment requires rotation.")
    else:
        selected_rotation = np.asarray(rotation, dtype=float).reshape(-1)
        if not (moment.size == selected_rotation.size or (
            moment.size == 1 and selected_rotation.size == 1
        )):
            raise ValueError("Reference-point moment and rotation layouts differ.")
        force = np.concatenate((force, moment))
        displacement = np.concatenate((displacement, selected_rotation))
    return GeneralizedWorkSample(
        name=getattr(load, "reference_name", None) or getattr(load, "name", "RP"),
        role="reference_point",
        force=force,
        displacement=displacement,
    )


@dataclass(frozen=True)
class CyclicEnergyFrame:
    """One accepted or trial cycle-block work--energy closure."""

    start_cycle: int
    end_cycle: int
    cycles: int
    resolved_cycle_work: float
    estimated_skipped_cycle_work: float
    block_external_work: float
    resolved_channel_work: dict[str, float]
    block_channel_work: dict[str, float]
    block_role_work: dict[str, float]
    energy_channel_increment: dict[str, float]
    accounted_energy_increment: float
    balance_error: float
    relative_balance_error: float
    estimation: str

    def summary(self) -> dict[str, object]:
        return {"kind": "cyclic_work_energy_frame", **self.__dict__}


class CyclicWorkEnergyLedger:
    """Transactional generalized-work and cycle-block energy accounting."""

    _SCHEMA = "agentfem.cyclic-work-energy-ledger.v1"

    def __init__(self, *, name="cyclic work-energy ledger"):
        self.name = str(name)
        self.frames: list[CyclicEnergyFrame] = []
        self._trial: CyclicEnergyFrame | None = None

    @staticmethod
    def _coerce_sample(value) -> GeneralizedWorkSample:
        if isinstance(value, GeneralizedWorkSample):
            return value
        if isinstance(value, dict):
            return GeneralizedWorkSample(**value)
        raise TypeError("Generalized-work entries must be samples or mappings.")

    def begin_block(
        self,
        stations,
        *,
        start_cycle: int,
        cycles: int,
        energy_endpoints=None,
    ) -> CyclicEnergyFrame:
        if self._trial is not None:
            raise RuntimeError("A cyclic energy block is already active.")
        points = tuple(stations)
        count = int(cycles)
        if len(points) < 2 or count < 1:
            raise ValueError("Cycle energy needs two stations and positive cycles.")
        channels_by_station = []
        for point in points:
            samples = {
                sample.name: sample
                for sample in (
                    self._coerce_sample(value)
                    for value in getattr(point, "generalized_work", ())
                )
            }
            if len(samples) != len(getattr(point, "generalized_work", ())):
                raise ValueError("Generalized-work channel names must be unique.")
            channels_by_station.append(samples)
        names = set(channels_by_station[0])
        if not names or any(set(item) != names for item in channels_by_station):
            raise ValueError(
                "Every cycle station must expose the same generalized-work channels."
            )
        endpoints = points if energy_endpoints is None else tuple(energy_endpoints)
        if len(endpoints) != 2:
            raise ValueError("Energy accounting requires exactly two block endpoints.")
        energy_by_endpoint = [
            dict(getattr(point, "energy_channels", {})) for point in endpoints
        ]
        energy_names = set(energy_by_endpoint[0])
        if not energy_names or set(energy_by_endpoint[1]) != energy_names:
            raise ValueError("Both block endpoints must expose the same energy channels.")
        for values in energy_by_endpoint:
            if any(not isfinite(float(value)) for value in values.values()):
                raise ValueError("Energy channels must contain finite total values.")
        channel_work = {name: 0.0 for name in sorted(names)}
        roles = {}
        for left, right in zip(channels_by_station[:-1], channels_by_station[1:]):
            for name in channel_work:
                first = left[name]
                second = right[name]
                if first.role != second.role or first.force.shape != second.force.shape:
                    raise ValueError(f"Generalized-work channel {name!r} changed contract.")
                increment = second.displacement - first.displacement
                channel_work[name] += 0.5 * float(
                    np.dot(first.force + second.force, increment)
                )
                roles[name] = first.role
        resolved = float(sum(channel_work.values()))
        skipped = resolved * max(count - 1, 0)
        block_channel_work = {
            name: float(value) * count for name, value in channel_work.items()
        }
        role_work: dict[str, float] = {}
        for name, value in block_channel_work.items():
            role = roles[name]
            role_work[role] = role_work.get(role, 0.0) + value
        increments = {
            name: float(energy_by_endpoint[-1][name])
            - float(energy_by_endpoint[0][name])
            for name in sorted(energy_names)
        }
        block_work = resolved + skipped
        accounted = float(sum(increments.values()))
        balance = block_work - accounted
        scale = max(abs(block_work), abs(accounted), np.finfo(float).eps)
        self._trial = CyclicEnergyFrame(
            start_cycle=int(start_cycle),
            end_cycle=int(start_cycle) + count,
            cycles=count,
            resolved_cycle_work=resolved,
            estimated_skipped_cycle_work=skipped,
            block_external_work=block_work,
            resolved_channel_work=channel_work,
            block_channel_work=block_channel_work,
            block_role_work=role_work,
            energy_channel_increment=increments,
            accounted_energy_increment=accounted,
            balance_error=balance,
            relative_balance_error=abs(balance) / scale,
            estimation=(
                "resolved_single_cycle"
                if count == 1
                else "representative_resolved_cycle_times_cycle_block"
            ),
        )
        return self._trial

    def commit(self) -> None:
        if self._trial is None:
            raise RuntimeError("No cyclic energy block is active.")
        self.frames.append(self._trial)
        self._trial = None

    def rollback(self) -> None:
        self._trial = None

    def snapshot(self) -> dict[str, object]:
        if self._trial is not None:
            raise RuntimeError("Commit or rollback energy state before checkpoint.")
        return {
            "schema": self._SCHEMA,
            "name": self.name,
            "frames": [frame.summary() for frame in self.frames],
        }

    def restore(self, snapshot) -> None:
        if snapshot.get("schema") != self._SCHEMA or snapshot.get("name") != self.name:
            raise ValueError("Cyclic energy-ledger identity differs.")
        self.frames = []
        for record in snapshot.get("frames", []):
            selected = dict(record)
            if selected.pop("kind", None) != "cyclic_work_energy_frame":
                raise ValueError("Cyclic energy frame schema differs.")
            self.frames.append(CyclicEnergyFrame(**selected))
        self._trial = None


def cyclic_work_energy_ledger(**options) -> CyclicWorkEnergyLedger:
    """Create a transactional cycle-block work--energy ledger."""

    return CyclicWorkEnergyLedger(**options)


__all__ = (
    "CyclicEnergyFrame",
    "CyclicWorkEnergyLedger",
    "GeneralizedWorkSample",
    "cyclic_work_energy_ledger",
    "generalized_work_sample",
    "reference_point_work_sample",
)

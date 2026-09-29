# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted-boundary energy evidence for conservative nonlinear paths."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AcceptedEnergyFrame:
    """Lightweight scalar state at one accepted load boundary."""

    load_factor: float
    natural_load_coordinate: float
    stored_energy_components: dict[str, float]

    @property
    def stored_energy(self) -> float:
        return float(sum(self.stored_energy_components.values()))

    def as_dict(self) -> dict[str, object]:
        return {
            "load_factor": float(self.load_factor),
            "natural_load_coordinate": float(self.natural_load_coordinate),
            "stored_energy_components": dict(self.stored_energy_components),
            "stored_energy": self.stored_energy,
        }

    @classmethod
    def from_dict(cls, payload) -> "AcceptedEnergyFrame":
        if not isinstance(payload, dict):
            raise TypeError("Accepted energy frames must be mappings.")
        factor = float(payload["load_factor"])
        coordinate = float(payload["natural_load_coordinate"])
        components = {
            str(name): float(value)
            for name, value in dict(payload["stored_energy_components"]).items()
        }
        if (
            not np.isfinite(factor)
            or not np.isfinite(coordinate)
            or not components
            or not all(np.isfinite(value) for value in components.values())
        ):
            raise ValueError("Accepted energy frames require finite scalar data.")
        return cls(factor, coordinate, components)


@dataclass
class AcceptedConservativeEnergyRecorder:
    """Record scalar work and stored-energy evidence at every accepted state.

    The callbacks receive the copied accepted solution, keeping the recorder
    independent of the nonlinear backend.  It retains only scalar evidence,
    not full spatial fields, so long load paths remain inexpensive.
    """

    stored_energy_evaluators: dict[str, object]
    natural_load_coordinate_evaluator: object | None = None
    proportional_dead_load: bool = True
    zero_prescribed_motion: bool = True
    frames: list[AcceptedEnergyFrame] = field(default_factory=list)

    def __post_init__(self) -> None:
        selected = {
            str(name): evaluator
            for name, evaluator in dict(self.stored_energy_evaluators).items()
        }
        if not selected or any(not callable(item) for item in selected.values()):
            raise TypeError("Stored-energy evaluators must be a non-empty mapping.")
        if (
            self.natural_load_coordinate_evaluator is not None
            and not callable(self.natural_load_coordinate_evaluator)
        ):
            raise TypeError("Natural-load coordinate evaluator must be callable.")
        self.stored_energy_evaluators = selected

    def _capture(self, snapshot) -> AcceptedEnergyFrame:
        solution = snapshot.solution
        coordinate = (
            0.0
            if self.natural_load_coordinate_evaluator is None
            else float(self.natural_load_coordinate_evaluator(solution))
        )
        components = {
            name: float(evaluator(solution))
            for name, evaluator in self.stored_energy_evaluators.items()
        }
        frame = AcceptedEnergyFrame(
            float(snapshot.load_factor),
            coordinate,
            components,
        )
        # Reuse durable validation at the point of capture.
        return AcceptedEnergyFrame.from_dict(frame.as_dict())

    def reset(self, snapshot) -> None:
        self.frames[:] = [self._capture(snapshot)]

    def prepare_resume(self, snapshot) -> None:
        if not self.frames:
            self.reset(snapshot)
            return
        if abs(self.frames[-1].load_factor - float(snapshot.load_factor)) > 1.0e-12:
            raise ValueError(
                "Accepted energy history does not end at the resume boundary."
            )

    def accept(self, snapshot) -> None:
        frame = self._capture(snapshot)
        if self.frames and frame.load_factor <= self.frames[-1].load_factor + 1.0e-12:
            raise ValueError("Accepted energy factors must increase strictly.")
        self.frames.append(frame)

    def snapshot_runtime_state(self) -> dict[str, object]:
        return {"frames": list(self.frames)}

    def restore_runtime_state(self, state) -> None:
        self.frames[:] = list(state["frames"])

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "schema": "agentfem.accepted-conservative-energy.v1",
            "proportional_dead_load": bool(self.proportional_dead_load),
            "zero_prescribed_motion": bool(self.zero_prescribed_motion),
            "frames": [item.as_dict() for item in self.frames],
        }

    def restore_checkpoint_state(self, record, *, current_snapshot) -> None:
        if not isinstance(record, dict):
            raise TypeError("Accepted conservative-energy state must be a mapping.")
        if record.get("schema") != "agentfem.accepted-conservative-energy.v1":
            raise ValueError("Unsupported accepted conservative-energy schema.")
        for key in ("proportional_dead_load", "zero_prescribed_motion"):
            if type(record.get(key)) is not bool:
                raise TypeError(f"Accepted energy checkpoint {key!r} must be boolean.")
        if record["proportional_dead_load"] != self.proportional_dead_load or record[
            "zero_prescribed_motion"
        ] != self.zero_prescribed_motion:
            raise ValueError("Accepted energy path semantics differ from checkpoint.")
        raw_frames = record.get("frames")
        if not isinstance(raw_frames, list):
            raise TypeError("Accepted energy checkpoint frames must be a list.")
        frames = [AcceptedEnergyFrame.from_dict(item) for item in raw_frames]
        if not frames or any(
            current.load_factor <= previous.load_factor + 1.0e-12
            for previous, current in zip(frames, frames[1:])
        ):
            raise ValueError("Accepted energy checkpoint path is not increasing.")
        if abs(frames[-1].load_factor - float(current_snapshot.load_factor)) > 1.0e-12:
            raise ValueError(
                "Accepted energy checkpoint does not end at the restored boundary."
            )
        expected = set(self.stored_energy_evaluators)
        if any(set(item.stored_energy_components) != expected for item in frames):
            raise ValueError("Accepted energy component schema changed at restart.")
        self.frames[:] = frames

    def evidence(self, *, accepted_factor: float) -> dict[str, object]:
        if (
            len(self.frames) < 2
            or abs(self.frames[-1].load_factor - float(accepted_factor)) > 1.0e-12
        ):
            return {
                "status": "unavailable",
                "reason": "accepted energy history is incomplete",
            }
        factors = np.asarray([item.load_factor for item in self.frames], dtype=float)
        coordinates = np.asarray(
            [item.natural_load_coordinate for item in self.frames],
            dtype=float,
        )
        natural_work = float(
            np.sum(0.5 * (factors[1:] + factors[:-1]) * np.diff(coordinates))
        )
        component_changes = {
            name: float(
                self.frames[-1].stored_energy_components[name]
                - self.frames[0].stored_energy_components[name]
            )
            for name in self.stored_energy_evaluators
        }
        stored_change = float(sum(component_changes.values()))
        reasons = []
        if not self.proportional_dead_load:
            reasons.append("natural loads are not a fixed proportional dead-load path")
        if not self.zero_prescribed_motion:
            reasons.append("nonzero prescribed-motion work is not yet recorded")
        scale = max(abs(natural_work), abs(stored_change), np.finfo(float).eps)
        error = natural_work - stored_change
        return {
            "status": "complete" if not reasons else "unavailable",
            "reason": None if not reasons else "; ".join(reasons),
            "integration": "accepted_boundary_trapezoidal_force_displacement",
            "sample_count": len(self.frames),
            "load_factors": tuple(float(value) for value in factors),
            "natural_load_coordinate": tuple(float(value) for value in coordinates),
            "natural_load_work": natural_work,
            "stored_energy_change": stored_change,
            "stored_energy_components": component_changes,
            "energy_balance_error": float(error),
            "relative_energy_balance_error": float(abs(error) / scale),
        }

    def summary(self) -> dict[str, object]:
        return {
            "kind": "accepted_conservative_energy",
            "frame_count": len(self.frames),
            "components": tuple(self.stored_energy_evaluators),
            "proportional_dead_load": bool(self.proportional_dead_load),
            "zero_prescribed_motion": bool(self.zero_prescribed_motion),
        }


__all__ = ("AcceptedConservativeEnergyRecorder", "AcceptedEnergyFrame")

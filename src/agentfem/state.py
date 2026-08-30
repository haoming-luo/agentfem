"""Shared state ownership and transaction contracts.

State is neither a constitutive law nor a solver.  It owns accepted/trial
history and restart boundaries.  How a trial begins may remain specific to an
increment, cycle, or material algorithm; commit, rollback, snapshot, and
restore capabilities stay explicit and inspectable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from . import fields, spaces, time
from .kernel import dofs


@runtime_checkable
class RestartableState(Protocol):
    """State whose accepted scientific identity can cross a restart."""

    def snapshot(self) -> object:
        """Return a detached representation of the owned state."""

        ...

    def restore(self, snapshot: object) -> None:
        """Restore a compatible detached representation."""

        ...


@runtime_checkable
class ReplaceableState(RestartableState, Protocol):
    """State with an atomic trial/accept/reject boundary."""

    def commit(self) -> None:
        """Accept the current trial state."""

        ...

    def rollback(self) -> None:
        """Reject the current trial and restore the accepted boundary."""

        ...


@dataclass(frozen=True)
class StateCapabilities:
    """Inspectable transaction features without guessing from class names."""

    restartable: bool
    replaceable: bool
    begins_trial: bool
    increment_transaction: bool
    cycle_transaction: bool

    def summary(self) -> dict[str, bool]:
        return {
            "restartable": self.restartable,
            "replaceable": self.replaceable,
            "begins_trial": self.begins_trial,
            "increment_transaction": self.increment_transaction,
            "cycle_transaction": self.cycle_transaction,
        }


def capabilities(value: object) -> StateCapabilities:
    """Describe the transaction boundary implemented by ``value``.

    ``begin`` is intentionally not required by :class:`ReplaceableState`:
    beginning a material trial, a load increment, or an ordered fatigue cycle
    needs different physical inputs.  Treating those operations as identical
    would hide rather than simplify the science.
    """

    def has(name: str) -> bool:
        return callable(getattr(value, name, None))

    restartable = has("snapshot") and has("restore")
    replaceable = restartable and has("commit") and has("rollback")
    return StateCapabilities(
        restartable=restartable,
        replaceable=replaceable,
        begins_trial=has("begin"),
        increment_transaction=(
            has("commit_increment")
            and has("rollback_increment")
            and restartable
        ),
        cycle_transaction=has("begin_cycle") and (
            has("commit_cycle") or has("commit")
        ) and has("rollback"),
    )


def require_restartable(value: object, *, name: str = "state") -> RestartableState:
    """Return ``value`` or fail with an addressable ownership error."""

    selected = capabilities(value)
    if not selected.restartable:
        raise TypeError(
            f"{name} must own snapshot() and restore() for restartable execution."
        )
    return value  # type: ignore[return-value]


def require_replaceable(value: object, *, name: str = "state") -> ReplaceableState:
    """Return ``value`` or fail unless it owns atomic accept/reject semantics."""

    selected = capabilities(value)
    if not selected.replaceable:
        raise TypeError(
            f"{name} must own commit(), rollback(), snapshot(), and restore()."
        )
    return value  # type: ignore[return-value]


def _array(value) -> np.ndarray:
    selected = fields.unwrap(value)
    return np.asarray(selected.x.array)


def _assign(value, data, *, label: str) -> None:
    selected = fields.unwrap(value)
    restored = np.asarray(data, dtype=selected.x.array.dtype)
    if restored.shape != selected.x.array.shape:
        raise ValueError(
            f"State field {label!r} shape changed from {restored.shape} "
            f"to {selected.x.array.shape}."
        )
    selected.x.array[:] = restored
    if callable(getattr(selected.x, "scatter_forward", None)):
        selected.x.scatter_forward()


def _snapshot_record(snapshot: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(snapshot, Mapping):
        raise TypeError(f"{label} snapshot must be a mapping, not {type(snapshot).__name__}.")
    return snapshot


@dataclass
class TransientState:
    """Accepted/trial fields for a first-order transient unknown."""

    current: object
    next: object

    @classmethod
    def create(cls, V, *, name: str = "Field"):
        return cls(
            current=spaces.named_function(V, name),
            next=spaces.named_function(V, name),
        )

    def accept_step(self) -> None:
        dofs.copy_function(self.current, self.next)

    def commit(self) -> None:
        self.accept_step()

    def rollback(self) -> None:
        dofs.copy_function(self.next, self.current)

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "agentfem.transient-state.v1",
            "current": _array(self.current).copy(),
            "next": _array(self.next).copy(),
        }

    def restore(self, snapshot: object) -> None:
        record = _snapshot_record(snapshot, label="Transient state")
        if record.get("schema") != "agentfem.transient-state.v1":
            raise ValueError("Unsupported transient-state snapshot schema.")
        _assign(self.current, record["current"], label="current")
        _assign(self.next, record["next"], label="next")


@dataclass
class SecondOrderDynamicsState:
    """Accepted/trial displacement, velocity, and acceleration fields."""

    u: object
    v: object
    a: object
    v_mid: object
    u_next: object
    v_next: object
    a_next: object

    @classmethod
    def create(
        cls,
        V,
        *,
        displacement_name: str = "Displacement",
        velocity_name: str = "Velocity",
        acceleration_name: str = "Acceleration",
    ):
        return cls(
            u=fields.wrap(spaces.named_function(V, displacement_name)),
            v=fields.wrap(spaces.named_function(V, velocity_name)),
            a=fields.wrap(spaces.named_function(V, acceleration_name)),
            v_mid=fields.wrap(spaces.named_function(V, f"{velocity_name}_Midstep")),
            u_next=fields.wrap(spaces.named_function(V, displacement_name)),
            v_next=fields.wrap(spaces.named_function(V, velocity_name)),
            a_next=fields.wrap(spaces.named_function(V, acceleration_name)),
        )

    def predict_displacement(self, dt: float) -> None:
        time.central_difference_predict_displacement(
            self.u_next, self.u, self.v, self.a, dt
        )

    def set_acceleration_from_residual(self, residual, inv_mass: np.ndarray) -> None:
        time.acceleration_from_residual(self.a_next, residual, inv_mass)

    def update_midstep_velocity(self, dt: float) -> None:
        time.central_difference_update_midstep_velocity(
            self.v_mid, self.v, self.a, dt
        )

    def correct_velocity(self, dt: float) -> None:
        time.central_difference_correct_velocity(
            self.v_next, self.v, self.a, self.a_next, dt
        )

    def update_velocity(self, dt: float) -> None:
        time.central_difference_update_velocity(
            self.v_next, self.v_mid, self.a_next, dt
        )

    def update_displacement(self) -> None:
        dofs.copy_function(self.u, self.u_next)

    def advance_state(self) -> None:
        dofs.copy_function(self.u, self.u_next)
        dofs.copy_function(self.v, self.v_next)
        dofs.copy_function(self.a, self.a_next)

    def accept_step(self) -> None:
        self.advance_state()

    def accept_displacement(self) -> None:
        self.update_displacement()

    def accept_velocity_acceleration(self) -> None:
        self.advance_velocity_acceleration()

    def advance_velocity_acceleration(self) -> None:
        dofs.copy_function(self.v, self.v_next)
        dofs.copy_function(self.a, self.a_next)

    def commit(self) -> None:
        self.advance_state()

    def rollback(self) -> None:
        dofs.copy_function(self.u_next, self.u)
        dofs.copy_function(self.v_next, self.v)
        dofs.copy_function(self.a_next, self.a)
        dofs.copy_function(self.v_mid, self.v)

    def snapshot(self) -> dict[str, object]:
        names = ("u", "v", "a", "v_mid", "u_next", "v_next", "a_next")
        return {
            "schema": "agentfem.second-order-dynamics-state.v1",
            "fields": {
                name: _array(getattr(self, name)).copy() for name in names
            },
        }

    def restore(self, snapshot: object) -> None:
        record = _snapshot_record(snapshot, label="Second-order dynamics state")
        if record.get("schema") != "agentfem.second-order-dynamics-state.v1":
            raise ValueError("Unsupported second-order dynamics snapshot schema.")
        raw_fields = record.get("fields")
        if not isinstance(raw_fields, Mapping):
            raise TypeError("Second-order dynamics snapshot fields must be a mapping.")
        stored = dict(raw_fields)
        names = ("u", "v", "a", "v_mid", "u_next", "v_next", "a_next")
        if set(stored) != set(names):
            raise ValueError("Second-order dynamics snapshot fields differ.")
        for name in names:
            _assign(getattr(self, name), stored[name], label=name)


ExplicitDynamicsState = SecondOrderDynamicsState


def second_order_state(field_or_space, **kwargs) -> SecondOrderDynamicsState:
    """Create a second-order state from a field or function space."""

    source = None
    if hasattr(field_or_space, "space"):
        V = field_or_space.space
        source = getattr(field_or_space, "value", None)
    elif hasattr(field_or_space, "function_space"):
        V = field_or_space.function_space
        source = field_or_space
    else:
        V = field_or_space
    selected = SecondOrderDynamicsState.create(V, **kwargs)
    if source is not None:
        selected.u = fields.wrap(source)
        selected.u_next.assign(source)
    return selected


__all__ = (
    "ExplicitDynamicsState",
    "ReplaceableState",
    "RestartableState",
    "SecondOrderDynamicsState",
    "StateCapabilities",
    "TransientState",
    "capabilities",
    "require_replaceable",
    "require_restartable",
    "second_order_state",
)

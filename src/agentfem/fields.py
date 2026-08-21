"""Application-level unknown fields.

This layer bundles the finite-element space, solution field, trial function,
and test function so beginner workflows can talk about displacement or
temperature instead of V/u/du/v bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from . import spaces
from .kernel import dofs


@dataclass
class Field:
    """Tensor-like finite-element field with immediate-value algebra.

    Arithmetic is intentionally eager: operations between compatible fields
    create a new DOLFINx function with computed dof values. This mirrors CAE
    field algebra such as Cast3M ``vp = vit0 + res1 * acc0``.
    """

    function: fem.Function
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            self.function.name = self.name

    @property
    def x(self):
        """Return the underlying vector data."""

        return self.function.x

    @property
    def function_space(self):
        """Return the underlying function space."""

        return self.function.function_space

    @property
    def ufl_shape(self):
        """Return the UFL value shape."""

        return self.function.ufl_shape

    @property
    def value(self):
        """Compatibility alias for workflows expecting a field value."""

        return self.function

    def __getattr__(self, name: str):
        return getattr(self.function, name)

    def __len__(self) -> int:
        shape = getattr(self.function, "ufl_shape", ())
        if len(shape) == 0:
            raise TypeError("Scalar fields do not have len().")
        return int(shape[0])

    def __add__(self, other):
        return _binary_field_op(self, other, np.add, "add")

    def __radd__(self, other):
        return _binary_field_op(other, self, np.add, "add")

    def __sub__(self, other):
        return _binary_field_op(self, other, np.subtract, "sub")

    def __rsub__(self, other):
        return _binary_field_op(other, self, np.subtract, "sub")

    def __mul__(self, other):
        return _binary_field_op(self, other, np.multiply, "mul")

    def __rmul__(self, other):
        return _binary_field_op(other, self, np.multiply, "mul")

    def __truediv__(self, other):
        return _binary_field_op(self, other, np.divide, "div")

    def __rtruediv__(self, other):
        return _binary_field_op(other, self, np.divide, "div")

    def __neg__(self):
        result = empty_like(self, name=f"neg_{self.function.name}")
        result.x.array[:] = -self.x.array
        result.x.scatter_forward()
        return result

    def assign(self, source) -> None:
        """Assign another compatible field, DOLFINx function, or scalar."""

        assign(self, source)

    def copy(self, *, name: str | None = None) -> "Field":
        """Return a numerical copy of this field."""

        result = empty_like(self, name=name or self.function.name)
        dofs.copy_function(result.function, self.function)
        return result

    def summary(self) -> dict[str, object]:
        """Return a compact field summary."""

        return {
            "name": self.function.name,
            "kind": "field",
            "shape": self.ufl_shape,
            "space": str(self.function_space.ufl_element()),
        }


@dataclass(frozen=True)
class UnknownField:
    """Finite-element unknown bundle for application-level workflows."""

    name: str
    space: object
    value: object
    trial: object
    test: object
    kind: str = "unknown"

    @property
    def function_space(self):
        """Compatibility alias for the underlying function space."""

        return self.space

    def summary(self) -> dict[str, object]:
        """Return an inspectable field summary."""

        element = self.space.ufl_element()
        return {
            "name": self.name,
            "kind": self.kind,
            "element": str(element),
            "value_name": getattr(self.value, "name", self.name),
        }

    def assign_from(self, other) -> None:
        """Copy values from another unknown field or DOLFINx function."""

        source = other.value if hasattr(other, "value") else other
        self.value.x.array[:] = source.x.array
        self.value.x.scatter_forward()

    def max_value(self) -> float:
        """Return the distributed maximum value of the field."""

        local = float(np.max(self.value.x.array))
        return self.value.function_space.mesh.comm.allreduce(local, op=MPI.MAX)

    def max_abs(self) -> float:
        """Return the distributed maximum absolute value of the field."""

        local = float(np.max(np.abs(self.value.x.array)))
        return self.value.function_space.mesh.comm.allreduce(local, op=MPI.MAX)


@dataclass(frozen=True)
class DisplacementPressureUnknown:
    """Mixed displacement/pressure unknown for hybrid solid mechanics.

    ``displacement`` and ``pressure`` are views of one monolithic mixed
    solution.  Users apply mechanical constraints to ``displacement`` while
    the nonlinear problem solves both fields together.
    """

    name: str
    space: object
    value: object
    trial: tuple[object, object]
    test: tuple[object, object]
    displacement: UnknownField
    pressure: UnknownField
    displacement_degree: int = 2
    pressure_degree: int = 0
    kind: str = "displacement_pressure"

    @property
    def function_space(self):
        return self.space

    @property
    def ufl_shape(self):
        return None

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "element": str(self.space.ufl_element()),
            "displacement_degree": int(self.displacement_degree),
            "pressure_degree": int(self.pressure_degree),
            "pressure_interpolation": "cellwise_discontinuous",
            "pressure_unknowns_per_cell": 1 if self.pressure_degree == 0 else None,
        }

    def collapsed_displacement(self, *, name: str = "U"):
        """Return a standalone displacement field copied from the mixed state."""

        result = self.value.sub(0).collapse()
        result.name = name
        return result

    def collapsed_pressure(self, *, name: str = "P"):
        """Return a standalone pressure field copied from the mixed state."""

        result = self.value.sub(1).collapse()
        result.name = name
        return result


@dataclass(frozen=True)
class VelocityPressureUnknown:
    """Taylor--Hood velocity/pressure unknown for incompressible flow."""

    name: str
    space: object
    value: object
    trial: tuple[object, object]
    test: tuple[object, object]
    velocity: UnknownField
    pressure: UnknownField
    velocity_degree: int = 2
    pressure_degree: int = 1
    kind: str = "velocity_pressure"

    @property
    def function_space(self):
        return self.space

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "element": str(self.space.ufl_element()),
            "velocity_degree": int(self.velocity_degree),
            "pressure_degree": int(self.pressure_degree),
            "stability_family": "Taylor-Hood",
        }

    def collapsed_velocity(self, *, name: str = "V"):
        """Return a standalone velocity field copied from the mixed state."""

        result = self.value.sub(0).collapse()
        result.name = name
        return result

    def collapsed_pressure(self, *, name: str = "P"):
        """Return a standalone pressure field copied from the mixed state."""

        result = self.value.sub(1).collapse()
        result.name = name
        return result


def scalar_unknown(domain, *, name: str = "Unknown", degree: int = 1, value=0.0) -> UnknownField:
    """Create a scalar finite-element unknown."""

    V = spaces.scalar_space(domain, degree=degree)
    return UnknownField(
        name=name,
        kind="scalar_unknown",
        space=V,
        value=spaces.named_function(V, name, value=value),
        trial=spaces.trial_function(V),
        test=spaces.test_function(V),
    )


def vector_unknown(
    domain,
    *,
    name: str = "Unknown",
    degree: int = 1,
    dim: int | None = None,
    value=0.0,
) -> UnknownField:
    """Create a vector finite-element unknown."""

    V = spaces.vector_space(domain, degree=degree, dim=dim)
    return UnknownField(
        name=name,
        kind="vector_unknown",
        space=V,
        value=spaces.named_function(V, name, value=value),
        trial=spaces.trial_function(V),
        test=spaces.test_function(V),
    )


def displacement(domain, *, degree: int = 1, dim: int | None = None, value=0.0) -> UnknownField:
    """Create a displacement unknown for mechanics workflows."""

    field = vector_unknown(domain, name="Displacement", degree=degree, dim=dim, value=value)
    return UnknownField(
        name=field.name,
        kind="displacement",
        space=field.space,
        value=field.value,
        trial=field.trial,
        test=field.test,
    )


def displacement_pressure(
    domain,
    *,
    displacement_degree: int = 2,
    pressure_degree: int = 0,
    name: str = "DisplacementPressure",
) -> DisplacementPressureUnknown:
    """Create a mixed displacement/pressure unknown.

    The default has quadratic displacement interpolation and one constant
    pressure value per cell.  Apply constraints through
    ``unknown.displacement`` and request pressure output through
    ``unknown.collapsed_pressure()``.
    """

    W = spaces.displacement_pressure_space(
        domain,
        displacement_degree=displacement_degree,
        pressure_degree=pressure_degree,
    )
    mixed_value = fem.Function(W, name=name)
    trial = tuple(ufl.TrialFunctions(W))
    test = tuple(ufl.TestFunctions(W))
    displacement_value = mixed_value.sub(0)
    displacement_value.name = "Displacement"
    pressure_value = mixed_value.sub(1)
    pressure_value.name = "Pressure"
    displacement_unknown = UnknownField(
        name="Displacement",
        kind="displacement_subfield",
        space=W.sub(0),
        value=displacement_value,
        trial=trial[0],
        test=test[0],
    )
    pressure_unknown = UnknownField(
        name="Pressure",
        kind="pressure_subfield",
        space=W.sub(1),
        value=pressure_value,
        trial=trial[1],
        test=test[1],
    )
    return DisplacementPressureUnknown(
        name=name,
        space=W,
        value=mixed_value,
        trial=trial,
        test=test,
        displacement=displacement_unknown,
        pressure=pressure_unknown,
        displacement_degree=int(displacement_degree),
        pressure_degree=int(pressure_degree),
    )


def velocity_pressure(
    domain,
    *,
    velocity_degree: int = 2,
    pressure_degree: int = 1,
    name: str = "VelocityPressure",
) -> VelocityPressureUnknown:
    """Create a Taylor--Hood incompressible-flow unknown."""

    W = spaces.velocity_pressure_space(
        domain,
        velocity_degree=velocity_degree,
        pressure_degree=pressure_degree,
    )
    mixed_value = fem.Function(W, name=name)
    trial = tuple(ufl.TrialFunctions(W))
    test = tuple(ufl.TestFunctions(W))
    velocity_value = mixed_value.sub(0)
    velocity_value.name = "Velocity"
    pressure_value = mixed_value.sub(1)
    pressure_value.name = "Pressure"
    return VelocityPressureUnknown(
        name=name,
        space=W,
        value=mixed_value,
        trial=trial,
        test=test,
        velocity=UnknownField(
            name="Velocity",
            kind="velocity_subfield",
            space=W.sub(0),
            value=velocity_value,
            trial=trial[0],
            test=test[0],
        ),
        pressure=UnknownField(
            name="Pressure",
            kind="pressure_subfield",
            space=W.sub(1),
            value=pressure_value,
            trial=trial[1],
            test=test[1],
        ),
        velocity_degree=int(velocity_degree),
        pressure_degree=int(pressure_degree),
    )


def temperature(domain, *, degree: int = 1, value=0.0) -> UnknownField:
    """Create a temperature unknown for heat-transfer workflows."""

    field = scalar_unknown(domain, name="Temperature", degree=degree, value=value)
    return UnknownField(
        name=field.name,
        kind="temperature",
        space=field.space,
        value=field.value,
        trial=field.trial,
        test=field.test,
    )


def wrap(function, *, name: str | None = None) -> Field:
    """Wrap a DOLFINx function as an AgentFEM field."""

    if isinstance(function, Field):
        if name is not None:
            function.function.name = name
        return function
    return Field(function=function, name=name)


def unwrap(field_or_function):
    """Return the underlying DOLFINx function when given an AgentFEM field."""

    if isinstance(field_or_function, Field):
        return field_or_function.function
    if hasattr(field_or_function, "value") and hasattr(field_or_function.value, "x"):
        return field_or_function.value
    return field_or_function


def empty_like(field_or_function, *, name: str | None = None) -> Field:
    """Create a zero-valued field with the same function space."""

    function = unwrap(field_or_function)
    return Field(fem.Function(function.function_space, name=name or "Field"))


def compute(expression, *, name: str | None = None) -> Field:
    """Return a computed field.

    For eager AgentFEM field algebra, ``expression`` is already a field. This
    helper gives user code a readable Cast3M-like spelling when a named result
    is desired.
    """

    if isinstance(expression, Field):
        if name is None:
            return expression
        return expression.copy(name=name)
    if hasattr(expression, "function_space") and hasattr(expression, "x"):
        return wrap(expression, name=name)
    raise TypeError("fields.compute currently requires a Field or DOLFINx Function.")


def assign(target, source) -> None:
    """Assign a scalar, compatible field, or DOLFINx function into ``target``."""

    target_function = unwrap(target)
    if np.isscalar(source):
        target_function.x.array[:] = source
        target_function.x.scatter_forward()
        return
    source_function = unwrap(source)
    require_same_space(target_function, source_function)
    dofs.copy_function(target_function, source_function)


def dot(left, right) -> float:
    """Return the distributed algebraic dot product of two compatible fields.

    This is an immediate numerical dof-vector operation, not a weak-form
    integral. Use operator-level helpers for mass-weighted or stiffness-weighted
    products such as ``x^T M y``.
    """

    require_same_space(left, right)
    left_function = unwrap(left)
    right_function = unwrap(right)
    local = float(np.dot(dofs.owned_array(left_function), dofs.owned_array(right_function)))
    return left_function.function_space.mesh.comm.allreduce(local, op=MPI.SUM)


def weighted_dot(left, weights, right=None) -> float:
    """Return ``left^T diag(weights) right`` for compatible fields.

    ``weights`` is normally a lumped mass/capacity/stiffness-like diagonal
    vector assembled on the same function space. If ``right`` is omitted, this
    returns ``left^T diag(weights) left``.
    """

    if right is None:
        right = left
    require_same_space(left, right)
    left_function = unwrap(left)
    right_function = unwrap(right)
    left_values = dofs.owned_array(left_function)
    right_values = dofs.owned_array(right_function)
    owned_weights = np.asarray(weights)[: len(left_values)]
    if len(owned_weights) != len(left_values):
        raise ValueError(
            "weighted_dot requires weights with at least the local owned dof length."
        )
    local = float(np.sum(owned_weights * left_values * right_values))
    return left_function.function_space.mesh.comm.allreduce(local, op=MPI.SUM)


def norm(field, *, weight=None) -> float:
    """Return the distributed algebraic norm of a field.

    Without ``weight`` this is ``sqrt(field^T field)``. With a lumped diagonal
    weight this is ``sqrt(field^T diag(weight) field)``.
    """

    value = weighted_dot(field, weight) if weight is not None else dot(field, field)
    return float(np.sqrt(value))


def require_same_space(left, right) -> None:
    """Raise if two fields/functions are not on the same function space."""

    left_function = unwrap(left)
    right_function = unwrap(right)
    if left_function.function_space is not right_function.function_space:
        raise ValueError(
            "Field algebra requires the same function space in this release. "
            f"Got {left_function.function_space!r} and {right_function.function_space!r}."
        )


def same_space(left, right) -> bool:
    """Return whether two fields/functions share the same function space."""

    return unwrap(left).function_space is unwrap(right).function_space


def _binary_field_op(left, right, op, op_name: str) -> Field:
    left_is_field = _is_field_like(left)
    right_is_field = _is_field_like(right)
    if not left_is_field and not right_is_field:
        raise TypeError("At least one operand must be a field.")

    template = left if left_is_field else right
    result = empty_like(template, name=f"{op_name}_{unwrap(template).name}")

    if left_is_field:
        left_values = unwrap(left).x.array
    else:
        left_values = left
    if right_is_field:
        right_values = unwrap(right).x.array
    else:
        right_values = right

    if left_is_field and right_is_field:
        require_same_space(left, right)

    result.x.array[:] = op(left_values, right_values)
    result.x.scatter_forward()
    return result


def _is_field_like(value) -> bool:
    return isinstance(value, Field) or (
        hasattr(value, "function_space") and hasattr(value, "x")
    )

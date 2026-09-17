# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Inspectable scalar controls assembled from physical point displacements."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from dolfinx import fem

from .. import fields as field_api


@dataclass(frozen=True)
class PointKinematicTerm:
    """One coefficient multiplying one displacement component at a point."""

    point: tuple[float, ...]
    component: int
    coefficient: float
    name: str = "point_displacement"

    def __post_init__(self) -> None:
        point = tuple(float(value) for value in self.point)
        component = int(self.component)
        coefficient = float(self.coefficient)
        name = str(self.name).strip()
        if not point or not all(isfinite(value) for value in point):
            raise ValueError("A kinematic-control point must be finite and nonempty.")
        if component < 0:
            raise ValueError("A kinematic-control component must be nonnegative.")
        if not isfinite(coefficient) or coefficient == 0.0:
            raise ValueError("A kinematic-control coefficient must be finite and nonzero.")
        if not name:
            raise ValueError("A kinematic-control term needs a name.")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "name", name)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "point": self.point,
            "component": self.component,
            "coefficient": self.coefficient,
        }


@dataclass(frozen=True)
class LinearKinematicControl:
    """One scalar generalized coordinate ``q = sum(a_i u_i)``.

    The first backend is deliberately serial and point based.  It is intended
    for rigid loading fixtures, crack-opening controls, and other small sets of
    exact nodal equations.  The control owns the coordinate definition; the
    active solver owns its conjugate force and work history.
    """

    target: object
    terms: tuple[PointKinematicTerm, ...]
    name: str = "linear_kinematic_control"
    unit: str | None = None
    tolerance: float = 1.0e-10
    _coefficient_vector: np.ndarray = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        function = field_api.unwrap(self.target)
        space = function.function_space
        if int(space.mesh.comm.size) != 1:
            raise NotImplementedError(
                "The initial linear kinematic-control backend is serial."
            )
        shape = tuple(function.ufl_shape)
        if len(shape) != 1:
            raise TypeError("Linear kinematic control requires a vector field.")
        terms = tuple(self.terms)
        if not terms or not all(isinstance(item, PointKinematicTerm) for item in terms):
            raise TypeError("Linear kinematic control requires point terms.")
        dimension = int(shape[0])
        geometric_dimension = int(space.mesh.geometry.dim)
        if any(item.component >= dimension for item in terms):
            raise ValueError("A kinematic-control component exceeds the field dimension.")
        if any(len(item.point) != geometric_dimension for item in terms):
            raise ValueError("Kinematic-control points must match the mesh dimension.")
        name = str(self.name).strip()
        tolerance = float(self.tolerance)
        if not name or not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Linear kinematic control needs a name and positive tolerance.")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tolerance", tolerance)
        # Resolve once so missing, duplicated, or ambiguous points fail before
        # a nonlinear solve starts without repeating geometric searches while
        # progress and result evidence query the coordinate.
        object.__setattr__(self, "_coefficient_vector", self._resolve_coefficients())

    def coefficients(self) -> np.ndarray:
        """Return the serial scalar-dof coefficient vector."""

        return self._coefficient_vector.copy()

    def _resolve_coefficients(self) -> np.ndarray:
        """Resolve physical point terms once at construction time."""

        function = field_api.unwrap(self.target)
        space = function.function_space
        values = np.zeros(function.x.array.size, dtype=float)
        for term in self.terms:
            collapsed, _ = space.sub(term.component).collapse()
            located = fem.locate_dofs_geometrical(
                (space.sub(term.component), collapsed),
                self._point_marker(term.point),
            )
            parent = np.asarray(located[0], dtype=np.int64)
            if parent.size != 1:
                raise ValueError(
                    f"Kinematic term {term.name!r} at {term.point!r} selected "
                    f"{parent.size} dofs; exactly one is required."
                )
            values[int(parent[0])] += term.coefficient
        active = np.flatnonzero(np.abs(values) > np.finfo(float).eps)
        if active.size == 0:
            raise ValueError("Linear kinematic-control terms cancel completely.")
        return values

    def coordinate(self, displacement=None) -> float:
        selected = field_api.unwrap(self.target if displacement is None else displacement)
        if selected.function_space is not field_api.unwrap(self.target).function_space:
            raise ValueError("Kinematic control and displacement must share one space.")
        return float(np.dot(self.coefficients(), selected.x.array))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "linear_kinematic_control",
            "equation": "q = sum(a_i * u_i)",
            "terms": tuple(item.summary() for item in self.terms),
            "unit": self.unit,
            "enforcement": "exact_scalar_lagrange_multiplier",
            "supports_parallel": False,
            "dual_owner": "active_equilibrium_provider",
        }

    def _point_marker(self, point):
        selected = np.asarray(point, dtype=float)
        tolerance = self.tolerance

        def marker(x):
            return np.all(
                np.isclose(
                    x[: selected.size],
                    selected[:, None],
                    rtol=0.0,
                    atol=tolerance,
                ),
                axis=0,
            )

        return marker


def point_kinematic_term(
    point,
    *,
    component: int,
    coefficient: float,
    name: str = "point_displacement",
) -> PointKinematicTerm:
    """Create one readable term of a generalized displacement coordinate."""

    return PointKinematicTerm(
        point=tuple(point),
        component=component,
        coefficient=coefficient,
        name=name,
    )


def linear_kinematic_control(
    target,
    terms,
    *,
    name: str = "linear_kinematic_control",
    unit: str | None = None,
    tolerance: float = 1.0e-10,
) -> LinearKinematicControl:
    """Create a scalar control with a provider-owned conjugate reaction."""

    return LinearKinematicControl(
        target=target,
        terms=tuple(terms),
        name=name,
        unit=unit,
        tolerance=tolerance,
    )


__all__ = (
    "LinearKinematicControl",
    "PointKinematicTerm",
    "linear_kinematic_control",
    "point_kinematic_term",
)

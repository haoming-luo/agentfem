# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Objective local kinematics for future finite-rotation surface providers.

This module owns geometry, not a shell finite element.  It deliberately keeps
director kinematics independent from interpolation, quadrature, locking
control, contact, and nonlinear solution.  A future fibrous-shell provider can
therefore consume the same tested measures without making a material law own
element technology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _columns(value, *, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != (3, 2) or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be one finite 3x2 tangent matrix.")
    if np.linalg.matrix_rank(selected) != 2:
        raise ValueError(f"{name} must contain two independent tangents.")
    return selected


def _director(value, *, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != (3,) or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be one finite 3-component vector.")
    norm = float(np.linalg.norm(selected))
    if norm <= np.finfo(float).eps:
        raise ValueError(f"{name} must have nonzero length.")
    return selected / norm


def _director_gradient(value, *, name: str) -> np.ndarray:
    if value is None:
        return np.zeros((3, 2))
    selected = np.asarray(value, dtype=float)
    if selected.shape != (3, 2) or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be one finite 3x2 matrix.")
    return selected


@dataclass(frozen=True)
class DirectorShellKinematics:
    """Finite-rotation surface measures at one material point.

    ``membrane_strain`` and ``curvature_change`` are symmetric covariant
    2x2 tensors. ``transverse_shear`` contains the two director/tangent dot
    products. All measures vanish under a superposed rigid rotation.
    """

    reference_metric: np.ndarray
    current_metric: np.ndarray
    membrane_strain: np.ndarray
    transverse_shear: np.ndarray
    curvature_change: np.ndarray
    current_normal: np.ndarray
    area_ratio: float

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "director_shell_kinematics",
            "reference_metric": self.reference_metric.tolist(),
            "current_metric": self.current_metric.tolist(),
            "membrane_strain": self.membrane_strain.tolist(),
            "transverse_shear": self.transverse_shear.tolist(),
            "curvature_change": self.curvature_change.tolist(),
            "current_normal": self.current_normal.tolist(),
            "area_ratio": self.area_ratio,
            "objectivity": "invariant_under_superposed_rigid_rotation",
        }


@dataclass(frozen=True)
class FiberCurveKinematics:
    """Objective bending measures for one material fibre curve.

    The signed in-plane component is measured along ``normal x direction``;
    the signed normal component is measured along the surface normal.  Their
    changes are the quantities required by fibrous-surface bending laws.  This
    local contract deliberately does not prescribe a shell interpolation or a
    second-gradient discretization.
    """

    reference_direction: np.ndarray
    current_direction: np.ndarray
    reference_in_plane_curvature: float
    current_in_plane_curvature: float
    in_plane_curvature_change: float
    reference_normal_curvature: float
    current_normal_curvature: float
    normal_curvature_change: float

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "fiber_curve_kinematics",
            "reference_direction": self.reference_direction.tolist(),
            "current_direction": self.current_direction.tolist(),
            "reference_in_plane_curvature": self.reference_in_plane_curvature,
            "current_in_plane_curvature": self.current_in_plane_curvature,
            "in_plane_curvature_change": self.in_plane_curvature_change,
            "reference_normal_curvature": self.reference_normal_curvature,
            "current_normal_curvature": self.current_normal_curvature,
            "normal_curvature_change": self.normal_curvature_change,
            "sign_convention": {
                "in_plane": "positive_along_surface_normal_cross_fiber",
                "normal": "positive_along_surface_normal",
            },
            "objectivity": "invariant_under_superposed_rigid_rotation",
        }


def _fiber_curve_components(tangents, direction, gradient, *, name: str):
    surface = _columns(tangents, name=f"{name}_tangents")
    fiber = _director(direction, name=f"{name}_direction")
    derivative = _director_gradient(gradient, name=f"{name}_direction_gradient")
    normal = np.cross(surface[:, 0], surface[:, 1])
    normal /= np.linalg.norm(normal)
    if abs(float(np.dot(fiber, normal))) > 1.0e-9:
        raise ValueError(f"{name}_direction must be tangent to the surface.")
    metric = surface.T @ surface
    coordinates = np.linalg.solve(metric, surface.T @ fiber)
    reconstructed = surface @ coordinates
    if not np.allclose(reconstructed, fiber, atol=1.0e-9, rtol=1.0e-9):
        raise ValueError(f"{name}_direction must lie in the surface tangent span.")
    directional_derivative = derivative @ coordinates
    # A unit-vector derivative is orthogonal to the vector. Remove only roundoff
    # drift so curvature components remain insensitive to numerical normalization.
    directional_derivative -= (
        float(np.dot(directional_derivative, fiber)) * fiber
    )
    in_plane_normal = np.cross(normal, fiber)
    return (
        fiber,
        float(np.dot(directional_derivative, in_plane_normal)),
        float(np.dot(directional_derivative, normal)),
    )


def fiber_curve_kinematics(
    reference_tangents,
    current_tangents,
    reference_direction,
    current_direction,
    *,
    current_direction_gradient,
    reference_direction_gradient=None,
) -> FiberCurveKinematics:
    """Evaluate in-plane and normal curvature changes of one fibre family.

    Direction gradients are derivatives of the *unit* fibre direction with
    respect to the two surface coordinates.  This makes the required
    second-gradient information explicit instead of hiding it in a fitted
    conventional-shell stiffness.
    """

    reference = _fiber_curve_components(
        reference_tangents,
        reference_direction,
        reference_direction_gradient,
        name="reference",
    )
    current = _fiber_curve_components(
        current_tangents,
        current_direction,
        current_direction_gradient,
        name="current",
    )
    return FiberCurveKinematics(
        reference_direction=reference[0],
        current_direction=current[0],
        reference_in_plane_curvature=reference[1],
        current_in_plane_curvature=current[1],
        in_plane_curvature_change=current[1] - reference[1],
        reference_normal_curvature=reference[2],
        current_normal_curvature=current[2],
        normal_curvature_change=current[2] - reference[2],
    )


def director_shell_kinematics(
    reference_tangents,
    current_tangents,
    director,
    *,
    director_gradient=None,
    reference_director=None,
    reference_director_gradient=None,
) -> DirectorShellKinematics:
    """Evaluate objective membrane, shear, and curvature measures.

    The tangents and unit-director data are expressed in a common
    three-dimensional Cartesian frame. Reference director data are optional
    for initially flat surfaces; curved or initially sheared references should
    supply both reference quantities. Supplied gradients are derivatives of
    the normalized directors with respect to the two surface coordinates.
    """

    A = _columns(reference_tangents, name="reference_tangents")
    a = _columns(current_tangents, name="current_tangents")
    d = _director(director, name="director")
    Dd = _director_gradient(director_gradient, name="director_gradient")

    cross_reference = np.cross(A[:, 0], A[:, 1])
    reference_normal = cross_reference / np.linalg.norm(cross_reference)
    D = (
        reference_normal
        if reference_director is None
        else _director(reference_director, name="reference_director")
    )
    DD = _director_gradient(
        reference_director_gradient,
        name="reference_director_gradient",
    )

    reference_metric = A.T @ A
    current_metric = a.T @ a
    membrane = 0.5 * (current_metric - reference_metric)
    transverse_shear = a.T @ d - A.T @ D
    reference_curvature = -0.5 * (A.T @ DD + DD.T @ A)
    current_curvature = -0.5 * (a.T @ Dd + Dd.T @ a)
    curvature_change = current_curvature - reference_curvature

    cross_current = np.cross(a[:, 0], a[:, 1])
    current_area = float(np.linalg.norm(cross_current))
    reference_area = float(np.linalg.norm(cross_reference))
    current_normal = cross_current / current_area
    return DirectorShellKinematics(
        reference_metric=reference_metric,
        current_metric=current_metric,
        membrane_strain=membrane,
        transverse_shear=transverse_shear,
        curvature_change=curvature_change,
        current_normal=current_normal,
        area_ratio=current_area / reference_area,
    )


__all__ = [
    "DirectorShellKinematics",
    "FiberCurveKinematics",
    "director_shell_kinematics",
    "fiber_curve_kinematics",
]

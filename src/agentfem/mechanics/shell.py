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


__all__ = ["DirectorShellKinematics", "director_shell_kinematics"]

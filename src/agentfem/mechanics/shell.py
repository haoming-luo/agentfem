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


@dataclass(frozen=True)
class RotationFreeEdgeBoundarySemantics:
    """Declare the two work-conjugate boundary pairs of a thin shell.

    This object is a provider-neutral scientific contract, not an executable
    boundary condition.  A rotation-free shell provider must lower the
    translational pair ``displacement/effective_force`` and the bending pair
    ``normal_rotation/bending_moment`` consistently with its own discrete
    surface gradient.  Keeping this distinction explicit prevents an ordinary
    displacement constraint from being mislabeled as a shell clamp.
    """

    translation_control: str
    bending_control: str
    name: str = "rotation_free_edge"

    def __post_init__(self) -> None:
        choices = {"essential", "natural"}
        translation = str(self.translation_control).strip().lower()
        bending = str(self.bending_control).strip().lower()
        if translation not in choices:
            raise ValueError(
                "translation_control must be 'essential' or 'natural'."
            )
        if bending not in choices:
            raise ValueError("bending_control must be 'essential' or 'natural'.")
        name = str(self.name).strip()
        if not name:
            raise ValueError("Rotation-free edge boundary name must not be empty.")
        object.__setattr__(self, "translation_control", translation)
        object.__setattr__(self, "bending_control", bending)
        object.__setattr__(self, "name", name)

    @property
    def shell_support(self) -> str:
        if self.translation_control == "essential" and self.bending_control == "essential":
            return "clamped"
        if self.translation_control == "essential":
            return "simply_supported"
        if self.bending_control == "essential":
            return "rotation_guided"
        return "free"

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "rotation_free_edge_boundary_semantics",
            "name": self.name,
            "shell_support": self.shell_support,
            "translation_control": self.translation_control,
            "bending_control": self.bending_control,
            "work_conjugate_pairs": (
                {
                    "kinematic": "boundary_displacement",
                    "dynamic": "effective_boundary_force",
                    "control": self.translation_control,
                },
                {
                    "kinematic": "boundary_normal_rotation",
                    "dynamic": "bending_moment",
                    "control": self.bending_control,
                },
            ),
            "effective_force_note": (
                "provider must include higher-order boundary terms; it is not "
                "assumed equal to a raw section-force resultant"
            ),
            "lowering": "unavailable_until_provider_verified",
        }


def rotation_free_edge_boundary(
    *,
    translation: str,
    bending: str,
    name: str = "rotation_free_edge",
) -> RotationFreeEdgeBoundarySemantics:
    """Create an inspectable rotation-free shell edge contract.

    ``essential`` selects the kinematic member of a conjugate pair;
    ``natural`` selects its force or moment member.  The contract deliberately
    has no backend object until a shell provider supplies verified lowering.
    """

    return RotationFreeEdgeBoundarySemantics(
        translation_control=translation,
        bending_control=bending,
        name=name,
    )


@dataclass(frozen=True)
class ReconstructedFiberCurvature:
    """Owned-cell fibre curvatures plus neighbour reconstruction evidence."""

    direction_gradients: np.ndarray
    in_plane_curvature: np.ndarray
    normal_curvature: np.ndarray
    reconstruction: object

    def __post_init__(self) -> None:
        gradients = np.asarray(self.direction_gradients, dtype=float)
        in_plane = np.asarray(self.in_plane_curvature, dtype=float)
        normal = np.asarray(self.normal_curvature, dtype=float)
        if gradients.ndim != 3 or gradients.shape[1:] != (3, 2):
            raise ValueError("direction_gradients must have shape (cells, 3, 2).")
        if in_plane.shape != (gradients.shape[0],) or normal.shape != in_plane.shape:
            raise ValueError("Curvature arrays require one scalar per owned cell.")
        if not all(np.all(np.isfinite(value)) for value in (gradients, in_plane, normal)):
            raise ValueError("Reconstructed fibre curvature must be finite.")
        object.__setattr__(self, "direction_gradients", gradients.copy())
        object.__setattr__(self, "in_plane_curvature", in_plane.copy())
        object.__setattr__(self, "normal_curvature", normal.copy())

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "reconstructed_fiber_curvature",
            "owned_cells": int(self.in_plane_curvature.size),
            "in_plane_curvature": self.in_plane_curvature.tolist(),
            "normal_curvature": self.normal_curvature.tolist(),
            "sign_convention": {
                "in_plane": "positive_along_surface_normal_cross_fiber",
                "normal": "positive_along_surface_normal",
            },
            "reconstruction": self.reconstruction.as_dict(),
            "maturity": "discrete_kinematic_foundation_not_shell_equilibrium",
        }


@dataclass(frozen=True)
class FibrousShellKinematicsExpressions:
    """Symbolic operator-owned measures consumed by a fibrous-shell law.

    Membrane measures come from the surface deformation, director shear from
    an independent material normal, and the two bending pairs from independent
    unit-fibre fields.  Compatibility between those fields remains an operator
    constraint rather than a constitutive assumption.
    """

    generalized_strain: object
    convected_warp: object
    convected_weft: object
    current_normal: object
    area_ratio: object

    @property
    def generalized_order(self) -> tuple[str, ...]:
        return (
            "warp_strain",
            "weft_strain",
            "trellising_angle",
            "director_shear_1",
            "director_shear_2",
            "warp_in_plane_curvature",
            "weft_in_plane_curvature",
            "warp_normal_curvature",
            "weft_normal_curvature",
        )


@dataclass(frozen=True)
class FibrousShellCompatibilityExpressions:
    """Minimal mixed-field compatibility residuals for a no-slip layer.

    The residual contains one director-unit equation followed by the three
    Cartesian convection equations for warp and weft.  Fibre unit length and
    surface tangency are consequences of the vector convection equations and
    are therefore diagnostics, not redundant multiplier constraints.
    """

    residual: object
    director_unit: object
    warp_convection: object
    weft_convection: object
    warp_unit_error: object
    weft_unit_error: object
    warp_tangency: object
    weft_tangency: object

    @property
    def residual_order(self) -> tuple[str, ...]:
        return (
            "director_unit",
            "warp_convection_x",
            "warp_convection_y",
            "warp_convection_z",
            "weft_convection_x",
            "weft_convection_y",
            "weft_convection_z",
        )


def _ufl_matrix(value, *, name: str, shape: tuple[int, int]):
    import ufl

    selected_shape = tuple(getattr(value, "ufl_shape", ()))
    if selected_shape:
        if selected_shape != shape:
            raise ValueError(f"{name} must have UFL shape {shape}.")
        return value
    selected = np.asarray(value, dtype=float)
    if selected.shape != shape or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be one finite {shape[0]}x{shape[1]} matrix.")
    return ufl.as_matrix(selected.tolist())


def _ufl_vector(value, *, name: str, size: int):
    import ufl

    selected_shape = tuple(getattr(value, "ufl_shape", ()))
    if selected_shape:
        if selected_shape != (size,):
            raise ValueError(f"{name} must have UFL shape ({size},).")
        return value
    selected = np.asarray(value, dtype=float)
    if selected.shape != (size,) or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be one finite {size}-component vector.")
    return ufl.as_vector(selected.tolist())


def _ufl_unit(vector):
    import ufl

    return vector / ufl.sqrt(ufl.inner(vector, vector))


def _ufl_surface_normal(tangents):
    import ufl

    return _ufl_unit(ufl.cross(tangents[:, 0], tangents[:, 1]))


def _ufl_surface_deformation(reference_tangents, current_tangents):
    import ufl

    reference_normal = _ufl_surface_normal(reference_tangents)
    current_normal = _ufl_surface_normal(current_tangents)
    reference_dual = ufl.dot(
        reference_tangents,
        ufl.inv(ufl.dot(ufl.transpose(reference_tangents), reference_tangents)),
    )
    deformation = ufl.dot(
        current_tangents,
        ufl.transpose(reference_dual),
    ) + ufl.outer(current_normal, reference_normal)
    return deformation, reference_normal, current_normal


def _ufl_fibre_curvature(tangents, direction, gradient):
    import ufl

    normal = _ufl_surface_normal(tangents)
    unit_direction = _ufl_unit(direction)
    metric = ufl.dot(ufl.transpose(tangents), tangents)
    coordinates = ufl.dot(
        ufl.inv(metric),
        ufl.dot(ufl.transpose(tangents), unit_direction),
    )
    derivative = ufl.dot(gradient, coordinates)
    derivative -= ufl.inner(derivative, unit_direction) * unit_direction
    return (
        ufl.inner(derivative, ufl.cross(normal, unit_direction)),
        ufl.inner(derivative, normal),
    )


def fibrous_shell_kinematics_ufl(
    reference_tangents,
    current_tangents,
    director,
    *,
    reference_fibers,
    current_fibers,
    current_fiber_gradients,
    reference_director=None,
    reference_fiber_gradients=None,
) -> FibrousShellKinematicsExpressions:
    """Build the nine objective fibrous-shell measures as UFL expressions.

    This function defines the kinematic hand-off between a future mixed shell
    operator and :meth:`DecoupledFibrousShell.generalized_expressions_ufl`.
    It does not choose finite-element spaces or enforce the unit, tangency, and
    convection constraints required by independent director/fibre fields.
    """

    import ufl

    A = _ufl_matrix(reference_tangents, name="reference_tangents", shape=(3, 2))
    a = _ufl_matrix(current_tangents, name="current_tangents", shape=(3, 2))
    d = _ufl_unit(_ufl_vector(director, name="director", size=3))
    D = (
        _ufl_surface_normal(A)
        if reference_director is None
        else _ufl_unit(_ufl_vector(reference_director, name="reference_director", size=3))
    )
    if len(reference_fibers) != 2 or len(current_fibers) != 2:
        raise ValueError("reference_fibers and current_fibers must contain warp and weft.")
    if len(current_fiber_gradients) != 2:
        raise ValueError("current_fiber_gradients must contain warp and weft gradients.")
    reference_gradients = (
        (np.zeros((3, 2)), np.zeros((3, 2)))
        if reference_fiber_gradients is None
        else reference_fiber_gradients
    )
    if len(reference_gradients) != 2:
        raise ValueError("reference_fiber_gradients must contain warp and weft gradients.")
    reference = tuple(
        _ufl_unit(_ufl_vector(item, name=f"reference_fibers[{index}]", size=3))
        for index, item in enumerate(reference_fibers)
    )
    current = tuple(
        _ufl_unit(_ufl_vector(item, name=f"current_fibers[{index}]", size=3))
        for index, item in enumerate(current_fibers)
    )
    current_gradients = tuple(
        _ufl_matrix(item, name=f"current_fiber_gradients[{index}]", shape=(3, 2))
        for index, item in enumerate(current_fiber_gradients)
    )
    reference_gradients = tuple(
        _ufl_matrix(item, name=f"reference_fiber_gradients[{index}]", shape=(3, 2))
        for index, item in enumerate(reference_gradients)
    )

    surface_deformation, _, current_normal = _ufl_surface_deformation(A, a)
    convected = tuple(surface_deformation * item for item in reference)
    stretches = tuple(ufl.sqrt(ufl.inner(item, item)) for item in convected)
    convected_units = tuple(item / stretch for item, stretch in zip(convected, stretches))
    reference_angle = ufl.acos(ufl.inner(reference[0], reference[1]))
    current_cosine = ufl.inner(convected_units[0], convected_units[1])
    current_cosine = ufl.max_value(-1.0, ufl.min_value(1.0, current_cosine))
    trellising = reference_angle - ufl.acos(current_cosine)
    transverse_shear = ufl.dot(ufl.transpose(a), d) - ufl.dot(ufl.transpose(A), D)

    reference_curvatures = tuple(
        _ufl_fibre_curvature(A, direction, gradient)
        for direction, gradient in zip(reference, reference_gradients)
    )
    current_curvatures = tuple(
        _ufl_fibre_curvature(a, direction, gradient)
        for direction, gradient in zip(current, current_gradients)
    )
    in_plane = tuple(
        current_curvatures[index][0] - reference_curvatures[index][0]
        for index in range(2)
    )
    normal = tuple(
        current_curvatures[index][1] - reference_curvatures[index][1]
        for index in range(2)
    )
    generalized = ufl.as_vector(
        (
            stretches[0] - 1.0,
            stretches[1] - 1.0,
            trellising,
            transverse_shear[0],
            transverse_shear[1],
            in_plane[0],
            in_plane[1],
            normal[0],
            normal[1],
        )
    )
    reference_area = ufl.sqrt(
        ufl.inner(ufl.cross(A[:, 0], A[:, 1]), ufl.cross(A[:, 0], A[:, 1]))
    )
    current_area = ufl.sqrt(
        ufl.inner(ufl.cross(a[:, 0], a[:, 1]), ufl.cross(a[:, 0], a[:, 1]))
    )
    return FibrousShellKinematicsExpressions(
        generalized_strain=generalized,
        convected_warp=convected_units[0],
        convected_weft=convected_units[1],
        current_normal=current_normal,
        area_ratio=current_area / reference_area,
    )


def fibrous_shell_compatibility_ufl(
    reference_tangents,
    current_tangents,
    director,
    *,
    reference_fibers,
    current_fibers,
) -> FibrousShellCompatibilityExpressions:
    """Return the minimal exact-constraint residual for mixed shell fields.

    A future operator may pair this residual with Lagrange multipliers,
    augmented multipliers, or a justified local condensation.  This function
    intentionally chooses no enforcement strategy and introduces no penalty
    scale.  Slip-enabled layers require a different explicit contract.
    """

    import ufl

    A = _ufl_matrix(reference_tangents, name="reference_tangents", shape=(3, 2))
    a = _ufl_matrix(current_tangents, name="current_tangents", shape=(3, 2))
    d = _ufl_vector(director, name="director", size=3)
    if len(reference_fibers) != 2 or len(current_fibers) != 2:
        raise ValueError("reference_fibers and current_fibers must contain warp and weft.")
    reference = tuple(
        _ufl_unit(_ufl_vector(item, name=f"reference_fibers[{index}]", size=3))
        for index, item in enumerate(reference_fibers)
    )
    current = tuple(
        _ufl_vector(item, name=f"current_fibers[{index}]", size=3)
        for index, item in enumerate(current_fibers)
    )
    surface_deformation, _, current_normal = _ufl_surface_deformation(A, a)
    convected = tuple(_ufl_unit(surface_deformation * item) for item in reference)
    warp_convection = current[0] - convected[0]
    weft_convection = current[1] - convected[1]
    director_unit = ufl.inner(d, d) - 1.0
    residual = ufl.as_vector(
        (
            director_unit,
            *warp_convection,
            *weft_convection,
        )
    )
    return FibrousShellCompatibilityExpressions(
        residual=residual,
        director_unit=director_unit,
        warp_convection=warp_convection,
        weft_convection=weft_convection,
        warp_unit_error=ufl.inner(current[0], current[0]) - 1.0,
        weft_unit_error=ufl.inner(current[1], current[1]) - 1.0,
        warp_tangency=ufl.inner(current_normal, current[0]),
        weft_tangency=ufl.inner(current_normal, current[1]),
    )


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


def reconstruct_fiber_curvature(
    domain,
    current_directions,
    current_tangents,
    *,
    rings: int = 2,
    weight_power: float = 1.0,
    condition_limit: float = 1.0e10,
) -> ReconstructedFiberCurvature:
    """Reconstruct in-plane and normal fibre curvature on owned cells.

    This first neighbour-element adapter uses a two-dimensional parameter mesh
    with three-dimensional unit-fibre directions and 3x2 current tangents.
    It consumes the generic rank-audited cell-gradient reconstruction and does
    not define bending energy, virtual work, boundary moments, or a shell Step.
    """

    from agentfem.mesh import reconstruct_cell_gradient

    domain = getattr(domain, "domain", domain)
    if int(domain.topology.dim) != 2 or int(domain.geometry.dim) != 2:
        raise ValueError(
            "The first reconstructed fibre-curvature adapter requires a "
            "two-dimensional parameter mesh; embedded-surface coordinate "
            "reconstruction is a separate promotion gate."
        )
    cell_map = domain.topology.index_map(domain.topology.dim)
    total = int(cell_map.size_local + cell_map.num_ghosts)
    directions = np.asarray(current_directions, dtype=float)
    tangents = np.asarray(current_tangents, dtype=float)
    if directions.shape != (total, 3) or not np.all(np.isfinite(directions)):
        raise ValueError(
            "current_directions must have shape (local_and_ghost_cells, 3)."
        )
    if tangents.shape != (total, 3, 2) or not np.all(np.isfinite(tangents)):
        raise ValueError(
            "current_tangents must have shape (local_and_ghost_cells, 3, 2)."
        )
    reconstruction = reconstruct_cell_gradient(
        domain,
        directions,
        rings=rings,
        weight_power=weight_power,
        condition_limit=condition_limit,
    )
    owned = int(cell_map.size_local)
    in_plane = np.empty(owned, dtype=float)
    normal = np.empty(owned, dtype=float)
    for cell in range(owned):
        _, in_plane[cell], normal[cell] = _fiber_curve_components(
            tangents[cell],
            directions[cell],
            reconstruction.gradients[cell],
            name=f"cell_{cell}_current",
        )
    return ReconstructedFiberCurvature(
        direction_gradients=reconstruction.gradients,
        in_plane_curvature=in_plane,
        normal_curvature=normal,
        reconstruction=reconstruction,
    )


def surface_deformation_gradient(
    reference_tangents,
    current_tangents,
) -> np.ndarray:
    """Return the three-dimensional tangential deformation lift.

    The map sends each reference tangent to its current counterpart and the
    reference unit normal to the current unit normal. Its action on an
    embedded reference fibre is therefore unique, while the unit normal lift
    merely completes the rank-two surface map for convenient constitutive use.
    No physical thickness stretch is implied.
    """

    reference = _columns(reference_tangents, name="reference_tangents")
    current = _columns(current_tangents, name="current_tangents")
    reference_normal = np.cross(reference[:, 0], reference[:, 1])
    reference_normal /= np.linalg.norm(reference_normal)
    current_normal = np.cross(current[:, 0], current[:, 1])
    current_normal /= np.linalg.norm(current_normal)
    reference_dual = reference @ np.linalg.inv(reference.T @ reference)
    return current @ reference_dual.T + np.outer(current_normal, reference_normal)


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
    "ReconstructedFiberCurvature",
    "FibrousShellCompatibilityExpressions",
    "FibrousShellKinematicsExpressions",
    "RotationFreeEdgeBoundarySemantics",
    "director_shell_kinematics",
    "fiber_curve_kinematics",
    "reconstruct_fiber_curvature",
    "rotation_free_edge_boundary",
    "fibrous_shell_compatibility_ufl",
    "fibrous_shell_kinematics_ufl",
    "surface_deformation_gradient",
]

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded planar reduction of full three-dimensional history updates.

The material remains three-dimensional. This module never commits state or
relabels a condensed tangent as the material's full dP/dF. It solves transverse
equilibrium from fixed accepted history, then returns a separate 4-by-4 tangent.
"""

from dataclasses import dataclass, replace

import numpy as np

from .user_material import (
    MaterialPointInput,
    MaterialPointOutput,
    MaterialTangentConvention,
    validated_material_update,
    _tangent_error_metrics,
    _MaterialInputMutationError,
    _VerificationMaterial,
)

_IN_PLANE = np.array((0, 1, 3, 4))  # row-major 11,12,21,22
_SHEAR = np.array((2, 5, 6, 7))
_PLANAR = np.array((0, 1, 3, 4, 8))


def _norm(values):
    """Avoid squaring dimensional stresses before normalizing their scale."""
    magnitude = float(np.max(np.abs(values), initial=0.0))
    if magnitude == 0:
        return 0.0
    with np.errstate(over="raise", invalid="raise"):
        return float(magnitude * np.linalg.norm(values / magnitude))


@dataclass(frozen=True)
class PlaneStressReduction:
    """Explicit reflection-symmetric planar assumption and local solve policy.

    ``symmetry="reflection_symmetric"`` is an author declaration, not inferred
    from an isotropic-looking response. Only positive transverse stiffness is
    admitted. Full director/shear reduction and shells are outside this scope.
    Absolute tolerance has stress units; the default uses a relative scale.
    """

    symmetry: str
    relative_tolerance: float = 1.0e-11
    absolute_tolerance: float = 0.0
    maximum_iterations: int = 30
    maximum_backtracks: int = 20

    def __post_init__(self):
        if self.symmetry != "reflection_symmetric":
            raise ValueError(
                "Declare symmetry='reflection_symmetric' for scalar plane stress."
            )
        if (
            isinstance(self.relative_tolerance, (bool, np.bool_))
            or not np.isfinite(self.relative_tolerance)
            or self.relative_tolerance <= 0
        ):
            raise ValueError(
                "Plane-stress relative tolerance must be finite and positive."
            )
        if (
            isinstance(self.absolute_tolerance, (bool, np.bool_))
            or not np.isfinite(self.absolute_tolerance)
            or self.absolute_tolerance < 0
        ):
            raise ValueError(
                "Plane-stress absolute tolerance must be finite and nonnegative."
            )
        for value in (self.maximum_iterations, self.maximum_backtracks):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(
                    "Plane-stress iteration limits must be positive integers."
                )
        object.__setattr__(self, "relative_tolerance", float(self.relative_tolerance))
        object.__setattr__(self, "absolute_tolerance", float(self.absolute_tolerance))

    def summary(self):
        return {
            "kind": "finite_strain_plane_stress_reduction",
            "symmetry": self.symmetry,
            "transverse_equation": "P33=0; transverse shear checked",
            "unknown": "positive_F33",
            "tangent": "full_update_Schur_complement_11_12_21_22",
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "maximum_iterations": int(self.maximum_iterations),
            "maximum_backtracks": int(self.maximum_backtracks),
        }


@dataclass(frozen=True)
class PlaneStressPointResponse:
    """Uncommitted original 3D material response plus its planar reduction."""

    material_response: MaterialPointOutput
    deformation_gradient: np.ndarray
    first_piola_stress: np.ndarray
    in_plane_tangent: np.ndarray
    iterations: int
    residual: float
    tolerance: float


@dataclass(frozen=True)
class PlaneStressTangentCheck:
    """Fixed-old-state numerical evidence for the reduced 4-by-4 derivative."""

    relative_error: float
    maximum_absolute_error: float
    relative_step: float
    tolerance: float
    accepted: bool
    reduction: PlaneStressReduction

    def summary(self):
        return {
            "kind": "plane_stress_tangent_check",
            "method": "central_difference_fixed_old_state_with_transverse_equilibrium",
            "relative_error": self.relative_error,
            "maximum_absolute_error": self.maximum_absolute_error,
            "relative_step": self.relative_step,
            "tolerance": self.tolerance,
            "accepted": self.accepted,
            "stress_measure": "first_piola",
            "kinematic_measure": "in_plane_deformation_gradient",
            "component_order": ["11", "12", "21", "22"],
            "storage": "matrix_4x4",
            "reduction": self.reduction.summary(),
            "scope": "local_discrete_derivative_not_material_validation",
        }


@dataclass(frozen=True)
class PlaneStressBatchResponse:
    """Reduction evidence accompanying, not replacing, full material responses."""

    deformation_gradient: np.ndarray
    in_plane_tangent: np.ndarray
    iterations: np.ndarray
    residual: np.ndarray
    tolerance: np.ndarray

    def __post_init__(self):
        F = np.asarray(self.deformation_gradient, dtype=float)
        count = len(F)
        expected = {
            "deformation_gradient": (count, 3, 3),
            "in_plane_tangent": (count, 4, 4),
            "iterations": (count,),
            "residual": (count,),
            "tolerance": (count,),
        }
        for name, shape in expected.items():
            array = np.asarray(getattr(self, name))
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Plane-stress {name} requires finite shape {shape}.")
            if name == "iterations" and (
                array.dtype.kind not in "iu" or np.any(array < 0)
            ):
                raise ValueError(
                    "Plane-stress iteration counts must be nonnegative integers."
                )
            object.__setattr__(self, name, array.copy())
        if (
            np.any(np.linalg.det(F) <= 0)
            or np.any(F[:, 2, 2] <= 0)
            or np.any(self.tolerance <= 0)
            or np.any(abs(self.residual) > self.tolerance)
        ):
            raise ValueError(
                "Plane-stress batch is not an admissible converged reduction."
            )

    def embedded_tangent(self):
        """Assembly-only embedding; not a full 3D constitutive derivative."""
        result = np.zeros((len(self.deformation_gradient), 9, 9))
        result[:, _IN_PLANE[:, None], _IN_PLANE] = self.in_plane_tangent
        return result


def _require_planar(F):
    scale = max(_norm(F), np.finfo(float).tiny)
    if (
        np.max(np.abs(F.ravel()[_SHEAR])) > 64 * np.finfo(float).eps * scale
        or F[2, 2] <= 0
    ):
        raise ValueError(
            "Scalar plane stress requires block-diagonal planar F and positive F33."
        )


def reduce_plane_stress(
    material, point: MaterialPointInput, *, reduction: PlaneStressReduction
):
    """Solve P33=0 with fixed old state/time and condense the discrete tangent.

    The supplied new F33 is a positive initial guess, not prescribed data.
    Every local candidate receives a fresh copy of the same accepted request.
    Failure propagates without publishing partial state. The returned original
    response has its actual Cauchy stress and full 3D tangent; neither is scaled
    to make a plane-strain conversion accidentally look correct.
    """
    if not isinstance(reduction, PlaneStressReduction):
        raise TypeError("A PlaneStressReduction declaration is required.")
    if (
        material.tangent_convention
        != MaterialTangentConvention.first_piola_deformation_gradient()
    ):
        raise ValueError(
            "Plane stress requires the full reference 9x9 dP/dF material tangent."
        )
    _require_planar(point.deformation_gradient_old)
    _require_planar(point.deformation_gradient_new)

    def evaluate(log_stretch):
        with np.errstate(over="raise", under="raise", invalid="raise"):
            stretch = float(np.exp(log_stretch))
        F = point.deformation_gradient_new.copy()
        F[2, 2] = stretch
        response = validated_material_update(
            material, replace(point, deformation_gradient_new=F)
        )
        P = np.linalg.det(F) * response.cauchy_stress @ np.linalg.inv(F).T
        if not np.isfinite(P).all():
            raise ValueError("Nonfinite plane-stress nominal stress.")
        return F, response, P

    eta = float(np.log(point.deformation_gradient_new[2, 2]))
    F, response, P = evaluate(eta)
    # Freeze a physical stress scale for this local solve. No implicit unit-size
    # floor: rescaling the same law from Pa to MPa must preserve the criterion.
    scale = max(
        _norm(response.consistent_tangent),
        _norm(P),
        np.finfo(float).tiny,
    )
    tolerance = reduction.absolute_tolerance + reduction.relative_tolerance * scale
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            "Plane-stress tolerance must be representable in the supplied units."
        )
    for iteration in range(reduction.maximum_iterations + 1):
        A = response.consistent_tangent
        stiffness_scale = max(_norm(A), np.finfo(float).tiny)
        if A[8, 8] <= 64 * np.finfo(float).eps * stiffness_scale:
            raise RuntimeError(
                "AFM-PLANE-STRESS-SINGULAR: positive transverse stiffness is required."
            )
        if abs(P[2, 2]) <= tolerance:
            symmetry_tolerance = (
                max(reduction.relative_tolerance, 1e-10) * stiffness_scale
            )
            if (
                np.max(np.abs(P.ravel()[_SHEAR])) > tolerance
                or np.max(np.abs(A[np.ix_(_SHEAR, _PLANAR)])) > symmetry_tolerance
                or np.max(np.abs(A[np.ix_(_PLANAR, _SHEAR)])) > symmetry_tolerance
            ):
                raise ValueError(
                    "AFM-PLANE-STRESS-SYMMETRY: transverse shear/coupling violates the declared planar reduction."
                )
            reduced = (
                A[np.ix_(_IN_PLANE, _IN_PLANE)]
                # Divide before the outer product: multiplying two dimensional
                # stiffnesses first can overflow/underflow despite a perfectly
                # representable condensed result after unit rescaling.
                - np.outer(A[_IN_PLANE, 8] / A[8, 8], A[8, _IN_PLANE])
            )
            if not np.isfinite(reduced).all():
                raise RuntimeError("Nonfinite condensed plane-stress tangent.")
            return PlaneStressPointResponse(
                response,
                F.copy(),
                P.copy(),
                reduced,
                iteration,
                float(P[2, 2]),
                tolerance,
            )
        if iteration == reduction.maximum_iterations:
            break
        direction = -(P[2, 2] / A[8, 8]) / F[2, 2]
        alpha = 1.0
        candidate = None
        for _ in range(reduction.maximum_backtracks):
            try:
                trial = evaluate(eta + alpha * direction)
                if abs(trial[2][2, 2]) < abs(P[2, 2]):
                    candidate = trial
                    break
            except _MaterialInputMutationError:
                raise  # A verification contract error cannot be fixed by backtracking.
            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ):
                pass
            alpha *= 0.5
        if candidate is None:
            raise RuntimeError("AFM-PLANE-STRESS-LOCAL: transverse line search failed.")
        eta += alpha * direction
        F, response, P = candidate
    raise RuntimeError(
        "AFM-PLANE-STRESS-LOCAL: transverse equilibrium did not converge."
    )


def check_plane_stress_tangent(
    material,
    point: MaterialPointInput,
    *,
    reduction: PlaneStressReduction,
    relative_step: float = 1e-6,
    tolerance: float = 1e-5,
) -> PlaneStressTangentCheck:
    """Check the reduced derivative by independently resolving each perturbation.

    Old F, history and physical dt remain fixed. Every positive and negative
    in-plane perturbation solves its own transverse equilibrium. Inadmissible
    perturbations fail explicitly; this checker does not silently replace a
    centered difference by a one-sided derivative at a constitutive switch.
    Choose a perturbation resolvable above the local equilibrium tolerance and
    verify each smooth branch separately for nonsmooth history laws.
    """
    step, tolerance = float(relative_step), float(tolerance)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("relative_step must be finite and positive.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    material = _VerificationMaterial(material)
    baseline = reduce_plane_stress(material, point, reduction=reduction)
    numerical = np.empty((4, 4))
    for column, component in enumerate(_IN_PLANE):
        increment = step * max(
            1.0, abs(float(point.deformation_gradient_new.flat[component]))
        )
        piola = []
        for sign in (1.0, -1.0):
            gradient = point.deformation_gradient_new.copy()
            gradient.flat[component] += sign * increment
            if (
                gradient.flat[component]
                == point.deformation_gradient_new.flat[component]
            ):
                raise ValueError(
                    "Tangent perturbation is not representable; increase relative_step."
                )
            response = reduce_plane_stress(
                material,
                replace(point, deformation_gradient_new=gradient),
                reduction=reduction,
            )
            piola.append(response.first_piola_stress.ravel()[_IN_PLANE])
        numerical[:, column] = (piola[0] - piola[1]) / (2 * increment)
    if not np.isfinite(numerical).all():
        raise ValueError(
            "Nonfinite reduced tangent differences; check perturbation and units."
        )
    error, absolute = _tangent_error_metrics(baseline.in_plane_tangent, numerical)
    return PlaneStressTangentCheck(
        error,
        absolute,
        step,
        tolerance,
        bool(np.isfinite(error) and error <= tolerance),
        reduction,
    )

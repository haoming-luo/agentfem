# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Finite-strain J2 material-point integration.

The first provider in this module uses a multiplicative decomposition
``F = Fe Fp`` and a quadratic Hencky elastic potential.  The local return is
performed in elastic logarithmic-strain space.  It is deliberately named for
that formulation instead of being presented as a generic finite-strain J2
implementation: different elastic potentials and objective integrations are
not interchangeable at large strain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import ClassVar

import numpy as np

from .user_material import (
    MaterialPointBatchInput,
    MaterialPointBatchOutput,
    MaterialPointInput,
    MaterialPointOutput,
    MaterialStateSchema,
    MaterialStateVariable,
    MaterialTangentConvention,
)


_IDENTITY_3D = np.eye(3)


@dataclass(frozen=True)
class _FiniteStrainJ2Integration:
    cauchy_stress: np.ndarray
    first_piola_stress: np.ndarray
    state: np.ndarray
    strain_energy_density: float
    elastic_energy_density: float
    hardening_energy_density: float
    plastic_dissipation_density: float
    trial_yield_function: float
    plastic_multiplier_increment: float


@dataclass(frozen=True)
class FiniteStrainJ2Logarithmic:
    """Multiplicative finite-strain J2 plasticity with Hencky elasticity.

    The elastic free energy is

    ``psi_e = mu ||dev(log(Ve))||^2 + K/2 tr(log(Ve))^2``

    and the yield stress is ``sigma_y + H p``.  Associated J2 flow is
    integrated by a radial return in principal elastic logarithmic-strain
    space.  The plastic flow is isochoric, so ``det(Fp)`` remains one from the
    declared identity initial state.

    The provider returns the derivative of the complete discrete material
    update, ``dP/dF``, with the old state held fixed.  The production path uses
    the spectral derivative of the radial return.  A central-difference path
    remains available as an independent implementation oracle.
    """

    stateful_constitutive: ClassVar[bool] = True
    stored_energy_component_names: ClassVar[tuple[str, ...]] = (
        "ELENER",
        "HARDENER",
    )

    young: float
    poisson: float
    yield_stress: float
    hardening_modulus: float = 0.0
    tangent_relative_step: float = 2.0e-6
    tangent_evaluation: str = "analytic_spectral"
    name: str = "finite-strain logarithmic J2 plasticity"
    state_schema: MaterialStateSchema = field(init=False, repr=False)
    tangent_convention: MaterialTangentConvention = field(init=False, repr=False)

    def __post_init__(self) -> None:
        values = (
            self.young,
            self.poisson,
            self.yield_stress,
            self.hardening_modulus,
            self.tangent_relative_step,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("Finite-strain J2 parameters must be finite.")
        if self.young <= 0.0 or self.yield_stress <= 0.0:
            raise ValueError("young and yield_stress must be positive.")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must satisfy -1 < nu < 0.5.")
        if self.hardening_modulus < 0.0:
            raise ValueError("hardening_modulus must be nonnegative.")
        if self.tangent_relative_step <= 0.0:
            raise ValueError("tangent_relative_step must be positive.")
        tangent_evaluation = str(self.tangent_evaluation).strip().lower()
        if tangent_evaluation not in {"analytic_spectral", "central_difference"}:
            raise ValueError(
                "tangent_evaluation must be analytic_spectral or central_difference."
            )
        object.__setattr__(self, "tangent_evaluation", tangent_evaluation)
        object.__setattr__(
            self,
            "state_schema",
            MaterialStateSchema(
                "agentfem.finite_strain_j2_logarithmic_state",
                (
                    MaterialStateVariable(
                        "plastic_deformation_gradient",
                        shape=(3, 3),
                        initial_value=_IDENTITY_3D,
                        unit="1",
                        description=(
                            "Plastic part Fp of the multiplicative deformation "
                            "gradient, initialized to the identity."
                        ),
                        output_name="FP",
                    ),
                    MaterialStateVariable(
                        "equivalent_plastic_strain",
                        unit="1",
                        description="Accumulated equivalent plastic strain.",
                        output_name="PEEQ",
                    ),
                    MaterialStateVariable(
                        "plastic_dissipation",
                        unit="consistent stress unit",
                        description=(
                            "Cumulative irrecoverable plastic dissipation per "
                            "reference volume."
                        ),
                        output_name="PDENER",
                    ),
                ),
                version="0.2.0",
            ),
        )
        object.__setattr__(
            self,
            "tangent_convention",
            MaterialTangentConvention.first_piola_deformation_gradient(),
        )

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def bulk_modulus(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    @property
    def properties(self) -> np.ndarray:
        return np.asarray(
            (
                self.young,
                self.poisson,
                self.yield_stress,
                self.hardening_modulus,
            ),
            dtype=float,
        )

    def current_yield_stress(self, equivalent_plastic_strain: float) -> float:
        selected = float(equivalent_plastic_strain)
        if not isfinite(selected) or selected < 0.0:
            raise ValueError(
                "equivalent_plastic_strain must be finite and nonnegative."
            )
        return self.yield_stress + self.hardening_modulus * selected

    def _validate_point(self, point: MaterialPointInput) -> None:
        if point.state_schema is not None and (
            point.state_schema.identity != self.state_schema.identity
        ):
            raise ValueError("Material-point state schema does not match the material.")
        if point.properties.size not in {0, 4}:
            raise ValueError(
                "FiniteStrainJ2Logarithmic expects no duplicated properties or "
                "[young, poisson, yield_stress, hardening_modulus]."
            )
        if point.properties.size and not np.allclose(
            point.properties,
            self.properties,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                "MaterialPointInput properties conflict with the provider values."
            )

    def _integrate(self, deformation_gradient, state_old) -> _FiniteStrainJ2Integration:
        deformation_gradient = np.asarray(deformation_gradient, dtype=float)
        if deformation_gradient.shape != (3, 3):
            raise ValueError("deformation_gradient must be a 3x3 matrix.")
        jacobian = float(np.linalg.det(deformation_gradient))
        if not np.all(np.isfinite(deformation_gradient)) or jacobian <= 0.0:
            raise ValueError("deformation_gradient must be finite with positive J.")

        state = self.state_schema.unpack(state_old)
        plastic_gradient = np.asarray(
            state["plastic_deformation_gradient"], dtype=float
        )
        equivalent_plastic_strain = float(state["equivalent_plastic_strain"])
        plastic_dissipation = float(state["plastic_dissipation"])
        if equivalent_plastic_strain < 0.0 or plastic_dissipation < 0.0:
            raise ValueError(
                "Committed equivalent plastic strain and plastic dissipation "
                "must be nonnegative."
            )
        expected_dissipation = self.yield_stress * equivalent_plastic_strain
        if not np.isclose(
            plastic_dissipation,
            expected_dissipation,
            rtol=1.0e-10,
            atol=256.0 * np.finfo(float).eps * max(1.0, abs(expected_dissipation)),
        ):
            raise ValueError(
                "Committed PDENER is inconsistent with the declared "
                "rate-independent linear-hardening J2 history."
            )
        plastic_jacobian = float(np.linalg.det(plastic_gradient))
        if plastic_jacobian <= 0.0:
            raise ValueError("The committed plastic deformation gradient is inverted.")
        if abs(plastic_jacobian - 1.0) > 1.0e-8:
            raise ValueError(
                "Finite-strain J2 requires an isochoric committed plastic state; "
                f"det(Fp)={plastic_jacobian:.16g}."
            )

        elastic_trial = deformation_gradient @ np.linalg.inv(plastic_gradient)
        left_vectors, stretches, right_vectors_transpose = np.linalg.svd(elastic_trial)
        if np.any(stretches <= 0.0):
            raise ValueError("Elastic principal stretches must be positive.")
        elastic_rotation = left_vectors @ right_vectors_transpose
        if np.linalg.det(elastic_rotation) <= 0.0:
            raise RuntimeError("Elastic polar decomposition produced a reflection.")

        logarithmic_strain_trial = np.log(stretches)
        volumetric_logarithmic_strain = float(np.sum(logarithmic_strain_trial))
        deviatoric_logarithmic_trial = (
            logarithmic_strain_trial - volumetric_logarithmic_strain / 3.0
        )
        deviatoric_kirchhoff_trial = (
            2.0 * self.shear_modulus * deviatoric_logarithmic_trial
        )
        equivalent_trial = float(
            np.sqrt(
                1.5 * np.dot(deviatoric_kirchhoff_trial, deviatoric_kirchhoff_trial)
            )
        )
        trial_yield = equivalent_trial - self.current_yield_stress(
            equivalent_plastic_strain
        )

        tolerance = (
            64.0
            * np.finfo(float).eps
            * max(
                self.young,
                self.yield_stress,
                equivalent_trial,
            )
        )
        plastic_increment = 0.0
        logarithmic_strain = logarithmic_strain_trial.copy()
        deviatoric_kirchhoff = deviatoric_kirchhoff_trial.copy()
        state_new = self.state_schema.validate(state_old)
        if trial_yield > tolerance:
            plastic_increment = trial_yield / (
                3.0 * self.shear_modulus + self.hardening_modulus
            )
            radial_scale = max(
                0.0,
                1.0 - 3.0 * self.shear_modulus * plastic_increment / equivalent_trial,
            )
            deviatoric_kirchhoff = radial_scale * deviatoric_kirchhoff_trial
            logarithmic_strain = (
                volumetric_logarithmic_strain / 3.0
                + deviatoric_kirchhoff / (2.0 * self.shear_modulus)
            )

            elastic_left_stretch = (
                left_vectors @ np.diag(np.exp(logarithmic_strain)) @ left_vectors.T
            )
            elastic_new = elastic_left_stretch @ elastic_rotation
            plastic_new = np.linalg.solve(elastic_new, deformation_gradient)
            plastic_new_jacobian = float(np.linalg.det(plastic_new))
            if abs(plastic_new_jacobian - 1.0) > 2.0e-10:
                raise RuntimeError(
                    "Isochoric finite-strain J2 update drifted from det(Fp)=1: "
                    f"{plastic_new_jacobian:.16g}."
                )
            state_new = np.concatenate(
                (
                    plastic_new.reshape(-1),
                    np.asarray(
                        [
                            equivalent_plastic_strain + plastic_increment,
                            plastic_dissipation + self.yield_stress * plastic_increment,
                        ],
                        dtype=float,
                    ),
                )
            )

        principal_kirchhoff = (
            self.bulk_modulus * volumetric_logarithmic_strain + deviatoric_kirchhoff
        )
        kirchhoff_stress = left_vectors @ np.diag(principal_kirchhoff) @ left_vectors.T
        kirchhoff_stress = 0.5 * (kirchhoff_stress + kirchhoff_stress.T)
        cauchy_stress = kirchhoff_stress / jacobian
        first_piola_stress = kirchhoff_stress @ np.linalg.inv(deformation_gradient).T
        elastic_energy = (
            self.shear_modulus
            * float(
                np.dot(
                    logarithmic_strain - np.mean(logarithmic_strain),
                    logarithmic_strain - np.mean(logarithmic_strain),
                )
            )
            + 0.5 * self.bulk_modulus * volumetric_logarithmic_strain**2
        )
        hardening_energy = (
            0.5
            * self.hardening_modulus
            * (equivalent_plastic_strain + plastic_increment) ** 2
        )
        return _FiniteStrainJ2Integration(
            cauchy_stress=cauchy_stress,
            first_piola_stress=first_piola_stress,
            state=state_new,
            strain_energy_density=elastic_energy + hardening_energy,
            elastic_energy_density=elastic_energy,
            hardening_energy_density=hardening_energy,
            plastic_dissipation_density=(
                plastic_dissipation + self.yield_stress * plastic_increment
            ),
            trial_yield_function=trial_yield,
            plastic_multiplier_increment=plastic_increment,
        )

    def _algorithmic_tangent(
        self,
        deformation_gradient,
        state_old,
        *,
        baseline: _FiniteStrainJ2Integration | None = None,
    ) -> np.ndarray:
        selected = np.asarray(deformation_gradient, dtype=float)
        baseline_piola = (
            self._integrate(selected, state_old).first_piola_stress
            if baseline is None
            else baseline.first_piola_stress
        )
        tangent = np.empty((9, 9), dtype=float)
        for column in range(9):
            row, component = divmod(column, 3)
            increment = self.tangent_relative_step * max(
                1.0, abs(float(selected[row, component]))
            )
            plus = selected.copy()
            minus = selected.copy()
            plus[row, component] += increment
            minus[row, component] -= increment
            plus_piola = self._integrate(plus, state_old).first_piola_stress
            if np.linalg.det(minus) > 0.0:
                minus_piola = self._integrate(minus, state_old).first_piola_stress
                derivative = (plus_piola - minus_piola) / (2.0 * increment)
            else:
                derivative = (plus_piola - baseline_piola) / increment
            tangent[:, column] = derivative.reshape(-1)
        return tangent

    def _analytic_algorithmic_tangent_batch(
        self,
        deformation_gradients,
        states_old,
        *,
        baseline,
    ) -> np.ndarray:
        """Return the spectral derivative of the complete ``P(F)`` update."""

        gradients = np.asarray(deformation_gradients, dtype=float)
        states = np.asarray(states_old, dtype=float)
        point_count = len(gradients)
        plastic_gradients = states[:, :9].reshape((-1, 3, 3))
        inverse_plastic = np.linalg.inv(plastic_gradients)
        elastic_trial = gradients @ inverse_plastic
        left_vectors, stretches, _right_vectors_transpose = np.linalg.svd(elastic_trial)
        eigenvalues = stretches**2
        logarithmic_trial = np.log(stretches)
        volumetric = np.sum(logarithmic_trial, axis=1)
        deviatoric_logarithmic = logarithmic_trial - volumetric[:, None] / 3.0
        deviatoric_trial = 2.0 * self.shear_modulus * deviatoric_logarithmic
        equivalent_trial = np.sqrt(1.5 * np.sum(deviatoric_trial**2, axis=1))
        equivalent_plastic_strain = states[:, 9]
        yield_level = (
            self.yield_stress + self.hardening_modulus * equivalent_plastic_strain
        )
        trial_yield = equivalent_trial - yield_level
        tolerance = (
            64.0
            * np.finfo(float).eps
            * np.maximum.reduce(
                (
                    np.full(point_count, self.young),
                    np.full(point_count, self.yield_stress),
                    equivalent_trial,
                )
            )
        )
        plastic = trial_yield > tolerance

        deviatoric_projector = np.eye(3) - np.ones((3, 3)) / 3.0
        elastic_principal_moduli = (
            self.bulk_modulus * np.ones((3, 3))
            + 2.0 * self.shear_modulus * deviatoric_projector
        )
        principal_moduli = np.broadcast_to(
            elastic_principal_moduli,
            (point_count, 3, 3),
        ).copy()
        radial_scale = np.ones(point_count, dtype=float)
        if np.any(plastic):
            denominator = 3.0 * self.shear_modulus + self.hardening_modulus
            radial_scale[plastic] = 1.0 - 3.0 * self.shear_modulus * trial_yield[
                plastic
            ] / (denominator * equivalent_trial[plastic])
            dyadic_coefficient = (
                9.0
                * self.shear_modulus**2
                * yield_level[plastic]
                / (denominator * equivalent_trial[plastic] ** 3)
            )
            principal_moduli[plastic] = (
                self.bulk_modulus * np.ones((3, 3))
                + 2.0
                * self.shear_modulus
                * radial_scale[plastic, None, None]
                * deviatoric_projector
                - dyadic_coefficient[:, None, None]
                * np.einsum(
                    "pi,pj->pij",
                    deviatoric_trial[plastic],
                    deviatoric_trial[plastic],
                )
            )

        principal_stress = (
            self.bulk_modulus * volumetric[:, None]
            + radial_scale[:, None] * deviatoric_trial
        )
        kirchhoff_stress = np.einsum(
            "pia,pa,pja->pij",
            left_vectors,
            principal_stress,
            left_vectors,
        )
        inverse_transpose = np.swapaxes(np.linalg.inv(gradients), 1, 2)
        first_piola = kirchhoff_stress @ inverse_transpose
        baseline_piola = np.asarray(baseline["first_piola_stress"], dtype=float)
        scale = np.maximum(
            1.0,
            np.max(np.abs(baseline_piola), axis=(1, 2)),
        )
        mismatch = np.max(np.abs(first_piola - baseline_piola), axis=(1, 2))
        if np.any(mismatch > 2.0e-11 * scale):
            raise RuntimeError(
                "Analytic finite-strain J2 tangent reconstructed a response "
                "that differs from the discrete return."
            )

        principal_derivative = principal_moduli / (2.0 * eigenvalues[:, None, :])
        divided_difference = np.zeros((point_count, 3, 3), dtype=float)
        for first in range(3):
            for second in range(3):
                if first == second:
                    continue
                difference = eigenvalues[:, first] - eigenvalues[:, second]
                repeated = np.abs(difference) <= (
                    1.0e-10
                    * np.maximum.reduce(
                        (
                            np.ones(point_count),
                            np.abs(eigenvalues[:, first]),
                            np.abs(eigenvalues[:, second]),
                        )
                    )
                )
                distinct = ~repeated
                divided_difference[distinct, first, second] = (
                    principal_stress[distinct, first]
                    - principal_stress[distinct, second]
                ) / difference[distinct]
                divided_difference[repeated, first, second] = (
                    principal_moduli[repeated, first, first]
                    - principal_moduli[repeated, first, second]
                ) / (2.0 * eigenvalues[repeated, first])

        tangent = np.empty((point_count, 9, 9), dtype=float)
        for column in range(9):
            row, component = divmod(column, 3)
            variation_elastic = np.zeros_like(elastic_trial)
            variation_elastic[:, row, :] = inverse_plastic[:, component, :]
            variation_left = variation_elastic @ np.swapaxes(
                elastic_trial, 1, 2
            ) + elastic_trial @ np.swapaxes(variation_elastic, 1, 2)
            principal_variation = np.einsum(
                "pia,pij,pjb->pab",
                left_vectors,
                variation_left,
                left_vectors,
            )
            variation_stress_principal = divided_difference * principal_variation
            diagonal_variation = np.diagonal(
                principal_variation,
                axis1=1,
                axis2=2,
            )
            diagonal_stress = np.einsum(
                "pij,pj->pi",
                principal_derivative,
                diagonal_variation,
            )
            indices = np.arange(3)
            variation_stress_principal[:, indices, indices] = diagonal_stress
            variation_kirchhoff = np.einsum(
                "pia,pab,pjb->pij",
                left_vectors,
                variation_stress_principal,
                left_vectors,
            )
            variation_gradient = np.zeros((3, 3), dtype=float)
            variation_gradient[row, component] = 1.0
            variation_inverse_transpose = -np.einsum(
                "pij,jk,pkl->pil",
                inverse_transpose,
                variation_gradient.T,
                inverse_transpose,
            )
            variation_piola = (
                variation_kirchhoff @ inverse_transpose
                + kirchhoff_stress @ variation_inverse_transpose
            )
            tangent[:, :, column] = variation_piola.reshape((-1, 9))
        return tangent

    def _integrate_batch(self, deformation_gradients, states_old):
        """Vectorize the discrete return over one rank-local point batch."""

        gradients = np.asarray(deformation_gradients, dtype=float)
        states = np.asarray(states_old, dtype=float)
        if gradients.ndim != 3 or gradients.shape[1:] != (3, 3):
            raise ValueError("deformation_gradients must have shape (n, 3, 3).")
        if states.shape != (len(gradients), self.state_schema.size):
            raise ValueError(
                "states_old must provide one complete state vector per point."
            )
        if not np.all(np.isfinite(gradients)) or not np.all(np.isfinite(states)):
            raise ValueError("Finite-strain J2 batch inputs must be finite.")
        jacobians = np.linalg.det(gradients)
        if np.any(jacobians <= 0.0):
            raise ValueError("Every deformation gradient must have positive J.")

        plastic_gradients = states[:, :9].reshape((-1, 3, 3))
        equivalent_plastic_strain = states[:, 9]
        plastic_dissipation = states[:, 10]
        if np.any(equivalent_plastic_strain < 0.0) or np.any(plastic_dissipation < 0.0):
            raise ValueError(
                "Committed equivalent plastic strain and plastic dissipation "
                "must be nonnegative."
            )
        expected_dissipation = self.yield_stress * equivalent_plastic_strain
        dissipation_tolerance = (
            256.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(expected_dissipation))
        )
        if np.any(
            np.abs(plastic_dissipation - expected_dissipation)
            > 1.0e-10 * np.abs(expected_dissipation) + dissipation_tolerance
        ):
            raise ValueError(
                "Committed PDENER is inconsistent with the declared "
                "rate-independent linear-hardening J2 history."
            )
        plastic_jacobians = np.linalg.det(plastic_gradients)
        if np.any(plastic_jacobians <= 0.0):
            raise ValueError("A committed plastic deformation gradient is inverted.")
        if np.any(np.abs(plastic_jacobians - 1.0) > 1.0e-8):
            raise ValueError(
                "Finite-strain J2 requires isochoric committed plastic states."
            )

        elastic_trial = gradients @ np.linalg.inv(plastic_gradients)
        left_vectors, stretches, right_vectors_transpose = np.linalg.svd(elastic_trial)
        if np.any(stretches <= 0.0):
            raise ValueError("Elastic principal stretches must be positive.")
        elastic_rotation = left_vectors @ right_vectors_transpose
        if np.any(np.linalg.det(elastic_rotation) <= 0.0):
            raise RuntimeError("Elastic polar decomposition produced a reflection.")

        logarithmic_trial = np.log(stretches)
        volumetric_logarithmic_strain = np.sum(logarithmic_trial, axis=1)
        deviatoric_logarithmic_trial = (
            logarithmic_trial - volumetric_logarithmic_strain[:, None] / 3.0
        )
        deviatoric_kirchhoff_trial = (
            2.0 * self.shear_modulus * deviatoric_logarithmic_trial
        )
        equivalent_trial = np.sqrt(1.5 * np.sum(deviatoric_kirchhoff_trial**2, axis=1))
        trial_yield = equivalent_trial - (
            self.yield_stress + self.hardening_modulus * equivalent_plastic_strain
        )
        tolerance = (
            64.0
            * np.finfo(float).eps
            * np.maximum.reduce(
                (
                    np.full_like(equivalent_trial, self.young),
                    np.full_like(equivalent_trial, self.yield_stress),
                    equivalent_trial,
                )
            )
        )
        plastic = trial_yield > tolerance
        plastic_increment = np.zeros_like(equivalent_trial)
        plastic_increment[plastic] = trial_yield[plastic] / (
            3.0 * self.shear_modulus + self.hardening_modulus
        )
        radial_scale = np.ones_like(equivalent_trial)
        radial_scale[plastic] = np.maximum(
            0.0,
            1.0
            - 3.0
            * self.shear_modulus
            * plastic_increment[plastic]
            / equivalent_trial[plastic],
        )
        deviatoric_kirchhoff = radial_scale[:, None] * deviatoric_kirchhoff_trial
        logarithmic_strain = volumetric_logarithmic_strain[
            :, None
        ] / 3.0 + deviatoric_kirchhoff / (2.0 * self.shear_modulus)

        states_new = states.copy()
        if np.any(plastic):
            selected_vectors = left_vectors[plastic]
            elastic_left_stretch = np.einsum(
                "nia,na,nja->nij",
                selected_vectors,
                np.exp(logarithmic_strain[plastic]),
                selected_vectors,
            )
            elastic_new = elastic_left_stretch @ elastic_rotation[plastic]
            plastic_new = np.linalg.solve(elastic_new, gradients[plastic])
            if np.any(np.abs(np.linalg.det(plastic_new) - 1.0) > 2.0e-10):
                raise RuntimeError(
                    "Isochoric finite-strain J2 batch update drifted from det(Fp)=1."
                )
            states_new[plastic, :9] = plastic_new.reshape((-1, 9))
            states_new[plastic, 9] += plastic_increment[plastic]
            states_new[plastic, 10] += self.yield_stress * plastic_increment[plastic]

        principal_kirchhoff = (
            self.bulk_modulus * volumetric_logarithmic_strain[:, None]
            + deviatoric_kirchhoff
        )
        kirchhoff_stress = np.einsum(
            "nia,na,nja->nij",
            left_vectors,
            principal_kirchhoff,
            left_vectors,
        )
        kirchhoff_stress = 0.5 * (
            kirchhoff_stress + np.swapaxes(kirchhoff_stress, 1, 2)
        )
        inverse_transpose = np.swapaxes(np.linalg.inv(gradients), 1, 2)
        cauchy_stress = kirchhoff_stress / jacobians[:, None, None]
        first_piola_stress = kirchhoff_stress @ inverse_transpose
        deviatoric_logarithmic = (
            logarithmic_strain - np.mean(logarithmic_strain, axis=1)[:, None]
        )
        elastic_energy = (
            self.shear_modulus * np.sum(deviatoric_logarithmic**2, axis=1)
            + 0.5 * self.bulk_modulus * volumetric_logarithmic_strain**2
        )
        hardening_energy = (
            0.5
            * self.hardening_modulus
            * (equivalent_plastic_strain + plastic_increment) ** 2
        )
        return {
            "cauchy_stress": cauchy_stress,
            "first_piola_stress": first_piola_stress,
            "state": states_new,
            "strain_energy_density": elastic_energy + hardening_energy,
            "elastic_energy_density": elastic_energy,
            "hardening_energy_density": hardening_energy,
            "plastic_dissipation_density": (
                plastic_dissipation + self.yield_stress * plastic_increment
            ),
            "trial_yield_function": trial_yield,
            "plastic_multiplier_increment": plastic_increment,
        }

    def _numerical_algorithmic_tangent_batch(
        self,
        deformation_gradients,
        states_old,
        *,
        baseline,
    ) -> np.ndarray:
        gradients = np.asarray(deformation_gradients, dtype=float)
        states = np.asarray(states_old, dtype=float)
        tangent = np.empty((len(gradients), 9, 9), dtype=float)
        for column in range(9):
            row, component = divmod(column, 3)
            increments = self.tangent_relative_step * np.maximum(
                1.0,
                np.abs(gradients[:, row, component]),
            )
            plus = gradients.copy()
            minus = gradients.copy()
            plus[:, row, component] += increments
            minus[:, row, component] -= increments
            plus_piola = self._integrate_batch(plus, states)["first_piola_stress"]
            central = np.linalg.det(minus) > 0.0
            derivative = (plus_piola - baseline["first_piola_stress"]) / increments[
                :, None, None
            ]
            if np.any(central):
                minus_piola = self._integrate_batch(
                    minus[central],
                    states[central],
                )["first_piola_stress"]
                derivative[central] = (plus_piola[central] - minus_piola) / (
                    2.0 * increments[central, None, None]
                )
            tangent[:, :, column] = derivative.reshape((-1, 9))
        return tangent

    def _selected_algorithmic_tangent_batch(
        self,
        deformation_gradients,
        states_old,
        *,
        baseline,
    ) -> np.ndarray:
        if self.tangent_evaluation == "analytic_spectral":
            return self._analytic_algorithmic_tangent_batch(
                deformation_gradients,
                states_old,
                baseline=baseline,
            )
        return self._numerical_algorithmic_tangent_batch(
            deformation_gradients,
            states_old,
            baseline=baseline,
        )

    def update(self, point: MaterialPointInput) -> MaterialPointOutput:
        """Advance one point and return Cauchy stress, state and ``dP/dF``."""

        self._validate_point(point)
        integrated = self._integrate(
            point.deformation_gradient_new,
            point.state_old,
        )
        return MaterialPointOutput(
            cauchy_stress=integrated.cauchy_stress,
            consistent_tangent=(
                self._analytic_algorithmic_tangent_batch(
                    np.asarray(point.deformation_gradient_new)[None, ...],
                    np.asarray(point.state_old)[None, ...],
                    baseline={
                        "first_piola_stress": integrated.first_piola_stress[None, ...]
                    },
                )[0]
                if self.tangent_evaluation == "analytic_spectral"
                else self._algorithmic_tangent(
                    point.deformation_gradient_new,
                    point.state_old,
                    baseline=integrated,
                )
            ),
            state_new=integrated.state,
            strain_energy_density=integrated.strain_energy_density,
            stored_energy_density_components={
                "ELENER": integrated.elastic_energy_density,
                "HARDENER": integrated.hardening_energy_density,
            },
            tangent_convention=self.tangent_convention,
            state_schema=self.state_schema,
            dissipation_density_increment=(
                integrated.plastic_dissipation_density - point.state_old[10]
            ),
        )

    def update_array_batch(self, request):
        """Columnar form of the same discrete update and algorithmic tangent."""
        from .material_array_batch import (
            MaterialPointArrayBatchInput, MaterialPointArrayBatchOutput,
        )

        if not isinstance(request, MaterialPointArrayBatchInput):
            raise TypeError("request must be a MaterialPointArrayBatchInput.")
        self._validate_point(request)
        if request.state_schema != self.state_schema:
            raise ValueError("Array batch state schema does not match the material.")
        gradients, states = request.deformation_gradient_new, request.state_old
        integrated = self._integrate_batch(gradients, states)
        return MaterialPointArrayBatchOutput(
            cauchy_stress=integrated["cauchy_stress"],
            consistent_tangent=self._selected_algorithmic_tangent_batch(
                gradients, states, baseline=integrated,
            ),
            state_new=integrated["state"],
            tangent_convention=self.tangent_convention,
            state_schema=self.state_schema,
            strain_energy_density=integrated["strain_energy_density"],
            stored_energy_density_components={
                "ELENER": integrated["elastic_energy_density"],
                "HARDENER": integrated["hardening_energy_density"],
            },
            dissipation_density_increment=integrated["plastic_dissipation_density"] - states[:, 10],
        )

    def update_batch(
        self,
        request: MaterialPointBatchInput,
    ) -> MaterialPointBatchOutput:
        """Advance all rank-local points through one vectorized NumPy path."""

        if not isinstance(request, MaterialPointBatchInput):
            raise TypeError("request must be a MaterialPointBatchInput.")
        for point in request.points:
            self._validate_point(point)
        gradients = np.asarray(
            [point.deformation_gradient_new for point in request.points],
            dtype=float,
        )
        states = np.asarray(
            [point.state_old for point in request.points],
            dtype=float,
        )
        integrated = self._integrate_batch(gradients, states)
        tangents = self._selected_algorithmic_tangent_batch(
            gradients,
            states,
            baseline=integrated,
        )
        return MaterialPointBatchOutput(
            tuple(
                MaterialPointOutput(
                    cauchy_stress=integrated["cauchy_stress"][index],
                    consistent_tangent=tangents[index],
                    state_new=integrated["state"][index],
                    strain_energy_density=integrated["strain_energy_density"][index],
                    stored_energy_density_components={
                        "ELENER": integrated["elastic_energy_density"][index],
                        "HARDENER": integrated["hardening_energy_density"][index],
                    },
                    tangent_convention=self.tangent_convention,
                    state_schema=self.state_schema,
                    dissipation_density_increment=(
                        integrated["plastic_dissipation_density"][index]
                        - states[index, 10]
                    ),
                )
                for index in range(request.point_count)
            )
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "finite_strain_j2_logarithmic",
            "name": self.name,
            "status": "experimental_material_point",
            "parameters": {
                "young": self.young,
                "poisson": self.poisson,
                "yield_stress": self.yield_stress,
                "hardening_modulus": self.hardening_modulus,
            },
            "numerical_parameters": {
                "tangent_relative_step": self.tangent_relative_step,
            },
            "kinematics": "multiplicative_F_equals_Fe_Fp",
            "elastic_potential": "quadratic_Hencky",
            "plastic_flow": "associated_isochoric_J2",
            "hardening": "linear_isotropic",
            "stored_energy_density": {
                "SENER": "ELENER + HARDENER",
                "ELENER": "quadratic_Hencky_elastic_free_energy",
                "HARDENER": "linear_isotropic_hardening_free_energy",
                "PDENER": (
                    "cumulative_rate_independent_plastic_dissipation_"
                    "yield_stress_times_equivalent_plastic_strain"
                ),
            },
            "tangent": self.tangent_convention.summary(),
            "tangent_evaluation": self.tangent_evaluation,
            "batch_update": "numpy_vectorized_rank_local_points",
            "state_schema": self.state_schema.summary(),
        }

    def as_dict(self) -> dict[str, object]:
        return self.summary()


def finite_strain_j2_logarithmic(
    *,
    young: float,
    poisson: float,
    yield_stress: float,
    hardening_modulus: float = 0.0,
    tangent_relative_step: float = 2.0e-6,
    tangent_evaluation: str = "analytic_spectral",
) -> FiniteStrainJ2Logarithmic:
    """Create the logarithmic finite-strain J2 material provider."""

    return FiniteStrainJ2Logarithmic(
        young=young,
        poisson=poisson,
        yield_stress=yield_stress,
        hardening_modulus=hardening_modulus,
        tangent_relative_step=tangent_relative_step,
        tangent_evaluation=tangent_evaluation,
    )

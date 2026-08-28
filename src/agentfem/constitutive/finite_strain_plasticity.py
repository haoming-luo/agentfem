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

    The first implementation returns the numerical derivative of the complete
    discrete material update, ``dP/dF``, with the old state held fixed.  This is
    a correctness-first algorithmic tangent for verification and initial
    global integration.  An analytically linearized production tangent is a
    separate performance milestone.
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
                ),
                version="0.1.0",
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
        plastic_jacobian = float(np.linalg.det(plastic_gradient))
        if plastic_jacobian <= 0.0:
            raise ValueError("The committed plastic deformation gradient is inverted.")
        if abs(plastic_jacobian - 1.0) > 1.0e-8:
            raise ValueError(
                "Finite-strain J2 requires an isochoric committed plastic state; "
                f"det(Fp)={plastic_jacobian:.16g}."
            )

        elastic_trial = deformation_gradient @ np.linalg.inv(plastic_gradient)
        left_vectors, stretches, right_vectors_transpose = np.linalg.svd(
            elastic_trial
        )
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
            np.sqrt(1.5 * np.dot(deviatoric_kirchhoff_trial, deviatoric_kirchhoff_trial))
        )
        trial_yield = equivalent_trial - self.current_yield_stress(
            equivalent_plastic_strain
        )

        tolerance = 64.0 * np.finfo(float).eps * max(
            self.young,
            self.yield_stress,
            equivalent_trial,
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
                1.0
                - 3.0
                * self.shear_modulus
                * plastic_increment
                / equivalent_trial,
            )
            deviatoric_kirchhoff = radial_scale * deviatoric_kirchhoff_trial
            logarithmic_strain = (
                volumetric_logarithmic_strain / 3.0
                + deviatoric_kirchhoff / (2.0 * self.shear_modulus)
            )

            elastic_left_stretch = (
                left_vectors
                @ np.diag(np.exp(logarithmic_strain))
                @ left_vectors.T
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
                        [equivalent_plastic_strain + plastic_increment], dtype=float
                    ),
                )
            )

        principal_kirchhoff = (
            self.bulk_modulus * volumetric_logarithmic_strain
            + deviatoric_kirchhoff
        )
        kirchhoff_stress = (
            left_vectors @ np.diag(principal_kirchhoff) @ left_vectors.T
        )
        kirchhoff_stress = 0.5 * (kirchhoff_stress + kirchhoff_stress.T)
        cauchy_stress = kirchhoff_stress / jacobian
        first_piola_stress = kirchhoff_stress @ np.linalg.inv(
            deformation_gradient
        ).T
        elastic_energy = (
            self.shear_modulus
            * float(np.dot(logarithmic_strain - np.mean(logarithmic_strain),
                           logarithmic_strain - np.mean(logarithmic_strain)))
            + 0.5
            * self.bulk_modulus
            * volumetric_logarithmic_strain**2
        )
        hardening_energy = 0.5 * self.hardening_modulus * (
            equivalent_plastic_strain + plastic_increment
        ) ** 2
        return _FiniteStrainJ2Integration(
            cauchy_stress=cauchy_stress,
            first_piola_stress=first_piola_stress,
            state=state_new,
            strain_energy_density=elastic_energy + hardening_energy,
            elastic_energy_density=elastic_energy,
            hardening_energy_density=hardening_energy,
            trial_yield_function=trial_yield,
            plastic_multiplier_increment=plastic_increment,
        )

    def _algorithmic_tangent(self, deformation_gradient, state_old) -> np.ndarray:
        selected = np.asarray(deformation_gradient, dtype=float)
        baseline = self._integrate(selected, state_old).first_piola_stress
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
                derivative = (plus_piola - baseline) / increment
            tangent[:, column] = derivative.reshape(-1)
        return tangent

    def update(self, point: MaterialPointInput) -> MaterialPointOutput:
        """Advance one point and return Cauchy stress, state and ``dP/dF``."""

        self._validate_point(point)
        integrated = self._integrate(
            point.deformation_gradient_new,
            point.state_old,
        )
        return MaterialPointOutput(
            cauchy_stress=integrated.cauchy_stress,
            consistent_tangent=self._algorithmic_tangent(
                point.deformation_gradient_new,
                point.state_old,
            ),
            state_new=integrated.state,
            strain_energy_density=integrated.strain_energy_density,
            stored_energy_density_components={
                "ELENER": integrated.elastic_energy_density,
                "HARDENER": integrated.hardening_energy_density,
            },
            tangent_convention=self.tangent_convention,
            state_schema=self.state_schema,
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
                "plastic_dissipation": "not_implemented",
            },
            "tangent": self.tangent_convention.summary(),
            "tangent_evaluation": "central_difference_of_discrete_return",
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
) -> FiniteStrainJ2Logarithmic:
    """Create the logarithmic finite-strain J2 material provider."""

    return FiniteStrainJ2Logarithmic(
        young=young,
        poisson=poisson,
        yield_stress=yield_stress,
        hardening_modulus=hardening_modulus,
        tangent_relative_step=tangent_relative_step,
    )

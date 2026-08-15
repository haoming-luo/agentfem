"""Small-strain J2 plasticity material-point integration.

The local radial-return mapping and analytical algorithmic tangent are kept
independent of the global DOLFINx driver. Integration-point storage lives in
``constitutive.quadrature`` and the global equilibrium path in
``mechanics.plasticity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import ClassVar

import numpy as np


def _symmetric_tensor(value, *, label: str) -> np.ndarray:
    tensor = np.asarray(value, dtype=float)
    if tensor.shape != (3, 3):
        raise ValueError(f"{label} must be a 3x3 tensor.")
    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"{label} must contain finite values.")
    if not np.allclose(tensor, tensor.T, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{label} must be symmetric.")
    return tensor.copy()


def deviatoric(tensor) -> np.ndarray:
    """Return the three-dimensional deviatoric part of a tensor."""

    selected = _symmetric_tensor(tensor, label="tensor")
    return selected - np.trace(selected) / 3.0 * np.eye(3)


def von_mises(stress) -> float:
    """Return ``sqrt(3/2 s:s)`` for a symmetric Cauchy stress."""

    s = deviatoric(stress)
    return float(sqrt(1.5 * np.tensordot(s, s)))


@dataclass(frozen=True)
class J2PlasticState:
    """History variables for small-strain isotropic J2 plasticity."""

    plastic_strain: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    equivalent_plastic_strain: float = 0.0

    def __post_init__(self) -> None:
        plastic_strain = _symmetric_tensor(
            self.plastic_strain,
            label="J2PlasticState.plastic_strain",
        )
        if abs(float(np.trace(plastic_strain))) > 1.0e-10:
            raise ValueError("J2 plastic strain must be deviatoric.")
        equivalent = float(self.equivalent_plastic_strain)
        if not isfinite(equivalent) or equivalent < 0.0:
            raise ValueError("equivalent_plastic_strain must be finite and nonnegative.")
        object.__setattr__(self, "plastic_strain", plastic_strain)
        object.__setattr__(self, "equivalent_plastic_strain", equivalent)


@dataclass(frozen=True)
class J2Update:
    """Result of one radial-return material-point update."""

    stress: np.ndarray
    state: J2PlasticState
    elastic: bool
    yield_function_trial: float
    plastic_multiplier_increment: float
    algorithmic_tangent: np.ndarray


@dataclass(frozen=True)
class J2LinearIsotropicHardening:
    """Rate-independent von Mises plasticity with linear isotropic hardening."""

    young: float
    poisson: float
    yield_stress: float
    hardening_modulus: float = 0.0
    name: str = "J2 linear isotropic hardening"
    stateful_constitutive: ClassVar[bool] = True

    def __post_init__(self) -> None:
        values = {
            "young": self.young,
            "poisson": self.poisson,
            "yield_stress": self.yield_stress,
            "hardening_modulus": self.hardening_modulus,
        }
        if not all(isfinite(float(value)) for value in values.values()):
            raise ValueError("J2 material parameters must be finite.")
        if self.young <= 0.0:
            raise ValueError("J2LinearIsotropicHardening.young must be positive.")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError(
                "J2LinearIsotropicHardening.poisson must satisfy -1 < nu < 0.5."
            )
        if self.yield_stress <= 0.0:
            raise ValueError("yield_stress must be positive.")
        if self.hardening_modulus < 0.0:
            raise ValueError("hardening_modulus must be nonnegative.")

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def bulk_modulus(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    def current_yield_stress(self, equivalent_plastic_strain: float) -> float:
        return self.yield_stress + self.hardening_modulus * float(
            equivalent_plastic_strain
        )

    def elastic_tangent(self) -> np.ndarray:
        """Return the symmetric three-dimensional elastic tangent."""

        identity = np.eye(3)
        symmetric_identity = 0.5 * (
            np.einsum("ik,jl->ijkl", identity, identity)
            + np.einsum("il,jk->ijkl", identity, identity)
        )
        deviatoric_identity = symmetric_identity - (
            np.einsum("ij,kl->ijkl", identity, identity) / 3.0
        )
        return (
            self.bulk_modulus * np.einsum("ij,kl->ijkl", identity, identity)
            + 2.0 * self.shear_modulus * deviatoric_identity
        )

    def update(
        self,
        total_strain,
        state: J2PlasticState | None = None,
        *,
        tolerance: float | None = None,
    ) -> J2Update:
        """Integrate one material point by closest-point radial return."""

        strain = _symmetric_tensor(total_strain, label="total_strain")
        old = J2PlasticState() if state is None else state
        elastic_strain_trial = strain - old.plastic_strain
        trial_stress = (
            2.0 * self.shear_modulus * deviatoric(elastic_strain_trial)
            + self.bulk_modulus * np.trace(elastic_strain_trial) * np.eye(3)
        )
        trial_deviator = deviatoric(trial_stress)
        q_trial = von_mises(trial_stress)
        yield_old = self.current_yield_stress(old.equivalent_plastic_strain)
        f_trial = q_trial - yield_old
        selected_tolerance = (
            max(1.0, yield_old) * 1.0e-12
            if tolerance is None
            else float(tolerance)
        )
        if selected_tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative.")
        if f_trial <= selected_tolerance:
            return J2Update(
                stress=trial_stress,
                state=old,
                elastic=True,
                yield_function_trial=float(f_trial),
                plastic_multiplier_increment=0.0,
                algorithmic_tangent=self.elastic_tangent(),
            )
        if q_trial <= 0.0:
            raise RuntimeError("Positive J2 yield function requires q_trial > 0.")
        increment = f_trial / (
            3.0 * self.shear_modulus + self.hardening_modulus
        )
        direction = 1.5 * trial_deviator / q_trial
        plastic_strain = old.plastic_strain + increment * direction
        equivalent = old.equivalent_plastic_strain + increment
        new_deviator = (
            1.0 - 3.0 * self.shear_modulus * increment / q_trial
        ) * trial_deviator
        pressure_part = np.trace(trial_stress) / 3.0 * np.eye(3)
        reduction = 1.0 - 3.0 * self.shear_modulus * increment / q_trial
        identity = np.eye(3)
        symmetric_identity = 0.5 * (
            np.einsum("ik,jl->ijkl", identity, identity)
            + np.einsum("il,jk->ijkl", identity, identity)
        )
        deviatoric_identity = symmetric_identity - (
            np.einsum("ij,kl->ijkl", identity, identity) / 3.0
        )
        flow_direction = 1.5 * trial_deviator / q_trial
        radial_coefficient = (
            1.0
            / (
                q_trial
                * (3.0 * self.shear_modulus + self.hardening_modulus)
            )
            - increment / q_trial**2
        )
        tangent = (
            self.bulk_modulus
            * np.einsum("ij,kl->ijkl", identity, identity)
            + 2.0 * self.shear_modulus * reduction * deviatoric_identity
            - 6.0
            * self.shear_modulus**2
            * radial_coefficient
            * np.einsum("ij,kl->ijkl", trial_deviator, flow_direction)
        )
        return J2Update(
            stress=pressure_part + new_deviator,
            state=J2PlasticState(plastic_strain, equivalent),
            elastic=False,
            yield_function_trial=float(f_trial),
            plastic_multiplier_increment=float(increment),
            algorithmic_tangent=tangent,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "j2_linear_isotropic_hardening",
            "kinematics": "small_strain",
            "young": self.young,
            "poisson": self.poisson,
            "yield_stress": self.yield_stress,
            "hardening_modulus": self.hardening_modulus,
            "maturity": "fem_integrated_3d",
            "fem_quadrature_driver": True,
            "algorithmic_tangent": "analytical_consistent",
        }


@dataclass(frozen=True)
class UniaxialPlasticState:
    """History variables for the exact one-dimensional counterpart."""

    plastic_strain: float = 0.0
    equivalent_plastic_strain: float = 0.0


def update_uniaxial(
    total_strain: float,
    material: J2LinearIsotropicHardening,
    state: UniaxialPlasticState | None = None,
) -> tuple[float, UniaxialPlasticState]:
    """Return stress and state for a one-dimensional bilinear material test."""

    selected = UniaxialPlasticState() if state is None else state
    trial = material.young * (float(total_strain) - selected.plastic_strain)
    yield_value = material.current_yield_stress(
        selected.equivalent_plastic_strain
    )
    function = abs(trial) - yield_value
    if function <= max(1.0, yield_value) * 1.0e-12:
        return float(trial), selected
    direction = 1.0 if trial >= 0.0 else -1.0
    increment = function / (material.young + material.hardening_modulus)
    new_state = UniaxialPlasticState(
        plastic_strain=selected.plastic_strain + direction * increment,
        equivalent_plastic_strain=selected.equivalent_plastic_strain + increment,
    )
    stress = direction * material.current_yield_stress(
        new_state.equivalent_plastic_strain
    )
    return float(stress), new_state

"""Small-strain J2 plasticity material-point integration.

The local radial-return mapping and analytical algorithmic tangent are kept
independent of the global DOLFINx driver. Integration-point storage lives in
``constitutive.quadrature`` and the global equilibrium path in
``mechanics.plasticity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import ClassVar, Iterable

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
    state: J2PlasticState | ChabocheState
    elastic: bool
    yield_function_trial: float
    plastic_multiplier_increment: float
    algorithmic_tangent: np.ndarray


@dataclass(frozen=True)
class ChabocheState:
    """History for small-strain combined isotropic/kinematic hardening."""

    plastic_strain: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    equivalent_plastic_strain: float = 0.0
    backstresses: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 3, 3))
    )

    def __post_init__(self) -> None:
        plastic = _symmetric_tensor(self.plastic_strain, label="plastic_strain")
        if abs(float(np.trace(plastic))) > 1.0e-10:
            raise ValueError("Chaboche plastic strain must be deviatoric.")
        equivalent = float(self.equivalent_plastic_strain)
        if not isfinite(equivalent) or equivalent < 0.0:
            raise ValueError(
                "equivalent_plastic_strain must be finite and nonnegative."
            )
        backstresses = np.asarray(self.backstresses, dtype=float)
        if backstresses.ndim != 3 or backstresses.shape[1:] != (3, 3):
            raise ValueError("backstresses must have shape (components, 3, 3).")
        selected = np.empty_like(backstresses)
        for index, value in enumerate(backstresses):
            selected[index] = deviatoric(
                _symmetric_tensor(value, label=f"backstresses[{index}]")
            )
        object.__setattr__(self, "plastic_strain", plastic)
        object.__setattr__(self, "equivalent_plastic_strain", equivalent)
        object.__setattr__(self, "backstresses", selected)

    @property
    def total_backstress(self) -> np.ndarray:
        return np.sum(self.backstresses, axis=0)


@dataclass(frozen=True)
class ChabocheCombinedHardening:
    """Small-strain J2 plasticity with nonlinear combined hardening.

    The isotropic radius follows ``Q * (1 - exp(-b p))`` and every
    Armstrong--Frederick component follows
    ``alpha_dot = 2/3 C ep_dot - gamma alpha p_dot``.  The local backward-
    Euler return is solved as one safeguarded scalar consistency equation.
    Its tangent is the numerical derivative of the fully discrete return map,
    so global Newton consumes the algorithmic response rather than the elastic
    predictor.
    """

    young: float
    poisson: float
    yield_stress: float
    backstress_moduli: tuple[float, ...]
    dynamic_recovery: tuple[float, ...]
    isotropic_saturation: float = 0.0
    isotropic_rate: float = 0.0
    name: str = "Chaboche combined hardening"
    local_tolerance: float = 1.0e-11
    local_maximum_iterations: int = 80
    stateful_constitutive: ClassVar[bool] = True

    def __post_init__(self) -> None:
        moduli = tuple(float(value) for value in self.backstress_moduli)
        recovery = tuple(float(value) for value in self.dynamic_recovery)
        if not moduli or len(moduli) != len(recovery):
            raise ValueError(
                "Chaboche requires equally sized, nonempty backstress_moduli "
                "and dynamic_recovery sequences."
            )
        values = {
            "young": self.young,
            "poisson": self.poisson,
            "yield_stress": self.yield_stress,
            "isotropic_saturation": self.isotropic_saturation,
            "isotropic_rate": self.isotropic_rate,
            "local_tolerance": self.local_tolerance,
        }
        if not all(isfinite(float(value)) for value in values.values()):
            raise ValueError("Chaboche material parameters must be finite.")
        if self.young <= 0.0 or self.yield_stress <= 0.0:
            raise ValueError("young and yield_stress must be positive.")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must satisfy -1 < nu < 0.5.")
        if any(not isfinite(value) or value <= 0.0 for value in moduli):
            raise ValueError("Every backstress modulus must be positive.")
        if any(not isfinite(value) or value < 0.0 for value in recovery):
            raise ValueError("Every dynamic-recovery coefficient must be nonnegative.")
        if self.isotropic_saturation < 0.0 or self.isotropic_rate < 0.0:
            raise ValueError("Isotropic saturation and rate must be nonnegative.")
        if (self.isotropic_saturation == 0.0) != (self.isotropic_rate == 0.0):
            raise ValueError(
                "isotropic_saturation and isotropic_rate must both be zero "
                "or both be positive."
            )
        if self.local_tolerance <= 0.0 or int(self.local_maximum_iterations) < 8:
            raise ValueError("Local tolerance must be positive and iterations >= 8.")
        object.__setattr__(self, "backstress_moduli", moduli)
        object.__setattr__(self, "dynamic_recovery", recovery)
        object.__setattr__(self, "local_maximum_iterations", int(self.local_maximum_iterations))

    @property
    def backstress_count(self) -> int:
        return len(self.backstress_moduli)

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def bulk_modulus(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    def current_yield_stress(self, equivalent_plastic_strain: float) -> float:
        equivalent = float(equivalent_plastic_strain)
        return float(
            self.yield_stress
            + self.isotropic_saturation
            * (1.0 - np.exp(-self.isotropic_rate * equivalent))
        )

    def elastic_tangent(self) -> np.ndarray:
        return _isotropic_elastic_tangent(self.bulk_modulus, self.shear_modulus)

    def initial_state(self) -> ChabocheState:
        return ChabocheState(
            backstresses=np.zeros((self.backstress_count, 3, 3))
        )

    def update(
        self,
        total_strain,
        state: ChabocheState | None = None,
        *,
        tolerance: float | None = None,
    ) -> J2Update:
        """Integrate one point and return its fully discrete tangent."""

        strain = _symmetric_tensor(total_strain, label="total_strain")
        old = self.initial_state() if state is None else state
        if not isinstance(old, ChabocheState):
            raise TypeError("ChabocheCombinedHardening requires ChabocheState.")
        if len(old.backstresses) != self.backstress_count:
            raise ValueError("State and material backstress counts differ.")
        stress, new_state, elastic, trial_value, increment = self._integrate(
            strain,
            old,
            tolerance=tolerance,
        )
        tangent = (
            self.elastic_tangent()
            if elastic
            else self._algorithmic_tangent(strain, old)
        )
        return J2Update(
            stress=stress,
            state=new_state,
            elastic=elastic,
            yield_function_trial=trial_value,
            plastic_multiplier_increment=increment,
            algorithmic_tangent=tangent,
        )

    def _integrate(self, strain, old, *, tolerance=None):
        elastic_trial = strain - old.plastic_strain
        trial_stress = (
            2.0 * self.shear_modulus * deviatoric(elastic_trial)
            + self.bulk_modulus * np.trace(elastic_trial) * np.eye(3)
        )
        trial_deviator = deviatoric(trial_stress)
        shifted_trial = trial_deviator - old.total_backstress
        q_trial = von_mises(shifted_trial)
        yield_old = self.current_yield_stress(old.equivalent_plastic_strain)
        trial_value = q_trial - yield_old
        selected_tolerance = (
            max(1.0, yield_old) * 1.0e-12
            if tolerance is None
            else float(tolerance)
        )
        if selected_tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative.")
        if trial_value <= selected_tolerance:
            return trial_stress, old, True, float(trial_value), 0.0

        def consistency(increment: float):
            theta = 1.0 / (
                1.0 + np.asarray(self.dynamic_recovery) * increment
            )
            base = trial_deviator - np.einsum(
                "a,aij->ij", theta, old.backstresses
            )
            q_base = von_mises(base)
            value = (
                q_base
                - 3.0 * self.shear_modulus * increment
                - increment
                * float(np.dot(theta, np.asarray(self.backstress_moduli)))
                - self.current_yield_stress(
                    old.equivalent_plastic_strain + increment
                )
            )
            return float(value), theta, base, q_base

        lower = 0.0
        upper = max(
            trial_value
            / (
                3.0 * self.shear_modulus
                + sum(self.backstress_moduli)
                + self.isotropic_saturation * self.isotropic_rate
            ),
            np.finfo(float).eps,
        )
        upper_value = consistency(upper)[0]
        for _ in range(self.local_maximum_iterations):
            if upper_value <= 0.0:
                break
            upper *= 2.0
            upper_value = consistency(upper)[0]
        else:
            raise RuntimeError("Could not bracket the Chaboche consistency root.")

        scale = max(1.0, yield_old, q_trial)
        increment = upper
        for _ in range(self.local_maximum_iterations):
            increment = 0.5 * (lower + upper)
            value = consistency(increment)[0]
            if abs(value) <= self.local_tolerance * scale:
                break
            if value > 0.0:
                lower = increment
            else:
                upper = increment
        else:
            raise RuntimeError("Chaboche local return did not converge.")

        _, theta, base, q_base = consistency(increment)
        if q_base <= 0.0:
            raise RuntimeError("Plastic Chaboche return requires a positive direction norm.")
        direction = 1.5 * base / q_base
        plastic = old.plastic_strain + increment * direction
        backstresses = np.empty_like(old.backstresses)
        for index, (modulus, factor) in enumerate(
            zip(self.backstress_moduli, theta, strict=True)
        ):
            backstresses[index] = factor * (
                old.backstresses[index]
                + (2.0 / 3.0) * modulus * increment * direction
            )
        pressure = np.trace(trial_stress) / 3.0 * np.eye(3)
        stress = pressure + trial_deviator - 2.0 * self.shear_modulus * increment * direction
        return (
            stress,
            ChabocheState(
                plastic_strain=plastic,
                equivalent_plastic_strain=(
                    old.equivalent_plastic_strain + increment
                ),
                backstresses=backstresses,
            ),
            False,
            float(trial_value),
            float(increment),
        )

    def _algorithmic_tangent(self, strain, old) -> np.ndarray:
        """Differentiate the complete discrete return map in symmetric strain."""

        scale = max(
            1.0e-5,
            float(np.linalg.norm(strain)),
            self.yield_stress / self.young,
        )
        step = np.cbrt(np.finfo(float).eps) * scale
        tangent = np.zeros((3, 3, 3, 3), dtype=float)
        for k in range(3):
            for l in range(k, 3):
                direction = np.zeros((3, 3), dtype=float)
                if k == l:
                    direction[k, l] = 1.0
                else:
                    direction[k, l] = direction[l, k] = 0.5
                plus = self._integrate(strain + step * direction, old)[0]
                minus = self._integrate(strain - step * direction, old)[0]
                derivative = (plus - minus) / (2.0 * step)
                tangent[:, :, k, l] = derivative
                tangent[:, :, l, k] = derivative
        return 0.5 * (tangent + np.swapaxes(tangent, 0, 1))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "chaboche_combined_hardening",
            "kinematics": "small_strain",
            "young": float(self.young),
            "poisson": float(self.poisson),
            "yield_stress": float(self.yield_stress),
            "isotropic_saturation": float(self.isotropic_saturation),
            "isotropic_rate": float(self.isotropic_rate),
            "backstress_moduli": list(self.backstress_moduli),
            "dynamic_recovery": list(self.dynamic_recovery),
            "maturity": "fem_integrated_experimental",
            "fem_quadrature_driver": True,
            "local_integration": "backward_euler_scalar_consistency",
            "algorithmic_tangent": "discrete_central_difference",
        }


def _isotropic_elastic_tangent(bulk: float, shear: float) -> np.ndarray:
    identity = np.eye(3)
    symmetric_identity = 0.5 * (
        np.einsum("ik,jl->ijkl", identity, identity)
        + np.einsum("il,jk->ijkl", identity, identity)
    )
    deviatoric_identity = symmetric_identity - (
        np.einsum("ij,kl->ijkl", identity, identity) / 3.0
    )
    return (
        bulk * np.einsum("ij,kl->ijkl", identity, identity)
        + 2.0 * shear * deviatoric_identity
    )


def chaboche(
    *,
    young: float,
    poisson: float,
    yield_stress: float,
    backstresses: Iterable[tuple[float, float]],
    isotropic_saturation: float = 0.0,
    isotropic_rate: float = 0.0,
    name: str = "Chaboche combined hardening",
) -> ChabocheCombinedHardening:
    """Create a combined-hardening material from ``(C, gamma)`` pairs."""

    selected = tuple((float(c), float(gamma)) for c, gamma in backstresses)
    return ChabocheCombinedHardening(
        young=young,
        poisson=poisson,
        yield_stress=yield_stress,
        backstress_moduli=tuple(item[0] for item in selected),
        dynamic_recovery=tuple(item[1] for item in selected),
        isotropic_saturation=isotropic_saturation,
        isotropic_rate=isotropic_rate,
        name=name,
    )


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

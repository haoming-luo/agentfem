# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Small-strain J2 plasticity material-point integration.

The local radial-return mapping and analytical algorithmic tangent are kept
independent of the global DOLFINx driver. Integration-point storage lives in
``constitutive.quadrature`` and the global equilibrium path in
``mechanics.plasticity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import ClassVar, Iterable, Literal

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
            raise ValueError(
                "equivalent_plastic_strain must be finite and nonnegative."
            )
        object.__setattr__(self, "plastic_strain", plastic_strain)
        object.__setattr__(self, "equivalent_plastic_strain", equivalent)


@dataclass(frozen=True)
class PlasticEnergyIncrement:
    """One accepted plastic substep written as an explicit energy ledger."""

    plastic_work: float = 0.0
    isotropic_stored_energy_change: float = 0.0
    kinematic_stored_energy_change: float = 0.0
    reference_yield_dissipation: float = 0.0
    dynamic_recovery_dissipation: float = 0.0
    backward_euler_dissipation: float = 0.0
    balance_residual: float = 0.0

    @property
    def recoverable_storage_change(self) -> float:
        return self.isotropic_stored_energy_change + self.kinematic_stored_energy_change

    @property
    def modeled_irreversible_dissipation(self) -> float:
        return self.reference_yield_dissipation + self.dynamic_recovery_dissipation

    @property
    def discrete_dissipation(self) -> float:
        return self.modeled_irreversible_dissipation + self.backward_euler_dissipation

    def as_dict(self) -> dict[str, float]:
        return {
            "plastic_work": self.plastic_work,
            "isotropic_stored_energy_change": (self.isotropic_stored_energy_change),
            "kinematic_stored_energy_change": (self.kinematic_stored_energy_change),
            "recoverable_storage_change": self.recoverable_storage_change,
            "reference_yield_dissipation": self.reference_yield_dissipation,
            "dynamic_recovery_dissipation": (self.dynamic_recovery_dissipation),
            "modeled_irreversible_dissipation": (self.modeled_irreversible_dissipation),
            "backward_euler_dissipation": self.backward_euler_dissipation,
            "discrete_dissipation": self.discrete_dissipation,
            "balance_residual": self.balance_residual,
        }


@dataclass(frozen=True)
class J2Update:
    """Result of one radial-return material-point update.

    ``algorithmic_tangent`` is absent only when the caller explicitly requests
    a response-only update.  Global Newton providers always request the
    consistent linearization.
    """

    stress: np.ndarray
    state: J2PlasticState | ChabocheState
    elastic: bool
    yield_function_trial: float
    plastic_multiplier_increment: float
    algorithmic_tangent: np.ndarray | None
    energy_increment: PlasticEnergyIncrement = PlasticEnergyIncrement()


@dataclass(frozen=True)
class _ChabocheBatchUpdate:
    """Vectorized internal response for one homogeneous quadrature batch."""

    stress: np.ndarray
    plastic_strain: np.ndarray
    equivalent_plastic_strain: np.ndarray
    backstresses: np.ndarray
    elastic: np.ndarray
    plastic_multiplier_increment: np.ndarray
    algorithmic_tangent: np.ndarray
    dynamic_recovery_dissipation: np.ndarray
    backward_euler_dissipation: np.ndarray


Linearization = Literal["none", "consistent"]


def _linearization(value: str) -> Linearization:
    selected = str(value).strip().lower()
    if selected not in {"none", "consistent"}:
        raise ValueError("linearization must be 'none' or 'consistent'.")
    return selected


@dataclass(frozen=True)
class TabulatedIsotropicHardening:
    """Piecewise-linear yield radius as a function of equivalent plastic strain.

    The table is an explicit scientific asset rather than an interpolation
    hidden inside a material driver.  Constant extrapolation matches the
    default used by Abaqus ``*CYCLIC HARDENING`` tables.
    """

    equivalent_plastic_strain: tuple[float, ...]
    yield_stress: tuple[float, ...]
    extrapolation: Literal["constant", "linear"] = "constant"

    def __post_init__(self) -> None:
        equivalent = tuple(float(value) for value in self.equivalent_plastic_strain)
        stress = tuple(float(value) for value in self.yield_stress)
        if len(equivalent) < 2 or len(equivalent) != len(stress):
            raise ValueError(
                "Tabulated isotropic hardening requires equally sized tables "
                "with at least two points."
            )
        if not all(isfinite(value) for value in (*equivalent, *stress)):
            raise ValueError("Tabulated isotropic hardening must be finite.")
        if equivalent[0] != 0.0 or any(
            right <= left for left, right in zip(equivalent, equivalent[1:])
        ):
            raise ValueError(
                "Equivalent plastic strain must start at zero and increase strictly."
            )
        if any(value <= 0.0 for value in stress):
            raise ValueError("Every tabulated yield stress must be positive.")
        if any(right < left for left, right in zip(stress, stress[1:])):
            raise ValueError(
                "TabulatedIsotropicHardening currently accepts hardening, not softening."
            )
        extrapolation = str(self.extrapolation).strip().lower()
        if extrapolation not in {"constant", "linear"}:
            raise ValueError("extrapolation must be 'constant' or 'linear'.")
        object.__setattr__(self, "equivalent_plastic_strain", equivalent)
        object.__setattr__(self, "yield_stress", stress)
        object.__setattr__(self, "extrapolation", extrapolation)

    @property
    def initial_yield_stress(self) -> float:
        return self.yield_stress[0]

    @property
    def maximum_slope(self) -> float:
        equivalent = np.asarray(self.equivalent_plastic_strain)
        stress = np.asarray(self.yield_stress)
        return float(np.max(np.diff(stress) / np.diff(equivalent), initial=0.0))

    def value(self, equivalent_plastic_strain: float) -> float:
        equivalent = float(equivalent_plastic_strain)
        if not isfinite(equivalent) or equivalent < 0.0:
            raise ValueError(
                "equivalent_plastic_strain must be finite and nonnegative."
            )
        points = np.asarray(self.equivalent_plastic_strain)
        values = np.asarray(self.yield_stress)
        if equivalent <= points[-1] or self.extrapolation == "constant":
            return float(np.interp(equivalent, points, values))
        slope = (values[-1] - values[-2]) / (points[-1] - points[-2])
        return float(values[-1] + slope * (equivalent - points[-1]))

    def slope(self, equivalent_plastic_strain: float) -> float:
        """Return the active piecewise-linear hardening modulus."""

        equivalent = float(equivalent_plastic_strain)
        if not isfinite(equivalent) or equivalent < 0.0:
            raise ValueError(
                "equivalent_plastic_strain must be finite and nonnegative."
            )
        points = np.asarray(self.equivalent_plastic_strain)
        values = np.asarray(self.yield_stress)
        if equivalent >= points[-1]:
            if self.extrapolation == "constant":
                return 0.0
            index = points.size - 2
        else:
            index = max(
                int(np.searchsorted(points, equivalent, side="right")) - 1,
                0,
            )
        return float(
            (values[index + 1] - values[index]) / (points[index + 1] - points[index])
        )

    def hardening_storage(self, equivalent_plastic_strain: float) -> float:
        """Return the integral of yield-radius growth above its initial value."""

        equivalent = float(equivalent_plastic_strain)
        if not isfinite(equivalent) or equivalent < 0.0:
            raise ValueError(
                "equivalent_plastic_strain must be finite and nonnegative."
            )
        points = np.asarray(self.equivalent_plastic_strain)
        radius = np.asarray(self.yield_stress) - self.initial_yield_stress
        upper = min(equivalent, float(points[-1]))
        storage = 0.0
        for index in range(points.size - 1):
            left = float(points[index])
            if upper <= left:
                break
            right = min(upper, float(points[index + 1]))
            fraction = (right - left) / (points[index + 1] - points[index])
            radius_right = radius[index] + fraction * (
                radius[index + 1] - radius[index]
            )
            storage += 0.5 * (radius[index] + radius_right) * (right - left)
        if equivalent > points[-1]:
            delta = equivalent - points[-1]
            if self.extrapolation == "constant":
                storage += radius[-1] * delta
            else:
                slope = (radius[-1] - radius[-2]) / (points[-1] - points[-2])
                storage += radius[-1] * delta + 0.5 * slope * delta**2
        return float(storage)

    def as_dict(self) -> dict[str, object]:
        return {
            "model": "tabulated_isotropic_hardening",
            "equivalent_plastic_strain": list(self.equivalent_plastic_strain),
            "yield_stress": list(self.yield_stress),
            "extrapolation": self.extrapolation,
        }


@dataclass(frozen=True)
class ChabocheState:
    """History for small-strain combined isotropic/kinematic hardening."""

    plastic_strain: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    equivalent_plastic_strain: float = 0.0
    backstresses: np.ndarray = field(default_factory=lambda: np.zeros((1, 3, 3)))

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
    isotropic_hardening: TabulatedIsotropicHardening | None = None
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
        if self.isotropic_hardening is not None:
            if not isinstance(
                self.isotropic_hardening,
                TabulatedIsotropicHardening,
            ):
                raise TypeError(
                    "isotropic_hardening must be TabulatedIsotropicHardening."
                )
            if self.isotropic_saturation != 0.0 or self.isotropic_rate != 0.0:
                raise ValueError(
                    "Choose either exponential or tabulated isotropic hardening."
                )
            if not np.isclose(
                self.yield_stress,
                self.isotropic_hardening.initial_yield_stress,
                rtol=1.0e-12,
                atol=0.0,
            ):
                raise ValueError(
                    "yield_stress must equal the first tabulated yield stress."
                )
        if self.local_tolerance <= 0.0 or int(self.local_maximum_iterations) < 8:
            raise ValueError("Local tolerance must be positive and iterations >= 8.")
        object.__setattr__(self, "backstress_moduli", moduli)
        object.__setattr__(self, "dynamic_recovery", recovery)
        object.__setattr__(
            self, "local_maximum_iterations", int(self.local_maximum_iterations)
        )

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
        if self.isotropic_hardening is not None:
            return self.isotropic_hardening.value(equivalent)
        return float(
            self.yield_stress
            + self.isotropic_saturation
            * (1.0 - np.exp(-self.isotropic_rate * equivalent))
        )

    def elastic_tangent(self) -> np.ndarray:
        return _isotropic_elastic_tangent(self.bulk_modulus, self.shear_modulus)

    def initial_state(self) -> ChabocheState:
        return ChabocheState(backstresses=np.zeros((self.backstress_count, 3, 3)))

    def update(
        self,
        total_strain,
        state: ChabocheState | None = None,
        *,
        tolerance: float | None = None,
        linearization: Linearization = "consistent",
    ) -> J2Update:
        """Integrate one point and optionally return its discrete tangent."""

        strain = _symmetric_tensor(total_strain, label="total_strain")
        selected_linearization = _linearization(linearization)
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
        tangent = None
        if selected_linearization == "consistent":
            tangent = (
                self.elastic_tangent()
                if elastic
                else self._algorithmic_tangent(strain, old, increment)
            )
        return J2Update(
            stress=stress,
            state=new_state,
            elastic=elastic,
            yield_function_trial=trial_value,
            plastic_multiplier_increment=increment,
            algorithmic_tangent=tangent,
            energy_increment=_plastic_energy_increment(
                self,
                old,
                new_state,
                stress,
            ),
        )

    def _update_batch(
        self,
        total_strain,
        plastic_strain,
        equivalent_plastic_strain,
        backstresses,
    ) -> _ChabocheBatchUpdate:
        """Integrate a homogeneous batch without changing the material law.

        This is an internal execution kernel for quadrature-point state.  It
        vectorizes the same safeguarded backward-Euler consistency solve and
        analytical tangent used by :meth:`update`; heterogeneous regional
        material maps deliberately retain explicit material dispatch.
        """

        strains = np.asarray(total_strain, dtype=float)
        old_plastic = np.asarray(plastic_strain, dtype=float)
        old_equivalent = np.asarray(equivalent_plastic_strain, dtype=float)
        old_backstresses = np.asarray(backstresses, dtype=float)
        count = len(strains)
        if strains.shape != (count, 3, 3):
            raise ValueError("total_strain batch must have shape (points, 3, 3).")
        if old_plastic.shape != strains.shape:
            raise ValueError("plastic_strain batch must match total_strain.")
        if old_equivalent.shape != (count,):
            raise ValueError("equivalent_plastic_strain must have shape (points,).")
        if old_backstresses.shape != (count, self.backstress_count, 3, 3):
            raise ValueError(
                "backstresses batch must have shape (points, components, 3, 3)."
            )
        if not all(
            np.all(np.isfinite(value))
            for value in (strains, old_plastic, old_equivalent, old_backstresses)
        ):
            raise ValueError("Chaboche batch inputs must contain only finite values.")
        if np.any(old_equivalent < 0.0):
            raise ValueError("Equivalent plastic strain must be nonnegative.")

        identity = np.eye(3)

        def deviatoric_many(value):
            trace = np.trace(value, axis1=-2, axis2=-1)
            return value - trace[..., None, None] * identity / 3.0

        def mises_many(value):
            selected = deviatoric_many(value)
            return np.sqrt(1.5 * np.sum(selected * selected, axis=(-2, -1)))

        def yield_many(equivalent):
            selected = np.asarray(equivalent, dtype=float)
            if self.isotropic_hardening is None:
                return self.yield_stress + self.isotropic_saturation * (
                    1.0 - np.exp(-self.isotropic_rate * selected)
                )
            return np.fromiter(
                (self.isotropic_hardening.value(value) for value in selected),
                dtype=float,
                count=len(selected),
            )

        def isotropic_slope_many(equivalent):
            selected = np.asarray(equivalent, dtype=float)
            if self.isotropic_hardening is None:
                return (
                    self.isotropic_saturation
                    * self.isotropic_rate
                    * np.exp(-self.isotropic_rate * selected)
                )
            return np.fromiter(
                (self.isotropic_hardening.slope(value) for value in selected),
                dtype=float,
                count=len(selected),
            )

        def isotropic_storage_many(equivalent):
            selected = np.asarray(equivalent, dtype=float)
            if self.isotropic_hardening is None:
                if self.isotropic_rate == 0.0:
                    return np.zeros_like(selected)
                return self.isotropic_saturation * (
                    selected
                    + (np.exp(-self.isotropic_rate * selected) - 1.0)
                    / self.isotropic_rate
                )
            return np.fromiter(
                (
                    self.isotropic_hardening.hardening_storage(value)
                    for value in selected
                ),
                dtype=float,
                count=len(selected),
            )

        elastic_trial = strains - old_plastic
        trial_deviator = 2.0 * self.shear_modulus * deviatoric_many(elastic_trial)
        pressure = (
            self.bulk_modulus
            * np.trace(elastic_trial, axis1=-2, axis2=-1)[:, None, None]
            * identity
        )
        trial_stress = pressure + trial_deviator
        shifted_trial = trial_deviator - np.sum(old_backstresses, axis=1)
        q_trial = mises_many(shifted_trial)
        yield_old = yield_many(old_equivalent)
        trial_value = q_trial - yield_old
        tolerance = np.maximum(1.0, yield_old) * 1.0e-12
        plastic_mask = trial_value > tolerance

        stresses = trial_stress.copy()
        new_plastic = old_plastic.copy()
        new_equivalent = old_equivalent.copy()
        new_backstresses = old_backstresses.copy()
        increments = np.zeros(count, dtype=float)
        tangents = np.broadcast_to(
            self.elastic_tangent(),
            (count, 3, 3, 3, 3),
        ).copy()
        dynamic_dissipation = np.zeros(count, dtype=float)
        backward_euler_dissipation = np.zeros(count, dtype=float)
        if not np.any(plastic_mask):
            return _ChabocheBatchUpdate(
                stress=stresses,
                plastic_strain=new_plastic,
                equivalent_plastic_strain=new_equivalent,
                backstresses=new_backstresses,
                elastic=~plastic_mask,
                plastic_multiplier_increment=increments,
                algorithmic_tangent=tangents,
                dynamic_recovery_dissipation=dynamic_dissipation,
                backward_euler_dissipation=backward_euler_dissipation,
            )

        selected = np.flatnonzero(plastic_mask)
        trial_dev = trial_deviator[selected]
        old_bs = old_backstresses[selected]
        old_peeq = old_equivalent[selected]
        trial_f = trial_value[selected]
        yield_radius = yield_old[selected]
        recovery = np.asarray(self.dynamic_recovery, dtype=float)
        moduli = np.asarray(self.backstress_moduli, dtype=float)

        def consistency(delta, *, with_modulus=False):
            theta = 1.0 / (1.0 + delta[:, None] * recovery[None, :])
            base = trial_dev - np.einsum("ma,maij->mij", theta, old_bs)
            q_base = mises_many(base)
            value = (
                q_base
                - 3.0 * self.shear_modulus * delta
                - delta * np.einsum("ma,a->m", theta, moduli)
                - yield_many(old_peeq + delta)
            )
            if not with_modulus:
                return value, theta, base, q_base, None, None
            direction = np.divide(
                1.5 * base,
                q_base[:, None, None],
                out=np.zeros_like(base),
                where=q_base[:, None, None] > 0.0,
            )
            backstress_direction = np.einsum(
                "ma,maij->mij",
                recovery[None, :] * theta**2,
                old_bs,
            )
            weighted_modulus = np.einsum("ma,a->m", theta, moduli)
            weighted_modulus_derivative = -np.einsum(
                "ma,a->m",
                recovery[None, :] * theta**2,
                moduli,
            )
            consistency_modulus = (
                3.0 * self.shear_modulus
                + weighted_modulus
                + delta * weighted_modulus_derivative
                + isotropic_slope_many(old_peeq + delta)
                - np.sum(direction * backstress_direction, axis=(-2, -1))
            )
            return (
                value,
                theta,
                base,
                q_base,
                backstress_direction,
                consistency_modulus,
            )

        maximum_isotropic_slope = (
            self.isotropic_saturation * self.isotropic_rate
            if self.isotropic_hardening is None
            else self.isotropic_hardening.maximum_slope
        )
        lower = np.zeros_like(trial_f)
        upper = np.maximum(
            trial_f
            / (
                3.0 * self.shear_modulus
                + np.sum(moduli)
                + maximum_isotropic_slope
            ),
            np.finfo(float).eps,
        )
        bracketed = np.zeros_like(trial_f, dtype=bool)
        for _ in range(self.local_maximum_iterations):
            upper_value = consistency(upper)[0]
            bracketed |= upper_value <= 0.0
            if np.all(bracketed):
                break
            upper[~bracketed] *= 2.0
        else:
            raise RuntimeError("Could not bracket the Chaboche batch consistency root.")

        scale = np.maximum.reduce((np.ones_like(yield_radius), yield_radius, q_trial[selected]))
        # The analytical tangent is more sensitive to the local root than the
        # stress update itself.  Resolve the batched root two decimal orders
        # beyond the public material tolerance, with a floating-point floor,
        # so the global Newton path retains the scalar algorithm's robustness.
        root_tolerance = np.maximum(
            32.0 * np.finfo(float).eps,
            0.01 * self.local_tolerance,
        ) * scale
        converged = np.zeros_like(trial_f, dtype=bool)
        delta = upper.copy()
        for _ in range(self.local_maximum_iterations):
            value, _, _, _, _, modulus = consistency(delta, with_modulus=True)
            newly_converged = (~converged) & (
                np.abs(value) <= root_tolerance
            )
            active = ~converged & ~newly_converged
            if not np.any(active):
                converged |= newly_converged
                break
            positive = active & (value > 0.0)
            negative = active & ~positive
            lower[positive] = delta[positive]
            upper[negative] = delta[negative]
            converged |= newly_converged
            candidate = delta + np.divide(
                value,
                modulus,
                out=np.full_like(value, np.nan),
                where=np.isfinite(modulus) & (modulus > 0.0),
            )
            safeguarded = (
                np.isfinite(candidate)
                & (candidate > lower)
                & (candidate < upper)
            )
            midpoint = 0.5 * (lower + upper)
            delta[active] = np.where(
                safeguarded[active],
                candidate[active],
                midpoint[active],
            )
        else:
            raise RuntimeError("Chaboche batch local return did not converge.")

        (
            _,
            theta,
            base,
            q_base,
            backstress_direction,
            denominator,
        ) = consistency(delta, with_modulus=True)
        if np.any(q_base <= 0.0):
            raise RuntimeError(
                "Plastic Chaboche batch return requires positive direction norms."
            )
        direction = 1.5 * base / q_base[:, None, None]
        plastic = old_plastic[selected] + delta[:, None, None] * direction
        backstress = theta[:, :, None, None] * (
            old_bs
            + (2.0 / 3.0)
            * moduli[None, :, None, None]
            * delta[:, None, None, None]
            * direction[:, None, :, :]
        )
        stress = (
            pressure[selected]
            + trial_dev
            - 2.0 * self.shear_modulus * delta[:, None, None] * direction
        )
        equivalent = old_peeq + delta

        if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
            raise RuntimeError(
                "Chaboche batch tangent has a nonpositive consistency modulus."
            )
        batch_tangent = np.zeros((len(selected), 3, 3, 3, 3), dtype=float)
        for k in range(3):
            for l in range(k, 3):
                perturbation = np.zeros((3, 3), dtype=float)
                if k == l:
                    perturbation[k, l] = 1.0
                else:
                    perturbation[k, l] = perturbation[l, k] = 0.5
                trial_increment = 2.0 * self.shear_modulus * deviatoric_many(
                    perturbation[None, :, :]
                )[0]
                plastic_increment = (
                    np.sum(direction * trial_increment, axis=(-2, -1)) / denominator
                )
                base_increment = (
                    trial_increment[None, :, :]
                    + backstress_direction * plastic_increment[:, None, None]
                )
                direction_increment = (
                    1.5 * base_increment
                    - direction
                    * np.sum(direction * base_increment, axis=(-2, -1))[:, None, None]
                ) / q_base[:, None, None]
                derivative = (
                    self.bulk_modulus * np.trace(perturbation) * identity
                    + trial_increment
                    - 2.0
                    * self.shear_modulus
                    * (
                        direction * plastic_increment[:, None, None]
                        + delta[:, None, None] * direction_increment
                    )
                )
                batch_tangent[:, :, :, k, l] = derivative
                batch_tangent[:, :, :, l, k] = derivative
        batch_tangent = 0.5 * (
            batch_tangent + np.swapaxes(batch_tangent, 1, 2)
        )

        old_kinematic = np.sum(
            3.0
            * np.sum(old_bs * old_bs, axis=(-2, -1))
            / (4.0 * moduli[None, :]),
            axis=1,
        )
        new_kinematic = np.sum(
            3.0
            * np.sum(backstress * backstress, axis=(-2, -1))
            / (4.0 * moduli[None, :]),
            axis=1,
        )
        dynamic = np.sum(
            3.0
            * recovery[None, :]
            * np.sum(backstress * backstress, axis=(-2, -1))
            * delta[:, None]
            / (2.0 * moduli[None, :]),
            axis=1,
        )
        delta_backstress = backstress - old_bs
        backward = np.sum(
            3.0
            * np.sum(delta_backstress * delta_backstress, axis=(-2, -1))
            / (4.0 * moduli[None, :]),
            axis=1,
        )
        isotropic_change = isotropic_storage_many(equivalent) - isotropic_storage_many(
            old_peeq
        )
        radius_new = yield_many(equivalent) - self.yield_stress
        backward += radius_new * delta - isotropic_change

        stresses[selected] = stress
        new_plastic[selected] = plastic
        new_equivalent[selected] = equivalent
        new_backstresses[selected] = backstress
        increments[selected] = delta
        tangents[selected] = batch_tangent
        dynamic_dissipation[selected] = dynamic
        backward_euler_dissipation[selected] = backward
        return _ChabocheBatchUpdate(
            stress=stresses,
            plastic_strain=new_plastic,
            equivalent_plastic_strain=new_equivalent,
            backstresses=new_backstresses,
            elastic=~plastic_mask,
            plastic_multiplier_increment=increments,
            algorithmic_tangent=tangents,
            dynamic_recovery_dissipation=dynamic_dissipation,
            backward_euler_dissipation=backward_euler_dissipation,
        )

    def _integrate(self, strain, old, *, tolerance=None):
        elastic_trial = strain - old.plastic_strain
        trial_stress = 2.0 * self.shear_modulus * deviatoric(
            elastic_trial
        ) + self.bulk_modulus * np.trace(elastic_trial) * np.eye(3)
        trial_deviator = deviatoric(trial_stress)
        shifted_trial = trial_deviator - old.total_backstress
        q_trial = von_mises(shifted_trial)
        yield_old = self.current_yield_stress(old.equivalent_plastic_strain)
        trial_value = q_trial - yield_old
        selected_tolerance = (
            max(1.0, yield_old) * 1.0e-12 if tolerance is None else float(tolerance)
        )
        if selected_tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative.")
        if trial_value <= selected_tolerance:
            return trial_stress, old, True, float(trial_value), 0.0

        def consistency(increment: float):
            theta = 1.0 / (1.0 + np.asarray(self.dynamic_recovery) * increment)
            base = trial_deviator - np.einsum("a,aij->ij", theta, old.backstresses)
            q_base = von_mises(base)
            value = (
                q_base
                - 3.0 * self.shear_modulus * increment
                - increment * float(np.dot(theta, np.asarray(self.backstress_moduli)))
                - self.current_yield_stress(old.equivalent_plastic_strain + increment)
            )
            return float(value), theta, base, q_base

        lower = 0.0
        upper = max(
            trial_value
            / (
                3.0 * self.shear_modulus
                + sum(self.backstress_moduli)
                + self.isotropic_saturation * self.isotropic_rate
                + (
                    0.0
                    if self.isotropic_hardening is None
                    else self.isotropic_hardening.maximum_slope
                )
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
            raise RuntimeError(
                "Plastic Chaboche return requires a positive direction norm."
            )
        direction = 1.5 * base / q_base
        plastic = old.plastic_strain + increment * direction
        backstresses = np.empty_like(old.backstresses)
        for index, (modulus, factor) in enumerate(
            zip(self.backstress_moduli, theta, strict=True)
        ):
            backstresses[index] = factor * (
                old.backstresses[index] + (2.0 / 3.0) * modulus * increment * direction
            )
        pressure = np.trace(trial_stress) / 3.0 * np.eye(3)
        stress = (
            pressure + trial_deviator - 2.0 * self.shear_modulus * increment * direction
        )
        return (
            stress,
            ChabocheState(
                plastic_strain=plastic,
                equivalent_plastic_strain=(old.equivalent_plastic_strain + increment),
                backstresses=backstresses,
            ),
            False,
            float(trial_value),
            float(increment),
        )

    def _algorithmic_tangent(self, strain, old, increment) -> np.ndarray:
        """Linearize the fully discrete backward-Euler return map.

        The local solve has one scalar unknown, the equivalent-plastic-strain
        increment. Differentiating that same consistency equation gives the
        algorithmic tangent consumed by global Newton without repeating the
        return map for twelve finite-difference perturbations.
        """

        elastic_trial = strain - old.plastic_strain
        trial_stress = 2.0 * self.shear_modulus * deviatoric(
            elastic_trial
        ) + self.bulk_modulus * np.trace(elastic_trial) * np.eye(3)
        trial_deviator = deviatoric(trial_stress)
        recovery = np.asarray(self.dynamic_recovery)
        moduli = np.asarray(self.backstress_moduli)
        theta = 1.0 / (1.0 + recovery * increment)
        base = trial_deviator - np.einsum(
            "a,aij->ij",
            theta,
            old.backstresses,
        )
        q_base = von_mises(base)
        if q_base <= 0.0:
            raise RuntimeError(
                "Plastic Chaboche tangent requires a positive return direction."
            )
        direction = 1.5 * base / q_base
        backstress_direction = np.einsum(
            "a,aij->ij",
            recovery * theta**2,
            old.backstresses,
        )
        weighted_modulus = float(np.dot(theta, moduli))
        weighted_modulus_derivative = -float(np.dot(recovery * theta**2, moduli))
        equivalent = old.equivalent_plastic_strain + increment
        if self.isotropic_hardening is None:
            isotropic_modulus = (
                self.isotropic_saturation
                * self.isotropic_rate
                * np.exp(-self.isotropic_rate * equivalent)
            )
        else:
            isotropic_modulus = self.isotropic_hardening.slope(equivalent)
        denominator = (
            3.0 * self.shear_modulus
            + weighted_modulus
            + increment * weighted_modulus_derivative
            + isotropic_modulus
            - float(np.sum(direction * backstress_direction))
        )
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError(
                "Chaboche discrete tangent has a nonpositive consistency modulus."
            )

        tangent = np.zeros((3, 3, 3, 3), dtype=float)
        for k in range(3):
            for l in range(k, 3):
                perturbation = np.zeros((3, 3), dtype=float)
                if k == l:
                    perturbation[k, l] = 1.0
                else:
                    perturbation[k, l] = perturbation[l, k] = 0.5
                trial_increment = 2.0 * self.shear_modulus * deviatoric(perturbation)
                plastic_increment = float(
                    np.sum(direction * trial_increment) / denominator
                )
                base_increment = (
                    trial_increment + backstress_direction * plastic_increment
                )
                direction_increment = (
                    1.5 * base_increment
                    - direction * np.sum(direction * base_increment)
                ) / q_base
                derivative = (
                    self.bulk_modulus * np.trace(perturbation) * np.eye(3)
                    + trial_increment
                    - 2.0
                    * self.shear_modulus
                    * (direction * plastic_increment + increment * direction_increment)
                )
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
            "isotropic_hardening": (
                None
                if self.isotropic_hardening is None
                else self.isotropic_hardening.as_dict()
            ),
            "backstress_moduli": list(self.backstress_moduli),
            "dynamic_recovery": list(self.dynamic_recovery),
            "maturity": "fem_integrated_experimental",
            "fem_quadrature_driver": True,
            "local_integration": "backward_euler_scalar_consistency",
            "algorithmic_tangent": "analytical_discrete_consistent",
            "response_only": True,
        }

    def history(
        self,
        path,
        *,
        initial_state: ChabocheState | None = None,
        linearization: Linearization = "none",
        control_tolerance: float = 1.0e-10,
        control_maximum_iterations: int = 30,
        name: str = "chaboche_material_history",
    ):
        """Create an inspectable strain-controlled material-point procedure."""

        from .._material_history import PlasticMaterialHistoryStep

        return PlasticMaterialHistoryStep(
            material=self,
            path=path,
            initial_state=initial_state,
            linearization=linearization,
            control_tolerance=control_tolerance,
            control_maximum_iterations=control_maximum_iterations,
            name=name,
        )


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
    isotropic_hardening: TabulatedIsotropicHardening | None = None,
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
        isotropic_hardening=isotropic_hardening,
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
        linearization: Linearization = "consistent",
    ) -> J2Update:
        """Integrate one material point by closest-point radial return."""

        strain = _symmetric_tensor(total_strain, label="total_strain")
        selected_linearization = _linearization(linearization)
        old = J2PlasticState() if state is None else state
        elastic_strain_trial = strain - old.plastic_strain
        trial_stress = 2.0 * self.shear_modulus * deviatoric(
            elastic_strain_trial
        ) + self.bulk_modulus * np.trace(elastic_strain_trial) * np.eye(3)
        trial_deviator = deviatoric(trial_stress)
        q_trial = von_mises(trial_stress)
        yield_old = self.current_yield_stress(old.equivalent_plastic_strain)
        f_trial = q_trial - yield_old
        selected_tolerance = (
            max(1.0, yield_old) * 1.0e-12 if tolerance is None else float(tolerance)
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
                algorithmic_tangent=(
                    self.elastic_tangent()
                    if selected_linearization == "consistent"
                    else None
                ),
                energy_increment=PlasticEnergyIncrement(),
            )
        if q_trial <= 0.0:
            raise RuntimeError("Positive J2 yield function requires q_trial > 0.")
        increment = f_trial / (3.0 * self.shear_modulus + self.hardening_modulus)
        direction = 1.5 * trial_deviator / q_trial
        plastic_strain = old.plastic_strain + increment * direction
        equivalent = old.equivalent_plastic_strain + increment
        new_deviator = (
            1.0 - 3.0 * self.shear_modulus * increment / q_trial
        ) * trial_deviator
        pressure_part = np.trace(trial_stress) / 3.0 * np.eye(3)
        tangent = None
        if selected_linearization == "consistent":
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
                1.0 / (q_trial * (3.0 * self.shear_modulus + self.hardening_modulus))
                - increment / q_trial**2
            )
            tangent = (
                self.bulk_modulus * np.einsum("ij,kl->ijkl", identity, identity)
                + 2.0 * self.shear_modulus * reduction * deviatoric_identity
                - 6.0
                * self.shear_modulus**2
                * radial_coefficient
                * np.einsum("ij,kl->ijkl", trial_deviator, flow_direction)
            )
        new_stress = pressure_part + new_deviator
        new_state = J2PlasticState(plastic_strain, equivalent)
        return J2Update(
            stress=new_stress,
            state=new_state,
            elastic=False,
            yield_function_trial=float(f_trial),
            plastic_multiplier_increment=float(increment),
            algorithmic_tangent=tangent,
            energy_increment=_plastic_energy_increment(
                self,
                old,
                new_state,
                new_stress,
            ),
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
            "response_only": True,
        }

    def history(
        self,
        path,
        *,
        initial_state: J2PlasticState | None = None,
        linearization: Linearization = "none",
        control_tolerance: float = 1.0e-10,
        control_maximum_iterations: int = 30,
        name: str = "j2_material_history",
    ):
        """Create an inspectable strain-controlled material-point procedure."""

        from .._material_history import PlasticMaterialHistoryStep

        return PlasticMaterialHistoryStep(
            material=self,
            path=path,
            initial_state=initial_state,
            linearization=linearization,
            control_tolerance=control_tolerance,
            control_maximum_iterations=control_maximum_iterations,
            name=name,
        )


def _isotropic_hardening_storage(material, equivalent: float) -> float:
    selected = float(equivalent)
    if isinstance(material, J2LinearIsotropicHardening):
        return 0.5 * material.hardening_modulus * selected**2
    if material.isotropic_hardening is not None:
        return material.isotropic_hardening.hardening_storage(selected)
    if material.isotropic_rate == 0.0:
        return 0.0
    return float(
        material.isotropic_saturation
        * (
            selected
            + (np.exp(-material.isotropic_rate * selected) - 1.0)
            / material.isotropic_rate
        )
    )


def _kinematic_hardening_storage(material, state) -> float:
    if not isinstance(material, ChabocheCombinedHardening):
        return 0.0
    return float(
        sum(
            3.0 * np.tensordot(alpha, alpha) / (4.0 * modulus)
            for alpha, modulus in zip(
                state.backstresses,
                material.backstress_moduli,
                strict=True,
            )
        )
    )


def _plastic_energy_increment(material, old, new, stress) -> PlasticEnergyIncrement:
    """Close the accepted backward-Euler plastic-work identity."""

    increment = float(new.equivalent_plastic_strain - old.equivalent_plastic_strain)
    if increment <= 0.0:
        return PlasticEnergyIncrement()
    plastic_strain_increment = new.plastic_strain - old.plastic_strain
    plastic_work = float(np.tensordot(stress, plastic_strain_increment))
    isotropic_change = _isotropic_hardening_storage(
        material,
        new.equivalent_plastic_strain,
    ) - _isotropic_hardening_storage(
        material,
        old.equivalent_plastic_strain,
    )
    kinematic_change = _kinematic_hardening_storage(
        material,
        new,
    ) - _kinematic_hardening_storage(
        material,
        old,
    )
    reference = float(material.yield_stress * increment)
    dynamic_recovery = 0.0
    backward_euler = 0.0
    if isinstance(material, ChabocheCombinedHardening):
        for alpha_old, alpha_new, modulus, recovery in zip(
            old.backstresses,
            new.backstresses,
            material.backstress_moduli,
            material.dynamic_recovery,
            strict=True,
        ):
            dynamic_recovery += (
                3.0
                * recovery
                * float(np.tensordot(alpha_new, alpha_new))
                * increment
                / (2.0 * modulus)
            )
            delta_alpha = alpha_new - alpha_old
            backward_euler += (
                3.0 * float(np.tensordot(delta_alpha, delta_alpha)) / (4.0 * modulus)
            )
        radius_new = (
            material.current_yield_stress(new.equivalent_plastic_strain)
            - material.yield_stress
        )
        backward_euler += radius_new * increment - isotropic_change
    else:
        radius_new = material.hardening_modulus * new.equivalent_plastic_strain
        backward_euler = radius_new * increment - isotropic_change
    residual = (
        plastic_work
        - isotropic_change
        - kinematic_change
        - reference
        - dynamic_recovery
        - backward_euler
    )
    return PlasticEnergyIncrement(
        plastic_work=plastic_work,
        isotropic_stored_energy_change=float(isotropic_change),
        kinematic_stored_energy_change=float(kinematic_change),
        reference_yield_dissipation=reference,
        dynamic_recovery_dissipation=float(dynamic_recovery),
        backward_euler_dissipation=float(backward_euler),
        balance_residual=float(residual),
    )


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
    yield_value = material.current_yield_stress(selected.equivalent_plastic_strain)
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

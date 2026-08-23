"""Power-law creep material-point relations and closed-form checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sinh
from typing import ClassVar

import numpy as np

from .plasticity import deviatoric, von_mises
from agentfem.materials.properties import ElasticIsotropicProperties


@dataclass(frozen=True)
class CreepHistory:
    """Integrated piecewise-constant stress history."""

    time: np.ndarray
    equivalent_creep_strain: np.ndarray
    creep_strain: np.ndarray | None = None

    @property
    def final_equivalent_strain(self) -> float:
        return float(self.equivalent_creep_strain[-1])

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "creep_history",
            "time": self.time.tolist(),
            "equivalent_creep_strain": self.equivalent_creep_strain.tolist(),
            "tensor_history": self.creep_strain is not None,
        }


@dataclass(frozen=True)
class CreepDamageState:
    """Local creep strain and scalar continuum-damage state."""

    equivalent_creep_strain: float = 0.0
    damage: float = 0.0
    creep_strain: np.ndarray | None = None

    def __post_init__(self) -> None:
        equivalent = float(self.equivalent_creep_strain)
        damage = float(self.damage)
        if not isfinite(equivalent) or equivalent < 0.0:
            raise ValueError("equivalent_creep_strain must be finite and nonnegative.")
        if not isfinite(damage) or not 0.0 <= damage < 1.0:
            raise ValueError("creep damage must satisfy 0 <= damage < 1.")
        tensor = self.creep_strain
        if tensor is not None:
            tensor = np.asarray(tensor, dtype=float)
            if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
                raise ValueError("creep_strain must be a finite 3x3 tensor.")
            tensor = tensor.copy()
        object.__setattr__(self, "equivalent_creep_strain", equivalent)
        object.__setattr__(self, "damage", damage)
        object.__setattr__(self, "creep_strain", tensor)


@dataclass(frozen=True)
class CreepDamageUpdate:
    """Accepted material-point increment from a creep-damage law."""

    state: CreepDamageState
    equivalent_increment: float
    damage_increment: float
    failed: bool = False


@dataclass(frozen=True)
class ImplicitCreepState:
    """Committed small-strain creep state at one integration point."""

    creep_strain: np.ndarray | None = None
    equivalent_creep_strain: float = 0.0

    def __post_init__(self) -> None:
        tensor = (
            np.zeros((3, 3), dtype=float)
            if self.creep_strain is None
            else np.asarray(self.creep_strain, dtype=float)
        )
        equivalent = float(self.equivalent_creep_strain)
        if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
            raise ValueError("creep_strain must be a finite 3x3 tensor.")
        if not np.isfinite(equivalent) or equivalent < 0.0:
            raise ValueError(
                "equivalent_creep_strain must be finite and nonnegative."
            )
        object.__setattr__(self, "creep_strain", tensor.copy())
        object.__setattr__(self, "equivalent_creep_strain", equivalent)


@dataclass(frozen=True)
class ImplicitCreepUpdate:
    """Backward-Euler material-point update and consistent tangent."""

    stress: np.ndarray
    state: ImplicitCreepState
    equivalent_increment: float
    algorithmic_tangent: np.ndarray
    local_iterations: int
    converged: bool = True


@dataclass(frozen=True)
class ImplicitCreepBatchUpdate:
    """Vectorized backward-Euler updates for one homogeneous material region."""

    stress: np.ndarray
    creep_strain: np.ndarray
    equivalent_creep_strain: np.ndarray
    equivalent_increment: np.ndarray
    algorithmic_tangent: np.ndarray
    local_iterations: np.ndarray
    converged: np.ndarray


@dataclass(frozen=True)
class IsotropicPowerLawCreepMaterial:
    """Isotropic elasticity with an implicit Mises power-law creep branch.

    The local corrector solves

    ``dgamma = dt * A(t_end) * (q_new / sigma_ref)**n``

    with ``q_new = q_trial - 3*G*dgamma``.  Differentiating this exact
    backward-Euler equation supplies the algorithmic tangent consumed by the
    global Newton solve.
    """

    elastic: ElasticIsotropicProperties
    creep: "PowerLawCreep"
    name: str = "isotropic power-law creep material"
    temperature_dependence: "ArrheniusPowerLawCreep | None" = None
    stateful_constitutive: ClassVar[bool] = True

    @property
    def young(self) -> float:
        return self.elastic.young

    @property
    def poisson(self) -> float:
        return self.elastic.poisson

    @property
    def density(self) -> float:
        return self.elastic.density

    @property
    def shear_modulus(self) -> float:
        return self.elastic.mu

    @property
    def bulk_modulus(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    def elastic_tangent(self) -> np.ndarray:
        identity = np.eye(3)
        symmetric_identity = 0.5 * (
            np.einsum("ik,jl->ijkl", identity, identity)
            + np.einsum("il,jk->ijkl", identity, identity)
        )
        deviatoric_identity = symmetric_identity - np.einsum(
            "ij,kl->ijkl", identity, identity
        ) / 3.0
        return (
            self.bulk_modulus * np.einsum("ij,kl->ijkl", identity, identity)
            + 2.0 * self.shear_modulus * deviatoric_identity
        )

    def stress_from_state(
        self,
        total_strain,
        state: ImplicitCreepState | None = None,
    ) -> np.ndarray:
        """Evaluate stress from a committed state without advancing time.

        This operation is deliberately separate from :meth:`update`.  Output
        recovery and checkpoint restore must never create a fictitious creep
        increment merely to reconstruct the accepted stress.
        """

        strain = np.asarray(total_strain, dtype=float)
        if strain.shape != (3, 3) or not np.all(np.isfinite(strain)):
            raise ValueError("total_strain must be a finite 3x3 tensor.")
        strain = 0.5 * (strain + strain.T)
        accepted = ImplicitCreepState() if state is None else state
        elastic_strain = strain - accepted.creep_strain
        return (
            2.0 * self.shear_modulus * deviatoric(elastic_strain)
            + self.bulk_modulus * np.trace(elastic_strain) * np.eye(3)
        )

    def update(
        self,
        total_strain,
        *,
        time_start: float,
        time_end: float,
        state: ImplicitCreepState | None = None,
        tolerance: float = 1.0e-12,
        maximum_iterations: int = 30,
        temperature: float | None = None,
    ) -> ImplicitCreepUpdate:
        """Integrate one strain-driven creep increment by backward Euler."""

        strain = np.asarray(total_strain, dtype=float)
        if strain.shape != (3, 3) or not np.all(np.isfinite(strain)):
            raise ValueError("total_strain must be a finite 3x3 tensor.")
        strain = 0.5 * (strain + strain.T)
        start = float(time_start)
        end = float(time_end)
        dt = end - start
        if not np.isfinite(start) or not np.isfinite(end) or start < 0.0 or dt <= 0.0:
            raise ValueError(
                "Creep update requires finite 0 <= time_start < time_end."
            )
        if tolerance <= 0.0 or maximum_iterations <= 0:
            raise ValueError("Local tolerance and maximum_iterations must be positive.")
        old = ImplicitCreepState() if state is None else state
        if self.temperature_dependence is None:
            if temperature is not None:
                raise ValueError(
                    "temperature was supplied to an isothermal creep material."
                )
            creep_law = self.creep
        else:
            if temperature is None:
                raise ValueError(
                    "Arrhenius creep requires an absolute temperature at every "
                    "integration point."
                )
            creep_law = self.temperature_dependence.at_temperature(temperature)
        elastic_trial = strain - old.creep_strain
        trial_stress = (
            2.0 * self.shear_modulus * deviatoric(elastic_trial)
            + self.bulk_modulus * np.trace(elastic_trial) * np.eye(3)
        )
        trial_deviator = deviatoric(trial_stress)
        q_trial = von_mises(trial_stress)
        if q_trial <= max(1.0, self.young) * tolerance:
            return ImplicitCreepUpdate(
                stress=trial_stress,
                state=old,
                equivalent_increment=0.0,
                algorithmic_tangent=self.elastic_tangent(),
                local_iterations=0,
            )

        upper = q_trial / (3.0 * self.shear_modulus)
        increment = min(dt * creep_law.equivalent_rate(q_trial, end), upper)
        lower = 0.0
        iterations = 0
        converged = False
        for iterations in range(1, maximum_iterations + 1):
            q = max(0.0, q_trial - 3.0 * self.shear_modulus * increment)
            rate = creep_law.equivalent_rate(q, end)
            residual = increment - dt * rate
            scale = max(1.0, upper, abs(increment), dt * rate)
            if abs(residual) <= tolerance * scale:
                converged = True
                break
            if residual > 0.0:
                upper = increment
            else:
                lower = increment
            rate_derivative = (
                0.0
                if q <= 0.0 or rate == 0.0
                else creep_law.stress_exponent * rate / q
            )
            derivative = 1.0 + 3.0 * self.shear_modulus * dt * rate_derivative
            candidate = increment - residual / derivative
            if not lower < candidate < upper:
                candidate = 0.5 * (lower + upper)
            increment = candidate
        if not converged:
            raise RuntimeError(
                "Implicit power-law creep local update did not converge within "
                f"{maximum_iterations} iterations."
            )

        q = max(0.0, q_trial - 3.0 * self.shear_modulus * increment)
        direction = 1.5 * trial_deviator / q_trial
        creep_strain = old.creep_strain + increment * direction
        reduction = q / q_trial
        pressure = np.trace(trial_stress) * np.eye(3) / 3.0
        stress = pressure + reduction * trial_deviator
        rate = creep_law.equivalent_rate(q, end)
        rate_derivative = (
            0.0
            if q <= 0.0 or rate == 0.0
            else creep_law.stress_exponent * rate / q
        )
        d_increment_d_qtrial = (
            0.0
            if rate_derivative == 0.0
            else dt * rate_derivative
            / (1.0 + 3.0 * self.shear_modulus * dt * rate_derivative)
        )
        identity = np.eye(3)
        symmetric_identity = 0.5 * (
            np.einsum("ik,jl->ijkl", identity, identity)
            + np.einsum("il,jk->ijkl", identity, identity)
        )
        deviatoric_identity = symmetric_identity - np.einsum(
            "ij,kl->ijkl", identity, identity
        ) / 3.0
        radial_coefficient = (
            d_increment_d_qtrial / q_trial - increment / q_trial**2
        )
        tangent = (
            self.bulk_modulus * np.einsum("ij,kl->ijkl", identity, identity)
            + 2.0 * self.shear_modulus * reduction * deviatoric_identity
            - 6.0
            * self.shear_modulus**2
            * radial_coefficient
            * np.einsum("ij,kl->ijkl", trial_deviator, direction)
        )
        return ImplicitCreepUpdate(
            stress=stress,
            state=ImplicitCreepState(
                creep_strain,
                old.equivalent_creep_strain + increment,
            ),
            equivalent_increment=float(increment),
            algorithmic_tangent=tangent,
            local_iterations=iterations,
        )

    def update_many(
        self,
        total_strain,
        *,
        time_start: float,
        time_end: float,
        creep_strain,
        equivalent_creep_strain,
        tolerance: float = 1.0e-12,
        maximum_iterations: int = 30,
    ) -> ImplicitCreepBatchUpdate:
        """Integrate many isothermal points with the scalar algorithm exactly.

        This is a performance path, not a second constitutive model. It uses
        the same safeguarded backward-Euler equations and consistent tangent
        as :meth:`update`, while evaluating a homogeneous quadrature region in
        NumPy batches. Temperature-dependent and heterogeneous regions retain
        the scalar dispatch where each point may select a different law.
        """

        if self.temperature_dependence is not None:
            raise ValueError(
                "update_many is the homogeneous isothermal path; "
                "temperature-dependent updates require point temperatures."
            )
        strain = np.asarray(total_strain, dtype=float)
        old_creep = np.asarray(creep_strain, dtype=float)
        old_equivalent = np.asarray(equivalent_creep_strain, dtype=float).reshape(-1)
        if strain.ndim != 3 or strain.shape[1:] != (3, 3):
            raise ValueError("total_strain must have shape (points, 3, 3).")
        if old_creep.shape != strain.shape or old_equivalent.shape != (len(strain),):
            raise ValueError("Batch creep state and strain layouts do not match.")
        if (
            np.any(~np.isfinite(strain))
            or np.any(~np.isfinite(old_creep))
            or np.any(~np.isfinite(old_equivalent))
            or np.any(old_equivalent < 0.0)
        ):
            raise ValueError("Batch strain and creep state must be finite and valid.")
        start = float(time_start)
        end = float(time_end)
        dt = end - start
        if not np.isfinite(start) or not np.isfinite(end) or start < 0.0 or dt <= 0.0:
            raise ValueError("Creep update requires finite 0 <= time_start < time_end.")
        if tolerance <= 0.0 or maximum_iterations <= 0:
            raise ValueError("Local tolerance and maximum_iterations must be positive.")

        strain = 0.5 * (strain + np.swapaxes(strain, 1, 2))
        identity = np.eye(3)
        elastic_trial = strain - old_creep
        trace = np.trace(elastic_trial, axis1=1, axis2=2)
        deviatoric_strain = elastic_trial - trace[:, None, None] * identity / 3.0
        trial_stress = (
            2.0 * self.shear_modulus * deviatoric_strain
            + self.bulk_modulus * trace[:, None, None] * identity
        )
        stress_trace = np.trace(trial_stress, axis1=1, axis2=2)
        trial_deviator = trial_stress - stress_trace[:, None, None] * identity / 3.0
        q_trial = np.sqrt(1.5 * np.sum(trial_deviator**2, axis=(1, 2)))
        active = q_trial > max(1.0, self.young) * tolerance

        upper = np.zeros_like(q_trial)
        upper[active] = q_trial[active] / (3.0 * self.shear_modulus)
        time_factor = (
            1.0
            if self.creep.time_exponent == 0.0
            else (end / self.creep.reference_time) ** self.creep.time_exponent
        )

        def rates(equivalent):
            return (
                self.creep.coefficient
                * (equivalent / self.creep.reference_stress)
                ** self.creep.stress_exponent
                * time_factor
            )

        increment = np.zeros_like(q_trial)
        increment[active] = np.minimum(
            dt * rates(q_trial[active]),
            upper[active],
        )
        lower = np.zeros_like(q_trial)
        converged = ~active
        local_iterations = np.zeros(len(strain), dtype=np.int32)
        for iteration in range(1, int(maximum_iterations) + 1):
            pending = active & ~converged
            if not np.any(pending):
                break
            q = np.maximum(
                0.0,
                q_trial[pending] - 3.0 * self.shear_modulus * increment[pending],
            )
            rate = rates(q)
            residual = increment[pending] - dt * rate
            scale = np.maximum.reduce(
                (
                    np.ones_like(rate),
                    upper[pending],
                    np.abs(increment[pending]),
                    dt * rate,
                )
            )
            selected = np.flatnonzero(pending)
            finished = np.abs(residual) <= tolerance * scale
            converged[selected[finished]] = True
            local_iterations[selected] = iteration
            remaining = ~finished
            if not np.any(remaining):
                continue
            indices = selected[remaining]
            residual = residual[remaining]
            q = q[remaining]
            rate = rate[remaining]
            positive = residual > 0.0
            upper[indices[positive]] = increment[indices[positive]]
            lower[indices[~positive]] = increment[indices[~positive]]
            derivative_rate = np.zeros_like(rate)
            nonzero = (q > 0.0) & (rate != 0.0)
            derivative_rate[nonzero] = (
                self.creep.stress_exponent * rate[nonzero] / q[nonzero]
            )
            derivative = 1.0 + 3.0 * self.shear_modulus * dt * derivative_rate
            candidate = increment[indices] - residual / derivative
            outside = (candidate <= lower[indices]) | (candidate >= upper[indices])
            candidate[outside] = 0.5 * (
                lower[indices[outside]] + upper[indices[outside]]
            )
            increment[indices] = candidate
        if not np.all(converged):
            index = int(np.flatnonzero(~converged)[0])
            raise RuntimeError(
                "Implicit power-law creep batch update did not converge at "
                f"point {index} within {maximum_iterations} iterations."
            )

        q = np.maximum(0.0, q_trial - 3.0 * self.shear_modulus * increment)
        direction = np.zeros_like(trial_deviator)
        direction[active] = (
            1.5
            * trial_deviator[active]
            / q_trial[active, None, None]
        )
        next_creep = old_creep + increment[:, None, None] * direction
        reduction = np.ones_like(q_trial)
        reduction[active] = q[active] / q_trial[active]
        pressure = stress_trace[:, None, None] * identity / 3.0
        stress = pressure + reduction[:, None, None] * trial_deviator

        rate = rates(q)
        rate_derivative = np.zeros_like(rate)
        nonzero = active & (q > 0.0) & (rate != 0.0)
        rate_derivative[nonzero] = (
            self.creep.stress_exponent * rate[nonzero] / q[nonzero]
        )
        d_increment_d_qtrial = np.zeros_like(q_trial)
        d_increment_d_qtrial[nonzero] = (
            dt
            * rate_derivative[nonzero]
            / (
                1.0
                + 3.0 * self.shear_modulus * dt * rate_derivative[nonzero]
            )
        )
        radial_coefficient = np.zeros_like(q_trial)
        radial_coefficient[active] = (
            d_increment_d_qtrial[active] / q_trial[active]
            - increment[active] / q_trial[active] ** 2
        )
        symmetric_identity = 0.5 * (
            np.einsum("ik,jl->ijkl", identity, identity)
            + np.einsum("il,jk->ijkl", identity, identity)
        )
        deviatoric_identity = symmetric_identity - np.einsum(
            "ij,kl->ijkl", identity, identity
        ) / 3.0
        tangent = np.broadcast_to(
            self.bulk_modulus * np.einsum("ij,kl->ijkl", identity, identity),
            (len(strain), 3, 3, 3, 3),
        ).copy()
        tangent += (
            2.0
            * self.shear_modulus
            * reduction[:, None, None, None, None]
            * deviatoric_identity
        )
        tangent -= (
            6.0
            * self.shear_modulus**2
            * radial_coefficient[:, None, None, None, None]
            * np.einsum("nij,nkl->nijkl", trial_deviator, direction)
        )
        return ImplicitCreepBatchUpdate(
            stress=stress,
            creep_strain=next_creep,
            equivalent_creep_strain=old_equivalent + increment,
            equivalent_increment=increment,
            algorithmic_tangent=tangent,
            local_iterations=local_iterations,
            converged=converged,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "isotropic_power_law_creep",
            "elastic": self.elastic.as_dict(),
            "creep": self.creep.as_dict(),
            "temperature_dependence": (
                None
                if self.temperature_dependence is None
                else self.temperature_dependence.as_dict()
            ),
            "integration": "backward_euler",
            "algorithmic_tangent": "analytical_consistent",
            "maturity": "fem_integrated_foundation",
            "fem_quadrature_driver": True,
        }


@dataclass(frozen=True)
class KachanovRabotnovCreep:
    """Classical scalar Kachanov--Rabotnov creep-damage coupling.

    For normalized equivalent stress ``s = q / reference_stress``, the local
    equations are

    ``eps_dot = A s**n / (1 - omega)**n`` and
    ``omega_dot = B s**m / (1 - omega)**phi``.

    The constant-stress update integrates both equations analytically.  This
    avoids time-step-dependent damage in material-point verification and gives
    a reference update for a future global quadrature driver.  Multiaxial use
    currently drives both equations with von Mises stress; alternative rupture
    stress measures must be introduced explicitly rather than hidden.
    """

    creep_coefficient: float
    creep_exponent: float
    damage_coefficient: float
    damage_exponent: float
    damage_power: float
    reference_stress: float = 1.0
    reference_time: float = 1.0
    failure_damage: float = 0.999
    name: str = "Kachanov-Rabotnov creep damage"

    def __post_init__(self) -> None:
        values = (
            self.creep_coefficient,
            self.creep_exponent,
            self.damage_coefficient,
            self.damage_exponent,
            self.damage_power,
            self.reference_stress,
            self.reference_time,
            self.failure_damage,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("KachanovRabotnovCreep parameters must be finite.")
        if self.creep_coefficient < 0.0 or self.damage_coefficient < 0.0:
            raise ValueError("K-R rate coefficients must be nonnegative.")
        if self.creep_exponent <= 0.0 or self.damage_exponent <= 0.0:
            raise ValueError("K-R stress exponents must be positive.")
        if self.damage_power < 0.0:
            raise ValueError("damage_power must be nonnegative.")
        if self.reference_stress <= 0.0 or self.reference_time <= 0.0:
            raise ValueError("K-R reference scales must be positive.")
        if not 0.0 < self.failure_damage < 1.0:
            raise ValueError("failure_damage must lie strictly between zero and one.")

    def update(
        self,
        stress,
        duration: float,
        state: CreepDamageState | None = None,
    ) -> CreepDamageUpdate:
        """Integrate one piecewise-constant stress interval exactly."""

        dt = float(duration)
        if not isfinite(dt) or dt < 0.0:
            raise ValueError("duration must be finite and nonnegative.")
        old = CreepDamageState() if state is None else state
        selected = np.asarray(stress, dtype=float)
        tensor_stress = selected.shape == (3, 3)
        if tensor_stress:
            q = von_mises(selected)
            if old.creep_strain is None and old.equivalent_creep_strain > 0.0:
                raise ValueError(
                    "Cannot switch a scalar K-R history to tensor flow because "
                    "its prior tensor direction is unavailable."
                )
        elif selected.ndim == 0:
            q = abs(float(selected))
            if old.creep_strain is not None:
                raise ValueError(
                    "Cannot switch a tensor K-R history to scalar stress."
                )
        else:
            raise ValueError("stress must be a scalar or symmetric 3x3 tensor.")
        normalized = q / self.reference_stress
        old_integrity = 1.0 - old.damage
        damage_scale = (
            self.damage_coefficient
            * normalized**self.damage_exponent
            * dt
            / self.reference_time
        )
        power = self.damage_power + 1.0
        remaining = old_integrity**power - power * damage_scale
        failed = remaining <= (1.0 - self.failure_damage) ** power
        new_integrity = max(
            1.0 - self.failure_damage,
            max(0.0, remaining) ** (1.0 / power),
        )
        new_damage = 1.0 - new_integrity

        equivalent_increment = self._creep_increment(
            normalized,
            old_integrity,
            new_integrity,
            dt,
        )
        tensor = old.creep_strain
        if tensor_stress:
            tensor = np.zeros((3, 3)) if tensor is None else tensor.copy()
            if q > 0.0:
                tensor += equivalent_increment * 1.5 * deviatoric(selected) / q
        new_state = CreepDamageState(
            old.equivalent_creep_strain + equivalent_increment,
            new_damage,
            tensor,
        )
        return CreepDamageUpdate(
            state=new_state,
            equivalent_increment=equivalent_increment,
            damage_increment=new_damage - old.damage,
            failed=failed,
        )

    def rupture_time(self, equivalent_stress: float) -> float:
        """Return time to ``failure_damage`` under constant stress."""

        normalized = abs(float(equivalent_stress)) / self.reference_stress
        rate = self.damage_coefficient * normalized**self.damage_exponent
        if rate == 0.0:
            return float("inf")
        integrity = 1.0 - self.failure_damage
        return float(
            self.reference_time
            * (1.0 - integrity ** (self.damage_power + 1.0))
            / ((self.damage_power + 1.0) * rate)
        )

    def _creep_increment(
        self,
        normalized_stress: float,
        old_integrity: float,
        new_integrity: float,
        duration: float,
    ) -> float:
        if normalized_stress == 0.0 or self.creep_coefficient == 0.0:
            return 0.0
        if self.damage_coefficient == 0.0:
            return float(
                self.creep_coefficient
                * normalized_stress**self.creep_exponent
                * duration
                / self.reference_time
                / old_integrity**self.creep_exponent
            )
        exponent = self.damage_power - self.creep_exponent + 1.0
        prefactor = (
            self.creep_coefficient
            / self.damage_coefficient
            * normalized_stress
            ** (self.creep_exponent - self.damage_exponent)
        )
        if abs(exponent) <= 1.0e-12:
            integral = log(old_integrity / new_integrity)
        else:
            integral = (
                old_integrity**exponent - new_integrity**exponent
            ) / exponent
        return float(prefactor * integral)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "kachanov_rabotnov_creep_damage",
            "creep_coefficient": self.creep_coefficient,
            "creep_exponent": self.creep_exponent,
            "damage_coefficient": self.damage_coefficient,
            "damage_exponent": self.damage_exponent,
            "damage_power": self.damage_power,
            "reference_stress": self.reference_stress,
            "reference_time": self.reference_time,
            "failure_damage": self.failure_damage,
            "damage_stress_measure": "von_mises",
            "maturity": "material_point_verified",
            "fem_quadrature_driver": False,
        }


@dataclass(frozen=True)
class SinhCreep:
    """Stress-sensitive hyperbolic-sine Mises creep law."""

    coefficient: float
    stress_scale: float
    exponent: float = 1.0
    reference_time: float = 1.0
    name: str = "hyperbolic-sine creep"

    def __post_init__(self) -> None:
        values = (self.coefficient, self.stress_scale, self.exponent, self.reference_time)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("SinhCreep parameters must be finite.")
        if self.coefficient < 0.0:
            raise ValueError("SinhCreep.coefficient must be nonnegative.")
        if self.stress_scale <= 0.0 or self.exponent <= 0.0 or self.reference_time <= 0.0:
            raise ValueError("SinhCreep scales and exponent must be positive.")

    def equivalent_rate(self, equivalent_stress: float) -> float:
        return float(
            self.coefficient
            * sinh(abs(float(equivalent_stress)) / self.stress_scale) ** self.exponent
            / self.reference_time
        )

    def tensor_increment(self, stress, duration: float) -> np.ndarray:
        selected = np.asarray(stress, dtype=float)
        q = von_mises(selected)
        if q == 0.0:
            return np.zeros((3, 3))
        equivalent = self.equivalent_rate(q) * float(duration)
        return equivalent * 1.5 * deviatoric(selected) / q

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "mises_hyperbolic_sine_creep",
            "coefficient": self.coefficient,
            "stress_scale": self.stress_scale,
            "exponent": self.exponent,
            "reference_time": self.reference_time,
            "maturity": "material_point_verified",
            "fem_quadrature_driver": False,
        }


@dataclass(frozen=True)
class ModifiedThetaProjection:
    """Three-parameter modified-theta representation of a creep curve.

    ``epsilon(t) = epsilon_0 + A(1-exp(-alpha*t)) + B(exp(alpha*t)-1)``.
    This object is deliberately a curve projection and life-assessment aid;
    it is not advertised as a stress-update law for a global FE solve.
    """

    initial_strain: float
    primary_strain: float
    tertiary_strain: float
    rate: float
    fit_rmse: float | None = None
    name: str = "modified theta projection"

    def __post_init__(self) -> None:
        values = (self.initial_strain, self.primary_strain, self.tertiary_strain, self.rate)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("ModifiedThetaProjection parameters must be finite.")
        if self.primary_strain < 0.0 or self.tertiary_strain < 0.0 or self.rate <= 0.0:
            raise ValueError("Modified-theta amplitudes must be nonnegative and rate positive.")
        if self.fit_rmse is not None and (not isfinite(self.fit_rmse) or self.fit_rmse < 0.0):
            raise ValueError("fit_rmse must be finite and nonnegative.")

    def strain(self, time):
        selected = np.asarray(time, dtype=float)
        if np.any(~np.isfinite(selected)) or np.any(selected < 0.0):
            raise ValueError("time must be finite and nonnegative.")
        value = (
            self.initial_strain
            + self.primary_strain * (-np.expm1(-self.rate * selected))
            + self.tertiary_strain * np.expm1(self.rate * selected)
        )
        return float(value) if value.ndim == 0 else value

    def strain_rate(self, time):
        selected = np.asarray(time, dtype=float)
        if np.any(~np.isfinite(selected)) or np.any(selected < 0.0):
            raise ValueError("time must be finite and nonnegative.")
        value = self.rate * (
            self.primary_strain * np.exp(-self.rate * selected)
            + self.tertiary_strain * np.exp(self.rate * selected)
        )
        return float(value) if value.ndim == 0 else value

    def time_to_strain(self, target: float, *, maximum_time: float) -> float:
        """Bracket and bisect the first time reaching a strain criterion."""

        selected_target = float(target)
        upper = float(maximum_time)
        if selected_target < self.initial_strain or upper <= 0.0:
            raise ValueError("target and maximum_time do not bracket a future criterion.")
        if self.strain(upper) < selected_target:
            raise ValueError("target strain is not reached within maximum_time.")
        lower = 0.0
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            if self.strain(midpoint) < selected_target:
                lower = midpoint
            else:
                upper = midpoint
        return upper

    @classmethod
    def fit(
        cls,
        times,
        strains,
        *,
        initial_strain: float | None = None,
        rate_bounds: tuple[float, float] | None = None,
        candidates: int = 320,
    ) -> "ModifiedThetaProjection":
        """Fit a deterministic nonnegative projection without SciPy."""

        selected_time = np.asarray(times, dtype=float)
        selected_strain = np.asarray(strains, dtype=float)
        if selected_time.ndim != 1 or selected_time.size < 4:
            raise ValueError("modified-theta fitting requires at least four times.")
        if selected_strain.shape != selected_time.shape:
            raise ValueError("times and strains must have identical one-dimensional shapes.")
        if np.any(~np.isfinite(selected_time)) or np.any(np.diff(selected_time) <= 0.0):
            raise ValueError("times must be finite and strictly increasing.")
        if np.any(~np.isfinite(selected_strain)):
            raise ValueError("strains must be finite.")
        epsilon0 = float(selected_strain[0] if initial_strain is None else initial_strain)
        relative_time = selected_time - selected_time[0]
        span = float(relative_time[-1])
        bounds = rate_bounds or (1.0e-3 / span, 5.0 / span)
        if bounds[0] <= 0.0 or bounds[1] <= bounds[0] or candidates < 8:
            raise ValueError("rate_bounds and candidates do not define a valid search.")
        best = None
        for alpha in np.geomspace(bounds[0], bounds[1], int(candidates)):
            basis = np.column_stack(
                (
                    -np.expm1(-alpha * relative_time),
                    np.expm1(alpha * relative_time),
                )
            )
            amplitudes, *_ = np.linalg.lstsq(
                basis,
                selected_strain - epsilon0,
                rcond=None,
            )
            if np.any(amplitudes < 0.0):
                continue
            residual = basis @ amplitudes + epsilon0 - selected_strain
            rmse = float(np.sqrt(np.mean(residual**2)))
            if best is None or rmse < best[0]:
                best = (rmse, float(alpha), amplitudes)
        if best is None:
            raise ValueError("data cannot be fit with nonnegative modified-theta amplitudes.")
        return cls(epsilon0, float(best[2][0]), float(best[2][1]), best[1], best[0])

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "modified_theta_projection",
            "initial_strain": self.initial_strain,
            "primary_strain": self.primary_strain,
            "tertiary_strain": self.tertiary_strain,
            "rate": self.rate,
            "fit_rmse": self.fit_rmse,
            "maturity": "curve_projection_verified",
            "fem_quadrature_driver": False,
        }


@dataclass(frozen=True)
class PowerLawCreep:
    """Mises time-hardening creep law.

    The equivalent creep rate is

    ``epsilon_dot = A (q / sigma_ref)^n (t / time_ref)^m``.

    With this normalized form ``A`` has units of inverse time.  Set
    ``reference_stress=1`` and use a consistent stress unit to reproduce the
    conventional dimensionful ``A q^n t^m`` notation.
    """

    coefficient: float
    stress_exponent: float
    time_exponent: float = 0.0
    reference_stress: float = 1.0
    reference_time: float = 1.0
    name: str = "Mises power-law creep"

    def __post_init__(self) -> None:
        values = (
            self.coefficient,
            self.stress_exponent,
            self.time_exponent,
            self.reference_stress,
            self.reference_time,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("PowerLawCreep parameters must be finite.")
        if self.coefficient < 0.0:
            raise ValueError("PowerLawCreep.coefficient must be nonnegative.")
        if self.stress_exponent <= 0.0:
            raise ValueError("PowerLawCreep.stress_exponent must be positive.")
        if self.time_exponent <= -1.0:
            raise ValueError("PowerLawCreep.time_exponent must be greater than -1.")
        if self.reference_stress <= 0.0 or self.reference_time <= 0.0:
            raise ValueError("PowerLawCreep reference scales must be positive.")

    def equivalent_rate(self, equivalent_stress: float, time: float) -> float:
        q = abs(float(equivalent_stress))
        selected_time = float(time)
        if selected_time < 0.0:
            raise ValueError("creep time must be nonnegative.")
        if selected_time == 0.0 and self.time_exponent < 0.0:
            return float("inf")
        time_factor = (
            1.0
            if self.time_exponent == 0.0
            else (selected_time / self.reference_time) ** self.time_exponent
        )
        return float(
            self.coefficient
            * (q / self.reference_stress) ** self.stress_exponent
            * time_factor
        )

    def constant_stress_strain(
        self,
        equivalent_stress: float,
        time: float,
    ) -> float:
        """Return the exact equivalent creep strain from zero to ``time``."""

        selected_time = float(time)
        if selected_time < 0.0:
            raise ValueError("creep time must be nonnegative.")
        q_factor = (
            abs(float(equivalent_stress)) / self.reference_stress
        ) ** self.stress_exponent
        normalized_time = selected_time / self.reference_time
        return float(
            self.coefficient
            * self.reference_time
            * q_factor
            * normalized_time ** (self.time_exponent + 1.0)
            / (self.time_exponent + 1.0)
        )

    def constant_stress_increment(
        self,
        equivalent_stress: float,
        time_start: float,
        time_end: float,
    ) -> float:
        if time_end < time_start:
            raise ValueError("time_end must be greater than or equal to time_start.")
        return self.constant_stress_strain(
            equivalent_stress,
            time_end,
        ) - self.constant_stress_strain(equivalent_stress, time_start)

    def tensor_increment(
        self,
        stress,
        time_start: float,
        time_end: float,
    ) -> np.ndarray:
        """Return an associative Mises creep-strain increment tensor."""

        selected = np.asarray(stress, dtype=float)
        q = von_mises(selected)
        if q == 0.0:
            return np.zeros((3, 3))
        equivalent_increment = self.constant_stress_increment(
            q,
            time_start,
            time_end,
        )
        return equivalent_increment * 1.5 * deviatoric(selected) / q

    def relaxation_stress(
        self,
        *,
        initial_stress: float,
        young: float,
        time: float,
    ) -> float:
        """Closed-form stress for a constant-total-strain relaxation test."""

        sigma0 = float(initial_stress)
        modulus = float(young)
        selected_time = float(time)
        if sigma0 <= 0.0 or modulus <= 0.0:
            raise ValueError("initial_stress and young must be positive.")
        if selected_time < 0.0:
            raise ValueError("time must be nonnegative.")
        exponent = self.stress_exponent
        if np.isclose(exponent, 1.0):
            power = (
                self.coefficient
                * modulus
                * self.reference_time
                / self.reference_stress
                * (selected_time / self.reference_time)
                ** (self.time_exponent + 1.0)
                / (self.time_exponent + 1.0)
            )
            return float(sigma0 * np.exp(-power))
        accumulated = (
            self.coefficient
            * modulus
            * (exponent - 1.0)
            * self.reference_time
            / self.reference_stress**exponent
            * (selected_time / self.reference_time)
            ** (self.time_exponent + 1.0)
            / (self.time_exponent + 1.0)
        )
        return float(
            (sigma0 ** (1.0 - exponent) + accumulated)
            ** (1.0 / (1.0 - exponent))
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "mises_power_law_time_hardening_creep",
            "coefficient": self.coefficient,
            "stress_exponent": self.stress_exponent,
            "time_exponent": self.time_exponent,
            "reference_stress": self.reference_stress,
            "reference_time": self.reference_time,
            "maturity": "material_point_verified",
            "fem_quadrature_driver": False,
        }


def isotropic_power_law(
    *,
    young: float,
    poisson: float,
    density: float,
    coefficient: float,
    stress_exponent: float,
    time_exponent: float = 0.0,
    reference_stress: float = 1.0,
    reference_time: float = 1.0,
    name: str = "isotropic power-law creep",
) -> IsotropicPowerLawCreepMaterial:
    """Create one Abaqus-style material record with elastic and creep data."""

    elastic = ElasticIsotropicProperties(
        name=f"{name} elastic",
        young=young,
        density=density,
        poisson=poisson,
    )
    law = PowerLawCreep(
        coefficient=coefficient,
        stress_exponent=stress_exponent,
        time_exponent=time_exponent,
        reference_stress=reference_stress,
        reference_time=reference_time,
        name=f"{name} creep law",
    )
    return IsotropicPowerLawCreepMaterial(elastic=elastic, creep=law, name=name)


@dataclass(frozen=True)
class ArrheniusPowerLawCreep:
    """Temperature-dependent Mises power-law creep.

    ``coefficient`` is the equivalent creep rate coefficient calibrated at
    ``reference_temperature``.  The normalized Arrhenius factor avoids
    silently changing the meaning of a fitted coefficient:

    ``A(T) = A_ref exp[-Q/R (1/T - 1/T_ref)]``.
    """

    coefficient: float
    stress_exponent: float
    activation_energy: float
    reference_temperature: float
    time_exponent: float = 0.0
    reference_stress: float = 1.0
    reference_time: float = 1.0
    gas_constant: float = 8.31446261815324
    name: str = "Arrhenius Mises power-law creep"

    def __post_init__(self) -> None:
        values = (
            self.coefficient,
            self.stress_exponent,
            self.activation_energy,
            self.reference_temperature,
            self.time_exponent,
            self.reference_stress,
            self.reference_time,
            self.gas_constant,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("ArrheniusPowerLawCreep parameters must be finite.")
        if self.activation_energy < 0.0:
            raise ValueError("activation_energy must be nonnegative.")
        if self.reference_temperature <= 0.0:
            raise ValueError("reference_temperature must be positive in kelvin.")
        if self.gas_constant <= 0.0:
            raise ValueError("gas_constant must be positive.")
        self.at_temperature(self.reference_temperature)

    def temperature_factor(self, temperature: float) -> float:
        selected = float(temperature)
        if not isfinite(selected) or selected <= 0.0:
            raise ValueError("temperature must be finite and positive in kelvin.")
        exponent = -self.activation_energy / self.gas_constant * (
            1.0 / selected - 1.0 / self.reference_temperature
        )
        return float(exp(exponent))

    def at_temperature(self, temperature: float) -> PowerLawCreep:
        """Return the isothermal local law at a selected absolute temperature."""

        return PowerLawCreep(
            coefficient=self.coefficient * self.temperature_factor(temperature),
            stress_exponent=self.stress_exponent,
            time_exponent=self.time_exponent,
            reference_stress=self.reference_stress,
            reference_time=self.reference_time,
            name=f"{self.name} at {float(temperature):g} K",
        )

    def equivalent_rate(
        self,
        equivalent_stress: float,
        time: float,
        *,
        temperature: float,
    ) -> float:
        return self.at_temperature(temperature).equivalent_rate(
            equivalent_stress,
            time,
        )

    def constant_stress_increment(
        self,
        equivalent_stress: float,
        time_start: float,
        time_end: float,
        *,
        temperature: float,
    ) -> float:
        return self.at_temperature(temperature).constant_stress_increment(
            equivalent_stress,
            time_start,
            time_end,
        )

    def tensor_increment(
        self,
        stress,
        time_start: float,
        time_end: float,
        *,
        temperature: float,
    ) -> np.ndarray:
        return self.at_temperature(temperature).tensor_increment(
            stress,
            time_start,
            time_end,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "arrhenius_mises_power_law_creep",
            "coefficient_at_reference_temperature": self.coefficient,
            "stress_exponent": self.stress_exponent,
            "time_exponent": self.time_exponent,
            "activation_energy": self.activation_energy,
            "gas_constant": self.gas_constant,
            "reference_temperature": self.reference_temperature,
            "reference_stress": self.reference_stress,
            "reference_time": self.reference_time,
            "maturity": "material_point_verified",
            "fem_quadrature_driver": False,
        }


def isotropic_arrhenius_power_law(
    *,
    young: float,
    poisson: float,
    density: float,
    coefficient: float,
    stress_exponent: float,
    activation_energy: float,
    reference_temperature: float,
    time_exponent: float = 0.0,
    reference_stress: float = 1.0,
    reference_time: float = 1.0,
    gas_constant: float = 8.31446261815324,
    name: str = "isotropic Arrhenius power-law creep",
) -> IsotropicPowerLawCreepMaterial:
    """Create elasticity plus a globally consumable Arrhenius creep law."""

    elastic = ElasticIsotropicProperties(
        name=f"{name} elastic",
        young=young,
        density=density,
        poisson=poisson,
    )
    dependence = ArrheniusPowerLawCreep(
        coefficient=coefficient,
        stress_exponent=stress_exponent,
        activation_energy=activation_energy,
        reference_temperature=reference_temperature,
        time_exponent=time_exponent,
        reference_stress=reference_stress,
        reference_time=reference_time,
        gas_constant=gas_constant,
        name=f"{name} creep law",
    )
    return IsotropicPowerLawCreepMaterial(
        elastic=elastic,
        creep=dependence.at_temperature(reference_temperature),
        temperature_dependence=dependence,
        name=name,
    )


def integrate_stress_history(
    law: PowerLawCreep,
    times,
    interval_stresses,
) -> CreepHistory:
    """Integrate a piecewise-constant scalar or tensor stress history.

    ``times`` contains interval boundaries and ``interval_stresses`` contains
    one stress value per interval. Each increment uses the law's exact
    time-hardening integral, avoiding a hidden forward-Euler approximation.
    This is a verified material-point driver, not yet a global FE creep step.
    """

    selected_times = np.asarray(times, dtype=float)
    selected_stresses = np.asarray(interval_stresses, dtype=float)
    if selected_times.ndim != 1 or selected_times.size < 2:
        raise ValueError("times must be a one-dimensional array of length >= 2.")
    if not np.all(np.isfinite(selected_times)):
        raise ValueError("times must be finite.")
    if selected_times[0] < 0.0 or np.any(np.diff(selected_times) <= 0.0):
        raise ValueError("times must be nonnegative and strictly increasing.")
    interval_count = selected_times.size - 1
    scalar_history = selected_stresses.ndim == 1
    tensor_history = (
        selected_stresses.ndim == 3
        and selected_stresses.shape[1:] == (3, 3)
    )
    if not scalar_history and not tensor_history:
        raise ValueError(
            "interval_stresses must have shape (intervals,) or (intervals, 3, 3)."
        )
    if selected_stresses.shape[0] != interval_count:
        raise ValueError("Provide exactly one stress value per time interval.")
    if not np.all(np.isfinite(selected_stresses)):
        raise ValueError("interval_stresses must be finite.")

    equivalent = np.zeros(selected_times.size)
    tensors = (
        np.zeros((selected_times.size, 3, 3))
        if tensor_history
        else None
    )
    for index in range(interval_count):
        start = float(selected_times[index])
        end = float(selected_times[index + 1])
        if scalar_history:
            increment = law.constant_stress_increment(
                float(selected_stresses[index]),
                start,
                end,
            )
        else:
            stress = selected_stresses[index]
            q = von_mises(stress)
            increment = law.constant_stress_increment(q, start, end)
            tensors[index + 1] = (
                tensors[index] + law.tensor_increment(stress, start, end)
            )
        equivalent[index + 1] = equivalent[index] + increment
    return CreepHistory(
        time=selected_times.copy(),
        equivalent_creep_strain=equivalent,
        creep_strain=tensors,
    )

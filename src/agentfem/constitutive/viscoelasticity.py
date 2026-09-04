"""Linear viscoelastic spectra with exact generalized-Maxwell updates."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np


def _positive_vector(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values.")
    return array


@dataclass(frozen=True)
class WLFShift:
    """Williams--Landel--Ferry time-temperature shift factor."""

    reference_temperature: float
    c1: float
    c2: float

    def __post_init__(self) -> None:
        values = (self.reference_temperature, self.c1, self.c2)
        if not np.all(np.isfinite(values)) or self.c1 <= 0.0 or self.c2 <= 0.0:
            raise ValueError(
                "WLF reference_temperature, c1, and c2 must be finite; "
                "c1 and c2 must be positive."
            )

    def factor(self, temperature) -> np.ndarray:
        temperature = np.asarray(temperature, dtype=float)
        difference = temperature - float(self.reference_temperature)
        denominator = float(self.c2) + difference
        if np.any(denominator <= 0.0) or np.any(np.isclose(denominator, 0.0)):
            raise ValueError(
                "WLF temperature lies at or below the model singularity; "
                "restrict the declared temperature range."
            )
        factor = np.power(10.0, -float(self.c1) * difference / denominator)
        if not np.all(np.isfinite(factor)) or np.any(factor <= 0.0):
            raise ValueError("WLF shift factor must remain finite and positive.")
        return factor

    def summary(self) -> dict[str, object]:
        return {
            "kind": "WLF",
            "reference_temperature": self.reference_temperature,
            "c1": self.c1,
            "c2": self.c2,
        }


@dataclass(frozen=True)
class ArrheniusShift:
    """Arrhenius time-temperature shift factor."""

    activation_energy: float
    reference_temperature: float
    gas_constant: float = 8.31446261815324

    def __post_init__(self) -> None:
        values = (
            self.activation_energy,
            self.reference_temperature,
            self.gas_constant,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Arrhenius parameters must be finite.")
        if self.activation_energy <= 0.0 or self.reference_temperature <= 0.0:
            raise ValueError("activation_energy and reference_temperature must be positive.")
        if self.gas_constant <= 0.0:
            raise ValueError("gas_constant must be positive.")

    def factor(self, temperature) -> np.ndarray:
        temperature = np.asarray(temperature, dtype=float)
        if np.any(temperature <= 0.0):
            raise ValueError("Arrhenius temperature must be absolute and positive.")
        exponent = self.activation_energy / self.gas_constant * (
            1.0 / temperature - 1.0 / self.reference_temperature
        )
        factor = np.exp(exponent)
        if not np.all(np.isfinite(factor)) or np.any(factor <= 0.0):
            raise ValueError("Arrhenius shift factor must remain finite and positive.")
        return factor

    def summary(self) -> dict[str, object]:
        return {
            "kind": "Arrhenius",
            "activation_energy": self.activation_energy,
            "reference_temperature": self.reference_temperature,
            "gas_constant": self.gas_constant,
        }


@dataclass(frozen=True)
class ViscoelasticUpdate:
    """One trial material-point update that can be committed atomically."""

    strain: np.ndarray
    overstress: np.ndarray
    stress: np.ndarray
    algorithmic_modulus: float
    dissipated_energy_increment: float


@dataclass
class MaxwellState:
    """Committed state for a generalized-Maxwell material point."""

    strain: np.ndarray
    overstress: np.ndarray
    dissipated_energy: float = 0.0

    @classmethod
    def zero(cls, branch_count: int, *, value_shape=()) -> "MaxwellState":
        if int(branch_count) <= 0:
            raise ValueError("branch_count must be positive.")
        shape = tuple(value_shape)
        return cls(
            strain=np.zeros(shape, dtype=float),
            overstress=np.zeros((int(branch_count), *shape), dtype=float),
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "strain": self.strain.copy(),
            "overstress": self.overstress.copy(),
            "dissipated_energy": float(self.dissipated_energy),
        }

    def restore(self, snapshot) -> None:
        strain = np.asarray(snapshot["strain"], dtype=float)
        overstress = np.asarray(snapshot["overstress"], dtype=float)
        if strain.shape != self.strain.shape or overstress.shape != self.overstress.shape:
            raise ValueError("Viscoelastic snapshot shape does not match the state.")
        self.strain[...] = strain
        self.overstress[...] = overstress
        self.dissipated_energy = float(snapshot["dissipated_energy"])

    def commit(self, update: ViscoelasticUpdate) -> None:
        if update.strain.shape != self.strain.shape or update.overstress.shape != self.overstress.shape:
            raise ValueError("Viscoelastic update shape does not match the state.")
        self.strain[...] = update.strain
        self.overstress[...] = update.overstress
        self.dissipated_energy += float(update.dissipated_energy_increment)


@dataclass(frozen=True)
class GeneralizedMaxwell:
    """Small-strain generalized-Maxwell relaxation spectrum.

    ``equilibrium_modulus`` is the long-time modulus and ``branch_moduli`` are
    the relaxing moduli. Their sum is the instantaneous modulus.
    """

    equilibrium_modulus: float
    branch_moduli: np.ndarray
    relaxation_times: np.ndarray
    shift: WLFShift | ArrheniusShift | None = None
    name: str = "generalized_maxwell"

    def __post_init__(self) -> None:
        if not np.isfinite(self.equilibrium_modulus) or self.equilibrium_modulus <= 0.0:
            raise ValueError("equilibrium_modulus must be finite and positive.")
        moduli = _positive_vector(self.branch_moduli, name="branch_moduli")
        times = _positive_vector(self.relaxation_times, name="relaxation_times")
        if moduli.size != times.size:
            raise ValueError("branch_moduli and relaxation_times must have equal length.")
        object.__setattr__(self, "branch_moduli", moduli.copy())
        object.__setattr__(self, "relaxation_times", times.copy())

    @classmethod
    def from_prony(
        cls,
        instantaneous_modulus: float,
        ratios,
        relaxation_times,
        *,
        shift: WLFShift | ArrheniusShift | None = None,
        name: str = "prony_series",
    ) -> "GeneralizedMaxwell":
        ratios = _positive_vector(ratios, name="ratios")
        if np.sum(ratios) >= 1.0:
            raise ValueError("Prony modulus ratios must sum to less than one.")
        if instantaneous_modulus <= 0.0:
            raise ValueError("instantaneous_modulus must be positive.")
        return cls(
            equilibrium_modulus=float(instantaneous_modulus) * (1.0 - float(np.sum(ratios))),
            branch_moduli=float(instantaneous_modulus) * ratios,
            relaxation_times=relaxation_times,
            shift=shift,
            name=name,
        )

    @property
    def instantaneous_modulus(self) -> float:
        return float(self.equilibrium_modulus + np.sum(self.branch_moduli))

    @property
    def prony_ratios(self) -> np.ndarray:
        return self.branch_moduli / self.instantaneous_modulus

    def shifted_relaxation_times(self, temperature=None) -> np.ndarray:
        if temperature is None or self.shift is None:
            return self.relaxation_times.copy()
        factor = np.asarray(self.shift.factor(temperature), dtype=float)
        if factor.ndim != 0:
            raise ValueError("A material-point update requires one scalar temperature.")
        return self.relaxation_times * float(factor)

    def relaxation_modulus(self, time, *, temperature=None) -> np.ndarray:
        time = np.asarray(time, dtype=float)
        if np.any(time < 0.0) or not np.all(np.isfinite(time)):
            raise ValueError("time must contain finite nonnegative values.")
        times = self.shifted_relaxation_times(temperature)
        return self.equilibrium_modulus + np.sum(
            self.branch_moduli * np.exp(-time[..., None] / times), axis=-1
        )

    def complex_modulus(self, angular_frequency, *, temperature=None) -> np.ndarray:
        omega = np.asarray(angular_frequency, dtype=float)
        if np.any(omega < 0.0) or not np.all(np.isfinite(omega)):
            raise ValueError("angular_frequency must contain finite nonnegative values.")
        times = self.shifted_relaxation_times(temperature)
        reduced = omega[..., None] * times
        branches = self.branch_moduli * (1j * reduced) / (1.0 + 1j * reduced)
        return self.equilibrium_modulus + np.sum(branches, axis=-1)

    def storage_modulus(self, angular_frequency, *, temperature=None) -> np.ndarray:
        return np.real(self.complex_modulus(angular_frequency, temperature=temperature))

    def loss_modulus(self, angular_frequency, *, temperature=None) -> np.ndarray:
        return np.imag(self.complex_modulus(angular_frequency, temperature=temperature))

    def loss_factor(self, angular_frequency, *, temperature=None) -> np.ndarray:
        storage = self.storage_modulus(angular_frequency, temperature=temperature)
        return self.loss_modulus(angular_frequency, temperature=temperature) / storage

    def update(
        self,
        state: MaxwellState,
        strain,
        dt: float,
        *,
        temperature=None,
    ) -> ViscoelasticUpdate:
        """Return an exact branch update for linear strain over one increment."""

        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        selected = np.asarray(strain, dtype=float)
        if selected.shape != state.strain.shape:
            raise ValueError("strain shape must match state.strain.")
        if not np.all(np.isfinite(selected)):
            raise ValueError("strain must contain only finite values.")
        if not np.all(np.isfinite(state.strain)) or not np.all(
            np.isfinite(state.overstress)
        ):
            raise ValueError("Maxwell state must contain only finite values.")
        if state.overstress.shape != (self.branch_moduli.size, *selected.shape):
            raise ValueError("state.overstress does not match the Maxwell spectrum.")
        times = self.shifted_relaxation_times(temperature)
        decay = np.exp(-float(dt) / times)
        integration = times / float(dt) * (1.0 - decay)
        reshape = (self.branch_moduli.size,) + (1,) * selected.ndim
        increment = selected - state.strain
        overstress = decay.reshape(reshape) * state.overstress + (
            self.branch_moduli * integration
        ).reshape(reshape) * increment
        stress = self.equilibrium_modulus * selected + np.sum(overstress, axis=0)
        tangent = float(
            self.equilibrium_modulus + np.sum(self.branch_moduli * integration)
        )
        # For a linear strain path, each branch overstress is
        # q(t) = b + a exp(-t/tau), where b = E tau strain_rate.  Integrating
        # q:q/(E tau) gives the exact, non-negative viscous dissipation.  This
        # avoids hiding an inconsistent energy update with an a-posteriori
        # ``max(0, ...)`` correction.
        strain_rate = increment / float(dt)
        branch_moduli = self.branch_moduli.reshape(reshape)
        branch_times = times.reshape(reshape)
        steady_overstress = branch_moduli * branch_times * strain_rate
        transient_overstress = state.overstress - steady_overstress
        integral_q_squared = (
            steady_overstress**2 * float(dt)
            + 2.0
            * steady_overstress
            * transient_overstress
            * branch_times
            * (1.0 - decay.reshape(reshape))
            + 0.5
            * transient_overstress**2
            * branch_times
            * (1.0 - decay.reshape(reshape) ** 2)
        )
        dissipation = float(
            np.sum(integral_q_squared / (branch_moduli * branch_times))
        )
        if dissipation < -1.0e-12 * max(1.0, abs(dissipation)):
            raise RuntimeError("Generalized-Maxwell dissipation became negative.")
        dissipation = max(0.0, dissipation)
        return ViscoelasticUpdate(
            strain=selected.copy(),
            overstress=overstress,
            stress=np.asarray(stress),
            algorithmic_modulus=tangent,
            dissipated_energy_increment=dissipation,
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "generalized_maxwell",
            "equilibrium_modulus": self.equilibrium_modulus,
            "instantaneous_modulus": self.instantaneous_modulus,
            "branch_moduli": self.branch_moduli.tolist(),
            "relaxation_times": self.relaxation_times.tolist(),
            "prony_ratios": self.prony_ratios.tolist(),
            "shift": None if self.shift is None else self.shift.summary(),
        }


def standard_linear_solid(
    *,
    equilibrium_modulus: float,
    relaxing_modulus: float,
    relaxation_time: float,
    shift: WLFShift | ArrheniusShift | None = None,
    name: str = "standard_linear_solid",
) -> GeneralizedMaxwell:
    """Create a standard linear solid as one Maxwell branch in parallel."""

    return GeneralizedMaxwell(
        equilibrium_modulus=equilibrium_modulus,
        branch_moduli=np.asarray([relaxing_modulus]),
        relaxation_times=np.asarray([relaxation_time]),
        shift=shift,
        name=name,
    )


@dataclass(frozen=True)
class PronyFit:
    """Deterministic fixed-spectrum relaxation fit with validation evidence."""

    model: GeneralizedMaxwell
    predicted: np.ndarray
    residual: np.ndarray
    root_mean_square_error: float
    relative_root_mean_square_error: float
    nonnegative: bool

    def summary(self) -> dict[str, object]:
        return {
            "model": self.model.summary(),
            "root_mean_square_error": self.root_mean_square_error,
            "relative_root_mean_square_error": self.relative_root_mean_square_error,
            "nonnegative": self.nonnegative,
            "sample_count": int(self.predicted.size),
        }


def fit_relaxation_prony(
    time,
    modulus,
    relaxation_times,
    *,
    nonnegative: bool = True,
    name: str = "fitted_prony_series",
) -> PronyFit:
    """Fit a relaxation spectrum for user-declared relaxation times.

    The nonlinear choice of spectrum is kept explicit. Once those times are
    declared, equilibrium and branch moduli form a transparent linear problem.
    """

    time = np.asarray(time, dtype=float)
    measured = np.asarray(modulus, dtype=float)
    times = _positive_vector(relaxation_times, name="relaxation_times")
    if time.ndim != 1 or measured.shape != time.shape or time.size < times.size + 1:
        raise ValueError("time and modulus must be equal one-dimensional arrays with enough samples.")
    if np.any(time < 0.0) or np.any(measured <= 0.0):
        raise ValueError("Relaxation time must be nonnegative and modulus positive.")
    design = np.column_stack((np.ones(time.size), np.exp(-time[:, None] / times)))
    if nonnegative:
        from ..dependencies import require

        nnls = require(
            "scipy.optimize",
            extra="identification",
            capability="nonnegative Prony-spectrum fitting",
        ).nnls
        coefficients, _ = nnls(design, measured)
    else:
        coefficients, *_ = np.linalg.lstsq(design, measured, rcond=None)
    if coefficients[0] <= 0.0 or np.any(coefficients[1:] <= 0.0):
        raise ValueError("Fitted spectrum is not strictly positive; revise relaxation_times or data.")
    model = GeneralizedMaxwell(coefficients[0], coefficients[1:], times, name=name)
    predicted = model.relaxation_modulus(time)
    residual = predicted - measured
    rmse = float(np.sqrt(np.mean(residual**2)))
    scale = float(np.sqrt(np.mean(measured**2)))
    return PronyFit(model, predicted, residual, rmse, rmse / scale, nonnegative)


__all__ = [
    "ArrheniusShift",
    "GeneralizedMaxwell",
    "MaxwellState",
    "PronyFit",
    "ViscoelasticUpdate",
    "WLFShift",
    "fit_relaxation_prony",
    "standard_linear_solid",
]

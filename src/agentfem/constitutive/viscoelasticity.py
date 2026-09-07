"""Linear viscoelastic spectra with exact generalized-Maxwell updates."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import TYPE_CHECKING

import numpy as np

from .user_material import MaterialStateSchema, MaterialStateVariable

if TYPE_CHECKING:
    from .._material_history import GeneralizedMaxwellHistoryStep


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
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
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
            raise ValueError(
                "activation_energy and reference_temperature must be positive."
            )
        if self.gas_constant <= 0.0:
            raise ValueError("gas_constant must be positive.")

    def factor(self, temperature) -> np.ndarray:
        temperature = np.asarray(temperature, dtype=float)
        if np.any(temperature <= 0.0):
            raise ValueError("Arrhenius temperature must be absolute and positive.")
        exponent = (
            self.activation_energy
            / self.gas_constant
            * (1.0 / temperature - 1.0 / self.reference_temperature)
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
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
    mechanical_work_increment: float = 0.0


@dataclass(frozen=True)
class IsotropicMaxwellUpdate:
    """One exact tensor-valued generalized-Maxwell trial update."""

    strain: np.ndarray
    shear_overstress: np.ndarray
    bulk_overstress: np.ndarray
    stress: np.ndarray
    consistent_tangent: np.ndarray
    stored_energy_density: float
    dissipated_energy_increment: float
    mechanical_work_increment: float
    state_new: np.ndarray


@dataclass
class MaxwellState:
    """Committed state for a generalized-Maxwell material point."""

    strain: np.ndarray
    overstress: np.ndarray
    dissipated_energy: float = 0.0

    @classmethod
    def zero(cls, branch_count: int, *, value_shape=()) -> "MaxwellState":
        if isinstance(branch_count, (bool, np.bool_)):
            raise ValueError("branch_count must be a positive integer.")
        try:
            selected_count = int(branch_count.__index__())
        except (AttributeError, TypeError) as exc:
            raise ValueError("branch_count must be a positive integer.") from exc
        if selected_count <= 0:
            raise ValueError("branch_count must be positive.")
        shape = tuple(value_shape)
        return cls(
            strain=np.zeros(shape, dtype=float),
            overstress=np.zeros((selected_count, *shape), dtype=float),
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
        if (
            strain.shape != self.strain.shape
            or overstress.shape != self.overstress.shape
        ):
            raise ValueError("Viscoelastic snapshot shape does not match the state.")
        dissipated_energy = float(snapshot["dissipated_energy"])
        if (
            not np.all(np.isfinite(strain))
            or not np.all(np.isfinite(overstress))
            or not np.isfinite(dissipated_energy)
            or dissipated_energy < 0.0
        ):
            raise ValueError(
                "Viscoelastic snapshot must contain finite state and nonnegative dissipation."
            )
        self.strain[...] = strain
        self.overstress[...] = overstress
        self.dissipated_energy = dissipated_energy

    def commit(self, update: ViscoelasticUpdate) -> None:
        if (
            update.strain.shape != self.strain.shape
            or update.overstress.shape != self.overstress.shape
        ):
            raise ValueError("Viscoelastic update shape does not match the state.")
        if (
            not np.all(np.isfinite(update.strain))
            or not np.all(np.isfinite(update.overstress))
            or not np.all(np.isfinite(update.stress))
            or not np.isfinite(update.algorithmic_modulus)
            or update.algorithmic_modulus <= 0.0
            or not np.isfinite(update.dissipated_energy_increment)
            or update.dissipated_energy_increment < 0.0
            or not np.isfinite(update.mechanical_work_increment)
            or not np.isfinite(self.dissipated_energy)
            or self.dissipated_energy < 0.0
        ):
            raise ValueError(
                "Viscoelastic update must contain finite state, a positive tangent, "
                "and nonnegative dissipation."
            )
        self.strain[...] = update.strain
        self.overstress[...] = update.overstress
        self.dissipated_energy += float(update.dissipated_energy_increment)

    def copy(self) -> "MaxwellState":
        """Return a detached accepted state for a new procedure or branch."""

        copied = MaxwellState.zero(
            self.overstress.shape[0],
            value_shape=self.strain.shape,
        )
        copied.restore(self.snapshot())
        return copied


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
            raise ValueError(
                "branch_moduli and relaxation_times must have equal length."
            )
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
        if not np.isfinite(instantaneous_modulus) or instantaneous_modulus <= 0.0:
            raise ValueError("instantaneous_modulus must be finite and positive.")
        return cls(
            equilibrium_modulus=float(instantaneous_modulus)
            * (1.0 - float(np.sum(ratios))),
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
        with np.errstate(over="ignore", invalid="ignore"):
            shifted = self.relaxation_times * float(factor)
        if not np.all(np.isfinite(shifted)) or np.any(shifted <= 0.0):
            raise ValueError(
                "Shifted relaxation times must remain finite and positive."
            )
        return shifted

    def initial_state(
        self,
        strain=0.0,
        *,
        condition: str = "equilibrated",
    ) -> MaxwellState:
        """Create an accepted state with an explicit loading-history meaning.

        ``equilibrated`` means the strain has been held until every Maxwell
        branch relaxed. ``instantaneous`` means the strain was just applied,
        so every branch initially carries its elastic share.
        """

        selected_strain = np.asarray(strain, dtype=float)
        if not np.all(np.isfinite(selected_strain)):
            raise ValueError("initial strain must contain only finite values.")
        selected_condition = str(condition).strip().lower().replace("-", "_")
        if selected_condition not in {"equilibrated", "instantaneous"}:
            raise ValueError(
                "condition must be 'equilibrated' or 'instantaneous'."
            )
        state = MaxwellState.zero(
            self.branch_moduli.size,
            value_shape=selected_strain.shape,
        )
        state.strain[...] = selected_strain
        if selected_condition == "instantaneous":
            reshape = (self.branch_moduli.size,) + (1,) * selected_strain.ndim
            state.overstress[...] = self.branch_moduli.reshape(reshape) * selected_strain
        return state

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
            raise ValueError(
                "angular_frequency must contain finite nonnegative values."
            )
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

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
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
        reduced_increment = float(dt) / times
        decay = np.exp(-reduced_increment)
        one_minus_decay = -np.expm1(-reduced_increment)
        one_minus_decay_squared = -np.expm1(-2.0 * reduced_increment)
        integration = one_minus_decay / reduced_increment
        reshape = (self.branch_moduli.size,) + (1,) * selected.ndim
        increment = selected - state.strain
        overstress = (
            decay.reshape(reshape) * state.overstress
            + (self.branch_moduli * integration).reshape(reshape) * increment
        )
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
            * one_minus_decay.reshape(reshape)
            + 0.5
            * transient_overstress**2
            * branch_times
            * one_minus_decay_squared.reshape(reshape)
        )
        dissipation = float(np.sum(integral_q_squared / (branch_moduli * branch_times)))
        if not np.isfinite(dissipation):
            raise RuntimeError("Generalized-Maxwell dissipation became non-finite.")
        if dissipation < -1.0e-12 * max(1.0, abs(dissipation)):
            raise RuntimeError("Generalized-Maxwell dissipation became negative.")
        dissipation = max(0.0, dissipation)
        equilibrium_work = 0.5 * self.equilibrium_modulus * float(
            np.sum(selected**2 - state.strain**2)
        )
        branch_work = float(
            np.sum(
                strain_rate
                * (
                    steady_overstress * float(dt)
                    + transient_overstress
                    * branch_times
                    * one_minus_decay.reshape(reshape)
                )
            )
        )
        mechanical_work = equilibrium_work + branch_work
        if not np.isfinite(mechanical_work):
            raise RuntimeError("Generalized-Maxwell mechanical work became non-finite.")
        return ViscoelasticUpdate(
            strain=selected.copy(),
            overstress=overstress,
            stress=np.asarray(stress),
            algorithmic_modulus=tangent,
            dissipated_energy_increment=dissipation,
            mechanical_work_increment=mechanical_work,
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

    def history(
        self,
        time,
        strain,
        *,
        temperature=None,
        initial_state: MaxwellState | None = None,
        name: str = "generalized_maxwell_history",
    ) -> "GeneralizedMaxwellHistoryStep":
        """Create an inspectable material-history procedure.

        This is a constitutive material-point route, not a claim that a global
        tensor-valued viscoelastic FEM provider has been selected.
        """

        from .._material_history import GeneralizedMaxwellHistoryStep

        return GeneralizedMaxwellHistoryStep(
            material=self,
            time=time,
            strain=strain,
            temperature=temperature,
            initial_state=initial_state,
            name=name,
        )


def _nonnegative_vector(value, *, name: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or (size is not None and array.size != size):
        expected = "a one-dimensional array" if size is None else f"{size} values"
        raise ValueError(f"{name} must contain {expected}.")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values.")
    return array


def _isotropic_tangent(bulk_modulus: float, shear_modulus: float) -> np.ndarray:
    identity = np.eye(3)
    symmetric_identity = 0.5 * (
        np.einsum("ik,jl->ijkl", identity, identity)
        + np.einsum("il,jk->ijkl", identity, identity)
    )
    volumetric = np.einsum("ij,kl->ijkl", identity, identity)
    deviatoric = symmetric_identity - volumetric / 3.0
    return bulk_modulus * volumetric + 2.0 * shear_modulus * deviatoric


def _exact_overstress_update(old, modulus, relaxation_time, increment, dt):
    """Return branch stress, tangent factor, work and dissipation.

    ``old`` and ``increment`` may be scalars or tensors.  Their inner product
    is the ordinary Euclidean contraction appropriate to the stored physical
    tensor, not engineering-shear Voigt components.
    """

    if modulus == 0.0:
        return np.zeros_like(old), 0.0, 0.0, 0.0
    reduced = float(dt) / float(relaxation_time)
    decay = np.exp(-reduced)
    one_minus_decay = -np.expm1(-reduced)
    one_minus_decay_squared = -np.expm1(-2.0 * reduced)
    integration = one_minus_decay / reduced
    rate = np.asarray(increment, dtype=float) / float(dt)
    steady = float(modulus) * float(relaxation_time) * rate
    transient = np.asarray(old, dtype=float) - steady
    new = decay * old + float(modulus) * integration * increment
    integral = (
        steady**2 * float(dt)
        + 2.0 * steady * transient * float(relaxation_time) * one_minus_decay
        + 0.5
        * transient**2
        * float(relaxation_time)
        * one_minus_decay_squared
    )
    work = float(
        np.sum(
            rate
            * (
                steady * float(dt)
                + transient * float(relaxation_time) * one_minus_decay
            )
        )
    )
    return np.asarray(new), float(integration), work, float(np.sum(integral))


@dataclass(frozen=True)
class IsotropicGeneralizedMaxwell:
    """Small-strain isotropic generalized-Maxwell solid for global FEM.

    The constitutive split is

    ``sigma = K tr(epsilon) I + 2 G dev(epsilon)``.

    Shear and bulk relaxation branches remain separate, matching common
    Prony-series input while avoiding any engineering-shear ambiguity.
    """

    equilibrium_bulk_modulus: float
    equilibrium_shear_modulus: float
    shear_branch_moduli: object
    bulk_branch_moduli: object
    relaxation_times: object
    shift: WLFShift | ArrheniusShift | None = None
    name: str = "isotropic_generalized_maxwell"

    # Public procedure dispatch uses this structural declaration rather than
    # importing every built-in material class into the model layer.
    stateful_constitutive = True

    def __post_init__(self) -> None:
        bulk = float(self.equilibrium_bulk_modulus)
        shear = float(self.equilibrium_shear_modulus)
        times = _positive_vector(self.relaxation_times, name="relaxation_times")
        shear_branches = _nonnegative_vector(
            self.shear_branch_moduli,
            name="shear_branch_moduli",
            size=times.size,
        )
        bulk_branches = _nonnegative_vector(
            self.bulk_branch_moduli,
            name="bulk_branch_moduli",
            size=times.size,
        )
        if not np.isfinite(bulk) or not np.isfinite(shear) or bulk <= 0.0 or shear <= 0.0:
            raise ValueError("Equilibrium bulk and shear moduli must be finite and positive.")
        if np.any((shear_branches == 0.0) & (bulk_branches == 0.0)):
            raise ValueError("Every relaxation time must own a shear or bulk branch.")
        if self.shift is not None and not isinstance(self.shift, (WLFShift, ArrheniusShift)):
            raise TypeError("shift must be WLFShift, ArrheniusShift, or None.")
        if not str(self.name).strip():
            raise ValueError("Viscoelastic material name must not be empty.")
        object.__setattr__(self, "equilibrium_bulk_modulus", bulk)
        object.__setattr__(self, "equilibrium_shear_modulus", shear)
        object.__setattr__(self, "shear_branch_moduli", shear_branches.copy())
        object.__setattr__(self, "bulk_branch_moduli", bulk_branches.copy())
        object.__setattr__(self, "relaxation_times", times.copy())

    @classmethod
    def from_prony(
        cls,
        *,
        instantaneous_young_modulus: float,
        instantaneous_poisson_ratio: float,
        shear_relaxation_ratios,
        relaxation_times,
        bulk_relaxation_ratios=None,
        shift: WLFShift | ArrheniusShift | None = None,
        name: str = "isotropic_generalized_maxwell",
    ) -> "IsotropicGeneralizedMaxwell":
        """Build from instantaneous elasticity and normalized Prony ratios."""

        young = float(instantaneous_young_modulus)
        poisson = float(instantaneous_poisson_ratio)
        if not np.isfinite(young) or young <= 0.0:
            raise ValueError("instantaneous_young_modulus must be finite and positive.")
        if not np.isfinite(poisson) or not -1.0 < poisson < 0.5:
            raise ValueError("instantaneous_poisson_ratio must lie between -1 and 0.5.")
        times = _positive_vector(relaxation_times, name="relaxation_times")
        shear_ratios = _nonnegative_vector(
            shear_relaxation_ratios,
            name="shear_relaxation_ratios",
            size=times.size,
        )
        bulk_ratios = _nonnegative_vector(
            np.zeros(times.size) if bulk_relaxation_ratios is None else bulk_relaxation_ratios,
            name="bulk_relaxation_ratios",
            size=times.size,
        )
        if np.sum(shear_ratios) >= 1.0 or np.sum(bulk_ratios) >= 1.0:
            raise ValueError("The sum of each Prony-ratio family must be less than one.")
        instantaneous_shear = young / (2.0 * (1.0 + poisson))
        instantaneous_bulk = young / (3.0 * (1.0 - 2.0 * poisson))
        return cls(
            equilibrium_bulk_modulus=instantaneous_bulk * (1.0 - np.sum(bulk_ratios)),
            equilibrium_shear_modulus=instantaneous_shear * (1.0 - np.sum(shear_ratios)),
            shear_branch_moduli=instantaneous_shear * shear_ratios,
            bulk_branch_moduli=instantaneous_bulk * bulk_ratios,
            relaxation_times=times,
            shift=shift,
            name=name,
        )

    @property
    def branch_count(self) -> int:
        return int(self.relaxation_times.size)

    @property
    def instantaneous_bulk_modulus(self) -> float:
        return float(self.equilibrium_bulk_modulus + np.sum(self.bulk_branch_moduli))

    @property
    def instantaneous_shear_modulus(self) -> float:
        return float(self.equilibrium_shear_modulus + np.sum(self.shear_branch_moduli))

    @property
    def state_schema(self) -> MaterialStateSchema:
        return MaterialStateSchema(
            name=f"isotropic_generalized_maxwell_{self.branch_count}_branch",
            version="1.0.0",
            variables=(
                MaterialStateVariable("strain", shape=(3, 3), output_name="E_ACCEPTED"),
                MaterialStateVariable(
                    "shear_overstress",
                    shape=(self.branch_count, 3, 3),
                    output_name="S_MAXWELL",
                ),
                MaterialStateVariable(
                    "bulk_overstress",
                    shape=(self.branch_count,),
                    output_name="P_MAXWELL",
                ),
                MaterialStateVariable(
                    "dissipated_energy",
                    output_name="VDENER",
                    description="Cumulative viscous dissipation density.",
                ),
            ),
        )

    def shifted_relaxation_times(self, temperature=None) -> np.ndarray:
        if self.shift is None:
            if temperature is not None:
                raise ValueError("temperature requires a declared time-temperature shift law.")
            return self.relaxation_times.copy()
        if temperature is None:
            raise ValueError("A shifted viscoelastic material requires temperature.")
        shifted = self.relaxation_times * float(np.asarray(self.shift.factor(temperature)))
        if not np.all(np.isfinite(shifted)) or np.any(shifted <= 0.0):
            raise ValueError("Shifted relaxation times must remain finite and positive.")
        return shifted

    def relaxation_moduli(self, time, *, temperature=None) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(time, dtype=float)
        if not np.all(np.isfinite(selected)) or np.any(selected < 0.0):
            raise ValueError("time must contain finite nonnegative values.")
        times = self.shifted_relaxation_times(temperature)
        decay = np.exp(-selected[..., None] / times)
        bulk = self.equilibrium_bulk_modulus + np.sum(self.bulk_branch_moduli * decay, axis=-1)
        shear = self.equilibrium_shear_modulus + np.sum(self.shear_branch_moduli * decay, axis=-1)
        return bulk, shear

    def initial_state(self, strain=None, *, condition: str = "equilibrated") -> np.ndarray:
        """Return a schema-ordered state with an explicit prior-history meaning."""

        selected = np.zeros((3, 3)) if strain is None else np.asarray(strain, dtype=float)
        if selected.shape != (3, 3) or not np.all(np.isfinite(selected)):
            raise ValueError("initial strain must be a finite 3x3 tensor.")
        selected = 0.5 * (selected + selected.T)
        mode = str(condition).strip().lower().replace("-", "_")
        if mode not in {"equilibrated", "instantaneous"}:
            raise ValueError("condition must be 'equilibrated' or 'instantaneous'.")
        state = self.state_schema.initial_state()
        state[:9] = selected.reshape(-1)
        if mode == "instantaneous":
            trace = float(np.trace(selected))
            deviator = selected - trace * np.eye(3) / 3.0
            offset = 9
            count = self.branch_count * 9
            state[offset : offset + count] = (
                2.0 * self.shear_branch_moduli[:, None, None] * deviator
            ).reshape(-1)
            offset += count
            state[offset : offset + self.branch_count] = (
                self.bulk_branch_moduli * trace
            )
        return state

    def update(
        self,
        state_old,
        strain,
        dt: float,
        *,
        temperature=None,
    ) -> IsotropicMaxwellUpdate:
        """Return the exact response for a linear strain path over ``dt``."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
        state = self.state_schema.unpack(state_old)
        old_strain = np.asarray(state["strain"], dtype=float)
        selected = np.asarray(strain, dtype=float)
        if selected.shape != (3, 3) or not np.all(np.isfinite(selected)):
            raise ValueError("strain must be a finite 3x3 tensor.")
        selected = 0.5 * (selected + selected.T)
        old_strain = 0.5 * (old_strain + old_strain.T)
        shear_old = np.asarray(state["shear_overstress"], dtype=float)
        bulk_old = np.asarray(state["bulk_overstress"], dtype=float)
        times = self.shifted_relaxation_times(temperature)
        identity = np.eye(3)
        trace_old = float(np.trace(old_strain))
        trace_new = float(np.trace(selected))
        dev_old = old_strain - trace_old * identity / 3.0
        dev_new = selected - trace_new * identity / 3.0

        shear_new = np.empty_like(shear_old)
        bulk_new = np.empty_like(bulk_old)
        shear_tangent = self.equilibrium_shear_modulus
        bulk_tangent = self.equilibrium_bulk_modulus
        work = (
            self.equilibrium_shear_modulus
            * float(np.sum(dev_new**2 - dev_old**2))
            + 0.5
            * self.equilibrium_bulk_modulus
            * (trace_new**2 - trace_old**2)
        )
        dissipation = 0.0
        for index, relaxation_time in enumerate(times):
            shear_new[index], factor, branch_work, integral = _exact_overstress_update(
                shear_old[index],
                2.0 * self.shear_branch_moduli[index],
                relaxation_time,
                dev_new - dev_old,
                dt,
            )
            shear_tangent += self.shear_branch_moduli[index] * factor
            work += branch_work
            if self.shear_branch_moduli[index] > 0.0:
                dissipation += integral / (
                    2.0 * self.shear_branch_moduli[index] * relaxation_time
                )
            bulk_new[index], factor, branch_work, integral = _exact_overstress_update(
                np.asarray(bulk_old[index]),
                self.bulk_branch_moduli[index],
                relaxation_time,
                np.asarray(trace_new - trace_old),
                dt,
            )
            bulk_tangent += self.bulk_branch_moduli[index] * factor
            work += branch_work
            if self.bulk_branch_moduli[index] > 0.0:
                dissipation += integral / (
                    self.bulk_branch_moduli[index] * relaxation_time
                )

        stress = (
            self.equilibrium_bulk_modulus * trace_new * identity
            + 2.0 * self.equilibrium_shear_modulus * dev_new
            + np.sum(shear_new, axis=0)
            + float(np.sum(bulk_new)) * identity
        )
        stored = (
            self.equilibrium_shear_modulus * float(np.sum(dev_new**2))
            + 0.5 * self.equilibrium_bulk_modulus * trace_new**2
        )
        for index in range(self.branch_count):
            if self.shear_branch_moduli[index] > 0.0:
                stored += float(np.sum(shear_new[index] ** 2)) / (
                    4.0 * self.shear_branch_moduli[index]
                )
            if self.bulk_branch_moduli[index] > 0.0:
                stored += float(bulk_new[index] ** 2) / (
                    2.0 * self.bulk_branch_moduli[index]
                )
        state_new = self.state_schema.initial_state()
        state_new[:9] = selected.reshape(-1)
        offset = 9
        count = self.branch_count * 9
        state_new[offset : offset + count] = shear_new.reshape(-1)
        offset += count
        state_new[offset : offset + self.branch_count] = bulk_new
        state_new[-1] = float(state["dissipated_energy"]) + dissipation
        return IsotropicMaxwellUpdate(
            strain=selected,
            shear_overstress=shear_new,
            bulk_overstress=bulk_new,
            stress=stress,
            consistent_tangent=_isotropic_tangent(bulk_tangent, shear_tangent),
            stored_energy_density=float(stored),
            dissipated_energy_increment=float(dissipation),
            mechanical_work_increment=float(work),
            state_new=state_new,
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "isotropic_generalized_maxwell",
            "instantaneous_bulk_modulus": self.instantaneous_bulk_modulus,
            "instantaneous_shear_modulus": self.instantaneous_shear_modulus,
            "equilibrium_bulk_modulus": self.equilibrium_bulk_modulus,
            "equilibrium_shear_modulus": self.equilibrium_shear_modulus,
            "shear_branch_moduli": self.shear_branch_moduli.tolist(),
            "bulk_branch_moduli": self.bulk_branch_moduli.tolist(),
            "relaxation_times": self.relaxation_times.tolist(),
            "shift": None if self.shift is None else self.shift.summary(),
            "state_schema": self.state_schema.summary(),
        }

    def as_dict(self) -> dict[str, object]:
        """Return the stable material record used by results and restart."""

        return self.summary()


def isotropic_generalized_maxwell(**kwargs) -> IsotropicGeneralizedMaxwell:
    """Create an isotropic tensor Prony solid from instantaneous properties."""

    return IsotropicGeneralizedMaxwell.from_prony(**kwargs)


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
        raise ValueError(
            "time and modulus must be equal one-dimensional arrays with enough samples."
        )
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
        raise ValueError(
            "Fitted spectrum is not strictly positive; revise relaxation_times or data."
        )
    model = GeneralizedMaxwell(coefficients[0], coefficients[1:], times, name=name)
    predicted = model.relaxation_modulus(time)
    residual = predicted - measured
    rmse = float(np.sqrt(np.mean(residual**2)))
    scale = float(np.sqrt(np.mean(measured**2)))
    return PronyFit(model, predicted, residual, rmse, rmse / scale, nonnegative)


__all__ = [
    "ArrheniusShift",
    "GeneralizedMaxwell",
    "IsotropicGeneralizedMaxwell",
    "IsotropicMaxwellUpdate",
    "MaxwellState",
    "PronyFit",
    "ViscoelasticUpdate",
    "WLFShift",
    "fit_relaxation_prony",
    "isotropic_generalized_maxwell",
    "standard_linear_solid",
]

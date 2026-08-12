"""Experimental constitutive contracts for material interfaces.

The first public slice is deliberately independent of DOLFINx assembly.  It
defines the local, irreversible traction--separation response that a paired
facet element consumes.  Keeping this material-point contract separate from
mesh topology and time integration makes its energy, rollback, and restart
semantics testable before it is used in a dynamic fracture calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class CohesiveResponse:
    """One Mode-I traction--separation update.

    ``traction`` is positive in opening and negative in compression.
    ``stored_energy`` and ``dissipated_energy`` are energies per undeformed
    interface area.  The latter is cumulative and irreversible.
    """

    opening: np.ndarray
    traction: np.ndarray
    tangent: np.ndarray
    maximum_opening: np.ndarray
    damage: np.ndarray
    stored_energy: np.ndarray
    dissipated_energy: np.ndarray


@dataclass(frozen=True)
class VectorCohesiveResponse:
    """Local-basis response of a two- or three-dimensional interface.

    Component zero is normal to the interface; the remaining components are
    tangential.  Keeping this material-point object in a local orthonormal
    basis makes one law reusable for line interfaces, surfaces, serial
    assembly and MPI owner assembly.
    """

    jump: np.ndarray
    traction: np.ndarray
    tangent: np.ndarray
    maximum_effective_separation: np.ndarray
    damage: np.ndarray
    stored_energy: np.ndarray
    dissipated_energy: np.ndarray
    mode_mixity: np.ndarray


@dataclass(frozen=True)
class MixedModeBilinearCohesiveLaw:
    """Bilinear mixed-mode cohesive law for proportional loading paths.

    Damage initiation uses a quadratic nominal-traction interaction.  The
    mixed fracture energy is selected by either a Benzeggagh--Kenane (BK) or
    power-law interaction.  The mode mix at first damage initiation is frozen
    in the material state; this gives an objective, restartable law with an
    analytical algorithmic tangent for proportional and mildly changing
    loading paths.  General non-proportional mixed-mode fatigue is outside
    this first contract.

    Compression is carried by an independent penalty and never initiates or
    advances damage.  Optional regularized Coulomb resistance is explicit in
    the summary because it is a smooth contact approximation, not a hidden
    stabilization.
    """

    normal_strength: float
    shear_strength: float
    normal_fracture_energy: float
    shear_fracture_energy: float
    normal_stiffness: float
    tangential_stiffness: float
    interaction: str = "bk"
    interaction_exponent: float = 1.45
    compression_stiffness: float | None = None
    residual_tangential_fraction: float = 0.0
    friction_coefficient: float = 0.0
    friction_regularization: float = 1.0e-8
    name: str = "bilinear mixed-mode cohesive law"

    def __post_init__(self) -> None:
        positive = (
            "normal_strength",
            "shear_strength",
            "normal_fracture_energy",
            "shear_fracture_energy",
            "normal_stiffness",
            "tangential_stiffness",
            "interaction_exponent",
            "friction_regularization",
        )
        for field_name in positive:
            value = float(getattr(self, field_name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        selected = str(self.interaction).strip().lower().replace("-", "_")
        if selected not in {"bk", "power"}:
            raise ValueError("interaction must be 'bk' or 'power'.")
        object.__setattr__(self, "interaction", selected)
        closure = (
            self.normal_stiffness
            if self.compression_stiffness is None
            else float(self.compression_stiffness)
        )
        if not isfinite(closure) or closure <= 0.0:
            raise ValueError("compression_stiffness must be finite and positive.")
        object.__setattr__(self, "compression_stiffness", closure)
        residual = float(self.residual_tangential_fraction)
        if not isfinite(residual) or not 0.0 <= residual <= 1.0:
            raise ValueError("residual_tangential_fraction must lie in [0, 1].")
        friction = float(self.friction_coefficient)
        if not isfinite(friction) or friction < 0.0:
            raise ValueError("friction_coefficient must be finite and nonnegative.")
        # Both pure-mode triangular envelopes must have a descending branch.
        checks = (
            (
                self.normal_strength,
                self.normal_fracture_energy,
                self.normal_stiffness,
                "normal",
            ),
            (
                self.shear_strength,
                self.shear_fracture_energy,
                self.tangential_stiffness,
                "shear",
            ),
        )
        for strength, energy, stiffness, label in checks:
            if 2.0 * energy / strength <= strength / stiffness:
                raise ValueError(
                    f"The {label} fracture energy is too small for the declared "
                    "strength and stiffness."
                )

    def transaction(self, size: int) -> "MixedModeCohesiveTransaction":
        return MixedModeCohesiveTransaction(self, size)

    def mixed_fracture_energy(self, shear_fraction) -> np.ndarray:
        beta = np.clip(_finite_array(shear_fraction, name="shear_fraction"), 0.0, 1.0)
        gi = float(self.normal_fracture_energy)
        gii = float(self.shear_fracture_energy)
        exponent = float(self.interaction_exponent)
        if self.interaction == "bk":
            return gi + (gii - gi) * beta**exponent
        inverse = ((1.0 - beta) / gi) ** exponent + (beta / gii) ** exponent
        return inverse ** (-1.0 / exponent)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": "cohesive_traction_separation",
            "mode": "mixed",
            "envelope": "bilinear",
            "damage_initiation": "quadratic_nominal_traction",
            "damage_evolution": "energy",
            "mixed_mode_interaction": self.interaction,
            "interaction_exponent": self.interaction_exponent,
            "normal_strength": self.normal_strength,
            "shear_strength": self.shear_strength,
            "normal_fracture_energy": self.normal_fracture_energy,
            "shear_fracture_energy": self.shear_fracture_energy,
            "normal_stiffness": self.normal_stiffness,
            "tangential_stiffness": self.tangential_stiffness,
            "total_initial_tangential_stiffness": (
                (1.0 + self.residual_tangential_fraction)
                * self.tangential_stiffness
            ),
            "compression_stiffness": self.compression_stiffness,
            "residual_tangential_fraction": self.residual_tangential_fraction,
            "residual_tangential_role": "parallel_nonfracturing_penalty",
            "friction": (
                "none"
                if self.friction_coefficient == 0.0
                else "regularized_coulomb_contact"
            ),
            "friction_coefficient": self.friction_coefficient,
            "state": [
                "maximum_effective_separation",
                "initiation_separation",
                "failure_separation",
                "initiation_stiffness",
            ],
            "path_scope": "proportional_or_mildly_changing_mode_mix",
            "maturity": "experimental_material_point",
        }


@dataclass(frozen=True)
class BilinearCohesiveLaw:
    """Irreversible bilinear Mode-I cohesive law.

    The virgin response reaches ``strength`` at ``peak_opening`` and then
    softens linearly to zero at ``failure_opening``.  Consequently, the exact
    area under the monotonic envelope is ``fracture_energy``.

    Unloading and reloading are secant-linear through the origin.  Compressive
    closure uses a separate penalty stiffness and does not heal or advance
    tensile damage.
    """

    strength: float
    fracture_energy: float
    initial_stiffness: float
    compression_stiffness: float | None = None
    name: str = "bilinear Mode-I cohesive law"

    def __post_init__(self) -> None:
        for field_name in ("strength", "fracture_energy", "initial_stiffness"):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        if self.compression_stiffness is not None and (
            not isfinite(float(self.compression_stiffness))
            or float(self.compression_stiffness) <= 0.0
        ):
            raise ValueError("compression_stiffness must be finite and positive.")
        if self.failure_opening <= self.peak_opening:
            minimum = self.strength**2 / (2.0 * self.initial_stiffness)
            raise ValueError(
                "The requested fracture energy is too small for the declared "
                "strength and initial stiffness. It must exceed "
                f"strength**2/(2*initial_stiffness) = {minimum:.16g}."
            )

    @property
    def peak_opening(self) -> float:
        """Opening at peak traction."""

        return float(self.strength / self.initial_stiffness)

    @property
    def failure_opening(self) -> float:
        """Opening at complete tensile decohesion."""

        return float(2.0 * self.fracture_energy / self.strength)

    @property
    def closure_stiffness(self) -> float:
        return float(
            self.initial_stiffness
            if self.compression_stiffness is None
            else self.compression_stiffness
        )

    def characteristic_length(self, elastic_modulus: float) -> float:
        """Return the declared cohesive length ``E*Gamma/strength^2``."""

        modulus = float(elastic_modulus)
        if not isfinite(modulus) or modulus <= 0.0:
            raise ValueError("elastic_modulus must be finite and positive.")
        return modulus * self.fracture_energy / self.strength**2

    def transaction(self, size: int) -> "CohesiveTransaction":
        """Create this law's trial/commit state container."""

        return CohesiveTransaction(self, size)

    def envelope_traction(self, opening) -> np.ndarray:
        """Return the monotonic tensile envelope traction."""

        value = _finite_array(opening, name="opening")
        positive = np.maximum(value, 0.0)
        d0 = self.peak_opening
        df = self.failure_opening
        traction = np.where(
            positive <= d0,
            self.initial_stiffness * positive,
            self.strength * np.maximum(df - positive, 0.0) / (df - d0),
        )
        return np.where(positive >= df, 0.0, traction)

    def envelope_work(self, opening) -> np.ndarray:
        """Integrate the monotonic envelope exactly up to ``opening``."""

        value = _finite_array(opening, name="opening")
        selected = np.clip(value, 0.0, self.failure_opening)
        d0 = self.peak_opening
        df = self.failure_opening
        elastic = 0.5 * self.initial_stiffness * selected**2
        softening = (
            0.5 * self.strength * d0
            + self.strength
            / (df - d0)
            * (df * (selected - d0) - 0.5 * (selected**2 - d0**2))
        )
        work = np.where(selected <= d0, elastic, softening)
        return np.where(value >= df, self.fracture_energy, work)

    def damage_from_maximum(self, maximum_opening) -> np.ndarray:
        """Return the secant damage associated with a maximum opening."""

        maximum = np.maximum(
            _finite_array(maximum_opening, name="maximum_opening"), 0.0
        )
        d0 = self.peak_opening
        df = self.failure_opening
        denominator = np.maximum(maximum * (df - d0), np.finfo(float).tiny)
        softening = df * (maximum - d0) / denominator
        return np.where(
            maximum <= d0,
            0.0,
            np.where(maximum >= df, 1.0, np.clip(softening, 0.0, 1.0)),
        )

    def update(self, opening, committed_maximum=0.0) -> CohesiveResponse:
        """Evaluate a trial state without mutating committed history."""

        value = _finite_array(opening, name="opening")
        committed = _finite_array(committed_maximum, name="committed_maximum")
        value, committed = np.broadcast_arrays(value, committed)
        if np.any(committed < 0.0):
            raise ValueError("committed_maximum cannot be negative.")

        maximum = np.maximum(committed, np.maximum(value, 0.0))
        damage = self.damage_from_maximum(maximum)
        tensile_stiffness = (1.0 - damage) * self.initial_stiffness
        is_compression = value < 0.0
        is_new_loading = value > committed
        traction = np.where(
            is_compression,
            self.closure_stiffness * value,
            tensile_stiffness * value,
        )
        envelope_tangent = np.where(
            value <= self.peak_opening,
            self.initial_stiffness,
            np.where(
                value < self.failure_opening,
                -self.strength / (self.failure_opening - self.peak_opening),
                0.0,
            ),
        )
        tangent = np.where(
            is_compression,
            self.closure_stiffness,
            np.where(is_new_loading, envelope_tangent, tensile_stiffness),
        )
        tensile_opening = np.maximum(value, 0.0)
        stored = np.where(
            is_compression,
            0.5 * self.closure_stiffness * value**2,
            0.5 * tensile_stiffness * tensile_opening**2,
        )

        envelope = self.envelope_traction(maximum)
        maximum_stored = 0.5 * envelope * maximum
        dissipated = np.maximum(
            self.envelope_work(maximum) - maximum_stored,
            0.0,
        )
        return CohesiveResponse(
            opening=np.array(value, dtype=float, copy=True),
            traction=np.asarray(traction, dtype=float),
            tangent=np.asarray(tangent, dtype=float),
            maximum_opening=np.asarray(maximum, dtype=float),
            damage=np.asarray(damage, dtype=float),
            stored_energy=np.asarray(stored, dtype=float),
            dissipated_energy=np.asarray(dissipated, dtype=float),
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": "cohesive_traction_separation",
            "mode": "normal",
            "envelope": "bilinear",
            "strength": self.strength,
            "fracture_energy": self.fracture_energy,
            "initial_stiffness": self.initial_stiffness,
            "compression_stiffness": self.closure_stiffness,
            "peak_opening": self.peak_opening,
            "failure_opening": self.failure_opening,
            "characteristic_length_definition": "E*Gamma/strength^2",
            "state": ["maximum_opening", "damage", "dissipated_energy"],
            "maturity": "experimental_material_point",
        }


class CohesiveTransaction:
    """Trial/commit/rollback state for a batch of cohesive points."""

    def __init__(self, law: BilinearCohesiveLaw, size: int):
        if int(size) <= 0:
            raise ValueError("CohesiveTransaction.size must be positive.")
        self.law = law
        self._committed_maximum = np.zeros(int(size), dtype=float)
        self._trial: CohesiveResponse | None = None

    @property
    def size(self) -> int:
        return int(self._committed_maximum.size)

    @property
    def committed_maximum(self) -> np.ndarray:
        return self._committed_maximum.copy()

    @property
    def trial(self) -> CohesiveResponse | None:
        return self._trial

    def begin(self, opening) -> CohesiveResponse:
        """Create a replaceable trial state from committed history."""

        values = _finite_array(opening, name="opening")
        if values.shape != self._committed_maximum.shape:
            raise ValueError(
                f"opening must have shape {self._committed_maximum.shape}, "
                f"got {values.shape}."
            )
        self._trial = self.evaluate(values)
        return self._trial

    def evaluate(self, opening) -> CohesiveResponse:
        """Evaluate from committed state without creating a transaction."""

        return self.law.update(opening, self._committed_maximum)

    def commit(self) -> None:
        if self._trial is None:
            raise RuntimeError("No cohesive trial state is available to commit.")
        self._committed_maximum[:] = self._trial.maximum_opening
        self._trial = None

    def rollback(self) -> None:
        """Discard the trial state without changing committed history."""

        self._trial = None

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "agentfem.cohesive-state.v1",
            "law": self.law.summary(),
            "maximum_opening": self._committed_maximum.tolist(),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("schema") != "agentfem.cohesive-state.v1":
            raise ValueError("Unsupported cohesive-state schema.")
        values = _finite_array(snapshot.get("maximum_opening"), name="maximum_opening")
        if values.shape != self._committed_maximum.shape:
            raise ValueError("Cohesive-state size does not match this transaction.")
        if np.any(values < 0.0):
            raise ValueError("Cohesive maximum opening cannot be negative.")
        self._committed_maximum[:] = values
        self._trial = None

    def initialize(self, maximum_opening) -> None:
        """Set an initial intact or pre-debonded state before execution."""

        if self._trial is not None:
            raise RuntimeError("Rollback the cohesive trial state before initialization.")
        values = _finite_array(maximum_opening, name="maximum_opening")
        values = np.broadcast_to(values, self._committed_maximum.shape)
        if np.any(values < 0.0):
            raise ValueError("Initial cohesive maximum opening cannot be negative.")
        self._committed_maximum[:] = values

    def state_arrays(self) -> dict[str, np.ndarray]:
        """Return checkpoint-ready committed arrays."""

        return {"maximum_opening": self._committed_maximum.copy()}

    def restore_state_arrays(self, arrays: dict[str, object]) -> None:
        """Restore checkpoint arrays through the same validation as initialize."""

        if set(arrays) != {"maximum_opening"}:
            raise ValueError("Cohesive-state fields differ from this law.")
        self.initialize(arrays["maximum_opening"])


@dataclass(frozen=True)
class _MixedModeTrialState:
    """Internal constitutive state resolved before traction evaluation."""

    normal: np.ndarray
    opening: np.ndarray
    closure: np.ndarray
    tangential: np.ndarray
    shear_norm: np.ndarray
    effective: np.ndarray
    mode_mixity: np.ndarray
    initiation_separation: np.ndarray
    failure_separation: np.ndarray
    initiation_stiffness: np.ndarray
    maximum: np.ndarray
    damage: np.ndarray
    damage_gradient: np.ndarray

    @property
    def intact(self) -> np.ndarray:
        return 1.0 - self.damage

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        return {
            "maximum_effective_separation": self.maximum,
            "initiation_separation": self.initiation_separation,
            "failure_separation": self.failure_separation,
            "initiation_stiffness": self.initiation_stiffness,
        }


def _mixed_mode_trial_state(law, values, committed) -> _MixedModeTrialState:
    """Resolve initiation, frozen envelope, damage, and its trial gradient."""

    normal = values[:, 0]
    opening = np.maximum(normal, 0.0)
    closure = np.minimum(normal, 0.0)
    tangential = values[:, 1:]
    shear_norm = np.linalg.norm(tangential, axis=1)
    effective = np.sqrt(opening**2 + shear_norm**2)
    elastic_measure = (
        law.normal_stiffness * opening**2
        + law.tangential_stiffness * shear_norm**2
    )
    quotient = np.sqrt(
        (law.normal_stiffness * opening / law.normal_strength) ** 2
        + (law.tangential_stiffness * shear_norm / law.shear_strength) ** 2
    )
    tiny = np.finfo(float).tiny
    candidate_d0 = np.divide(
        effective,
        quotient,
        out=np.zeros_like(effective),
        where=quotient > tiny,
    )
    candidate_k = np.divide(
        elastic_measure,
        effective**2,
        out=np.full_like(effective, law.normal_stiffness),
        where=effective > tiny,
    )
    mode_mixity = np.divide(
        law.tangential_stiffness * shear_norm**2,
        elastic_measure,
        out=np.zeros_like(effective),
        where=elastic_measure > tiny,
    )
    candidate_gc = law.mixed_fracture_energy(mode_mixity)
    candidate_df = np.divide(
        2.0 * candidate_gc,
        candidate_k * candidate_d0,
        out=np.zeros_like(effective),
        where=(candidate_k * candidate_d0) > tiny,
    )

    d0 = committed["initiation_separation"].copy()
    df = committed["failure_separation"].copy()
    k0 = committed["initiation_stiffness"].copy()
    first_initiation = (d0 == 0.0) & (quotient >= 1.0)
    if np.any(candidate_df[first_initiation] <= candidate_d0[first_initiation]):
        raise ValueError(
            "Mixed-mode fracture energy gives failure before damage initiation."
        )
    d0[first_initiation] = candidate_d0[first_initiation]
    df[first_initiation] = candidate_df[first_initiation]
    k0[first_initiation] = candidate_k[first_initiation]
    maximum = np.maximum(committed["maximum_effective_separation"], effective)
    initiated = d0 > 0.0
    denominator = np.maximum(maximum * (df - d0), tiny)
    damage = np.where(
        initiated & (maximum > d0),
        np.where(
            maximum >= df,
            1.0,
            np.clip(df * (maximum - d0) / denominator, 0.0, 1.0),
        ),
        0.0,
    )
    evolving = (
        initiated
        & (effective > committed["maximum_effective_separation"])
        & (maximum > d0)
        & (maximum < df)
        & (effective > tiny)
    )
    slope = np.zeros_like(effective)
    slope[evolving] = (
        df[evolving]
        * d0[evolving]
        / (df[evolving] - d0[evolving])
        / maximum[evolving] ** 2
    )
    damage_gradient = np.zeros_like(values)
    damage_gradient[:, 0] = slope * np.divide(
        opening,
        effective,
        out=np.zeros_like(opening),
        where=(normal > 0.0) & (effective > tiny),
    )
    damage_gradient[:, 1:] = (
        np.divide(
            slope,
            effective,
            out=np.zeros_like(slope),
            where=effective > tiny,
        )[:, None]
        * tangential
    )
    return _MixedModeTrialState(
        normal=normal,
        opening=opening,
        closure=closure,
        tangential=tangential,
        shear_norm=shear_norm,
        effective=effective,
        mode_mixity=mode_mixity,
        initiation_separation=d0,
        failure_separation=df,
        initiation_stiffness=k0,
        maximum=maximum,
        damage=damage,
        damage_gradient=damage_gradient,
    )


def _add_regularized_contact_friction(law, trial, traction, tangent) -> None:
    """Add the declared smooth compression-friction contribution in place."""

    if law.friction_coefficient == 0.0:
        return
    active = trial.normal < 0.0
    pressure = np.where(
        active, -law.compression_stiffness * trial.normal, 0.0
    )
    tiny = np.finfo(float).tiny
    unit = np.divide(
        trial.tangential,
        trial.shear_norm[:, None],
        out=np.zeros_like(trial.tangential),
        where=trial.shear_norm[:, None] > tiny,
    )
    argument = trial.shear_norm / law.friction_regularization
    saturation = np.tanh(argument)
    friction = (
        law.friction_coefficient
        * pressure
        * trial.damage
        * saturation
    )
    traction[:, 1:] += friction[:, None] * unit
    tangent[:, 1:, 0] += (
        -law.friction_coefficient
        * law.compression_stiffness
        * trial.damage
        * saturation
        * active
    )[:, None] * unit
    dimension = traction.shape[1]
    for point in range(traction.shape[0]):
        if not active[point] or trial.shear_norm[point] <= tiny:
            continue
        direction = unit[point]
        projector = np.eye(dimension - 1) - np.outer(direction, direction)
        derivative = (
            saturation[point] / trial.shear_norm[point] * projector
            + (1.0 - saturation[point] ** 2)
            / law.friction_regularization
            * np.outer(direction, direction)
        )
        tangent[point, 1:, 1:] += (
            law.friction_coefficient
            * pressure[point]
            * (
                trial.damage[point] * derivative
                + saturation[point]
                * np.outer(direction, trial.damage_gradient[point, 1:])
            )
        )


class MixedModeCohesiveTransaction:
    """Trial/commit state for :class:`MixedModeBilinearCohesiveLaw`."""

    _FIELDS = (
        "maximum_effective_separation",
        "initiation_separation",
        "failure_separation",
        "initiation_stiffness",
    )

    def __init__(self, law: MixedModeBilinearCohesiveLaw, size: int):
        if int(size) <= 0:
            raise ValueError("MixedModeCohesiveTransaction.size must be positive.")
        self.law = law
        self._state = {
            name: np.zeros(int(size), dtype=float) for name in self._FIELDS
        }
        self._trial: VectorCohesiveResponse | None = None
        self._trial_state: dict[str, np.ndarray] | None = None

    @property
    def size(self) -> int:
        return int(self._state["maximum_effective_separation"].size)

    @property
    def trial(self) -> VectorCohesiveResponse | None:
        return self._trial

    @property
    def committed_maximum(self) -> np.ndarray:
        return self._state["maximum_effective_separation"].copy()

    def _evaluate(self, jump) -> tuple[VectorCohesiveResponse, dict[str, np.ndarray]]:
        values = _finite_array(jump, name="jump")
        if values.ndim != 2 or values.shape[0] != self.size or values.shape[1] < 2:
            raise ValueError(
                "mixed-mode jump must have shape (transaction size, 2 or 3)."
            )
        law = self.law
        trial = _mixed_mode_trial_state(law, values, self._state)
        # The residual branch is a genuine parallel elastic penalty. It is
        # present from the virgin state, never participates in damage
        # initiation/evolution, and therefore stores rather than dissipates
        # its work. ``tangential_stiffness`` remains the damageable branch.
        shear_scale = trial.intact + law.residual_tangential_fraction

        traction = np.zeros_like(values)
        traction[:, 0] = np.where(
            trial.normal < 0.0,
            law.compression_stiffness * trial.normal,
            trial.intact * law.normal_stiffness * trial.normal,
        )
        traction[:, 1:] = (
            shear_scale * law.tangential_stiffness
        )[:, None] * trial.tangential

        dimension = values.shape[1]
        tangent = np.zeros((self.size, dimension, dimension), dtype=float)
        tangent[:, 0, 0] = np.where(
            trial.normal < 0.0,
            law.compression_stiffness,
            trial.intact * law.normal_stiffness,
        )
        tangential_indices = np.arange(1, dimension)
        tangent[:, tangential_indices, tangential_indices] = (
            shear_scale * law.tangential_stiffness
        )[:, None]

        traction_damage_derivative = np.zeros_like(values)
        traction_damage_derivative[:, 0] = (
            -law.normal_stiffness * trial.opening
        )
        traction_damage_derivative[:, 1:] = -(
            law.tangential_stiffness * trial.tangential
        )
        tangent += np.einsum(
            "fi,fj->fij", traction_damage_derivative, trial.damage_gradient
        )
        _add_regularized_contact_friction(law, trial, traction, tangent)

        stored = (
            0.5 * trial.intact * law.normal_stiffness * trial.opening**2
            + 0.5
            * shear_scale
            * law.tangential_stiffness
            * trial.shear_norm**2
            + 0.5 * law.compression_stiffness * trial.closure**2
        )
        tiny = np.finfo(float).tiny
        capped = np.minimum(trial.maximum, trial.failure_separation)
        elastic_work = 0.5 * trial.initiation_stiffness * capped**2
        softening_work = (
            0.5
            * trial.initiation_stiffness
            * trial.initiation_separation**2
            + trial.initiation_stiffness
            * trial.initiation_separation
            / np.maximum(
                trial.failure_separation - trial.initiation_separation,
                tiny,
            )
            * (
                trial.failure_separation
                * (capped - trial.initiation_separation)
                - 0.5 * (capped**2 - trial.initiation_separation**2)
            )
        )
        envelope_work = np.where(
            capped <= trial.initiation_separation,
            elastic_work,
            softening_work,
        )
        maximum_stored = (
            0.5
            * trial.intact
            * trial.initiation_stiffness
            * trial.maximum**2
        )
        dissipated = np.where(
            trial.initiation_separation > 0.0,
            np.maximum(envelope_work - maximum_stored, 0.0),
            0.0,
        )
        response = VectorCohesiveResponse(
            jump=values.copy(),
            traction=traction,
            tangent=tangent,
            maximum_effective_separation=trial.maximum,
            damage=trial.damage,
            stored_energy=stored,
            dissipated_energy=dissipated,
            mode_mixity=trial.mode_mixity,
        )
        return response, trial.checkpoint_arrays()

    def evaluate(self, jump) -> VectorCohesiveResponse:
        return self._evaluate(jump)[0]

    def begin(self, jump) -> VectorCohesiveResponse:
        self._trial, self._trial_state = self._evaluate(jump)
        return self._trial

    def commit(self) -> None:
        if self._trial is None or self._trial_state is None:
            raise RuntimeError("No mixed-mode cohesive trial is available to commit.")
        for name in self._FIELDS:
            self._state[name][:] = self._trial_state[name]
        self._trial = None
        self._trial_state = None

    def rollback(self) -> None:
        self._trial = None
        self._trial_state = None

    def initialize(self, maximum_effective_separation) -> None:
        if self._trial is not None:
            raise RuntimeError(
                "Rollback the mixed-mode cohesive trial before initialization."
            )
        values = _finite_array(
            maximum_effective_separation,
            name="maximum_effective_separation",
        )
        values = np.broadcast_to(values, (self.size,))
        if np.any(values < 0.0):
            raise ValueError("Initial effective separation cannot be negative.")
        if np.any(values > 0.0):
            raise ValueError(
                "A positive mixed-mode separation does not determine the "
                "initiation mode or failure envelope. Use initialize_failed() "
                "for a precrack or restore_state_arrays() for a complete state."
            )
        for field in self._FIELDS:
            self._state[field].fill(0.0)

    def initialize_failed(self, mask) -> None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (self.size,):
            raise ValueError("Mixed-mode precrack mask has an invalid shape.")
        d0 = self.law.normal_strength / self.law.normal_stiffness
        df = 2.0 * self.law.normal_fracture_energy / self.law.normal_strength
        self._state["maximum_effective_separation"][selected] = df
        self._state["initiation_separation"][selected] = d0
        self._state["failure_separation"][selected] = df
        self._state["initiation_stiffness"][selected] = self.law.normal_stiffness

    def state_arrays(self) -> dict[str, np.ndarray]:
        return {name: values.copy() for name, values in self._state.items()}

    def restore_state_arrays(self, arrays: dict[str, object]) -> None:
        if set(arrays) != set(self._FIELDS):
            raise ValueError("Mixed-mode cohesive-state fields differ.")
        restored = {}
        for name in self._FIELDS:
            values = _finite_array(arrays[name], name=name)
            if values.shape != (self.size,) or np.any(values < 0.0):
                raise ValueError(f"Mixed-mode state field {name!r} is invalid.")
            restored[name] = values
        initiated = restored["initiation_separation"] > 0.0
        uninitiated = ~initiated
        # A point may carry a positive elastic maximum below initiation while
        # the frozen damage envelope remains unset. Only a partial envelope is
        # invalid; rejecting the elastic maximum would make an ordinary
        # pre-initiation checkpoint impossible to restore.
        if np.any(restored["failure_separation"][uninitiated] > 0.0) or np.any(
            restored["initiation_stiffness"][uninitiated] > 0.0
        ):
            raise ValueError(
                "Mixed-mode checkpoint contains an incomplete uninitiated state."
            )
        if np.any(
            restored["failure_separation"][initiated]
            <= restored["initiation_separation"][initiated]
        ):
            raise ValueError("Mixed-mode checkpoint contains an invalid envelope.")
        if np.any(restored["initiation_stiffness"][initiated] <= 0.0) or np.any(
            restored["maximum_effective_separation"][initiated]
            < restored["initiation_separation"][initiated]
        ):
            raise ValueError("Mixed-mode checkpoint contains an incomplete envelope.")
        for name in self._FIELDS:
            self._state[name][:] = restored[name]
        self.rollback()

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "agentfem.mixed-mode-cohesive-state.v1",
            "law": self.law.summary(),
            "state": {name: values.tolist() for name, values in self._state.items()},
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("schema") != "agentfem.mixed-mode-cohesive-state.v1":
            raise ValueError("Unsupported mixed-mode cohesive-state schema.")
        if snapshot.get("law") != self.law.summary():
            raise ValueError("Mixed-mode cohesive law differs from checkpoint.")
        self.restore_state_arrays(snapshot.get("state", {}))


@dataclass(frozen=True)
class PairedLineFacets:
    """Deterministically paired zero-thickness line facets for a 2D mesh.

    Positive-side node order is permuted to match the negative-side geometry.
    ``normals`` point in the caller-declared direction from the negative side
    toward the positive side.  Coincident geometry alone cannot infer that
    direction, hence ``normal_hint`` is mandatory in the constructor helper.
    """

    negative_nodes: np.ndarray
    positive_nodes: np.ndarray
    normals: np.ndarray
    lengths: np.ndarray
    tolerance: float
    facet_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        negative = np.asarray(self.negative_nodes, dtype=np.int64)
        positive = np.asarray(self.positive_nodes, dtype=np.int64)
        normals = np.asarray(self.normals, dtype=float)
        lengths = np.asarray(self.lengths, dtype=float).reshape(-1)
        tolerance = float(self.tolerance)
        if negative.ndim != 2 or negative.shape[1] != 2:
            raise ValueError("PairedLineFacets negative_nodes must have shape (facets, 2).")
        if positive.shape != negative.shape:
            raise ValueError("PairedLineFacets positive_nodes must match negative_nodes.")
        if normals.shape[0] != negative.shape[0] or normals.ndim != 2:
            raise ValueError("PairedLineFacets normals must provide one vector per facet.")
        if (
            lengths.shape != (negative.shape[0],)
            or np.any(~np.isfinite(lengths))
            or np.any(lengths <= 0.0)
        ):
            raise ValueError("PairedLineFacets lengths must be positive per facet.")
        if np.any(~np.isfinite(normals)):
            raise ValueError("PairedLineFacets normals must be finite.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("PairedLineFacets tolerance must be finite and positive.")
        keys = tuple(str(value) for value in self.facet_keys)
        if keys and (len(keys) != negative.shape[0] or len(set(keys)) != len(keys)):
            raise ValueError("PairedLineFacets facet_keys must be unique per facet.")
        object.__setattr__(self, "negative_nodes", negative.copy())
        object.__setattr__(self, "positive_nodes", positive.copy())
        object.__setattr__(self, "normals", normals.copy())
        object.__setattr__(self, "lengths", lengths.copy())
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "facet_keys", keys)

    @property
    def number_of_facets(self) -> int:
        return int(self.negative_nodes.shape[0])

    @property
    def number_of_points(self) -> int:
        return 2 * self.number_of_facets

    @property
    def quadrature_points_per_facet(self) -> int:
        return 2

    @property
    def measures(self) -> np.ndarray:
        """Reference line measure per facet."""

        return self.lengths.copy()

    def summary(self) -> dict[str, object]:
        return {
            "kind": "paired_line_facets",
            "number_of_facets": self.number_of_facets,
            "quadrature_points_per_facet": 2,
            "reference_length": float(np.sum(self.lengths)),
            "pairing_tolerance": self.tolerance,
            "dof_sides": "independent",
            "state_identity": self.identity(),
        }

    def identity(self) -> dict[str, object]:
        """Return the durable physical identity of the paired line facets."""

        if self.facet_keys:
            keys = self.facet_keys
            scope = "ordered_reference_facet_geometry"
        else:
            keys = tuple(
                ":".join(str(int(value)) for value in (*negative, *positive))
                for negative, positive in zip(
                    self.negative_nodes, self.positive_nodes, strict=True
                )
            )
            scope = "legacy_node_order"
        digest = sha256()
        for key in keys:
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
        digest.update(np.rint(self.normals / self.tolerance).astype("<i8").tobytes())
        digest.update(np.rint(self.lengths / self.tolerance).astype("<i8").tobytes())
        return {
            "schema": "agentfem.cohesive-interface-identity.v1",
            "sha256": digest.hexdigest(),
            "scope": scope,
            "number_of_facets": self.number_of_facets,
            "quadrature_points_per_facet": self.quadrature_points_per_facet,
            "pairing_tolerance": self.tolerance,
            "facet_keys": list(keys),
            "orientation_sensitive": True,
        }


@dataclass(frozen=True)
class PairedSurfaceFacets:
    """Deterministically paired zero-thickness triangular facets in 3D.

    The two sides have independent node identities but coincident reference
    geometry.  Positive-side node order is permuted to the negative-side
    geometry.  ``normal_hint`` remains an explicit physical convention: a
    coincident surface cannot infer which material side is positive.
    """

    negative_nodes: np.ndarray
    positive_nodes: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    tolerance: float
    facet_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        negative = np.asarray(self.negative_nodes, dtype=np.int64)
        positive = np.asarray(self.positive_nodes, dtype=np.int64)
        normals = np.asarray(self.normals, dtype=float)
        areas = np.asarray(self.areas, dtype=float).reshape(-1)
        tolerance = float(self.tolerance)
        if negative.ndim != 2 or negative.shape[1] != 3:
            raise ValueError(
                "PairedSurfaceFacets negative_nodes must have shape (facets, 3)."
            )
        if positive.shape != negative.shape:
            raise ValueError(
                "PairedSurfaceFacets positive_nodes must match negative_nodes."
            )
        if normals.shape != (negative.shape[0], 3):
            raise ValueError(
                "PairedSurfaceFacets normals must provide one 3D vector per facet."
            )
        if (
            areas.shape != (negative.shape[0],)
            or np.any(~np.isfinite(areas))
            or np.any(areas <= 0.0)
        ):
            raise ValueError("PairedSurfaceFacets areas must be positive per facet.")
        if np.any(~np.isfinite(normals)):
            raise ValueError("PairedSurfaceFacets normals must be finite.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("PairedSurfaceFacets tolerance must be finite and positive.")
        keys = tuple(str(value) for value in self.facet_keys)
        if keys and (len(keys) != negative.shape[0] or len(set(keys)) != len(keys)):
            raise ValueError("PairedSurfaceFacets facet_keys must be unique per facet.")
        object.__setattr__(self, "negative_nodes", negative.copy())
        object.__setattr__(self, "positive_nodes", positive.copy())
        object.__setattr__(self, "normals", normals.copy())
        object.__setattr__(self, "areas", areas.copy())
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "facet_keys", keys)

    @property
    def number_of_facets(self) -> int:
        return int(self.negative_nodes.shape[0])

    @property
    def quadrature_points_per_facet(self) -> int:
        return 3

    @property
    def number_of_points(self) -> int:
        return self.quadrature_points_per_facet * self.number_of_facets

    @property
    def measures(self) -> np.ndarray:
        """Reference area per facet."""

        return self.areas.copy()

    def summary(self) -> dict[str, object]:
        return {
            "kind": "paired_triangular_surface_facets",
            "number_of_facets": self.number_of_facets,
            "quadrature_points_per_facet": self.quadrature_points_per_facet,
            "reference_area": float(np.sum(self.areas)),
            "pairing_tolerance": self.tolerance,
            "dof_sides": "independent",
            "state_identity": self.identity(),
        }

    def identity(self) -> dict[str, object]:
        if self.facet_keys:
            keys = self.facet_keys
            scope = "ordered_reference_facet_geometry"
        else:
            keys = tuple(
                ":".join(str(int(value)) for value in (*negative, *positive))
                for negative, positive in zip(
                    self.negative_nodes, self.positive_nodes, strict=True
                )
            )
            scope = "legacy_node_order"
        digest = sha256()
        for key in keys:
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
        digest.update(np.rint(self.normals / self.tolerance).astype("<i8").tobytes())
        digest.update(
            np.rint(np.sqrt(self.areas) / self.tolerance)
            .astype("<i8")
            .tobytes()
        )
        return {
            "schema": "agentfem.cohesive-interface-identity.v1",
            "sha256": digest.hexdigest(),
            "scope": scope,
            "number_of_facets": self.number_of_facets,
            "quadrature_points_per_facet": self.quadrature_points_per_facet,
            "pairing_tolerance": self.tolerance,
            "facet_keys": list(keys),
            "orientation_sensitive": True,
        }

@dataclass(frozen=True)
class SplitInterfaceMesh:
    """Array-level result of splitting one conforming interface manifold."""

    coordinates: np.ndarray
    cells: np.ndarray
    negative_facets: np.ndarray
    positive_facets: np.ndarray
    original_to_duplicate: dict[int, int]
    positive_cells: np.ndarray

    def summary(self) -> dict[str, object]:
        return {
            "kind": "split_zero_thickness_interface_mesh",
            "geometric_dimension": int(self.coordinates.shape[1]),
            "facet_nodes": int(self.negative_facets.shape[1]),
            "number_of_cells": int(self.cells.shape[0]),
            "number_of_original_interface_nodes": len(self.original_to_duplicate),
            "number_of_interface_facets": int(self.negative_facets.shape[0]),
            "independent_sides": True,
        }

    def identity(self) -> dict[str, object]:
        """Return the rank-independent audited array identity."""

        digest = sha256()
        for values in (
            self.coordinates,
            self.cells,
            self.negative_facets,
            self.positive_facets,
            self.positive_cells,
        ):
            selected = np.ascontiguousarray(values)
            digest.update(str(selected.dtype).encode("ascii"))
            digest.update(np.asarray(selected.shape, dtype="<i8").tobytes())
            digest.update(selected.tobytes())
        for source, target in sorted(self.original_to_duplicate.items()):
            digest.update(np.asarray((source, target), dtype="<i8").tobytes())
        return {
            "schema": "agentfem.split-interface-mesh-identity.v1",
            "sha256": digest.hexdigest(),
            **self.summary(),
        }


@dataclass(frozen=True)
class NamedSplitInterfaceMesh:
    """One solver mesh carrying several disjoint named cohesive surfaces."""

    coordinates: np.ndarray
    cells: np.ndarray
    interfaces: dict[str, SplitInterfaceMesh]

    def __post_init__(self) -> None:
        points = np.asarray(self.coordinates, dtype=float)
        cells = np.asarray(self.cells, dtype=int)
        records = dict(self.interfaces)
        if not records:
            raise ValueError("NamedSplitInterfaceMesh requires at least one interface.")
        if any(not str(name).strip() for name in records):
            raise ValueError("Named split interfaces require nonempty names.")
        for name, split in records.items():
            if not isinstance(split, SplitInterfaceMesh):
                raise TypeError(f"Named interface {name!r} is not a SplitInterfaceMesh.")
            if not np.array_equal(split.coordinates, points) or not np.array_equal(
                split.cells, cells
            ):
                raise ValueError(
                    f"Named interface {name!r} does not share the combined solver mesh."
                )
        object.__setattr__(self, "coordinates", points.copy())
        object.__setattr__(self, "cells", cells.copy())
        object.__setattr__(self, "interfaces", records)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.interfaces)

    def __getitem__(self, name: str) -> SplitInterfaceMesh:
        return self.interfaces[str(name)]

    def combined(self) -> SplitInterfaceMesh:
        """Return the topology-only union used to create one DOLFINx mesh."""

        records = tuple(self.interfaces.values())
        mapping = {}
        for split in records:
            overlap = set(mapping) & set(split.original_to_duplicate)
            if overlap:
                raise RuntimeError(
                    "Named interfaces unexpectedly share duplicated source nodes."
                )
            mapping.update(split.original_to_duplicate)
        return SplitInterfaceMesh(
            coordinates=self.coordinates.copy(),
            cells=self.cells.copy(),
            negative_facets=np.vstack([item.negative_facets for item in records]),
            positive_facets=np.vstack([item.positive_facets for item in records]),
            original_to_duplicate=mapping,
            positive_cells=np.unique(
                np.concatenate([item.positive_cells for item in records])
            ),
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "named_split_interface_mesh",
            "geometric_dimension": int(self.coordinates.shape[1]),
            "number_of_cells": int(self.cells.shape[0]),
            "interfaces": {
                name: split.summary() for name, split in self.interfaces.items()
            },
            "disjoint_interface_nodes": True,
            "single_solver_mesh": True,
        }

    def identity(self) -> dict[str, object]:
        digest = sha256()
        digest.update(np.ascontiguousarray(self.coordinates).tobytes())
        digest.update(np.ascontiguousarray(self.cells).tobytes())
        for name in sorted(self.interfaces):
            digest.update(name.encode("utf-8"))
            digest.update(self.interfaces[name].identity()["sha256"].encode("ascii"))
        return {
            "schema": "agentfem.named-split-interface-mesh-identity.v1",
            "sha256": digest.hexdigest(),
            **self.summary(),
        }


@dataclass(frozen=True)
class InterfaceRigidModeAudit:
    """Rigid-body constraint rank of a split-interface model."""

    geometric_dimension: int
    connected_bodies: int
    rigid_body_parameters: int
    constraint_rank: int
    nullity: int
    singular_values: tuple[float, ...]
    interface_kinematics: dict[str, str]

    @property
    def well_posed(self) -> bool:
        return self.nullity == 0

    def summary(self) -> dict[str, object]:
        return {
            "kind": "split_interface_rigid_mode_audit",
            "geometric_dimension": self.geometric_dimension,
            "connected_bodies": self.connected_bodies,
            "rigid_body_parameters": self.rigid_body_parameters,
            "constraint_rank": self.constraint_rank,
            "nullity": self.nullity,
            "well_posed": self.well_posed,
            "interface_kinematics": dict(self.interface_kinematics),
            "singular_values": list(self.singular_values),
        }


def audit_split_interface_rigid_modes(
    split: SplitInterfaceMesh | NamedSplitInterfaceMesh,
    *,
    constrained_components,
    tangential="free",
    active_facets=None,
    rank_tolerance: float | None = None,
    error_if_singular: bool = False,
) -> InterfaceRigidModeAudit:
    """Audit rigid translations and rotations before creating a solver mesh.

    ``constrained_components`` maps input-node ids to constrained displacement
    component ids.  The audit operates on the actual disconnected bulk graph
    produced by interface splitting.  Normal-only interfaces contribute only
    normal compatibility; ``tie``, ``degraded`` and ``mixed`` contribute the
    full intact vector connection.  This is a deterministic model preflight,
    not a costly sparse eigensolve.
    """

    if isinstance(split, NamedSplitInterfaceMesh):
        points = split.coordinates
        cells = split.cells
        records = split.interfaces
    elif isinstance(split, SplitInterfaceMesh):
        points = split.coordinates
        cells = split.cells
        records = {"interface": split}
    else:
        raise TypeError("Rigid-mode audit requires a split interface mesh.")
    dimension = int(points.shape[1])
    if dimension not in {2, 3}:
        raise ValueError("Rigid-mode audit supports two or three dimensions.")
    selected_modes = (
        {name: tangential for name in records}
        if isinstance(tangential, str)
        else dict(tangential)
    )
    if set(selected_modes) != set(records):
        raise ValueError("Tangential modes must match the named interfaces.")
    normalized_modes = {}
    for name, mode in selected_modes.items():
        selected = str(mode).strip().lower().replace("-", "_")
        selected = {
            "none": "free",
            "normal_only": "free",
            "cohesive": "degraded",
        }.get(selected, selected)
        if selected not in {"free", "tie", "degraded", "mixed"}:
            raise ValueError(f"Unsupported tangential mode for interface {name!r}.")
        normalized_modes[name] = selected
    selected_active = (
        {
            name: np.ones(item.negative_facets.shape[0], dtype=bool)
            for name, item in records.items()
        }
        if active_facets is None
        else dict(active_facets)
    )
    if set(selected_active) != set(records):
        raise ValueError("Active-facet masks must match the named interfaces.")
    for name, interface in records.items():
        mask = np.asarray(selected_active[name], dtype=bool)
        if mask.shape != (interface.negative_facets.shape[0],):
            raise ValueError(f"Active-facet mask for interface {name!r} is invalid.")
        selected_active[name] = mask

    parent = np.arange(points.shape[0], dtype=int)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return int(value)

    def union(left, right):
        a, b = find(int(left)), find(int(right))
        if a != b:
            parent[b] = a

    active_nodes = np.unique(cells)
    for cell in np.asarray(cells, dtype=int):
        for node in cell[1:]:
            union(cell[0], node)
    roots = sorted({find(int(node)) for node in active_nodes})
    root_to_component = {root: index for index, root in enumerate(roots)}
    node_component = np.full(points.shape[0], -1, dtype=int)
    for node in active_nodes:
        node_component[int(node)] = root_to_component[find(int(node))]
    body_count = len(roots)
    parameters_per_body = 3 if dimension == 2 else 6
    total_parameters = body_count * parameters_per_body
    centroids = np.vstack(
        [
            points[active_nodes[node_component[active_nodes] == index]].mean(axis=0)
            for index in range(body_count)
        ]
    )

    def rigid_map(point, component):
        offset = np.asarray(point, dtype=float) - centroids[component]
        if dimension == 2:
            x, y = offset
            return np.array([[1.0, 0.0, -y], [0.0, 1.0, x]])
        x, y, z = offset
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0, z, -y],
                [0.0, 1.0, 0.0, -z, 0.0, x],
                [0.0, 0.0, 1.0, y, -x, 0.0],
            ]
        )

    rows = []
    declared_constraints = dict(constrained_components)
    for node, components in declared_constraints.items():
        node = int(node)
        if node < 0 or node >= points.shape[0] or node_component[node] < 0:
            raise ValueError(f"Constrained input node {node} is not in the solver cells.")
        body = int(node_component[node])
        mapping = rigid_map(points[node], body)
        for component in components:
            component = int(component)
            if component < 0 or component >= dimension:
                raise ValueError("A constrained displacement component is invalid.")
            row = np.zeros(total_parameters, dtype=float)
            start = body * parameters_per_body
            row[start : start + parameters_per_body] = mapping[component]
            rows.append(row)

    for name, interface in records.items():
        mode = normalized_modes[name]
        negative = interface.negative_facets
        positive = interface.positive_facets
        for facet_index, (minus_nodes, plus_nodes) in enumerate(
            zip(negative, positive, strict=True)
        ):
            if not selected_active[name][facet_index]:
                continue
            minus_body = int(node_component[int(minus_nodes[0])])
            plus_body = int(node_component[int(plus_nodes[0])])
            if minus_body < 0 or plus_body < 0 or minus_body == plus_body:
                raise RuntimeError(
                    f"Interface {name!r} facet {facet_index} does not join two split bodies."
                )
            if mode == "free":
                if dimension == 2:
                    segment = points[minus_nodes[1]] - points[minus_nodes[0]]
                    direction = np.array([-segment[1], segment[0]], dtype=float)
                else:
                    direction = np.cross(
                        points[minus_nodes[1]] - points[minus_nodes[0]],
                        points[minus_nodes[2]] - points[minus_nodes[0]],
                    )
                direction /= np.linalg.norm(direction)
                directions = (direction,)
            else:
                directions = tuple(np.eye(dimension))
            for minus_node, plus_node in zip(minus_nodes, plus_nodes, strict=True):
                minus_map = rigid_map(points[minus_node], minus_body)
                plus_map = rigid_map(points[plus_node], plus_body)
                for direction in directions:
                    row = np.zeros(total_parameters, dtype=float)
                    minus_start = minus_body * parameters_per_body
                    plus_start = plus_body * parameters_per_body
                    row[minus_start : minus_start + parameters_per_body] -= direction @ minus_map
                    row[plus_start : plus_start + parameters_per_body] += direction @ plus_map
                    rows.append(row)

    matrix = (
        np.vstack(rows) if rows else np.empty((0, total_parameters), dtype=float)
    )
    singular = np.linalg.svd(matrix, compute_uv=False)
    if rank_tolerance is None:
        tolerance = (
            max(matrix.shape, default=1)
            * np.finfo(float).eps
            * (float(singular[0]) if singular.size else 1.0)
        )
    else:
        tolerance = float(rank_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("rank_tolerance must be finite and nonnegative.")
    rank = int(np.sum(singular > tolerance))
    audit = InterfaceRigidModeAudit(
        geometric_dimension=dimension,
        connected_bodies=body_count,
        rigid_body_parameters=total_parameters,
        constraint_rank=rank,
        nullity=total_parameters - rank,
        singular_values=tuple(float(value) for value in singular),
        interface_kinematics=normalized_modes,
    )
    if error_if_singular and not audit.well_posed:
        raise RuntimeError(
            "Split-interface model retains "
            f"{audit.nullity} unconstrained rigid-body mode(s). "
            "Add physical boundary constraints or a shear-carrying interface; "
            "do not suppress the mode with an arbitrary point fix."
        )
    return audit


def create_dolfinx_split_mesh(
    split: SplitInterfaceMesh | NamedSplitInterfaceMesh,
    *,
    comm=None,
    cell_type: str | None = None,
    input_order: str = "counterclockwise",
):
    """Create an executable DOLFINx mesh for an audited split interface.

    ``split_conforming_line_interface`` deliberately operates on plain arrays
    so imported meshes can be audited before a solver owns them.  This adapter
    is the corresponding execution boundary.  ``input_order`` is explicit
    because conventional CAE quadrilaterals enumerate vertices around the
    perimeter whereas the Basix reference quadrilateral uses tensor-product
    vertex order.

    In MPI, rank zero supplies the audited global arrays and DOLFINx
    partitions them with SCOTCH.  Split interfaces are disconnected in the
    bulk-cell adjacency graph, so the cohesive force adapter must communicate
    the two sides explicitly by input-node and physical-facet identity.
    """

    import basix.ufl
    import ufl
    from dolfinx import graph as dolfinx_graph
    from dolfinx import mesh as dolfinx_mesh
    from mpi4py import MPI

    if isinstance(split, NamedSplitInterfaceMesh):
        split = split.combined()
    if not isinstance(split, SplitInterfaceMesh):
        raise TypeError(
            "create_dolfinx_split_mesh requires SplitInterfaceMesh or "
            "NamedSplitInterfaceMesh."
        )
    selected_comm = MPI.COMM_SELF if comm is None else comm
    identities = selected_comm.allgather(split.identity())
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError(
            "Every MPI rank must provide the same audited SplitInterfaceMesh."
        )
    nodes_per_cell = int(split.cells.shape[1])
    geometric_dimension = int(split.coordinates.shape[1])
    inferred = {
        (2, 3): "triangle",
        (2, 4): "quadrilateral",
        (3, 4): "tetrahedron",
    }.get((geometric_dimension, nodes_per_cell))
    selected_cell = inferred if cell_type is None else str(cell_type).strip().lower()
    if selected_cell not in {"triangle", "quadrilateral", "tetrahedron"}:
        raise ValueError(
            "The split-interface adapter supports 2D triangles/quadrilaterals "
            "and 3D tetrahedra."
        )
    expected_nodes = 3 if selected_cell == "triangle" else 4
    if nodes_per_cell != expected_nodes:
        raise ValueError(
            f"cell_type={selected_cell!r} requires {expected_nodes} nodes per cell."
        )
    ordering = str(input_order).strip().lower().replace("-", "_")
    if ordering not in {"counterclockwise", "dolfinx"}:
        raise ValueError("input_order must be 'counterclockwise' or 'dolfinx'.")
    cells = np.asarray(split.cells, dtype=np.int64).copy()
    if selected_cell == "quadrilateral" and ordering == "counterclockwise":
        cells = cells[:, [0, 1, 3, 2]]
    coordinate_element = basix.ufl.element(
        "Lagrange",
        selected_cell,
        1,
        shape=(geometric_dimension,),
    )
    input_cells = cells if selected_comm.rank == 0 else np.empty(
        (0, cells.shape[1]), dtype=np.int64
    )
    input_coordinates = (
        np.asarray(split.coordinates, dtype=float)
        if selected_comm.rank == 0
        else np.empty((0, geometric_dimension), dtype=float)
    )
    partitioner = None
    if selected_comm.size > 1:
        # SCOTCH handles the disconnected bulk graph created by duplicating
        # cohesive-interface nodes, including small laboratory meshes for
        # which the default ParMETIS path may have an empty adjacency graph.
        try:
            partitioner = dolfinx_mesh.create_cell_partitioner(
                dolfinx_graph.partitioner_scotch(),
                dolfinx_mesh.GhostMode.shared_facet,
                geometric_dimension,
            )
        except Exception as exc:
            raise RuntimeError(
                "Distributed split-interface meshes require a DOLFINx build "
                "with SCOTCH partitioning support."
            ) from exc
    domain = dolfinx_mesh.create_mesh(
        selected_comm,
        input_cells,
        ufl.Mesh(coordinate_element),
        input_coordinates,
        partitioner=partitioner,
    )
    # The input indices are the durable bridge from the audited array mesh to
    # the solver mesh, including two distinct ids at coincident coordinates.
    expected = set(range(split.coordinates.shape[0]))
    retained = set(np.asarray(domain.geometry.input_global_indices, dtype=int))
    global_retained = set().union(*selected_comm.allgather(retained))
    if global_retained != expected:
        raise RuntimeError(
            "DOLFINx did not retain every split input-node identity; cohesive "
            "DOF recovery would be unsafe."
        )
    return domain


@dataclass(frozen=True)
class CohesiveFacetResponse:
    """Trial force, kinematics and energy from paired interface facets.

    ``traction`` remains the scalar normal traction for compatibility.
    Standard vector fields are carried explicitly so result writers do not
    need to reconstruct interface physics from nodal displacement later.
    """

    internal_force: np.ndarray
    opening: np.ndarray
    traction: np.ndarray
    damage: np.ndarray
    stored_energy: float
    dissipated_energy: float
    jump: np.ndarray | None = None
    tangential_jump: np.ndarray | None = None
    traction_vector: np.ndarray | None = None
    tangential_traction: np.ndarray | None = None
    mode_mixity: np.ndarray | None = None


@dataclass(frozen=True)
class ModeIKinematicsAudit:
    """Accepted-state check that a declared Mode-I path remains Mode-I."""

    maximum_normal_jump: float
    maximum_tangential_jump: float
    tangential_to_normal_ratio: float
    ratio_limit: float

    @property
    def accepted(self) -> bool:
        return self.tangential_to_normal_ratio <= self.ratio_limit

    def summary(self) -> dict[str, object]:
        return {
            "kind": "mode_i_kinematics_audit",
            "maximum_normal_jump": self.maximum_normal_jump,
            "maximum_tangential_jump": self.maximum_tangential_jump,
            "tangential_to_normal_ratio": self.tangential_to_normal_ratio,
            "ratio_limit": self.ratio_limit,
            "accepted": self.accepted,
        }


def audit_mode_i_kinematics(
    response: CohesiveFacetResponse,
    *,
    ratio_limit: float = 0.1,
    absolute_tolerance: float = 1.0e-12,
    error_if_exceeded: bool = False,
) -> ModeIKinematicsAudit:
    """Check tangential jump without changing cohesive history."""

    limit = float(ratio_limit)
    absolute = float(absolute_tolerance)
    if not isfinite(limit) or limit < 0.0:
        raise ValueError("ratio_limit must be finite and nonnegative.")
    if not isfinite(absolute) or absolute <= 0.0:
        raise ValueError("absolute_tolerance must be finite and positive.")
    if response.tangential_jump is None:
        raise ValueError("The response predates vector interface kinematics.")
    normal = float(np.max(np.abs(response.opening), initial=0.0))
    tangential = np.asarray(response.tangential_jump, dtype=float)
    tangential_norm = np.linalg.norm(tangential, axis=-1)
    maximum_tangential = float(np.max(tangential_norm, initial=0.0))
    ratio = maximum_tangential / max(normal, absolute)
    audit = ModeIKinematicsAudit(
        maximum_normal_jump=normal,
        maximum_tangential_jump=maximum_tangential,
        tangential_to_normal_ratio=ratio,
        ratio_limit=limit,
    )
    if error_if_exceeded and not audit.accepted:
        raise RuntimeError(
            "Declared Mode-I interface developed excessive tangential jump: "
            f"max|jump_t|/max|jump_n|={ratio:.6g} > {limit:.6g}."
        )
    return audit


@dataclass(frozen=True)
class CohesiveElementTangents:
    """Element-node layouts and consistent scalar-dof tangent matrices.

    ``nodes`` stores the negative trace followed by the positive trace for
    every paired facet.  Each matrix uses node-major component ordering.  The
    array contract stays independent of PETSc so serial and distributed
    consumers can lower the same verified interface kernel.
    """

    nodes: np.ndarray
    matrices: np.ndarray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=int)
        matrices = np.asarray(self.matrices, dtype=float)
        if nodes.ndim != 2:
            raise ValueError("Cohesive tangent nodes must be a two-dimensional array.")
        if matrices.ndim != 3 or matrices.shape[0] != nodes.shape[0]:
            raise ValueError("Cohesive tangent matrices and facets are incompatible.")
        if matrices.shape[1] != matrices.shape[2]:
            raise ValueError("Every cohesive element tangent must be square.")
        if not np.all(np.isfinite(matrices)):
            raise ValueError("Cohesive element tangents must be finite.")
        object.__setattr__(self, "nodes", nodes.copy())
        object.__setattr__(self, "matrices", matrices.copy())


def _mode_i_element_tangents(
    *,
    negative_nodes,
    positive_nodes,
    normals,
    shape_values,
    point_tangent,
    point_scale,
) -> CohesiveElementTangents:
    """Integrate ``B_jump.T * dt/ddelta * B_jump`` per paired facet."""

    negative = np.asarray(negative_nodes, dtype=int)
    positive = np.asarray(positive_nodes, dtype=int)
    normal = np.asarray(normals, dtype=float)
    shape = np.asarray(shape_values, dtype=float)
    tangent = np.asarray(point_tangent, dtype=float)
    scale = np.asarray(point_scale, dtype=float)
    facet_count, nodes_per_side = negative.shape
    dimension = normal.shape[1]
    if tangent.shape != (facet_count, shape.shape[0]):
        raise ValueError("Material tangent and cohesive quadrature layout differ.")
    if scale.shape not in {(facet_count,), (facet_count, shape.shape[0])}:
        raise ValueError("Cohesive quadrature scale has an invalid shape.")
    weighted = tangent * (scale[:, None] if scale.ndim == 1 else scale)
    side_scalar = np.einsum("qi,fq,qj->fij", shape, weighted, shape)
    # Construct the trace signs explicitly to preserve one matrix per physical
    # facet and the conventional [negative, positive] element ordering.
    trace_sign = np.concatenate(
        (-np.ones(nodes_per_side), np.ones(nodes_per_side))
    )
    signed_scalar = (
        side_scalar[:, np.tile(np.arange(nodes_per_side), 2)[:, None],
                    np.tile(np.arange(nodes_per_side), 2)[None, :]]
        * trace_sign[None, :, None]
        * trace_sign[None, None, :]
    )
    normal_projector = np.einsum("fi,fj->fij", normal, normal)
    block = np.einsum("fab,fij->faibj", signed_scalar, normal_projector)
    matrices = block.reshape(
        (facet_count, 2 * nodes_per_side * dimension, 2 * nodes_per_side * dimension)
    )
    return CohesiveElementTangents(
        nodes=np.concatenate((negative, positive), axis=1),
        matrices=matrices,
    )


def _vector_element_tangents(
    *,
    negative_nodes,
    positive_nodes,
    shape_values,
    point_tangent,
    point_scale,
) -> CohesiveElementTangents:
    """Integrate a full local/global vector interface tangent."""

    negative = np.asarray(negative_nodes, dtype=int)
    positive = np.asarray(positive_nodes, dtype=int)
    shape = np.asarray(shape_values, dtype=float)
    tangent = np.asarray(point_tangent, dtype=float)
    scale = np.asarray(point_scale, dtype=float)
    facets, nodes_per_side = negative.shape
    points = shape.shape[0]
    if tangent.ndim != 4 or tangent.shape[:2] != (facets, points):
        raise ValueError("Vector tangent and cohesive quadrature layout differ.")
    dimension = tangent.shape[2]
    if tangent.shape[3] != dimension:
        raise ValueError("Every interface point tangent must be square.")
    weights = scale[:, None] if scale.ndim == 1 else scale
    if weights.shape not in {(facets, 1), (facets, points)}:
        raise ValueError("Cohesive quadrature scale has an invalid shape.")
    weighted = np.broadcast_to(weights, (facets, points))
    trace_sign = np.concatenate(
        (-np.ones(nodes_per_side), np.ones(nodes_per_side))
    )
    local = np.tile(np.arange(nodes_per_side), 2)
    trace_nodes = 2 * nodes_per_side
    matrices = np.zeros(
        (facets, trace_nodes * dimension, trace_nodes * dimension), dtype=float
    )
    for row in range(trace_nodes):
        for column in range(trace_nodes):
            block = np.einsum(
                "q,fqde,fq->fde",
                shape[:, local[row]] * shape[:, local[column]],
                tangent,
                weighted,
            )
            block *= trace_sign[row] * trace_sign[column]
            row_slice = slice(row * dimension, (row + 1) * dimension)
            column_slice = slice(column * dimension, (column + 1) * dimension)
            matrices[:, row_slice, column_slice] = block
    return CohesiveElementTangents(
        nodes=np.concatenate((negative, positive), axis=1),
        matrices=matrices,
    )


def _interface_frames(normals) -> np.ndarray:
    """Return deterministic right-handed frames with normal in column zero."""

    normal = np.asarray(normals, dtype=float)
    dimension = normal.shape[1]
    frames = np.zeros((normal.shape[0], dimension, dimension), dtype=float)
    frames[:, :, 0] = normal
    if dimension == 2:
        frames[:, :, 1] = np.column_stack((-normal[:, 1], normal[:, 0]))
        return frames
    if dimension != 3:
        raise ValueError("Interface frames require geometric dimension 2 or 3.")
    axes = np.eye(3)
    for index, direction in enumerate(normal):
        seed = axes[int(np.argmin(np.abs(axes @ direction)))]
        first = np.cross(direction, seed)
        first /= np.linalg.norm(first)
        second = np.cross(direction, first)
        frames[index, :, 1] = first
        frames[index, :, 2] = second
    return frames


def _validate_tangential_mode(mode: str, stiffness, law) -> tuple[str, float]:
    selected = str(mode).strip().lower().replace("-", "_")
    aliases = {"none": "free", "normal_only": "free", "cohesive": "degraded"}
    selected = aliases.get(selected, selected)
    if selected not in {"free", "tie", "degraded", "mixed"}:
        raise ValueError(
            "tangential must be 'free', 'tie', 'degraded', or 'mixed'."
        )
    mixed = getattr(law, "summary", lambda: {})().get("mode") == "mixed"
    if mixed:
        if selected not in {"mixed"}:
            raise ValueError("A mixed-mode law requires tangential='mixed'.")
        return selected, float(law.tangential_stiffness)
    if selected == "mixed":
        raise TypeError("tangential='mixed' requires a mixed-mode cohesive law.")
    if selected == "free":
        if stiffness is not None and float(stiffness) != 0.0:
            raise ValueError("tangential_stiffness is incompatible with free slip.")
        return selected, 0.0
    value = law.initial_stiffness if stiffness is None else float(stiffness)
    if not isfinite(value) or value <= 0.0:
        raise ValueError("tangential_stiffness must be finite and positive.")
    return selected, value


def _point_interface_response(
    *, state, law, jump_global, normals, tangential, tangential_stiffness, begin
):
    """Evaluate one shared vector kinematics contract for every assembler."""

    facets, points, dimension = jump_global.shape
    frames = _interface_frames(normals)
    local_jump = np.einsum("fdi,fqd->fqi", frames, jump_global)
    flat_local = local_jump.reshape((-1, dimension))
    if tangential == "mixed":
        material = state.begin(flat_local) if begin else state.evaluate(flat_local)
        local_traction = material.traction.reshape((facets, points, dimension))
        local_tangent = material.tangent.reshape((facets, points, dimension, dimension))
        damage = material.damage.reshape((facets, points))
        stored = material.stored_energy.reshape((facets, points))
        dissipated = material.dissipated_energy.reshape((facets, points))
        mixity = material.mode_mixity.reshape((facets, points))
    else:
        opening = flat_local[:, 0]
        material = state.begin(opening) if begin else state.evaluate(opening)
        normal_traction = material.traction.reshape((facets, points))
        damage = material.damage.reshape((facets, points))
        local_traction = np.zeros((facets, points, dimension), dtype=float)
        local_traction[:, :, 0] = normal_traction
        local_tangent = np.zeros(
            (facets, points, dimension, dimension), dtype=float
        )
        local_tangent[:, :, 0, 0] = material.tangent.reshape((facets, points))
        if tangential != "free":
            scale = (
                np.ones_like(damage)
                if tangential == "tie"
                else 1.0 - damage
            )
            local_traction[:, :, 1:] = (
                tangential_stiffness * scale[:, :, None] * local_jump[:, :, 1:]
            )
            tangent_scale = tangential_stiffness * scale
            for component in range(1, dimension):
                local_tangent[:, :, component, component] = tangent_scale
            if tangential == "degraded":
                opening_values = local_jump[:, :, 0]
                tensile = opening_values > np.finfo(float).eps
                ds = np.zeros_like(opening_values)
                normal_tangent = material.tangent.reshape((facets, points))
                initial = float(law.initial_stiffness)
                ds[tensile] = (
                    normal_tangent[tensile] / initial - scale[tensile]
                ) / opening_values[tensile]
                local_tangent[:, :, 1:, 0] = (
                    tangential_stiffness
                    * local_jump[:, :, 1:]
                    * ds[:, :, None]
                )
        stored = material.stored_energy.reshape((facets, points)).copy()
        if tangential != "free":
            stored += (
                0.5
                * tangential_stiffness
                * scale
                * np.sum(local_jump[:, :, 1:] ** 2, axis=2)
            )
        dissipated = material.dissipated_energy.reshape((facets, points))
        shear_measure = np.sum(local_jump[:, :, 1:] ** 2, axis=2)
        denominator = np.maximum(local_jump[:, :, 0] ** 2 + shear_measure, np.finfo(float).tiny)
        mixity = shear_measure / denominator
    traction_global = np.einsum("fdi,fqi->fqd", frames, local_traction)
    tangent_global = np.einsum(
        "fdi,fqij,fej->fqde", frames, local_tangent, frames
    )
    tangential_jump = jump_global - (
        local_jump[:, :, 0, None] * normals[:, None, :]
    )
    tangential_traction = traction_global - (
        local_traction[:, :, 0, None] * normals[:, None, :]
    )
    return {
        "material": material,
        "jump": jump_global,
        "opening": local_jump[:, :, 0],
        "tangential_jump": tangential_jump,
        "traction": local_traction[:, :, 0],
        "traction_vector": traction_global,
        "tangential_traction": tangential_traction,
        "damage": damage,
        "stored_energy": stored,
        "dissipated_energy": dissipated,
        "mode_mixity": mixity,
        "tangent": tangent_global,
    }


class ModeICohesiveFacetAssembler:
    """Two-point line integration for a fixed-path 2D interface.

    This is an assembly kernel, not a mesh adapter.  Node numbers refer to a
    coordinate/displacement array in which the two coincident sides already
    own independent rows.  A future DOLFINx adapter maps these nodal forces to
    distributed vector degrees of freedom and assigns every facet pair one
    deterministic MPI owner.
    """

    _GAUSS = np.array(
        [
            [0.5 * (1.0 + 1.0 / np.sqrt(3.0)), 0.5 * (1.0 - 1.0 / np.sqrt(3.0))],
            [0.5 * (1.0 - 1.0 / np.sqrt(3.0)), 0.5 * (1.0 + 1.0 / np.sqrt(3.0))],
        ],
        dtype=float,
    )

    def __init__(
        self,
        topology: PairedLineFacets,
        law,
        *,
        number_of_nodes: int,
        thickness: float = 1.0,
        tangential: str = "free",
        tangential_stiffness: float | None = None,
    ):
        if int(number_of_nodes) <= 0:
            raise ValueError("number_of_nodes must be positive.")
        if not isfinite(float(thickness)) or float(thickness) <= 0.0:
            raise ValueError("thickness must be finite and positive.")
        largest = int(
            max(np.max(topology.negative_nodes), np.max(topology.positive_nodes))
        )
        if largest >= int(number_of_nodes):
            raise ValueError("Paired facet node number exceeds number_of_nodes.")
        self.topology = topology
        self.law = law
        self.number_of_nodes = int(number_of_nodes)
        self.thickness = float(thickness)
        self.tangential, self.tangential_stiffness = _validate_tangential_mode(
            tangential, tangential_stiffness, law
        )
        self.state = _transaction_for_law(law, topology.number_of_points)
        configure_dimension = getattr(self.state, "configure_dimension", None)
        if callable(configure_dimension):
            configure_dimension(self.topology.normals.shape[1])
        self._trial: CohesiveFacetResponse | None = None
        self.last_committed_response: CohesiveFacetResponse | None = None

    def initialize_precrack(self, facets) -> None:
        """Mark selected facet indices as fully separated before execution."""

        selected = np.asarray(facets)
        if selected.dtype == bool:
            if selected.shape != (self.topology.number_of_facets,):
                raise ValueError("Boolean precrack mask has the wrong facet shape.")
            mask = selected
        else:
            mask = np.zeros(self.topology.number_of_facets, dtype=bool)
            indices = np.asarray(selected, dtype=int)
            if np.any(indices < 0) or np.any(indices >= mask.size):
                raise ValueError("Precrack facet index is out of range.")
            mask[indices] = True
        point_mask = np.repeat(mask, 2)
        initialize_failed = getattr(self.state, "initialize_failed", None)
        if callable(initialize_failed):
            initialize_failed(point_mask)
        else:
            maximum = np.zeros((self.topology.number_of_facets, 2), dtype=float)
            maximum[mask, :] = self.law.failure_opening
            self.state.initialize(maximum.reshape(-1))

    def begin(self, displacement) -> CohesiveFacetResponse:
        """Assemble a replaceable trial force from nodal displacement."""

        values = _finite_array(displacement, name="displacement")
        if values.ndim != 2 or values.shape[0] != self.number_of_nodes:
            raise ValueError(
                "displacement must have shape (number_of_nodes, geometric_dimension)."
            )
        if values.shape[1] != self.topology.normals.shape[1]:
            raise ValueError("Displacement and interface normal dimensions differ.")

        negative = values[self.topology.negative_nodes]
        positive = values[self.topology.positive_nodes]
        nodal_jump = positive - negative
        jump_at_points = np.einsum("qi,fid->fqd", self._GAUSS, nodal_jump)
        point = _point_interface_response(
            state=self.state,
            law=self.law,
            jump_global=jump_at_points,
            normals=self.topology.normals,
            tangential=self.tangential,
            tangential_stiffness=self.tangential_stiffness,
            begin=True,
        )

        force = np.zeros_like(values)
        point_scale = 0.5 * self.topology.lengths * self.thickness
        for local_node in range(2):
            vector = np.sum(
                self._GAUSS[:, local_node][None, :, None]
                * point["traction_vector"],
                axis=1,
            ) * point_scale[:, None]
            np.add.at(force, self.topology.positive_nodes[:, local_node], vector)
            np.add.at(force, self.topology.negative_nodes[:, local_node], -vector)

        stored = float(
            np.sum(
                point["stored_energy"]
                * point_scale[:, None]
            )
        )
        dissipated = float(
            np.sum(
                point["dissipated_energy"]
                * point_scale[:, None]
            )
        )
        self._trial = CohesiveFacetResponse(
            internal_force=force,
            opening=point["opening"],
            traction=point["traction"],
            damage=point["damage"],
            stored_energy=stored,
            dissipated_energy=dissipated,
            jump=point["jump"],
            tangential_jump=point["tangential_jump"],
            traction_vector=point["traction_vector"],
            tangential_traction=point["tangential_traction"],
            mode_mixity=point["mode_mixity"],
        )
        return self._trial

    def tangent_elements(self, displacement) -> CohesiveElementTangents:
        """Return the consistent 2D interface tangent at committed history."""

        values = _finite_array(displacement, name="displacement")
        if values.shape != (
            self.number_of_nodes,
            self.topology.normals.shape[1],
        ):
            raise ValueError("Displacement shape differs from the cohesive interface.")
        jump = (
            values[self.topology.positive_nodes]
            - values[self.topology.negative_nodes]
        )
        point = _point_interface_response(
            state=self.state,
            law=self.law,
            jump_global=np.einsum("qi,fid->fqd", self._GAUSS, jump),
            normals=self.topology.normals,
            tangential=self.tangential,
            tangential_stiffness=self.tangential_stiffness,
            begin=False,
        )
        return _vector_element_tangents(
            negative_nodes=self.topology.negative_nodes,
            positive_nodes=self.topology.positive_nodes,
            shape_values=self._GAUSS,
            point_tangent=point["tangent"],
            point_scale=0.5 * self.topology.lengths * self.thickness,
        )

    def commit(self) -> None:
        if self._trial is None:
            raise RuntimeError("No cohesive facet trial response is available to commit.")
        self.last_committed_response = self._trial
        self.state.commit()
        self._trial = None

    def material_point_response(
        self,
        response: CohesiveFacetResponse | None = None,
    ) -> CohesiveResponse:
        """Recover current per-quadrature-point constitutive quantities.

        ``CohesiveFacetResponse`` intentionally stores interface-integrated
        energies for global balance.  Research field output needs the local
        energy densities instead; this method preserves that distinction.
        """

        selected = self.last_committed_response if response is None else response
        if selected is None:
            raise RuntimeError("No cohesive facet response is available.")
        if self.tangential == "mixed":
            frames = _interface_frames(self.topology.normals)
            local = np.einsum("fdi,fqd->fqi", frames, selected.jump)
            return self.state.evaluate(local.reshape((-1, local.shape[-1])))
        return self.state.evaluate(np.asarray(selected.opening, dtype=float).reshape(-1))

    def cycle_kinematics(
        self, response: CohesiveFacetResponse | None = None
    ) -> np.ndarray:
        """Return true local-basis point kinematics for one cycle extremum."""

        selected = self.last_committed_response if response is None else response
        if selected is None:
            raise RuntimeError("No cohesive facet response is available.")
        if self.tangential != "mixed":
            return np.asarray(selected.opening, dtype=float).reshape(-1).copy()
        frames = _interface_frames(self.topology.normals)
        local = np.einsum("fdi,fqd->fqi", frames, selected.jump)
        return local.reshape((-1, local.shape[-1])).copy()

    def rollback(self) -> None:
        self.state.rollback()
        self._trial = None


class ModeICohesiveSurfaceAssembler:
    """Three-point integration of linear triangular interfaces in 3D."""

    _QUADRATURE = np.array(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=float,
    )

    def __init__(
        self,
        topology: PairedSurfaceFacets,
        law,
        *,
        number_of_nodes: int,
        tangential: str = "free",
        tangential_stiffness: float | None = None,
    ):
        if int(number_of_nodes) <= 0:
            raise ValueError("number_of_nodes must be positive.")
        largest = int(
            max(np.max(topology.negative_nodes), np.max(topology.positive_nodes))
        )
        if largest >= int(number_of_nodes):
            raise ValueError("Paired surface node number exceeds number_of_nodes.")
        self.topology = topology
        self.law = law
        self.number_of_nodes = int(number_of_nodes)
        self.thickness = 1.0  # compatibility: surface measure already has area
        self.tangential, self.tangential_stiffness = _validate_tangential_mode(
            tangential, tangential_stiffness, law
        )
        self.state = _transaction_for_law(law, topology.number_of_points)
        configure_dimension = getattr(self.state, "configure_dimension", None)
        if callable(configure_dimension):
            configure_dimension(self.topology.normals.shape[1])
        self._trial: CohesiveFacetResponse | None = None
        self.last_committed_response: CohesiveFacetResponse | None = None

    def initialize_precrack(self, facets) -> None:
        selected = np.asarray(facets)
        if selected.dtype == bool:
            if selected.shape != (self.topology.number_of_facets,):
                raise ValueError("Boolean precrack mask has the wrong facet shape.")
            mask = selected
        else:
            mask = np.zeros(self.topology.number_of_facets, dtype=bool)
            indices = np.asarray(selected, dtype=int)
            if np.any(indices < 0) or np.any(indices >= mask.size):
                raise ValueError("Precrack facet index is out of range.")
            mask[indices] = True
        point_mask = np.repeat(mask, self.topology.quadrature_points_per_facet)
        initialize_failed = getattr(self.state, "initialize_failed", None)
        if callable(initialize_failed):
            initialize_failed(point_mask)
        else:
            maximum = np.zeros(
                (self.topology.number_of_facets, self.topology.quadrature_points_per_facet),
                dtype=float,
            )
            maximum[mask, :] = self.law.failure_opening
            self.state.initialize(maximum.reshape(-1))

    def begin(self, displacement) -> CohesiveFacetResponse:
        values = _finite_array(displacement, name="displacement")
        if values.shape != (self.number_of_nodes, 3):
            raise ValueError("3D cohesive displacement must have shape (nodes, 3).")
        jump = (
            values[self.topology.positive_nodes]
            - values[self.topology.negative_nodes]
        )
        jump_at_points = np.einsum("qi,fid->fqd", self._QUADRATURE, jump)
        points = self.topology.quadrature_points_per_facet
        point = _point_interface_response(
            state=self.state,
            law=self.law,
            jump_global=jump_at_points,
            normals=self.topology.normals,
            tangential=self.tangential,
            tangential_stiffness=self.tangential_stiffness,
            begin=True,
        )

        force = np.zeros_like(values)
        point_scale = self.topology.areas / float(points)
        for local_node in range(3):
            vector = np.sum(
                self._QUADRATURE[:, local_node][None, :, None]
                * point["traction_vector"],
                axis=1,
            ) * point_scale[:, None]
            np.add.at(force, self.topology.positive_nodes[:, local_node], vector)
            np.add.at(force, self.topology.negative_nodes[:, local_node], -vector)
        stored = float(
            np.sum(point["stored_energy"] * point_scale[:, None])
        )
        dissipated = float(
            np.sum(
                point["dissipated_energy"]
                * point_scale[:, None]
            )
        )
        self._trial = CohesiveFacetResponse(
            internal_force=force,
            opening=point["opening"],
            traction=point["traction"],
            damage=point["damage"],
            stored_energy=stored,
            dissipated_energy=dissipated,
            jump=point["jump"],
            tangential_jump=point["tangential_jump"],
            traction_vector=point["traction_vector"],
            tangential_traction=point["tangential_traction"],
            mode_mixity=point["mode_mixity"],
        )
        return self._trial

    def tangent_elements(self, displacement) -> CohesiveElementTangents:
        """Return the consistent 3D interface tangent at committed history."""

        values = _finite_array(displacement, name="displacement")
        if values.shape != (self.number_of_nodes, 3):
            raise ValueError("3D cohesive displacement must have shape (nodes, 3).")
        jump = (
            values[self.topology.positive_nodes]
            - values[self.topology.negative_nodes]
        )
        points = self.topology.quadrature_points_per_facet
        point = _point_interface_response(
            state=self.state,
            law=self.law,
            jump_global=np.einsum("qi,fid->fqd", self._QUADRATURE, jump),
            normals=self.topology.normals,
            tangential=self.tangential,
            tangential_stiffness=self.tangential_stiffness,
            begin=False,
        )
        return _vector_element_tangents(
            negative_nodes=self.topology.negative_nodes,
            positive_nodes=self.topology.positive_nodes,
            shape_values=self._QUADRATURE,
            point_tangent=point["tangent"],
            point_scale=self.topology.areas / float(points),
        )

    def commit(self) -> None:
        if self._trial is None:
            raise RuntimeError("No cohesive surface trial response is available.")
        self.last_committed_response = self._trial
        self.state.commit()
        self._trial = None

    def rollback(self) -> None:
        self.state.rollback()
        self._trial = None

    def material_point_response(
        self, response: CohesiveFacetResponse | None = None
    ) -> CohesiveResponse:
        selected = self.last_committed_response if response is None else response
        if selected is None:
            raise RuntimeError("No cohesive surface response is available.")
        if self.tangential == "mixed":
            frames = _interface_frames(self.topology.normals)
            local = np.einsum("fdi,fqd->fqi", frames, selected.jump)
            return self.state.evaluate(local.reshape((-1, local.shape[-1])))
        return self.state.evaluate(np.asarray(selected.opening, dtype=float).reshape(-1))

    def cycle_kinematics(
        self, response: CohesiveFacetResponse | None = None
    ) -> np.ndarray:
        selected = self.last_committed_response if response is None else response
        if selected is None:
            raise RuntimeError("No cohesive surface response is available.")
        if self.tangential != "mixed":
            return np.asarray(selected.opening, dtype=float).reshape(-1).copy()
        frames = _interface_frames(self.topology.normals)
        local = np.einsum("fdi,fqd->fqi", frames, selected.jump)
        return local.reshape((-1, local.shape[-1])).copy()


def pair_coincident_surface_facets(
    coordinates,
    negative_facets,
    positive_facets,
    *,
    normal_hint,
    tolerance: float = 1.0e-10,
) -> PairedSurfaceFacets:
    """Pair coincident three-node triangular facets in 3D."""

    points = _finite_array(coordinates, name="coordinates")
    negative = np.asarray(negative_facets, dtype=int)
    positive = np.asarray(positive_facets, dtype=int)
    hint = _finite_array(normal_hint, name="normal_hint")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Paired surface facets require 3D coordinates.")
    if negative.ndim != 2 or negative.shape[1] != 3:
        raise ValueError("negative_facets must have shape (facets, 3).")
    if positive.shape != negative.shape:
        raise ValueError("positive_facets must match negative_facets.")
    if hint.shape != (3,) or np.linalg.norm(hint) == 0.0:
        raise ValueError("normal_hint must be one nonzero 3D vector.")
    if not isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    if np.any(negative < 0) or np.any(positive < 0):
        raise ValueError("Facet node numbers cannot be negative.")
    if np.any(negative >= points.shape[0]) or np.any(positive >= points.shape[0]):
        raise ValueError("Facet node number exceeds the coordinate array.")
    if np.intersect1d(negative.reshape(-1), positive.reshape(-1)).size:
        raise ValueError("Cohesive surface sides require independent node identities.")

    positive_by_key = {}
    for index, facet in enumerate(positive):
        key = tuple(sorted(tuple(np.rint(point / tolerance).astype(np.int64)) for point in points[facet]))
        if key in positive_by_key:
            raise ValueError("Positive cohesive surface contains duplicate geometry.")
        positive_by_key[key] = int(index)
    ordered_positive = np.empty_like(negative)
    for index, facet in enumerate(negative):
        source = points[facet]
        key = tuple(sorted(tuple(np.rint(point / tolerance).astype(np.int64)) for point in source))
        candidate_index = positive_by_key.pop(key, None)
        if candidate_index is None:
            raise ValueError(f"Negative surface facet {index} has no coincident partner.")
        candidates = positive[candidate_index]
        target = points[candidates]
        permutation = []
        for point in source:
            distances = np.linalg.norm(target - point, axis=1)
            match = int(np.argmin(distances))
            if distances[match] > tolerance or match in permutation:
                raise ValueError("Coincident surface node pairing is ambiguous.")
            permutation.append(match)
        ordered_positive[index] = candidates[np.asarray(permutation, dtype=int)]
    if positive_by_key:
        raise ValueError("Positive cohesive surface contains unmatched facets.")

    cross = np.cross(
        points[negative[:, 1]] - points[negative[:, 0]],
        points[negative[:, 2]] - points[negative[:, 0]],
    )
    double_area = np.linalg.norm(cross, axis=1)
    if np.any(double_area <= tolerance**2):
        raise ValueError("Interface facets must have positive reference area.")
    normals = cross / double_area[:, None]
    hint = hint / np.linalg.norm(hint)
    signs = np.where(np.einsum("fd,d->f", normals, hint) >= 0.0, 1.0, -1.0)
    normals *= signs[:, None]
    facet_keys = tuple(
        sha256(
            np.asarray(
                sorted(
                    tuple(value)
                    for value in np.rint(points[facet] / float(tolerance)).astype("<i8")
                ),
                dtype="<i8",
            ).tobytes()
        ).hexdigest()
        for facet in negative
    )
    return PairedSurfaceFacets(
        negative_nodes=negative.copy(),
        positive_nodes=ordered_positive,
        normals=normals,
        areas=0.5 * double_area,
        tolerance=float(tolerance),
        facet_keys=facet_keys,
    )


def pair_coincident_line_facets(
    coordinates,
    negative_facets,
    positive_facets,
    *,
    normal_hint,
    tolerance: float = 1.0e-10,
) -> PairedLineFacets:
    """Pair coincident two-node line facets with a declared normal direction."""

    points = _finite_array(coordinates, name="coordinates")
    negative = np.asarray(negative_facets, dtype=int)
    positive = np.asarray(positive_facets, dtype=int)
    hint = _finite_array(normal_hint, name="normal_hint")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("The first paired-facet kernel requires 2D coordinates.")
    if negative.ndim != 2 or negative.shape[1] != 2:
        raise ValueError("negative_facets must have shape (facets, 2).")
    if positive.ndim != 2 or positive.shape[1] != 2:
        raise ValueError("positive_facets must have shape (facets, 2).")
    if negative.shape[0] != positive.shape[0]:
        raise ValueError("The two interface sides must contain the same facet count.")
    if hint.shape != (2,) or np.linalg.norm(hint) == 0.0:
        raise ValueError("normal_hint must be one nonzero 2D vector.")
    if not isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    if np.any(negative < 0) or np.any(positive < 0):
        raise ValueError("Facet node numbers cannot be negative.")
    if np.any(negative >= points.shape[0]) or np.any(positive >= points.shape[0]):
        raise ValueError("Facet node number exceeds the coordinate array.")
    if np.intersect1d(negative.reshape(-1), positive.reshape(-1)).size:
        raise ValueError(
            "Cohesive interface sides must use independent node identities; "
            "split or duplicate the shared interface nodes first."
        )

    negative_centroids = np.mean(points[negative], axis=1)
    positive_centroids = np.mean(points[positive], axis=1)
    distances = np.linalg.norm(
        negative_centroids[:, None, :] - positive_centroids[None, :, :], axis=2
    )
    ordered_positive = np.empty_like(negative)
    used: set[int] = set()
    for index in range(negative.shape[0]):
        candidates = np.flatnonzero(distances[index] <= tolerance)
        candidates = np.asarray([item for item in candidates if int(item) not in used])
        matches = []
        for candidate in candidates:
            source = points[negative[index]]
            target = points[positive[candidate]]
            direct = np.max(np.linalg.norm(source - target, axis=1))
            reverse = np.max(np.linalg.norm(source - target[::-1], axis=1))
            if min(direct, reverse) <= tolerance:
                matches.append((int(candidate), direct <= reverse))
        if len(matches) != 1:
            raise ValueError(
                "Every negative interface facet must have exactly one coincident "
                f"positive partner; facet {index} has {len(matches)}."
            )
        candidate, direct = matches[0]
        used.add(candidate)
        ordered_positive[index] = (
            positive[candidate] if direct else positive[candidate, ::-1]
        )

    segment = points[negative[:, 1]] - points[negative[:, 0]]
    lengths = np.linalg.norm(segment, axis=1)
    if np.any(lengths <= tolerance):
        raise ValueError("Interface facets must have positive reference length.")
    normals = np.column_stack((-segment[:, 1], segment[:, 0])) / lengths[:, None]
    hint = hint / np.linalg.norm(hint)
    signs = np.where(np.einsum("fd,d->f", normals, hint) >= 0.0, 1.0, -1.0)
    normals *= signs[:, None]
    quantized = np.rint(points[negative] / float(tolerance)).astype("<i8")
    facet_keys = tuple(
        sha256(np.ascontiguousarray(item).tobytes()).hexdigest()
        for item in quantized
    )
    return PairedLineFacets(
        negative_nodes=negative.copy(),
        positive_nodes=ordered_positive,
        normals=normals,
        lengths=lengths,
        tolerance=float(tolerance),
        facet_keys=facet_keys,
    )


def split_conforming_line_interface(
    coordinates,
    cells,
    interface_facets,
    *,
    positive_cells,
) -> SplitInterfaceMesh:
    """Duplicate nodes on a declared conforming 2D cell interface.

    ``positive_cells`` is an explicit set of cell indices whose interface
    nodes are replaced by duplicates.  Requiring this side identity avoids a
    fragile geometric guess and lets an Abaqus/Gmsh adapter preserve source
    surface semantics.  Every interface facet must separate exactly one
    selected cell from one unselected cell.
    """

    points = _finite_array(coordinates, name="coordinates")
    connectivity = np.asarray(cells, dtype=int)
    facets = np.asarray(interface_facets, dtype=int)
    selected = np.asarray(positive_cells)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Interface splitting currently requires 2D coordinates.")
    if connectivity.ndim != 2 or connectivity.shape[0] == 0:
        raise ValueError("cells must be one nonempty connectivity block.")
    if facets.ndim != 2 or facets.shape[1] != 2 or facets.shape[0] == 0:
        raise ValueError("interface_facets must have shape (facets, 2).")
    if np.any(connectivity < 0) or np.any(connectivity >= points.shape[0]):
        raise ValueError("Cell connectivity contains an invalid node number.")
    if np.any(facets < 0) or np.any(facets >= points.shape[0]):
        raise ValueError("Interface connectivity contains an invalid node number.")
    if selected.dtype == bool:
        if selected.shape != (connectivity.shape[0],):
            raise ValueError("Boolean positive_cells mask has the wrong shape.")
        positive_mask = selected.copy()
    else:
        positive_mask = np.zeros(connectivity.shape[0], dtype=bool)
        indices = np.asarray(selected, dtype=int)
        if np.any(indices < 0) or np.any(indices >= connectivity.shape[0]):
            raise ValueError("positive_cells contains an out-of-range cell index.")
        positive_mask[indices] = True
    if not np.any(positive_mask) or np.all(positive_mask):
        raise ValueError("Interface splitting requires cells on both declared sides.")

    for facet_index, facet in enumerate(facets):
        incident = np.flatnonzero(
            np.sum(np.isin(connectivity, facet), axis=1) == 2
        )
        if incident.size != 2:
            raise ValueError(
                f"Interface facet {facet_index} must have exactly two incident cells; "
                f"found {incident.size}."
            )
        sides = positive_mask[incident]
        if int(np.sum(sides)) != 1:
            raise ValueError(
                f"Interface facet {facet_index} does not separate one positive and "
                "one negative cell."
            )

    interface_nodes = np.unique(facets)
    duplicates = np.arange(
        points.shape[0], points.shape[0] + interface_nodes.size, dtype=int
    )
    mapping = {
        int(source): int(target)
        for source, target in zip(interface_nodes, duplicates, strict=True)
    }
    split_points = np.vstack((points, points[interface_nodes]))
    split_cells = connectivity.copy()
    for source, target in mapping.items():
        rows, columns = np.nonzero(
            positive_mask[:, None] & (split_cells == source)
        )
        split_cells[rows, columns] = target
    positive_facets_array = np.vectorize(mapping.__getitem__, otypes=[int])(facets)
    return SplitInterfaceMesh(
        coordinates=split_points,
        cells=split_cells,
        negative_facets=facets.copy(),
        positive_facets=np.asarray(positive_facets_array, dtype=int),
        original_to_duplicate=mapping,
        positive_cells=np.flatnonzero(positive_mask),
    )


def split_conforming_surface_interface(
    coordinates,
    cells,
    interface_facets,
    *,
    positive_cells,
) -> SplitInterfaceMesh:
    """Duplicate nodes on a declared conforming triangular surface in 3D."""

    points = _finite_array(coordinates, name="coordinates")
    connectivity = np.asarray(cells, dtype=int)
    facets = np.asarray(interface_facets, dtype=int)
    selected = np.asarray(positive_cells)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Surface splitting requires 3D coordinates.")
    if (
        connectivity.ndim != 2
        or connectivity.shape[0] == 0
        or connectivity.shape[1] != 4
    ):
        raise ValueError(
            "3D surface splitting currently requires one linear tetrahedron block."
        )
    if facets.ndim != 2 or facets.shape[1] != 3 or facets.shape[0] == 0:
        raise ValueError("interface_facets must have shape (facets, 3).")
    if np.any(connectivity < 0) or np.any(connectivity >= points.shape[0]):
        raise ValueError("Cell connectivity contains an invalid node number.")
    if np.any(facets < 0) or np.any(facets >= points.shape[0]):
        raise ValueError("Interface connectivity contains an invalid node number.")
    if selected.dtype == bool:
        if selected.shape != (connectivity.shape[0],):
            raise ValueError("Boolean positive_cells mask has the wrong shape.")
        positive_mask = selected.copy()
    else:
        positive_mask = np.zeros(connectivity.shape[0], dtype=bool)
        indices = np.asarray(selected, dtype=int)
        if np.any(indices < 0) or np.any(indices >= connectivity.shape[0]):
            raise ValueError("positive_cells contains an out-of-range cell index.")
        positive_mask[indices] = True
    if not np.any(positive_mask) or np.all(positive_mask):
        raise ValueError("Interface splitting requires cells on both declared sides.")
    for facet_index, facet in enumerate(facets):
        incident = np.flatnonzero(
            np.sum(np.isin(connectivity, facet), axis=1) == facet.size
        )
        if incident.size != 2:
            raise ValueError(
                f"Interface facet {facet_index} must have exactly two incident cells; "
                f"found {incident.size}."
            )
        if int(np.sum(positive_mask[incident])) != 1:
            raise ValueError(
                f"Interface facet {facet_index} does not separate one positive "
                "and one negative cell."
            )
    interface_nodes = np.unique(facets)
    duplicates = np.arange(
        points.shape[0], points.shape[0] + interface_nodes.size, dtype=int
    )
    mapping = {
        int(source): int(target)
        for source, target in zip(interface_nodes, duplicates, strict=True)
    }
    split_points = np.vstack((points, points[interface_nodes]))
    split_cells = connectivity.copy()
    for source, target in mapping.items():
        rows, columns = np.nonzero(positive_mask[:, None] & (split_cells == source))
        split_cells[rows, columns] = target
    positive_facets_array = np.vectorize(mapping.__getitem__, otypes=[int])(facets)
    return SplitInterfaceMesh(
        coordinates=split_points,
        cells=split_cells,
        negative_facets=facets.copy(),
        positive_facets=np.asarray(positive_facets_array, dtype=int),
        original_to_duplicate=mapping,
        positive_cells=np.flatnonzero(positive_mask),
    )


def split_conforming_named_interfaces(
    coordinates,
    cells,
    named_interfaces,
) -> NamedSplitInterfaceMesh:
    """Atomically split several disjoint conforming cohesive manifolds.

    Each mapping value supplies ``interface_facets`` and ``positive_cells``.
    Interface nodes must be disjoint in this first multi-crack contract. This
    gives every duplicate a durable identity and fits separated surface cracks;
    intersecting cohesive networks require an explicit junction topology.
    """

    points = _finite_array(coordinates, name="coordinates")
    connectivity = np.asarray(cells, dtype=int)
    specifications = tuple(dict(named_interfaces).items())
    if not specifications:
        raise ValueError("At least one named interface is required.")
    names = tuple(str(name).strip() for name, _record in specifications)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Named interfaces require unique nonempty names.")
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError("Named interface splitting requires 2D or 3D coordinates.")
    splitter = (
        split_conforming_line_interface
        if points.shape[1] == 2
        else split_conforming_surface_interface
    )
    validated = []
    occupied_nodes: set[int] = set()
    for normalized_name, (_declared_name, record) in zip(
        names, specifications, strict=True
    ):
        if isinstance(record, dict):
            try:
                facets = record["interface_facets"]
                positive_cells = record["positive_cells"]
            except KeyError as exc:
                raise ValueError(
                    f"Named interface {normalized_name!r} requires "
                    "interface_facets and positive_cells."
                ) from exc
        else:
            try:
                facets, positive_cells = record
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "A named interface must be a mapping or a "
                    "(interface_facets, positive_cells) pair."
                ) from exc
        checked = splitter(
            points,
            connectivity,
            facets,
            positive_cells=positive_cells,
        )
        nodes = set(int(value) for value in np.unique(checked.negative_facets))
        overlap = occupied_nodes & nodes
        if overlap:
            raise ValueError(
                "Named cohesive interfaces must not share source nodes; "
                f"interface {normalized_name!r} overlaps nodes {sorted(overlap)[:8]}."
            )
        occupied_nodes.update(nodes)
        validated.append((normalized_name, checked))

    point_blocks = [points]
    split_cells = connectivity.copy()
    cursor = int(points.shape[0])
    pending = []
    for name, checked in validated:
        source_nodes = np.asarray(sorted(checked.original_to_duplicate), dtype=int)
        duplicate_nodes = np.arange(cursor, cursor + source_nodes.size, dtype=int)
        cursor += source_nodes.size
        mapping = {
            int(source): int(target)
            for source, target in zip(source_nodes, duplicate_nodes, strict=True)
        }
        point_blocks.append(points[source_nodes])
        positive_mask = np.zeros(connectivity.shape[0], dtype=bool)
        positive_mask[np.asarray(checked.positive_cells, dtype=int)] = True
        for source, target in mapping.items():
            rows, columns = np.nonzero(
                positive_mask[:, None] & (split_cells == source)
            )
            split_cells[rows, columns] = target
        positive_facets = np.vectorize(mapping.__getitem__, otypes=[int])(
            checked.negative_facets
        )
        pending.append(
            (
                name,
                checked.negative_facets.copy(),
                np.asarray(positive_facets, dtype=int),
                mapping,
                np.asarray(checked.positive_cells, dtype=int),
            )
        )
    combined_points = np.vstack(point_blocks)
    records = {
        name: SplitInterfaceMesh(
            coordinates=combined_points,
            cells=split_cells,
            negative_facets=negative,
            positive_facets=positive,
            original_to_duplicate=mapping,
            positive_cells=positive_cells,
        )
        for name, negative, positive, mapping, positive_cells in pending
    }
    return NamedSplitInterfaceMesh(
        coordinates=combined_points,
        cells=split_cells,
        interfaces=records,
    )


def split_conforming_cell_interface(
    coordinates,
    cells,
    *,
    positive_cells,
) -> SplitInterfaceMesh:
    """Split the internal facet separating two declared cell partitions.

    This engineering adapter derives the conforming line facets from cell
    connectivity instead of asking a user or mesh importer to enumerate every
    edge twice.  ``positive_cells`` retains the important physical decision:
    which material/region side receives the duplicated nodes.  Triangles and
    perimeter-ordered quadrilaterals are supported; non-manifold edges and an
    empty partition interface are rejected before a solver mesh is created.
    """

    points = _finite_array(coordinates, name="coordinates")
    connectivity = np.asarray(cells, dtype=int)
    selected = np.asarray(positive_cells)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError("Cell-interface recovery requires 2D or 3D coordinates.")
    supported = (
        points.shape[1] == 2 and connectivity.ndim == 2 and connectivity.shape[1] in {3, 4}
    ) or (
        points.shape[1] == 3 and connectivity.ndim == 2 and connectivity.shape[1] == 4
    )
    if not supported:
        raise ValueError(
            "Cell-interface recovery supports 2D triangles/quadrilaterals "
            "or one 3D tetrahedron block."
        )
    if np.any(connectivity < 0) or np.any(connectivity >= points.shape[0]):
        raise ValueError("Cell connectivity contains an invalid node number.")
    if selected.dtype == bool:
        if selected.shape != (connectivity.shape[0],):
            raise ValueError("Boolean positive_cells mask has the wrong shape.")
        positive_mask = selected.copy()
    else:
        positive_mask = np.zeros(connectivity.shape[0], dtype=bool)
        indices = np.asarray(selected, dtype=int)
        if np.any(indices < 0) or np.any(indices >= connectivity.shape[0]):
            raise ValueError("positive_cells contains an out-of-range cell index.")
        positive_mask[indices] = True
    if not np.any(positive_mask) or np.all(positive_mask):
        raise ValueError("Interface recovery requires cells on both declared sides.")

    facet_owners: dict[tuple[int, ...], list[int]] = {}
    for cell_index, cell in enumerate(connectivity):
        if points.shape[1] == 2:
            local_facets = [
                (int(cell[local]), int(cell[(local + 1) % cell.size]))
                for local in range(cell.size)
            ]
        else:
            local_facets = [
                (int(cell[0]), int(cell[2]), int(cell[1])),
                (int(cell[0]), int(cell[1]), int(cell[3])),
                (int(cell[1]), int(cell[2]), int(cell[3])),
                (int(cell[2]), int(cell[0]), int(cell[3])),
            ]
        for facet in local_facets:
            key = tuple(sorted(facet))
            facet_owners.setdefault(key, []).append(int(cell_index))
    non_manifold = {
        facet: owners for facet, owners in facet_owners.items() if len(owners) > 2
    }
    if non_manifold:
        facet, owners = next(iter(non_manifold.items()))
        raise ValueError(
            "Cell-interface recovery requires a manifold mesh; "
            f"facet {facet} has {len(owners)} incident cells."
        )
    interface = [
        facet
        for facet, owners in facet_owners.items()
        if len(owners) == 2
        and bool(positive_mask[owners[0]]) != bool(positive_mask[owners[1]])
    ]
    if not interface:
        raise ValueError(
            "The declared cell partitions do not share a conforming internal facet."
        )
    splitter = (
        split_conforming_line_interface
        if points.shape[1] == 2
        else split_conforming_surface_interface
    )
    return splitter(
        points, connectivity, np.asarray(sorted(interface), dtype=int),
        positive_cells=positive_mask,
    )


@dataclass(frozen=True)
class CohesiveSurface:
    """Public description of a fixed-path zero-thickness interface."""

    law: object
    mode: str = "normal"
    name: str = "cohesive surface"
    maturity: str = "experimental"

    def __post_init__(self) -> None:
        selected = str(self.mode).strip().lower().replace("-", "_")
        if selected not in {"normal", "mixed"}:
            raise ValueError("CohesiveSurface.mode must be 'normal' or 'mixed'.")
        if selected == "mixed" and getattr(
            self.law, "summary", lambda: {}
        )().get("mode") != "mixed":
            raise TypeError("mode='mixed' requires a mixed-mode cohesive law.")
        object.__setattr__(self, "mode", selected)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "zero_thickness_cohesive_surface",
            "mode": self.mode,
            "law": self.law.summary(),
            "topology_requirement": "paired coincident facets with independent dofs",
            "maturity": self.maturity,
        }


def bilinear_cohesive(
    *,
    strength: float,
    fracture_energy: float,
    initial_stiffness: float,
    compression_stiffness: float | None = None,
    name: str = "bilinear Mode-I cohesive law",
) -> BilinearCohesiveLaw:
    """Create a bilinear Mode-I cohesive law."""

    return BilinearCohesiveLaw(
        strength=strength,
        fracture_energy=fracture_energy,
        initial_stiffness=initial_stiffness,
        compression_stiffness=compression_stiffness,
        name=name,
    )


def mixed_mode_bilinear_cohesive(
    *,
    normal_strength: float,
    shear_strength: float,
    normal_fracture_energy: float,
    shear_fracture_energy: float,
    normal_stiffness: float,
    tangential_stiffness: float,
    interaction: str = "bk",
    interaction_exponent: float = 1.45,
    compression_stiffness: float | None = None,
    residual_tangential_fraction: float = 0.0,
    friction_coefficient: float = 0.0,
    friction_regularization: float = 1.0e-8,
    name: str = "bilinear mixed-mode cohesive law",
) -> MixedModeBilinearCohesiveLaw:
    """Create a quadratic-initiation, energy-evolution mixed-mode law."""

    return MixedModeBilinearCohesiveLaw(
        normal_strength=normal_strength,
        shear_strength=shear_strength,
        normal_fracture_energy=normal_fracture_energy,
        shear_fracture_energy=shear_fracture_energy,
        normal_stiffness=normal_stiffness,
        tangential_stiffness=tangential_stiffness,
        interaction=interaction,
        interaction_exponent=interaction_exponent,
        compression_stiffness=compression_stiffness,
        residual_tangential_fraction=residual_tangential_fraction,
        friction_coefficient=friction_coefficient,
        friction_regularization=friction_regularization,
        name=name,
    )


def cohesive_surface(
    *,
    law,
    mode: str = "normal",
    name: str = "cohesive surface",
) -> CohesiveSurface:
    """Declare a fixed-path zero-thickness cohesive interface."""

    return CohesiveSurface(law=law, mode=mode, name=name)


def cohesive_characteristic_length(
    *, young: float, fracture_energy: float, strength: float
) -> float:
    """Return the declared scale ``E * Gamma / strength**2``.

    Different cohesive-zone conventions introduce order-one factors.  This
    helper intentionally reports the unscaled definition and leaves the
    convention visible in benchmark metadata.
    """

    values = (float(young), float(fracture_energy), float(strength))
    if any(not isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("young, fracture_energy, and strength must be positive.")
    return float(young * fracture_energy / strength**2)


def _finite_array(value, *, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must contain only finite values.")
    return selected


def _transaction_for_law(law, size: int):
    """Create a cohesive state transaction without fixing one law family."""

    factory = getattr(law, "transaction", None)
    if factory is None or not callable(factory):
        raise TypeError(
            "A cohesive law must provide transaction(size), update semantics, "
            "and a summary contract."
        )
    transaction = factory(int(size))
    required = ("begin", "commit", "rollback", "initialize", "evaluate")
    missing = [name for name in required if not callable(getattr(transaction, name, None))]
    if missing:
        raise TypeError(f"Cohesive transaction is missing operations: {missing}.")
    return transaction


# Clear current names; the ModeI-prefixed classes remain compatible aliases.
CohesiveFacetAssembler = ModeICohesiveFacetAssembler
CohesiveSurfaceAssembler = ModeICohesiveSurfaceAssembler


# Portable state lives in a separate module so the local cohesive material and
# facet kernels remain independently testable.  Re-export it from the public
# interface namespace because users should not need to know the storage owner.
from .cohesive_checkpoint import (  # noqa: E402
    COHESIVE_CHECKPOINT_SCHEMA,
    FacetOwnership,
    deterministic_facet_ownership,
    load_portable_cohesive_state,
    save_portable_cohesive_state,
)


__all__ = [
    "BilinearCohesiveLaw",
    "MixedModeBilinearCohesiveLaw",
    "MixedModeCohesiveTransaction",
    "CohesiveResponse",
    "VectorCohesiveResponse",
    "CohesiveSurface",
    "CohesiveTransaction",
    "COHESIVE_CHECKPOINT_SCHEMA",
    "CohesiveElementTangents",
    "CohesiveFacetAssembler",
    "CohesiveFacetResponse",
    "CohesiveSurfaceAssembler",
    "InterfaceRigidModeAudit",
    "ModeIKinematicsAudit",
    "ModeICohesiveFacetAssembler",
    "ModeICohesiveSurfaceAssembler",
    "FacetOwnership",
    "NamedSplitInterfaceMesh",
    "PairedLineFacets",
    "PairedSurfaceFacets",
    "SplitInterfaceMesh",
    "bilinear_cohesive",
    "audit_mode_i_kinematics",
    "audit_split_interface_rigid_modes",
    "mixed_mode_bilinear_cohesive",
    "cohesive_characteristic_length",
    "cohesive_surface",
    "create_dolfinx_split_mesh",
    "deterministic_facet_ownership",
    "load_portable_cohesive_state",
    "pair_coincident_line_facets",
    "pair_coincident_surface_facets",
    "save_portable_cohesive_state",
    "split_conforming_line_interface",
    "split_conforming_named_interfaces",
    "split_conforming_surface_interface",
    "split_conforming_cell_interface",
]

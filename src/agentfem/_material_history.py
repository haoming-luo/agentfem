# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Material-point procedure orchestration separate from constitutive laws."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from .constitutive.plasticity import (
    ChabocheCombinedHardening,
    ChabocheState,
    J2LinearIsotropicHardening,
    J2PlasticState,
    _linearization,
    von_mises,
)
from .constitutive.viscoelasticity import GeneralizedMaxwell, MaxwellState


def _history_coordinate(value, *, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.ndim != 1 or selected.size < 2:
        raise ValueError(f"{name} must contain at least two one-dimensional values.")
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(np.diff(selected) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return selected.copy()


def _history_temperature(value, *, count: int) -> np.ndarray | None:
    if value is None:
        return None
    selected = np.asarray(value, dtype=float)
    if selected.ndim == 0:
        selected = np.full(count, float(selected))
    if selected.shape != (count,) or not np.all(np.isfinite(selected)):
        raise ValueError("temperature must be finite and scalar or match time.")
    return selected.copy()


@dataclass(frozen=True)
class MaterialLoadingPath:
    """Tensor-valued material loading with exact, refinable physical knots."""

    coordinate: object
    strain: object
    temperature: object | None = None
    name: str = "material_loading_path"
    coordinate_name: str = "time"
    coordinate_unit: str | None = None

    def __post_init__(self) -> None:
        coordinate = _history_coordinate(self.coordinate, name="coordinate")
        strain = np.asarray(self.strain, dtype=float)
        if strain.shape != (coordinate.size, 3, 3):
            raise ValueError("strain must have shape (path points, 3, 3).")
        if not np.all(np.isfinite(strain)):
            raise ValueError("strain must contain only finite values.")
        if not np.allclose(strain, np.swapaxes(strain, 1, 2), rtol=0.0, atol=1.0e-12):
            raise ValueError("Every material-path strain tensor must be symmetric.")
        temperature = _history_temperature(
            self.temperature,
            count=coordinate.size,
        )
        name = str(self.name).strip()
        coordinate_name = str(self.coordinate_name).strip()
        if not name or not coordinate_name:
            raise ValueError("Material path names must be nonempty.")
        strain = strain.copy()
        coordinate.setflags(write=False)
        strain.setflags(write=False)
        if temperature is not None:
            temperature.setflags(write=False)
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "strain", strain)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coordinate_name", coordinate_name)

    @property
    def point_count(self) -> int:
        return int(self.coordinate.size)

    @property
    def segment_count(self) -> int:
        return self.point_count - 1

    @property
    def fingerprint(self) -> str:
        record = {
            "schema": "agentfem.material-loading-path",
            "schema_version": "0.1.0",
            "coordinate_name": self.coordinate_name,
            "coordinate_unit": self.coordinate_unit,
            "coordinate": self.coordinate.tolist(),
            "strain": self.strain.tolist(),
            "temperature": (
                None if self.temperature is None else self.temperature.tolist()
            ),
        }
        encoded = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{sha256(encoded).hexdigest()}"

    def refine(self, substeps_per_segment: int = 2) -> "MaterialLoadingPath":
        """Subdivide every segment without moving or deleting any source knot."""

        substeps = int(substeps_per_segment)
        if substeps < 1 or substeps != substeps_per_segment:
            raise ValueError("substeps_per_segment must be a positive integer.")
        if substeps == 1:
            return self
        coordinates = []
        strains = []
        temperatures = [] if self.temperature is not None else None
        for index in range(self.segment_count):
            fractions = np.arange(substeps, dtype=float) / substeps
            start = self.coordinate[index]
            end = self.coordinate[index + 1]
            for fraction in fractions:
                coordinates.append(start + fraction * (end - start))
                strains.append(
                    self.strain[index]
                    + fraction * (self.strain[index + 1] - self.strain[index])
                )
                if temperatures is not None:
                    temperatures.append(
                        self.temperature[index]
                        + fraction
                        * (self.temperature[index + 1] - self.temperature[index])
                    )
        coordinates.append(self.coordinate[-1])
        strains.append(self.strain[-1])
        if temperatures is not None:
            temperatures.append(self.temperature[-1])
        return MaterialLoadingPath(
            coordinate=np.asarray(coordinates),
            strain=np.asarray(strains),
            temperature=(
                None if temperatures is None else np.asarray(temperatures)
            ),
            name=f"{self.name}_refined",
            coordinate_name=self.coordinate_name,
            coordinate_unit=self.coordinate_unit,
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.material-loading-path",
            "schema_version": "0.1.0",
            "name": self.name,
            "coordinate_name": self.coordinate_name,
            "coordinate_unit": self.coordinate_unit,
            "point_count": self.point_count,
            "segment_count": self.segment_count,
            "strain_shape": (3, 3),
            "temperature": self.temperature is not None,
            "interpolation": "piecewise_linear_exact_knots",
            "fingerprint": self.fingerprint,
        }


def material_strain_path(
    coordinate,
    strain,
    *,
    temperature=None,
    name: str = "material_strain_path",
    coordinate_name: str = "time",
    coordinate_unit: str | None = None,
) -> MaterialLoadingPath:
    """Create a strain-controlled material path with explicit physical knots."""

    return MaterialLoadingPath(
        coordinate=coordinate,
        strain=strain,
        temperature=temperature,
        name=name,
        coordinate_name=coordinate_name,
        coordinate_unit=coordinate_unit,
    )


def _plastic_stored_energy(material, strain, stress, state):
    elastic = 0.5 * float(np.tensordot(stress, strain - state.plastic_strain))
    equivalent = float(state.equivalent_plastic_strain)
    isotropic = 0.0
    kinematic = 0.0
    if isinstance(material, J2LinearIsotropicHardening):
        isotropic = 0.5 * material.hardening_modulus * equivalent**2
    else:
        if material.isotropic_rate > 0.0:
            isotropic = material.isotropic_saturation * (
                equivalent
                + (np.exp(-material.isotropic_rate * equivalent) - 1.0)
                / material.isotropic_rate
            )
        kinematic = sum(
            3.0 * float(np.tensordot(alpha, alpha)) / (4.0 * modulus)
            for alpha, modulus in zip(
                state.backstresses,
                material.backstress_moduli,
                strict=True,
            )
        )
    return float(elastic), float(isotropic), float(kinematic)


def _plastic_yield_function(material, stress, state) -> float:
    shifted = stress
    if isinstance(state, ChabocheState):
        shifted = stress - state.total_backstress
    return float(
        von_mises(shifted)
        - material.current_yield_stress(state.equivalent_plastic_strain)
    )


@dataclass(frozen=True)
class PlasticMaterialHistoryResponse:
    """Accepted J2/Chaboche path with typed state and partial energy evidence."""

    path: MaterialLoadingPath
    stress: np.ndarray
    plastic_strain: np.ndarray
    equivalent_plastic_strain: np.ndarray
    yield_function: np.ndarray
    elastic: np.ndarray
    plastic_multiplier_increment: np.ndarray
    algorithmic_tangent: np.ndarray | None
    backstresses: np.ndarray | None
    elastic_stored_energy: np.ndarray
    isotropic_hardening_stored_energy: np.ndarray
    kinematic_hardening_stored_energy: np.ndarray
    plastic_work: np.ndarray
    reference_yield_dissipation: np.ndarray
    final_state: J2PlasticState | ChabocheState
    material: J2LinearIsotropicHardening | ChabocheCombinedHardening
    linearization: str
    name: str

    def to_result(self):
        from . import procedures
        from .results import SimulationResult

        result = SimulationResult(name=self.name)
        histories = {
            "strain": self.path.strain,
            "stress": self.stress,
            "plastic_strain": self.plastic_strain,
            "equivalent_plastic_strain": self.equivalent_plastic_strain,
            "yield_function": self.yield_function,
            "elastic_state": self.elastic.astype(float),
            "plastic_multiplier_increment": self.plastic_multiplier_increment,
            "elastic_stored_energy": self.elastic_stored_energy,
            "isotropic_hardening_stored_energy": (
                self.isotropic_hardening_stored_energy
            ),
            "kinematic_hardening_stored_energy": (
                self.kinematic_hardening_stored_energy
            ),
            "plastic_work": self.plastic_work,
            "reference_yield_dissipation": self.reference_yield_dissipation,
        }
        if self.algorithmic_tangent is not None:
            histories["algorithmic_tangent"] = self.algorithmic_tangent
        if self.backstresses is not None:
            histories["backstress_components"] = self.backstresses
            histories["total_backstress"] = np.sum(self.backstresses, axis=1)
        result.add_histories(
            self.path.coordinate,
            histories,
            abscissa_name=self.path.coordinate_name,
            abscissa_unit=self.path.coordinate_unit,
            descriptions={
                "plastic_work": (
                    "Cumulative signed end-step stress work on the accepted "
                    "plastic-strain increments; it is not a dissipation claim."
                ),
                "reference_yield_dissipation": (
                    "Cumulative sigma_y0 times PEEQ component. Chaboche "
                    "dynamic-recovery dissipation is not included."
                ),
                "elastic_state": "One for an elastic update and zero for a plastic update.",
            },
        )
        plastic_mask = ~self.elastic
        active_residual = (
            float(np.max(np.abs(self.yield_function[plastic_mask])))
            if np.any(plastic_mask)
            else 0.0
        )
        result.add_quantities(
            {
                "final_equivalent_plastic_strain": (
                    self.equivalent_plastic_strain[-1]
                ),
                "maximum_plastic_consistency_residual": active_residual,
                "accepted_increment_count": self.path.segment_count,
            },
            kind="constitutive_history",
        )
        chaboche = isinstance(self.material, ChabocheCombinedHardening)
        result.metadata["procedure"] = procedures.material_history().summary()
        result.metadata["material_history"] = {
            "schema": "agentfem.plastic-material-history.v1",
            "control": "strain",
            "linearization": self.linearization,
            "path_fingerprint": self.path.fingerprint,
            "state_commit": "accepted_increment",
            "restart": {
                "in_memory_final_state": True,
                "portable_checkpoint": False,
            },
        }
        result.metadata["energy"] = {
            "plastic_work": "signed_work_not_dissipation",
            "reference_yield_dissipation": "available",
            "dynamic_recovery_dissipation": (
                "unavailable" if chaboche else "not_applicable"
            ),
            "irreversible_dissipation": (
                "unavailable_dynamic_recovery_not_closed"
                if chaboche
                else "reference_yield_dissipation"
            ),
            "complete_discrete_energy_balance": False,
        }
        result.metadata["scope"] = {
            "level": "material_point",
            "global_fem_provider": False,
            "kinematics": "three_dimensional_small_strain",
        }
        result.add_scientific_inputs(
            material=self.material,
            loading_path=self.path,
        )
        return result


@dataclass
class PlasticMaterialHistoryStep:
    """Atomic strain-controlled material history for J2-family materials."""

    material: J2LinearIsotropicHardening | ChabocheCombinedHardening
    path: MaterialLoadingPath
    initial_state: J2PlasticState | ChabocheState | None = None
    linearization: str = "none"
    name: str = "plastic_material_history"
    last_response: PlasticMaterialHistoryResponse | None = None

    @property
    def procedure(self):
        from . import procedures

        return procedures.material_history()

    def solve(self) -> PlasticMaterialHistoryResponse:
        path = self.path
        if not isinstance(path, MaterialLoadingPath):
            raise TypeError("Plastic material history requires MaterialLoadingPath.")
        linearization = _linearization(self.linearization)
        if path.temperature is not None:
            raise ValueError(
                "Small-strain J2/Chaboche history does not yet accept temperature."
            )
        if self.initial_state is None:
            if not np.allclose(path.strain[0], 0.0, rtol=0.0, atol=1.0e-14):
                raise ValueError(
                    "A history without initial_state must start at zero strain."
                )
            state = self.material.initial_state() if isinstance(
                self.material, ChabocheCombinedHardening
            ) else J2PlasticState()
        else:
            state = self.initial_state
        if isinstance(self.material, ChabocheCombinedHardening):
            if not isinstance(state, ChabocheState):
                raise TypeError("Chaboche history requires ChabocheState.")
        elif not isinstance(state, J2PlasticState):
            raise TypeError("J2 history requires J2PlasticState.")

        updates = []
        for strain in path.strain:
            update = self.material.update(
                strain,
                state,
                linearization=linearization,
            )
            updates.append(update)
            state = update.state

        stress = np.asarray([item.stress for item in updates])
        plastic_strain = np.asarray(
            [item.state.plastic_strain for item in updates]
        )
        equivalent = np.asarray(
            [item.state.equivalent_plastic_strain for item in updates]
        )
        yield_function = np.asarray(
            [
                _plastic_yield_function(self.material, item.stress, item.state)
                for item in updates
            ]
        )
        elastic = np.asarray([item.elastic for item in updates], dtype=bool)
        increments = np.asarray(
            [item.plastic_multiplier_increment for item in updates]
        )
        tangent = (
            None
            if linearization == "none"
            else np.asarray([item.algorithmic_tangent for item in updates])
        )
        backstresses = (
            np.asarray([item.state.backstresses for item in updates])
            if isinstance(self.material, ChabocheCombinedHardening)
            else None
        )
        stored = np.asarray(
            [
                _plastic_stored_energy(
                    self.material,
                    path.strain[index],
                    item.stress,
                    item.state,
                )
                for index, item in enumerate(updates)
            ]
        )
        plastic_work = np.zeros(path.point_count)
        for index in range(1, path.point_count):
            plastic_work[index] = plastic_work[index - 1] + float(
                np.tensordot(
                    stress[index],
                    plastic_strain[index] - plastic_strain[index - 1],
                )
            )
        response = PlasticMaterialHistoryResponse(
            path=path,
            stress=stress,
            plastic_strain=plastic_strain,
            equivalent_plastic_strain=equivalent,
            yield_function=yield_function,
            elastic=elastic,
            plastic_multiplier_increment=increments,
            algorithmic_tangent=tangent,
            backstresses=backstresses,
            elastic_stored_energy=stored[:, 0],
            isotropic_hardening_stored_energy=stored[:, 1],
            kinematic_hardening_stored_energy=stored[:, 2],
            plastic_work=plastic_work,
            reference_yield_dissipation=(
                self.material.yield_stress * equivalent
            ),
            final_state=state,
            material=self.material,
            linearization=linearization,
            name=self.name,
        )
        self.last_response = response
        return response

    def solve_result(self):
        return self.solve().to_result()


def _stored_energy(material: GeneralizedMaxwell, state: MaxwellState) -> float:
    equilibrium = 0.5 * material.equilibrium_modulus * float(
        np.sum(state.strain**2)
    )
    reshape = (material.branch_moduli.size,) + (1,) * state.strain.ndim
    branches = 0.5 * float(
        np.sum(state.overstress**2 / material.branch_moduli.reshape(reshape))
    )
    return equilibrium + branches


@dataclass(frozen=True)
class GeneralizedMaxwellHistoryResponse:
    """Accepted constitutive path and energy evidence from one history."""

    time: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    branch_overstress: np.ndarray
    algorithmic_modulus: np.ndarray
    stored_energy: np.ndarray
    dissipated_energy: np.ndarray
    mechanical_work: np.ndarray
    energy_balance_error: np.ndarray
    temperature: np.ndarray | None
    final_state: MaxwellState
    material: GeneralizedMaxwell
    name: str

    def to_result(self):
        """Convert the accepted path to the common scientific result."""

        from . import procedures
        from .results import SimulationResult

        result = SimulationResult(name=self.name)
        result.add_histories(
            self.time,
            {
                "strain": self.strain,
                "stress": self.stress,
                "branch_overstress": self.branch_overstress,
                "algorithmic_modulus": self.algorithmic_modulus,
                "stored_energy": self.stored_energy,
                "dissipated_energy": self.dissipated_energy,
                "mechanical_work": self.mechanical_work,
                "energy_balance_error": self.energy_balance_error,
            },
            abscissa_name="time",
            abscissa_unit="s",
            descriptions={
                "branch_overstress": "Accepted Maxwell-branch internal state.",
                "algorithmic_modulus": (
                    "Instantaneous modulus at the initial state and exact "
                    "increment tangent thereafter."
                ),
                "stored_energy": "Recoverable generalized-Maxwell energy density.",
                "dissipated_energy": "Cumulative nonnegative viscous dissipation.",
                "mechanical_work": "Cumulative constitutive work density.",
                "energy_balance_error": (
                    "Mechanical work minus stored energy and dissipation."
                ),
            },
        )
        if self.temperature is not None:
            result.add_history(
                "temperature",
                self.time,
                self.temperature,
                abscissa_name="time",
                abscissa_unit="s",
                description="Temperature consumed by the accepted shift update.",
            )
        result.add_quantities(
            {
                "final_stress": self.stress[-1],
                "final_dissipated_energy": self.dissipated_energy[-1],
                "maximum_absolute_energy_balance_error": np.max(
                    np.abs(self.energy_balance_error)
                ),
            },
            kind="constitutive_history",
        )
        result.metadata["procedure"] = procedures.viscoelastic_history().summary()
        result.metadata["state"] = {
            "schema": "agentfem.maxwell-state.v1",
            "branch_count": int(self.final_state.overstress.shape[0]),
            "value_shape": tuple(self.final_state.strain.shape),
            "accepted_increments": int(self.time.size - 1),
            "restartable": True,
            "increment_transaction": "trial_update_then_commit",
        }
        result.metadata["scope"] = {
            "level": "material_point",
            "global_fem_provider": False,
            "kinematics": "componentwise_small_strain",
        }
        scientific_inputs = {
            "material": self.material,
            "time": self.time,
            "strain": self.strain,
        }
        if self.temperature is not None:
            scientific_inputs["temperature"] = self.temperature
        result.add_scientific_inputs(scientific_inputs)
        return result


@dataclass
class GeneralizedMaxwellHistoryStep:
    """Atomic, restartable material-history execution for one Prony spectrum."""

    material: GeneralizedMaxwell
    time: object
    strain: object
    temperature: object | None = None
    initial_state: MaxwellState | None = None
    name: str = "generalized_maxwell_history"
    last_response: GeneralizedMaxwellHistoryResponse | None = None

    @property
    def procedure(self):
        from . import procedures

        return procedures.viscoelastic_history()

    def solve(self) -> GeneralizedMaxwellHistoryResponse:
        time = _history_coordinate(self.time, name="time")
        strain = np.asarray(self.strain, dtype=float)
        if strain.shape[0:1] != time.shape or not np.all(np.isfinite(strain)):
            raise ValueError(
                "strain must be finite and its first dimension must match time."
            )
        value_shape = strain.shape[1:]
        temperature = _history_temperature(self.temperature, count=time.size)
        state = (
            self.material.initial_state(np.zeros(value_shape, dtype=float))
            if self.initial_state is None
            else self.initial_state.copy()
        )
        if state.overstress.shape != (
            self.material.branch_moduli.size,
            *value_shape,
        ):
            raise ValueError("initial_state shape does not match the strain history.")
        if not np.allclose(state.strain, strain[0], rtol=0.0, atol=1.0e-14):
            raise ValueError(
                "The first strain sample must match the accepted initial_state strain."
            )

        stress = np.empty_like(strain)
        branch_overstress = np.empty(
            (time.size, self.material.branch_moduli.size, *value_shape),
            dtype=float,
        )
        tangent = np.empty(time.size, dtype=float)
        stored = np.empty(time.size, dtype=float)
        dissipated = np.empty(time.size, dtype=float)
        work = np.empty(time.size, dtype=float)
        stress[0] = self.material.equilibrium_modulus * state.strain + np.sum(
            state.overstress,
            axis=0,
        )
        branch_overstress[0] = state.overstress
        tangent[0] = self.material.instantaneous_modulus
        stored[0] = _stored_energy(self.material, state)
        dissipated[0] = state.dissipated_energy
        work[0] = stored[0] + dissipated[0]

        for index in range(1, time.size):
            accepted = state.snapshot()
            try:
                update = self.material.update(
                    state,
                    strain[index],
                    float(time[index] - time[index - 1]),
                    temperature=(
                        None if temperature is None else float(temperature[index])
                    ),
                )
                state.commit(update)
            except Exception:
                state.restore(accepted)
                raise
            stress[index] = update.stress
            branch_overstress[index] = update.overstress
            tangent[index] = update.algorithmic_modulus
            stored[index] = _stored_energy(self.material, state)
            dissipated[index] = state.dissipated_energy
            work[index] = work[index - 1] + update.mechanical_work_increment

        response = GeneralizedMaxwellHistoryResponse(
            time=time,
            strain=strain.copy(),
            stress=stress,
            branch_overstress=branch_overstress,
            algorithmic_modulus=tangent,
            stored_energy=stored,
            dissipated_energy=dissipated,
            mechanical_work=work,
            energy_balance_error=work - stored - dissipated,
            temperature=temperature,
            final_state=state.copy(),
            material=self.material,
            name=self.name,
        )
        self.last_response = response
        return response

    def solve_result(self):
        """Execute the material history and return a common result contract."""

        return self.solve().to_result()


__all__ = [
    "GeneralizedMaxwellHistoryResponse",
    "GeneralizedMaxwellHistoryStep",
    "MaterialLoadingPath",
    "PlasticMaterialHistoryResponse",
    "PlasticMaterialHistoryStep",
    "material_strain_path",
]

"""Material-point procedure orchestration separate from constitutive laws."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


__all__ = ["GeneralizedMaxwellHistoryResponse", "GeneralizedMaxwellHistoryStep"]

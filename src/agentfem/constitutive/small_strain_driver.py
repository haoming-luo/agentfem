# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Atomic quadrature driver for generic small-strain materials."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np
from mpi4py import MPI

from .quadrature import MaterialQuadratureState
from .quadrature import QuadratureField
from .small_strain_material import (
    SmallStrainMaterialBatchInput,
    SmallStrainMaterialBatchOutput,
    SmallStrainUserMaterial,
    update_small_strain_material_batch,
)


class SmallStrainMaterialBatchError(RuntimeError):
    """A rank-local material update failed before any trial state was accepted."""

    code = "AFM-MATERIAL-BATCH-001"


@dataclass(frozen=True)
class SmallStrainQuadratureBatchResult:
    """Validated rank-local response and transaction status."""

    response: SmallStrainMaterialBatchOutput
    committed: bool

    @property
    def point_count(self) -> int:
        return self.response.point_count

    def summary(self) -> dict[str, object]:
        statuses = self.response.applicability_status
        return {
            "kind": "small_strain_quadrature_batch_result",
            "point_count": self.point_count,
            "committed": self.committed,
            "applicability_counts": {
                name: statuses.count(name)
                for name in ("in_domain", "warning", "out_of_domain", "invalid_state")
                if name in statuses
            },
            "minimum_suggested_time_scale": float(
                np.min(self.response.suggested_time_scale)
            ),
        }


@dataclass
class SmallStrainMaterialQuadratureResponse:
    """Live stress, tangent, energy and state fields for global Newton."""

    state: MaterialQuadratureState
    accepted_strain: QuadratureField
    trial_strain: QuadratureField
    stress: QuadratureField
    tangent: QuadratureField
    stored_energy_density: QuadratureField
    dissipated_energy_density: QuadratureField
    energy_density_components: dict[str, QuadratureField] = field(default_factory=dict)
    last_response: SmallStrainMaterialBatchOutput | None = None

    @classmethod
    def create(cls, domain, state_schema, *, degree: int = 2, scheme: str = "default"):
        common = {"degree": int(degree), "scheme": str(scheme)}
        return cls(
            state=MaterialQuadratureState.create(
                domain,
                state_schema,
                **common,
            ),
            accepted_strain=QuadratureField.create(
                domain, name="E_ACCEPTED", value_shape=(6,), **common
            ),
            trial_strain=QuadratureField.create(
                domain, name="E_TRIAL", value_shape=(6,), **common
            ),
            stress=QuadratureField.create(domain, name="S", value_shape=(6,), **common),
            tangent=QuadratureField.create(
                domain, name="DDSDDE", value_shape=(6, 6), **common
            ),
            stored_energy_density=QuadratureField.create(
                domain, name="SENER", **common
            ),
            dissipated_energy_density=QuadratureField.create(
                domain, name="DENER", **common
            ),
        )

    @property
    def domain(self):
        return self.state.domain

    @property
    def degree(self) -> int:
        return self.state.degree

    @property
    def scheme(self) -> str:
        return self.state.scheme

    @property
    def measure(self):
        return self.state.measure

    @property
    def transaction(self):
        return self.state.transaction

    def compile_strain(self, expression):
        return self.state.compile_expression(expression, value_shape=(6,))

    def evaluate_strain(self, expression) -> np.ndarray:
        return self.state.evaluate_expression(expression, value_shape=(6,))

    def update(self, strain_values, material) -> dict[str, float | int]:
        selected = np.asarray(strain_values, dtype=float).reshape((-1, 6))
        result = update_small_strain_quadrature_state(
            material,
            self.state,
            strain_old=self.accepted_strain.values,
            strain_new=selected,
            time=1.0,
            time_increment=1.0,
            parameters=material.parameters,
            commit=False,
        )
        response = result.response
        self.trial_strain.assign(selected)
        self.stress.assign(response.cauchy_stress)
        self.tangent.assign(response.consistent_tangent)
        self.stored_energy_density.assign(response.stored_energy_density)
        self.dissipated_energy_density.assign(response.dissipated_energy_density)
        for name, values in response.energy_density_components.items():
            selected = self.energy_density_components.get(name)
            if selected is None:
                selected = QuadratureField.create(
                    self.domain,
                    name=f"ENERGY_{name.upper()}",
                    degree=self.degree,
                    scheme=self.scheme,
                )
                self.energy_density_components[name] = selected
            selected.assign(values)
        self.last_response = response
        owned = int(
            self.domain.topology.index_map(self.domain.topology.dim).size_local
        ) * len(self.stress.points)
        plastic = np.asarray(
            response.diagnostics.get("plastic", np.zeros(response.point_count)),
            dtype=bool,
        )
        increment = np.asarray(
            response.diagnostics.get(
                "plastic_increment", np.zeros(response.point_count)
            ),
            dtype=float,
        )
        return {
            "points": int(self.domain.comm.allreduce(owned)),
            "plastic_points": int(
                self.domain.comm.allreduce(
                    int(np.count_nonzero(plastic[:owned])), op=MPI.SUM
                )
            ),
            "maximum_plastic_increment": float(
                self.domain.comm.allreduce(
                    float(np.max(increment[:owned], initial=0.0)), op=MPI.MAX
                )
            ),
        }

    def commit(self) -> None:
        self.state.commit()
        self.accepted_strain.assign(self.trial_strain.values)

    def rollback(self) -> None:
        self.state.rollback()
        self.trial_strain.assign(self.accepted_strain.values)

    def snapshot(self) -> dict[str, object]:
        return {
            "material_state": self.state.snapshot(),
            "accepted_strain": self.accepted_strain.values.copy(),
        }

    def restore(self, snapshot) -> None:
        self.state.restore(snapshot["material_state"])
        self.accepted_strain.assign(snapshot["accepted_strain"])
        self.rollback()

    def equivalent_stress(self) -> QuadratureField:
        values = np.asarray(self.stress.values, dtype=float)
        mean = values[:, :3].mean(axis=1, keepdims=True)
        deviator = values.copy()
        deviator[:, :3] -= mean
        weights = np.asarray((1.0, 1.0, 1.0, 2.0, 2.0, 2.0))
        mises = np.sqrt(np.maximum(0.0, 1.5 * np.sum(weights * deviator**2, axis=1)))
        output = QuadratureField.create(
            self.domain,
            name="MISES",
            degree=self.degree,
            scheme=self.scheme,
        )
        output.assign(mises)
        return output

    def summary(self) -> dict[str, object]:
        statuses = (
            ()
            if self.last_response is None
            else self.last_response.applicability_status
        )
        return {
            "kind": "small_strain_material_quadrature_response",
            "degree": self.degree,
            "scheme": self.scheme,
            "state": self.state.summary(),
            "applicability_counts": {
                name: statuses.count(name)
                for name in ("in_domain", "warning", "out_of_domain", "invalid_state")
                if name in statuses
            },
            "energy_density_components": tuple(sorted(self.energy_density_components)),
        }


def update_small_strain_quadrature_state(
    material: SmallStrainUserMaterial,
    state: MaterialQuadratureState,
    *,
    strain_old,
    strain_new,
    time: float,
    time_increment: float,
    parameters,
    temperature=None,
    temperature_increment=None,
    field_variables=None,
    commit: bool = False,
    allow_out_of_domain: bool = False,
) -> SmallStrainQuadratureBatchResult:
    """Update every visible integration point as one MPI-atomic transaction.

    No trial field is changed until every rank has produced a valid complete
    response.  Callers may reuse the trial response during Newton iterations,
    then commit only after global equilibrium converges.
    """

    if state.state_schema.summary() != material.state_schema.summary():
        raise ValueError("Quadrature and material state schemas differ.")
    old = np.asarray(strain_old, dtype=float)
    new = np.asarray(strain_new, dtype=float)
    point_count = len(state.reference_field.values)
    if old.shape != (point_count, 6) or new.shape != old.shape:
        raise ValueError(
            f"Quadrature strains must both have shape ({point_count}, 6); "
            f"received old={old.shape}, new={new.shape}."
        )
    request = SmallStrainMaterialBatchInput(
        strain_old=old,
        strain_new=new,
        time=time,
        time_increment=time_increment,
        parameters=parameters,
        state_old=state.committed_state_vectors(),
        state_schema=state.state_schema,
        temperature=temperature,
        temperature_increment=temperature_increment,
        field_variables=field_variables,
    )
    response = None
    problem = None
    try:
        response = update_small_strain_material_batch(material, request)
        if any(status == "invalid_state" for status in response.applicability_status):
            raise ValueError(
                "Material returned invalid_state for an integration point."
            )
        if not allow_out_of_domain and any(
            status == "out_of_domain" for status in response.applicability_status
        ):
            raise ValueError(
                "Material reported out_of_domain. Explicitly configure a reviewed "
                "fallback or allow_out_of_domain for exploratory material-point work."
            )
    except Exception as exc:
        problem = f"{type(exc).__name__}: {exc}"
    problems = state.domain.comm.allgather(problem)
    if any(item is not None for item in problems):
        state.rollback()
        details = "; ".join(
            f"rank {rank}: {item}"
            for rank, item in enumerate(problems)
            if item is not None
        )
        raise SmallStrainMaterialBatchError(
            f"{SmallStrainMaterialBatchError.code}: atomic small-strain material "
            f"update failed ({details})."
        )
    state.assign_trial_state_vectors(response.state_new)
    if commit:
        state.commit()
    return SmallStrainQuadratureBatchResult(response=response, committed=bool(commit))


__all__ = [
    "SmallStrainMaterialBatchError",
    "SmallStrainMaterialQuadratureResponse",
    "SmallStrainQuadratureBatchResult",
    "update_small_strain_quadrature_state",
]

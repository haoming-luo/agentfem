# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Global equilibrium for generic small-strain material providers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from .. import amplitudes
from .. import steps as step_controls
from ..constitutive import SmallStrainMaterialQuadratureResponse
from ..learning import LearnedConstitutiveMaterial
from ..solvers import newton
from .plasticity import J2PlasticityStep


def _strain_voigt(displacement):
    dimension = int(displacement.ufl_shape[0])
    strain = ufl.sym(ufl.grad(displacement))
    if dimension == 3:
        return ufl.as_vector(
            (
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                strain[0, 1],
                strain[1, 2],
                strain[0, 2],
            )
        )
    if dimension == 2:
        return ufl.as_vector((strain[0, 0], strain[1, 1], 0.0, strain[0, 1], 0.0, 0.0))
    raise NotImplementedError(
        "Generic small-strain materials require 2D plane strain or 3D."
    )


@dataclass(frozen=True)
class SmallStrainMaterialEnergyFrame:
    step_coordinate: float
    load_amplitude: float
    stored_energy: float
    dissipated_energy: float
    internal_energy: float
    generalized_reaction: float | None
    external_work: float | None
    energy_balance_error: float | None

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, record) -> "SmallStrainMaterialEnergyFrame":
        return cls(
            **{
                name: (None if record.get(name) is None else float(record[name]))
                for name in cls.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class SmallStrainMaterialLoadPathInfo:
    """Provider-neutral nonlinear load-path evidence."""

    increments: tuple[object, ...]
    attempts: tuple[object, ...]
    incrementation: object

    @property
    def converged(self) -> bool:
        return (
            bool(self.increments)
            and all(item.converged for item in self.increments)
            and bool(self.attempts)
            and self.attempts[-1].converged
        )

    @property
    def completed_step(self) -> bool:
        return self.converged and abs(self.increments[-1].load_factor - 1.0) <= 1.0e-12

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "small_strain_material_load_path",
            "converged": self.converged,
            "completed_step": self.completed_step,
            "accepted_increment_count": len(self.increments),
            "attempt_count": len(self.attempts),
            "incrementation": self.incrementation.summary(),
            "increments": [item.as_dict() for item in self.increments],
            "attempts": [item.as_dict() for item in self.attempts],
        }


@dataclass
class SmallStrainMaterialStep(J2PlasticityStep):
    """Ordinary nonlinear Step for provider-backed small-strain materials."""

    material: LearnedConstitutiveMaterial
    state: SmallStrainMaterialQuadratureResponse

    def __post_init__(self) -> None:
        # The inherited Newton lifecycle expects a constitutive-specific
        # evaluator. Native J2 stores a symmetric tensor, whereas this public
        # protocol stores the declared six-component tensor-shear vector.
        self._strain_evaluator = self.state.compile_strain(_strain_voigt(self.solution))

    def solve(self, *, until: float = 1.0):
        """Advance the shared nonlinear path without leaking J2 result language."""

        solution = super().solve(until=until)
        info = self.last_solve_info
        self.last_solve_info = SmallStrainMaterialLoadPathInfo(
            tuple(info.increments),
            tuple(info.attempts),
            info.incrementation,
        )
        return solution

    def _integral(self, field) -> float:
        local = fem.assemble_scalar(fem.form(field.function * self.state.measure))
        return float(self.state.domain.comm.allreduce(local, op=MPI.SUM))

    def _record_energy(self, step_coordinate: float) -> None:
        stored = self._integral(self.state.stored_energy_density)
        increment_dissipation = self._integral(self.state.dissipated_energy_density)
        dissipated = max(0.0, increment_dissipation)
        if self.energy_history:
            dissipated += self.energy_history[-1].dissipated_energy
        internal = stored + dissipated
        amplitude = float(self.amplitude(step_coordinate))
        generalized = self._generalized_reaction()
        if generalized is None:
            external = None
            balance = None
        else:
            previous_amplitude = 0.0
            previous_reaction = 0.0
            previous_work = 0.0
            if self.energy_history:
                previous = self.energy_history[-1]
                previous_amplitude = previous.load_amplitude
                previous_reaction = float(previous.generalized_reaction)
                previous_work = float(previous.external_work)
            external = previous_work + 0.5 * (previous_reaction + generalized) * (
                amplitude - previous_amplitude
            )
            balance = external - internal
        self.energy_history.append(
            SmallStrainMaterialEnergyFrame(
                step_coordinate=float(step_coordinate),
                load_amplitude=amplitude,
                stored_energy=stored,
                dissipated_energy=dissipated,
                internal_energy=internal,
                generalized_reaction=generalized,
                external_work=external,
                energy_balance_error=balance,
            )
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "small_strain_material_step",
            "name": self.name,
            "study": None if self.study is None else self.study.summary(),
            "procedure": self.procedure.summary(),
            "material": self.material.summary(),
            "state": self.state.summary(),
            "incrementation": self.incrementation.summary(),
            "solver": self.solver_options.summary(),
            "num_bcs": len(self.bcs),
            "accepted_load_factor": self.accepted_load_factor,
            "last_solve": None
            if self.last_solve_info is None
            else self.last_solve_info.as_dict(),
        }

    def solve_result(self, *, output=None, strict_output: bool = False, metadata=None):
        from ..results import add_execution_trace, complete_result, from_solution

        solution = (
            self.solve() if self.accepted_load_factor < 1.0 - 1.0e-12 else self.solution
        )
        result = from_solution(
            solution,
            name=self.name,
            metadata={
                "step": self.summary(),
                "material": self.material.summary(),
                **({} if metadata is None else dict(metadata)),
            },
        )
        add_execution_trace(result, self.execution_events)
        result.add_scientific_inputs(
            learned_constitutive=self.material.specification,
        )
        result.add_field(
            "S",
            self.state.stress.function,
            unit="Pa",
            location="quadrature_points",
            description="Cauchy stress in declared Voigt order.",
            processing={
                "representation": "quadrature_values",
                "component_order": self.material.tangent_convention.component_order,
                "shear_convention": self.material.tangent_convention.shear_convention,
            },
        )
        result.add_field(
            "MISES",
            self.state.equivalent_stress().function,
            unit="Pa",
            location="quadrature_points",
            description="Pointwise von Mises stress.",
        )
        for variable in self.material.state_schema.variables:
            result.add_field(
                variable.output_name or variable.name,
                self.state.state.committed[variable.name].function,
                unit=variable.unit,
                location="quadrature_points",
                description=variable.description,
                processing={"representation": "committed_material_state"},
            )
        for name, component in sorted(self.state.energy_density_components.items()):
            result.add_field(
                f"ENERGY_{name.upper()}",
                component.function,
                unit="J/m^3",
                location="quadrature_points",
                description=f"Provider-declared {name.replace('_', ' ')} density.",
                processing={
                    "representation": "quadrature_values",
                    "source": "learned_constitutive_provider",
                },
            )
        if self.state.last_response is not None:
            diagnostics = self.state.last_response.diagnostics
            inference = np.asarray(
                diagnostics.get("inference_seconds_per_point", (0.0,)),
                dtype=float,
            )
            result.add_quantities(
                {
                    "maximum_yield_residual": float(
                        np.max(np.abs(diagnostics.get("yield_residual", (0.0,))))
                    ),
                    "out_of_domain_integration_points": self.state.last_response.applicability_status.count(
                        "out_of_domain"
                    ),
                    "warning_integration_points": self.state.last_response.applicability_status.count(
                        "warning"
                    ),
                    "latest_batch_inference_seconds": float(np.sum(inference)),
                    "cutback_or_retry_attempts": max(
                        0,
                        len(self.attempted_increments) - len(self.accepted_increments),
                    ),
                },
                kind="diagnostic",
            )
        if self.energy_history:
            coordinate = [item.step_coordinate for item in self.energy_history]
            result.add_histories(
                coordinate,
                {
                    "stored_energy": [
                        item.stored_energy for item in self.energy_history
                    ],
                    "dissipated_energy": [
                        item.dissipated_energy for item in self.energy_history
                    ],
                    "internal_energy": [
                        item.internal_energy for item in self.energy_history
                    ],
                },
                abscissa_name="step_coordinate",
                abscissa_unit=None,
            )
        return complete_result(
            self,
            result,
            output=output,
            strict_output=strict_output,
        )

    def save_checkpoint(self, path, *, portable: bool | None = None):
        """Save one MPI-portable checkpoint without serializing provider code."""

        from ..checkpointing import (
            atomic_write_text,
            checkpoint_file_record,
            save_portable_state_bundle,
        )
        from ..results import CheckpointRecord

        if portable is False:
            raise ValueError(
                "Learned-material checkpoints use the portable format on every "
                "rank count; executable provider objects are never serialized."
            )
        selected = Path(path)
        if selected.suffix:
            selected = selected.with_suffix("")
        manifest = selected.with_name(selected.name + ".checkpoint.json")
        bundle = save_portable_state_bundle(manifest, state={"U": self.solution})
        quadrature = self.state.state.save(
            manifest.with_name(f"{selected.name}.{bundle['generation']}.quadrature"),
            material=self.material,
        )
        payload = {
            "schema": "agentfem.small-strain-material-checkpoint.v1",
            "step_identity": self._portable_checkpoint_identity(),
            "coordinate": float(self.accepted_load_factor),
            "nodal_state": bundle["record"],
            "nodal_identity": bundle["identities"],
            "quadrature_state": checkpoint_file_record(quadrature),
            "accepted_increments": [
                item.as_dict() for item in self.accepted_increments
            ],
            "attempted_increments": [
                item.as_dict() for item in self.attempted_increments
            ],
            "execution_events": [item.as_dict() for item in self.execution_events],
            "energy_history": [item.as_dict() for item in self.energy_history],
            "next_increment_size": self.next_increment_size,
        }
        comm = self.state.domain.comm
        error = None
        if comm.rank == 0:
            try:
                atomic_write_text(
                    manifest, json.dumps(payload, indent=2, sort_keys=True) + "\n"
                )
            except Exception as exc:  # pragma: no cover - filesystem failure
                error = f"{type(exc).__name__}: {exc}"
        error = comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(
                f"Small-strain material checkpoint write failed: {error}"
            )
        comm.barrier()
        record = CheckpointRecord(
            name=f"{self.name}_{self.accepted_load_factor:g}",
            path=manifest,
            schema=payload["schema"],
            step_name=self.name,
            coordinate_name="load_factor",
            coordinate_value=self.accepted_load_factor,
            portable=True,
            metadata={
                "state_variables": (
                    "U",
                    *tuple(self.state.transaction.names),
                ),
                "provider": self.material.specification.provider,
                "specification_fingerprint": (self.material.specification.fingerprint),
            },
        )
        self.checkpoints.append(record)
        return manifest

    def load_checkpoint(self, path) -> None:
        """Restore nodal and named material state under any MPI partition."""

        from ..checkpointing import (
            load_portable_state_bundle,
            validate_checkpoint_record,
        )
        from ..mechanics.plasticity import J2IncrementInfo
        from ..solvers import SolveEvent

        manifest = Path(path)
        if not manifest.name.endswith(".checkpoint.json"):
            manifest = manifest.with_suffix("").with_name(
                manifest.with_suffix("").name + ".checkpoint.json"
            )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("schema") != "agentfem.small-strain-material-checkpoint.v1":
            raise ValueError("Unsupported small-strain material checkpoint schema.")
        current = json.loads(
            json.dumps(self._portable_checkpoint_identity(), sort_keys=True)
        )
        if payload.get("step_identity") != current:
            raise ValueError(
                "Checkpoint model, provider, mesh, procedure, or increment "
                "identity differs from the current Step."
            )
        load_portable_state_bundle(
            manifest,
            state={"U": self.solution},
            record=payload["nodal_state"],
            identities=payload["nodal_identity"],
        )
        self.state.state.load(
            validate_checkpoint_record(manifest.parent, payload["quadrature_state"]),
            material=self.material,
        )
        self.accepted_load_factor = float(payload["coordinate"])
        self.accepted_increments[:] = [
            J2IncrementInfo.from_dict(item) for item in payload["accepted_increments"]
        ]
        self.attempted_increments[:] = [
            J2IncrementInfo.from_dict(item) for item in payload["attempted_increments"]
        ]
        self.execution_events[:] = [
            SolveEvent.from_dict(item) for item in payload["execution_events"]
        ]
        self.energy_history[:] = [
            SmallStrainMaterialEnergyFrame.from_dict(item)
            for item in payload["energy_history"]
        ]
        self.next_increment_size = payload.get("next_increment_size")
        self._apply_loading(self.accepted_load_factor)
        self.last_solve_info = SmallStrainMaterialLoadPathInfo(
            tuple(self.accepted_increments),
            tuple(self.attempted_increments),
            self.incrementation,
        )
        accepted = self.state.evaluate_strain(self._strain_evaluator)
        self.state.accepted_strain.assign(accepted)
        self.state.trial_strain.assign(accepted)
        self.state.update(accepted, self.material)

    def _portable_checkpoint_identity(self) -> dict[str, object]:
        from ..checkpointing import function_portable_identity

        return {
            "step_name": self.name,
            "procedure": self.procedure.summary(),
            "material": self.material.as_dict(),
            "amplitude": self.amplitude.summary(),
            "incrementation": self.incrementation.summary(),
            "solution": function_portable_identity(self.solution),
            "quadrature": self.state.state.summary()["transaction"],
        }


def small_strain_material_step(
    *,
    displacement,
    material,
    external_force=None,
    constraints=(),
    study=None,
    incrementation=None,
    solver_options=None,
    quadrature_degree: int = 2,
    progress=True,
    status_file=None,
    amplitude=None,
    name: str = "small_strain_material",
) -> SmallStrainMaterialStep:
    """Build a nonlinear 2D-plane-strain or 3D provider-material Step."""

    if not isinstance(material, LearnedConstitutiveMaterial):
        raise TypeError(
            "small_strain_material_step currently requires a "
            "LearnedConstitutiveMaterial."
        )
    domain = displacement.value.function_space.mesh
    dimension = int(domain.geometry.dim)
    if dimension not in {2, 3}:
        raise NotImplementedError(
            "Small-strain material equilibrium requires 2D or 3D."
        )
    if dimension == 2 and getattr(study, "assumption", None) != "plane_strain":
        raise NotImplementedError("The first 2D generic material Step is plane strain.")
    response = SmallStrainMaterialQuadratureResponse.create(
        domain,
        material.state_schema,
        degree=quadrature_degree,
    )
    selected_amplitude = (
        amplitudes.ramp()
        if amplitude is None
        else amplitudes.as_amplitude(
            amplitude,
            name="small_strain_material_amplitude",
        )
    )
    if not np.isclose(selected_amplitude(0.0), 0.0):
        raise ValueError("A small-strain material amplitude must start at zero.")
    load_factor = fem.Constant(domain, PETSc.ScalarType(0.0))
    strain_test = _strain_voigt(displacement.test)
    strain_trial = _strain_voigt(displacement.trial)
    weights = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)
    stress = response.stress.function
    tangent = response.tangent.function
    weighted_stress = ufl.as_vector(
        tuple(weights[index] * stress[index] for index in range(6))
    )
    residual = ufl.dot(weighted_stress, strain_test) * response.measure
    if external_force is not None:
        residual -= load_factor * external_force.expression
    tangent_action = ufl.dot(tangent, strain_trial)
    weighted_tangent_action = ufl.as_vector(
        tuple(weights[index] * tangent_action[index] for index in range(6))
    )
    jacobian = ufl.dot(weighted_tangent_action, strain_test) * response.measure
    selected_bcs = []
    prescribed_values = []
    for item in constraints or ():
        if hasattr(item, "bcs"):
            selected_bcs.extend(item.bcs)
            declared = getattr(item, "dirichlet", ())
        elif hasattr(item, "bc"):
            selected_bcs.append(item.bc)
            declared = (item,)
        else:
            selected_bcs.append(item)
            declared = ()
        for constraint in declared:
            value = getattr(constraint, "value", None)
            if value is not None and hasattr(value, "value"):
                prescribed_values.append(
                    (value, np.asarray(value.value, dtype=float).copy(), constraint.bc)
                )
    if not selected_bcs:
        raise ValueError(
            "Small-strain material equilibrium requires strong constraints."
        )
    return SmallStrainMaterialStep(
        name=name,
        solution=displacement.value,
        material=material,
        state=response,
        residual_form=fem.form(residual),
        tangent_form=fem.form(jacobian),
        load_factor=load_factor,
        amplitude=selected_amplitude,
        bcs=tuple(selected_bcs),
        prescribed_values=tuple(prescribed_values),
        incrementation=step_controls.normalize(incrementation),
        solver_options=newton() if solver_options is None else solver_options,
        study=study,
        progress=progress,
        status_file=status_file,
    )


__all__ = [
    "SmallStrainMaterialEnergyFrame",
    "SmallStrainMaterialLoadPathInfo",
    "SmallStrainMaterialStep",
    "small_strain_material_step",
]

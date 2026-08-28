"""Experimental global total-Lagrangian finite-strain plasticity path."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from petsc4py import PETSc
from mpi4py import MPI

from .. import amplitudes
from .. import steps as step_controls
from ..constitutive import FiniteStrainJ2Logarithmic
from ..constitutive import MaterialQuadratureResponse
from ..solvers import NewtonSolverOptions, newton, solve_matrix_system


@dataclass(frozen=True)
class FiniteStrainPlasticityIncrementInfo:
    increment: int
    attempt: int
    start_load_factor: float
    load_factor: float
    converged: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    plastic_points: int
    maximum_plastic_increment: float
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, record) -> "FiniteStrainPlasticityIncrementInfo":
        return cls(
            increment=int(record["increment"]),
            attempt=int(record["attempt"]),
            start_load_factor=float(record["start_load_factor"]),
            load_factor=float(record["load_factor"]),
            converged=bool(record["converged"]),
            iterations=int(record["iterations"]),
            initial_residual_norm=float(record["initial_residual_norm"]),
            residual_norm=float(record["residual_norm"]),
            plastic_points=int(record["plastic_points"]),
            maximum_plastic_increment=float(record["maximum_plastic_increment"]),
            rejection_reason=record.get("rejection_reason"),
        )


@dataclass
class ExperimentalFiniteStrainPlasticityStep:
    """Total-Lagrangian Newton path consuming a neutral material provider.

    This class is intentionally not registered under ``model.step`` yet.  It
    exists to prove the provider/transaction/global-residual interface before
    the public lowering route is promoted.
    """

    name: str
    solution: object
    accepted_solution: object
    material: FiniteStrainJ2Logarithmic
    response: MaterialQuadratureResponse
    residual_form: object
    tangent_form: object
    deformation_gradient_old: object
    deformation_gradient_new: object
    load_factor: object
    amplitude: amplitudes.Amplitude
    bcs: tuple[object, ...]
    prescribed_values: tuple[tuple[object, np.ndarray, object], ...]
    incrementation: object
    solver_options: NewtonSolverOptions
    accepted_load_factor: float = field(default=0.0, init=False)
    accepted_increments: list[FiniteStrainPlasticityIncrementInfo] = field(
        default_factory=list, init=False
    )
    attempted_increments: list[FiniteStrainPlasticityIncrementInfo] = field(
        default_factory=list, init=False
    )
    next_increment_size: float | None = field(default=None, init=False)

    def _apply_loading(self, coordinate: float) -> None:
        factor = self.amplitude(coordinate)
        self.load_factor.value = PETSc.ScalarType(factor)
        for constant, target, _bc in self.prescribed_values:
            selected = factor * target
            constant.value = (
                PETSc.ScalarType(selected.item())
                if selected.ndim == 0 or selected.size == 1
                else np.asarray(selected, dtype=PETSc.ScalarType)
            )

    def _evaluate_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.response.state.evaluate_expression(
                self.deformation_gradient_old,
                value_shape=(3, 3),
            ),
            self.response.state.evaluate_expression(
                self.deformation_gradient_new,
                value_shape=(3, 3),
            ),
        )

    def _update_response(
        self,
        *,
        start_factor: float,
        target_factor: float,
    ):
        old_gradient, new_gradient = self._evaluate_gradients()
        committed = self.response.state.committed_state_vectors()
        result = self.response.update(
            self.material,
            deformation_gradient_old=old_gradient,
            deformation_gradient_new=new_gradient,
            time=target_factor,
            time_increment=max(target_factor - start_factor, np.finfo(float).eps),
            commit=False,
        )
        increments = result.state_new[:, -1] - committed[:, -1]
        return result, float(np.max(increments, initial=0.0))

    def _correction_rhs(self):
        residual = fem_petsc.assemble_vector(self.residual_form)
        fem_petsc.apply_lifting(
            residual,
            [self.tangent_form],
            [self.bcs],
            x0=[self.solution.x.petsc_vec],
            alpha=-1.0,
        )
        residual.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        residual.scale(-1.0)
        fem_petsc.set_bc(
            residual,
            self.bcs,
            x0=self.solution.x.petsc_vec,
            alpha=1.0,
        )
        return residual, float(residual.norm())

    def _assign_trial(self, base, direction, alpha: float) -> None:
        self.solution.x.array[:] = base
        self.solution.x.array[: len(direction)] += alpha * direction
        self.solution.x.scatter_forward()

    def _line_search(
        self,
        base,
        direction,
        base_norm: float,
        *,
        start_factor: float,
        target_factor: float,
    ) -> float:
        options = self.solver_options
        alpha = 1.0
        if options.line_search in {None, "basic"}:
            self._assign_trial(base, direction, alpha)
            return alpha
        while alpha + 1.0e-15 >= options.minimum_step_length:
            self._assign_trial(base, direction, alpha)
            self._update_response(
                start_factor=start_factor,
                target_factor=target_factor,
            )
            rhs, trial_norm = self._correction_rhs()
            rhs.destroy()
            if np.isfinite(trial_norm) and trial_norm < base_norm:
                return alpha
            alpha *= options.line_search_reduction
        self.solution.x.array[:] = base
        self.solution.x.scatter_forward()
        self.response.rollback()
        return 0.0

    def _solve_increment(
        self,
        *,
        increment: int,
        attempt: int,
        start_factor: float,
        target_factor: float,
    ) -> FiniteStrainPlasticityIncrementInfo:
        initial_norm = None
        norm = float("inf")
        maximum_increment = 0.0
        plastic_points = 0
        converged = False
        iteration = 0
        for iteration in range(self.solver_options.maximum_iterations + 1):
            result, maximum_increment = self._update_response(
                start_factor=start_factor,
                target_factor=target_factor,
            )
            old_state = self.response.state.committed_state_vectors()
            points_per_cell = len(self.response.state.reference_field.points)
            cell_map = self.response.domain.topology.index_map(
                self.response.domain.topology.dim
            )
            owned_points = int(cell_map.size_local) * points_per_cell
            local_plastic_points = int(
                np.count_nonzero(
                    result.state_new[:owned_points, -1]
                    > old_state[:owned_points, -1] + 1.0e-14
                )
            )
            plastic_points = int(
                self.response.domain.comm.allreduce(local_plastic_points, op=MPI.SUM)
            )
            maximum_increment = float(
                self.response.domain.comm.allreduce(maximum_increment, op=MPI.MAX)
            )
            rhs, norm = self._correction_rhs()
            if initial_norm is None:
                initial_norm = norm
            threshold = (
                self.solver_options.absolute_tolerance
                + self.solver_options.relative_tolerance * initial_norm
            )
            if np.isfinite(norm) and norm <= threshold:
                rhs.destroy()
                converged = True
                break
            if iteration == self.solver_options.maximum_iterations:
                rhs.destroy()
                break
            tangent = fem_petsc.assemble_matrix(self.tangent_form, bcs=self.bcs)
            tangent.assemble()
            correction = rhs.duplicate()
            correction.set(0.0)
            linear = solve_matrix_system(
                tangent,
                rhs,
                correction,
                self.solver_options.linear_solver,
                raise_on_failure=False,
            )
            tangent.destroy()
            rhs.destroy()
            if not linear.converged:
                correction.destroy()
                break
            base = self.solution.x.array.copy()
            direction = correction.array_r.copy()
            correction.destroy()
            alpha = self._line_search(
                base,
                direction,
                norm,
                start_factor=start_factor,
                target_factor=target_factor,
            )
            if alpha == 0.0:
                break
        return FiniteStrainPlasticityIncrementInfo(
            increment=increment,
            attempt=attempt,
            start_load_factor=start_factor,
            load_factor=target_factor,
            converged=converged,
            iterations=iteration,
            initial_residual_norm=float(initial_norm or 0.0),
            residual_norm=float(norm),
            plastic_points=plastic_points,
            maximum_plastic_increment=maximum_increment,
        )

    def _restore_accepted(self) -> None:
        self.solution.x.array[:] = self.accepted_solution.x.array
        self.solution.x.scatter_forward()
        self.response.rollback()
        self._apply_loading(self.accepted_load_factor)
        self._update_response(
            start_factor=self.accepted_load_factor,
            target_factor=self.accepted_load_factor,
        )

    def solve(self, *, until: float = 1.0):
        """Advance the normalized load path with commit/cutback discipline."""

        selected_until = float(until)
        if not self.accepted_load_factor < selected_until <= 1.0:
            raise ValueError(
                "until must exceed the accepted load factor and be at most one."
            )
        accepted = self.accepted_load_factor
        proposed = (
            self.incrementation.initial
            if isinstance(self.incrementation, step_controls.AutomaticIncrementation)
            else None
        )
        cutbacks = 0
        while accepted < selected_until - 1.0e-12:
            increment = len(self.accepted_increments) + 1
            if isinstance(self.incrementation, step_controls.AutomaticIncrementation):
                if len(self.accepted_increments) >= self.incrementation.max_increments:
                    raise RuntimeError("Finite-strain J2 reached max_increments.")
                target = min(selected_until, accepted + proposed)
            else:
                remaining = [
                    value
                    for value in self.incrementation.load_factors
                    if value > accepted + 1.0e-12
                ]
                if not remaining:
                    raise RuntimeError("Fixed finite-strain J2 path is incomplete.")
                target = min(selected_until, remaining[0])
            self._apply_loading(target)
            info = self._solve_increment(
                increment=increment,
                attempt=cutbacks + 1,
                start_factor=accepted,
                target_factor=target,
            )
            limit = getattr(self.incrementation, "maximum_inelastic_increment", None)
            if (
                info.converged
                and limit is not None
                and info.maximum_plastic_increment > limit
            ):
                info = FiniteStrainPlasticityIncrementInfo(
                    **{
                        **info.as_dict(),
                        "converged": False,
                        "rejection_reason": (
                            "maximum equivalent plastic-strain increment "
                            f"{info.maximum_plastic_increment:.6g} exceeds {limit:.6g}"
                        ),
                    }
                )
            self.attempted_increments.append(info)
            if info.converged:
                self.response.commit()
                self.accepted_solution.x.array[:] = self.solution.x.array
                self.accepted_solution.x.scatter_forward()
                self.accepted_increments.append(info)
                size = target - accepted
                accepted = target
                self.accepted_load_factor = target
                cutbacks = 0
                if isinstance(
                    self.incrementation, step_controls.AutomaticIncrementation
                ):
                    proposed = self.incrementation.after_convergence(
                        size, info.iterations
                    )
                    self.next_increment_size = proposed
                continue

            self._restore_accepted()
            if not isinstance(
                self.incrementation, step_controls.AutomaticIncrementation
            ):
                raise RuntimeError(
                    f"{self.name}: fixed increment failed at load factor {target}."
                )
            cutbacks += 1
            proposed = self.incrementation.after_failure(target - accepted)
            self.next_increment_size = proposed
            if (
                cutbacks > self.incrementation.max_cutbacks
                or proposed < self.incrementation.minimum
            ):
                raise RuntimeError(
                    f"{self.name}: automatic incrementation exhausted cutbacks."
                )
        return self.solution

    def _checkpoint_identity(self) -> dict[str, object]:
        from ..checkpointing import function_partition_identity

        return {
            "step_name": self.name,
            "material": self.material.summary(),
            "state_schema": self.response.state.state_schema.summary(),
            "incrementation": self.incrementation.summary(),
            "amplitude": self.amplitude.summary(),
            "solution": function_partition_identity(self.solution),
        }

    def save_checkpoint(self, path, *, portable: bool | None = None) -> Path:
        """Save the accepted global and material state."""

        comm = self.solution.function_space.mesh.comm
        selected_portable = comm.size != 1 if portable is None else bool(portable)
        if selected_portable:
            return self._save_portable_checkpoint(path)
        if comm.size != 1:
            raise ValueError("Distributed checkpoints require portable=True.")
        from ..checkpointing import atomic_savez

        selected = Path(path)
        if selected.suffix != ".npz":
            selected = selected.with_suffix(".npz")
        snapshot = self.response.snapshot()
        atomic_savez(
            selected,
            schema="agentfem.finite-strain-j2-experimental-checkpoint.v1",
            identity=json.dumps(self._checkpoint_identity(), sort_keys=True),
            solution=self.solution.x.array,
            accepted_solution=self.accepted_solution.x.array,
            accepted_load_factor=self.accepted_load_factor,
            state_names=json.dumps(tuple(snapshot)),
            **snapshot,
            accepted_increments=json.dumps(
                [item.as_dict() for item in self.accepted_increments]
            ),
            attempted_increments=json.dumps(
                [item.as_dict() for item in self.attempted_increments]
            ),
            next_increment_size=(
                np.nan
                if self.next_increment_size is None
                else self.next_increment_size
            ),
        )
        return selected

    def load_checkpoint(self, path) -> None:
        """Restore a serial or cross-partition portable checkpoint."""

        selected = Path(path)
        if selected.suffix == ".json" or selected.name.endswith(".checkpoint.json"):
            payload = json.loads(selected.read_text(encoding="utf-8"))
            if payload.get("schema") == (
                "agentfem.finite-strain-j2-experimental-checkpoint.v2"
            ):
                self._load_portable_checkpoint(selected, payload)
                return
        if self.solution.function_space.mesh.comm.size != 1:
            raise ValueError(
                "This finite-strain J2 checkpoint is partition-bound; use the "
                "portable v2 manifest for distributed restore."
            )
        with np.load(selected, allow_pickle=False) as data:
            if str(data["schema"]) != (
                "agentfem.finite-strain-j2-experimental-checkpoint.v1"
            ):
                raise ValueError("Unsupported finite-strain J2 checkpoint schema.")
            stored_identity = json.loads(str(data["identity"]))
            current_identity = json.loads(
                json.dumps(self._checkpoint_identity(), sort_keys=True)
            )
            if stored_identity != current_identity:
                raise ValueError(
                    "Finite-strain J2 checkpoint material, state, loading, "
                    "increment control, or function layout differs."
                )
            solution = np.asarray(data["solution"])
            accepted_solution = np.asarray(data["accepted_solution"])
            if (
                solution.size != self.solution.x.array.size
                or accepted_solution.size != self.accepted_solution.x.array.size
            ):
                raise ValueError("Finite-strain J2 checkpoint dof layout differs.")
            self.solution.x.array[:] = solution
            self.solution.x.scatter_forward()
            self.accepted_solution.x.array[:] = accepted_solution
            self.accepted_solution.x.scatter_forward()
            state_names = tuple(json.loads(str(data["state_names"])))
            if state_names != tuple(self.response.state.transaction.names):
                raise ValueError("Finite-strain J2 checkpoint state names differ.")
            self.response.restore({name: data[name] for name in state_names})
            self.accepted_load_factor = float(data["accepted_load_factor"])
            self.accepted_increments[:] = [
                FiniteStrainPlasticityIncrementInfo.from_dict(item)
                for item in json.loads(str(data["accepted_increments"]))
            ]
            self.attempted_increments[:] = [
                FiniteStrainPlasticityIncrementInfo.from_dict(item)
                for item in json.loads(str(data["attempted_increments"]))
            ]
            size = float(data["next_increment_size"])
            self.next_increment_size = size if np.isfinite(size) else None
        self._apply_loading(self.accepted_load_factor)
        self._update_response(
            start_factor=self.accepted_load_factor,
            target_factor=self.accepted_load_factor,
        )

    def _portable_checkpoint_identity(self) -> dict[str, object]:
        from ..checkpointing import function_portable_identity

        return {
            "step_name": self.name,
            "material": self.material.summary(),
            "state_schema": self.response.state.state_schema.summary(),
            "incrementation": self.incrementation.summary(),
            "amplitude": self.amplitude.summary(),
            "solution": function_portable_identity(self.solution),
        }

    def _save_portable_checkpoint(self, path) -> Path:
        from ..checkpointing import (
            atomic_write_text,
            checkpoint_file_record,
            save_portable_state_bundle,
        )

        selected = Path(path)
        if selected.suffix:
            selected = selected.with_suffix("")
        manifest = selected.with_name(selected.name + ".checkpoint.json")
        bundle = save_portable_state_bundle(
            manifest,
            state={"U": self.solution, "U_ACCEPTED": self.accepted_solution},
        )
        quadrature = self.response.state.save(
            manifest.with_name(
                f"{selected.name}.{bundle['generation']}.quadrature"
            ),
            material=self.material,
        )
        payload = {
            "schema": "agentfem.finite-strain-j2-experimental-checkpoint.v2",
            "identity": self._portable_checkpoint_identity(),
            "coordinate": self.accepted_load_factor,
            "nodal_state": bundle["record"],
            "nodal_identity": bundle["identities"],
            "quadrature_state": checkpoint_file_record(quadrature),
            "accepted_increments": [
                item.as_dict() for item in self.accepted_increments
            ],
            "attempted_increments": [
                item.as_dict() for item in self.attempted_increments
            ],
            "next_increment_size": self.next_increment_size,
        }
        comm = self.solution.function_space.mesh.comm
        error = None
        if comm.rank == 0:
            try:
                atomic_write_text(
                    manifest,
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(
                f"Finite-strain J2 checkpoint manifest write failed: {error}"
            )
        comm.barrier()
        return manifest

    def _load_portable_checkpoint(self, manifest: Path, payload: dict) -> None:
        from ..checkpointing import (
            load_portable_state_bundle,
            validate_checkpoint_record,
        )

        current = json.loads(
            json.dumps(self._portable_checkpoint_identity(), sort_keys=True)
        )
        if payload.get("identity") != current:
            raise ValueError(
                "Portable finite-strain J2 checkpoint scientific identity differs."
            )
        load_portable_state_bundle(
            manifest,
            state={"U": self.solution, "U_ACCEPTED": self.accepted_solution},
            record=payload["nodal_state"],
            identities=payload["nodal_identity"],
        )
        self.response.state.load(
            validate_checkpoint_record(
                manifest.parent, payload["quadrature_state"]
            ),
            material=self.material,
        )
        self.accepted_load_factor = float(payload["coordinate"])
        self.accepted_increments[:] = [
            FiniteStrainPlasticityIncrementInfo.from_dict(item)
            for item in payload["accepted_increments"]
        ]
        self.attempted_increments[:] = [
            FiniteStrainPlasticityIncrementInfo.from_dict(item)
            for item in payload["attempted_increments"]
        ]
        self.next_increment_size = payload.get("next_increment_size")
        self._apply_loading(self.accepted_load_factor)
        self._update_response(
            start_factor=self.accepted_load_factor,
            target_factor=self.accepted_load_factor,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "experimental_finite_strain_plasticity_step",
            "name": self.name,
            "maturity": "experimental_global_mpi_restart",
            "accepted_load_factor": self.accepted_load_factor,
            "material": self.material.summary(),
            "response": self.response.summary(),
            "incrementation": self.incrementation.summary(),
            "solver": self.solver_options.summary(),
            "accepted_increments": tuple(
                item.as_dict() for item in self.accepted_increments
            ),
            "attempted_increments": tuple(
                item.as_dict() for item in self.attempted_increments
            ),
        }


def experimental_finite_strain_j2_step(
    *,
    displacement,
    material: FiniteStrainJ2Logarithmic,
    external_force=None,
    constraints=(),
    incrementation=None,
    solver_options=None,
    quadrature_degree: int = 2,
    amplitude=None,
    name: str = "finite_strain_j2_experimental",
) -> ExperimentalFiniteStrainPlasticityStep:
    """Build the gated 3D global patch for logarithmic finite-strain J2."""

    if not isinstance(material, FiniteStrainJ2Logarithmic):
        raise TypeError("The experimental step requires FiniteStrainJ2Logarithmic.")
    solution = displacement.value
    domain = solution.function_space.mesh
    if domain.geometry.dim != 3:
        raise NotImplementedError("The experimental finite-strain J2 step is 3D only.")
    response = MaterialQuadratureResponse.create(
        domain,
        material.state_schema,
        degree=quadrature_degree,
    )
    accepted_solution = fem.Function(solution.function_space, name="U_ACCEPTED")
    identity = ufl.Identity(3)
    deformation_gradient_old = response.state.compile_expression(
        identity + ufl.grad(accepted_solution), value_shape=(3, 3)
    )
    deformation_gradient_new = response.state.compile_expression(
        identity + ufl.grad(solution), value_shape=(3, 3)
    )
    gradient_test = ufl.grad(displacement.test)
    gradient_trial = ufl.grad(displacement.trial)
    first_piola = response.first_piola_stress.function
    tangent = response.tangent.function
    i, j, k, l = ufl.indices(4)
    tangent_action = ufl.as_tensor(
        tangent[i, j, k, l] * gradient_trial[k, l], (i, j)
    )
    load_factor = fem.Constant(domain, PETSc.ScalarType(0.0))
    residual = ufl.inner(first_piola, gradient_test) * response.measure
    if external_force is not None:
        residual -= load_factor * external_force.expression
    jacobian = ufl.inner(tangent_action, gradient_test) * response.measure

    selected_bcs = []
    prescribed_values = []
    for item in constraints or ():
        if hasattr(item, "bcs"):
            selected_bcs.extend(item.bcs)
            records = getattr(item, "dirichlet", ())
        elif hasattr(item, "bc"):
            selected_bcs.append(item.bc)
            records = (item,)
        else:
            selected_bcs.append(item)
            records = ()
        for record in records:
            value = getattr(record, "value", None)
            if value is not None and hasattr(value, "value"):
                prescribed_values.append(
                    (
                        value,
                        np.asarray(value.value, dtype=float).copy(),
                        record.bc,
                    )
                )
    selected_amplitude = (
        amplitudes.ramp()
        if amplitude is None
        else amplitudes.as_amplitude(amplitude, name="finite_strain_j2_amplitude")
    )
    if not np.isclose(selected_amplitude(0.0), 0.0):
        raise ValueError("Finite-strain J2 amplitude must start at zero.")
    return ExperimentalFiniteStrainPlasticityStep(
        name=name,
        solution=solution,
        accepted_solution=accepted_solution,
        material=material,
        response=response,
        residual_form=fem.form(residual),
        tangent_form=fem.form(jacobian),
        deformation_gradient_old=deformation_gradient_old,
        deformation_gradient_new=deformation_gradient_new,
        load_factor=load_factor,
        amplitude=selected_amplitude,
        bcs=tuple(selected_bcs),
        prescribed_values=tuple(prescribed_values),
        incrementation=step_controls.normalize(incrementation),
        solver_options=newton() if solver_options is None else solver_options,
    )

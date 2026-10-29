# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded total-Lagrangian equilibrium for finite-strain history providers.

Local equations belong to UserMaterial. This Procedure owns physical time,
Newton trials, accepted quadrature responses and atomic serial restart. It
does not infer a multiaxial law, objective rate or tangent from a model name.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from petsc4py import PETSc

from .. import amplitudes, checkpointing, procedures, steps
from ..constitutive.material_driver import MaterialQuadratureResponse
from ..constitutive.user_material import UserMaterial, MaterialTangentConvention
from ..solvers import NewtonSolverOptions, newton, solve_matrix_system
from ..diagnostics import StandardRunReporter, SolveEventRecorder, compose_reporters
from ..events import SolveEvent
from ._incremental_runtime import NewtonCorrection, NewtonEvaluation, run_newton_attempt


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


class FiniteStrainMaterialStep:
    """Experimental serial 3D/plane-strain, strong-BC history-material Step."""

    schema = "agentfem.finite-strain-material-step.v1"

    def __init__(
        self,
        *,
        displacement,
        material,
        duration,
        external_force=None,
        constraints=(),
        study=None,
        incrementation=None,
        solver_options=None,
        quadrature_degree=2,
        amplitude=None,
        properties=(),
        name="finite_strain_material",
        progress=True,
        time_unit=None,
    ):
        from .. import constraints as constraint_api
        from .._step_provider_support import is_standalone_h1_vector_field

        self.solution = displacement.value
        domain = self.solution.function_space.mesh
        dimension = domain.geometry.dim
        if domain.comm.size != 1:
            raise NotImplementedError(
                "AFM-USER-FINITE-MPI: implicit user-material MPI is not verified."
            )
        if dimension not in (2, 3) or domain.topology.dim != dimension:
            raise NotImplementedError(
                "AFM-USER-FINITE-DIMENSION: use a 3D solid or 2D plane strain."
            )
        if self.solution.ufl_shape != (dimension,):
            raise ValueError(
                "Displacement must have one component per spatial dimension."
            )
        if not is_standalone_h1_vector_field(displacement):
            raise NotImplementedError(
                "AFM-USER-FINITE-SPACE: use a standalone H1-conforming displacement; no mixed-pressure or DG interface terms are supplied."
            )
        if dimension == 2 and getattr(study, "assumption", None) != "plane_strain":
            raise NotImplementedError(
                "AFM-USER-FINITE-REDUCTION: only explicit plane strain is admitted in 2D."
            )
        if not isinstance(material, UserMaterial):
            raise TypeError("A finite-strain UserMaterial is required.")
        convention = MaterialTangentConvention.first_piola_deformation_gradient()
        if material.tangent_convention != convention:
            raise ValueError(
                "AFM-USER-FINITE-TANGENT: declare the reference 9x9 dP/dF convention."
            )
        if not material.state_schema.size:
            raise ValueError(
                "History-material integration requires a nonempty named state schema."
            )
        if not callable(getattr(material, "summary", None)):
            raise TypeError(
                "UserMaterial.summary() must record parameters and implementation identity for restart."
            )
        self.material_identity = _json(material.summary())
        if not isinstance(self.material_identity, dict) or not self.material_identity:
            raise ValueError(
                "Material summary must be a nonempty JSON scientific identity."
            )
        self.duration = float(duration)
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Physical duration must be finite and positive.")
        self.time_unit = None if time_unit is None else str(time_unit).strip()
        if self.time_unit == "":
            raise ValueError("time_unit must be nonempty when declared.")
        if np.any(self.solution.x.array != 0):
            raise ValueError(
                "Start from zero displacement or restore a complete Step checkpoint."
            )
        self.name, self.material, self.study = (
            name or "finite_strain_material",
            material,
            study,
        )
        self.properties = np.asarray(properties, dtype=float).reshape(-1).copy()
        if not np.isfinite(self.properties).all():
            raise ValueError("Material properties must be finite.")
        self.procedure = procedures.nonlinear_static(stateful=True)
        self.incrementation = steps.normalize(incrementation)
        if (
            isinstance(self.incrementation, steps.FixedIncrementation)
            and self.incrementation.load_factors[-1] != 1.0
        ):
            raise ValueError(
                "The finite-history normalized fixed path must end at exactly 1."
            )
        if (
            getattr(self.incrementation, "maximum_inelastic_increment", None)
            is not None
        ):
            raise NotImplementedError(
                "Generic material states have no universal inelastic scalar; use provider suggested_time_scale."
            )
        self.solver_options = newton() if solver_options is None else solver_options
        if not isinstance(self.solver_options, NewtonSolverOptions):
            raise TypeError(
                "Finite-strain history equilibrium requires solvers.newton() options."
            )
        if not self.solver_options.error_if_not_converged:
            raise NotImplementedError(
                "This bounded Step always raises on exhausted increments; non-raising failure mode is not admitted."
            )
        self.amplitude = (
            amplitudes.ramp()
            if amplitude is None
            else amplitudes.as_amplitude(amplitude)
        )
        if not np.isclose(self.amplitude(0), 0):
            raise ValueError("Load amplitude must start at zero.")
        self._load_knots = ()
        if self.amplitude.kind == "tabular":
            self._load_knots = tuple(
                float(t)
                for t in self.amplitude.summary()["metadata"]["times"]
                if 0 < t < 1
            )
        self.progress = progress
        self._reporter_failure = None
        self.execution_events = []
        recorder = SolveEventRecorder(self.execution_events)
        self._reporter = (
            compose_reporters(recorder, StandardRunReporter(domain.comm))
            if progress is True
            else recorder
            if progress in (False, None)
            else compose_reporters(recorder, progress)
        )
        state_names = [v.output_name or v.name for v in material.state_schema.variables]
        if len(set(state_names)) != len(state_names) or set(state_names) & {
            "P",
            "S",
            "SENER",
            "DPDF",
            "RF",
        }:
            raise ValueError(
                "State output names must be unique and not reserved stress/energy names."
            )
        component_names = tuple(
            str(key).strip().upper()
            for key in getattr(material, "stored_energy_component_names", ())
        )
        if (
            any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) for key in component_names)
            or len(set(component_names)) != len(component_names)
            or set(component_names)
            & (set(state_names) | {"P", "S", "SENER", "DPDF", "RF"})
        ):
            raise ValueError(
                "Stored-energy component names must be unique and not collide with response/state fields."
            )
        raw_names = ["S", "P", "SENER", *state_names, *component_names]
        published_names = [
            self.solution.name,
            "RF",
            *raw_names,
            *(f"{name}_CELL" for name in raw_names),
        ]
        if len(set(published_names)) != len(published_names):
            raise ValueError(
                "Material output names collide with displacement, raw responses or recovered _CELL fields."
            )
        self.response = MaterialQuadratureResponse.create(
            domain,
            material.state_schema,
            degree=quadrature_degree,
            stored_energy_component_names=component_names,
        )
        self.state = self.response.state
        self._responses = {
            "P": self.response.first_piola_stress.function,
            "S": self.response.cauchy_stress.function,
            "DPDF": self.response.tangent.function,
            "SENER": self.response.strain_energy_density.function,
            **{
                key: field.function
                for key, field in self.response.stored_energy_density_components.items()
            },
        }
        self.bcs, self.prescribed = [], []
        for item in constraint_api.constraint_assets(constraints):
            if not isinstance(
                item,
                (
                    constraint_api.DirichletConstraint,
                    constraint_api.RemoteDisplacementConstraint,
                ),
            ):
                raise NotImplementedError(
                    "AFM-USER-FINITE-BC: only strong displacement constraints are admitted."
                )
        for item in constraint_api.dirichlet_constraints(constraints):
            if isinstance(item, constraint_api.TimeDependentDirichlet):
                raise NotImplementedError(
                    "Use end displacement and a normalized amplitude; absolute-time BCs are not yet admitted."
                )
            if not hasattr(getattr(item, "value", None), "value"):
                raise NotImplementedError(
                    "The initial history-material route requires constant end-displacement data."
                )
            self.bcs.append(item.bc)
            self.prescribed.append((item.value, np.asarray(item.value.value).copy()))
        F = ufl.Identity(dimension) + ufl.grad(self.solution)
        if dimension == 2:
            F = ufl.as_matrix(((F[0, 0], F[0, 1], 0), (F[1, 0], F[1, 1], 0), (0, 0, 1)))
        self._F_evaluator = self.state.compile_expression(F, value_shape=(3, 3))
        self._accepted_F = self._gradient()
        self._accepted_U = self.solution.x.array.copy()
        self.load_factor = fem.Constant(domain, PETSc.ScalarType(0))
        P = self.response.first_piola_stress.function
        A = self.response.tangent.function
        grad_v, grad_du = ufl.grad(displacement.test), ufl.grad(displacement.trial)
        residual = (
            sum(
                P[i, j] * grad_v[i, j]
                for i in range(dimension)
                for j in range(dimension)
            )
            * self.state.measure
        )
        tangent = (
            sum(
                A[i, j, k, l] * grad_du[k, l] * grad_v[i, j]
                for i in range(dimension)
                for j in range(dimension)
                for k in range(dimension)
                for l in range(dimension)
            )
            * self.state.measure
        )
        if external_force is not None:
            if any(
                c is self.solution for c in external_force.expression.coefficients()
            ):
                raise NotImplementedError(
                    "AFM-USER-FINITE-FOLLOWER: displacement-dependent loads need their own verified tangent."
                )
            residual -= self.load_factor * external_force.expression
        self.residual_form, self.tangent_form = fem.form(residual), fem.form(tangent)
        self.accepted_load_factor = 0.0
        self.accepted_increments, self.attempted_increments = [], []
        self.checkpoints = []
        self.restored_from = None
        self.next_increment_size = None
        self._energy_defined = False
        self._busy = False
        self._at_accepted = True
        self._apply_loading(0)
        self._external_force = external_force
        # Store coefficient values as well as form signatures: changing a dead
        # load or prescribed end value must invalidate restart identity.
        load_identity = self._load_identity()
        self._identity = _json(
            {
                "name": self.name,
                "material": self.material_identity,
                "state_schema": material.state_schema.summary(),
                "tangent": convention.summary(),
                "properties": self.properties.tolist(),
                "duration": self.duration,
                "time_unit": self.time_unit,
                "study": None if study is None else study.summary(),
                "solution": checkpointing.function_partition_identity(self.solution),
                "quadrature_degree": quadrature_degree,
                "amplitude": self.amplitude.summary(),
                "incrementation": self.incrementation.summary(),
                "load": load_identity,
                "constraints": [
                    {"dofs": bc.dof_indices()[0].tolist(), "value": value.tolist()}
                    for bc, (_, value) in zip(self.bcs, self.prescribed)
                ],
            }
        )

    @property
    def time(self):
        return self.duration * self.accepted_load_factor

    def _gradient(self):
        return self.state.evaluate_expression(self._F_evaluator, value_shape=(3, 3))

    def _check_accepted_solution(self):
        # F alone cannot detect a rigid translation of the displacement field.
        if not np.array_equal(
            self.solution.x.array, self._accepted_U
        ) or not np.allclose(
            self._gradient(), self._accepted_F, rtol=1e-12, atol=1e-13
        ):
            raise ValueError(
                "Displacement was changed outside the accepted Step state."
            )

    def _load_identity(self):
        if self._external_force is None:
            return None
        form = self._external_force.expression
        return {
            "signature": form.signature(),
            "coefficients": [
                sha256(np.asarray(c.x.array).tobytes()).hexdigest()
                for c in form.coefficients()
            ],
            "constants": [np.asarray(c.value).tolist() for c in form.constants()],
        }

    def _check_input_identity(self):
        if _json(self.material.summary()) != self.material_identity:
            raise ValueError("Material scientific identity changed during the path.")
        if (
            _json(self.material.state_schema.summary())
            != self._identity["state_schema"]
            or self.material.tangent_convention
            != MaterialTangentConvention.first_piola_deformation_gradient()
            or self.properties.tolist() != self._identity["properties"]
            or self.duration != self._identity["duration"]
            or self.time_unit != self._identity["time_unit"]
            or _json(self.amplitude.summary()) != self._identity["amplitude"]
            or _json(self._load_identity()) != self._identity["load"]
        ):
            raise ValueError(
                "History-material scientific inputs changed after Step construction."
            )

    def _emit(self, kind, **kwargs):
        coordinate = kwargs.get("target_factor", self.accepted_load_factor)
        kwargs.setdefault("time", coordinate * self.duration)
        kwargs.setdefault("coordinate_name", "time")
        kwargs.setdefault("coordinate_value", coordinate * self.duration)
        kwargs.setdefault("coordinate_unit", self.time_unit)
        try:
            self._reporter.emit(SolveEvent(kind, self.name, **kwargs))
        except Exception as error:
            # Observability is not a constitutive rejection: a broken reporter
            # must not trigger numerical cutback or change the physical path.
            self._reporter_failure = error
            raise

    def checkpoint_capabilities(self):
        return checkpointing.CheckpointCapabilities(
            schemas=(self.schema,),
            boundary="accepted_increment",
            payload_scope="full_restart_state",
            state_components=(
                "displacement",
                "committed material state",
                "accepted P/S/dP/dF",
                "physical time and load path",
            ),
            atomic_publication=True,
            rank_count_portability="unsupported",
            identity_scope=(
                "same mesh partition and space",
                "provider summary and state schema",
                "loads, constraints and duration",
            ),
            limitations=("serial same-layout restart", "manual checkpoint cadence"),
            evidence=(
                "continuous/restarted two-law paths",
                "atomic corrupt-restore rejection",
            ),
        )

    def _apply_loading(self, coordinate):
        factor = float(self.amplitude(coordinate))
        if not np.isfinite(factor):
            raise ValueError("Nonfinite load amplitude.")
        self.load_factor.value = PETSc.ScalarType(factor)
        for constant, target in self.prescribed:
            constant.value = factor * target

    def _snapshot(self):
        return {
            "U": self.solution.x.array.copy(),
            "F": self._accepted_F.copy(),
            "state": self.state.snapshot(),
            "response": {
                key: value.x.array.copy() for key, value in self._responses.items()
            },
            "energy_defined": self._energy_defined,
        }

    def _restore(self, snapshot):
        self.solution.x.array[:] = snapshot["U"]
        self.solution.x.scatter_forward()
        self._accepted_U = snapshot["U"].copy()
        self._accepted_F = snapshot["F"].copy()
        self.state.restore(snapshot["state"])
        for key, values in snapshot["response"].items():
            self._responses[key].x.array[:] = values
            self._responses[key].x.scatter_forward()
        self._energy_defined = snapshot["energy_defined"]
        self._apply_loading(self.accepted_load_factor)
        self._at_accepted = True

    def _evaluate(self, target, start):
        result = self.response.update(
            self.material,
            deformation_gradient_old=self._accepted_F,
            deformation_gradient_new=self._gradient(),
            time=target * self.duration,
            time_increment=(target - start) * self.duration,
            properties=self.properties,
        )
        rhs = fem_petsc.assemble_vector(self.residual_form)
        try:
            fem_petsc.apply_lifting(
                rhs,
                [self.tangent_form],
                [self.bcs],
                x0=[self.solution.x.petsc_vec],
                alpha=-1,
            )
            rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            rhs.scale(-1)
            fem_petsc.set_bc(rhs, self.bcs, x0=self.solution.x.petsc_vec, alpha=1)
            return NewtonEvaluation(result, rhs, float(rhs.norm()))
        except BaseException:
            rhs.destroy()
            raise

    def _attempt(self, start, target):
        def correct(evaluation, iteration):
            matrix = fem_petsc.assemble_matrix(self.tangent_form, bcs=self.bcs)
            correction = evaluation.residual.duplicate()
            try:
                matrix.assemble()
                correction.set(0)
                info = solve_matrix_system(
                    matrix,
                    evaluation.residual,
                    correction,
                    self.solver_options.linear_solver,
                    raise_on_failure=False,
                )
                if not info.converged:
                    return NewtonCorrection(
                        False, rejection_reason="Linear solve failed."
                    )
                direction = correction.array_r.copy()
            finally:
                correction.destroy()
                matrix.destroy()
            base = self.solution.x.array.copy()
            alpha = 1.0
            while alpha >= self.solver_options.minimum_step_length:
                self.solution.x.array[:] = base
                self.solution.x.array[: len(direction)] += alpha * direction
                self.solution.x.scatter_forward()
                try:
                    candidate = self._evaluate(target, start)
                except (RuntimeError, ValueError):
                    candidate = None
                if candidate is not None:
                    if (
                        self.solver_options.line_search in (None, "basic")
                        or candidate.residual_norm < evaluation.residual_norm
                    ):
                        return NewtonCorrection(True, alpha, candidate)
                    candidate.residual.destroy()
                alpha *= self.solver_options.line_search_reduction
            return NewtonCorrection(
                False, rejection_reason="Line search rejected all candidates."
            )

        return run_newton_attempt(
            self.solver_options,
            evaluate=lambda: self._evaluate(target, start),
            correct=correct,
            release=lambda r: r.destroy(),
            on_iteration=lambda i, norm, alpha: self._emit(
                "iteration",
                increment=len(self.accepted_increments) + 1,
                start_factor=start,
                target_factor=target,
                iteration=i,
                residual_norm=norm,
                step_length=alpha,
            ),
        )

    def solve(self, *, until=1.0, on_accept=None):
        until = float(until)
        if not self.accepted_load_factor < until <= 1 or not np.isfinite(until):
            raise ValueError(
                "until must exceed the accepted normalized coordinate and be <= 1."
            )
        if self._busy:
            raise RuntimeError("A material Step cannot be advanced recursively.")
        self._check_accepted_solution()
        self._busy = True
        self._reporter_failure = None
        automatic = isinstance(self.incrementation, steps.AutomaticIncrementation)
        proposed = self.next_increment_size or (
            self.incrementation.initial if automatic else 1
        )
        cutbacks = 0
        try:
            self._emit(
                "step_started",
                target_factor=self.accepted_load_factor,
                incrementation="implicit history material / "
                + self.incrementation.summary()["kind"],
            )
            while self.accepted_load_factor < until:
                self._check_input_identity()
                start = self.accepted_load_factor
                if automatic:
                    if (
                        len(self.accepted_increments)
                        >= self.incrementation.max_increments
                    ):
                        raise RuntimeError("Maximum accepted increments reached.")
                    target = min(until, start + proposed)
                else:
                    target = min(
                        until,
                        next(v for v in self.incrementation.load_factors if v > start),
                    )
                target = min(
                    target, next((k for k in self._load_knots if k > start), 1.0)
                )
                if target <= start:
                    raise RuntimeError(
                        "Increment is too small to advance the normalized coordinate."
                    )
                snapshot = self._snapshot()
                self._at_accepted = False
                accepted = False
                try:
                    self._emit(
                        "increment_started",
                        increment=len(self.accepted_increments) + 1,
                        attempt=cutbacks + 1,
                        start_factor=start,
                        target_factor=target,
                    )
                    self._apply_loading(target)
                    # All line-search candidates remain in the prescribed
                    # displacement space. Scale only free-DOF corrections, not
                    # the essential boundary values themselves.
                    fem_petsc.set_bc(self.solution.x.petsc_vec, self.bcs)
                    self.solution.x.scatter_forward()
                    outcome = self._attempt(start, target)
                    scale = (
                        outcome.payload.minimum_suggested_time_scale
                        if outcome.payload is not None
                        else 1.0
                    )
                    accepted = outcome.converged and scale >= 1 - 1e-12
                    reason = (
                        outcome.rejection_reason
                        if scale >= 1 - 1e-12
                        else "Material requested cutback."
                    )
                    iterations, norm = outcome.iterations, outcome.residual_norm
                except (RuntimeError, ValueError) as error:
                    if error is self._reporter_failure:
                        self._restore(snapshot)
                        raise
                    scale, iterations, norm, reason = 1.0, 0, float("inf"), str(error)
                except BaseException:
                    self._restore(snapshot)
                    raise
                record = {
                    "start": start,
                    "coordinate": target,
                    "time": target * self.duration,
                    "dt": (target - start) * self.duration,
                    "iterations": iterations,
                    "residual_norm": norm if np.isfinite(norm) else None,
                    "accepted": accepted,
                    "reason": reason,
                }
                self.attempted_increments.append(record)
                if accepted:
                    try:
                        new_F = self._gradient()
                        self.response.commit()
                    except BaseException:
                        self._restore(snapshot)
                        record["accepted"] = False
                        record["reason"] = "Accepted-state commit failed."
                        raise
                    self._accepted_F = new_F
                    self._accepted_U = self.solution.x.array.copy()
                    self._energy_defined = bool(
                        np.all(outcome.payload.strain_energy_density_defined)
                    )
                    self.accepted_load_factor = target
                    self.accepted_increments.append(record.copy())
                    self._at_accepted = True
                    successful_attempt = cutbacks + 1
                    cutbacks = 0
                    if automatic:
                        proposed = self.incrementation.after_convergence(
                            target - start, iterations
                        )
                        self.next_increment_size = proposed
                    self._emit(
                        "increment_converged",
                        increment=len(self.accepted_increments),
                        attempt=successful_attempt,
                        start_factor=start,
                        target_factor=target,
                        iteration=iterations,
                        residual_norm=norm,
                    )
                    if on_accept is not None:
                        on_accept(self)
                    continue
                self._restore(snapshot)
                cutbacks += 1
                if not automatic:
                    raise RuntimeError(f"{self.name}: rejected increment: {reason}")
                proposed = min(
                    self.incrementation.after_failure(target - start),
                    (target - start) * scale,
                )
                self.next_increment_size = proposed
                if (
                    cutbacks > self.incrementation.max_cutbacks
                    or proposed < self.incrementation.minimum
                ):
                    raise RuntimeError(f"{self.name}: cutback exhausted: {reason}")
                self._emit(
                    "increment_cutback",
                    increment=len(self.accepted_increments) + 1,
                    attempt=cutbacks,
                    start_factor=start,
                    target_factor=target,
                    next_increment=proposed,
                    message=reason,
                )
            self._emit(
                "step_completed" if until == 1.0 else "step_paused",
                target_factor=self.accepted_load_factor,
                increment=len(self.accepted_increments),
                attempt=len(self.attempted_increments),
            )
        except BaseException as error:
            if error is not self._reporter_failure:
                try:
                    self._emit(
                        "step_failed",
                        target_factor=self.accepted_load_factor,
                        message=str(error),
                    )
                except Exception as reporter_error:
                    if hasattr(error, "add_note"):
                        error.add_note(
                            f"Failure reporting also failed: {reporter_error}"
                        )
            raise
        finally:
            self._busy = False
        return self.solution

    def reaction_field(self, *, name="RF"):
        """Full accepted nodal residual; only constrained DOFs are reactions."""
        if not self._at_accepted or not self.accepted_increments:
            raise RuntimeError("Reaction output requires an accepted increment.")
        residual = fem_petsc.assemble_vector(self.residual_form)
        try:
            residual.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            reaction = fem.Function(self.solution.function_space, name=name)
            reaction.x.array[: residual.getLocalSize()] = residual.array_r
            reaction.x.scatter_forward()
            return reaction
        finally:
            residual.destroy()

    def _field_metadata(self, name):
        if name in {"P", "S"}:
            return {
                "stress_measure": "first_piola_kirchhoff" if name == "P" else "cauchy",
                "configuration": "reference" if name == "P" else "current",
                "component_shape": [3, 3],
                "component_order": "row_major",
            }
        if name == "SENER":
            return {
                "availability": "provider_defined_at_current_accepted_state",
                "not_total_work": True,
            }
        if name in self.response.stored_energy_density_components:
            return {
                "availability": "provider_declared_stored_energy_component",
                "not_total_work": True,
            }
        variable = next(
            v
            for v in self.material.state_schema.variables
            if (v.output_name or v.name) == name
        )
        return {
            "state_name": variable.name,
            "component_shape": list(variable.shape),
            "unit": variable.unit,
            "description": variable.description,
        }

    def _visual_fields(self, *, include_energy=True):
        from ..results import recover_integration_point_field

        sources = {
            "S": self.response.cauchy_stress,
            "P": self.response.first_piola_stress,
            **{
                v.output_name or v.name: self.state.committed[v.name]
                for v in self.material.state_schema.variables
            },
        }
        if include_energy and self._energy_defined:
            sources["SENER"] = self.response.strain_energy_density
            sources.update(self.response.stored_energy_density_components)
        recovered = [
            recover_integration_point_field(source, name=f"{name}_CELL")
            for name, source in sources.items()
        ]
        for name, item in zip(sources, recovered):
            item.processing.update(self._field_metadata(name))
        return sources, recovered

    def solve_result(self, *, output=None, metadata=None):
        from .. import io, results
        from ..results.lifecycle import execution_context, attach_checkpoint_contract

        context = execution_context(self)
        if output is None and context is not None:
            output = context.configured_output
        if output is not None and not isinstance(output, (str, Path)):
            raise NotImplementedError(
                "This experimental Step accepts a result path; declarative OutputPlan is not yet supported."
            )
        if output is not None and Path(output).suffix.lower() not in {".xdmf", ".xmf"}:
            raise ValueError(
                "The bounded history-material Step publishes a serial XDMF field series."
            )
        output_start = self.time
        writer = (
            io.ResultWriter(Path(output), self.state.domain, storage="frame_files")
            if output
            else None
        )
        with writer if writer is not None else nullcontext():

            def publish(step):
                # Optional energy availability can change along a provider's
                # path. Do not invent zero energy or change a live field schema.
                _, recovered = step._visual_fields(include_energy=False)
                writer.write(
                    step.time, step.solution, *(item.field for item in recovered)
                )

            if self.accepted_load_factor < 1:
                self.solve(on_accept=publish if writer else None)
            elif writer:
                publish(self)
        result = results.from_solution(
            self.solution,
            name=self.name,
            metadata={"step": self.summary(), **(metadata or {})},
        )
        result.metadata["solve"] = {
            "converged": self.accepted_load_factor == 1.0,
            "accepted_increments": len(self.accepted_increments),
            "attempts": len(self.attempted_increments),
            "scope": "nonlinear_equilibrium_not_scientific_validation",
        }
        if context is not None:
            result.metadata["execution_context"] = context.summary()
        engineering = getattr(self, "engineering_step_summary", None)
        if engineering is not None:
            result.metadata["engineering_step"] = deepcopy(engineering)
        attach_checkpoint_contract(self, result)
        for record in self.checkpoints:
            result.add_checkpoint(record)
        if self.restored_from is not None:
            result.metadata["restart"] = self.restored_from.as_dict()
        results.add_execution_trace(result, self.execution_events)
        result.add_history(
            "load_coordinate",
            [r["time"] for r in self.accepted_increments],
            [r["coordinate"] for r in self.accepted_increments],
            abscissa_name="time",
            abscissa_unit=self.time_unit,
            description="Normalized loading coordinate, distinct from physical time.",
        )
        sources, recovered = self._visual_fields()
        for name, source in sources.items():
            result.add_field(
                name,
                source.function,
                location="quadrature_points",
                sampling=source,
                unit=self._field_metadata(name).get("unit"),
                processing={
                    "method": "accepted_constitutive_update",
                    "committed": True,
                    "representation": "quadrature_values",
                    "postprocessed": False,
                    **self._field_metadata(name),
                },
            )
        for item in recovered:
            result.add_field(
                item.name,
                item.field,
                unit=item.processing.get("unit"),
                location=item.location,
                processing=item.processing,
            )
        result.add_field(
            "RF",
            self.reaction_field(),
            description="Full nodal residual; extract reactions on constrained DOFs only.",
            processing={"method": "assembled_equilibrium_residual", "committed": True},
        )
        if writer:
            for name, path in writer.artifacts().items():
                result.add_artifact(name, path)
            result.metadata["field_output"] = {
                "recommended_visualization_artifact": str(output),
                "geometry": "reference",
                "coordinate": "physical_time",
                "output_scope": "continuation_segment"
                if output_start
                else "accepted_path",
                "start_time": output_start,
            }
            result.metadata["visualization"] = {
                "primary_artifact": "field_output",
                "kind": "accepted_time_series",
                "includes_endpoint": True,
                "geometry": "reference",
            }
        return result

    def summary(self):
        return {
            "kind": "finite_strain_user_material_step",
            "name": self.name,
            "maturity": "experimental",
            "material": self.material_identity,
            "duration": self.duration,
            "time": self.time,
            "time_unit": self.time_unit,
            "accepted_load_factor": self.accepted_load_factor,
            "accepted_increments": list(self.accepted_increments),
            "attempted_increments": list(self.attempted_increments),
            "stored_energy_available": self._energy_defined,
            "limitations": [
                "serial",
                "homogeneous material",
                "strong BCs",
                "no contact or MPC",
                "3D or plane strain",
                "no general energy-balance claim",
            ],
        }

    def save_checkpoint(self, path, *, portable=False):
        if portable:
            raise NotImplementedError(
                "This Step currently supports serial same-layout restart only."
            )
        if not self._at_accepted or not self.accepted_increments:
            raise RuntimeError("Checkpoint requires an accepted increment.")
        if not self.amplitude.serializable:
            raise NotImplementedError(
                "Checkpoint requires an amplitude with declared serializable scientific identity."
            )
        self._check_input_identity()
        self._check_accepted_solution()
        selected = Path(path).with_suffix(".npz")
        snapshot = self._snapshot()
        arrays = {
            "U": snapshot["U"],
            "F": snapshot["F"],
            **{
                f"state_{i}": snapshot["state"][v.name]
                for i, v in enumerate(self.material.state_schema.variables)
            },
            **{
                f"response_{key}": values
                for key, values in snapshot["response"].items()
            },
        }
        checkpointing.atomic_savez(
            selected,
            schema=self.schema,
            identity=json.dumps(self._identity, sort_keys=True),
            ledger=json.dumps(
                {
                    "accepted": self.accepted_increments,
                    "attempted": self.attempted_increments,
                    "coordinate": self.accepted_load_factor,
                    "next_increment": self.next_increment_size,
                    "energy_defined": self._energy_defined,
                },
                allow_nan=False,
            ),
            **arrays,
        )
        self.checkpoints.append(self._checkpoint_record(selected))
        return selected

    def _checkpoint_record(self, path):
        from ..results import CheckpointRecord

        return CheckpointRecord(
            name=f"{self.name}_{self.accepted_load_factor:.17g}",
            path=path,
            schema=self.schema,
            step_name=self.name,
            coordinate_name="time",
            coordinate_value=self.time,
            portable=False,
            metadata={
                "time_unit": self.time_unit,
                "coordinate": self.accepted_load_factor,
                "scope": "serial_same_layout",
                "material": self.material_identity,
            },
        )

    def _validate_ledger(self, ledger):
        """Validate all attempted stations before modifying any live state."""
        coordinate = float(ledger["coordinate"])
        if (
            not 0 < coordinate <= 1
            or not isinstance(ledger["accepted"], list)
            or not ledger["accepted"]
        ):
            raise ValueError("Invalid checkpoint accepted coordinate.")
        if not isinstance(ledger["attempted"], list):
            raise ValueError("Invalid checkpoint attempt ledger.")
        previous, accepted = 0.0, []
        for record in ledger["attempted"]:
            norm = record["residual_norm"]
            if (
                type(record["accepted"]) is not bool
                or record["start"] != previous
                or not previous < record["coordinate"] <= 1
                or record["time"] != record["coordinate"] * self.duration
                or record["dt"] != (record["coordinate"] - previous) * self.duration
                or type(record["iterations"]) is not int
                or record["iterations"] < 0
                or (norm is not None and (not np.isfinite(norm) or norm < 0))
                or (record["accepted"] and norm is None)
            ):
                raise ValueError("Invalid checkpoint physical-time/attempt ledger.")
            if record["accepted"]:
                previous = record["coordinate"]
                accepted.append(record)
        if (
            accepted != ledger["accepted"]
            or previous != coordinate
            or type(ledger["energy_defined"]) is not bool
        ):
            raise ValueError("Checkpoint ledger does not end at the accepted state.")
        next_size = ledger["next_increment"]
        if next_size is not None and (not np.isfinite(next_size) or next_size <= 0):
            raise ValueError("Invalid checkpoint continuation increment.")
        return coordinate, next_size

    def load_checkpoint(self, path):
        if self._busy:
            raise RuntimeError("Cannot restore during an active solve.")
        self._check_input_identity()
        snapshot = self._snapshot()
        with np.load(path, allow_pickle=False) as data:
            if (
                str(data["schema"].item()) != self.schema
                or json.loads(str(data["identity"].item())) != self._identity
            ):
                raise ValueError(
                    "History-material checkpoint scientific identity differs."
                )
            ledger = json.loads(str(data["ledger"].item()))
            coordinate, next_size = self._validate_ledger(ledger)
            candidate = {
                "U": data["U"].copy(),
                "F": data["F"].copy(),
                "energy_defined": ledger["energy_defined"],
                "state": {
                    v.name: data[f"state_{i}"].copy()
                    for i, v in enumerate(self.material.state_schema.variables)
                },
                "response": {
                    key: data[f"response_{key}"].copy() for key in self._responses
                },
            }
            pairs = [(candidate[key], snapshot[key]) for key in ("U", "F")]
            pairs += [
                (candidate[group][key], value)
                for group in ("state", "response")
                for key, value in snapshot[group].items()
            ]
            if any(a.shape != b.shape or not np.isfinite(a).all() for a, b in pairs):
                raise ValueError(
                    "Checkpoint arrays must have matching shapes and finite values."
                )
            if np.any(np.linalg.det(candidate["F"]) <= 0):
                raise ValueError("Checkpoint deformation gradient is inverted.")
        try:
            self._restore(candidate)
            if not np.allclose(
                self._gradient(), candidate["F"], rtol=1e-12, atol=1e-13
            ):
                raise ValueError(
                    "Checkpoint displacement and deformation gradient disagree."
                )
            self._apply_loading(coordinate)
            prescribed = self.solution.x.array.copy()
            for bc in self.bcs:
                bc.set(prescribed)
            if not np.allclose(
                prescribed, self.solution.x.array, rtol=1e-12, atol=1e-13
            ):
                raise ValueError(
                    "Checkpoint displacement violates its prescribed boundary state."
                )
        except BaseException:
            self._restore(snapshot)
            raise
        self.accepted_load_factor = coordinate
        self.accepted_increments = ledger["accepted"]
        self.attempted_increments = ledger["attempted"]
        self.next_increment_size = next_size
        self.restored_from = self._checkpoint_record(path)

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded 3D coupling experiment, not a production Step provider.

Compare staggered implicit Euler to an independently written mixed UFL block
system. The cube permits prescribed inward flux and affine dilation, while
keeping path work distinct from the discrete quadratic balance. This does not
cover general thermodynamic boundary work or a production coupled procedure.
"""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from hashlib import sha256
from math import isfinite

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

from agentfem import (
    checkpointing,
    constitutive,
    fields,
    mesh,
    operators,
    results,
    solvers,
    state,
)
from agentfem.provenance import collective_call
from agentfem.time._staggered import StaggeredFieldIteration


class _ThermoelasticPrototype:
    """Research driver; its single accepted-time owner avoids nested steppers."""

    def __init__(
        self,
        *,
        comm=MPI.COMM_WORLD,
        cells=2,
        dt=0.1,
        alpha=0.002,
        nonuniform=False,
        inward_heat_flux=0.0,
        dilation_rate=None,
        temperature_rate=None,
    ):
        configuration = (
            cells,
            dt,
            alpha,
            nonuniform,
            inward_heat_flux,
            dilation_rate,
            temperature_rate,
        )
        if any(item != configuration for item in comm.allgather(configuration)):
            raise ValueError(
                "AFM-COUPLING-003: prototype configuration differs across ranks."
            )
        if (
            cells < 1
            or cells**3 < comm.size
            or not isfinite(dt)
            or dt <= 0
            or not isfinite(alpha)
            or not isfinite(inward_heat_flux)
            or (dilation_rate is not None and not isfinite(dilation_rate))
            or (temperature_rate is not None and not isfinite(temperature_rate))
            or (temperature_rate is not None and inward_heat_flux != 0)
        ):
            raise ValueError("Positive mesh size/dt and finite alpha required.")
        self.domain = mesh.cuboid(
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (cells,) * 3,
            comm=comm,
            cell_type="hexahedron",
        )
        self.dt, self.completed_steps = dt, 0
        self.cells = cells
        self.alpha, self.nonuniform = alpha, nonuniform
        self.inward_heat_flux = float(inward_heat_flux)
        self.dilation_rate = dilation_rate
        self.prescribed_dilation = fem.Constant(self.domain, 0.0)
        self.temperature_rate = temperature_rate
        self.prescribed_temperature = fem.Constant(self.domain, 0.0)
        self.capacity, self.t0, self.k = 100.0, 300.0, 1.0
        self.material = constitutive.thermoelastic(
            young=1000.0,
            poisson=0.25,
            density=1.0,
            specific_heat=self.capacity,
            conductivity=self.k,
            thermal_expansion=alpha,
            reference_temperature=self.t0,
        )
        self.lam, self.mu = self.material.lambda_, self.material.mu
        self.beta = (3 * self.lam + 2 * self.mu) * alpha
        self.u = fields.displacement(self.domain)
        self.theta = fields.temperature(self.domain, value=0.0)
        self.u_old = fem.Function(self.u.space)
        self.theta_old = fem.Function(self.theta.space)
        self._iteration = StaggeredFieldIteration(
            {
                "displacement": self.u.value,
                "temperature": self.theta.value,
            }
        )
        self.dx = ufl.Measure("dx", domain=self.domain)
        self.ds = ufl.Measure("ds", domain=self.domain)
        self.flux = fem.Constant(self.domain, self.inward_heat_flux)
        self.volume = self._integral(1.0)
        x = ufl.SpatialCoordinate(self.domain)
        self.source = fem.Constant(self.domain, 10.0) * (1 + x[0] if nonuniform else 1)
        self.history = []
        self._in_window = False
        self.restart_source = None
        self._resources = ExitStack()

        a_u = operators.elastic_stiffness(self.u, self.material).expression
        f_u = operators.thermal_expansion_vector(
            self.u, self.t0 + self.theta.value, self.material
        ).expression
        q = self.theta.test
        a_t = (
            self.capacity / self.dt * self.theta.trial * q
            + self.k * ufl.inner(ufl.grad(self.theta.trial), ufl.grad(q))
        ) * self.dx
        feedback = operators.thermoelastic_heat_source(
            q,
            self.u.value - self.u_old,
            coupling_coefficient=self.beta,
            reference_temperature=self.t0,
            dt=self.dt,
            measure=self.dx,
        )
        f_t = (
            (self.capacity / self.dt * self.theta_old + self.source) * q * self.dx
            + feedback.expression
            + self.flux * q * self.ds
        )

        # Independent coupled reference: raw mixed-space forms, no feedback
        # helper, eigenstrain helper or staggered update is reused here.
        cell = self.domain.basix_cell()
        W = fem.functionspace(
            self.domain,
            mixed_element(
                [element("Lagrange", cell, 1, shape=(3,)), element("Lagrange", cell, 1)]
            ),
        )
        self.reference = fem.Function(W)
        self.reference_old = fem.Function(W)
        u, t = ufl.TrialFunctions(W)
        v, s = ufl.TestFunctions(W)
        old_u, old_t = ufl.split(self.reference_old)
        eps = lambda value: ufl.sym(ufl.grad(value))
        a_ref = (
            self.lam * ufl.div(u) * ufl.div(v)
            + 2 * self.mu * ufl.inner(eps(u), eps(v))
            - self.beta * t * ufl.div(v)
            + self.capacity / self.dt * t * s
            + self.k * ufl.inner(ufl.grad(t), ufl.grad(s))
            + self.beta * self.t0 / self.dt * ufl.div(u) * s
        ) * self.dx
        f_ref = (
            (
                self.source
                + self.capacity / self.dt * old_t
                + self.beta * self.t0 / self.dt * ufl.div(old_u)
            )
            * s
            * self.dx
        )
        f_ref += self.flux * s * self.ds
        thermal_bcs = self._thermal_bcs(self.theta.space)
        self._thermal_residual = fem.form(ufl.action(a_t, self.theta.value) - f_t)
        self._thermal_boundary_dofs = (
            thermal_bcs[0].dof_indices()[0][: thermal_bcs[0].dof_indices()[1]]
            if thermal_bcs
            else np.array([], dtype=np.int32)
        )
        try:
            self.heat = self._resources.enter_context(
                solvers.prepare_linear_problem(
                    a_t, f_t, self.theta.value, bcs=thermal_bcs
                )
            )
            self.solid = self._resources.enter_context(
                solvers.prepare_linear_problem(
                    a_u, f_u, self.u.value, bcs=self._rollers(self.u.space)
                )
            )
            self.monolithic = self._resources.enter_context(
                solvers.prepare_linear_problem(
                    a_ref,
                    f_ref,
                    self.reference,
                    bcs=self._rollers(W.sub(0)) + self._thermal_bcs(W.sub(1)),
                )
            )
        except BaseException:
            self.close()
            raise

    def _rollers(self, V):
        bcs = []
        for axis in range(3):
            facets = dmesh.locate_entities_boundary(
                self.domain, 2, lambda x, axis=axis: np.isclose(x[axis], 0.0)
            )
            dofs = fem.locate_dofs_topological(V.sub(axis), 2, facets)
            bcs.append(fem.dirichletbc(0.0, dofs, V.sub(axis)))
            if self.dilation_rate is not None:
                upper = dmesh.locate_entities_boundary(
                    self.domain, 2, lambda x, axis=axis: np.isclose(x[axis], 1.0)
                )
                upper_dofs = fem.locate_dofs_topological(V.sub(axis), 2, upper)
                bcs.append(
                    fem.dirichletbc(self.prescribed_dilation, upper_dofs, V.sub(axis))
                )
        return bcs

    def _thermal_bcs(self, V):
        if self.temperature_rate is None:
            return []
        facets = dmesh.locate_entities_boundary(
            self.domain,
            2,
            lambda x: np.any(np.isclose(x, 0) | np.isclose(x, 1), axis=0),
        )
        dofs = fem.locate_dofs_topological(V, 2, facets)
        return [fem.dirichletbc(self.prescribed_temperature, dofs, V)]

    @contextmanager
    def _motion_window(self):
        if self._in_window:
            raise RuntimeError(
                "AFM-COUPLING-004: nested coupled window is unsupported."
            )
        self._in_window = True
        previous = float(self.prescribed_dilation.value)
        old_temperature = float(self.prescribed_temperature.value)
        if self.dilation_rate is not None:
            self.prescribed_dilation.value = (
                self.dilation_rate * (self.completed_steps + 1) * self.dt
            )
        if self.temperature_rate is not None:
            self.prescribed_temperature.value = (
                self.temperature_rate * (self.completed_steps + 1) * self.dt
            )
        try:
            yield
        except BaseException:
            self.prescribed_dilation.value = previous
            self.prescribed_temperature.value = old_temperature
            raise
        finally:
            self._in_window = False

    def _checkpoint_arguments(self, total_steps):
        declaration = (total_steps, self.completed_steps, self._in_window)
        if any(item != declaration for item in self.domain.comm.allgather(declaration)):
            raise ValueError(
                "AFM-COUPLING-003: checkpoint boundary differs across ranks."
            )
        if self._in_window:
            raise RuntimeError(
                "AFM-COUPLING-004: checkpoint requires an accepted boundary."
            )
        if (
            not isinstance(total_steps, int)
            or total_steps < self.completed_steps
            or total_steps < 1
        ):
            raise ValueError("Checkpoint total_steps must cover the accepted path.")
        return dict(
            step_kind="thermoelastic_verification_prototype",
            step_name="coupled_cube",
            procedure={"method": "staggered_backward_euler", "schema": 1},
            dt=self.dt,
            total_steps=total_steps,
            time_inputs={
                "kind": "time_input_plan",
                "restart_identity_bound": True,
                "identity": {
                    "cells": self.cells,
                    "alpha": self.alpha,
                    "capacity": self.capacity,
                    "t0": self.t0,
                    "k": self.k,
                    "lambda": self.lam,
                    "mu": self.mu,
                    "nonuniform_source": self.nonuniform,
                    "inward_heat_flux": self.inward_heat_flux,
                    "dilation_rate": self.dilation_rate,
                    "temperature_rate": self.temperature_rate,
                },
            },
        )

    def save_checkpoint(self, path, *, total_steps, portable=True):
        """Save one joint accepted state using the existing transient envelope."""
        arguments = self._checkpoint_arguments(total_steps)
        return checkpointing.save_transient_checkpoint(
            path,
            **arguments,
            completed_steps=self.completed_steps,
            state={
                "displacement": self.u_old,
                "temperature": self.theta_old,
                "reference_displacement": self.reference_old.sub(0).collapse(),
                "reference_temperature": self.reference_old.sub(1).collapse(),
            },
            accepted_times=[item["time"] for item in self.history],
            history_records=self.history,
            auxiliary_state={"history_sha256": self._history_digest(self.history)},
            portable=portable,
        )

    @staticmethod
    def _history_digest(history):
        return sha256(
            json.dumps(history, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def load_checkpoint(self, path, *, total_steps):
        """Stage fields and evidence before mutating any live accepted state."""
        arguments = self._checkpoint_arguments(total_steps)
        staged = {
            "displacement": fem.Function(self.u.space),
            "temperature": fem.Function(self.theta.space),
            "reference_displacement": self.reference_old.sub(0).collapse(),
            "reference_temperature": self.reference_old.sub(1).collapse(),
        }
        metadata = checkpointing.load_transient_checkpoint(
            path, **arguments, state=staged
        )
        count = metadata["completed_steps"]
        history = metadata["history_records"]
        if (
            type(count) is not int
            or not 0 <= count <= total_steps
            or len(history) != count
            or metadata["time"] != count * self.dt
            or metadata["accepted_times"] != [(i + 1) * self.dt for i in range(count)]
            or (metadata.get("auxiliary_state") or {}).get("history_sha256")
            != self._history_digest(history)
        ):
            raise ValueError(
                "AFM-COUPLING-005: inconsistent accepted checkpoint history."
            )
        json.dumps(history, allow_nan=False)
        for i, record in enumerate(history):
            if record["step"] != i + 1 or record["time"] != (i + 1) * self.dt:
                raise ValueError("AFM-COUPLING-005: inconsistent checkpoint time.")
        with state.field_transaction(
            u=self.u,
            t=self.theta,
            u_old=self.u_old,
            t_old=self.theta_old,
            reference=self.reference,
            reference_old=self.reference_old,
        ):
            for target, name in (
                (self.u.value, "displacement"),
                (self.u_old, "displacement"),
                (self.theta.value, "temperature"),
                (self.theta_old, "temperature"),
            ):
                self._copy(target, staged[name])
            for target in (self.reference, self.reference_old):
                target.sub(0).interpolate(staged["reference_displacement"])
                target.sub(1).interpolate(staged["reference_temperature"])
                target.x.scatter_forward()
        self.completed_steps, self.history = count, deepcopy(history)
        self.prescribed_dilation.value = (
            0.0 if self.dilation_rate is None else self.dilation_rate * count * self.dt
        )
        self.restart_source = {
            "manifest": metadata["manifest_path"],
            "mode": metadata["restart_mode"],
            "accepted_step": count,
        }
        self.prescribed_temperature.value = (
            0.0
            if self.temperature_rate is None
            else self.temperature_rate * count * self.dt
        )
        return metadata

    def _integral(self, expression):
        local = fem.assemble_scalar(fem.form(expression * self.dx))
        value = float(self.domain.comm.allreduce(local, op=MPI.SUM).real)
        if not isfinite(value):
            raise RuntimeError(
                "AFM-COUPLING-002: non-finite global diagnostic; window rejected."
            )
        return value

    def _norm(self, expression):
        return float(
            np.sqrt(
                max(
                    0.0, self._integral(ufl.inner(expression, expression)) / self.volume
                )
            )
        )

    def _boundary_integral(self, expression):
        local = fem.assemble_scalar(fem.form(expression * self.ds))
        value = float(self.domain.comm.allreduce(local, op=MPI.SUM).real)
        if not isfinite(value):
            raise RuntimeError("AFM-COUPLING-002: non-finite boundary diagnostic.")
        return value

    @staticmethod
    def _copy(target, source):
        target.x.array[:] = source.x.array
        target.x.scatter_forward()

    def advance(
        self,
        *,
        relaxation=1.0,
        max_iterations=200,
        rtol=1e-9,
        temperature_atol=1e-11,
        displacement_atol=1e-13,
        acceptance_check=None,
    ):
        """Accept one joint window or restore both fields without moving time."""
        controls = (relaxation, rtol, temperature_atol, displacement_atol)
        declaration = (*controls, max_iterations, acceptance_check is not None)
        if any(item != declaration for item in self.domain.comm.allgather(declaration)):
            raise ValueError(
                "AFM-COUPLING-003: iteration controls differ across ranks."
            )
        if (
            not all(isfinite(v) for v in controls)
            or not 0 < relaxation <= 1
            or min(rtol, temperature_atol, displacement_atol) <= 0
            or max_iterations < 1
        ):
            raise ValueError(
                "Finite positive tolerances, 0 < relaxation <= 1 and iterations required."
            )
        with (
            self._motion_window(),
            state.field_transaction(
                displacement=self.u, temperature=self.theta, reference=self.reference
            ),
        ):
            residuals = self._iteration.run(
                (self.heat.solve, self.solid.solve),
                absolute_tolerances={
                    "displacement": displacement_atol,
                    "temperature": temperature_atol,
                },
                rtol=rtol,
                relaxation=relaxation,
                max_iterations=max_iterations,
                decoupled=self.beta == 0,
            )
            iteration = len(residuals)
            self.monolithic.solve()
            u_ref, t_ref = (self.reference.sub(i).collapse() for i in range(2))
            record = {
                "step": self.completed_steps + 1,
                "time": (self.completed_steps + 1) * self.dt,
                "outer_iterations": iteration,
                "residuals": residuals,
                "iteration_policy": {
                    "relaxation": relaxation,
                    "max_iterations": max_iterations,
                    "rtol": rtol,
                    "temperature_atol": temperature_atol,
                    "displacement_atol": displacement_atol,
                },
                "convergence_reason": "decoupled_blocks"
                if self.beta == 0
                else "both_field_residuals",
                "displacement_reference_error": self._norm(self.u.value - u_ref),
                "temperature_reference_error": self._norm(self.theta.value - t_ref),
            }
            record.update(self._balance())
            if acceptance_check is not None:
                collective_call(
                    lambda: acceptance_check(record),
                    comm=self.domain.comm,
                    label="accept thermoelastic window",
                )
        self._copy(self.u_old, self.u.value)
        self._copy(self.theta_old, self.theta.value)
        self._copy(self.reference_old, self.reference)
        self.completed_steps += 1
        self.history.append(record)
        return record

    def _balance(self):
        u, old = self.u.value, self.u_old
        t, t_old = self.theta.value, self.theta_old
        du, delta = u - old, t - t_old

        def elastic(a, b):
            return self.lam * ufl.div(a) * ufl.div(b) + 2 * self.mu * ufl.inner(
                ufl.sym(ufl.grad(a)), ufl.sym(ufl.grad(b))
            )

        change = self._integral(
            0.5 * (elastic(u, u) - elastic(old, old))
            + self.capacity / (2 * self.t0) * (t * t - t_old * t_old)
        )
        numerical = self._integral(
            0.5 * elastic(du, du) + self.capacity / (2 * self.t0) * delta * delta
        )
        conduction = (
            self.dt
            * self.k
            / self.t0
            * self._integral(ufl.inner(ufl.grad(t), ufl.grad(t)))
        )
        supply = self.dt / self.t0 * self._integral(self.source * t)
        boundary_heat = self.dt * self._boundary_integral(self.flux)
        boundary_supply = self.dt / self.t0 * self._boundary_integral(self.flux * t)
        dirichlet_heat = dirichlet_supply = 0.0
        if self.temperature_rate is not None:
            residual = fem_petsc.assemble_vector(self._thermal_residual)
            try:
                residual.ghostUpdate(
                    addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                )
                dofs = self._thermal_boundary_dofs
                local_heat = float(np.sum(residual.array[dofs]))
                local_supply = float(
                    np.dot(residual.array[dofs], self.theta.value.x.array[dofs])
                )
                dirichlet_heat = self.dt * self.domain.comm.allreduce(
                    local_heat, op=MPI.SUM
                )
                dirichlet_supply = (
                    self.dt
                    / self.t0
                    * self.domain.comm.allreduce(local_supply, op=MPI.SUM)
                )
            finally:
                residual.destroy()
        # Dilation is a unit affine virtual displacement on the prescribed
        # faces. Its weak residual gives the conjugate reaction, without
        # extrapolating boundary stresses. Old and new accepted stations are
        # distinct from the endpoint work in the backward-Euler identity.
        reaction = self._integral(
            (3 * self.lam + 2 * self.mu) * ufl.div(u) - 3 * self.beta * t
        )
        old_reaction = self._integral(
            (3 * self.lam + 2 * self.mu) * ufl.div(old) - 3 * self.beta * t_old
        )
        increment = 0.0 if self.dilation_rate is None else self.dilation_rate * self.dt
        endpoint_work = reaction * increment
        path_work = 0.5 * (reaction + old_reaction) * increment
        heat = self._integral(
            self.capacity * delta
            + self.beta * self.t0 * ufl.div(du)
            - self.dt * self.source
        )
        return {
            "dilation_reaction": reaction if self.dilation_rate is not None else None,
            "prescribed_motion_path_work": path_work,
            "prescribed_motion_endpoint_work": endpoint_work,
            "boundary_heat_input": boundary_heat,
            "prescribed_temperature_heat_input": dirichlet_heat,
            "prescribed_temperature_quadratic_supply": dirichlet_supply,
            "volume_heat_input": self.dt * self._integral(self.source),
            "quadratic_boundary_supply": boundary_supply,
            "linearized_heat_balance_absolute": abs(
                heat - boundary_heat - dirichlet_heat
            ),
            "quadratic_balance_absolute": abs(
                change
                + numerical
                + conduction
                - supply
                - boundary_supply
                - endpoint_work
                - dirichlet_supply
            ),
            "quadratic_energy_change": change,
            "time_discretization_term": numerical,
            "conduction_term": conduction,
            "quadratic_supply": supply,
        }

    def result(self):
        if not self.history:
            raise RuntimeError("No accepted coupled window is available.")
        result = results.SimulationResult("thermoelastic_coupling_prototype")
        result.add_field("Displacement", self.u.value.copy(), unit="m")
        result.add_field("TemperatureDeparture", self.theta.value.copy(), unit="K")
        for name, unit in (
            ("temperature_reference_error", "K"),
            ("displacement_reference_error", "m"),
            ("linearized_heat_balance_absolute", "J"),
            ("quadratic_balance_absolute", "J"),
        ):
            result.add_quantity(
                "maximum_" + name,
                max(item[name] for item in self.history),
                unit=unit,
                kind="diagnostic",
                description="Maximum over accepted windows; field errors are volume-normalized physical RMS norms.",
            )
        result.metadata.update(
            {
                "maturity": "experimental_benchmark_only",
                "inputs": {
                    "domain": "unit_cube",
                    "element": "hexahedron_Q1",
                    "boundary_conditions": "prescribed inward heat flux; lower-face rollers; upper-face dilation if configured",
                    "inward_heat_flux": self.inward_heat_flux,
                    "dilation_rate": self.dilation_rate,
                    "temperature_rate": self.temperature_rate,
                    "cells_per_axis": self.cells,
                    "dt": self.dt,
                    "young": 1000.0,
                    "poisson": 0.25,
                    "alpha": self.alpha,
                    "volumetric_capacity_constant_strain": self.capacity,
                    "reference_temperature": self.t0,
                    "conductivity": self.k,
                    "source": "10*(1+x)" if self.nonuniform else "10",
                    "unit_system": "SI",
                },
                "residual_norm": "volume_normalized_FE_RMS",
                "residual_units": {"temperature": "K", "displacement": "m"},
                "accepted_steps": self.completed_steps,
                "restart_source": deepcopy(self.restart_source),
                "history": deepcopy(self.history),
                "scope": "homogeneous_small_strain_3d_prescribed_flux_optional_dilation",
                "energy_semantics": "linearized_heat_and_quadratic_identity_not_general_first_law",
                "matrix_assemblies": {
                    "thermal": self.heat.matrix_assembly_count,
                    "mechanical": self.solid.matrix_assembly_count,
                    "reference": self.monolithic.matrix_assembly_count,
                },
                "solve_counts": {
                    "thermal": self.heat.solve_count,
                    "mechanical": self.solid.solve_count,
                },
            }
        )
        result.add_scientific_inputs(
            model=result.metadata["inputs"],
            iteration_policies=[record["iteration_policy"] for record in self.history],
        )
        return result

    def close(self):
        self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded 3D coupling experiment, not a production Step provider.

Compare staggered implicit Euler to an independently written mixed UFL block
system. The homogeneous, insulated cube uses symmetry rollers and no applied
mechanical work. This does not cover general thermodynamic boundary work.
"""

from contextlib import ExitStack
from copy import deepcopy
from math import isfinite

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, operators, results, solvers, state
from agentfem.provenance import collective_call


class _ThermoelasticPrototype:
    """Research driver; its single accepted-time owner avoids nested steppers."""

    def __init__(
        self, *, comm=MPI.COMM_WORLD, cells=2, dt=0.1, alpha=0.002, nonuniform=False
    ):
        configuration = (cells, dt, alpha, nonuniform)
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
        self.u_iter = fem.Function(self.u.space)
        self.t_iter = fem.Function(self.theta.space)
        self.dx = ufl.Measure("dx", domain=self.domain)
        self.volume = self._integral(1.0)
        x = ufl.SpatialCoordinate(self.domain)
        self.source = fem.Constant(self.domain, 10.0) * (1 + x[0] if nonuniform else 1)
        self.history = []
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
            self.capacity / self.dt * self.theta_old + self.source
        ) * q * self.dx + feedback.expression

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
        try:
            self.heat = self._resources.enter_context(
                solvers.prepare_linear_problem(a_t, f_t, self.theta.value)
            )
            self.solid = self._resources.enter_context(
                solvers.prepare_linear_problem(
                    a_u, f_u, self.u.value, bcs=self._rollers(self.u.space)
                )
            )
            self.monolithic = self._resources.enter_context(
                solvers.prepare_linear_problem(
                    a_ref, f_ref, self.reference, bcs=self._rollers(W.sub(0))
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
        return bcs

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
        residuals = []
        with state.field_transaction(
            displacement=self.u, temperature=self.theta, reference=self.reference
        ):
            for iteration in range(1, max_iterations + 1):
                self._copy(self.u_iter, self.u.value)
                self._copy(self.t_iter, self.theta.value)
                self.heat.solve()
                self.solid.solve()
                # Both residuals precede relaxation; near-zero scales rely on
                # absolute tolerances. Physical integrals exclude MPI ghosts.
                ru = self._norm(self.u.value - self.u_iter)
                rt = self._norm(self.theta.value - self.t_iter)
                su, st = self._norm(self.u.value), self._norm(self.theta.value)
                residuals.append(
                    {
                        "iteration": iteration,
                        "temperature_absolute": rt,
                        "displacement_absolute": ru,
                        "temperature_relative": rt / st if st else None,
                        "displacement_relative": ru / su if su else None,
                    }
                )
                if self.beta == 0 or (
                    rt <= temperature_atol + rtol * st
                    and ru <= displacement_atol + rtol * su
                ):
                    break
                if iteration == max_iterations:
                    raise RuntimeError(
                        "AFM-COUPLING-001: outer iteration exhausted; window rejected."
                    )
                for value, previous in (
                    (self.u.value, self.u_iter),
                    (self.theta.value, self.t_iter),
                ):
                    value.x.array[:] = previous.x.array + relaxation * (
                        value.x.array - previous.x.array
                    )
                    value.x.scatter_forward()
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
        heat = self._integral(
            self.capacity * delta
            + self.beta * self.t0 * ufl.div(du)
            - self.dt * self.source
        )
        return {
            "linearized_heat_balance_absolute": abs(heat),
            "quadratic_balance_absolute": abs(change + numerical + conduction - supply),
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
                    "boundary_conditions": "insulated; normal displacement zero on x=0, y=0, z=0",
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
                "history": deepcopy(self.history),
                "scope": "homogeneous_small_strain_3d_insulated_zero_mechanical_work",
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

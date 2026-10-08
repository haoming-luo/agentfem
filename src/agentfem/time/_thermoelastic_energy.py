# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Discrete heat and work evidence, not a nonlinear first-law claim."""

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc


def _owned_boundary_dofs(bcs):
    arrays = [bc.dof_indices()[0][: bc.dof_indices()[1]] for bc in bcs]
    return np.unique(np.concatenate(arrays)) if arrays else np.array([], dtype=np.int32)


class ThermoelasticEnergyEvidence:
    def __init__(self, step, mechanical_bcs, thermal_bcs, heat_load, mechanical_load):
        self.step = step
        self.comm = step.comm
        self.ub = _owned_boundary_dofs(mechanical_bcs)
        self.tb = _owned_boundary_dofs(thermal_bcs)
        self.owned_u = (
            step.u.space.dofmap.index_map.size_local * step.u.space.dofmap.index_map_bs
        )
        self.old_external = step.u.value.copy()
        self.old_external.x.array[:] = 0
        self.old_reaction = step.u.value.copy()
        self.old_reaction.x.array[:] = 0
        u, old = step.u.value, step.u_old
        t, t_old = step.theta.value, step.theta_old
        domain = step.u.space.mesh
        dx = ufl.Measure("dx", domain=domain)
        dt, t0, capacity = (
            step.dt,
            step.material.reference_temperature,
            step.blocks.capacity,
        )
        du, delta = u - old, t - t_old
        lam, mu = step.material.lambda_, step.material.mu

        def elastic(a, b):
            return lam * ufl.div(a) * ufl.div(b) + 2 * mu * ufl.inner(
                ufl.sym(ufl.grad(a)), ufl.sym(ufl.grad(b))
            )

        elastic_change = 0.5 * (elastic(u, u) - elastic(old, old))
        self.forms = {
            "elastic_strain_energy_change": fem.form(elastic_change * dx),
            "quadratic_energy_change": fem.form(
                (elastic_change + capacity / (2 * t0) * (t * t - t_old * t_old)) * dx
            ),
            "time_discretization_term": fem.form(
                (0.5 * elastic(du, du) + capacity / (2 * t0) * delta * delta) * dx
            ),
            "conduction_term": fem.form(
                dt
                * step.material.conductivity
                / t0
                * ufl.inner(ufl.grad(t), ufl.grad(t))
                * dx
            ),
            "linearized_heat_storage_change": fem.form(
                (capacity * delta + step.blocks.beta * t0 * ufl.div(du)) * dx
            ),
        }
        if heat_load is not None:
            heat_load = getattr(heat_load, "expression", heat_load)
            self.forms["natural_heat_input"] = fem.form(
                dt
                * ufl.replace(heat_load, {step.theta.test: fem.Constant(domain, 1.0)})
            )
            self.forms["quadratic_natural_supply"] = fem.form(
                dt / t0 * ufl.replace(heat_load, {step.theta.test: t})
            )
        self.mechanical_residual = fem.form(
            ufl.action(step.blocks.mechanical_matrix, u) - step.blocks.mechanical_rhs
        )
        self.thermal_residual = fem.form(
            ufl.action(step.blocks.thermal_matrix, t) - step.blocks.thermal_rhs
        )
        self.external_form = (
            None
            if mechanical_load is None
            else fem.form(getattr(mechanical_load, "expression", mechanical_load))
        )
        self._assign_force(self.old_reaction, self.mechanical_residual, self.ub)
        self._assign_force(self.old_external, self.external_form)

    def _vector(self, form):
        vector = fem_petsc.assemble_vector(form)
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return vector

    def _assign_force(self, target, form, indices=None):
        target.x.array[:] = 0
        if form is not None:
            vector = self._vector(form)
            try:
                selected = slice(0, self.owned_u) if indices is None else indices
                target.x.array[selected] = vector.array[selected]
            finally:
                vector.destroy()
        target.x.scatter_forward()

    def evaluate(self):
        step = self.step
        values = {
            name: float(self.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM).real)
            for name, form in self.forms.items()
        }
        values.setdefault("natural_heat_input", 0.0)
        values.setdefault("quadratic_natural_supply", 0.0)
        reaction, external = self.old_reaction.copy(), self.old_external.copy()
        self._assign_force(reaction, self.mechanical_residual, self.ub)
        self._assign_force(external, self.external_form)
        thermal = self._vector(self.thermal_residual)
        try:
            heat = step.dt * float(np.sum(thermal.array[self.tb]))
            weighted = (
                step.dt
                / step.material.reference_temperature
                * float(
                    np.dot(thermal.array[self.tb], step.theta.value.x.array[self.tb])
                )
            )
        finally:
            thermal.destroy()
        values["prescribed_temperature_heat_input"] = self.comm.allreduce(
            heat, op=MPI.SUM
        )
        values["quadratic_prescribed_temperature_supply"] = self.comm.allreduce(
            weighted, op=MPI.SUM
        )
        delta = (step.u.value.x.array - step.u_old.x.array)[: self.owned_u]
        for name, current, previous in (
            ("prescribed_motion", reaction, self.old_reaction),
            ("natural_load", external, self.old_external),
        ):
            endpoint = float(np.dot(current.x.array[: self.owned_u], delta))
            path = 0.5 * float(
                np.dot((current.x.array + previous.x.array)[: self.owned_u], delta)
            )
            values[name + "_endpoint_work"] = self.comm.allreduce(endpoint, op=MPI.SUM)
            values[name + "_path_work"] = self.comm.allreduce(path, op=MPI.SUM)
        values["linearized_heat_balance_absolute"] = abs(
            values["linearized_heat_storage_change"]
            - values["natural_heat_input"]
            - values["prescribed_temperature_heat_input"]
        )
        values["quadratic_balance_absolute"] = abs(
            values["quadratic_energy_change"]
            + values["time_discretization_term"]
            + values["conduction_term"]
            - values["quadratic_natural_supply"]
            - values["quadratic_prescribed_temperature_supply"]
            - values["prescribed_motion_endpoint_work"]
            - values["natural_load_endpoint_work"]
        )
        if not all(np.isfinite(value) for value in values.values()):
            raise RuntimeError("AFM-COUPLING-002: non-finite heat/work evidence.")
        return values, reaction, external

    def commit(self, reaction, external):
        for target, value in (
            (self.old_reaction, reaction),
            (self.old_external, external),
        ):
            target.x.array[:] = value.x.array
            target.x.scatter_forward()

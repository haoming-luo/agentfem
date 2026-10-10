# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded elastic interface Procedure using the shared solver/result owners."""

import numpy as np
import ufl


class ElasticInterfaceStep:
    """One proportional, small-strain elastic equilibrium from a stress-free origin."""

    def __init__(
        self,
        model,
        target,
        force,
        *,
        constraints=None,
        solver_options=None,
        name="elastic_interface",
    ):
        from . import constraints as api, loads
        from ._elastic_cohesive import ElasticCohesiveLaw
        from ._nonmatching_force import NonmatchingCohesiveForce
        from .solvers import LinearSolverOptions

        if not isinstance(force, NonmatchingCohesiveForce) or not isinstance(
            force.assembler.law, ElasticCohesiveLaw
        ):
            raise TypeError(
                "Elastic interface Step requires a nonmatching elastic law; damage needs an incremental provider."
            )
        if force.displacement is not target.value:
            raise ValueError(
                "Interface and Step must share the same displacement field."
            )
        if force.assembler.pairing.method not in {"coplanar-triangle-common-refinement", "coplanar-affine-q1-common-refinement"}:
            raise NotImplementedError(
                "The first elastic Step requires reviewed common-refinement integration."
            )
        if model.boundary_models or model.eigenstrains:
            raise NotImplementedError(
                "Elastic interface Step does not yet accept other boundary models or eigenstrain."
            )
        assets = tuple(model.constraints if constraints is None else constraints)
        if any(
            not isinstance(item, api.DirichletConstraint)
            for item in api.constraint_assets(assets)
        ):
            raise NotImplementedError(
                "Elastic interface Step requires ordinary strong constraints."
            )
        if solver_options is not None and not isinstance(
            solver_options, LinearSolverOptions
        ):
            raise TypeError("Elastic interface Step requires LinearSolverOptions.")
        model.check(target=target)
        update = model._time_update_callback()
        if update is not None:
            update(1.0)
        self.model, self.target, self.force = model, target, force
        self.name, self.solver_options = name, solver_options
        self.bcs = tuple(item.bc for item in api.dirichlet_constraints(assets))
        self.K = model.stiffness(target)
        self.F = (
            model.external_force(target)
            if model.loads
            else model.external_force(
                target,
                load=loads.body_force((0.0, 0.0, 0.0), target=target, name="zero_load"),
            )
        )
        self.last_solve_info = None
        self.evidence = None
        self.reaction = None

    def solve(self):
        from dolfinx import fem
        from dolfinx.fem import petsc as fem_petsc
        from . import operators, solvers
        from .fracture import _owned_bc_global_dofs

        field = self.target.value
        previous = field.x.array.copy()
        resources = []
        try:
            matrix = operators.assemble_matrix(self.K)
            resources.append(matrix)
            self.force.add_to_matrix(matrix)
            matrix.assemble()
            load = operators.assemble_vector(self.F)
            rhs = load.copy()
            prescribed = field.x.petsc_vec.duplicate()
            resources.extend((load, rhs, prescribed))
            prescribed.set(0)
            fem_petsc.set_bc(prescribed, list(self.bcs))
            fixed = _owned_bc_global_dofs(field, self.bcs)
            original = matrix.copy()
            resources.append(original)
            matrix.zeroRowsColumns(fixed, diag=1, x=prescribed, b=rhs)
            self.last_solve_info = solvers.solve_matrix_system(
                matrix,
                rhs,
                field.x.petsc_vec,
                self.solver_options,
                raise_on_failure=True,
            )
            if not np.all(np.isfinite(field.x.array)):
                raise ValueError("Non-finite interface solution.")
            reaction = fem.Function(field.function_space, name="RF")
            original.mult(field.x.petsc_vec, reaction.x.petsc_vec)
            reaction.x.array[:] -= load.array
            response = self.force.begin()
            free = np.setdiff1d(np.arange(field.x.array.size), fixed)
            bulk_energy = 0.5 * float(
                fem.assemble_scalar(
                    fem.form(ufl.action(ufl.action(self.K.expression, field), field))
                )
            )
            total_energy = bulk_energy + response.stored_energy
            # Avoid naming endpoint force times displacement as path work:
            # the factor 1/2 is specific to this proportional linear path.
            natural_work = 0.5 * field.x.petsc_vec.dot(load)
            prescribed_work = 0.5 * float(
                np.dot(field.x.array[fixed], reaction.x.array[fixed])
            )
            self.evidence = {
                "free_residual_norm": float(np.linalg.norm(reaction.x.array[free])),
                "interface_stored_energy": response.stored_energy,
                "bulk_strain_energy": bulk_energy,
                "natural_load_work": float(natural_work),
                "prescribed_motion_work": prescribed_work,
                "energy_balance_residual": float(total_energy)
                - natural_work
                - prescribed_work,
            }
            self.force.commit()
            self.reaction = reaction
            return field
        except Exception:
            field.x.array[:] = previous
            field.x.scatter_forward()
            self.force.rollback()
            self.evidence = None
            raise
        finally:
            for resource in reversed(resources):
                resource.destroy()

    def solve_result(
        self, *, output=None, field_variables=None, metadata=None, strict_output=False
    ):
        from .results import from_solution
        from .results.lifecycle import complete_result

        if field_variables is not None and not set(field_variables).issubset(
            {"U", "RF"}
        ):
            raise ValueError("Elastic interface Step currently exports U and RF only.")
        solution = self.solve()
        units = self.model.unit_system
        result = from_solution(
            solution,
            name=self.name,
            field_name="U",
            unit=None if units is None else units.length,
            metadata={
                "step": self.summary(),
                "solve": self.last_solve_info.as_dict(),
                "energy_path": "proportional_linear_from_stress_free_origin",
                "maturity": "experimental_serial_small_strain_elastic",
            },
        )
        if field_variables is None or "RF" in field_variables:
            result.add_field(
                "RF",
                self.reaction,
                location="nodes",
                unit=None if units is None else units.force,
            )
        for key, value in self.evidence.items():
            unit = (
                None
                if units is None
                else (units.force if key == "free_residual_norm" else units.energy)
            )
            result.add_quantity(key, value, unit=unit, kind="diagnostic")
        return complete_result(
            self, result, output=output, metadata=metadata, strict_output=strict_output
        )

    def summary(self):
        return {
            "kind": "elastic_interface_step",
            "name": self.name,
            "pairing": self.force.assembler.pairing.summary(),
            "law": self.force.assembler.law.summary(),
            "restart": "not_supported",
            "finite_rotation": False,
            "energy_path": "proportional_linear_from_stress_free_origin",
        }


def lower(model, request):
    if any(request.option(key) is not None for key in ("K", "F", "material")):
        raise ValueError(
            "Elastic interface Step currently lowers registered materials and loads; direct K/F/material overrides are unsupported."
        )
    return ElasticInterfaceStep(
        model,
        request.target,
        request.option("cohesive_force"),
        constraints=request.option("constraints"),
        solver_options=request.option("solver_options"),
        name=request.option("name") or "elastic_interface",
    )

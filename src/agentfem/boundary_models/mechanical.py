"""Reusable mechanical weak boundary models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import ufl
from mpi4py import MPI

from agentfem.ir.values import describe_value
from agentfem.kernel import constants
from agentfem.operators import OperatorForm


@dataclass(frozen=True)
class ElasticFoundation:
    """Distributed linear spring support on a solid boundary."""

    stiffness: object
    location: object
    mode: str = "isotropic"
    normal: object | None = None
    name: str = "elastic_foundation"

    def __post_init__(self) -> None:
        if isinstance(self.stiffness, Real) and self.stiffness < 0.0:
            raise ValueError("Foundation stiffness must be non-negative.")
        if self.location is None or not hasattr(self.location, "measure"):
            raise ValueError("ElasticFoundation requires a boundary region.")
        selected = str(self.mode).lower().replace("-", "_")
        if selected not in {"isotropic", "normal"}:
            raise ValueError("Foundation mode must be isotropic or normal.")
        object.__setattr__(self, "mode", selected)

    def operator(self, displacement):
        trial, test = displacement.trial, displacement.test
        expression = self._expression(trial, test)
        return OperatorForm(
            name=f"K_{self.name}",
            expression=expression,
            kind="elastic_foundation_operator",
            role="matrix",
            family="mechanical_boundary",
            metadata={"mode": self.mode},
        )

    def _expression(self, trial, test):
        """Return the reviewed weak operator for arbitrary trial/test fields."""

        if self.mode == "normal":
            normal = self.normal or ufl.FacetNormal(self.location.domain)
            expression = (
                self.stiffness
                * ufl.dot(trial, normal)
                * ufl.dot(test, normal)
                * self.location.measure
            )
        else:
            expression = (
                self.stiffness
                * ufl.inner(trial, test)
                * self.location.measure
            )
        return expression

    def capabilities(self):
        """Declare the foundation's reaction and stored-energy semantics."""

        from agentfem.constraints import ConstraintCapabilities

        return ConstraintCapabilities(
            kind="weak_constraint",
            enforcement="weak_elastic_boundary_operator",
            analyses=("linear_static",),
            procedures=("direct_linear_solve",),
            strict=True,
            supports_parallel=True,
            reaction_evidence="provider_dual_required",
            work_evidence="internal_energy_operator",
        )

    def dual_evidence(self, problem):
        """Return the foundation reaction without counting its energy twice."""

        solution_getter = getattr(problem, "_solution", None)
        if not callable(solution_getter) or getattr(problem, "last_solve_info", None) is None:
            raise RuntimeError(
                "Elastic-foundation dual evidence requires a converged linear "
                "system problem."
            )

        from dolfinx import fem
        import dolfinx.fem.petsc as fem_petsc
        from petsc4py import PETSc

        from agentfem.constraints import constraint_dual

        solution = solution_getter()
        trial = ufl.TrialFunction(solution.function_space)
        test = ufl.TestFunction(solution.function_space)
        expression = self._expression(trial, test)
        internal = fem_petsc.assemble_vector(
            fem.form(ufl.action(expression, solution))
        )
        try:
            internal.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            distribution = fem.Function(
                solution.function_space,
                name=f"{self.name}_reaction",
            )
            owned = int(
                solution.function_space.dofmap.index_map.size_local
                * solution.function_space.dofmap.index_map_bs
            )
            distribution.x.array[:owned] = -np.asarray(internal.array_r[:owned])
            distribution.x.scatter_forward()
        finally:
            internal.destroy()

        values = np.asarray(distribution.x.array[:owned], dtype=float)
        shape = tuple(getattr(solution, "ufl_shape", ()))
        if len(shape) != 1:
            raise NotImplementedError(
                "Elastic-foundation reaction evidence requires a vector solid field."
            )
        components = int(shape[0])
        if values.size % components:
            raise RuntimeError(
                "Elastic-foundation reaction storage is incompatible with the field."
            )
        local_resultant = np.sum(values.reshape((-1, components)), axis=0)
        resultant = np.empty(components, dtype=float)
        comm = solution.function_space.mesh.comm
        comm.Allreduce(local_resultant, resultant, op=MPI.SUM)
        internal_action = ufl.action(expression, solution)
        stored_local = fem.assemble_scalar(
            fem.form(0.5 * ufl.action(internal_action, solution))
        )
        stored_energy = float(comm.allreduce(float(stored_local), op=MPI.SUM))
        diagnostics = {
            "status": "complete",
            "reaction_distribution": distribution.name,
            "distribution_location": "nodes",
            "resultant_norm": float(np.linalg.norm(resultant)),
            "stored_energy": stored_energy,
            "stored_energy_accounting": "included_in_system_strain_energy",
            "comm_size": int(comm.size),
        }
        return constraint_dual(
            self,
            force=resultant,
            coordinate=None,
            resultant=resultant,
            distribution=distribution,
            diagnostics=diagnostics,
            role="weak_constraint",
            source="elastic_foundation_weak_operator_action",
            complete=True,
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "elastic_foundation",
            "location": getattr(self.location, "name", None),
            "mode": self.mode,
            "stiffness": describe_value(self.stiffness),
        }


def elastic_foundation(
    *, on=None, location=None, stiffness, mode: str = "isotropic", normal=None,
    name: str = "elastic_foundation",
) -> ElasticFoundation:
    selected = location if location is not None else on
    if on is not None and location is not None:
        raise ValueError("Pass either on=... or location=..., not both.")
    if selected is None or not hasattr(selected, "domain"):
        raise ValueError("elastic_foundation requires a boundary region.")
    return ElasticFoundation(
        stiffness=constants.constant(selected.domain, stiffness),
        location=selected,
        mode=mode,
        normal=normal,
        name=name,
    )


__all__ = ["ElasticFoundation", "elastic_foundation"]

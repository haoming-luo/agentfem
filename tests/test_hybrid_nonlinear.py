# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from agentfem import mesh, operators, solvers


class _CubicDofEnergy:
    def __init__(self, coefficient: float) -> None:
        self.coefficient = float(coefficient)

    def energy(self, state) -> float:
        owned = state.x.petsc_vec.array_r
        return 0.25 * self.coefficient * float(np.sum(owned**4))

    def residual(self, state):
        result = state.x.petsc_vec.duplicate()
        result.array[:] = self.coefficient * state.x.petsc_vec.array_r**3
        return result

    def tangent_action(self, state, increment):
        result = state.x.petsc_vec.duplicate()
        result.array[:] = (
            3.0
            * self.coefficient
            * state.x.petsc_vec.array_r**2
            * increment.x.petsc_vec.array_r
        )
        return result

    def as_dict(self):
        return {"kind": "test_cubic_dof_energy"}


def _residual_norm(problem) -> float:
    problem._assemble_residual(
        problem.solver,
        problem.solution.x.petsc_vec,
        problem.residual_vector,
    )
    return float(problem.residual_vector.norm())


def test_hybrid_newton_solves_local_form_plus_matrix_free_energy():
    domain = dolfinx_mesh.create_unit_interval(MPI.COMM_SELF, 8)
    space = fem.functionspace(domain, ("Lagrange", 1))
    solution = fem.Function(space, name="U")
    solution.interpolate(lambda x: 0.02 + 0.01 * x[0])
    target = fem.Constant(domain, PETSc.ScalarType(0.15))
    test = ufl.TestFunction(space)
    trial = ufl.TrialFunction(space)
    residual = (solution - target) * test * ufl.dx
    jacobian = trial * test * ufl.dx
    contribution = _CubicDofEnergy(0.02)
    options = solvers.NonlinearSolverOptions(
        ksp_type="gmres",
        pc_type="lu",
        rtol=1.0e-11,
        atol=1.0e-12,
        max_it=30,
    )

    with solvers._prepare_hybrid_nonlinear_problem(
        residual,
        solution,
        (contribution,),
        jacobian_form=jacobian,
        options=options,
        petsc_options_prefix="agentfem_test_hybrid_",
    ) as problem:
        solved, info = problem.solve()
        assert info.converged
        assert _residual_norm(problem) < 2.0e-11
        assert np.all(solved.x.array > 0.0)
        assert np.all(solved.x.array < 0.15)
        assert problem.summary()["status"] == "internal_promotion_gate"


def test_hybrid_newton_applies_essential_boundary_once():
    domain = dolfinx_mesh.create_unit_interval(MPI.COMM_SELF, 8)
    space = fem.functionspace(domain, ("Lagrange", 1))
    solution = fem.Function(space, name="U")
    target = fem.Constant(domain, PETSc.ScalarType(0.15))
    test = ufl.TestFunction(space)
    trial = ufl.TrialFunction(space)
    residual = (solution - target) * test * ufl.dx
    jacobian = trial * test * ufl.dx
    boundary = dolfinx_mesh.locate_entities_boundary(
        domain,
        0,
        lambda x: np.isclose(x[0], 0.0),
    )
    dofs = fem.locate_dofs_topological(space, 0, boundary)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, space)
    options = solvers.NonlinearSolverOptions(
        ksp_type="gmres",
        pc_type="lu",
        rtol=1.0e-11,
        atol=1.0e-12,
    )

    with solvers._prepare_hybrid_nonlinear_problem(
        residual,
        solution,
        (_CubicDofEnergy(0.02),),
        bcs=(bc,),
        jacobian_form=jacobian,
        options=options,
        petsc_options_prefix="agentfem_test_hybrid_bc_",
    ) as problem:
        solved, info = problem.solve()
        assert info.converged
        assert _residual_norm(problem) < 2.0e-11
        assert solved.x.array[dofs[0]] == pytest.approx(0.0, abs=1.0e-14)
        assert problem.constrained_local_dofs == (int(dofs[0]),)


def test_hybrid_newton_rejects_preonly_with_nonlocal_tangent():
    domain = dolfinx_mesh.create_unit_interval(MPI.COMM_SELF, 2)
    space = fem.functionspace(domain, ("Lagrange", 1))
    solution = fem.Function(space)
    test = ufl.TestFunction(space)
    residual = solution * test * ufl.dx
    with pytest.raises(ValueError, match="iterative KSP"):
        solvers._prepare_hybrid_nonlinear_problem(
            residual,
            solution,
            (_CubicDofEnergy(0.02),),
            options=solvers.NonlinearSolverOptions(),
        )


def test_hybrid_newton_converges_with_displacement_fiber_bending():
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        4,
        3,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    displacement = fem.Function(space, name="U")
    transfer = operators.cell_average_gradient(space)
    angle = 0.35
    kinematics = operators.convected_cell_fiber(
        transfer,
        reference_tangents=np.array(
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
        ),
        reference_tangent_coordinates=np.array(
            (np.cos(angle), np.sin(angle))
        ),
    )
    bending = operators.displacement_fiber_bending(
        kinematics,
        mesh.cell_gradient_operator(domain, rings=2),
        cell_weights=mesh.owned_cell_measures(domain),
        in_plane_stiffness=0.03,
        normal_stiffness=0.05,
    )
    test = ufl.TestFunction(space)
    trial = ufl.TrialFunction(space)
    load = fem.Constant(domain, np.array((0.0, 0.0, 2.0e-3)))
    residual = (
        ufl.inner(displacement, test) * ufl.dx
        - ufl.inner(load, test) * ufl.dx
    )
    jacobian = ufl.inner(trial, test) * ufl.dx
    left_facets = dolfinx_mesh.locate_entities_boundary(
        domain,
        1,
        lambda x: np.isclose(x[0], 0.0),
    )
    left_dofs = fem.locate_dofs_topological(space, 1, left_facets)
    fixed = fem.dirichletbc(np.zeros(3), left_dofs, space)
    options = solvers.NonlinearSolverOptions(
        ksp_type="gmres",
        pc_type="lu",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_it=30,
    )

    with solvers._prepare_hybrid_nonlinear_problem(
        residual,
        displacement,
        (bending,),
        bcs=(fixed,),
        jacobian_form=jacobian,
        options=options,
        petsc_options_prefix="agentfem_test_hybrid_bending_",
    ) as problem:
        solved, info = problem.solve()
        assert info.converged
        assert _residual_norm(problem) < 2.0e-10
        assert bending.energy(solved) > 0.0
        assert np.max(np.abs(solved.x.array)) < 1.0e-2
        fixed_scalar_dofs, owned = fixed.dof_indices()
        np.testing.assert_allclose(
            solved.x.array[fixed_scalar_dofs[:owned]],
            0.0,
            atol=2.0e-13,
        )

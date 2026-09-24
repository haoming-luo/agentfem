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

from agentfem import (
    constitutive,
    materials,
    mechanics,
    mesh,
    operators,
    procedures,
    solvers,
    steps,
)
from agentfem._nonlinear_problems import IncrementalNonlinearVariationalProblem


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


class _NoOpValuePath:
    def __init__(self) -> None:
        self.factors = []

    def update(self, factor: float) -> None:
        self.factors.append(float(factor))


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
        physical = problem.assemble_physical_residual()
        reaction = float(physical.array_r[dofs[0]])
        assert abs(reaction) > 1.0e-5
        assert problem.free_residual_norm() < 2.0e-11
        physical.destroy()


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


def test_hybrid_newton_restores_solution_after_failed_attempt():
    domain = dolfinx_mesh.create_unit_interval(MPI.COMM_SELF, 8)
    space = fem.functionspace(domain, ("Lagrange", 1))
    solution = fem.Function(space, name="U")
    solution.interpolate(lambda x: 0.2 + 0.05 * x[0])
    accepted = solution.x.array.copy()
    target = fem.Constant(domain, PETSc.ScalarType(1.0))
    test = ufl.TestFunction(space)
    trial = ufl.TrialFunction(space)
    residual = (solution - target) * test * ufl.dx
    jacobian = trial * test * ufl.dx
    options = solvers.NonlinearSolverOptions(
        ksp_type="gmres",
        pc_type="lu",
        rtol=1.0e-14,
        atol=1.0e-14,
        max_it=1,
        error_if_not_converged=True,
    )

    with solvers._prepare_hybrid_nonlinear_problem(
        residual,
        solution,
        (_CubicDofEnergy(20.0),),
        jacobian_form=jacobian,
        options=options,
        petsc_options_prefix="agentfem_test_hybrid_failure_",
    ) as problem:
        with pytest.raises((RuntimeError, PETSc.Error)):
            problem.solve()
        np.testing.assert_allclose(solution.x.array, accepted, atol=0.0)
        assert problem.attempt_count == 1
        assert problem.accepted_solve_count == 0
        assert problem.summary()["failure_state"].startswith("restored")


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


def test_hybrid_bending_total_energy_residual_and_tangent_are_consistent():
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        4,
        3,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    displacement = fem.Function(space, name="U")
    displacement.interpolate(
        lambda x: np.vstack(
            (
                0.02 * x[0] ** 2,
                -0.015 * x[0] * x[1],
                0.025 * x[1] ** 2 + 0.01 * x[0],
            )
        )
    )
    increment = fem.Function(space, name="DU")
    increment.interpolate(
        lambda x: np.vstack(
            (
                -0.03 * x[0] + 0.01 * x[1],
                0.02 * x[0] ** 2,
                -0.015 * x[0] * x[1] + 0.005 * x[1],
            )
        )
    )
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
    load = fem.Constant(domain, np.array((4.0e-4, -3.0e-4, 8.0e-4)))
    local_energy_expression = (
        0.5 * ufl.inner(displacement, displacement) * ufl.dx
        - ufl.inner(load, displacement) * ufl.dx
    )
    residual = (
        ufl.inner(displacement, test) * ufl.dx
        - ufl.inner(load, test) * ufl.dx
    )
    jacobian = ufl.inner(trial, test) * ufl.dx
    options = solvers.NonlinearSolverOptions(
        ksp_type="gmres",
        pc_type="lu",
    )

    with solvers._prepare_hybrid_nonlinear_problem(
        residual,
        displacement,
        (bending,),
        jacobian_form=jacobian,
        options=options,
        petsc_options_prefix="agentfem_test_hybrid_consistency_",
    ) as problem:
        base = displacement.x.array.copy()
        epsilon = 1.0e-6

        def total_energy(values):
            displacement.x.array[:] = values
            displacement.x.scatter_forward()
            return fem.assemble_scalar(fem.form(local_energy_expression)) + bending.energy(
                displacement
            )

        plus_energy = total_energy(base + epsilon * increment.x.array)
        minus_energy = total_energy(base - epsilon * increment.x.array)
        displacement.x.array[:] = base
        displacement.x.scatter_forward()
        problem._assemble_residual(
            problem.solver,
            displacement.x.petsc_vec,
            problem.residual_vector,
        )
        exact_work = np.vdot(
            increment.x.petsc_vec.array_r,
            problem.residual_vector.array_r,
        )
        assert (plus_energy - minus_energy) / (2.0 * epsilon) == pytest.approx(
            exact_work,
            rel=4.0e-7,
            abs=4.0e-9,
        )

        plus = displacement.x.petsc_vec.duplicate()
        minus = displacement.x.petsc_vec.duplicate()
        displacement.x.petsc_vec.copy(plus)
        displacement.x.petsc_vec.copy(minus)
        plus.axpy(epsilon, increment.x.petsc_vec)
        minus.axpy(-epsilon, increment.x.petsc_vec)
        plus_residual = problem.residual_vector.duplicate()
        minus_residual = problem.residual_vector.duplicate()
        problem._assemble_residual(problem.solver, plus, plus_residual)
        problem._assemble_residual(problem.solver, minus, minus_residual)
        displacement.x.array[:] = base
        displacement.x.scatter_forward()
        problem._assemble_jacobian(
            problem.solver,
            displacement.x.petsc_vec,
            problem.additive.operator,
            problem.local_matrix,
        )
        exact_tangent = problem.local_matrix.createVecLeft()
        problem.additive.operator.mult(
            increment.x.petsc_vec,
            exact_tangent,
        )
        finite_difference = (
            plus_residual.array_r - minus_residual.array_r
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            exact_tangent.array_r,
            finite_difference,
            rtol=2.0e-6,
            atol=8.0e-8,
        )
        for vector in (
            plus,
            minus,
            plus_residual,
            minus_residual,
            exact_tangent,
        ):
            vector.destroy()


def test_hybrid_newton_uses_standard_load_path_progress_and_result_lifecycle():
    domain = dolfinx_mesh.create_unit_interval(MPI.COMM_SELF, 8)
    space = fem.functionspace(domain, ("Lagrange", 1))
    solution = fem.Function(space, name="Displacement")
    factor = fem.Constant(domain, PETSc.ScalarType(0.0))
    target = fem.Constant(domain, PETSc.ScalarType(0.12))
    test = ufl.TestFunction(space)
    trial = ufl.TrialFunction(space)
    residual = (solution - factor * target) * test * ufl.dx
    jacobian = trial * test * ufl.dx
    boundary = dolfinx_mesh.locate_entities_boundary(
        domain,
        0,
        lambda x: np.isclose(x[0], 0.0),
    )
    dofs = fem.locate_dofs_topological(space, 0, boundary)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, space)
    value_path = _NoOpValuePath()
    options = solvers.NonlinearSolverOptions(
        ksp_type="gmres",
        pc_type="lu",
        rtol=1.0e-11,
        atol=1.0e-12,
        error_if_not_converged=False,
    )

    with solvers._prepare_hybrid_nonlinear_problem(
        residual,
        solution,
        (_CubicDofEnergy(0.02),),
        bcs=(bc,),
        jacobian_form=jacobian,
        options=options,
        petsc_options_prefix="agentfem_test_hybrid_lifecycle_",
    ) as prepared:

        def acceptance_check():
            physical = prepared.assemble_physical_residual()
            try:
                reaction = float(physical.array_r[dofs[0]])
            finally:
                physical.destroy()
            free_residual_norm = prepared.free_residual_norm()
            return {
                "accepted": free_residual_norm < 2.0e-10,
                "fixed_reaction": reaction,
                "free_residual_norm": free_residual_norm,
            }

        problem = IncrementalNonlinearVariationalProblem(
            residual_form=residual,
            solution=solution,
            factor=factor,
            value_path=value_path,
            acceptance_check=acceptance_check,
            bcs=[bc],
            jacobian_form=jacobian,
            incrementation=steps.fixed(4),
            solver_options=options,
            progress=False,
            name="hybrid_lifecycle_gate",
            procedure=procedures.nonlinear_static(),
            _attempt_solver=prepared.solve,
            _attempt_backend="assembled_local_plus_matrix_free_nonlocal",
        )
        result = problem.solve_result()

    assert result.status == "completed"
    assert result.metadata["solve"]["accepted_increment_count"] == 4
    assert result.metadata["solve"]["attempt_count"] == 4
    assert result.metadata["problem"]["attempt_backend"].endswith("nonlocal")
    assert result.metadata["execution"]["event_count"] == 10
    assert value_path.factors == pytest.approx((0.0, 0.25, 0.5, 0.75, 1.0))
    for increment in result.metadata["solve"]["increments"]:
        assert increment["checks"]["free_residual_norm"] < 2.0e-10
        assert abs(increment["checks"]["fixed_reaction"]) > 1.0e-6


def test_injected_nonlinear_attempt_uses_standard_cutback_and_rollback():
    domain = dolfinx_mesh.create_unit_interval(MPI.COMM_SELF, 2)
    space = fem.functionspace(domain, ("Lagrange", 1))
    solution = fem.Function(space, name="Displacement")
    factor = fem.Constant(domain, PETSc.ScalarType(0.0))
    value_path = _NoOpValuePath()
    calls = 0

    def solve_attempt():
        nonlocal calls
        calls += 1
        if calls == 1:
            solution.x.array[:] = 99.0
            return solution, solvers.NonlinearSolveInfo(-3, 2, 1.0)
        solution.x.array[:] = float(factor.value)
        solution.x.scatter_forward()
        return solution, solvers.NonlinearSolveInfo(2, 1, 0.0)

    problem = IncrementalNonlinearVariationalProblem(
        residual_form=None,
        solution=solution,
        factor=factor,
        value_path=value_path,
        bcs=[],
        incrementation=steps.automatic(
            initial=0.5,
            minimum=0.125,
            maximum=0.5,
            max_cutbacks=3,
            cutback_factor=0.5,
        ),
        progress=False,
        name="injected_cutback_gate",
        procedure=procedures.nonlinear_static(),
        _attempt_solver=solve_attempt,
        _attempt_backend="test_injected",
    )
    solved = problem.solve()

    assert np.all(solved.x.array == pytest.approx(1.0))
    assert problem.last_solve_info.converged
    assert len(problem.last_solve_info.attempts) > len(
        problem.last_solve_info.increments
    )
    assert problem.last_solve_info.attempts[0].converged is False
    assert problem.last_solve_info.increments[0].start_load_factor == 0.0
    assert problem.last_solve_info.increments[0].load_factor == pytest.approx(0.25)
    assert any(
        event.kind == "increment_cutback" for event in problem.execution_events
    )
    assert 99.0 not in solution.x.array


def test_embedded_fabric_membrane_and_bending_build_separate_preconditioner():
    domain = dolfinx_mesh.create_unit_square(
        MPI.COMM_SELF,
        4,
        2,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    displacement = fem.Function(space, name="Displacement")
    factor = fem.Constant(domain, PETSc.ScalarType(0.0))
    test = ufl.TestFunction(space)
    trial = ufl.TrialFunction(space)
    reference_tangents = ufl.as_matrix(
        ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
    )
    current_tangents = reference_tangents + ufl.grad(displacement)
    current_normal = ufl.cross(
        current_tangents[:, 0], current_tangents[:, 1]
    )
    current_normal /= ufl.sqrt(ufl.inner(current_normal, current_normal))
    current_warp = current_tangents[:, 0] / ufl.sqrt(
        ufl.inner(current_tangents[:, 0], current_tangents[:, 0])
    )
    current_weft = current_tangents[:, 1] / ufl.sqrt(
        ufl.inner(current_tangents[:, 1], current_tangents[:, 1])
    )
    zero_gradient = ufl.as_matrix(np.zeros((3, 2)).tolist())
    kinematics = mechanics.fibrous_shell_kinematics_ufl(
        reference_tangents,
        current_tangents,
        current_normal,
        reference_fibers=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        current_fibers=(current_warp, current_weft),
        current_fiber_gradients=(zero_gradient, zero_gradient),
    )
    axial = constitutive.tabulated_response(
        (-0.1, 0.0, 0.1),
        (-5.0, 0.0, 5.0),
        extrapolation="linear",
    )
    shear = constitutive.tabulated_response(
        (0.0, 0.2),
        (0.0, 1.0),
        symmetry="odd",
        extrapolation="linear",
    )
    membrane = constitutive.decoupled_fabric_surface(
        frame=materials.fiber_frame(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        warp_tension=axial,
        weft_tension=axial,
        shear=shear,
        bending_stiffness=np.zeros((3, 3)),
        tension_only=False,
    )
    shell_law = constitutive.decoupled_fibrous_shell(
        membrane,
        transverse_shear_stiffness=np.zeros((2, 2)),
        in_plane_bending_stiffness=np.zeros((2, 2)),
        normal_bending_stiffness=np.diag((0.02, 0.0)),
    )
    local_response = shell_law.generalized_expressions_ufl(
        kinematics.generalized_strain
    )
    load = fem.Constant(domain, PETSc.ScalarType(-2.0e-3))
    foundation = fem.Constant(domain, PETSc.ScalarType(0.1))
    # The tabulated fabric law contains nested conditional expressions.  Leave
    # production quadrature policy to the caller, but keep this structural
    # preconditioner test bounded and deterministic: its contract is matrix
    # ownership, not high-order constitutive integration.
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})
    potential = (
        local_response.energy_channels["membrane"] * dx
        + 0.5 * foundation * displacement[2] ** 2 * dx
        - factor * load * displacement[2] * dx
    )
    residual = ufl.derivative(potential, displacement, test)
    jacobian = ufl.derivative(residual, displacement, trial)

    # A rotation-free curvature operator needs both displacement and its
    # boundary slope for a true clamp.  Until the public boundary-moment/
    # rotation contract exists, two nodal lines make that promotion gate
    # explicit instead of silently treating u=0 as a clamped shell edge.
    fixed_dofs = fem.locate_dofs_geometrical(
        space,
        lambda x: x[0] <= 0.25 + 1.0e-12,
    )
    fixed = fem.dirichletbc(
        np.zeros(3, dtype=PETSc.ScalarType),
        fixed_dofs,
        space,
    )
    transfer = operators.cell_average_gradient(space)
    convected = operators.convected_cell_fiber(
        transfer,
        reference_tangents=np.array(
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
        ),
        reference_tangent_coordinates=np.array((1.0, 0.0)),
    )
    bending = operators.displacement_fiber_bending(
        convected,
        mesh.cell_gradient_operator(domain, rings=2),
        cell_weights=mesh.owned_cell_measures(domain),
        in_plane_stiffness=0.0,
        normal_stiffness=0.02,
    )
    options = solvers.NonlinearSolverOptions(
        ksp_type="gmres",
        pc_type="lu",
        rtol=1.0e-10,
        atol=1.0e-11,
        max_it=30,
        error_if_not_converged=False,
    )

    with solvers._prepare_hybrid_nonlinear_problem(
        residual,
        displacement,
        (bending,),
        bcs=(fixed,),
        jacobian_form=jacobian,
        preconditioner_form=(
            jacobian
            + 0.02
            * ufl.inner(ufl.grad(trial[2]), ufl.grad(test[2]))
            * dx
        ),
        options=options,
        petsc_options_prefix="agentfem_test_fabric_strip_",
    ) as prepared:
        summary = prepared.summary()
        assert summary["preconditioner"].startswith("independent")
        assert prepared.additive.preconditioner is prepared.preconditioner_matrix
        assert prepared.preconditioner_matrix is not prepared.local_matrix
        assert prepared.additive.operator.getSizes() == prepared.local_matrix.getSizes()

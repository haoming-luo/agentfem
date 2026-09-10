from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import (
    constitutive,
    constraints,
    fields,
    mesh,
    models,
    operators,
    results,
    solvers,
    studies,
)
from agentfem.mesh import abaqus
from portable_affine_j2_periodic_driver import _step as _finite_strain_j2_step


pytest.importorskip("dolfinx_mpc")


def test_rectangular_periodic_mpc_is_public_strict_and_distributed():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 3, 2)
    space = fem.functionspace(domain, ("Lagrange", 1))

    periodicity = constraints.rectangular_periodic_mpc(space)
    global_slaves = domain.comm.allreduce(
        periodicity.backend.num_local_slaves,
        op=MPI.SUM,
    )

    assert global_slaves > 0
    summary = periodicity.summary()
    assert summary.pop("tolerance") > 0.0
    diagnostics = summary.pop("diagnostics")
    assert summary == {
        "name": "rectangular_periodic_mpc",
        "kind": "periodic_constraint",
        "method": "dolfinx_mpc",
        "enforcement": "exact_multi_point_constraint",
        "lower": (0.0, 0.0),
        "upper": (1.0, 1.0),
        "axes": (0, 1),
        "strict": True,
        "supports_parallel": True,
    }
    assert diagnostics["status"] == "valid"
    assert diagnostics["global_slave_dofs"] == global_slaves
    assert diagnostics["global_master_relations"] == global_slaves
    assert diagnostics["unmatched_slave_dofs"] == 0
    assert diagnostics["multiply_matched_slave_dofs"] == 0
    assert diagnostics["nonunit_coefficients_detected"] is False
    assert diagnostics["comm_size"] == MPI.COMM_WORLD.size
    assert diagnostics["reaction_distribution"] == "available_after_converged_solve"
    assert diagnostics["macroscopic_work"] == "available_after_converged_solve"
    assert periodicity.diagnostics() == diagnostics


def test_prepared_mpc_linear_problem_reuses_lifecycle_and_updates_load():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 4, 3)
    space = fem.functionspace(domain, ("Lagrange", 1))
    solution = fem.Function(space, name="temperature")
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    source = fem.Constant(domain, 1.0)
    lhs = (ufl.inner(ufl.grad(trial), ufl.grad(test)) + trial * test) * ufl.dx
    rhs = source * test * ufl.dx
    periodicity = constraints.rectangular_periodic_mpc(space)
    problem = solvers.prepare_mpc_linear_problem(
        lhs,
        rhs,
        solution,
        periodicity,
        options=solvers.direct_solver(package="mumps"),
        petsc_options_prefix="agentfem_test_periodic_mass_",
    )

    with problem:
        first = problem.solve().x.array.copy()
        source.value = 2.0
        second = problem.solve().x.array.copy()

        assert problem.last_solve_info is not None
        assert problem.last_solve_info.converged
        assert problem.solve_count == 2
        assert np.max(np.abs(first - 1.0)) < 1.0e-10
        assert np.max(np.abs(second - 2.0)) < 1.0e-10
        summary = problem.summary()
        assert summary["matrix_allocation_reused"] is True
        assert summary["matrix_values_reassembled"] is True
        assert summary["constraint"]["diagnostics"]["status"] == "valid"

    assert problem.closed
    assert problem.summary() == summary


def test_prepared_mpc_linear_problem_transfers_vector_field_layout():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 4, 3)
    space = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    solution = fem.Function(space, name="displacement")
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    source = fem.Constant(domain, np.asarray((1.0, -2.0)))
    lhs = (ufl.inner(ufl.grad(trial), ufl.grad(test)) + ufl.inner(trial, test)) * ufl.dx
    rhs = ufl.inner(source, test) * ufl.dx
    periodicity = constraints.rectangular_periodic_mpc(space)

    solved = solvers.solve_mpc_linear_problem(
        lhs,
        rhs,
        solution,
        periodicity,
        options=solvers.direct_solver(package="mumps"),
        petsc_options_prefix="agentfem_test_periodic_vector_",
    )
    values = solved.x.array.reshape(-1, 2)

    assert np.max(np.abs(values[:, 0] - 1.0)) < 1.0e-10
    assert np.max(np.abs(values[:, 1] + 2.0)) < 1.0e-10


def test_model_step_lowers_exact_mpc_and_publishes_dual_evidence():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 4, 3)
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
        name="periodic_vector_step",
    )
    displacement = model.field(fields.displacement(domain))
    source = fem.Function(displacement.space, name="prescribed_source")
    source.interpolate(
        lambda x: np.vstack((np.ones_like(x[0]), -2.0 * np.ones_like(x[0])))
    )
    periodicity = constraints.rectangular_periodic_mpc(displacement)

    step = model.step(
        target=displacement,
        K=operators.mass_operator(displacement),
        F=operators.mass_action_vector(source, displacement),
        constraints=periodicity,
        solver_options=solvers.direct_solver(package="mumps"),
    )
    simulation = step.solve_result()
    values = displacement.value.x.array.reshape(-1, 2)

    assert np.max(np.abs(values[:, 0] - 1.0)) < 1.0e-10
    assert np.max(np.abs(values[:, 1] + 2.0)) < 1.0e-10
    problem_summary = simulation.metadata["step"]["problem"]
    assert problem_summary["constraint_provider"]["method"] == "dolfinx_mpc"
    assert problem_summary["last_solve"]["converged"] is True
    contract = simulation.metadata["constraint_balance_contract"]
    assert contract["force_balance_available"] is True
    assert contract["work_balance_available"] is True
    dual = simulation.metadata["constraint_duals"][0]
    assert dual["source"] == "exact_mpc_slave_residual_multiplier_recovery"
    assert dual["force_complete"] is True
    assert dual["work_complete"] is True
    assert dual["diagnostics"]["status"] == "complete"
    assert dual["diagnostics"]["relation_count"] > 0
    assert dual["diagnostics"]["constraint_gap_linf_norm"] < 1.0e-12
    assert abs(dual["diagnostics"]["constraint_virtual_work"]) < 1.0e-12
    assert dual["diagnostics"]["resultant_norm"] < 1.0e-12
    reaction_name = dual["distribution"]["name"]
    assert reaction_name in simulation.fields
    assert abs(simulation.metadata["static_work"]["energy_balance_error"]) < 1.0e-12


def test_exact_mpc_recovers_nonzero_multiplier_distribution():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 6, 5)
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
        name="periodic_dual_distribution",
    )
    displacement = model.field(fields.displacement(domain))
    source = fem.Function(displacement.space, name="nonperiodic_source")
    source.interpolate(
        lambda x: np.vstack((x[0] + 0.25 * x[1], x[1] - 0.5 * x[0]))
    )
    periodicity = constraints.rectangular_periodic_mpc(displacement)
    stiffness = operators.combine(
        operators.diffusion_operator(displacement, conductivity=1.0),
        operators.mass_operator(displacement, density=0.5),
        name="regularized_periodic_stiffness",
    )

    simulation = model.step(
        target=displacement,
        K=stiffness,
        F=operators.mass_action_vector(source, displacement),
        constraints=periodicity,
        solver_options=solvers.direct_solver(package="mumps"),
    ).solve_result()

    dual = simulation.metadata["constraint_duals"][0]
    diagnostics = dual["diagnostics"]
    assert diagnostics["multiplier_l2_norm"] > 1.0e-5
    assert diagnostics["multiplier_linf_norm"] > 1.0e-6
    assert diagnostics["constraint_gap_linf_norm"] < 1.0e-11
    assert abs(diagnostics["constraint_virtual_work"]) < 1.0e-11
    assert diagnostics["resultant_norm"] < 1.0e-11
    distribution = simulation.fields[dual["distribution"]["name"]].field
    assert np.max(np.abs(distribution.x.array)) > 1.0e-6


@pytest.mark.parametrize(
    ("mode", "stiffness"),
    (
        ("isotropic", 5.0e3),
        ("matrix", ((5.0e3, 5.0e2), (5.0e2, 2.5e3))),
    ),
)
def test_elastic_foundation_owns_reaction_without_double_counting_energy(
    mode,
    stiffness,
):
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 8, 5)
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
        name="weak_foundation_balance",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        constitutive.elasticity.isotropic_elastic(
            young=2.0e3,
            poisson=0.25,
            density=1.0,
        )
    )
    left = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="foundation_face",
    )
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="loaded_face",
    )
    model.elastic_foundation(on=left, stiffness=stiffness, mode=mode)
    model.traction((10.0, 0.0), on=right)

    simulation = model.step(target=displacement).solve_result()

    contract = simulation.metadata["constraint_balance_contract"]
    assert contract["force_balance_available"] is True
    assert contract["work_balance_available"] is True
    dual = simulation.metadata["constraint_duals"][0]
    assert dual["role"] == "weak_constraint"
    assert dual["force_complete"] is True
    assert dual["work_complete"] is False
    assert dual["diagnostics"]["stored_energy"] > 0.0
    assert dual["diagnostics"]["stored_energy_accounting"] == (
        "included_in_system_strain_energy"
    )
    np.testing.assert_allclose(dual["resultant"], [-10.0, 0.0], atol=1.0e-9)
    assert simulation.quantities["relative_force_balance_error"].value < 1.0e-10
    assert abs(simulation.metadata["static_work"]["energy_balance_error"]) < 1.0e-10
    assert dual["distribution"]["name"] in simulation.fields


def test_steady_heat_step_uses_the_same_exact_mpc_lowering():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 4, 3)
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        mesh=domain,
        name="periodic_scalar_step",
    )
    temperature = model.field(fields.temperature(domain))
    source = fem.Function(temperature.space, name="prescribed_temperature")
    source.x.array[:] = 3.0
    source.x.scatter_forward()
    periodicity = constraints.rectangular_periodic_mpc(temperature)

    simulation = model.step(
        target=temperature,
        K=operators.mass_operator(temperature),
        F=operators.mass_action_vector(source, temperature),
        constraints=periodicity,
        solver_options=solvers.direct_solver(package="mumps"),
    ).solve_result()

    assert np.max(np.abs(temperature.value.x.array - 3.0)) < 1.0e-10
    provider = simulation.metadata["step"]["problem"]["constraint_provider"]
    assert provider["enforcement"] == "exact_multi_point_constraint"


def test_transient_heat_step_uses_exact_mpc_on_every_increment():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 4, 3)
    model = models.create(
        study=studies.transient_heat_transfer(dimension=2),
        mesh=domain,
        name="periodic_transient_scalar_step",
    )
    temperature = model.field(fields.temperature(domain, value=5.0))
    model.material(
        constitutive.thermoelastic(
            young=1.0,
            poisson=0.25,
            density=2.0,
            thermal_expansion=0.0,
            conductivity=3.0,
            specific_heat=4.0,
            reference_temperature=1.0,
        )
    )
    periodicity = constraints.rectangular_periodic_mpc(temperature)

    step = model.step(
        target=temperature,
        dt=0.1,
        steps=2,
        constraints=periodicity,
        solver_options=solvers.direct_solver(package="mumps"),
        progress=False,
    )
    simulation = step.solve_result()

    assert np.max(np.abs(temperature.value.x.array - 5.0)) < 1.0e-10
    assert step.completed_steps == 2
    problem_summary = simulation.metadata["step"]["problem"]["problem"]
    assert problem_summary["constraint_provider"]["method"] == "dolfinx_mpc"
    assert problem_summary["linear_lifecycle"]["solve_count"] == 2
    assert problem_summary["linear_lifecycle"]["matrix_allocation_reused"] is True


def test_model_step_rejects_multiple_exact_mpc_providers():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    source = fem.Function(displacement.space)
    first = constraints.rectangular_periodic_mpc(displacement, name="periodic_first")
    second = constraints.rectangular_periodic_mpc(displacement, name="periodic_second")

    with pytest.raises(ValueError, match="AFM-CONSTRAINT-MPC-002"):
        model.step(
            target=displacement,
            K=operators.mass_operator(displacement),
            F=operators.mass_action_vector(source, displacement),
            constraints=(first, second),
        )


def test_model_step_rejects_exact_mpc_provider_without_backend():
    class IncompleteExactMPC:
        name = "incomplete_exact_mpc"

        @staticmethod
        def capabilities():
            return constraints.ConstraintCapabilities(
                kind="periodic_constraint",
                enforcement="exact_multi_point_constraint",
            )

    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    model = models.create(
        study=studies.steady_heat_transfer(dimension=2),
        mesh=domain,
    )
    temperature = model.field(fields.temperature(domain))
    source = fem.Function(temperature.space)

    with pytest.raises(TypeError, match="AFM-CONSTRAINT-MPC-001"):
        model.step(
            target=temperature,
            K=operators.mass_operator(temperature),
            F=operators.mass_action_vector(source, temperature),
            constraints=IncompleteExactMPC(),
        )


def test_prepared_mpc_linear_problem_rejects_another_function_space():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    constrained_space = fem.functionspace(domain, ("Lagrange", 1))
    other_space = fem.functionspace(domain, ("Lagrange", 2))
    trial = ufl.TrialFunction(constrained_space)
    test = ufl.TestFunction(constrained_space)
    periodicity = constraints.rectangular_periodic_mpc(constrained_space)

    with pytest.raises(ValueError, match="compatible FunctionSpace"):
        solvers.prepare_mpc_linear_problem(
            trial * test * ufl.dx,
            test * ufl.dx,
            fem.Function(other_space),
            periodicity,
        )


def test_prepared_mpc_linear_problem_rejects_late_overlapping_bc():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 3, 2)
    space = fem.functionspace(domain, ("Lagrange", 1))
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    periodicity = constraints.rectangular_periodic_mpc(space, axes=(0,))
    right_dofs = fem.locate_dofs_geometrical(
        space,
        lambda x: np.isclose(x[0], 1.0),
    )
    right = fem.dirichletbc(fem.Constant(domain, 0.0), right_dofs, space)

    with pytest.raises(ValueError, match="Dirichlet and MPC slave DOFs overlap"):
        solvers.prepare_mpc_linear_problem(
            trial * test * ufl.dx,
            test * ufl.dx,
            fem.Function(space),
            periodicity,
            bcs=(right,),
        )


def test_prepared_mpc_linear_problem_accepts_bc_declared_with_constraint():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 3, 2)
    space = fem.functionspace(domain, ("Lagrange", 1))
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    anchor_dofs = fem.locate_dofs_geometrical(
        space,
        lambda x: np.logical_and(np.isclose(x[0], 1.0), np.isclose(x[1], 0.0)),
    )
    anchor = fem.dirichletbc(fem.Constant(domain, 0.0), anchor_dofs, space)
    periodicity = constraints.rectangular_periodic_mpc(
        space,
        axes=(0,),
        bcs=(anchor,),
    )

    problem = solvers.prepare_mpc_linear_problem(
        (ufl.inner(ufl.grad(trial), ufl.grad(test)) + trial * test) * ufl.dx,
        test * ufl.dx,
        fem.Function(space),
        periodicity,
        bcs=(anchor,),
        options=solvers.direct_solver(package="mumps"),
        petsc_options_prefix="agentfem_test_periodic_with_anchor_",
    )

    assert problem.solve_count == 0
    assert problem.summary()["num_bcs"] == 1
    problem.close()


def test_prepared_mpc_close_is_terminal_idempotent_and_keeps_borrowed_mpc():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 3, 2)
    space = fem.functionspace(domain, ("Lagrange", 1))
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    solution = fem.Function(space)
    periodicity = constraints.rectangular_periodic_mpc(space)
    problem = solvers.prepare_mpc_linear_problem(
        (ufl.inner(ufl.grad(trial), ufl.grad(test)) + trial * test) * ufl.dx,
        test * ufl.dx,
        solution,
        periodicity,
        options=solvers.direct_solver(package="mumps"),
        petsc_options_prefix="agentfem_test_periodic_close_",
    )
    native_problem = problem.problem
    summary = problem.summary()
    expected_solution = solution.x.array.copy()
    global_slaves = domain.comm.allreduce(
        periodicity.backend.num_local_slaves,
        op=MPI.SUM,
    )

    problem.close()
    problem.close()

    assert problem.closed
    assert problem.problem is None
    assert problem.summary() == summary
    assert all(
        getattr(native_problem, name) is None
        for name in ("_solver", "_A", "_b", "_x", "_P_mat")
    )
    # The provider and public field are borrowed rather than owned by the
    # prepared numerical lifecycle.
    assert domain.comm.allreduce(
        periodicity.backend.num_local_slaves,
        op=MPI.SUM,
    ) == global_slaves
    assert solution.function_space is space
    np.testing.assert_array_equal(solution.x.array, expected_solution)
    with pytest.raises(RuntimeError, match="PreparedMPCLinearProblem is closed"):
        problem.solve()

    # Closing the numerical allocation must not consume the provider: a fresh
    # prepared solve can reuse the same exact MPC graph on every rank.
    with solvers.prepare_mpc_linear_problem(
        (ufl.inner(ufl.grad(trial), ufl.grad(test)) + trial * test) * ufl.dx,
        test * ufl.dx,
        solution,
        periodicity,
        options=solvers.direct_solver(package="mumps"),
        petsc_options_prefix="agentfem_test_periodic_reopen_",
    ) as replacement:
        replacement.solve()
        assert np.max(np.abs(solution.x.array - 1.0)) < 1.0e-10
    assert replacement.closed


def test_distributed_abaqus_equation_mapping_and_source_order():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    space = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    displacement = fem.Function(space, name="Displacement")
    nodes = abaqus.AbaqusNodeTable(
        labels=np.arange(1, 10),
        coordinates=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.5],
                [1.0, 0.5],
                [0.5, 0.0],
                [0.5, 1.0],
                [0.5, 0.5],
                [1.0, 1.0],
            ]
        ),
    )
    equations = abaqus.AbaqusEquationSet(
        tuple(
            abaqus.LinearEquation(
                (
                    abaqus.EquationTerm(5, component, -1.0),
                    abaqus.EquationTerm(4, component, 1.0),
                    abaqus.EquationTerm(2, component, 1.0),
                    abaqus.EquationTerm(1, component, -1.0),
                )
            )
            for component in (1, 2)
        )
    )
    target_f = np.diag([1.10, 1.0 / 1.10])
    periodicity = constraints.abaqus_periodic_cell(
        displacement,
        nodes=nodes,
        equations=equations,
        deformation_gradient=target_f,
        anchor_node=1,
        reference_nodes=(2, 3),
    )

    reduction = periodicity.distributed_reduction()
    correction = reduction.correction()
    reduction.validate_prefix_layout(correction)
    owned_scalars = int(
        space.dofmap.index_map.size_local * space.dofmap.index_map_bs
    )
    for bc in reduction.bcs:
        dofs, owned_position = bc.dof_indices()
        assert np.all(dofs[:-1] <= dofs[1:])
        assert np.all(dofs[:owned_position] < owned_scalars)
        assert np.all(dofs[owned_position:] >= owned_scalars)
    periodicity.apply_affine_increment(0.0, 1.0)

    global_slaves = domain.comm.allreduce(
        reduction.mpc.num_local_slaves,
        op=MPI.SUM,
    )
    source_values = abaqus.displacement_in_source_order(
        displacement,
        nodes,
    )

    assert global_slaves == 2
    assert reduction.reduced_size == space.dofmap.index_map.size_global * 2 - 8
    assert periodicity.mismatch() < 1.0e-13
    np.testing.assert_allclose(
        source_values,
        nodes.coordinates @ (target_f - np.eye(2)).T,
        atol=1.0e-13,
    )


def test_distributed_two_phase_affine_j2_requires_and_solves_fluctuation():
    step, _, periodicity = _finite_strain_j2_step(
        MPI.COMM_WORLD,
        two_phase=True,
    )

    step.solve()

    assert step.accepted_load_factor == pytest.approx(1.0)
    assert periodicity.mismatch() < 1.0e-10
    assert all(item.iterations > 0 for item in step.accepted_increments)
    assert all(item.residual_norm < 1.0e-8 for item in step.accepted_increments)


def test_parallel_element_volume_writes_owned_then_ghost_values():
    domain = dolfinx_mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    space = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    displacement = fem.Function(space, name="Displacement")
    material = constitutive.neo_hookean(young=1000.0, poisson=0.3)

    (element_volume,) = results.finite_strain_cell_fields(
        displacement,
        material,
        variables=("EVOL",),
    )
    owned = element_volume.function_space.dofmap.index_map.size_local
    total = domain.comm.allreduce(
        float(np.sum(element_volume.x.array[:owned])),
        op=MPI.SUM,
    )

    assert total == pytest.approx(1.0)

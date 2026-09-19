from __future__ import annotations

from pathlib import Path
import tempfile
import uuid

import numpy as np
import pytest
from dolfinx import fem
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from agentfem import (
    assembly,
    backends,
    constraints,
    fields,
    loads,
    mechanics,
    mesh,
    operators,
    results,
    solvers,
)


def test_distributed_additive_tangent_uses_local_preconditioner():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("distributed additive tangent evidence requires two ranks")
    local = PETSc.Mat().createAIJ(
        size=((2, None), (2, None)),
        nnz=1,
        comm=comm,
    )
    start, end = local.getOwnershipRange()
    diagonal = 3.0 + np.arange(start, end, dtype=float)
    for row, value in zip(range(start, end), diagonal, strict=True):
        local.setValue(row, row, value)
    local.assemble()
    scale = 0.5 + 0.25 * comm.rank

    def nonlocal_action(source, target):
        target.array[:] = scale * source.array_r

    constrained = (0,) if comm.rank == 0 else ()
    additive = backends.create_additive_tangent_matrix(
        local,
        (nonlocal_action,),
        constrained_local_dofs=constrained,
    )
    exact = local.createVecRight()
    exact.array[:] = np.linspace(
        0.25 + comm.rank,
        0.75 + comm.rank,
        exact.getLocalSize(),
    )
    right = local.createVecLeft()
    additive.operator.mult(exact, right)
    solved = local.createVecRight()
    ksp = PETSc.KSP().create(comm)
    ksp.setType("gmres")
    ksp.getPC().setType("jacobi")
    ksp.setTolerances(rtol=1.0e-12, atol=1.0e-14, max_it=20)
    ksp.setOperators(additive.operator, additive.preconditioner)
    ksp.solve(right, solved)

    local_error = float(np.max(np.abs(solved.array_r - exact.array_r)))
    assert comm.allreduce(local_error, op=MPI.MAX) < 2.0e-12
    assert ksp.getConvergedReason() > 0
    assert additive.summary()["preconditioner"] == "assembled_local_matrix"
    ksp.destroy()
    exact.destroy()
    right.destroy()
    solved.destroy()
    additive.close()
    local.destroy()


def _hybrid_bending_observables(comm):
    domain = dolfinx_mesh.create_unit_square(
        comm,
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
        mesh.cell_gradient_operator(domain, rings=1),
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
        petsc_options_prefix="agentfem_test_parallel_hybrid_",
    ) as problem:
        solved, info = problem.solve()
        displacement_integral = comm.allreduce(
            fem.assemble_scalar(fem.form(solved[2] * ufl.dx)),
            op=MPI.SUM,
        )
        bending_energy = comm.allreduce(bending.energy(solved), op=MPI.SUM)
        maximum = comm.allreduce(
            float(np.max(np.abs(solved.x.petsc_vec.array_r))),
            op=MPI.MAX,
        )
        physical_residual = problem.assemble_physical_residual()
        fixed_dofs, fixed_owned = fixed.dof_indices()
        fixed_dofs = fixed_dofs[:fixed_owned]
        vertical_dofs = fixed_dofs[fixed_dofs % 3 == 2]
        vertical_reaction = comm.allreduce(
            float(np.sum(physical_residual.array_r[vertical_dofs])),
            op=MPI.SUM,
        )
        physical_residual.destroy()
        free_residual_norm = problem.free_residual_norm()
        return np.array(
            (
                displacement_integral,
                bending_energy,
                maximum,
                vertical_reaction,
                free_residual_norm,
                float(info.iterations),
            )
        )


def test_distributed_hybrid_bending_matches_serial_observables():
    comm = MPI.COMM_WORLD
    if comm.size != 2:
        pytest.skip("hybrid partition evidence is frozen for two ranks")
    distributed = _hybrid_bending_observables(comm)
    serial = _hybrid_bending_observables(MPI.COMM_SELF)

    np.testing.assert_allclose(distributed[:4], serial[:4], rtol=2.0e-9, atol=2.0e-12)
    assert distributed[4] < 2.0e-10
    assert serial[4] < 2.0e-10
    assert distributed[5] == serial[5]


def test_cell_neighborhood_keeps_partition_interface_pairs_complete():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("partition-neighborhood evidence requires at least two ranks")
    domain = dolfinx_mesh.create_unit_square(
        comm,
        4,
        2,
        cell_type=dolfinx_mesh.CellType.quadrilateral,
    )
    neighborhood = mesh.cell_neighborhood(domain)
    owned_cells = int(domain.topology.index_map(domain.topology.dim).size_local)

    global_interior = comm.allreduce(
        neighborhood.owned_interior_facets,
        op=MPI.SUM,
    )
    global_exterior = comm.allreduce(
        neighborhood.owned_exterior_facets,
        op=MPI.SUM,
    )
    cross_partition = comm.allreduce(
        sum(
            any(cell >= owned_cells for cell in pair.cell_locals)
            for pair in neighborhood.pairs
        ),
        op=MPI.SUM,
    )
    facet_ids = comm.allgather(
        tuple(pair.facet_global for pair in neighborhood.pairs)
    )
    flattened = tuple(value for rank_ids in facet_ids for value in rank_ids)

    assert global_interior == 10
    assert global_exterior == 12
    assert cross_partition > 0
    assert len(flattened) == len(set(flattened))
    assert all(len(pair.cell_globals) == 2 for pair in neighborhood.pairs)
    geometry = mesh.cell_neighborhood_geometry(domain, neighborhood)
    assert len(geometry.facets) == neighborhood.owned_interior_facets
    assert all(item.center_distance > 0.0 for item in geometry.facets)
    cell_map = domain.topology.index_map(domain.topology.dim)
    owned_measures = mesh.owned_cell_measures(domain)
    global_measure = comm.allreduce(float(np.sum(owned_measures)), op=MPI.SUM)
    assert global_measure == pytest.approx(1.0, rel=1.0e-13)
    local_and_ghost = np.arange(
        cell_map.size_local + cell_map.num_ghosts,
        dtype=np.int32,
    )
    centroids = dolfinx_mesh.compute_midpoints(
        domain,
        domain.topology.dim,
        local_and_ghost,
    )[:, :2]
    gradient = np.array((2.0, -3.0))
    values = centroids @ gradient + 7.0
    difference = mesh.cell_pair_directional_difference(geometry, values)
    np.testing.assert_allclose(
        difference.values,
        difference.directions @ gradient,
        atol=1.0e-13,
    )
    stencil = mesh.cell_stencil_neighborhood(domain)
    remote_owned_pairs = sum(not pair.facet_owned for pair in stencil.pairs)
    global_remote_owned_pairs = comm.allreduce(remote_owned_pairs, op=MPI.SUM)
    assert global_remote_owned_pairs > 0
    assert all(any(pair.cell_owned) for pair in stencil.pairs)
    reconstructed = mesh.reconstruct_cell_gradient(domain, values, rings=2)
    np.testing.assert_allclose(
        reconstructed.gradients,
        np.broadcast_to(gradient, reconstructed.gradients.shape),
        atol=1.0e-12,
    )
    angle = 0.4 * centroids[:, 0]
    directions = np.column_stack(
        (np.cos(angle), np.sin(angle), np.zeros_like(angle))
    )
    tangents = np.broadcast_to(
        np.array(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
        (len(centroids), 3, 2),
    ).copy()
    curvature = mechanics.reconstruct_fiber_curvature(
        domain,
        directions,
        tangents,
        rings=2,
    )
    exact = 0.4 * np.cos(angle[: cell_map.size_local])
    local_error = float(np.max(np.abs(curvature.in_plane_curvature - exact)))
    global_error = comm.allreduce(local_error, op=MPI.MAX)
    assert global_error < 2.0e-2
    incomplete_operator = mesh.cell_gradient_operator(domain, rings=2)
    assert not incomplete_operator.partition_complete
    with pytest.raises(ValueError, match="partition-complete"):
        operators.cell_gradient_energy(
            incomplete_operator,
            cell_weights=owned_measures,
            stiffness=2.5,
        )
    operator = mesh.cell_gradient_operator(domain, rings=1)
    assert operator.partition_complete
    duals = np.column_stack(
        (
            np.linspace(0.5, 1.5, operator.owned_cells),
            np.linspace(-0.25, 0.75, operator.owned_cells),
        )
    )
    local_lhs = np.vdot(operator.apply(values).gradients, duals)
    local_rhs = np.vdot(values, operator.apply_adjoint(duals))
    assert local_lhs == pytest.approx(local_rhs, rel=1.0e-13, abs=1.0e-13)
    gradient_energy = operators.cell_gradient_energy(
        operator,
        cell_weights=owned_measures,
        stiffness=2.5,
    )
    direction = centroids[:, 0] ** 2 - 0.3 * centroids[:, 1]
    epsilon = 1.0e-5
    finite_difference = (
        gradient_energy.energy(values + epsilon * direction)
        - gradient_energy.energy(values - epsilon * direction)
    ) / (2.0 * epsilon)
    exact_derivative = np.vdot(gradient_energy.residual(values), direction)
    assert finite_difference == pytest.approx(
        exact_derivative, rel=2.0e-9, abs=2.0e-9
    )
    bending = operators.fiber_direction_bending(
        operator,
        current_tangents=tangents[: operator.owned_cells],
        cell_weights=owned_measures,
        in_plane_stiffness=1.75,
        normal_stiffness=2.0,
    )
    direction_increment = np.column_stack(
        (
            0.1 + centroids[:, 0],
            -0.2 + centroids[:, 1] ** 2,
            np.zeros(len(centroids)),
        )
    )
    bending_response = bending.evaluate(directions)
    bending_finite_difference = (
        bending.evaluate(directions + epsilon * direction_increment).energy
        - bending.evaluate(directions - epsilon * direction_increment).energy
    ) / (2.0 * epsilon)
    assert bending_finite_difference == pytest.approx(
        np.vdot(bending_response.residual, direction_increment),
        rel=2.0e-7,
        abs=2.0e-9,
    )
    bending_residual_difference = (
        bending.evaluate(directions + epsilon * direction_increment).residual
        - bending.evaluate(directions - epsilon * direction_increment).residual
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        bending.tangent_action(directions, direction_increment),
        bending_residual_difference,
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    direction_space = fem.functionspace(domain, ("DG", 0, (3,)))
    assembled = assembly.assemble_cell_residual(
        direction_space,
        bending_response.residual,
    )
    owned_increment = np.empty_like(assembled.array_r)
    block_size = int(direction_space.dofmap.index_map_bs)
    for cell in range(cell_map.size_local):
        dof = int(direction_space.dofmap.cell_dofs(cell)[0])
        owned_increment[dof * block_size : (dof + 1) * block_size] = (
            direction_increment[cell]
        )
    global_assembled_work = comm.allreduce(
        float(np.vdot(assembled.array_r, owned_increment)), op=MPI.SUM
    )
    global_energy_difference = comm.allreduce(
        float(bending_finite_difference), op=MPI.SUM
    )
    assembled.destroy()
    assert global_assembled_work == pytest.approx(
        global_energy_difference, rel=2.0e-7, abs=2.0e-9
    )
    displacement_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    displacement = fem.Function(displacement_space, name="U")
    exact_displacement_gradient = np.array(
        ((2.0, -3.0), (0.5, 1.25), (-1.0, 4.0))
    )
    displacement.interpolate(
        lambda x: exact_displacement_gradient @ x[:2]
        + np.array((4.0, -2.0, 1.0))[:, None]
    )
    transfer = operators.cell_average_gradient(displacement_space)
    transferred = transfer.apply(displacement)
    transfer_block_size = int(transfer.target_space.dofmap.index_map_bs)
    transferred_values = np.empty(
        (operator.total_cells, *transfer.value_shape), dtype=float
    )
    for cell in range(operator.total_cells):
        dof = int(transfer.target_space.dofmap.cell_dofs(cell)[0])
        transferred_values[cell] = transferred.x.array[
            dof * transfer_block_size : (dof + 1) * transfer_block_size
        ].reshape(transfer.value_shape)
    np.testing.assert_allclose(
        transferred_values,
        np.broadcast_to(exact_displacement_gradient, transferred_values.shape),
        atol=2.0e-13,
    )
    transfer_duals = np.empty_like(transferred_values)
    for component in range(3):
        transfer_duals[:, component, 0] = (
            (comm.rank + 1.0) * (component + 1.0) * (0.5 + centroids[:, 0])
        )
        transfer_duals[:, component, 1] = (
            (comm.rank + 1.0) * (component + 1.0) * (-0.25 + centroids[:, 1])
        )
    transfer_adjoint = transfer.apply_adjoint(transfer_duals)
    local_transfer_work = float(np.vdot(transferred_values, transfer_duals))
    local_source_work = float(
        np.vdot(displacement.x.petsc_vec.array_r, transfer_adjoint.array_r)
    )
    transfer_adjoint.destroy()
    assert comm.allreduce(local_transfer_work, op=MPI.SUM) == pytest.approx(
        comm.allreduce(local_source_work, op=MPI.SUM),
        rel=2.0e-13,
        abs=2.0e-13,
    )
    fiber_transfer = operators.convected_cell_fiber(
        transfer,
        reference_tangents=np.array(
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
        ),
        reference_tangent_coordinates=np.array((np.cos(0.35), np.sin(0.35))),
    )
    displacement_increment = fem.Function(displacement_space, name="DU")
    displacement_increment.interpolate(
        lambda x: np.vstack(
            (-0.2 * x[0] + 0.1 * x[1], 0.3 * x[0], -0.15 * x[1])
        )
    )
    fiber_increment = fiber_transfer.directional_derivative(
        displacement,
        displacement_increment,
    )
    fiber_state = fiber_transfer.apply(displacement)
    direction_duals = np.empty_like(fiber_state.direction)
    tangent_duals = np.empty_like(fiber_state.current_tangents)
    stretch_duals = np.empty_like(fiber_state.stretch)
    direction_duals[:, 0] = (comm.rank + 1.0) * (0.5 + centroids[:, 0])
    direction_duals[:, 1] = (comm.rank + 1.0) * (-0.3 + centroids[:, 1])
    direction_duals[:, 2] = (comm.rank + 1.0) * 0.2
    tangent_duals[:] = (comm.rank + 1.0) * np.array(
        ((0.2, -0.1), (0.3, 0.4), (-0.25, 0.15))
    )
    stretch_duals[:] = (comm.rank + 1.0) * (0.7 + centroids[:, 0])
    fiber_adjoint = fiber_transfer.apply_adjoint(
        displacement,
        direction_duals=direction_duals,
        tangent_duals=tangent_duals,
        stretch_duals=stretch_duals,
    )
    local_fiber_work = float(
        np.vdot(fiber_increment.direction_increment, direction_duals)
        + np.vdot(fiber_increment.tangent_increment, tangent_duals)
        + np.vdot(fiber_increment.stretch_increment, stretch_duals)
    )
    local_displacement_work = float(
        np.vdot(
            displacement_increment.x.petsc_vec.array_r,
            fiber_adjoint.array_r,
        )
    )
    fiber_adjoint.destroy()
    assert comm.allreduce(local_fiber_work, op=MPI.SUM) == pytest.approx(
        comm.allreduce(local_displacement_work, op=MPI.SUM),
        rel=3.0e-13,
        abs=3.0e-13,
    )
    bend_displacement = fem.Function(displacement_space, name="U_bend")
    bend_displacement.interpolate(
        lambda x: np.vstack(
            (
                0.08 * x[0] ** 2 + 0.03 * x[1],
                0.06 * x[0] * x[1] - 0.02 * x[0],
                0.05 * x[0] ** 2 + 0.04 * x[1] ** 2,
            )
        )
    )
    bend_increment = fem.Function(displacement_space, name="DU_bend")
    bend_increment.interpolate(
        lambda x: np.vstack(
            (
                -0.03 * x[0] + 0.02 * x[1],
                0.04 * x[0] ** 2,
                -0.02 * x[0] * x[1] + 0.01 * x[1],
            )
        )
    )

    def displacement_bending_response(field):
        state = fiber_transfer.apply(field)
        model = operators.fiber_direction_bending(
            operator,
            current_tangents=state.current_tangents[: operator.owned_cells],
            cell_weights=owned_measures,
            in_plane_stiffness=2.5,
            normal_stiffness=4.0,
        )
        return state, model, model.evaluate(state.direction)

    bend_state, bend_model, bend_response = displacement_bending_response(
        bend_displacement
    )
    bend_tangent_duals = np.zeros_like(bend_state.current_tangents)
    bend_tangent_duals[: operator.owned_cells] = (
        bend_response.surface_tangent_residual
    )
    bend_residual = fiber_transfer.apply_adjoint(
        bend_displacement,
        direction_duals=bend_response.residual,
        tangent_duals=bend_tangent_duals,
    )
    plus = fem.Function(displacement_space)
    minus = fem.Function(displacement_space)
    plus.x.array[:] = bend_displacement.x.array + epsilon * bend_increment.x.array
    minus.x.array[:] = bend_displacement.x.array - epsilon * bend_increment.x.array
    plus.x.scatter_forward()
    minus.x.scatter_forward()
    local_energy_difference = (
        displacement_bending_response(plus)[2].energy
        - displacement_bending_response(minus)[2].energy
    ) / (2.0 * epsilon)
    local_residual_work = float(
        np.vdot(bend_increment.x.petsc_vec.array_r, bend_residual.array_r)
    )
    bend_residual.destroy()
    assert comm.allreduce(local_energy_difference, op=MPI.SUM) == pytest.approx(
        comm.allreduce(local_residual_work, op=MPI.SUM),
        rel=5.0e-7,
        abs=5.0e-9,
    )
    bend_kinematic_increment = fiber_transfer.directional_derivative(
        bend_displacement,
        bend_increment,
    )
    bend_response_increment = bend_model.linearized_response(
        bend_state.direction,
        bend_kinematic_increment.direction_increment,
        surface_tangent_increment=bend_kinematic_increment.tangent_increment[
            : operator.owned_cells
        ],
    )
    bend_tangent_dual_increments = np.zeros_like(
        bend_state.current_tangents
    )
    bend_tangent_dual_increments[: operator.owned_cells] = (
        bend_response_increment.surface_tangent_residual_increment
    )
    bend_tangent = fiber_transfer.apply_adjoint_derivative(
        bend_displacement,
        bend_increment,
        direction_duals=bend_response.residual,
        direction_dual_increments=(
            bend_response_increment.direction_residual_increment
        ),
        tangent_duals=bend_tangent_duals,
        tangent_dual_increments=bend_tangent_dual_increments,
    )

    def displacement_bending_residual(field):
        trial_state, _, trial_response = displacement_bending_response(field)
        trial_tangent_duals = np.zeros_like(trial_state.current_tangents)
        trial_tangent_duals[: operator.owned_cells] = (
            trial_response.surface_tangent_residual
        )
        return fiber_transfer.apply_adjoint(
            field,
            direction_duals=trial_response.residual,
            tangent_duals=trial_tangent_duals,
        )

    plus_bend_residual = displacement_bending_residual(plus)
    minus_bend_residual = displacement_bending_residual(minus)
    local_tangent_error = float(
        np.max(
            np.abs(
                bend_tangent.array_r
                - (
                    plus_bend_residual.array_r
                    - minus_bend_residual.array_r
                )
                / (2.0 * epsilon)
            )
        )
    )
    bend_tangent.destroy()
    plus_bend_residual.destroy()
    minus_bend_residual.destroy()
    assert comm.allreduce(local_tangent_error, op=MPI.MAX) < 8.0e-8


def test_distributed_abaqus_regions_quality_and_remote_resultant():
    pytest.importorskip("meshio")
    comm = MPI.COMM_WORLD
    token = comm.bcast(uuid.uuid4().hex if comm.rank == 0 else None, root=0)
    directory = Path(tempfile.gettempdir()) / f"agentfem-parallel-mesh-{token}"
    source = directory / "two_tetra.inp"
    converted = directory / "two_tetra.xdmf"
    if comm.rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        source.write_text(
            "\n".join(
                (
                    "*Heading",
                    "*Node",
                    "1, 0., 0., 0.",
                    "2, 1., 0., 0.",
                    "3, 0., 1., 0.",
                    "4, 0., 0., 1.",
                    "5, 1., 1., 1.",
                    "*Nset, nset=FIXED",
                    "1, 4",
                    "*Element, type=C3D4, elset=SOLID",
                    "1, 1, 2, 3, 4",
                    "2, 2, 3, 4, 5",
                    "*Surface, name=LOADED, type=ELEMENT",
                    "1, S1",
                )
            ),
            encoding="utf-8",
        )
    comm.barrier()

    imported = mesh.read_abaqus_mesh(
        source,
        converted,
        comm=comm,
        cell_type="tetra",
        reuse_conversion=False,
    )
    fixed_nodes = imported.node_set("FIXED")
    loaded = imported.boundary("LOADED", tag=17)
    fixed = constraints.fixed(fields.displacement(imported.domain), on=fixed_nodes)
    quality = mesh.audit_quality(imported.domain, threshold=0.1, strict=True)
    remote = loads.remote_force(
        (3.0, -4.0, 5.0),
        reference_point=(1.0 / 3.0, 1.0 / 3.0, 0.0),
        on=loaded,
    )
    resultant = results.boundary_resultant(remote.traction, on=loaded)

    assert fixed_nodes.summary()["global_nodes"] == 2
    assert loaded.audit(strict=True)["global_tagged_facets"] == 1
    assert len(fixed.bcs) == 3
    assert quality.global_cells == 2
    np.testing.assert_allclose(resultant, (3.0, -4.0, 5.0), atol=1.0e-11)

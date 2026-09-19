from __future__ import annotations

from pathlib import Path
import tempfile
import uuid

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import constraints, fields, loads, mechanics, mesh, operators, results


def test_cell_neighborhood_keeps_partition_interface_pairs_complete():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("partition-neighborhood evidence requires at least two ranks")
    domain = dolfinx_mesh.create_unit_square(comm, 4, 2)
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

    assert global_interior == 18
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
    operator = mesh.cell_gradient_operator(domain, rings=2)
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

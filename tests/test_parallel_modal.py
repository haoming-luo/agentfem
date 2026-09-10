from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from agentfem import fields, mesh, models, operators, studies
from agentfem.constitutive import isotropic_elastic
from agentfem.operators.identity import _function_content_identity
from agentfem.provenance import collective_scientific_input_manifest
from agentfem.results import SimulationResult


def test_structural_modes_use_one_distributed_reduced_eigenproblem():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed modal verification requires at least two ranks")

    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (8, 2),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.modal_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain, degree=2))
    model.material(
        isotropic_elastic(
            young=210.0e9,
            poisson=0.3,
            density=7800.0,
        )
    )
    model.clamp(
        displacement,
        on=mesh.boundary(
            domain,
            lambda x: np.isclose(x[0], 0.0),
            name="fixed_end",
            tag=1,
        ),
    )

    step = model.step(target=displacement, modes=3)
    result = step.solve_result()
    frequency = np.asarray(result.quantity("frequencies"))
    gathered = MPI.COMM_WORLD.allgather(frequency)

    for rank_frequency in gathered:
        np.testing.assert_allclose(rank_frequency, frequency, rtol=1.0e-10)
    assert frequency[0] == pytest.approx(163.27832561, rel=1.0e-7)
    solve = result.metadata["solve"]
    assert solve["free_dofs"] > 0
    assert solve["constrained_dofs"] > 0
    assert solve["mass_orthogonality_error"] < 1.0e-10
    assert solve["stiffness_diagonalization_error"] < 1.0e-10
    assert solve["selected_clusters_complete"]
    assert solve["repeated_modes_compare_as"] == "invariant_subspace"
    assert sum(
        cluster["multiplicity"] for cluster in solve["eigenvalue_clusters"]
    ) == 3
    assert len(solve["orientation_anchor_dofs"]) == 3
    gathered_anchors = MPI.COMM_WORLD.allgather(solve["orientation_anchor_dofs"])
    assert all(item == gathered_anchors[0] for item in gathered_anchors)
    gathered_clusters = MPI.COMM_WORLD.allgather(solve["eigenvalue_clusters"])
    assert all(item == gathered_clusters[0] for item in gathered_clusters)
    manifest = result.scientific_input_manifest()
    assert manifest["complete"] is True
    gathered_manifests = MPI.COMM_WORLD.allgather(manifest["fingerprint"])
    assert all(item == gathered_manifests[0] for item in gathered_manifests)
    executable = result.scientific_inputs["modal_executable_identity"]
    assert executable["complete"] is True
    gathered_fingerprints = MPI.COMM_WORLD.allgather(executable["fingerprint"])
    assert all(item == gathered_fingerprints[0] for item in gathered_fingerprints)

    accepted_request = step._executed_request_manifest
    if MPI.COMM_WORLD.rank == 1:
        step._executed_request_manifest = None
    with pytest.raises(RuntimeError, match="require a completed solve"):
        step.scientific_inputs()
    step._executed_request_manifest = accepted_request
    MPI.COMM_WORLD.barrier()


def test_modal_rank_local_coefficient_drift_fails_together():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed identity failure requires at least two ranks")

    comm = MPI.COMM_WORLD
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 1),
        comm=comm,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.modal_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain, degree=1))
    model.material(
        isotropic_elastic(
            young=210.0e9,
            poisson=0.3,
            density=7800.0,
        )
    )
    model.clamp(
        displacement,
        on=mesh.boundary(
            domain,
            lambda x: np.isclose(x[0], 0.0),
            name="fixed_end",
            tag=1,
        ),
    )
    scale = fem.Constant(domain, PETSc.ScalarType(1.0))
    stiffness = operators.scale(model.stiffness(displacement), scale)
    step = model.step(target=displacement, modes=1, K=stiffness)
    step.solve()

    if comm.rank == 0:
        scale.value[...] = 1.01
    with pytest.raises(ValueError, match="cannot establish.*executable identity"):
        step.scientific_inputs()

    comm.barrier()


def test_rank_local_identity_construction_failure_fails_before_coordinate_collective():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed identity failure requires at least two ranks")

    comm = MPI.COMM_WORLD
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=comm,
        cell_type="quadrilateral",
    )
    space = fem.functionspace(domain, ("Lagrange", 1))
    valid = fem.Function(space)
    coefficient = object() if comm.rank == 1 else valid

    with pytest.raises(
        RuntimeError,
        match="build local coefficient identity inputs.*rank 1",
    ):
        _function_content_identity(coefficient, domain=domain)

    comm.barrier()


def test_rank_local_scientific_input_manifest_difference_fails_collectively():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed manifest failure requires at least two ranks")

    comm = MPI.COMM_WORLD
    with pytest.raises(RuntimeError, match="differs across MPI ranks"):
        collective_scientific_input_manifest(
            {"rank_dependent_value": comm.rank},
            comm=comm,
            label="rank_dependent_test",
            require_nonempty=True,
        )

    comm.barrier()


def test_rank_local_complete_result_manifest_difference_fails_collectively():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed manifest failure requires at least two ranks")

    comm = MPI.COMM_WORLD
    result = SimulationResult(
        name="rank_dependent_result",
        metadata={"rank_dependent_value": comm.rank},
        scientific_inputs={"problem": "shared"},
    )
    with pytest.raises(RuntimeError, match="SimulationResult manifest differs"):
        result.collective_manifest(comm, include_histories=True)

    comm.barrier()

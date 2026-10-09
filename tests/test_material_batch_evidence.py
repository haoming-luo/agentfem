# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from agentfem import constitutive


def law():
    return constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=1, hardening_modulus=2
    )


@pytest.mark.parametrize("chunk", [1, 3, 7, 1024])
def test_material_chunk_partition_does_not_change_response_or_state(chunk):
    material = law()
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    state = constitutive.MaterialQuadratureState.create(
        domain, material.state_schema, degree=1
    )
    count = len(state.committed_state_vectors())
    gradient = np.tile(np.eye(3), (count, 1, 1))
    gradient[:, 0, 0] = np.linspace(1.001, 1.08, count)
    common = dict(
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=gradient,
        time=0,
        time_increment=0.1,
    )
    reference = constitutive.update_material_points(
        material, state, max_batch_points=count, **common
    )
    state.rollback()
    selected = constitutive.update_material_points(
        material, state, max_batch_points=chunk, **common
    )
    for name in (
        "cauchy_stress",
        "consistent_tangent",
        "state_new",
        "strain_energy_density",
        "dissipation_density_increment",
    ):
        np.testing.assert_allclose(
            getattr(selected, name), getattr(reference, name), rtol=1e-12, atol=1e-12
        )
    assert selected.provider_batch_calls == (count + chunk - 1) // chunk
    assert np.all(selected.dissipation_density_increment_defined)
    np.testing.assert_array_equal(state.trial_state_vectors(), selected.state_new)
    np.testing.assert_array_equal(
        state.committed_state_vectors(),
        np.tile(material.state_schema.initial_state(), (count, 1)),
    )


def test_later_chunk_failure_is_collective_and_preserves_entire_state():
    material = law()
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    state = constitutive.MaterialQuadratureState.create(
        domain, material.state_schema, degree=1
    )
    before = state.committed_state_vectors().copy()

    class FailingBatch:
        name = "later chunk failure"
        state_schema = material.state_schema
        tangent_convention = material.tangent_convention
        stored_energy_component_names = material.stored_energy_component_names
        calls = 0

        def update(self, point):
            raise AssertionError("Batch path must not use scalar update")

        def update_batch(self, request):
            self.calls += 1
            if domain.comm.rank == domain.comm.size - 1 and self.calls == 3:
                raise ValueError("injected late chunk failure")
            return material.update_batch(request)

    provider = FailingBatch()
    with pytest.raises(RuntimeError, match="late chunk failure"):
        constitutive.update_material_points(
            provider,
            state,
            deformation_gradient_old=np.eye(3),
            deformation_gradient_new=1.01 * np.eye(3),
            time=0,
            time_increment=0.1,
            max_batch_points=2,
            commit=True,
        )
    np.testing.assert_array_equal(state.committed_state_vectors(), before)
    np.testing.assert_array_equal(state.trial_state_vectors(), before)
    assert provider.calls >= 3


@pytest.mark.parametrize("chunk", [0, -1, True, 2.5])
def test_invalid_batch_bound_is_rejected_before_state_mutation(chunk):
    material = law()
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    state = constitutive.MaterialQuadratureState.create(
        domain, material.state_schema, degree=1
    )
    before = state.committed_state_vectors().copy()
    constitutive.update_material_points(
        material,
        state,
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=np.diag([1.08, 1, 1]),
        time=0,
        time_increment=0.1,
    )
    assert not np.array_equal(state.trial_state_vectors(), before)
    with pytest.raises(RuntimeError, match="max_batch_points"):
        constitutive.update_material_points(
            material,
            state,
            deformation_gradient_old=np.eye(3),
            deformation_gradient_new=np.eye(3),
            time=0,
            time_increment=0.1,
            max_batch_points=chunk,
        )
    np.testing.assert_array_equal(state.committed_state_vectors(), before)
    np.testing.assert_array_equal(state.trial_state_vectors(), before)


def test_missing_energy_is_not_reported_as_defined_zero():
    material = law()

    class StressOnly:
        name = "stress-only evidence test"
        state_schema = material.state_schema
        tangent_convention = material.tangent_convention

        def update(self, point):
            return replace(
                material.update(point),
                strain_energy_density=None,
                stored_energy_density_components={},
                dissipation_density_increment=None,
            )

    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    state = constitutive.MaterialQuadratureState.create(
        domain, material.state_schema, degree=1
    )
    result = constitutive.update_material_points(
        StressOnly(),
        state,
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=np.eye(3),
        time=0,
        time_increment=0.1,
    )
    assert np.all(result.strain_energy_density == 0)  # compatibility storage only
    assert not np.any(result.strain_energy_density_defined)
    assert result.summary()["stored_energy_defined_points"] == 0
    assert result.summary()["dissipation_increment_defined_points"] == 0
    assert not np.any(result.dissipation_density_increment_defined)


def test_finite_j2_dissipation_increment_tracks_existing_accepted_history():
    material = law()
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    state = constitutive.MaterialQuadratureState.create(
        domain, material.state_schema, degree=1
    )
    old_f = np.eye(3)
    increments = []
    for stretch in (1.0, 1.04, 1.08, 1.079, 1.04):
        old_state = state.committed_state_vectors().copy()
        new_f = np.diag([stretch, 1 / np.sqrt(stretch), 1 / np.sqrt(stretch)])
        result = constitutive.update_material_points(
            material,
            state,
            deformation_gradient_old=old_f,
            deformation_gradient_new=new_f,
            time=0,
            time_increment=0.1,
            commit=True,
            max_batch_points=2,
        )
        assert np.all(result.dissipation_density_increment_defined)
        np.testing.assert_allclose(
            result.dissipation_density_increment,
            material.yield_stress * (result.state_new[:, 9] - old_state[:, 9]),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            result.dissipation_density_increment,
            result.state_new[:, 10] - old_state[:, 10],
            atol=1e-14,
        )
        assert np.all(result.dissipation_density_increment >= 0)
        increments.append(result.dissipation_density_increment)
        old_f = new_f
    np.testing.assert_array_equal(increments[0], 0)
    np.testing.assert_allclose(
        np.sum(increments, axis=0), state.committed_state_vectors()[:, 10], atol=1e-14
    )


def test_quadrature_response_empty_rank_preserves_component_and_summary_contract():
    material = law()
    from dolfinx import graph

    def owner_zero(comm, parts, types, cells):
        count = (
            cells.num_nodes
            if hasattr(cells, "num_nodes")
            else sum(array.size // 8 for array in cells)
        )
        result = graph.adjacencylist(np.zeros((count, 1), dtype=np.int32))
        return getattr(result, "_cpp_object", result)

    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [[0, 0, 0], [1, 1, 1]],
        [1, 1, 1],
        cell_type=mesh.CellType.hexahedron,
        partitioner=owner_zero if MPI.COMM_WORLD.size > 1 else None,
    )
    response = constitutive.MaterialQuadratureResponse.create(
        domain,
        material.state_schema,
        degree=1,
        stored_energy_component_names=material.stored_energy_component_names,
    )
    result = response.update(
        material,
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=1.01 * np.eye(3),
        time=0,
        time_increment=0.1,
        commit=True,
    )
    assert set(result.stored_energy_density_components) == set(
        material.stored_energy_component_names
    )
    assert result.summary()["stored_energy_defined_points"] == result.point_count
    assert result.minimum_suggested_time_scale > 0
    if not result.point_count:
        assert result.material_group_count == 0
        assert response.first_piola_stress.values.shape == (0, 3, 3)
    counts = domain.comm.allgather(result.point_count)
    if domain.comm.size > 1:
        assert 0 in counts

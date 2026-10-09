# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from agentfem import constitutive


def law():
    return constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=1, hardening_modulus=2
    )


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

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import interfaces
from agentfem import mesh as mesh_api
from test_nonmatching_global import _blocks


@pytest.mark.parametrize("topology", ["tetrahedron", "hexahedron"])
@pytest.mark.parametrize("axis,side", [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
def test_boundary_trace_outward_orientation_all_faces(topology, axis, side):
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, 2, 2, 2, cell_type=getattr(mesh.CellType, topology)
    )
    displacement = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    boundary = mesh_api.boundary_region(domain, lambda x: np.isclose(x[axis], side))
    trace, dofs = interfaces.reference_trace_from_boundary(displacement, boundary)
    surface = getattr(trace, "triangulated", trace)
    expected = np.zeros(3)
    expected[axis] = 2 * side - 1
    np.testing.assert_allclose(
        surface.facet_normals,
        np.broadcast_to(expected, surface.facet_normals.shape),
        atol=1e-13,
    )
    np.testing.assert_array_equal(
        trace.vertices, displacement.function_space.tabulate_dof_coordinates()[dofs]
    )
    assert not dofs.flags.writeable


@pytest.mark.parametrize("topology", ["tetrahedron", "hexahedron"])
def test_coincident_nodes_keep_independent_parent_dofs(topology):
    domain = _blocks(1, 2, cell_type=topology)
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    displacement = fem.Function(space)
    domain.topology.create_connectivity(2, 3)
    facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 0))
    xyz = space.tabulate_dof_coordinates()
    tags = []
    for facet in facets:
        cell = domain.topology.connectivity(2, 3).links(facet)[0]
        tags.append(2 if xyz[space.dofmap.cell_dofs(cell), 2].mean() > 0 else 1)
    facet_tags = mesh.meshtags(domain, 2, facets, np.array(tags, dtype=np.int32))
    traces = []
    for tag in (1, 2):
        boundary = mesh_api.BoundaryRegion(
            str(tag),
            domain,
            None,
            tag,
            facet_tags,
            ufl.Measure("ds", domain=domain, subdomain_data=facet_tags),
        )
        traces.append(interfaces.reference_trace_from_boundary(displacement, boundary))
    (negative, dn), (positive, dp) = traces
    assert not np.intersect1d(dn, dp).size
    pair = interfaces.pair_reference_traces(negative, positive, tolerance=1e-10)
    np.testing.assert_allclose(pair.weights.sum(), 1)
    assert all(
        side["relative_nodal_measure_error_l2"] < 1e-12
        for side in pair.constant_traction_audit()["sides"].values()
    )

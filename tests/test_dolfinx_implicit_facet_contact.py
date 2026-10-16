from __future__ import annotations

import basix.ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import pytest

from agentfem import boundary_models
from agentfem import mesh as agent_mesh
from agentfem.boundary_models.dolfinx_implicit_facet_contact import (
    dolfinx_implicit_facet_contact,
)


def _model():
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1, cell_type=mesh.CellType.tetrahedron)
    space = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)))
    displacement = fem.Function(space, name="Displacement")
    facets = mesh.locate_entities_boundary(
        domain, 2, lambda x: np.isclose(x[0], 0.0)
    )
    tags = mesh.meshtags(domain, 2, facets, np.full(len(facets), 7, dtype=np.int32))
    slave = agent_mesh.tagged_boundary_region(domain, tags, tag=7, name="slave")
    tool = boundary_models.triangulated_rigid_surface(
        vertices=((0.05, 0, 0), (0.05, 0, 1), (0.05, 1, 1), (0.05, 1, 0)),
        triangles=((0, 1, 2), (0, 2, 3)),
    )
    pair = boundary_models.rigid_contact_pair(
        slave,
        boundary_models.rigid_body(tool, reference_point=(0.05, 0, 0)),
        penalty=200,
    )
    operator = dolfinx_implicit_facet_contact(pair, displacement)
    return displacement, operator


def test_dolfinx_implicit_facet_tangent_matches_current_residual_difference():
    displacement, operator = _model()
    base = displacement.x.array.copy()
    base.reshape((-1, 3))[:, 0] = 0.1
    displacement.x.array[:] = base
    contribution = operator.evaluate()
    assert contribution.active_points > 0
    assert contribution.assembly.potential_energy > 0
    assert operator.summary()["execution_scope"] == "serial_fixed_frictionless_reference_area"
    rng = np.random.default_rng(231)
    direction = rng.normal(size=base.shape).reshape((-1, 3))
    step = 1e-7
    displacement.x.array[:] = base + step * direction.reshape(-1)
    plus = operator.evaluate()
    displacement.x.array[:] = base - step * direction.reshape(-1)
    minus = operator.evaluate()
    displacement.x.array[:] = base
    difference = (
        plus.assembly.nodal_structural_residual
        - minus.assembly.nodal_structural_residual
    ) / (2 * step)
    np.testing.assert_allclose(contribution.tangent_action(direction), difference, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        np.sum(contribution.assembly.nodal_structural_residual, axis=0),
        -contribution.assembly.contact_force_on_structure,
        atol=1e-12,
    )


def test_implicit_facet_rejects_unreviewed_tool_motion():
    displacement, operator = _model()
    motion = boundary_models.prescribed_rigid_motion(translation=(0.01, 0, 0))
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        motion, start_time=0, end_time=1,
    )
    pair = boundary_models.rigid_contact_pair(
        operator.pair.slave_boundary,
        boundary_models.rigid_body(operator.surface, motion_schedule=schedule),
        penalty=200,
    )
    with pytest.raises(NotImplementedError, match="work and pose rollback"):
        dolfinx_implicit_facet_contact(pair, displacement)

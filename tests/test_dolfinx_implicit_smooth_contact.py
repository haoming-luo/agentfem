"""Independent UFL energy reference for the curved rigid-contact Operator."""

from __future__ import annotations

import basix.ufl
from dolfinx import fem, mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np
import pytest
import ufl

from agentfem import boundary_models
from agentfem import mesh as agent_mesh
from agentfem.boundary_models.dolfinx_implicit_facet_contact import (
    dolfinx_implicit_rigid_contact,
)


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("contact_quadrature_degree", [None, 5, 7])
@pytest.mark.parametrize(
    "cell_type", [dolfinx_mesh.CellType.tetrahedron, dolfinx_mesh.CellType.hexahedron],
)
def test_curved_contact_matches_independent_ufl_energy_gradient_and_hessian(
    geometry, degree, cell_type, contact_quadrature_degree,
):
    if contact_quadrature_degree is not None and (
        cell_type != dolfinx_mesh.CellType.hexahedron or degree != 1
    ):
        pytest.skip("Refined contact integration is bounded to CG1/Q1 hexes.")
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, 1, 1, 1, cell_type=cell_type,
    )
    space = fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            domain.basix_cell(),
            degree,
            shape=(3,),
        ),
    )
    displacement = fem.Function(space)
    displacement.x.array.reshape(-1, 3)[:] = (0.08, -0.02, 0.03)
    facets = dolfinx_mesh.locate_entities_boundary(
        domain,
        2,
        lambda x: np.isclose(x[0], 0.0),
    )
    tags = dolfinx_mesh.meshtags(
        domain,
        2,
        facets,
        np.full(len(facets), 7, dtype=np.int32),
    )
    slave = agent_mesh.tagged_boundary_region(domain, tags, tag=7, name="slave")
    center = (0.65, 0.5, 0.5)
    radius = 1.0
    penalty = 200.0
    surface = (
        boundary_models.rigid_sphere(center=center, radius=radius)
        if geometry == "sphere"
        else boundary_models.rigid_cylinder(
            axis_point=center,
            axis_direction=(0, 0, 1),
            radius=radius,
        )
    )
    pair = boundary_models.rigid_contact_pair(
        slave,
        boundary_models.rigid_body(
            surface,
            reference_point=center,
        ),
        penalty=penalty,
    )
    contribution = dolfinx_implicit_rigid_contact(
        pair, displacement,
        contact_quadrature_degree=contact_quadrature_degree,
    ).evaluate()
    assert contribution.active_points > 0

    current = ufl.SpatialCoordinate(domain) + displacement
    offset = current - ufl.as_vector(center)
    squared_distance = (
        ufl.inner(offset, offset)
        if geometry == "sphere"
        else offset[0] ** 2 + offset[1] ** 2
    )
    gap = ufl.sqrt(squared_distance) - radius
    negative_gap = ufl.min_value(gap, 0.0)
    ds = ufl.Measure(
        "ds", domain=domain, subdomain_data=tags,
        metadata={
            "quadrature_degree": (
                2 * degree if contact_quadrature_degree is None
                else contact_quadrature_degree
            ),
        },
    )
    potential = 0.5 * penalty * negative_gap**2 * ds(7)
    test = ufl.TestFunction(space)
    trial = ufl.TrialFunction(space)
    residual_form = fem.form(ufl.derivative(potential, displacement, test))
    tangent_form = fem.form(
        ufl.derivative(
            ufl.derivative(potential, displacement, test),
            displacement,
            trial,
        )
    )
    reference_energy = fem.assemble_scalar(fem.form(potential))
    reference_residual = fem.petsc.assemble_vector(residual_form)
    reference_tangent = fem.petsc.assemble_matrix(tangent_form)
    reference_tangent.assemble()
    direction = reference_residual.duplicate()
    action = reference_residual.duplicate()
    try:
        np.testing.assert_allclose(
            contribution.assembly.potential_energy,
            reference_energy,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            contribution.assembly.nodal_structural_residual.reshape(-1),
            reference_residual.array,
            rtol=1e-12,
            atol=1e-11,
        )
        direction.array[:] = np.linspace(-0.3, 0.2, direction.array.size)
        reference_tangent.mult(direction, action)
        np.testing.assert_allclose(
            contribution.tangent_action(direction.array.reshape(-1, 3)).reshape(-1),
            action.array,
            rtol=1e-11,
            atol=1e-10,
        )
    finally:
        action.destroy()
        direction.destroy()
        reference_tangent.destroy()
        reference_residual.destroy()

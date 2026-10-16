"""The bounded J2/contact composition must fail before a distributed solve."""

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import boundary_models, constitutive, fields, mesh, models, studies


@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="two MPI ranks required")
@pytest.mark.parametrize("tool_kind", ["analytical_plane", "triangulated_plane"])
def test_distributed_j2_rigid_contact_is_rejected_at_step_build(tool_kind):
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    model = models.create(
        study=studies.nonlinear_static(physics="solid_mechanics", dimension=3),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    model.clamp(displacement, on=left)
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=1_000.0,
            poisson=0.0,
            yield_stress=2.0,
            hardening_modulus=100.0,
        )
    )
    if tool_kind == "analytical_plane":
        model.rigid_obstacle_contact(
            on=right, penalty=500.0, normal=(-1.0, 0.0, 0.0)
        )
    else:
        surface = boundary_models.triangulated_rigid_surface(
            vertices=((1.01, -0.5, -0.5), (1.01, -0.5, 1.5),
                      (1.01, 1.5, 1.5), (1.01, 1.5, -0.5)),
            triangles=((0, 1, 2), (0, 2, 3)),
        )
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            right, boundary_models.rigid_body(surface), penalty=500.0,
        ))
    capability = models.step_capability(
        model, target=displacement, options={"material": material}
    )
    assert not capability["supported"]
    assert capability["rejection_reasons"][0]["message"].startswith(
        "AFM-J2-CONTACT-MPI:"
    )
    with pytest.raises(NotImplementedError, match="AFM-J2-CONTACT-MPI"):
        model.step(target=displacement, material=material)

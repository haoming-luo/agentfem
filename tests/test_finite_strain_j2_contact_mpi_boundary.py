"""The bounded J2/contact composition must fail before a distributed solve."""

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import boundary_models, constitutive, fields, mesh, models, studies

from periodic_cube_fixture import periodic_unit_cube


@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="two MPI ranks required")
@pytest.mark.parametrize("tool_kind", [
    "analytical_plane", "triangulated_plane", "moving_triangulated_plane",
    "sphere", "cylinder", "closed_convex_friction", "reviewed_closed_general",
])
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
        if tool_kind in ("closed_convex_friction", "reviewed_closed_general"):
            factory = (
                boundary_models.closed_triangulated_rigid_solid
                if tool_kind == "reviewed_closed_general"
                else boundary_models.convex_polyhedral_rigid_surface
            )
            surface = factory(
                vertices=(
                    (1.01, -0.5, -0.5), (1.11, -0.5, -0.5),
                    (1.11, 1.5, -0.5), (1.01, 1.5, -0.5),
                    (1.01, -0.5, 1.5), (1.11, -0.5, 1.5),
                    (1.11, 1.5, 1.5), (1.01, 1.5, 1.5),
                ),
                triangles=(
                    (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                    (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
                    (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
                ),
                **(
                    {"reviewed_embedded": True}
                    if tool_kind == "reviewed_closed_general" else {}
                ),
            )
        else:
            surface = (
                boundary_models.rigid_sphere(
                center=(1.6, 0.5, 0.5), radius=0.75,
            ) if tool_kind == "sphere" else
            boundary_models.rigid_cylinder(
                axis_point=(1.6, 0.5, 0.0),
                axis_direction=(0.0, 0.0, 1.0), radius=0.75,
            ) if tool_kind == "cylinder" else
            boundary_models.triangulated_rigid_surface(
                vertices=((1.01, -0.5, -0.5), (1.01, -0.5, 1.5),
                          (1.01, 1.5, 1.5), (1.01, 1.5, -0.5)),
                triangles=((0, 1, 2), (0, 2, 3)),
            )
            )
        schedule = None
        if tool_kind == "moving_triangulated_plane":
            motion = boundary_models.prescribed_rigid_motion(
                translation=(-0.01, 0.0, 0.0),
                reference_point=(1.01, 0.0, 0.0),
            )
            schedule = boundary_models.prescribed_rigid_motion_schedule(
                motion, start_time=0.0, end_time=1.0,
            )
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            right,
            boundary_models.rigid_body(surface, motion_schedule=schedule),
            penalty=500.0,
            friction_coefficient=(
                0.2 if tool_kind == "closed_convex_friction" else None
            ),
            tangential_penalty=(
                100.0 if tool_kind == "closed_convex_friction" else None
            ),
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


@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="two MPI ranks required")
def test_distributed_periodic_j2_fixed_curved_contact_is_rejected_early():
    fixture = periodic_unit_cube(MPI.COMM_WORLD)
    model = models.create(
        study=studies.nonlinear_static(physics="solid_mechanics", dimension=3),
        mesh=fixture.domain,
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0, poisson=0.3,
            yield_stress=200.0, hardening_modulus=2_000.0,
        )
    )
    periodicity = model.constraint(fixture.constraint(displacement))
    right = mesh.boundary(
        fixture.domain, lambda x: np.isclose(x[0], 1.0), name="slave",
    )
    surface = boundary_models.rigid_sphere(
        center=(1.6, 0.5, 0.5), radius=0.75,
    )
    model.add_boundary_model(
        boundary_models.rigid_contact_pair(
            right, boundary_models.rigid_body(surface), penalty=5_000.0,
        )
    )
    capability = models.step_capability(
        model, target=displacement,
        options={"material": material, "constraints": periodicity},
    )
    assert not capability["supported"]
    assert any(
        item["message"].startswith("AFM-J2-CONTACT-MPC")
        for item in capability["rejection_reasons"]
    )
    with pytest.raises(NotImplementedError, match="AFM-J2-CONTACT-MPC"):
        model.step(target=displacement, material=material,
                   constraints=periodicity)

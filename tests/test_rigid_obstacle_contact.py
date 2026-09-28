from __future__ import annotations

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, results, studies


def _left(x):
    return np.isclose(x[0], 0.0)


def _right(x):
    return np.isclose(x[0], 1.0)


def _contact_model():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (8, 4),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(
            dimension=2,
            assumption="plane_stress",
            nonlinear=True,
        ),
        mesh=domain,
        name="rigid_plane_contact_patch",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        constitutive.elasticity.isotropic_elastic(
            young=1.0e3,
            poisson=0.0,
            density=1.0,
        )
    )
    left = mesh.boundary(domain, _left, name="left", tag=1)
    right = mesh.boundary(domain, _right, name="contact_and_load", tag=2)
    model.clamp(displacement, on=left)
    contact = model.rigid_obstacle_contact(
        on=right,
        penalty=1.0e4,
        normal=(-1.0, 0.0),
    )
    model.traction((10.0, 0.0), on=right)
    return model, displacement, contact


def test_rigid_obstacle_contact_contract_is_explicitly_bounded():
    model, displacement, contact = _contact_model()

    capability = contact.capabilities().summary()
    assert capability == {
        "kind": "contact_constraint",
        "enforcement": "one_sided_rigid_plane_penalty",
        "analyses": ("nonlinear_static",),
        "procedures": ("incremental_newton",),
        "strict": False,
        "supports_parallel": True,
        "reaction_evidence": "provider_dual_required",
        "work_evidence": "internal_energy_operator",
    }
    assert contact.summary()["friction"] == "none"
    assert contact.summary()["obstacle"] == "fixed_plane"
    assert model.step(target=displacement, progress=False).contact_provider is contact


def test_rigid_obstacle_contact_closes_force_and_reports_energy():
    model, displacement, _contact = _contact_model()
    result = model.step(target=displacement, progress=False).solve_result()

    expected_displacement = 10.0 / (1.0e3 + 1.0e4)
    right_values = results.probe(displacement, at=(1.0, 0.5))
    # The field is uniform in y and linear in x for this nu=0 patch problem.
    assert right_values[0] == pytest.approx(
        expected_displacement,
        rel=2.0e-6,
    )

    dual = result.metadata["constraint_duals"][0]
    assert dual["role"] == "contact_constraint"
    assert dual["source"] == "rigid_obstacle_penalty_potential"
    assert dual["force_complete"] is True
    assert dual["work_complete"] is True
    np.testing.assert_allclose(dual["coordinate"], (0.0, 0.0), atol=0.0)
    assert dual["diagnostics"]["contact_energy"] > 0.0
    assert dual["diagnostics"]["penetration_l2_norm"] > 0.0
    assert dual["diagnostics"]["active_contact_measure"] == pytest.approx(1.0)
    np.testing.assert_allclose(
        dual["resultant"],
        (-1.0e4 * expected_displacement, 0.0),
        rtol=2.0e-6,
        atol=1.0e-9,
    )
    assert result.quantities["relative_force_balance_error"].value < 1.0e-8
    assert result.metadata["static_work"]["status"] == "unavailable"
    path_work = result.metadata["constraint_path_work"]
    assert path_work["status"] == "complete"
    assert path_work["sample_count"] >= 2
    assert path_work["channels"]["rigid_obstacle_contact"]["value"] == pytest.approx(
        0.0
    )
    assert result.quantities["rigid_obstacle_contact_path_work"].value == pytest.approx(
        0.0
    )
    assert dual["distribution"]["name"] in result.fields


def test_rigid_obstacle_contact_rejects_nonpositive_penalty():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
    )
    model = models.create(
        study=studies.static_solid(
            dimension=2,
            assumption="plane_stress",
            nonlinear=True,
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    right = mesh.boundary(domain, _right, name="right")

    with pytest.raises(ValueError, match="positive"):
        model.rigid_obstacle_contact(
            on=right,
            penalty=0.0,
            normal=(-1.0, 0.0),
        )
    assert displacement is not None


def test_open_rigid_obstacle_has_zero_force_and_energy():
    model, displacement, _contact = _contact_model()
    model.boundary_models.clear()
    right = next(
        region for region in model.regions if region.name == "contact_and_load"
    )
    model.rigid_obstacle_contact(
        on=right,
        penalty=1.0e4,
        normal=(-1.0, 0.0),
        initial_gap=0.1,
        name="open_contact",
    )

    result = model.step(target=displacement, progress=False).solve_result()
    dual = result.metadata["constraint_duals"][0]

    np.testing.assert_allclose(dual["resultant"], (0.0, 0.0), atol=1.0e-12)
    assert dual["diagnostics"]["contact_energy"] == pytest.approx(0.0)
    assert dual["diagnostics"]["penetration_l2_norm"] == pytest.approx(0.0)
    assert dual["diagnostics"]["active_contact_measure"] == pytest.approx(0.0)
    assert result.quantities["relative_force_balance_error"].value < 1.0e-9


def test_rigid_obstacle_requires_unit_normal():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
    )
    model = models.create(
        study=studies.static_solid(
            dimension=2,
            assumption="plane_stress",
            nonlinear=True,
        ),
        mesh=domain,
    )
    right = mesh.boundary(domain, _right, name="right")

    with pytest.raises(ValueError, match="unit vector"):
        model.rigid_obstacle_contact(
            on=right,
            penalty=1.0,
            normal=(-2.0, 0.0),
        )


def test_rigid_obstacle_contact_retains_vector_resultant_in_3d():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True),
        mesh=domain,
        name="rigid_plane_contact_patch_3d",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        constitutive.elasticity.isotropic_elastic(
            young=1.0e3,
            poisson=0.0,
            density=1.0,
        )
    )
    left = mesh.boundary(domain, _left, name="left", tag=1)
    right = mesh.boundary(domain, _right, name="contact_and_load", tag=2)
    model.clamp(displacement, on=left)
    model.rigid_obstacle_contact(
        on=right,
        penalty=1.0e4,
        normal=(-1.0, 0.0, 0.0),
    )
    model.traction((10.0, 0.0, 0.0), on=right)

    result = model.step(target=displacement, progress=False).solve_result()
    dual = result.metadata["constraint_duals"][0]

    expected_displacement = 10.0 / (1.0e3 + 1.0e4)
    np.testing.assert_allclose(
        dual["resultant"],
        (-1.0e4 * expected_displacement, 0.0, 0.0),
        rtol=2.0e-6,
        atol=1.0e-9,
    )
    assert result.quantities["relative_force_balance_error"].value < 1.0e-8

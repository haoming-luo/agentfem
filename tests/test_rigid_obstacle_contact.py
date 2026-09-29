from __future__ import annotations

import json

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import (
    checkpointing,
    constitutive,
    fields,
    mesh,
    models,
    results,
    steps,
    studies,
)


def _left(x):
    return np.isclose(x[0], 0.0)


def _right(x):
    return np.isclose(x[0], 1.0)


def _contact_model(*, traction=10.0, prescribed_y=None):
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
    model.traction((float(traction), 0.0), on=right)
    if prescribed_y is not None:
        model.prescribe(
            displacement,
            float(prescribed_y),
            on=right,
            component=1,
            name="prescribed_right_y",
        )
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
    work = result.metadata["static_work"]
    expected_natural_work = 0.5 * 10.0 * expected_displacement
    expected_bulk_energy = 0.5 * 1.0e3 * expected_displacement**2
    expected_contact_energy = 0.5 * 1.0e4 * expected_displacement**2
    assert work["status"] == "complete"
    assert work["natural_load_work"] == pytest.approx(expected_natural_work)
    assert work["stored_energy_components"][
        "bulk_strain_energy"
    ] == pytest.approx(expected_bulk_energy)
    assert work["stored_energy_components"]["contact_energy"] == pytest.approx(
        expected_contact_energy
    )
    assert work["relative_energy_balance_error"] < 1.0e-12
    assert result.quantity("external_work") == pytest.approx(expected_natural_work)
    assert result.quantity("stored_energy_change") == pytest.approx(
        expected_bulk_energy + expected_contact_energy
    )
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


def test_rigid_contact_energy_fails_closed_for_unrecorded_prescribed_work():
    model, displacement, _contact = _contact_model(prescribed_y=0.01)
    result = model.step(target=displacement, progress=False).solve_result()

    work = result.metadata["static_work"]
    assert work["status"] == "unavailable"
    assert "prescribed-motion work" in work["reason"]
    assert work["prescribed_motion_work"] is None
    assert "external_work" not in result.quantities


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


@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 1,
    reason="cross-rank restart is covered by the portable contact driver",
)
def test_rigid_contact_checkpoint_restores_solution_path_and_duals(tmp_path):
    reference_model, reference_u, _ = _contact_model()
    reference_step = reference_model.step(
        target=reference_u,
        incrementation=steps.fixed(4),
        progress=False,
    )
    reference_step.solve()
    reference_values = reference_u.value.x.array.copy()
    reference_duals = reference_step.constraint_dual_history.checkpoint_state()
    reference_energy = reference_step.accepted_history_recorders[
        "conservative_energy"
    ].checkpoint_state()

    partial_model, partial_u, _ = _contact_model()
    partial = partial_model.step(
        target=partial_u,
        incrementation=steps.fixed(4),
        progress=False,
    )
    partial.solve(until=0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "contact")

    restarted_model, restarted_u, _ = _contact_model()
    restarted = restarted_model.step(
        target=restarted_u,
        incrementation=steps.fixed(4),
        progress=False,
    )
    restarted.load_checkpoint(checkpoint)
    assert restarted.accepted_load_factor == pytest.approx(0.5)
    assert len(restarted.accepted_increments) == 2
    assert len(restarted.constraint_dual_history.records) == 3
    restarted.solve()

    np.testing.assert_allclose(
        restarted_u.value.x.array, reference_values, rtol=0.0, atol=1e-12
    )
    assert restarted.constraint_dual_history.checkpoint_state() == reference_duals
    assert (
        restarted.accepted_history_recorders[
            "conservative_energy"
        ].checkpoint_state()
        == reference_energy
    )
    assert [item.load_factor for item in restarted.accepted_increments] == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]


@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 1,
    reason="manifest corruption is a serial filesystem test",
)
def test_rigid_contact_checkpoint_rejects_corrupt_duals_atomically(tmp_path):
    model, displacement, _ = _contact_model()
    step = model.step(
        target=displacement,
        incrementation=steps.fixed(2),
        progress=False,
    )
    step.solve(until=0.5)
    checkpoint = step.save_checkpoint(tmp_path / "contact-corrupt")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["constraint_dual_history"]["records"][-1]["duals"][0]["force"][0] = float(
        "nan"
    )
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    restarted_model, restarted_u, _ = _contact_model()
    restarted = restarted_model.step(
        target=restarted_u,
        incrementation=steps.fixed(2),
        progress=False,
    )
    before = restarted_u.value.x.array.copy()
    with pytest.raises(ValueError, match="Invalid constraint dual"):
        restarted.load_checkpoint(checkpoint)

    np.testing.assert_array_equal(restarted_u.value.x.array, before)
    assert restarted.accepted_load_factor == pytest.approx(0.0)
    assert restarted.accepted_increments == []
    assert restarted.constraint_dual_history.records == []


@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 1,
    reason="manifest corruption is a serial filesystem test",
)
def test_rigid_contact_checkpoint_rejects_corrupt_energy_history_atomically(tmp_path):
    model, displacement, _ = _contact_model()
    step = model.step(
        target=displacement,
        incrementation=steps.fixed(2),
        progress=False,
    )
    step.solve(until=0.5)
    checkpoint = step.save_checkpoint(tmp_path / "contact-corrupt-energy")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["accepted_observer_state"]["conservative_energy"]["frames"][-1][
        "load_factor"
    ] = 0.4
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    restarted_model, restarted_u, _ = _contact_model()
    restarted = restarted_model.step(
        target=restarted_u,
        incrementation=steps.fixed(2),
        progress=False,
    )
    before = restarted_u.value.x.array.copy()
    with pytest.raises(ValueError, match="does not end at the restored boundary"):
        restarted.load_checkpoint(checkpoint)

    np.testing.assert_array_equal(restarted_u.value.x.array, before)
    assert restarted.accepted_load_factor == pytest.approx(0.0)
    assert restarted.accepted_history_recorders["conservative_energy"].frames == []


@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 1,
    reason="cross-rank restart is covered by the portable contact driver",
)
def test_rigid_contact_automatic_checkpoints_are_result_assets(tmp_path):
    model, displacement, _ = _contact_model()
    step = model.step(
        target=displacement,
        incrementation=steps.fixed(4),
        checkpoint=checkpointing.every(
            2,
            directory=tmp_path,
            final=True,
            portable=True,
        ),
        progress=False,
    )
    result = step.solve_result()

    assert [item.coordinate_value for item in step.checkpoints] == [0.5, 1.0]
    assert len(result.checkpoints) == 2
    assert all(item.path.is_file() for item in step.checkpoints)


@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 1,
    reason="retention is a serial filesystem test",
)
def test_rigid_contact_checkpoint_retention_removes_complete_old_generations(
    tmp_path,
):
    model, displacement, _ = _contact_model()
    step = model.step(
        target=displacement,
        incrementation=steps.fixed(4),
        checkpoint=checkpointing.every(
            1,
            directory=tmp_path,
            keep_last=2,
            portable=True,
        ),
        progress=False,
    )
    step.solve()

    assert [item.coordinate_value for item in step.checkpoints] == [0.75, 1.0]
    manifests = sorted(tmp_path.glob("*.checkpoint.json"))
    payloads = sorted(tmp_path.glob("*.portable.npz"))
    assert [path.name for path in manifests] == [
        item.path.name for item in step.checkpoints
    ]
    assert len(payloads) == 2


@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 1,
    reason="identity mutation is a serial manifest test",
)
def test_rigid_contact_checkpoint_binds_load_identity(tmp_path):
    model, displacement, _ = _contact_model()
    step = model.step(
        target=displacement,
        incrementation=steps.fixed(2),
        progress=False,
    )
    step.solve(until=0.5)
    checkpoint = step.save_checkpoint(tmp_path / "contact-load")

    changed_model, changed_u, _ = _contact_model(traction=11.0)
    changed = changed_model.step(
        target=changed_u,
        incrementation=steps.fixed(2),
        progress=False,
    )
    with pytest.raises(ValueError, match="loads, constraints"):
        changed.load_checkpoint(checkpoint)
    assert changed.accepted_load_factor == pytest.approx(0.0)


@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 1,
    reason="simulated filesystem failure is a serial unit test",
)
def test_checkpoint_publication_failure_rolls_back_accepted_boundary(
    tmp_path,
    monkeypatch,
):
    model, displacement, _ = _contact_model()
    step = model.step(
        target=displacement,
        incrementation=steps.fixed(2),
        checkpoint=checkpointing.every(1, directory=tmp_path),
        progress=False,
    )

    def fail_checkpoint(*_args, **_kwargs):
        raise OSError("simulated checkpoint failure")

    monkeypatch.setattr(step, "save_checkpoint", fail_checkpoint)
    with pytest.raises(OSError, match="simulated checkpoint failure"):
        step.solve()

    assert step.accepted_load_factor == pytest.approx(0.0)
    assert step.accepted_increments == []
    assert step.attempted_increments == []
    assert len(step.snapshots) == 1
    assert step.checkpoints == []
    assert len(step.constraint_dual_history.records) == 1
    assert step.constraint_dual_history.records[0]["load_factor"] == pytest.approx(0.0)
    assert [event.kind for event in step.execution_events] == ["step_started"]
    energy_frames = step.accepted_history_recorders["conservative_energy"].frames
    assert len(energy_frames) == 1
    assert energy_frames[0].load_factor == pytest.approx(0.0)

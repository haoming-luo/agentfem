"""Public standard-boundary finite-strain J2 workflow contracts."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from basix.ufl import element
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh as dolfinx_mesh
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI

from agentfem import (
    amplitudes,
    boundary_models,
    checkpointing,
    constitutive,
    constraints,
    coordinates,
    fields,
    loads,
    mesh,
    models,
    results,
    solvers,
    steps,
    studies,
)
from agentfem.step_providers import step_capability


def _standard_patch(
    *,
    incrementation,
    amplitude=None,
    output=None,
    checkpoint=None,
    progress=False,
    status_file=None,
    solver_options=None,
):
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="public_finite_strain_j2_standard_patch",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1
    )
    right = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2
    )
    y_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="y_symmetry", tag=3
    )
    z_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[2], 0.0), name="z_symmetry", tag=4
    )
    model.fix(displacement, on=left, component=0, value=0.0)
    model.fix(displacement, on=y_symmetry, component=1, value=0.0)
    model.fix(displacement, on=z_symmetry, component=2, value=0.0)
    model.fix(displacement, on=right, component=0, value=0.02)
    material = constitutive.finite_strain_j2_logarithmic(
        young=200_000.0,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2_000.0,
    )
    material_record = model.material(material)
    step = model.step(
        target=displacement,
        material=material_record,
        incrementation=incrementation,
        solver_options=solver_options or solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        amplitude=amplitude,
        output=output,
        checkpoint=checkpoint,
        progress=progress,
        status_file=status_file,
    )
    return model, displacement, material, step


def test_corrected_inner_linear_policy_rejects_legacy_transport_identity(tmp_path, monkeypatch):
    policy = solvers.newton(linear_solver=solvers.LinearSolverOptions(
        ksp_type="gmres", pc_type="ilu", rtol=1e-12, atol=1e-14, max_it=200,
    ))
    _, _, _, partial = _standard_patch(incrementation=steps.fixed(4), solver_options=policy)
    partial.solve(until=.5)
    correct = partial.save_checkpoint(tmp_path / "correct_linear_transport")
    original = solvers.NewtonSolverOptions.summary
    def legacy_summary(self):
        record = original(self)
        record.pop("linear_tolerance_transport", None)
        return record
    # Simulate only the old identity schema, not an old numerical solution.
    with monkeypatch.context() as patch:
        patch.setattr(solvers.NewtonSolverOptions, "summary", legacy_summary)
        legacy = partial.save_checkpoint(tmp_path / "legacy_linear_transport")
    _, _, _, resumed = _standard_patch(incrementation=steps.fixed(4), solver_options=policy)
    initial = resumed.solution.x.array.copy()
    with pytest.raises(ValueError, match="identity"):
        resumed.load_checkpoint(legacy)
    np.testing.assert_array_equal(resumed.solution.x.array, initial)
    assert resumed.accepted_load_factor == 0.
    resumed.load_checkpoint(correct)
    resumed.solve()
    partial.solve()
    np.testing.assert_allclose(resumed.solution.x.array, partial.solution.x.array,
                               atol=2e-10, rtol=2e-8)
    np.testing.assert_allclose(resumed.response.state.committed_state_vectors(),
                               partial.response.state.committed_state_vectors(),
                               atol=2e-10, rtol=2e-8)


def _j2_contact_patch(*, incrementation=None, penalty=500.0,
                      yield_stress=2.0):
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics", dimension=3,
        ),
        mesh=domain,
        name="finite_j2_fixed_plane_contact",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    model.clamp(displacement, on=left)
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=1_000.0,
            poisson=0.0,
            yield_stress=yield_stress,
            hardening_modulus=100.0,
        )
    )
    contact = model.rigid_obstacle_contact(
        on=right,
        penalty=penalty,
        normal=(-1.0, 0.0, 0.0),
    )
    model.traction((10.0, 0.0, 0.0), on=right)
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(3) if incrementation is None else incrementation,
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=25,
            line_search="backtracking",
        ),
        progress=False,
    )
    return model, displacement, material, contact, step


def test_public_j2_fixed_plane_contact_uses_one_newton_and_energy_ledger():
    model, displacement, material, contact, step = _j2_contact_patch()
    capability = step_capability(
        model, target=displacement, options={"material": material},
    )
    assert capability["supported"]
    assert capability["provider"]["name"] == "finite_strain_j2_strong_static"

    assert step.contact_provider is contact
    result = step.solve_result()
    assert step.last_solve_info.converged
    assert max(frame.contact_potential for frame in step.energy_history) > 0.0
    assert step.energy_history[-1].internal_energy == pytest.approx(
        step.energy_history[-1].stored_energy
        + step.energy_history[-1].plastic_dissipation
        + step.energy_history[-1].contact_potential
    )
    assert result.histories["contact_potential"].values[-1] > 0.0
    assert result.metadata["constraint_duals"][0]["constraint_name"] == contact.name
    assert result.metadata["constraint_balance_contract"]["force_balance_available"]
    assert result.metadata["constraint_balance_contract"]["work_balance_available"]
    assert result.metadata["constraint_work_evidence"][0]["value"] == 0.0
    assert result.quantity(f"{contact.name}_contact_energy") > 0.0
    assert np.max(step.response.state.committed_state_vectors()[:, -2]) > 0.0
    strong = np.sum(
        result.fields["RF"].field.x.array.reshape((-1, 3)), axis=0
    )
    contact_resultant = np.asarray(
        result.metadata["constraint_duals"][0]["resultant"], dtype=float
    )
    np.testing.assert_allclose(
        strong + contact_resultant + np.array([10.0, 0.0, 0.0]),
        0.0,
        rtol=0.0, atol=1.0e-6,
    )


def test_public_j2_contact_default_result_output_keeps_nodal_primary(tmp_path):
    _, _, _, _, step = _j2_contact_patch()
    path = tmp_path / "fields.xdmf"

    result = step.solve_result(output=path, strict_output=True)

    assert result.status == "completed"
    assert path.is_file()
    assert path.with_suffix(".h5").is_file()
    assert (tmp_path / "fields.integration-points.h5").is_file()
    assert result.metadata["field_output_fields"]["included"][0] == "Displacement"
    grids = ET.parse(path).findall(".//Grid[@GridType='Uniform']")
    assert len(grids) == 1
    names = {item.attrib["Name"] for item in grids[0].findall("Attribute")}
    assert {"U", "UMAG", "PEEQ_CELL"} <= names


@pytest.mark.parametrize("penalty", (250.0, 500.0, 1_000.0))
def test_public_j2_contact_elastic_limit_matches_analytical_spring_pair(penalty):
    from scipy.optimize import brentq

    _, displacement, _, _, step = _j2_contact_patch(
        penalty=penalty, yield_stress=1.0e9
    )
    step.solve()
    right = results.probe(displacement, at=(1.0, 0.5, 0.5))
    # For nu=0, homogeneous uniaxial Hencky elasticity has
    # P = E log(1+u)/(1+u); the fixed-plane penalty contributes k*u.
    expected = brentq(
        lambda u: 1_000.0 * np.log1p(u) / (1.0 + u) + penalty * u - 10.0,
        0.0, 0.05,
    )
    assert right[0] == pytest.approx(expected, rel=2.0e-5)
    assert np.max(step.response.state.committed_state_vectors()[:, -2]) == pytest.approx(
        0.0, abs=1.0e-12,
    )


def test_j2_contact_two_disconnected_columns_have_distinct_analytical_active_sets():
    from scipy.optimize import brentq

    base = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    original = np.asarray(base.geometry.x, dtype=float)
    second = original.copy()
    second[:, 0] *= 0.99
    second[:, 1] += 2.0
    connectivity = np.asarray(base.geometry.dofmaps[0], dtype=np.int64)
    domain = mesh.from_arrays(
        cells=np.vstack((connectivity, connectivity + len(original))),
        coordinates=np.vstack((original, second)),
        coordinate_element=ufl.Mesh(
            element("Lagrange", "tetrahedron", 1, shape=(3,))
        ),
        comm=MPI.COMM_SELF,
    )
    model = models.create(
        study=studies.nonlinear_static(physics="solid_mechanics", dimension=3),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0) | np.isclose(x[0], 0.99),
        name="both_right_faces",
    )
    model.clamp(displacement, on=left)
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=1_000.0,
            poisson=0.0,
            yield_stress=1.0e9,
            hardening_modulus=100.0,
        )
    )
    contact = model.rigid_obstacle_contact(
        on=right,
        penalty=500.0,
        surface=boundary_models.rigid_plane(
            point=(1.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0)
        ),
    )
    model.traction((5.0, 0.0, 0.0), on=right)
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(4),
        progress=False,
    )
    result = step.solve_result()
    first = results.probe(displacement, at=(1.0, 0.5, 0.5))[0]
    second_value = results.probe(displacement, at=(0.99, 2.5, 0.5))[0]
    first_reference = brentq(
        lambda u: 1_000.0 * np.log1p(u) / (1.0 + u) + 500.0 * u - 5.0,
        0.0, 0.02,
    )
    second_reference = brentq(
        lambda u: 1_000.0 * np.log1p(u / 0.99) / (1.0 + u / 0.99) - 5.0,
        0.0, 0.02,
    )
    assert first == pytest.approx(first_reference, rel=2.0e-5)
    assert second_value == pytest.approx(second_reference, rel=2.0e-5)
    assert second_value < 0.01  # The short column never reaches the tool.
    assert result.metadata["constraint_duals"][0]["constraint_name"] == contact.name
    assert result.quantity(f"{contact.name}_contact_energy") > 0.0
    dual = contact.dual_evidence(step)
    nodal_coordinates = displacement.value.function_space.tabulate_dof_coordinates()
    nodal_force = dual.distribution.x.array.reshape((-1, 3))
    np.testing.assert_allclose(
        nodal_force[nodal_coordinates[:, 1] > 1.5], 0.0, rtol=0.0, atol=1.0e-12
    )


def test_public_j2_contact_restart_preserves_state_and_potential(tmp_path):
    _, _, _, _, uninterrupted = _j2_contact_patch()
    uninterrupted.solve()
    _, _, _, _, partial = _j2_contact_patch()
    partial.solve(until=1.0 / 3.0)
    saved = partial.save_checkpoint(tmp_path / "contact_restart")
    _, _, _, _, resumed = _j2_contact_patch()
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, uninterrupted.solution.x.array,
        rtol=2.0e-8, atol=2.0e-10,
    )
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        uninterrupted.response.state.committed_state_vectors(),
        rtol=2.0e-8, atol=2.0e-10,
    )
    assert resumed.energy_history[-1].contact_potential == pytest.approx(
        uninterrupted.energy_history[-1].contact_potential,
        rel=2.0e-8,
    )

    _, _, _, _, wrong_contact = _j2_contact_patch(penalty=600.0)
    with pytest.raises(ValueError, match="identity"):
        wrong_contact.load_checkpoint(saved)
    assert wrong_contact.accepted_load_factor == pytest.approx(0.0)
    assert not wrong_contact.energy_history


def test_public_j2_contact_energy_failure_rolls_back_joint_state(monkeypatch):
    _, _, _, _, step = _j2_contact_patch()
    step.solve(until=1.0 / 3.0)
    accepted_u = step.solution.x.array.copy()
    accepted_state = step.response.state.committed_state_vectors().copy()
    accepted_energy = tuple(step.energy_history)

    def fail_contact_energy():
        raise ValueError("intentional contact energy failure")

    monkeypatch.setattr(step, "_contact_potential", fail_contact_energy)
    with pytest.raises(RuntimeError, match="intentional contact energy failure"):
        step.solve(until=2.0 / 3.0)
    assert step.accepted_load_factor == pytest.approx(1.0 / 3.0)
    np.testing.assert_allclose(step.solution.x.array, accepted_u)
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(), accepted_state,
    )
    assert tuple(step.energy_history) == accepted_energy


def _j2_moving_contact_patch(*, rotation=0.0, cells=1, amplitude=None):
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, cells, cells, cells
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics", dimension=3,
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    model.clamp(displacement, on=left)
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=1_000.0, poisson=0.0, yield_stress=1.0e9,
            hardening_modulus=100.0,
        )
    )
    contact = model.rigid_obstacle_contact(
        on=right,
        penalty=500.0,
        surface=boundary_models.rigid_plane(
            point=(1.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        ),
        motion=boundary_models.prescribed_rigid_motion(
            translation=(-0.01, 0.0, 0.0),
            rotation=rotation,
            reference_point=(1.0, 0.5, 0.5),
        ),
    )
    capability = step_capability(
        model, target=displacement, options={"material": material},
    )
    assert capability["supported"]
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(6),
        amplitude=amplitude,
        progress=False,
    )
    return model, displacement, material, contact, step


def test_public_j2_prescribed_plane_records_tool_work():
    from scipy.optimize import brentq

    _, displacement, _, _, step = _j2_moving_contact_patch()
    result = step.solve_result()
    right_u = results.probe(displacement, at=(1.0, 0.5, 0.5))[0]
    expected = brentq(
        lambda u: 1_000.0 * np.log1p(u) / (1.0 + u)
        + 500.0 * (u + 0.01),
        -0.01, 0.0,
    )
    assert right_u == pytest.approx(expected, rel=2.0e-5)
    assert result.histories["tool_work"].values[-1] > 0.0
    assert step.energy_history[-1].tool_work == pytest.approx(
        result.histories["tool_work"].values[-1]
    )
    assert abs(step.energy_history[-1].mechanical_energy_residual) / (
        step.energy_history[-1].external_work
    ) < 0.01
    assert result.metadata["constraint_work_evidence"][0]["value"] == pytest.approx(
        step.energy_history[-1].tool_work
    )
    assert np.linalg.norm(result.metadata["constraint_duals"][0]["force"]) > 0.0


def test_public_j2_tool_uses_shared_nonlinear_step_amplitude():
    _, _, _, contact, step = _j2_moving_contact_patch(
        amplitude=amplitudes.tabular(
            (0.0, 0.5, 1.0), (0.0, 0.25, 1.0)
        )
    )
    step.solve(until=0.5)
    assert float(np.asarray(contact.motion_factor.value).reshape(-1)[0]) == pytest.approx(0.25)
    assert float(np.asarray(contact.surface_point.value).reshape(-1)[0]) == pytest.approx(0.9975)
    step.solve()
    assert float(np.asarray(contact.motion_factor.value).reshape(-1)[0]) == pytest.approx(1.0)
    assert float(np.asarray(contact.surface_point.value).reshape(-1)[0]) == pytest.approx(0.99)


def test_j2_rotating_tool_force_is_conjugate_to_contact_potential():
    _, _, _, contact, step = _j2_moving_contact_patch(
        rotation=(0.0, 0.0, 0.02), cells=2
    )
    result = step.solve_result()
    energy_at_end = step._contact_potential()
    increment = 1.0e-6
    try:
        contact.update_motion(1.0 - increment)
        energy_before_end = step._contact_potential()
    finally:
        contact.update_motion(1.0)
    generalized_force = np.asarray(
        result.metadata["constraint_duals"][0]["force"], dtype=float
    )
    path_direction = contact.motion.generalized_coordinate(1.0)
    assert (energy_at_end - energy_before_end) / increment == pytest.approx(
        float(np.dot(generalized_force, path_direction)), rel=2.0e-4
    )


def test_j2_rotating_plane_tool_work_refines_on_multicell_patch():
    # A tilted plane gives nonuniform face contact. This is a spatial
    # self-consistency check, not an independent curved-tool reference.
    work = []
    for cells in (1, 2, 3):
        _, _, _, _, step = _j2_moving_contact_patch(
            rotation=(0.0, 0.0, 0.02), cells=cells
        )
        step.solve()
        frame = step.energy_history[-1]
        assert np.isfinite(frame.mechanical_energy_residual)
        work.append(frame.tool_work)
    assert abs(work[2] - work[1]) < abs(work[1] - work[0])


def test_public_j2_moving_tool_restart_and_rollback(monkeypatch, tmp_path):
    _, _, _, _, reference = _j2_moving_contact_patch()
    reference.solve()
    _, _, _, contact, partial = _j2_moving_contact_patch()
    partial.solve(until=1.0 / 3.0)
    accepted_u = partial.solution.x.array.copy()
    accepted_state = partial.response.state.committed_state_vectors().copy()
    accepted_point = np.asarray(contact.surface_point.value).copy()
    saved = partial.save_checkpoint(tmp_path / "moving_contact")

    def fail_contact_energy():
        raise ValueError("intentional moving contact failure")

    monkeypatch.setattr(partial, "_contact_potential", fail_contact_energy)
    with pytest.raises(RuntimeError, match="intentional moving contact failure"):
        partial.solve(until=0.5)
    assert partial.accepted_load_factor == pytest.approx(1.0 / 3.0)
    np.testing.assert_allclose(partial.solution.x.array, accepted_u)
    np.testing.assert_allclose(
        partial.response.state.committed_state_vectors(), accepted_state,
    )
    np.testing.assert_allclose(contact.surface_point.value, accepted_point)

    _, _, _, restored_contact, restored = _j2_moving_contact_patch()
    restored.load_checkpoint(saved)
    np.testing.assert_allclose(restored_contact.surface_point.value, accepted_point)
    restored.solve()
    np.testing.assert_allclose(restored.solution.x.array, reference.solution.x.array)
    np.testing.assert_allclose(
        restored.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
    )
    assert restored.energy_history[-1].tool_work == pytest.approx(
        reference.energy_history[-1].tool_work,
        rel=2.0e-8,
    )


def test_public_j2_rotating_plane_records_force_and_moment():
    _, _, _, contact, step = _j2_moving_contact_patch(rotation=(0.0, 0.0, 0.02))
    result = step.solve_result()
    dual = result.metadata["constraint_duals"][0]
    assert np.isfinite(result.histories["tool_work"].values[-1])
    assert result.histories["tool_work"].values[-1] > 0.0
    assert abs(dual["diagnostics"]["contact_moment"][2]) > 1.0e-6
    assert abs(np.asarray(contact.normal.value)[1]) > 1.0e-3


def test_public_standard_j2_provider_result_output_progress_and_checkpoint(tmp_path):
    output = results.output_plan(
        tmp_path / "output",
        field=results.field_output(
            "U",
            "F",
            "S",
            "MISES",
            "SENER",
            "PEEQ",
            every=1,
            configuration="reference",
        ),
        requests=(
            results.probe_history(
                "right_ux",
                at=(1.0, 0.0, 0.0),
                component=0,
            ),
        ),
        presentation=None,
        basename="standard_j2",
    )
    loading = amplitudes.tabular(
        (0.0, 0.5, 1.0),
        (0.0, 0.25, 1.0),
        name="quadratic_like_ramp",
    )
    model, displacement, material, step = _standard_patch(
        incrementation=steps.fixed(4),
        amplitude=loading,
        output=output,
        checkpoint=checkpointing.every(
            2,
            directory=tmp_path / "checkpoints",
        ),
        progress=True,
        status_file=tmp_path / "status.log",
    )
    capability = step_capability(
        model,
        target=displacement,
        options={"material": material},
    )
    assert capability["supported"]
    assert capability["provider"]["name"] == "finite_strain_j2_strong_static"

    result = step.solve_result()

    assert step.last_solve_info.converged
    assert step.summary()["maturity"] == "experimental_global_mpi_restart"
    assert step.summary()["evidence_level"] == "internal_serial_mpi_restart_verified"
    assert len(step.snapshots) == 5
    assert len(step.checkpoints) == 2
    assert [item.coordinate_value for item in step.checkpoints] == pytest.approx(
        [0.5, 1.0]
    )
    np.testing.assert_allclose(
        result.histories["load_amplitude"].values,
        (0.0, 0.125, 0.25, 0.625, 1.0),
    )
    np.testing.assert_allclose(
        result.histories["right_ux"].values,
        (0.0, 0.0025, 0.005, 0.0125, 0.02),
        atol=2.0e-10,
    )
    assert {
        "Displacement",
        "F",
        "P",
        "S",
        "MISES",
        "SENER",
        "ELENER",
        "HARDENER",
        "FP",
        "PEEQ",
        "PDENER",
        "RF",
    } <= set(result.fields)
    assert result.fields["PEEQ"].location == "quadrature_points"
    state = step.response.state.committed_state_vectors()
    assert np.max(state[:, -2]) > 0.0
    assert np.max(state[:, -1]) > 0.0
    assert result.quantity("plastic_dissipation") > 0.0
    np.testing.assert_allclose(
        result.fields["PDENER"].field.x.array,
        step.response.state.committed["plastic_dissipation"].function.x.array,
    )
    for plastic_gradient in state[:, :9].reshape((-1, 3, 3)):
        assert np.linalg.det(plastic_gradient) == pytest.approx(
            1.0,
            abs=5.0e-10,
        )
    event_names = tuple(
        item["kind"] for item in result.metadata["execution"]["events"]
    )
    assert event_names.count("step_started") == 1
    assert event_names.count("increment_converged") == 4
    assert event_names.count("step_completed") == 1
    assert (tmp_path / "output" / "standard_j2.xdmf").is_file()
    assert (tmp_path / "output" / "standard_j2.h5").is_file()
    assert (
        tmp_path / "output" / "standard_j2.integration-points.h5"
    ).is_file()
    assert result.artifacts["integration_points_hdf5"].is_file()
    assert result.metadata["output_plan"]["status"] == "completed"
    assert len(step.energy_history) == 4
    assert step.energy_history[-1].prescribed_motion_work > 0.0
    assert step.energy_history[-1].natural_load_work == pytest.approx(0.0)
    assert result.metadata["mechanical_energy_ledger"]["status"] == (
        "complete_accepted_path"
    )
    assert {
        "stored_energy",
        "plastic_dissipation",
        "internal_energy",
        "prescribed_motion_work",
        "external_work",
        "mechanical_energy_residual",
    } <= set(result.histories)
    status = (tmp_path / "status.log").read_text(encoding="utf-8")
    assert "CONVERGED" in status
    assert "COMPLETED" in status


@pytest.mark.parametrize("portable", [False, True])
def test_standard_j2_checkpoint_retention_uses_record_schema(tmp_path, portable):
    directory = tmp_path / ("portable" if portable else "serial")
    _, _, _, step = _standard_patch(
        incrementation=steps.fixed(3),
        checkpoint=checkpointing.every(
            1,
            directory=directory,
            keep_last=2,
            portable=portable,
        ),
    )

    step.solve()

    assert len(step.checkpoints) == 2
    assert [item.coordinate_value for item in step.checkpoints] == pytest.approx(
        [2.0 / 3.0, 1.0]
    )
    assert not tuple(directory.glob("*00000001*"))
    assert tuple(directory.glob("*00000002*"))
    assert tuple(directory.glob("*00000003*"))


def test_standard_j2_real_cutback_and_manual_restart_are_equivalent(tmp_path):
    automatic = steps.automatic(
        initial=1.0,
        minimum=1.0e-4,
        maximum=1.0,
        max_increments=100,
        max_cutbacks=12,
        cutback_factor=0.5,
        maximum_inelastic_increment=0.006,
    )
    _, _, _, cutback = _standard_patch(incrementation=automatic)
    cutback.solve()
    rejected = [item for item in cutback.attempted_increments if not item.converged]
    assert rejected
    assert "maximum equivalent plastic-strain increment" in (
        rejected[0].rejection_reason or ""
    )

    _, _, _, reference = _standard_patch(incrementation=steps.fixed(4))
    reference.solve()
    expected_u = reference.solution.x.array.copy()
    expected_state = reference.response.state.committed_state_vectors().copy()
    expected_p = reference.response.first_piola_stress.values.copy()

    _, _, _, partial = _standard_patch(incrementation=steps.fixed(4))
    partial.solve(until=0.5)
    assert partial.execution_events[-1].kind == "step_paused"
    saved = partial.save_checkpoint(tmp_path / "manual_restart")
    _, _, _, restarted = _standard_patch(incrementation=steps.fixed(4))
    restarted.load_checkpoint(saved)
    restarted.solve()
    resumed_events = tuple(item.kind for item in restarted.execution_events)
    assert "step_paused" in resumed_events
    assert "step_resumed" in resumed_events
    assert resumed_events.index("step_resumed") > resumed_events.index("step_paused")
    assert resumed_events[-1] == "step_completed"

    np.testing.assert_allclose(restarted.solution.x.array, expected_u, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(
        restarted.response.state.committed_state_vectors(),
        expected_state,
        rtol=2e-8,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        restarted.response.state.committed_state_vectors()[:, -1],
        reference.response.state.committed_state_vectors()[:, -1],
        rtol=2e-8,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        restarted.response.first_piola_stress.values,
        expected_p,
        rtol=2e-8,
        atol=2e-8,
    )
    assert len(restarted.energy_history) == len(reference.energy_history)
    for name in restarted.energy_history[0].__dataclass_fields__:
        actual = [getattr(item, name) for item in restarted.energy_history]
        expected = [getattr(item, name) for item in reference.energy_history]
        if any(value is None for value in actual + expected):
            # Unavailable evidence must remain unavailable after restart;
            # it is not a numeric zero and must not enter a floating compare.
            assert actual == expected
            continue
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2.0e-10,
            atol=2.0e-12,
        )


def test_public_standard_j2_consumes_registered_reference_body_force():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="finite_strain_j2_body_force",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="clamp", tag=1
    )
    model.fix(displacement, on=left, value=(0.0, 0.0, 0.0))
    material = constitutive.finite_strain_j2_logarithmic(
        young=1_000.0,
        poisson=0.3,
        yield_stress=1_000.0,
        hardening_modulus=10.0,
    )
    model.material(material)
    load = model.body_force(
        (0.0, 0.0, -1.0),
        target=displacement,
        name="dead_reference_body_force",
    )
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-10,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )
    result = step.solve_result()

    assert step.last_solve_info.converged
    assert np.all(np.isfinite(displacement.value.x.array))
    assert np.min(displacement.value.x.array) < 0.0
    reaction = result.fields["RF"].field.x.array.reshape((-1, 3)).sum(axis=0)
    np.testing.assert_allclose(reaction, (0.0, 0.0, 1.0), atol=2.0e-8)
    assert step.summary()["external_load"] == step.load_identity
    assert step.load_identity[0]["name"] == load.name
    assert step.energy_history[-1].natural_load_work > 0.0
    assert step.energy_history[-1].prescribed_motion_work == pytest.approx(0.0)


def test_public_standard_j2_lowers_registered_material_regions():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="regional_finite_strain_j2_standard_patch",
    )
    displacement = model.field(fields.displacement(domain))
    left_boundary = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="left"
    )
    right_boundary = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.0), name="right"
    )
    y_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="y_symmetry"
    )
    z_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[2], 0.0), name="z_symmetry"
    )
    model.fix(displacement, on=left_boundary, component=0, value=0.0)
    model.fix(displacement, on=y_symmetry, component=1, value=0.0)
    model.fix(displacement, on=z_symmetry, component=2, value=0.0)
    model.fix(displacement, on=right_boundary, component=0, value=0.01)
    regions = mesh.partition_cells(
        domain,
        soft=lambda x: x[0] <= 0.5,
        hard=lambda x: x[0] > 0.5,
    )
    model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=120_000.0,
            poisson=0.3,
            yield_stress=100.0,
            hardening_modulus=1_000.0,
        ),
        region=regions.soft,
    )
    model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=250.0,
            hardening_modulus=2_000.0,
        ),
        region=regions.hard,
    )

    capability = step_capability(model, target=displacement)
    assert capability["supported"]
    assert capability["provider"]["name"] == "finite_strain_j2_strong_static"
    step = model.step(
        target=displacement,
        incrementation=steps.fixed(2),
        progress=False,
    )
    step.solve()

    assert isinstance(step.material, constitutive.QuadratureMaterialMap)
    assert step.last_solve_info.converged
    assert np.max(step.response.state.committed_state_vectors()[:, -2]) > 0.0


def test_standard_j2_remote_displacement_uses_the_shared_value_path():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="finite_strain_j2_remote_displacement",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    y_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="y_symmetry"
    )
    z_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[2], 0.0), name="z_symmetry"
    )
    model.fix(displacement, on=left, component=0, value=0.0)
    model.fix(displacement, on=y_symmetry, component=1, value=0.0)
    model.fix(displacement, on=z_symmetry, component=2, value=0.0)
    remote = model.remote_displacement(
        displacement,
        reference_point=coordinates.reference_point((1.0, 0.0, 0.0)),
        on=right,
        translation=(0.02, 0.0, 0.0),
    )
    reference = remote.reference_values.copy()
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2_000.0,
        )
    )
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.at(0.5, 1.0),
        amplitude=amplitudes.tabular(
            (0.0, 0.5, 1.0),
            (0.0, 0.25, 1.0),
        ),
        progress=False,
    )

    step.solve(until=0.5)

    np.testing.assert_allclose(remote.value.x.array, 0.25 * reference)
    assert step.summary()["accepted_load_factor"] == pytest.approx(0.5)
    assert step.value_path.summary()["field_values"] == 1


def test_standard_j2_constraint_capability_sees_nested_periodic_assets():
    model, displacement, material, _step = _standard_patch(
        incrementation=steps.fixed(1)
    )
    ordinary = tuple(model.constraints)
    periodic = constraints.PeriodicConstraintSpec(
        slave_marker=lambda x: np.isclose(x[0], 1.0),
        master_marker=lambda x: np.isclose(x[0], 0.0),
        map_slave_to_master=lambda x: x,
    )
    nested = constraints.ConstraintSet(
        dirichlet=list(ordinary),
        periodic=[periodic],
    )

    assets = constraints.constraint_assets((nested,))
    assert assets[-1] is periodic
    assert len(assets) == len(ordinary) + 1
    capability = step_capability(
        model,
        target=displacement,
        options={"material": material, "constraints": nested},
    )
    assert not capability["supported"]


def test_standard_j2_follower_pressure_has_external_tangent_and_path_work():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="finite_strain_j2_follower_pressure",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="right_pressure",
    )
    model.fix(displacement, on=left, value=(0.0, 0.0, 0.0))
    model.pressure(
        5.0,
        on=right,
        configuration="current",
        displacement=displacement,
    )
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=1_000.0,
            poisson=0.3,
            yield_stress=10_000.0,
            hardening_modulus=10.0,
        )
    )

    capability = step_capability(
        model,
        target=displacement,
        options={"material": material},
    )
    assert capability["supported"]
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(10),
        progress=False,
    )
    result = step.solve_result()

    assert step.last_solve_info.converged
    assert np.min(displacement.value.x.array) < 0.0
    energy = step.energy_history[-1]
    assert energy.natural_load_work > 0.0
    assert energy.prescribed_motion_work == pytest.approx(0.0)
    energy_scale = max(abs(energy.external_work), abs(energy.internal_energy))
    assert abs(energy.mechanical_energy_residual) / energy_scale < 2.0e-3
    assert result.metadata["mechanical_energy_ledger"]["follower_loads"] == (
        "consistent_external_tangent_and_current_configuration_path_work"
    )

    base = step.solution.x.array.copy()
    direction = np.sin(np.arange(base.size, dtype=float) + 0.5)
    external = step.load_factor * step.external_force.expression
    external_tangent = fem.form(
        ufl.derivative(external, step.solution, displacement.trial)
    )
    matrix = fem_petsc.assemble_matrix(external_tangent)
    matrix.assemble()
    direction_vector = step.solution.x.petsc_vec.duplicate()
    direction_vector.array[:] = direction
    tangent_action = direction_vector.duplicate()
    matrix.mult(direction_vector, tangent_action)

    def assembled_external(alpha):
        step.solution.x.array[:] = base + alpha * direction
        step.solution.x.scatter_forward()
        return step._assembled_external_force_values()

    increment = 1.0e-7
    finite_difference = (
        assembled_external(increment) - assembled_external(-increment)
    ) / (2.0 * increment)
    np.testing.assert_allclose(
        tangent_action.array_r,
        finite_difference,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    step.solution.x.array[:] = base
    step.solution.x.scatter_forward()
    tangent_action.destroy()
    direction_vector.destroy()
    matrix.destroy()


def test_standard_j2_load_capability_sees_follower_inside_public_containers():
    model, displacement, material, _step = _standard_patch(
        incrementation=steps.fixed(1)
    )
    right = mesh.boundary(
        displacement.value.function_space.mesh,
        lambda x: np.isclose(x[0], 1.0),
        name="wrapped_right_pressure",
    )
    follower = loads.pressure(
        1.0,
        on=right,
        configuration="current",
        displacement=displacement,
    )
    driven = loads.with_amplitude(
        follower,
        amplitudes.ramp(),
        domain=displacement.value.function_space.mesh,
    )
    model.load(loads.LoadSet.create(driven))

    visible = loads.load_assets(model.loads)
    physical = loads.load_assets(model.loads, unwrap_amplitudes=True)
    assert visible == (driven,)
    assert physical == (follower,)
    assert physical[0].configuration == "current"

    capability = step_capability(
        model,
        target=displacement,
        options={"material": material},
    )
    assert capability["supported"]


def test_standard_j2_rolls_back_if_accepted_state_finalization_fails(monkeypatch):
    _, _, _, step = _standard_patch(
        incrementation=steps.automatic(
            initial=0.5,
            minimum=0.05,
            maximum=0.5,
            max_increments=10,
        )
    )
    initial_solution = step.solution.x.array.copy()
    initial_state = step.response.state.committed_state_vectors().copy()

    def fail_checkpoint():
        raise OSError("intentional checkpoint failure")

    monkeypatch.setattr(step, "_write_scheduled_checkpoint", fail_checkpoint)
    with pytest.raises(RuntimeError, match="intentional checkpoint failure"):
        step.solve()

    assert step.accepted_load_factor == pytest.approx(0.0)
    assert not step.accepted_increments
    assert not step.energy_history
    assert step.next_increment_size is None
    np.testing.assert_allclose(step.solution.x.array, initial_solution)
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(), initial_state
    )
    assert step.execution_events[-1].kind == "step_failed"


def test_standard_j2_work_ledger_converges_with_path_refinement():
    _, _, _, coarse = _standard_patch(incrementation=steps.fixed(4))
    coarse_result = coarse.solve_result()
    _, _, _, fine = _standard_patch(incrementation=steps.fixed(40))
    fine_result = fine.solve_result()

    coarse_error = coarse_result.quantity("relative_mechanical_energy_residual")
    fine_error = fine_result.quantity("relative_mechanical_energy_residual")
    assert fine_error < coarse_error
    assert fine_error < 2.0e-5
    assert fine.energy_history[-1].internal_energy == pytest.approx(
        coarse.energy_history[-1].internal_energy,
        rel=2.0e-9,
    )


def test_standard_j2_loading_update_failure_is_atomic(monkeypatch):
    _, _, _, step = _standard_patch(incrementation=steps.fixed(2))
    initial_solution = step.solution.x.array.copy()
    initial_state = step.response.state.committed_state_vectors().copy()
    original_update = step.value_path.update
    calls = 0

    def fail_second_update(factor):
        nonlocal calls
        calls += 1
        if calls == 2:
            constant, _reference = step.value_path.constants[-1]
            constant.value = 123.0
            raise ValueError("intentional amplitude update failure")
        original_update(factor)

    monkeypatch.setattr(step.value_path, "update", fail_second_update)
    with pytest.raises(RuntimeError, match="loading update failed"):
        step.solve()

    assert step.accepted_load_factor == pytest.approx(0.0)
    np.testing.assert_allclose(step.solution.x.array, initial_solution)
    np.testing.assert_allclose(
        step.response.state.committed_state_vectors(), initial_state
    )
    restored_loading = step._snapshot_loading_state()
    np.testing.assert_allclose(restored_loading["load_factor"], 0.0)
    for value in restored_loading["prescribed_values"]["constants"]:
        np.testing.assert_allclose(value, 0.0)
    assert step.execution_events[-1].kind == "step_failed"


def test_standard_j2_serial_checkpoint_restore_is_atomic(monkeypatch, tmp_path):
    _, _, _, source = _standard_patch(incrementation=steps.fixed(2))
    source.solve(until=0.5)
    saved = source.save_checkpoint(tmp_path / "valid")

    _, _, _, receiver = _standard_patch(incrementation=steps.fixed(2))
    initial_solution = receiver.solution.x.array.copy()
    initial_accepted = receiver.accepted_solution.x.array.copy()
    initial_state = receiver.state_transaction.snapshot_runtime_state()
    original_restore = receiver.response.restore
    calls = 0

    def fail_once(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("intentional state restore failure")
        original_restore(state)

    monkeypatch.setattr(receiver.response, "restore", fail_once)
    with pytest.raises(ValueError, match="intentional state restore failure"):
        receiver.load_checkpoint(saved)

    assert receiver.accepted_load_factor == pytest.approx(0.0)
    assert not receiver.accepted_increments
    assert not receiver.checkpoints
    np.testing.assert_allclose(receiver.solution.x.array, initial_solution)
    np.testing.assert_allclose(
        receiver.accepted_solution.x.array,
        initial_accepted,
    )
    restored = receiver.state_transaction.snapshot_runtime_state()
    np.testing.assert_allclose(
        restored["first_piola_stress"],
        initial_state["first_piola_stress"],
    )


def test_standard_j2_checkpoint_requires_an_accepted_boundary(tmp_path):
    _, _, _, step = _standard_patch(incrementation=steps.fixed(2))
    step.solve(until=0.5)

    step.solution.x.array[0] += 1.0e-3
    with pytest.raises(RuntimeError, match="U equals U_ACCEPTED"):
        step.save_checkpoint(tmp_path / "trial_serial")
    with pytest.raises(RuntimeError, match="U equals U_ACCEPTED"):
        step.save_checkpoint(tmp_path / "trial_portable", portable=True)

    step.solution.x.array[:] = step.accepted_solution.x.array
    step.solution.x.scatter_forward()
    step.state_transaction.accepted_factor = 0.25
    with pytest.raises(RuntimeError, match="fully accepted material state"):
        step.save_checkpoint(tmp_path / "mismatched_coordinate")


def test_standard_j2_legacy_checkpoint_disables_incomplete_work_ledger(tmp_path):
    _, _, _, source = _standard_patch(incrementation=steps.fixed(4))
    source.solve(until=0.5)
    current = source.save_checkpoint(tmp_path / "current")
    legacy = tmp_path / "legacy_without_work_history.npz"
    with np.load(current, allow_pickle=False) as data:
        payload = {
            name: np.asarray(data[name]).copy()
            for name in data.files
            if name not in {"energy_history", "energy_history_complete"}
        }
    np.savez(legacy, **payload)

    _, _, _, restarted = _standard_patch(incrementation=steps.fixed(4))
    restarted.load_checkpoint(legacy)
    restarted.solve()

    assert not restarted.energy_history_complete
    assert restarted.energy_history == []
    assert restarted.summary()["energy_history_complete"] is False


def test_standard_j2_portable_checkpoint_restore_is_atomic(monkeypatch, tmp_path):
    _, _, _, source = _standard_patch(incrementation=steps.fixed(2))
    source.solve(until=0.5)
    manifest = source.save_checkpoint(tmp_path / "portable", portable=True)

    _, _, _, receiver = _standard_patch(incrementation=steps.fixed(2))
    initial_solution = receiver.solution.x.array.copy()
    initial_state = receiver.state_transaction.snapshot_runtime_state()

    def fail_quadrature_load(*_args, **_kwargs):
        raise ValueError("intentional quadrature restore failure")

    monkeypatch.setattr(receiver.response.state, "load", fail_quadrature_load)
    with pytest.raises(RuntimeError, match="quadrature restore failure"):
        receiver.load_checkpoint(manifest)

    assert receiver.accepted_load_factor == pytest.approx(0.0)
    assert not receiver.accepted_increments
    assert not receiver.checkpoints
    np.testing.assert_allclose(receiver.solution.x.array, initial_solution)
    restored = receiver.state_transaction.snapshot_runtime_state()
    np.testing.assert_allclose(
        restored["first_piola_stress"],
        initial_state["first_piola_stress"],
    )


def test_standard_j2_rejects_legacy_experimental_checkpoint_identity(tmp_path):
    _, _, _, step = _standard_patch(incrementation=steps.fixed(1))
    portable = tmp_path / "legacy.checkpoint.json"
    portable.write_text(
        json.dumps(
            {
                "schema": (
                    "agentfem.finite-strain-j2-experimental-checkpoint.v2"
                )
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Legacy experimental"):
        step.load_checkpoint(portable)

    partition_bound = tmp_path / "legacy.npz"
    np.savez(
        partition_bound,
        schema="agentfem.finite-strain-j2-experimental-checkpoint.v1",
    )
    with pytest.raises(ValueError, match="Legacy experimental"):
        step.load_checkpoint(partition_bound)

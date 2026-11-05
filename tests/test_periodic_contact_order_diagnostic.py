# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""The order diagnostic must not leave quadratic periodic trace DOFs free."""
from pathlib import Path
import runpy

import numpy as np
import pytest


def functions():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]
                             / "tools/diagnose_periodic_contact_order.py"))


def test_energy_increment_audit_localizes_without_correcting_work():
    from types import SimpleNamespace
    audit = functions()["energy_increment_audit"]
    frames = [SimpleNamespace(index=i, factor=f, values=(v,))
              for i, (f, v) in enumerate(((0., 0.), (.1, .2), (.2, .4)))]
    duals = [{"load_factor": f.factor, "force": [q], "outgoing_force": [q]}
             for f, q in zip(frames, (0., 2., 2.), strict=True)]
    result = audit(duals, frames)
    assert result["first_increment"]["residual"] == pytest.approx(-.1)
    assert result["largest_absolute_residual"]["index"] == 1
    assert result["residual_sum"] == pytest.approx(-.1)
    assert result["residual_after_first_increment"] == pytest.approx(0.)
    # Kinks need outgoing force on the left, incoming force on the right.
    duals[1]["outgoing_force"] = [4.]
    assert audit(duals, frames)["residual_after_first_increment"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="same accepted stations"):
        audit(duals, frames[:-1])
    duals[1]["load_factor"] += .01
    with pytest.raises(ValueError, match="coordinates differ"):
        audit(duals, frames)


@pytest.mark.parametrize("initial,automatic", [("nan", True), ("0", True),
                                              ("0.1", True), ("0.01", False)])
def test_initial_increment_diagnostic_rejects_ambiguous_or_invalid_control(tmp_path, monkeypatch, initial, automatic):
    import sys
    args = ["diagnostic", "--initial-increment", initial, "--output", str(tmp_path / "report.json")]
    if automatic:
        args.extend(["--plastic-increment-limit", "0.00075"])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as exc:
        functions()["main"]()
    assert exc.value.code == 2
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("mesh_family,cells", [("uniform", 4), ("edge_sine", 8),
                                              ("normal_cosine", 8), ("patch_edge_sine", 8)])
def test_all_boundary_field_dofs_obey_periodic_motion_and_guides(degree, mesh_family, cells):
    step = functions()["make_step"](degree=degree, cells=cells, mesh_family=mesh_family,
                                    increments=4, yield_stress=1e6)
    coordinates = step.solution.function_space.tabulate_dof_coordinates()
    reduction = step.constraint.reduction(.25)
    from scipy.sparse import csr_matrix
    T = csr_matrix((reduction.coefficients, reduction.column_indices, reduction.row_offsets),
                   shape=(reduction.full_size, reduction.reduced_size))
    # An arbitrary reduced vector exercises the constraint null space, not only
    # the trivial affine solution which would conceal missing midpoint rows.
    q = np.random.default_rng(241).normal(size=reduction.reduced_size)
    u = (T @ q + reduction.offset).reshape(-1, 3)
    keys = {tuple(np.rint(x * 1e12).astype(int)): i for i, x in enumerate(coordinates)}
    F = np.diag([1.02, 1.02**-.5, 1.02**-.5])
    checked = 0
    for i, x in enumerate(coordinates):
        axes = np.isclose(x, 1.)
        if np.any(axes):
            wrapped = np.where(axes, 0., x)
            j = keys[tuple(np.rint(wrapped * 1e12).astype(int))]
            np.testing.assert_allclose(u[i] - u[j], (F - np.eye(3)) @ (x - wrapped), atol=2e-14)
            checked += 1
        if np.isclose(x[2], 0.) or np.isclose(x[2], 1.):
            assert u[i, 0] == pytest.approx(.02 * x[0], abs=2e-14)
    assert checked > 50
    if degree == 2:
        assert len(coordinates) > 125
        assert step.contact_provider.adapter.interpolation_degree == 2


@pytest.mark.parametrize("cells", [8, 12, 16])
def test_edge_mesh_keeps_patch_geometry_and_positive_spacing(cells):
    lines = functions()["diagnostic_coordinate_lines"](cells, "edge_sine")
    assert np.all(np.diff(lines, axis=1) > 0.)
    np.testing.assert_array_equal(lines[:, [0, -1]], [[0., 1.]] * 3)
    np.testing.assert_allclose(lines + lines[:, ::-1], 1., atol=3e-16)
    np.testing.assert_array_equal(lines[1:, ::cells // 4], [np.linspace(0., 1., 5)] * 2)
    # Adjacent to each patch edge, both sides are finer than the uniform grid.
    for edge in (cells // 4, 3 * cells // 4):
        assert np.all(np.diff(lines[1, edge - 1:edge + 2]) < 1. / cells)


def test_edge_mesh_rejects_ineffective_or_unknown_family():
    lines = functions()["diagnostic_coordinate_lines"]
    with pytest.raises(ValueError, match="at least eight"):
        lines(4, "edge_sine")
    with pytest.raises(ValueError, match="Unknown"):
        lines(8, "guess")


@pytest.mark.parametrize("cells", [8, 12, 16])
def test_directional_grading_is_exact_factorization_of_existing_mesh(cells):
    lines = functions()["diagnostic_coordinate_lines"]
    uniform = lines(cells, "uniform")
    combined = lines(cells, "edge_sine")
    normal = lines(cells, "normal_cosine")
    edge = lines(cells, "patch_edge_sine")
    np.testing.assert_array_equal(normal[0], combined[0])
    np.testing.assert_array_equal(normal[1:], uniform[1:])
    np.testing.assert_array_equal(edge[0], uniform[0])
    np.testing.assert_array_equal(edge[1:], combined[1:])
    for family in ("normal_cosine", "patch_edge_sine"):
        with pytest.raises(ValueError, match="at least eight"):
            lines(4, family)


@pytest.mark.parametrize("mesh_family", ["edge_sine", "normal_cosine", "patch_edge_sine"])
def test_edge_mesh_keeps_volume_patch_and_checkpoint_identity(tmp_path, mesh_family):
    import ufl
    from dolfinx import fem
    from agentfem import mesh
    make = functions()["make_step"]
    step = make(cells=8, degree=2, increments=4, yield_stress=1e6,
                mesh_family=mesh_family, contact_degree=11)
    domain = step.solution.function_space.mesh
    assert mesh.audit_quality(domain, strict=True).global_cells == 6 * 8**3
    assert fem.assemble_scalar(fem.form(1. * ufl.dx(domain=domain))) == pytest.approx(1., abs=1e-12)
    step.solve(until=.25)
    samples = step.contact_provider.accepted_point_samples()
    partition = functions()["patch_pressure_partition"](samples)
    assert partition["total_reference_area"] == pytest.approx(.25)
    assert partition["smooth_partition"]["center"]["weighted_reference_area"] == pytest.approx(1./9.)
    checkpoint = step.save_checkpoint(tmp_path / "graded.npz")
    wrong = make(cells=8, degree=2, increments=4, yield_stress=1e6, contact_degree=11)
    before = wrong.solution.x.array.copy()
    with pytest.raises(ValueError, match="identity"):
        wrong.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(wrong.solution.x.array, before)


def test_interrupted_diagnostic_does_not_leave_running_or_complete_record(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    main = functions()["main"]
    output = tmp_path / "interrupted.json"
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--cells", "4", "--degrees", "2",
                                     "--output", str(output)])
    def solve(**kwargs):
        raise KeyboardInterrupt
    monkeypatch.setitem(main.__globals__, "make_step", lambda **kwargs: SimpleNamespace(
        solution=SimpleNamespace(x=SimpleNamespace(array=np.zeros(3))), solve=solve))
    with pytest.raises(KeyboardInterrupt):
        main()
    row = json.loads(output.read_text())["records"][0]
    assert row["status"] == "interrupted" and row["complete_cycle"] is False
    assert row["stations"] == [] and row["setup_solve_observe_seconds"] >= 0.


def test_quadratic_periodic_gauss_contact_accepts_current_total_residual():
    diagnostic = functions()
    step = diagnostic["make_step"](degree=2, increments=8)
    step.solve(until=.25)
    assert step.constraint.mismatch(.25) < 1e-12
    assert step.contact_provider.evaluate().active_points > 0
    observation = diagnostic["observe_penalty"](step, .25)
    assert observation["peeq_reference_volume_mean"] > 0
    assert "normal_relative_work" not in observation
    assert observation["normal_relative_work_status"] == "unavailable_on_this_diagnostic_route"
    assert observation["energy"]["plastic_dissipation"] > 0.
    assert observation["energy"]["contact_potential"] > 0.
    assert observation["energy_status"] == "accepted_trapezoidal_work_not_exact_continuous_path"
    assert observation["energy_increment_audit"]["residual_sum"] == pytest.approx(
        observation["energy"]["mechanical_energy_residual"], abs=1e-12)
    samples = step.contact_provider.accepted_point_samples()
    assert samples["measure_configuration"] == "reference"
    adapter = step.contact_provider.adapter
    interpolated_u = adapter.trace.evaluate(
        step.solution.x.array.reshape(-1, 3)
    ).query_points
    np.testing.assert_allclose(samples["slave_positions"] - samples["reference_positions"],
                               interpolated_u, atol=2e-15)
    partition = diagnostic["patch_pressure_partition"](samples)
    assert partition["total_reference_area"] == pytest.approx(.25)
    assert partition["smooth_partition"]["center"]["weighted_reference_area"] == pytest.approx(1./9.)
    assert sum(r["weighted_normal_force"] for r in partition["smooth_partition"].values()) == pytest.approx(
        partition["normal_force_magnitude"], abs=1e-12)
    assert sum(r["normal_force_magnitude"] for r in partition["regions"].values()) == pytest.approx(
        -observation["force_on_structure"][0], abs=1e-12)
    samples["reference_positions"][:] = -1.
    assert np.all(step.contact_provider.accepted_point_samples()["reference_positions"] >= 0.)
    full = step.assemble_full_residual()
    T = step.constraint.reduction(.25).matrix(step.solution.function_space.mesh.comm)
    r = T.createVecRight()
    try:
        T.multTranspose(full, r)
        assert r.norm() < 1e-7
    finally:
        full.destroy()
        r.destroy()
        T.destroy()


def test_duplicate_field_coordinates_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        functions()["periodic_field_equations"](np.zeros((2, 3)))


def test_diagnostic_uses_existing_accepted_checkpoint_and_status_contract(tmp_path):
    from agentfem import checkpointing
    diagnostic = functions()
    step = diagnostic["make_step"](degree=2, increments=4, yield_stress=1e6,
        checkpoint=checkpointing.every(1, directory=tmp_path / "checkpoints", keep_last=2),
        status_file=tmp_path / "progress.sta")
    step.solve(until=.25)
    assert "PAUSED" in (tmp_path / "progress.sta").read_text()
    assert len(step.checkpoints) == 1
    restored = diagnostic["make_step"](degree=2, increments=4, yield_stress=1e6)
    restored.load_checkpoint(step.checkpoints[-1].path)
    assert restored.accepted_load_factor == .25
    np.testing.assert_array_equal(restored.solution.x.array, step.solution.x.array)
    np.testing.assert_array_equal(restored.response.state.committed_state_vectors(),
                                  step.response.state.committed_state_vectors())
    np.testing.assert_array_equal(restored.contact_provider.accepted_point_samples()["normal_pressures"],
                                  step.contact_provider.accepted_point_samples()["normal_pressures"])


def test_patch_partition_preserves_force_and_rejects_wrong_measure():
    partition = functions()["patch_pressure_partition"]
    samples = {"reference_positions": np.array([[1., .26, .5], [1., .5, .5]]),
               "reference_weights": np.array([.1, .15]),
               "normal_pressures": np.array([4., 2.]), "measure_configuration": "reference"}
    result = partition(samples)
    assert result["regions"]["edge"]["normal_force_magnitude"] == pytest.approx(.4)
    assert result["regions"]["interior"]["normal_force_magnitude"] == pytest.approx(.3)
    assert result["normal_force_magnitude"] == pytest.approx(.7)
    with pytest.raises(ValueError, match="reference-area"):
        partition({**samples, "measure_configuration": "current"})
    with pytest.raises(ValueError, match="cover"):
        partition({**samples, "reference_weights": np.array([.1, .1])})


@pytest.mark.parametrize("increments", [0, 1, 6, -4])
def test_fixed_diagnostic_rejects_invalid_station_partition(tmp_path, monkeypatch, increments):
    import sys
    output = tmp_path / "invalid.json"
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--increments", str(increments),
                                    "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        functions()["main"]()
    assert error.value.code == 2
    assert not output.exists()


def test_diagnostic_refuses_existing_checkpoint_directory(tmp_path, monkeypatch):
    import sys
    output = tmp_path / "evidence.json"
    checkpoint = tmp_path / "previous"
    checkpoint.mkdir()
    retained = checkpoint / "keep.txt"
    retained.write_text("previous accepted state")
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--checkpoint-directory", str(checkpoint),
                                     "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        functions()["main"]()
    assert error.value.code == 2
    assert retained.read_text() == "previous accepted state"
    assert not output.exists()


@pytest.mark.parametrize("selection", [[], ["--cells", "4", "--degrees", "2"]])
def test_diagnostic_restart_rejects_ambiguous_or_missing_source(tmp_path, monkeypatch, selection):
    import sys
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(sys, "argv", ["diagnostic", *selection,
        "--restart", str(tmp_path / "missing.npz"), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        functions()["main"]()
    assert error.value.code == 2
    assert not output.exists()


def test_diagnostic_cli_restart_uses_public_state_and_preserves_original(tmp_path, monkeypatch):
    from hashlib import sha256
    import json
    import sys
    diagnostic = functions()
    partial = diagnostic["make_step"](degree=2, increments=4, yield_stress=1e6)
    partial.solve(until=.25)
    checkpoint = partial.save_checkpoint(tmp_path / "partial.npz")
    original = checkpoint.read_bytes()
    output = tmp_path / "resumed.json"
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--cells", "4", "--degrees", "2",
        "--increments", "4", "--yield-stress", "1000000", "--until", ".5",
        "--restart", str(checkpoint), "--patch-partition", "--output", str(output)])
    diagnostic["main"]()
    report = json.loads(output.read_text())
    row = report["records"][0]
    assert report["restart_source_sha256"] == sha256(original).hexdigest()
    assert checkpoint.read_bytes() == original
    assert row["status"] == "completed" and row["resumed_from"] == .25
    assert row["accepted_through"] == .5 and not row["complete_cycle"]
    assert row["accepted_increments"] == 2 and len(row["stations"]) == 1
    continuous = diagnostic["make_step"](degree=2, increments=4, yield_stress=1e6)
    continuous.solve(until=.5)
    expected = diagnostic["observe_penalty"](continuous, .5)
    np.testing.assert_allclose(row["stations"][0]["force_on_structure"],
                               expected["force_on_structure"], atol=1e-11)
    for key, value in expected["energy"].items():
        assert row["stations"][0]["energy"][key] == pytest.approx(value, abs=1e-12)
    # Changed physical input remains a failed diagnostic, not a forced restore.
    output_wrong = tmp_path / "wrong.json"
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--cells", "4", "--degrees", "2",
        "--increments", "4", "--yield-stress", "1000000", "--penalty", "20000",
        "--until", ".5", "--restart", str(checkpoint), "--output", str(output_wrong)])
    with pytest.raises(RuntimeError, match="failed; evidence preserved"):
        diagnostic["main"]()
    failed = json.loads(output_wrong.read_text())["records"][0]
    assert failed["status"] == "failed" and "identity" in failed["error"]
    assert checkpoint.read_bytes() == original


@pytest.mark.parametrize("limit,measured", [(None, .001), (.000375, .0003), (.000375, .0004)])
@pytest.mark.parametrize("until", [.25, 1.])
def test_diagnostic_records_and_enforces_selected_path_policy(tmp_path, monkeypatch, limit, measured, until):
    import json
    import sys
    from types import SimpleNamespace
    main = functions()["main"]
    output = tmp_path / "comparison.json"
    arguments = ["diagnostic", "--cells", "4", "--degrees", "2", "--output", str(output),
                 "--until", str(until)]
    if limit is not None:
        arguments += ["--plastic-increment-limit", str(limit)]
    monkeypatch.setattr(sys, "argv", arguments)
    policies = []
    def make(**kwargs):
        policies.append(kwargs["incrementation"])
        history = []
        def solve(*, until):
            history.append(SimpleNamespace(converged=True, iterations=2,
                checks={"maximum_plastic_increment": measured}))
        return SimpleNamespace(solve=solve, accepted_increments=history, attempted_increments=history,
            solution=SimpleNamespace(x=SimpleNamespace(array=np.zeros(3))))
    monkeypatch.setitem(main.__globals__, "make_step", make)
    monkeypatch.setitem(main.__globals__, "observe_penalty", lambda step, factor: {"factor": factor})
    if limit is not None and measured > limit:
        with pytest.raises(RuntimeError, match="failed; evidence preserved"):
            main()
    else:
        main()
    report = json.loads(output.read_text())
    assert "constitutive/finite_strain_plasticity.py" in report["source_sha256"]
    assert "boundary_models/dolfinx_implicit_facet_contact.py" in report["source_sha256"]
    assert report["incrementation"] == json.loads(json.dumps(policies[0].summary()))
    row = report["records"][0]
    assert row["maximum_accepted_plastic_increment"] == measured
    assert row["rejected_attempts"] == 0 and row["newton_iterations"] == 8 * until
    assert row["complete_cycle"] == (until == 1.)
    assert row["accepted_through"] == until
    assert (row["status"] == "failed") == (limit is not None and measured > limit)
    assert report["accuracy_claim"].startswith("none:")

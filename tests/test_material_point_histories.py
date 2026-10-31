# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Co-location, accepted-only sampling and atomic restart through model.step."""

import json
from dataclasses import replace

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import amplitudes, constitutive, fields, histories, mesh, models, results, solvers, steps, studies, units


def _step(*, requests=None, joint=False, regional=False, automatic=False):
    domain = mesh.cuboid((0., 0., 0.), (2., 1., 1.), (2, 1, 1),
                         cell_type="hexahedron", comm=MPI.COMM_SELF)
    model = models.create(study=studies.creep_solid(), mesh=domain, units=units.n_mm_mpa())
    u = model.field(fields.displacement(domain))
    temperature = model.field(fields.temperature(domain))
    path = histories.temperature(temperature)
    for time in (0., 4.):
        temperature.value.interpolate(lambda x, current=time: 800. + 20. * x[0] + 10. * current)
        temperature.value.x.scatter_forward()
        path.record(time)
    elastic = constitutive.thermoelastic(
        young=200_000., poisson=.3, density=1., thermal_expansion=1.e-5,
        conductivity=1., specific_heat=1., reference_temperature=800.)
    def material(coefficient):
        law = constitutive.isotropic_arrhenius_power_law(
            elastic=elastic, coefficient=coefficient, stress_exponent=3., reference_stress=100.,
            activation_energy=120_000., reference_temperature=800.)
        if joint:
            law = constitutive.CreepPlasticityMaterial(creep=law, plastic=constitutive.ChabocheCombinedHardening(
                young=200_000., poisson=.3, yield_stress=200., backstress_moduli=(12_000., 3_000.),
                dynamic_recovery=(100., 8.), isotropic_saturation=60., isotropic_rate=10.))
        return law
    if regional:
        regions = mesh.partition_cells(domain, left=lambda x: x[0] <= 1., right=lambda x: x[0] > 1.)
        model.material(material(1.e-6), region=regions.left)
        model.material(material(3.e-6), region=regions.right)
    else:
        model.material(material(1.e-6))
    for component, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component + 1), component=component, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=2., tag=4), component=0, value=.008)
    if requests is None:
        requests = (results.material_point("a", cell=0), results.material_point("b", cell=1, point=1))
    control = steps.automatic(initial=.25, minimum=.01, maximum=.25) if automatic else steps.fixed(8)
    return model.step(target=u, duration=4., temperature=path,
                      material_points=requests, incrementation=control,
                      amplitude=amplitudes.tabular([0., 1., 2., 3., 4.], [0., 1., 1., -.5, -.5]),
                      progress=False, solver_options=solvers.newton(maximum_iterations=30, line_search="backtracking"))


@pytest.mark.parametrize("joint,regional", [(False, False), (True, False), (False, True)])
def test_colocated_fields_use_original_cell_keys_and_physical_time(joint, regional):
    step = _step(joint=joint, regional=regional)
    result = step.solve_result()
    identity = result.metadata["material_point_history"]
    assert identity["scientific_identity"]["units"]["base"]["length"] == "mm"
    for point in identity["points"]:
        prefix = point["name"]
        temp = result.histories[prefix + ".TEMP"]
        np.testing.assert_allclose(temp.values, 800. + 20. * point["reference_position"][0] + 10. * temp.abscissa)
        assert temp.unit == "K"
        assert temp.abscissa_unit == "s"
        local = int(np.flatnonzero(step.state.stress.owned_cell_keys == point["cell"])[0])
        index = local * len(step.state.stress.points) + point["point"]
        np.testing.assert_array_equal(result.histories[prefix + ".S"].latest, step.state.stress.values[index])
        np.testing.assert_array_equal(result.histories[prefix + ".E"].latest,
                                      step.state.evaluate_strain(step._strain_evaluator)[index])
        np.testing.assert_array_equal(result.histories[prefix + ".CE"].latest, step.state.creep_strain.values[index])
        if joint:
            np.testing.assert_array_equal(result.histories[prefix + ".PE"].latest, step.state.plastic_fields["plastic_strain"].values[index])
            assert result.histories[prefix + ".ALPHA"].value_shape == (2, 3, 3)
    assert not np.array_equal(result.histories["a.TEMP"].values, result.histories["b.TEMP"].values)
    assert len(result.histories["a.S"].abscissa) == len(step.accepted_increments) == 8


@pytest.mark.parametrize("portable", [False, True])
def test_resume_preserves_full_point_trajectory_and_result_archive(tmp_path, portable):
    full = _step(joint=True)
    full.solve()
    partial = _step(joint=True)
    partial.solve(until=2.)
    checkpoint = partial.save_checkpoint(tmp_path / "state", portable=portable)
    resumed = _step(joint=True)
    resumed.load_checkpoint(checkpoint)
    result = resumed.solve_result()
    for expected, actual in zip(full.material_point_histories, resumed.material_point_histories, strict=True):
        assert expected.name == actual.name
        np.testing.assert_array_equal(expected.abscissa, actual.abscissa)
        np.testing.assert_allclose(expected.values, actual.values, rtol=1.e-8, atol=1.e-8)
    # Consumer receives ordinary tensor HistoryResult objects, not a new file format.
    assert result.histories["a.S"].value_shape == (3, 3)
    assert result.metadata["material_point_history"]["quadrature_rule"]
    archive = result.write_manifest(tmp_path / "result.json", include_histories=True, history_storage="npz")
    recovered = {item.name: item for item in results.read_histories(archive)}
    for item in resumed.material_point_histories:
        np.testing.assert_array_equal(recovered[item.name].values, item.values)
    assert json.loads(archive.read_text())["metadata"]["material_point_history"] == result.metadata["material_point_history"]


@pytest.mark.parametrize("portable", [False, True])
def test_corrupt_point_history_rejects_entire_restore_atomically(tmp_path, portable):
    source = _step(joint=True)
    source.solve(until=2.)
    checkpoint = source.save_checkpoint(tmp_path / "state", portable=portable)
    if portable:
        payload = json.loads(checkpoint.read_text())
        payload["material_point_history"]["frames"][0]["values"]["TEMP"][0] += 1.
        checkpoint.write_text(json.dumps(payload))
    else:
        with np.load(checkpoint, allow_pickle=False) as data:
            payload = dict(data)
        history = json.loads(str(payload["material_point_history"]))
        history["frames"][0]["values"]["TEMP"][0] += 1.
        payload["material_point_history"] = json.dumps(history)
        np.savez(checkpoint, **payload)
    target = _step(joint=True)
    target.solve(until=1.)
    old_u, old_state = target.solution.x.array.copy(), target.state.snapshot()
    old_history = target._point_history.payload()
    with pytest.raises(ValueError, match="integrity"):
        target.load_checkpoint(checkpoint)
    assert target.accepted_time == target.temperature.active_time == 1.
    assert target._point_history.payload() == old_history
    np.testing.assert_array_equal(target.solution.x.array, old_u)
    for key, values in old_state.items():
        np.testing.assert_array_equal(target.state.snapshot()[key], values)


def test_changed_selection_and_missing_recording_cannot_silently_resume(tmp_path):
    source = _step()
    source.solve(until=1.)
    checkpoint = source.save_checkpoint(tmp_path / "state")
    for requests in ((), (results.material_point("a", cell=1),)):
        receiver = _step(requests=requests)
        with pytest.raises(ValueError, match="(identity|declare the same)"):
            receiver.load_checkpoint(checkpoint)
        assert receiver.accepted_time == 0.


def test_failed_trial_is_not_a_material_point_sample(monkeypatch):
    step = _step(automatic=True)
    original = step._solve_increment
    rejected = []
    def fail_once(*args, **kwargs):
        info = original(*args, **kwargs)
        if not rejected:
            rejected.append(info.end_time)
            return replace(info, converged=False, rejection_reason="injected test rejection")
        return info
    monkeypatch.setattr(step, "_solve_increment", fail_once)
    step.solve()
    recorded = step.material_point_histories[0].abscissa
    np.testing.assert_array_equal(recorded, [item.end_time for item in step.accepted_increments])
    assert len(step.attempted_increments) > len(recorded)


@pytest.mark.parametrize("kwargs", [{"cell": True}, {"cell": -1}, {"cell": .1}, {"cell": 0, "point": False}])
def test_selector_rejects_ambiguous_ids(kwargs):
    with pytest.raises(ValueError):
        results.material_point("test", **kwargs)


def test_selector_rejects_unknown_point_and_duplicate_names_before_solve():
    for requests in ((results.material_point("a", cell=99),),
                     (results.material_point("a", cell=0, point=99),),
                     (results.material_point("a", cell=0), results.material_point("a", cell=1))):
        with pytest.raises(ValueError, match="(Unknown|unique)"):
            _step(requests=requests)


def test_reference_geometry_change_is_rejected_before_continuation():
    step = _step()
    step.solve(until=1.)
    step.state.domain.geometry.x[:, 0] *= 1.1
    with pytest.raises(ValueError, match="unchanged reference"):
        step.solve()
    assert step.accepted_time == 1.


def test_sampling_failure_rolls_back_to_last_accepted_point(monkeypatch):
    step = _step()
    step.solve(until=1.)
    previous = step._point_history.payload()
    displacement = step.solution.x.array.copy()
    state = step.state.snapshot()
    def fail(*args, **kwargs):
        raise ValueError("injected sample failure")
    with monkeypatch.context() as patch:
        patch.setattr(step._point_history, "prepare", fail)
        with pytest.raises(ValueError, match="injected sample"):
            step.solve()
    assert step.accepted_time == step.temperature.active_time == 1.
    assert step._point_history.payload() == previous
    np.testing.assert_array_equal(step.solution.x.array, displacement)
    for key, value in state.items():
        np.testing.assert_array_equal(step.state.snapshot()[key], value)
    step.solve()
    assert len(step.material_point_histories[0].abscissa) == 8


def test_legacy_serial_checkpoint_with_thermal_history_is_compatible(tmp_path):
    source = _step(requests=())
    source.solve(until=2.)
    checkpoint = source.save_checkpoint(tmp_path / "legacy")
    with np.load(checkpoint, allow_pickle=False) as data:
        payload = dict(data)
    payload.pop("material_point_history")
    identity = json.loads(str(payload["step_identity"]))
    identity["temperature"] = source._temperature_summary()
    payload["step_identity"] = json.dumps(identity)
    np.savez(checkpoint, **payload)
    resumed = _step(requests=())
    resumed.load_checkpoint(checkpoint)
    assert resumed.accepted_time == resumed.temperature.active_time == 2.

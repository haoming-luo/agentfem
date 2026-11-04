"""One-way prescribed thermal inputs through the existing history Procedure."""

import xml.etree.ElementTree as ET
import json
from pathlib import Path
import shutil
import importlib.util
from dataclasses import replace

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import amplitudes, constitutive as c, fields, histories, mesh, models, steps, studies
from agentfem.mechanics._prescribed_temperature import freeze_uniform_temperature


class ThermalHistory:
    name = "synthetic thermal Green-strain material"
    tangent_convention = c.MaterialTangentConvention.first_piola_deformation_gradient()
    state_schema = c.MaterialStateSchema("thermal_test", (
        c.MaterialStateVariable("EXPOSURE", unit="K*s"),
        c.MaterialStateVariable("DELTA_T", unit="K"),
    ))
    lame, shear, alpha, reference = 100., 50., 1.e-4, 300.

    def __init__(self, limit=np.inf):
        self.limit = limit
        self.requests = []

    def summary(self):
        return {"kind": "synthetic_thermal_history", "version": 1,
                "limit": None if np.isinf(self.limit) else self.limit}

    def update(self, point):
        assert point.temperature is not None and point.temperature_increment is not None
        self.requests.append((point.time, point.time_increment, point.temperature,
                              point.temperature_increment, point.state_old.copy()))
        F, identity = point.deformation_gradient_new, np.eye(3)
        E = .5 * (F.T @ F - identity) - self.alpha * (point.temperature - self.reference) * identity
        S = self.lame * np.trace(E) * identity + 2 * self.shear * E
        C = (self.lame * np.einsum("ij,kl->ijkl", identity, identity)
             + self.shear * (np.einsum("ik,jl->ijkl", identity, identity)
                             + np.einsum("il,jk->ijkl", identity, identity)))
        A = np.einsum("ik,LJ->iJkL", identity, S) + np.einsum("iA,AJBL,kB->iJkL", F, C, F)
        state = point.state_old + np.array([
            (point.temperature - .5 * point.temperature_increment) * point.time_increment,
            point.temperature_increment,
        ])
        return c.MaterialPointOutput(
            cauchy_stress=F @ S @ F.T / np.linalg.det(F),
            consistent_tangent=A.reshape(9, 9), state_new=state,
            state_schema=self.state_schema, tangent_convention=self.tangent_convention,
            suggested_time_scale=min(1., self.limit / point.time_increment),
        )

    def update_batch(self, batch):
        return c.MaterialPointBatchOutput(tuple(self.update(p) for p in batch.points))


def history(values=(300., 360., 340., 400.)):
    result = histories.temperature(300.)
    for t, value in zip((0., .6, 1.4, 2.), values, strict=True):
        result.record(t, value)
    return result


def build(temperature=None, *, assumption=None, constrain=False, increments=None, limit=np.inf):
    dimension = 3 if assumption is None else 2
    domain = (mesh.cuboid((0., 0., 0.), (1., 1., 1.), (1, 1, 1),
                         cell_type="hexahedron", comm=MPI.COMM_SELF) if dimension == 3 else
              mesh.rectangle((0., 0.), (1., 1.), (1, 1), cell_type="quadrilateral", comm=MPI.COMM_SELF))
    study = studies.static_solid(dimension=dimension, nonlinear=True,
                                **({"assumption": assumption} if assumption else {}))
    model = models.create(study=study, mesh=domain)
    u = model.field(fields.displacement(domain))
    material = ThermalHistory(limit)
    model.material(material)
    for i, axis in enumerate(("x", "y", "z")[:dimension]):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=i+1), component=i, value=0.)
    if constrain:
        model.fix(u, on=mesh.face(domain, axis="x", value=1., tag=4), component=0, value=0.)
    options = {"plane_stress": c.PlaneStressReduction(symmetry="reflection_symmetric")} if assumption == "plane_stress" else {}
    return model.step(target=u, duration=2., time_unit="s", progress=False,
                      temperature=history() if temperature is None else temperature,
                      amplitude=amplitudes.tabular([0., .5, 1.], [0., 1., 0.]),
                      incrementation=steps.fixed(4) if increments is None else increments,
                      **options), material


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf, True, 300j, [300.]])
def test_invalid_scalar_temperature(bad):
    with pytest.raises((ValueError, TypeError)):
        freeze_uniform_temperature(bad, 2.)


def test_history_freeze_validation_and_source_not_mutated():
    source = history()
    active = source.active_time
    frozen = freeze_uniform_temperature(source, 2.)
    assert frozen.at(.3) == 330.
    assert source.active_time == active
    source.record(2., 900.)
    assert frozen.at(2.) == 400.
    for option, value in (("unit", "degC"), ("coordinate_name", "load_factor"), ("interpolation", "previous")):
        bad = history()
        setattr(bad, option, value)
        with pytest.raises(ValueError):
            freeze_uniform_temperature(bad, 2.)
    with pytest.raises(ValueError, match="cover"):
        freeze_uniform_temperature(source, 3.)
    with pytest.raises(ValueError, match="positive"):
        freeze_uniform_temperature(history((300., -1., 300., 300.)), 2.)


@pytest.mark.parametrize("assumption", [None, "plane_strain", "plane_stress"])
def test_free_thermal_patch_time_knots_and_history(assumption, tmp_path):
    step, material = build(assumption=assumption)
    result = step.solve_result(output=tmp_path / "thermal.xdmf")
    times = np.array([r["time"] for r in step.accepted_increments])
    assert {.6, 1., 1.4, 2.} <= set(times)
    expected_E = .01 if assumption != "plane_strain" else .01 * 400. / 300.
    F = step._accepted_F
    np.testing.assert_allclose(F[:, 0, 0], np.sqrt(1 + 2 * expected_E), rtol=1e-8)
    np.testing.assert_allclose(F[:, 1, 1], F[:, 0, 0], atol=1e-9)
    np.testing.assert_allclose(step.state.committed["EXPOSURE"].values, 700., atol=1e-9)
    np.testing.assert_allclose(step.state.committed["DELTA_T"].values, 100., atol=1e-10)
    np.testing.assert_allclose(step._temperature_field.values, 400.)
    assert result.fields["TEMP"].unit == "K"
    assert "no heat equation" in result.fields["TEMP"].processing["coupling"]
    for t, dt, T, dT, _ in material.requests:
        assert T == pytest.approx(np.interp(t, (0., .6, 1.4, 2.), (300., 360., 340., 400.)))
        assert T-dT == pytest.approx(np.interp(t-dt, (0., .6, 1.4, 2.), (300., 360., 340., 400.)))
    frames = ET.parse(tmp_path / "thermal.xdmf").findall(".//Grid[@GridType='Uniform']")
    assert len(frames) == len(times)
    assert all({"TEMP_CELL", "S_CELL", "Displacement"} <= {a.get("Name") for a in frame.findall("Attribute")} for frame in frames)


def test_constrained_reaction_and_fixed_temperature_tangent():
    step, material = build(constrain=True)
    step.solve()
    # E11=0, S22=S33=0 gives this analytical nominal axial stress.
    expected = -material.shear * .01 * (3 * material.lame + 2 * material.shear) / (material.lame + material.shear)
    np.testing.assert_allclose(step.response.first_piola_stress.values[:, 0, 0], expected, rtol=1e-8)
    point = c.MaterialPointInput(deformation_gradient_old=np.eye(3),
        deformation_gradient_new=np.diag([1.02, 1.01, 1.01]),
        time=1., time_increment=.1, temperature=350., temperature_increment=5.,
        properties=np.array([]), state_old=np.zeros(2), state_schema=material.state_schema)
    report = c.check_material_tangent(material, point)
    assert report.accepted


def test_thermal_checkpoint_continuation_and_changed_input(tmp_path):
    full, _ = build()
    full.solve()
    partial, _ = build()
    partial.solve(until=.5)
    checkpoint = partial.save_checkpoint(tmp_path / "accepted")
    resumed, _ = build()
    resumed.load_checkpoint(checkpoint)
    resumed.solve()
    for key, field in full._responses.items():
        np.testing.assert_allclose(resumed._responses[key].x.array, field.x.array, atol=1e-8)
    for name, values in full.state.snapshot().items():
        np.testing.assert_allclose(resumed.state.snapshot()[name], values, atol=1e-10)
    changed, _ = build(history((300., 361., 340., 400.)))
    before = changed._snapshot()
    with pytest.raises(ValueError, match="identity"):
        changed.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(changed.solution.x.array, before["U"])

    with np.load(checkpoint, allow_pickle=False) as archive:
        corrupted = {name: archive[name].copy() for name in archive.files}
    corrupted["response_TEMP"] += 1.
    bad_path = tmp_path / "wrong-temperature.npz"
    np.savez(bad_path, **corrupted)
    fresh, _ = build()
    initial = fresh._snapshot()
    with pytest.raises(ValueError, match="temperature disagrees"):
        fresh.load_checkpoint(bad_path)
    for name, field in fresh._responses.items():
        np.testing.assert_array_equal(field.x.array, initial["response"][name])
    assert fresh.accepted_load_factor == 0.


def test_constant_temperature_and_scalar_provider():
    step, material = build(300.)
    # The scalar material entry uses the same prescribed thermal channels.
    material.update_batch = None
    result = step.solve_result()
    np.testing.assert_allclose(step.solution.x.array, 0., atol=1e-13)
    np.testing.assert_allclose(step.state.committed["EXPOSURE"].values, 600.)
    np.testing.assert_allclose(step.state.committed["DELTA_T"].values, 0.)
    np.testing.assert_allclose(result.histories["temperature"].values, 300.)
    assert all(request[3] == 0. for request in material.requests)


def test_step_freezes_source_and_protects_accepted_temperature():
    source = history()
    step, _ = build(source)
    source.record(2., 900.)
    step.solve(until=.5)
    assert step._temperature_input.at(2.) == 400.
    step._temperature_field.function.x.array[:] += 1.
    with pytest.raises(ValueError, match="outside the accepted Step state"):
        step.solve()


def test_failed_thermal_trial_restores_accepted_fields(monkeypatch):
    step, material = build()
    step.solve(until=.5)
    before = step._snapshot()
    original = material.update

    def fail(point):
        original(point)
        raise RuntimeError("thermal material failure")

    monkeypatch.setattr(material, "update", fail)
    with pytest.raises(RuntimeError):
        step.solve()
    assert step.accepted_load_factor == .5
    np.testing.assert_array_equal(step.solution.x.array, before["U"])
    for name, field in step._responses.items():
        np.testing.assert_array_equal(field.x.array, before["response"][name])
    for name, values in step.state.snapshot().items():
        np.testing.assert_array_equal(values, before["state"][name])


def test_thermal_cutback_preserves_delta_and_commits_once():
    step, _ = build(limit=.21, increments=steps.automatic(initial=.5, minimum=.005, maximum=.5))
    step.solve()
    assert any(not r["accepted"] for r in step.attempted_increments)
    np.testing.assert_allclose(step.state.committed["EXPOSURE"].values, 700., atol=1e-8)
    np.testing.assert_allclose(step.state.committed["DELTA_T"].values, 100., atol=1e-9)


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_thermal_example_uses_standard_project_run(tmp_path, mechanism):
    from agentfem import cli
    source = Path(__file__).parents[1] / "examples/history_material"
    for name in ("case.py", "material.py", "agentfem.toml", "README.md"):
        shutil.copy2(source / name, tmp_path / name)
    assert cli.main(["check", "--project", str(tmp_path)]) == 0
    assert cli.main(["run", "--project", str(tmp_path), "--name", "thermal", "--",
                     "--material", mechanism, "--temperature-cycle", "--output-every", "4"]) == 0
    output = tmp_path / "outputs/001-thermal"
    record = json.loads((output / "result.json").read_text())
    assert record["status"] == "completed"
    assert record["metadata"]["step"]["temperature"]["unit"] == "K"
    frames = ET.parse(output / "results.xdmf").findall(".//Grid[@GridType='Uniform']")
    assert len(frames) < 10
    assert all("TEMP_CELL" in {a.get("Name") for a in frame.findall("Attribute")} for frame in frames)


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_thermal_example_evolved_state_tangent_and_objectivity(mechanism):
    spec = importlib.util.spec_from_file_location(
        "thermal_example", Path(__file__).parents[1] / "examples/history_material/material.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    material = module.HistoryMaterial(mechanism=mechanism, thermal_expansion=1.e-4)
    size = 10 if mechanism == "viscoelastic" else 2
    point = c.MaterialPointInput(
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=np.diag([1.03, 1.01, .99]),
        time=.2, time_increment=.2, temperature=340., temperature_increment=40.,
        properties=np.array([]), state_old=np.zeros(size), state_schema=material.state_schema)
    preload = material.update(point)
    following = replace(point, deformation_gradient_old=point.deformation_gradient_new,
                        deformation_gradient_new=np.array([[1.06, .02, 0.], [0., 1.01, 0.], [0., 0., .99]]),
                        time=.4, temperature=360., temperature_increment=20., state_old=preload.state_new)
    assert c.check_material_tangent(material, following).accepted
    response = material.update(following)
    angle = .71
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    rotated = material.update(replace(
        following, deformation_gradient_old=rotation @ following.deformation_gradient_old,
        deformation_gradient_new=rotation @ following.deformation_gradient_new))
    np.testing.assert_allclose(rotated.cauchy_stress,
                               rotation @ response.cauchy_stress @ rotation.T, atol=1e-12)
    np.testing.assert_allclose(rotated.state_new, response.state_new, atol=1e-12)

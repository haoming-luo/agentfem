"""Physical-time material consumption through the ordinary structural Step."""

import json
from dataclasses import replace
from pathlib import Path
import runpy

import numpy as np
import pytest
from dolfinx import mesh as dm
from mpi4py import MPI

from agentfem import (
    amplitudes,
    constitutive,
    fields,
    loads,
    mesh,
    models,
    solvers,
    steps,
    studies,
)
from test_small_strain_material_step import LinearBatchMaterial


class RelaxingMaterial(LinearBatchMaterial):
    """Isotropic standard linear solid; backward Euler local integration."""

    rate_independent = False
    stress_free_initial_state = True
    name = "standard_linear_solid_reference"

    def __init__(self, max_dt=np.inf):
        super().__init__()
        self.max_dt = max_dt
        self.calls = []
        self.state_schema = constitutive.MaterialStateSchema(
            "test.viscoelastic_time_state",
            (
                constitutive.MaterialStateVariable("EV", shape=(3, 3)),
                constitutive.MaterialStateVariable("AGE", unit="s"),
            ),
            version="1.0",
        )

    def update_batch(self, request):
        self.calls.append((request.time, request.time_increment))
        dt = request.time_increment
        strain = request.strain_new
        trace = np.trace(strain, axis1=1, axis2=2)[:, None, None]
        dev = strain - trace * np.eye(3) / 3
        state = request.state_old.copy()
        viscous = (state[:, :9].reshape(-1, 3, 3) + dt * dev) / (1 + dt)
        state[:, :9] = viscous.reshape(-1, 9)
        state[:, 9] += dt
        stress = 1000 * trace * np.eye(3) + 200 * dev + 400 * (dev - viscous)
        shear = 100 + 200 / (1 + dt)
        tangent = np.zeros((6, 6))
        tangent[:3, :3] = 1000 - 2 * shear / 3
        tangent[np.arange(6), np.arange(6)] += 2 * shear
        return constitutive.SmallStrainMaterialPointBatchOutput(
            cauchy_stress=stress,
            consistent_tangent=np.broadcast_to(tangent, (len(strain), 6, 6)).copy(),
            state_new=state,
            state_schema=self.state_schema,
            tangent_convention=self.tangent_convention,
            suggested_time_scale=min(1.0, self.max_dt / dt),
        )


class KelvinVoigtMaterial(RelaxingMaterial):
    name = "kelvin_voigt_reference"

    def update_batch(self, request):
        result = super().update_batch(request)
        dt = request.time_increment
        strain = request.strain_new
        trace = np.trace(strain, axis1=1, axis2=2)[:, None, None]
        dev = strain - trace * np.eye(3) / 3
        rate = (strain - request.strain_old) / dt
        rate -= np.trace(rate, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
        shear = 100 + 20 / dt
        tangent = np.zeros((6, 6))
        tangent[:3, :3] = 1000 - 2 * shear / 3
        tangent[np.arange(6), np.arange(6)] += 2 * shear
        return constitutive.SmallStrainMaterialPointBatchOutput(
            cauchy_stress=1000 * trace * np.eye(3) + 200 * dev + 40 * rate,
            consistent_tangent=np.broadcast_to(tangent, (len(strain), 6, 6)).copy(),
            state_new=result.state_new,
            state_schema=self.state_schema,
            tangent_convention=self.tangent_convention,
        )


class EnergeticRelaxingMaterial(RelaxingMaterial):
    provides_stored_energy_density = True
    provides_dissipation_density_increment = True
    stored_energy_component_names = ("BULK", "EQUILIBRIUM", "BRANCH")

    def update_batch(self, request):
        response = super().update_batch(request)
        trace = np.trace(request.strain_new, axis1=1, axis2=2)
        dev = request.strain_new - trace[:, None, None] * np.eye(3) / 3
        viscous = response.state_new[:, :9].reshape(-1, 3, 3)
        rate = (
            viscous - request.state_old[:, :9].reshape(-1, 3, 3)
        ) / request.time_increment
        components = {
            "BULK": 500 * trace**2,
            "EQUILIBRIUM": 100 * np.sum(dev**2, axis=(1, 2)),
            "BRANCH": 200 * np.sum((dev - viscous) ** 2, axis=(1, 2)),
        }
        return replace(
            response,
            stored_energy_density=sum(components.values()),
            stored_energy_density_components=components,
            dissipation_density_increment=400
            * np.sum(rate**2, axis=(1, 2))
            * request.time_increment,
        )


def make_step(
    material=None,
    duration=2.0,
    increments=None,
    amplitude=None,
    traction=None,
    time_unit="s",
    **options,
):
    domain = dm.create_box(
        MPI.COMM_WORLD,
        [np.zeros(3), np.array([1.0, 0.2, 0.2])],
        [2, 1, 1],
        cell_type=dm.CellType.tetrahedron,
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain
    )
    u = model.field(fields.displacement(domain))
    material = RelaxingMaterial() if material is None else material
    model.material(material)
    for axis, component, length in (("x", 0, 1.0), ("y", 1, 0.2), ("z", 2, 0.2)):
        for value in (0.0, length):
            if axis == "x" and value == 1 and traction is not None:
                model.load(
                    loads.traction(
                        (traction, 0.0, 0.0), on=mesh.face(domain, axis="x", value=1.0)
                    )
                )
                continue
            model.fix(
                u,
                on=mesh.face(domain, axis=axis, value=value),
                component=component,
                value=0.002 if axis == "x" and value == 1 else 0.0,
            )
    return model.step(
        target=u,
        duration=duration,
        time_unit=time_unit,
        progress=False,
        incrementation=steps.fixed(8) if increments is None else increments,
        amplitude=amplitude,
        **options,
    )


@pytest.mark.parametrize("provider", [RelaxingMaterial, KelvinVoigtMaterial])
def test_two_rate_materials_share_structural_consumer(provider):
    material = provider()
    step = make_step(material)
    assert material.calls == []  # initialization must not consume fictitious time
    step.solve()
    assert step.last_solve_info.completed_step
    assert step.time == 2.0
    np.testing.assert_allclose(step.state.committed["AGE"].values, 2.0, atol=1e-13)
    assert all(dt == 0.25 for _, dt in material.calls)
    if provider is KelvinVoigtMaterial:
        expected = 2 + 200 * (0.002 * 2 / 3) + 40 * (0.001 * 2 / 3)
    else:
        ev = 0.0
        for index in range(1, 9):
            ev = (ev + 0.25 * (0.002 * index / 8 * 2 / 3)) / 1.25
        expected = 2 + 200 * (0.002 * 2 / 3) + 400 * (0.002 * 2 / 3 - ev)
    np.testing.assert_allclose(
        step.response.cauchy_stress.values[..., 0, 0], expected, rtol=1e-10
    )


def test_cutback_does_not_advance_material_clock():
    step = make_step(
        RelaxingMaterial(max_dt=0.25),
        increments=steps.automatic(
            initial=0.5,
            maximum=0.5,
            minimum=0.001,
            growth_factor=1.5,
            cutback_factor=0.5,
        ),
    )
    step.solve()
    assert step.last_solve_info.completed_step
    assert len(step.attempted_increments) > len(step.accepted_increments)
    np.testing.assert_allclose(step.state.committed["AGE"].values, 2.0, atol=1e-13)
    assert all(
        info.load_factor - info.start_load_factor <= 0.125 + 1e-12
        for info in step.accepted_increments
    )


def test_restart_uses_saved_response_without_provider_call(tmp_path):
    uninterrupted = make_step()
    uninterrupted.solve()
    paused = make_step()
    paused.solve(until=0.5)
    checkpoint = paused.save_checkpoint(tmp_path / "physical")
    restored = make_step()
    restored.load_checkpoint(checkpoint)
    assert restored.material.calls == []
    assert restored.time == 1.0
    np.testing.assert_array_equal(
        restored.response.cauchy_stress.values, paused.response.cauchy_stress.values
    )
    np.testing.assert_array_equal(
        restored.response.tangent.values, paused.response.tangent.values
    )
    restored.solve()
    np.testing.assert_allclose(
        restored.solution.x.array, uninterrupted.solution.x.array, atol=1e-14
    )
    for name in ("EV", "AGE"):
        np.testing.assert_allclose(
            restored.state.committed[name].values,
            uninterrupted.state.committed[name].values,
            atol=1e-14,
        )
    np.testing.assert_allclose(
        restored.response.cauchy_stress.values,
        uninterrupted.response.cauchy_stress.values,
        atol=1e-12,
    )
    assert json.loads(checkpoint.read_text())["schema"].endswith("v2")


@pytest.mark.parametrize("duration", [0, -1, np.nan, np.inf, True])
def test_invalid_duration_rejected(duration):
    with pytest.raises(ValueError, match="duration"):
        make_step(duration=duration)


def test_restart_duration_identity_and_initial_checkpoint(tmp_path):
    step = make_step()
    with pytest.raises(ValueError, match="accepted increment"):
        step.save_checkpoint(tmp_path / "initial")
    step.solve(until=0.5)
    with pytest.raises(NotImplementedError, match="serial"):
        step.save_checkpoint(tmp_path / "portable", portable=True)
    checkpoint = step.save_checkpoint(tmp_path / "valid")
    other = make_step(duration=3.0)
    with pytest.raises(ValueError, match="identity"):
        other.load_checkpoint(checkpoint)
    assert other.time == 0.0
    assert other.material.calls == []


def test_relaxation_temporal_refinement():
    # Ramp for one second, then hold for one second. Exact Maxwell branch
    # stress at t=2 is 400 * dev(strain_final) * (1-exp(-1))*exp(-1).
    amplitude = amplitudes.tabular([0.0, 0.5, 1.0], [0.0, 1.0, 1.0])
    exact = (
        2
        + 200 * (0.002 * 2 / 3)
        + 400 * (0.002 * 2 / 3) * (1 - np.exp(-1)) * np.exp(-1)
    )
    errors = []
    for count in (8, 16, 32):
        step = make_step(increments=steps.fixed(count), amplitude=amplitude)
        step.solve()
        errors.append(
            abs(np.mean(step.response.cauchy_stress.values[..., 0, 0]) - exact)
        )
    assert errors[1] < 0.6 * errors[0]
    assert errors[2] < 0.6 * errors[1]


def test_off_grid_hold_knot_and_physical_result_history():
    step = make_step(
        increments=steps.fixed(4),
        amplitude=amplitudes.tabular([0.0, 0.3, 0.7, 1.0], [0.0, 1.0, 1.0, 0.0]),
    )
    result = step.solve_result()
    coordinates = [info.load_factor for info in step.accepted_increments]
    assert 0.3 in coordinates and 0.7 in coordinates
    assert result.metadata["physical_time"]["accepted_time"] == 2.0
    history = result.histories["load_amplitude"]
    np.testing.assert_allclose(history.abscissa, np.array([0.0, *coordinates]) * 2)
    np.testing.assert_allclose(step.state.committed["AGE"].values, 2.0)


@pytest.mark.parametrize("defect", ["nan", "shape", "layout", "missing", "diagnostics"])
def test_corrupt_response_rejected_without_live_mutation(tmp_path, defect):
    from agentfem import checkpointing

    source = make_step()
    source.solve(until=0.75)
    checkpoint = source.save_checkpoint(tmp_path / "source")
    payload = json.loads(checkpoint.read_text())
    path = checkpoint.parent / payload["accepted_response"]["path"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if defect == "nan":
        arrays["stress"].flat[0] = np.nan
    elif defect == "shape":
        arrays["tangent"] = arrays["tangent"][:1]
    elif defect == "layout":
        arrays["cell_keys"] = arrays["cell_keys"][::-1]
    elif defect == "diagnostics":
        arrays["diagnostics"] = np.array("not-json")
    else:
        del arrays["stress"]
    checkpointing.atomic_savez(path, **arrays)
    payload["accepted_response"] = checkpointing.checkpoint_file_record(path)
    checkpointing.atomic_write_text(checkpoint, json.dumps(payload))
    live = make_step()
    live.solve(until=0.25)
    before_u = live.solution.x.array.copy()
    before_state = live.state.snapshot()
    before_response = live._response_snapshot()
    before_calls = len(live.material.calls)
    with pytest.raises(ValueError, match="validation"):
        live.load_checkpoint(checkpoint)
    assert len(live.material.calls) == before_calls
    assert live.time == 0.5
    np.testing.assert_array_equal(live.solution.x.array, before_u)
    for key, values in before_state.items():
        np.testing.assert_array_equal(live.state.snapshot()[key], values)
    for key, values in before_response[0].items():
        np.testing.assert_array_equal(live._response_snapshot()[0][key], values)


def test_declared_initial_state_required():
    material = RelaxingMaterial()
    material.stress_free_initial_state = False
    with pytest.raises(NotImplementedError, match="stress_free_initial_state"):
        make_step(material)


def test_nonzero_initial_history_rejected():
    material = RelaxingMaterial()
    material.state_schema = constitutive.MaterialStateSchema(
        "test.nonzero_initial",
        (constitutive.MaterialStateVariable("EV", initial_value=1.0),),
        version="1.0",
    )
    with pytest.raises(NotImplementedError, match="zero named history"):
        make_step(material)


def test_failed_fixed_step_retains_accepted_response():
    material = RelaxingMaterial(max_dt=0.25)
    step = make_step(material)
    step.solve(until=0.25)
    before = step._response_snapshot()
    material.max_dt = 0.01
    with pytest.raises(RuntimeError, match="fixed increment"):
        step.solve()
    assert step.time == 0.5
    np.testing.assert_allclose(step.state.committed["AGE"].values, 0.5)
    for key, values in before[0].items():
        np.testing.assert_array_equal(step._response_snapshot()[0][key], values)


def test_force_controlled_creep_matches_independent_scalar_recurrence():
    step = make_step(
        traction=2.0, amplitude=amplitudes.tabular([0.0, 0.5, 1.0], [0.0, 1.0, 1.0])
    )
    step.solve()
    ev = 0.0
    for index in range(1, 9):
        stress = 2 * min(index / 4, 1)
        strain = (stress + 400 * ev / 1.25) / (1000 + 4 / 3 * (100 + 200 / 1.25))
        ev = (ev + 0.25 * 2 / 3 * strain) / 1.25
    np.testing.assert_allclose(step._accepted_strain[..., 0, 0], strain, rtol=1e-10)
    np.testing.assert_allclose(
        step.response.cauchy_stress.values[..., 0, 0], 2.0, rtol=1e-10
    )


@pytest.mark.parametrize(
    "change", ["duration", "remove_duration", "unit", "policy", "target"]
)
def test_live_physical_path_inputs_cannot_change(change, tmp_path):
    step = make_step()
    step.solve(until=0.5)
    if change == "duration":
        step.duration = 3.0
    elif change == "remove_duration":
        step.duration = None
    elif change == "unit":
        step.time_unit = "h"
    elif change == "policy":
        step.incrementation = steps.fixed(10)
    else:
        step.prescribed_values[-1][1][...] += 0.01
    for action in (step.solve, lambda: step.save_checkpoint(tmp_path / "invalid")):
        with pytest.raises(ValueError, match="scientific inputs changed"):
            action()
    assert not list(tmp_path.iterdir())


def test_changed_traction_identity_rejects_restore(tmp_path):
    source = make_step(traction=2.0)
    source.solve(until=0.5)
    checkpoint = source.save_checkpoint(tmp_path / "force")
    receiver = make_step(traction=3.0)
    with pytest.raises(ValueError, match="identity"):
        receiver.load_checkpoint(checkpoint)
    assert receiver.material.calls == []


def test_scalar_example_material_checkpoint_and_batch_equivalence(tmp_path):
    provider = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "examples/small_strain_relaxation/material.py"
        )
    )["StandardLinearSolid"]
    scalar = make_step(provider())
    scalar.solve(until=0.5)
    checkpoint = scalar.save_checkpoint(tmp_path / "scalar")
    resumed = make_step(provider())
    resumed.load_checkpoint(checkpoint)
    resumed.solve()
    batch = make_step()
    batch.solve()
    np.testing.assert_allclose(
        resumed.response.cauchy_stress.values,
        batch.response.cauchy_stress.values,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        resumed.state.committed["EV"].values,
        batch.state.committed["EV"].values,
        atol=1e-14,
    )


@pytest.mark.parametrize("exception", [KeyboardInterrupt, ValueError])
def test_interrupted_trial_restores_accepted_state(tmp_path, monkeypatch, exception):
    step = make_step()
    step.solve(until=0.25)
    before_u = step.solution.x.array.copy()
    before_response = step._response_snapshot()
    before_state = step.state.snapshot()

    def fail():
        step.solution.x.array[:] += 0.1
        raise exception("injected assembly interruption")

    with monkeypatch.context() as patch:
        patch.setattr(step, "_correction_rhs", fail)
        with pytest.raises(exception, match="interruption"):
            step.solve()
    assert step.time == 0.5
    np.testing.assert_array_equal(step.solution.x.array, before_u)
    for key, values in before_state.items():
        np.testing.assert_array_equal(step.state.snapshot()[key], values)
    for key, values in before_response[0].items():
        np.testing.assert_array_equal(step._response_snapshot()[0][key], values)
    checkpoint = step.save_checkpoint(tmp_path / "interrupted")
    restarted = make_step()
    restarted.load_checkpoint(checkpoint)
    restarted.solve()
    step.solve()
    np.testing.assert_allclose(
        restarted.response.cauchy_stress.values, step.response.cauchy_stress.values
    )


@pytest.mark.skipif(MPI.COMM_WORLD.size != 2, reason="Two-rank admission check")
def test_physical_time_rejects_unverified_mpi_composition():
    with pytest.raises(NotImplementedError, match="serial execution"):
        make_step()


def test_unimplemented_control_is_not_silently_ignored():
    with pytest.raises(NotImplementedError, match="inelastic scalar"):
        make_step(increments=steps.automatic(maximum_inelastic_increment=0.01))
    with pytest.raises(NotImplementedError, match="raise on exhausted"):
        make_step(solver_options=solvers.newton(error_if_not_converged=False))


def test_declared_energy_response_survives_restart(tmp_path):
    source = make_step(EnergeticRelaxingMaterial())
    source.solve(until=0.5)
    checkpoint = source.save_checkpoint(tmp_path / "energy")
    restored = make_step(EnergeticRelaxingMaterial())
    restored.load_checkpoint(checkpoint)
    assert restored.material.calls == []
    for key, field in source._response_fields().items():
        np.testing.assert_array_equal(
            field.values, restored._response_fields()[key].values
        )
    restored.solve()
    source.solve()
    for key, field in source._response_fields().items():
        np.testing.assert_allclose(
            field.values, restored._response_fields()[key].values, atol=1e-13
        )


def test_failed_response_publication_preserves_previous_checkpoint(
    tmp_path, monkeypatch
):
    from agentfem import checkpointing

    step = make_step()
    step.solve(until=0.5)
    checkpoint = step.save_checkpoint(tmp_path / "latest")
    previous = checkpoint.read_bytes()
    step.solve(until=0.75)

    original_savez = checkpointing.atomic_savez

    def fail(path, **arrays):
        if str(path).endswith(".response.npz"):
            raise OSError("injected response disk failure")
        return original_savez(path, **arrays)

    with monkeypatch.context() as patch:
        patch.setattr(checkpointing, "atomic_savez", fail)
        with pytest.raises(OSError, match="disk failure"):
            step.save_checkpoint(tmp_path / "latest")
    assert checkpoint.read_bytes() == previous
    restored = make_step()
    restored.load_checkpoint(checkpoint)
    assert restored.time == 1.0

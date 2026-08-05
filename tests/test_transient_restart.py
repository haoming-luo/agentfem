from __future__ import annotations

import json

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, studies


def _left(x):
    return np.isclose(x[0], 0.0)


def _right(x):
    return np.isclose(x[0], 1.0)


def _dynamic_step(*, implicit: bool):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (3, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_stress",
            method="newmark" if implicit else "explicit",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        constitutive.isotropic_elastic(
            young=2.0e5,
            poisson=0.3,
            density=1.0e3,
        )
    )
    model.clamp(
        displacement,
        on=mesh.boundary(domain, _left, name="left", tag=1),
    )
    model.traction(
        (10.0, 0.0),
        on=mesh.boundary(domain, _right, name="right", tag=2),
    )
    return model.step(
        target=displacement,
        dt=1.0e-4,
        steps=4,
        progress=False,
    )


@pytest.mark.parametrize("implicit", [False, True])
def test_dynamics_restart_matches_uninterrupted_state_and_energy(tmp_path, implicit):
    reference = _dynamic_step(implicit=implicit)
    reference.run()

    partial = _dynamic_step(implicit=implicit)
    partial.run(until_step=2)
    checkpoint = partial.save_checkpoint(
        tmp_path / ("implicit" if implicit else "explicit")
    )
    manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert manifest["completed_steps"] == 2
    assert manifest["portable"] is False
    assert manifest["portability"] == "same mesh partition and MPI size"

    restarted = _dynamic_step(implicit=implicit)
    restarted.load_checkpoint(checkpoint)
    restarted.run()

    assert restarted.completed_steps == reference.completed_steps == 4
    np.testing.assert_allclose(
        restarted.state.u.value.x.array,
        reference.state.u.value.x.array,
    )
    np.testing.assert_allclose(
        restarted.state.v.value.x.array,
        reference.state.v.value.x.array,
    )
    np.testing.assert_allclose(
        restarted.state.a.value.x.array,
        reference.state.a.value.x.array,
    )
    assert restarted.history_records == pytest.approx(reference.history_records)
    simulation = restarted.solve_result()
    assert {
        "kinetic_energy",
        "strain_energy",
        "total_mechanical_energy",
    } <= set(simulation.histories)
    assert next(iter(simulation.checkpoints.values())).portable is False


def _heat_step():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (3, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.transient_heat_transfer(dimension=2),
        mesh=domain,
    )
    temperature = model.field(fields.temperature(domain, value=400.0))
    model.material(
        constitutive.thermoelastic(
            young=1.0e9,
            poisson=0.3,
            density=1000.0,
            thermal_expansion=1.0e-5,
            conductivity=10.0,
            specific_heat=500.0,
            reference_temperature=300.0,
        )
    )
    model.convection(
        on=mesh.boundary(domain, _right, name="right", tag=1),
        coefficient=25.0,
        ambient_temperature=300.0,
    )
    return model.step(
        target=temperature,
        dt=0.5,
        steps=4,
        progress=False,
    )


def test_heat_restart_matches_uninterrupted_state_and_thermal_history(tmp_path):
    reference = _heat_step()
    reference.run()

    partial = _heat_step()
    partial.run(until_step=2)
    checkpoint = partial.save_checkpoint(tmp_path / "heat")
    restarted = _heat_step()
    restarted.load_checkpoint(checkpoint)
    result = restarted.solve_result()

    np.testing.assert_allclose(restarted.current.x.array, reference.current.x.array)
    np.testing.assert_allclose(restarted.previous.x.array, reference.previous.x.array)
    np.testing.assert_allclose(
        result.histories["thermal_content"].values,
        [item["thermal_content"] for item in reference.history_records],
    )
    assert np.all(np.diff(result.histories["thermal_content"].values) <= 0.0)


def test_transient_checkpoint_rejects_a_different_time_contract(tmp_path):
    partial = _dynamic_step(implicit=True)
    partial.run(until_step=1)
    checkpoint = partial.save_checkpoint(tmp_path / "contract")
    incompatible = _dynamic_step(implicit=True)
    incompatible.dt = 2.0e-4

    with pytest.raises(ValueError, match="time increment differs"):
        incompatible.load_checkpoint(checkpoint)

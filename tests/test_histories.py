from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import constitutive, fields, histories, materials, mesh, models, studies


def test_scalar_history_interpolates_and_rejects_silent_extrapolation(tmp_path):
    history = histories.temperature(300.0)
    history.record(0.0)
    history.source = 500.0
    history.record(10.0)

    assert history.sample(2.5).item() == pytest.approx(350.0)
    with pytest.raises(ValueError, match="covers"):
        history.sample(10.1)

    saved = history.save(tmp_path / "temperature")
    restored = histories.FieldHistory.load(saved, source=0.0)
    assert restored.sample(7.5).item() == pytest.approx(450.0)
    assert restored.value == pytest.approx(500.0)
    assert restored.active_time == pytest.approx(10.0)
    assert restored.scientific_identity() == history.scientific_identity()


def test_finite_element_history_applies_interpolated_field():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    temperature = fields.temperature(domain, value=300.0)
    history = histories.temperature(temperature)
    history.record(0.0)
    temperature.value.x.array[:] = 500.0
    temperature.value.x.scatter_forward()
    history.record(2.0)

    history.apply(0.5)

    np.testing.assert_allclose(temperature.value.x.array, 350.0)
    assert history.active_time == pytest.approx(0.5)


def test_nodal_history_roundtrips_through_portable_coordinate_archive(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    temperature = fields.temperature(domain, value=300.0)
    history = histories.temperature(temperature)
    history.record(0.0)
    temperature.value.interpolate(lambda x: 400.0 + 50.0 * x[0])
    history.record(2.0)

    archive = history.save(tmp_path / "portable_temperature")
    restored_field = fields.temperature(domain, value=0.0)
    restored = histories.FieldHistory.load(archive, source=restored_field)

    np.testing.assert_allclose(restored.sample(1.0), history.sample(1.0))
    np.testing.assert_allclose(restored_field.value.x.array, history.value.x.array)
    assert restored.active_time == pytest.approx(2.0)
    assert restored.portable_identity() == history.portable_identity()
    with np.load(archive, allow_pickle=False) as payload:
        assert str(payload["schema"]) == "agentfem.field-history.v2"
        assert payload["coordinates"].ndim == 2
        assert payload["snapshots"].shape[0] == 2


def test_transient_heat_step_captures_accepted_temperature_history():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
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
        on=mesh.face(domain, axis="x", value=1.0, name="right", tag=1),
        coefficient=25.0,
        ambient_temperature=300.0,
    )
    step = model.step(target=temperature, dt=0.5, steps=2, progress=False)
    thermal_history = step.capture_history(name="temperature", unit="K")

    step.run()

    assert thermal_history.times == pytest.approx((0.0, 0.5, 1.0))
    assert thermal_history.frame_count == 3
    assert np.mean(thermal_history.sample(1.0)) < np.mean(
        thermal_history.sample(0.0)
    )
    assert step.summary()["captured_histories"][0]["unit"] == "K"


def test_temperature_property_table_is_bounded_and_inspectable():
    young = materials.temperature_property(
        [300.0, 500.0],
        [200.0e9, 160.0e9],
        name="young",
        unit="Pa",
    )

    assert young(400.0) == pytest.approx(180.0e9)
    with pytest.raises(ValueError, match="covers"):
        young(600.0)
    assert young.as_dict()["temperature_unit"] == "K"


def test_temperature_dependent_properties_drive_sequential_thermal_expansion():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.linear_static(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_stress",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    temperature = fields.temperature(domain, value=400.0)
    table_options = {"extrapolation": "constant"}
    material = model.material(
        constitutive.temperature_dependent_thermoelastic(
            young=materials.temperature_property(
                [300.0, 500.0], [200.0e9, 160.0e9], name="young", **table_options
            ),
            poisson=0.3,
            density=7800.0,
            thermal_expansion=materials.temperature_property(
                [300.0, 500.0], [10.0e-6, 20.0e-6], name="alpha", **table_options
            ),
            conductivity=45.0,
            specific_heat=500.0,
            reference_temperature=300.0,
        )
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0, name="left", tag=1),
        component=0,
        value=0.0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="y", value=0.0, name="bottom", tag=2),
        component=1,
        value=0.0,
    )
    step = model.step(
        target=displacement,
        K=model.stiffness(displacement, temperature=temperature),
        F=model.thermal_expansion(displacement, temperature),
    )

    step.solve()

    coordinates = displacement.space.tabulate_dof_coordinates()
    values = displacement.value.x.array.reshape((-1, 2))
    right = np.isclose(coordinates[:, 0], 1.0)
    np.testing.assert_allclose(values[right, 0], 15.0e-6 * 100.0, rtol=2.0e-9)
    assert material.at_temperature(400.0).young == pytest.approx(180.0e9)

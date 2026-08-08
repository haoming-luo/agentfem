from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import constitutive, fields, fracture, mesh, models, results, studies


def _trace():
    coordinate = np.linspace(0.0, 1.0, 101)
    time = np.linspace(0.0, 0.4, 9)
    fronts = 0.2 + 1.25 * time
    damage = np.asarray(
        [1.0 / (1.0 + np.exp((coordinate - front) / 0.01)) for front in fronts]
    )
    return fracture.CohesiveInterfaceTrace(
        time=time,
        path_coordinate=coordinate,
        opening=0.02 * damage,
        traction=10.0 * (1.0 - damage),
        damage=damage,
        dissipated_energy_density=0.1 * damage,
        metadata={"configuration": "reference"},
    )


def test_cohesive_trace_round_trip_and_multi_observer_front(tmp_path):
    trace = _trace()
    restored = fracture.CohesiveInterfaceTrace.read(
        trace.write(tmp_path / "interface_trace")
    )
    ensemble = fracture.cohesive_front_ensemble(
        restored,
        damage_thresholds=(0.5, 0.75),
        opening_thresholds=(0.01,),
        dissipation_thresholds=(0.05,),
        fit_window=5,
    )

    np.testing.assert_allclose(restored.damage, trace.damage)
    np.testing.assert_allclose(ensemble.histories[0].position, 0.2 + 1.25 * trace.time, atol=2.0e-3)
    np.testing.assert_allclose(ensemble.histories[0].speed[1:-1], 1.25, atol=1.0e-2)
    assert len(ensemble.histories) == 4
    assert ensemble.summary()["maximum_speed_spread"] < 5.0e-3


def test_curve_mach_cone_and_rectilinear_field_comparisons():
    x = np.linspace(0.0, 1.0, 11)
    curve = fracture.compare_curve(x, 2.0 * x, x[::2], 2.0 * x[::2])
    assert curve.normalized_root_mean_square_error < 1.0e-14
    assert curve.correlation == pytest.approx(1.0)

    mach = fracture.compare_mach_cone(
        crack_speed=2.0,
        shear_wave_speed=1.0,
        observed_angle=30.0,
        unit="degree",
    )
    assert mach.root_mean_square_error == pytest.approx(0.0, abs=1.0e-12)

    reference_x = np.linspace(0.0, 1.0, 7)
    reference_y = np.linspace(-0.5, 0.5, 5)
    simulation_x = np.linspace(0.0, 1.0, 4)
    simulation_y = np.linspace(-0.5, 0.5, 3)
    observed = reference_y[:, None] + 2.0 * reference_x[None, :]
    simulated = simulation_y[:, None] + 2.0 * simulation_x[None, :]
    comparison = fracture.compare_rectilinear_field(
        reference_x,
        reference_y,
        observed,
        simulation_x,
        simulation_y,
        simulated,
        quantity_name="SED",
    )
    assert comparison.samples == reference_x.size * reference_y.size
    assert comparison.normalized_root_mean_square_error < 1.0e-14


def test_finite_strain_ked_is_a_standard_cell_field():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.5),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    displacement = fields.displacement(domain)
    velocity = fields.displacement(domain)
    velocity.value.name = "V"
    velocity.value.interpolate(
        lambda x: np.vstack((2.0 * np.ones_like(x[0]), np.zeros_like(x[0])))
    )
    material = constitutive.neo_hookean(
        young=1000.0,
        poisson=0.3,
        density=3.0,
    )

    (ked,) = results.finite_strain_cell_fields(
        displacement,
        material,
        variables=("KED",),
        velocity=velocity,
    )

    assert ked.name == "KED"
    np.testing.assert_allclose(ked.x.array, 6.0)


def test_explicit_output_refreshes_live_sener_ked_and_j(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_strain",
            method="explicit",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.neo_hookean(
            young=1000.0,
            poisson=0.3,
            density=2.0,
        )
    )
    step = model.step(
        target=displacement,
        material=material,
        dt=1.0e-6,
        steps=2,
        save_every=1,
        progress=False,
    )
    live = results.finite_strain_dynamic_cell_fields(
        step.state.u,
        step.state.v,
        material,
        variables=("SENER", "KED", "J"),
    )

    step.run(
        output=tmp_path / "dynamic_fields.xdmf",
        fields=(step.state.u, step.state.v, live),
    )

    assert (tmp_path / "dynamic_fields.xdmf").is_file()
    assert len(step.last_output_fields) == 5
    assert tuple(field.name for field in step.last_output_fields[-3:]) == (
        "SENER", "KED", "J"
    )
    np.testing.assert_allclose(live.fields[-1].x.array, 1.0)

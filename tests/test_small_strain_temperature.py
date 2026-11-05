"""Prescribed thermal input shares the accepted small-strain material clock."""

from dataclasses import replace
import numpy as np
import pytest
from agentfem import histories, results
from test_small_strain_physical_time import RelaxingMaterial, make_step


class ThermalMaterial(RelaxingMaterial):
    name = "thermal_standard_linear_solid_reference"

    def __init__(self):
        super().__init__()
        self.thermal_calls = []

    def update_batch(self, request):
        t = np.asarray(request.temperature)
        delta = np.asarray(request.temperature_increment)
        self.thermal_calls.append(
            (request.time, request.time_increment, t.copy(), delta.copy())
        )
        return super().update_batch(
            replace(
                request,
                strain_new=request.strain_new
                - 1e-5 * (t - 300)[:, None, None] * np.eye(3),
                strain_old=request.strain_old
                - 1e-5 * (t - delta - 300)[:, None, None] * np.eye(3),
            )
        )


def path():
    temperature = histories.temperature(300.0)
    temperature.record(0.0, 300.0)
    temperature.record(0.7, 340.0)
    temperature.record(2.0, 350.0)
    return temperature


def build(**kwargs):
    return make_step(
        ThermalMaterial(),
        temperature=path(),
        material_points=(results.material_point("a", cell=0),),
        **kwargs,
    )


def test_thermal_knots_and_restrained_stress():
    step = build()
    step.solve()
    history = {h.name: h for h in step.material_point_histories}
    assert np.any(np.isclose(history["a.TEMP"].abscissa, 0.7))
    assert history["a.TEMP"].latest == 350.0
    # Thermal shift is spherical: trace(stress)=3*K*(trace(strain)-3*alpha*dT).
    np.testing.assert_allclose(
        np.trace(step.response.cauchy_stress.values, axis1=1, axis2=2),
        3000 * (0.002 - 3e-5 * 50),
        atol=1e-10,
    )
    for end, dt, t, delta in step.material.thermal_calls:
        np.testing.assert_allclose(t, np.interp(end, [0, 0.7, 2], [300, 340, 350]))
        np.testing.assert_allclose(
            delta, t - np.interp(end - dt, [0, 0.7, 2], [300, 340, 350]), atol=1e-12
        )


def test_thermal_restart_exact(tmp_path):
    full = build()
    full.solve()
    partial = build()
    partial.solve(until=0.5)
    checkpoint = partial.save_checkpoint(tmp_path / "mid")
    resumed = build()
    resumed.load_checkpoint(checkpoint)
    resumed.solve()
    for expected, actual in zip(
        full.material_point_histories, resumed.material_point_histories, strict=True
    ):
        np.testing.assert_array_equal(expected.abscissa, actual.abscissa)
        np.testing.assert_array_equal(expected.values, actual.values)


@pytest.mark.parametrize("value", [0.0, -1.0, True, np.inf])
def test_invalid_temperature_rejected(value):
    with pytest.raises(ValueError):
        make_step(temperature=value)


def test_temperature_requires_physical_duration():
    from test_small_strain_material_step import LinearBatchMaterial

    with pytest.raises(ValueError, match="duration"):
        make_step(LinearBatchMaterial(), duration=None, temperature=300.0)


def test_changed_thermal_path_rejects_restart_atomically(tmp_path):
    source = build()
    source.solve(until=0.5)
    checkpoint = source.save_checkpoint(tmp_path / "mid")
    target = make_step(
        ThermalMaterial(),
        temperature=300.0,
        material_points=(results.material_point("a", cell=0),),
    )
    target.solve(until=0.25)
    displacement = target.solution.x.array.copy()
    frame = target._point_history.payload()
    with pytest.raises(ValueError):
        target.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(target.solution.x.array, displacement)
    assert target._point_history.payload() == frame
    assert target.time == 0.5


def test_input_is_frozen_and_result_temperature_is_accepted():
    thermal = path()
    step = make_step(ThermalMaterial(), temperature=thermal)
    thermal.record(2.0, 900.0)
    result = step.solve_result()
    np.testing.assert_allclose(result.fields["TEMP"].field.x.array, 350.0)
    assert result.metadata["prescribed_temperature"]["times"] == [0.0, 0.7, 2.0]


def test_free_expansion_uses_material_owned_thermal_strain():
    from agentfem import fields, mesh, models, studies, steps
    from mpi4py import MPI

    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        cell_type="hexahedron",
        comm=MPI.COMM_SELF,
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain
    )
    u = model.field(fields.displacement(domain))
    model.material(ThermalMaterial())
    for component, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0.0), component=component)
    step = model.step(
        target=u,
        duration=2.0,
        temperature=path(),
        incrementation=steps.fixed(4),
        progress=False,
    )
    step.solve()
    np.testing.assert_allclose(
        step._accepted_strain,
        np.broadcast_to(0.0005 * np.eye(3), step._accepted_strain.shape),
        atol=1e-12,
    )
    np.testing.assert_allclose(step.response.cauchy_stress.values, 0.0, atol=1e-9)


def test_cutback_does_not_accumulate_trial_temperature_or_age():
    from agentfem import steps

    material = ThermalMaterial()
    material.max_dt = 0.15
    step = make_step(
        material,
        temperature=path(),
        material_points=(results.material_point("a", cell=0),),
        increments=steps.automatic(
            initial=0.5,
            maximum=0.5,
            minimum=0.001,
            growth_factor=1.5,
            cutback_factor=0.5,
        ),
    )
    step.solve()
    assert len(step.attempted_increments) > len(step.accepted_increments)
    np.testing.assert_allclose(step.state.committed["AGE"].values, 2.0, atol=1e-12)
    for end, dt, t, delta in material.thermal_calls:
        np.testing.assert_allclose(
            delta,
            t - np.interp(end - dt, [0.0, 0.7, 2.0], [300.0, 340.0, 350.0]),
            atol=1e-12,
        )
    thermal = next(h for h in step.material_point_histories if h.name == "a.TEMP")
    assert len(thermal.abscissa) == len(step.accepted_increments)
    np.testing.assert_allclose(
        thermal.values.reshape(-1),
        np.interp(thermal.abscissa, [0.0, 0.7, 2.0], [300.0, 340.0, 350.0]),
    )


def test_temperature_and_stress_survive_real_vtk_warp(tmp_path):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    step = build()
    output = tmp_path / "results.xdmf"
    step.solve_result(output=output)
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(str(output))
    reader.Update()
    grid = reader.GetOutputDataObject(0)
    assert grid.IsA("vtkUnstructuredGrid")
    np.testing.assert_allclose(vtk_to_numpy(grid.GetCellData().GetArray("TEMP")), 350.0)
    warp = vtk.vtkWarpVector()
    warp.SetInputData(grid)
    warp.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U"
    )
    warp.Update()
    assert warp.GetOutput().GetNumberOfCells() == grid.GetNumberOfCells()
    for name in ("S_CELL", "TEMP", "AGE_CELL"):
        assert warp.GetOutput().GetCellData().GetArray(name) is not None

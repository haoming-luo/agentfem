"""Numerical composition, not experimental multiaxial P92 qualification."""

import numpy as np
import pytest
from mpi4py import MPI
from agentfem import amplitudes, fields, loads, mesh, models, results, steps, studies
from tools.p92_tensor_material_reference import P92TensorResearchMaterial
from tools.p92_published_recovery_reference import implicit_update, published_parameters
from agentfem import constitutive as c
from dataclasses import replace

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size != 1, reason="Serial physical-time route"
)


def point(material):
    return c.SmallStrainMaterialPointInput(
        strain_old=np.zeros((3, 3)),
        strain_new=np.array(
            [
                [0.004, 0.0007, -0.0003],
                [0.0007, -0.001, 0.0002],
                [-0.0003, 0.0002, -0.001],
            ]
        ),
        time=1.0,
        time_increment=1.0,
        parameters=material.parameters,
        state_old=np.zeros(18),
        state_schema=material.state_schema,
        parameter_schema=material.parameter_schema,
    )


@pytest.mark.parametrize("temperature", [600, 650])
def test_public_tangent_checker_includes_shear_conversion(temperature):
    material = P92TensorResearchMaterial(temperature_C=temperature, poisson=0.3)
    request = point(material)
    evidence = c.check_small_strain_material_tangent(
        material, request, relative_step=1e-8
    )
    assert evidence.accepted
    np.testing.assert_array_equal(request.state_old, np.zeros(18))


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 600.0},
        {"temperature_increment": 0.0},
        {"field_variables": {"x": 1.0}},
    ],
)
def test_unsupported_fields_rejected(change):
    material = P92TensorResearchMaterial(temperature_C=600, poisson=0.3)
    with pytest.raises(ValueError):
        material.update(replace(point(material), **change))


def make_step(temperature=600, *, cyclic=False, increments=None, traction=None,
              material_points=()):
    domain = mesh.cuboid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1))
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain
    )
    u = model.field(fields.displacement(domain))
    model.material(P92TensorResearchMaterial(temperature_C=temperature, poisson=0.3))
    for axis, component in (("x", 0), ("y", 1), ("z", 2)):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0.0), component=component)
    end = mesh.face(domain, axis="x", value=1.0)
    if traction is None:
        model.fix(u, on=end, component=0, value=0.004)
    else:
        model.load(loads.traction((traction, 0.0, 0.0), on=end))
    return model.step(
        target=u,
        duration=24.0 if cyclic else 8.0,
        time_unit="s",
        material_points=material_points,
        incrementation=steps.fixed(increments or (96 if cyclic else 32)),
        amplitude=(
            amplitudes.tabular(
                [0.0, 4 / 24, 8 / 24, 16 / 24, 20 / 24, 1.0],
                [0.0, 1.0, 1.0, -1.0, -1.0, 0.0],
            )
            if cyclic
            else amplitudes.tabular([0.0, 0.5, 1.0], [0.0, 1.0, 1.0])
        ),
    )


@pytest.mark.parametrize("temperature", [600, 650])
def test_structure_matches_scalar_free_lateral_history(temperature):
    step = make_step(temperature)
    step.solve()
    expected = np.zeros(5)
    for index in range(32):
        expected, _ = implicit_update(
            expected,
            0.004 / 16 if index < 16 else 0.0,
            0.25,
            published_parameters(temperature),
        )
    stress = step.response.cauchy_stress.values
    np.testing.assert_allclose(stress[..., 0, 0], expected[0], rtol=2e-7, atol=1e-6)
    np.testing.assert_allclose(stress[..., 1, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(stress[..., 2, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(step.state.committed["R"].values, expected[3], rtol=2e-7)
    np.testing.assert_allclose(step.state.committed["P"].values, expected[4], rtol=2e-7)


def test_research_structure_restart(tmp_path):
    full = make_step()
    full.solve()
    partial = make_step()
    partial.solve(until=0.5)
    path = partial.save_checkpoint(tmp_path / "hold_start")
    restored = make_step()
    restored.load_checkpoint(path)
    restored.solve()
    for name in ("SIGMA_M", "X1_DEV", "X2_DEV", "R", "P"):
        np.testing.assert_allclose(
            restored.state.committed[name].values,
            full.state.committed[name].values,
            atol=1e-10,
            rtol=1e-10,
        )


def test_research_point_history_restores_without_resetting_material_clock(tmp_path):
    from tools.p92_tensor_normalization import tensor

    selection = (results.material_point("specimen", cell=0),)
    full = make_step(material_points=selection)
    full.solve()
    partial = make_step(material_points=selection)
    partial.solve(until=.5)
    path = partial.save_checkpoint(tmp_path / "selected_hold")
    resumed = make_step(material_points=selection)
    resumed.load_checkpoint(path)
    resumed.solve()
    reference = {h.name: h for h in full.material_point_histories}
    actual = {h.name: h for h in resumed.material_point_histories}
    assert reference.keys() == actual.keys()
    for name, history in reference.items():
        np.testing.assert_array_equal(history.abscissa, actual[name].abscissa)
        np.testing.assert_array_equal(history.values, actual[name].values)
    np.testing.assert_allclose(
        actual["specimen.S"].values,
        [tensor(v) for v in actual["specimen.SIGMA_M"].values],
        atol=1e-10,
    )
    assert actual["specimen.S"].abscissa[-1] == 8.0
    # P is accumulated inelastic strain, not a signed fatigue strain input.
    assert np.all(np.diff(actual["specimen.P"].values) >= -1e-12)


@pytest.mark.parametrize(
    "poisson", [True, False, 0.3 + 0j, [0.3], float("nan"), 0.5, -1.0]
)
def test_reference_rejects_ambiguous_elastic_assumption(poisson):
    with pytest.raises(ValueError):
        P92TensorResearchMaterial(temperature_C=600, poisson=poisson)


@pytest.mark.parametrize("temperature", [600, 650])
def test_structural_time_refinement_toward_continuous_scalar_reference(temperature):
    from tools.p92_published_recovery_reference import integrate_segment

    parameters = published_parameters(temperature)
    loaded = integrate_segment(np.zeros(5), 4.0, 0.001, parameters, rtol=1e-10).y[:, -1]
    reference = integrate_segment(loaded, 4.0, 0.0, parameters, rtol=1e-10).y[:, -1]
    errors = []
    for count in (16, 32, 64):
        step = make_step(temperature, increments=count)
        step.solve()
        observed = float(np.mean(step.response.cauchy_stress.values[..., 0, 0]))
        errors.append(abs(observed - reference[0]))
    assert errors[1] < 0.7 * errors[0], errors
    assert errors[2] < 0.7 * errors[1], errors


@pytest.mark.parametrize("temperature", [600, 650])
def test_cyclic_structure_matches_scalar_at_every_reversal_and_hold(temperature):
    step = make_step(temperature, cyclic=True)
    expected = np.zeros(5)
    old_strain = 0.0
    for endpoint in (4.0, 8.0, 16.0, 20.0, 24.0):
        start = step.time
        for time in np.arange(start + 0.25, endpoint + 0.125, 0.25):
            strain = np.interp(
                time, [0, 4, 8, 16, 20, 24], [0, 0.004, 0.004, -0.004, -0.004, 0]
            )
            expected, _ = implicit_update(
                expected, strain - old_strain, 0.25, published_parameters(temperature)
            )
            old_strain = strain
        step.solve(until=endpoint / 24)
        np.testing.assert_allclose(
            step.response.cauchy_stress.values[..., 0, 0],
            expected[0],
            rtol=3e-7,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            step.state.committed["R"].values, expected[3], rtol=3e-7
        )
        np.testing.assert_allclose(
            step.state.committed["P"].values, expected[4], rtol=3e-7
        )


@pytest.mark.parametrize("temperature", [600, 650])
def test_force_controlled_creep_refines_to_independent_time_integrator(temperature):
    from scipy.integrate import solve_ivp
    from tools.p92_published_recovery_reference import rates

    parameters = published_parameters(temperature)
    old = np.zeros(4)
    for start, end in ((0.0, 4.0), (4.0, 8.0)):

        def rhs(time, history):
            stress = 300.0 * min(time / 4.0, 1.0)
            return rates(np.r_[stress, history], 0.0, parameters)[1:]

        solved = solve_ivp(
            rhs, (start, end), old, method="Radau", rtol=1e-10, atol=1e-12
        )
        assert solved.success
        old = solved.y[:, -1]
    errors = []
    for count in (16, 32, 64):
        step = make_step(temperature, increments=count, traction=300.0)
        step.solve(until=0.5)
        before_hold = step.state.committed["P"].values.copy()
        step.solve()
        accumulated = step.state.committed["P"].values
        assert np.all(accumulated > before_hold)
        np.testing.assert_allclose(
            step.response.cauchy_stress.values[..., 0, 0], 300.0, rtol=1e-8, atol=1e-6
        )
        errors.append(abs(float(np.mean(accumulated)) - old[-1]))
    assert errors[1] < 0.75 * errors[0], errors
    assert errors[2] < 0.75 * errors[1], errors

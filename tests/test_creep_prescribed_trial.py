# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Newton trials stay on the new Dirichlet space, with accepted rollback."""
import numpy as np
import pytest
from mpi4py import MPI

from agentfem import amplitudes, constitutive, fields, mesh, models, steps, studies


def make_step(family):
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (1, 1, 1),
                         cell_type="hexahedron", comm=MPI.COMM_SELF)
    model = models.create(study=studies.creep_solid(), mesh=domain)
    u = model.field(fields.displacement(domain))
    elastic = dict(young=200_000., poisson=.3, density=1.)
    creep = (constitutive.isotropic_sinh(**elastic, coefficient=1.e-5, stress_scale=100., exponent=2.)
             if family == "sinh" else constitutive.isotropic_power_law(
                 **elastic, coefficient=1.e-5, reference_stress=100., stress_exponent=3.))
    if family == "joint":
        creep = constitutive.CreepPlasticityMaterial(
            creep=creep, plastic=constitutive.J2LinearIsotropicHardening(
                young=200_000., poisson=.3, yield_stress=100., hardening_modulus=1000.))
    model.material(creep)
    for i, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=i + 1), component=i, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=1., tag=4), component=0, value=.002)
    return model.step(target=u, duration=2., progress=False, incrementation=steps.fixed(8),
                      amplitude=amplitudes.tabular([0., .5, 1., 1.5, 2.], [0., 1., 1., -1., -1.]))


def test_small_strain_plasticity_trials_obey_prescribed_boundary(monkeypatch):
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (1, 1, 1),
                         cell_type="hexahedron", comm=MPI.COMM_SELF)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(constitutive.J2LinearIsotropicHardening(
        young=200_000., poisson=.3, yield_stress=100., hardening_modulus=1000.))
    for i, axis in enumerate(("x", "y", "z")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=i + 1), component=i, value=0.)
    model.fix(u, on=mesh.face(domain, axis="x", value=1., tag=4), component=0, value=.002)
    step = model.step(target=u, progress=False, incrementation=steps.fixed(8),
                      amplitude=amplitudes.tabular([0., .25, .5, .75, 1.], [0., 1., 1., -1., -1.]))
    original = step.state.update
    calls = []

    def observe(*args, **kwargs):
        actual = step.solution.x.array.copy()
        expected = actual.copy()
        for bc in step.bcs:
            bc.set(expected)
        np.testing.assert_allclose(actual, expected, atol=1.e-15, rtol=0.)
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(step.state, "update", observe)
    step.solve()
    assert calls


@pytest.mark.parametrize("family", ["power", "sinh", "joint"])
def test_every_material_trial_obeys_target_boundary_and_failed_step_rolls_back(monkeypatch, family):
    step = make_step(family)
    original = step.state.update
    calls = []

    def observe(*args, **kwargs):
        actual = step.solution.x.array.copy()
        expected = actual.copy()
        for bc in step.bcs:
            bc.set(expected)
        np.testing.assert_allclose(actual, expected, atol=1.e-15, rtol=0.)
        calls.append(kwargs["time_end"])
        return original(*args, **kwargs)

    monkeypatch.setattr(step.state, "update", observe)
    step.solve(until=1.)
    assert calls
    saved_u = step.solution.x.array.copy()
    saved_state = step.state.snapshot()
    saved_stress = step.state.stress.values.copy()

    def fail(*args, **kwargs):
        observe(*args, **kwargs)
        raise RuntimeError("injected trial failure after imposing new boundary")

    monkeypatch.setattr(step.state, "update", fail)
    with pytest.raises(RuntimeError, match="fixed creep increment"):
        step.solve()
    assert step.accepted_time == 1.
    np.testing.assert_array_equal(step.solution.x.array, saved_u)
    # solve() refreshes stress from accepted strain/state before its attempt;
    # equivalent elastic expressions can differ in their last rounding bits.
    np.testing.assert_allclose(step.state.stress.values, saved_stress, rtol=1.e-13, atol=1.e-12)
    for name, values in saved_state.items():
        np.testing.assert_array_equal(step.state.snapshot()[name], values)
    expected = saved_u.copy()
    for bc in step.bcs:
        bc.set(expected)
    np.testing.assert_allclose(expected, saved_u, atol=1.e-15, rtol=0.)

    monkeypatch.setattr(step.state, "update", observe)
    step.solve()
    full = make_step(family)
    full.solve()
    np.testing.assert_allclose(step.solution.x.array, full.solution.x.array, atol=1.e-12)
    np.testing.assert_allclose(step.state.stress.values, full.state.stress.values, atol=1.e-7)
    for name, values in full.state.snapshot().items():
        np.testing.assert_allclose(step.state.snapshot()[name], values, atol=1.e-12)

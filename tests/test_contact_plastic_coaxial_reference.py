# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent scalar reference for a *confined*, homogeneous plastic column.

F = diag(lambda, 1, 1), Fp = diag(exp(p), exp(-p/2), exp(-p/2)).
For quadratic Hencky elasticity the trial equivalent Kirchhoff stress is
abs(2 mu log(lambda) - 3 mu p). Associative J2 flow and linear hardening
reduce consistency to one scalar increment. Equilibrium additionally requires
the nominal axial stress plus the reference-area penalty pressure to equal
the applied reference traction. No AgentFEM constitutive/contact/solver code
is used by the scalar reference. This derivation applies only to this coaxial
fixture; it is neither a second FE solver nor general external validation.
"""
from math import copysign, log1p

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
from scipy.optimize import brentq

from agentfem import amplitudes, boundary_models, constitutive, fields, mesh, models, solvers, steps, studies
from test_contact_face_science_research import _wide_closed_tool


E, NU, YIELD, HARDENING = 1000., .3, 2., 100.
MU, BULK = E / (2 * (1 + NU)), E / (3 * (1 - 2 * NU))
GAP, PENALTY, SIDE = .002, 500., .1
KNOTS, AMPLITUDES = (0., .25, .5, .75, 1.), (0., 1., -.5, 0., 1.)


def scalar_response(displacement, old_p, old_peeq):
    strain = log1p(displacement)
    signed_trial = 2 * MU * strain - 3 * MU * old_p
    plastic_increment = max(0., (abs(signed_trial) - YIELD - HARDENING * old_peeq)
                            / (3 * MU + HARDENING))
    p = old_p + copysign(plastic_increment, signed_trial)
    peeq = old_peeq + plastic_increment
    axial_kirchhoff = BULK * strain + 2 * MU * (2 * strain / 3 - p)
    nominal = axial_kirchhoff / (1 + displacement)
    return nominal, p, peeq


def scalar_equilibrium(traction, old_p, old_peeq):
    displacement = brentq(
        lambda u: scalar_response(u, old_p, old_peeq)[0]
        + PENALTY * max(0., u - GAP) - traction,
        -.1, .1, xtol=5e-15,
    )
    nominal, p, peeq = scalar_response(displacement, old_p, old_peeq)
    return displacement, nominal, p, peeq


def column(cells, integration, increments=32):
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
    domain.geometry.x[:, 1:] *= SIDE
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.), name="right")
    y = mesh.boundary(domain, lambda x: np.isclose(x[1], 0.) | np.isclose(x[1], SIDE), name="y_guides")
    z = mesh.boundary(domain, lambda x: np.isclose(x[2], 0.) | np.isclose(x[2], SIDE), name="z_guides")
    model.fix(displacement, on=left, component=0, value=0.)
    model.fix(displacement, on=y, component=1, value=0.)
    model.fix(displacement, on=z, component=2, value=0.)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=E, poisson=NU, yield_stress=YIELD, hardening_modulus=HARDENING,
    ))
    surface = (_wide_closed_tool(initial_overlap=-GAP) if integration != "gauss"
               else boundary_models.rigid_plane(point=(1 + GAP, 0., 0.), normal=(-1., 0., 0.)))
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        right, boundary_models.rigid_body(surface), penalty=PENALTY,
        initial_penetration_limit=0., name="stop",
    ))
    model.traction((30., 0., 0.), on=right)
    step = model.step(
        target=displacement, material=material,
        amplitude=amplitudes.tabular(KNOTS, AMPLITUDES, name="cyclic_force"),
        contact_integration=integration, incrementation=steps.fixed(increments), progress=False,
        solver_options=solvers.newton(relative_tolerance=1e-10, absolute_tolerance=1e-11,
                                      maximum_iterations=50, line_search="backtracking"),
    )
    return displacement.value, step


@pytest.mark.parametrize("cells", [1, 2])
@pytest.mark.parametrize("integration", ["gauss", "certified_closed_two_feature_research"])
def test_cyclic_plastic_column_matches_independent_scalar_history(cells, integration):
    field, step = column(cells, integration)
    coordinates = field.function_space.tabulate_dof_coordinates()
    p = peeq = 0.
    opened = closed = reversed_flow = False
    last_p = 0.
    for index in range(1, 33):
        factor = index / 32
        applied = 30 * np.interp(factor, KNOTS, AMPLITUDES)
        u, nominal, p, peeq = scalar_equilibrium(applied, p, peeq)
        step.solve(until=factor)
        expected_displacement = np.zeros_like(coordinates)
        expected_displacement[:, 0] = u * coordinates[:, 0]
        np.testing.assert_allclose(field.x.array.reshape(-1, 3), expected_displacement, atol=2e-9, rtol=2e-7)
        state = step.response.state.committed_state_vectors()
        expected_fp = np.diag(np.exp([p, -p / 2, -p / 2])).reshape(9)
        np.testing.assert_allclose(state[:, :9], np.broadcast_to(expected_fp, state[:, :9].shape), atol=2e-9, rtol=2e-7)
        np.testing.assert_allclose(state[:, 9], peeq, atol=2e-9, rtol=2e-7)
        np.testing.assert_allclose(state[:, 10], YIELD * peeq, atol=2e-9, rtol=2e-7)
        contact = step.contact_provider.evaluate()
        pressure = PENALTY * max(0., u - GAP)
        # Both integration routes expose the same residual contract. Physical
        # force on the structure is its negative, not a point-pressure sum.
        force = -np.sum(contact.nodal_structural_residual, axis=0)
        np.testing.assert_allclose(force, [-SIDE**2 * pressure, 0., 0.], atol=2e-9, rtol=2e-7)
        assert nominal + pressure == pytest.approx(applied, abs=2e-10)
        opened |= pressure == 0 and peeq > 0
        closed |= pressure > 0
        reversed_flow |= p < last_p - 1e-5
        last_p = p
    assert opened and closed and reversed_flow and peeq > .01


def test_scalar_reference_is_not_an_elastic_only_or_monotone_path():
    p = peeq = 0.
    history = []
    for factor in np.arange(1, 33) / 32:
        u, _, p, peeq = scalar_equilibrium(30 * np.interp(factor, KNOTS, AMPLITUDES), p, peeq)
        history.append((u, p, peeq))
    values = np.asarray(history)
    assert np.any(np.diff(values[:, 1]) < -1e-4)
    assert np.all(np.diff(values[:, 2]) >= 0)
    assert np.any(values[:, 0] > GAP) and np.any(values[:, 0] < GAP)

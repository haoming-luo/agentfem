# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Nonuniform Poisson-coupled plastic compression: independent enforcement.

The reference uses the same finite-J2 bulk material but no contact Operator,
search, multiplier or penalty. Normal reactions come from the full residual
at a separately prescribed displacement boundary. This is an enforcement
reference on each mesh, not an independent constitutive or continuum solution.
"""

import numpy as np
import pytest
from dolfinx import mesh as dm
from mpi4py import MPI

from agentfem import (
    boundary_models, constitutive, fields, mesh, models, solvers, steps, studies,
)


def _build(cells, *, penalty=None):
    domain = dm.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    u = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.), name="right")
    model.clamp(u, on=left)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000., poisson=.3, yield_stress=2., hardening_modulus=100.,
    ))
    options = {}
    if penalty is None:
        model.fix(u, on=right, component=0, value=-.02)
    else:
        tool = boundary_models.rigid_body(
            boundary_models.rigid_plane(point=(1., 0., 0.), normal=(-1., 0., 0.)),
            motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                boundary_models.prescribed_rigid_motion(translation=(-.02, 0., 0.)),
                start_time=0., end_time=1.,
            ),
        )
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            right, tool, penalty=penalty, initial_penetration_limit=0.,
        ))
        options['normal_augmentation'] = steps.normal_augmentation(
            penetration_tolerance=1e-10, complementarity_tolerance=1e-9,
            maximum_augmentations=120, length_unit='m', pressure_unit='Pa',
        )
    return model.step(
        target=u, material=material, incrementation=steps.fixed(8), progress=False,
        solver_options=solvers.newton(
            relative_tolerance=1e-9, absolute_tolerance=1e-10,
            maximum_iterations=50, line_search='backtracking',
        ),
        **options,
    )


@pytest.mark.parametrize('cells', [1, 2, 3])
def test_nonuniform_plastic_press_matches_no_contact_displacement_reference(cells):
    reference = _build(cells)
    states = []
    for factor in np.linspace(1/8, 1., 8):
        reference.solve(until=float(factor))
        states.append((
            reference.solution.x.array.copy(),
            reference.response.state.committed_state_vectors().copy(),
            reference.reaction_field().x.array.reshape(-1, 3).copy(),
            reference.energy_history[-1].prescribed_motion_work,
        ))
    assert np.max(states[-1][1][:, 9]) > 0.
    for penalty in (1000., 5000.):
        target = _build(cells, penalty=penalty)
        for factor, (u, state, reaction, work) in zip(
            np.linspace(1/8, 1., 8), states, strict=True,
        ):
            target.solve(until=float(factor))
            provider = target.contact_provider
            sample = provider.accepted_point_samples()
            nodes = provider.adapter.trace._direct_nodes
            pressure = -reaction[nodes, 0] / sample['reference_weights']
            # Independent constrained solution must be physically admissible:
            # otherwise imposing displacement could artificially pull the body.
            assert np.min(pressure) > 0.
            assert np.ptp(pressure) > .1
            np.testing.assert_allclose(target.solution.x.array, u, atol=3e-8, rtol=0.)
            np.testing.assert_allclose(
                target.response.state.committed_state_vectors(), state,
                atol=1e-7, rtol=1e-5,
            )
            np.testing.assert_allclose(sample['normal_pressures'], pressure,
                                       atol=2e-5, rtol=1e-5)
            frame = target.energy_history[-1]
            assert frame.tool_work == pytest.approx(work, abs=2e-7)
            assert frame.normal_contact_work_bound < 1e-8
            assert np.max(sample['penetrations']) <= 1e-10

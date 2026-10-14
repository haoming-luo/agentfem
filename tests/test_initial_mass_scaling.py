# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Initial reference planning is reproducible, not nonlinear-path certification."""
from types import SimpleNamespace

import numpy as np
import pytest

from agentfem import elements, time
from agentfem.time._initial_mass_scaling import _deformable_reference_rows
from test_response_material_step import make_step, contact_step
from test_response_deformable_step import make_step as deformable_step


def policy(**changes):
    controls = dict(target_increment=1e-4, reference_modulus=1e8,
                    maximum_added_mass_ratio=100.)
    controls.update(changes)
    return time.InitialMassScaling(**controls)


@pytest.mark.parametrize('storage', ('fields', 'owned_arrays'))
def test_generated_mass_matches_explicit_factor_execution(storage):
    generated = make_step(mass_scaling=policy(), material_storage=storage)
    factors = generated.numerical_inertia.factors
    assert factors.max() > 1
    explicit = make_step(mass_scaling_factor=factors, material_storage=storage)
    generated.run()
    explicit.run()
    for field in ('u', 'v', 'a'):
        np.testing.assert_array_equal(getattr(generated.state, field).value.x.array,
                                      getattr(explicit.state, field).value.x.array)
    assert generated.history_records == explicit.history_records
    report = generated.summary()['mass_scaling_preparation']
    assert report['generated_from_current_mesh']
    assert not report['automatic_stability_certificate']
    assert not report['nonlinear_material_tangent_included']
    physical = report['increment_diagnostics']['physical_mass']
    numerical = report['increment_diagnostics']['numerical_mass']
    assert physical['classification'] == 'outside_supplied_bound'
    assert numerical['classification'] in ('inside_supplied_bound', 'roundoff_boundary')
    assert not numerical['nonlinear_path_certified']
    assert len(numerical['controllers'][0]['reference_coordinates']) == 3


def test_initial_planner_restart_identity(tmp_path):
    complete = make_step(mass_scaling=policy())
    complete.run()
    interrupted = make_step(mass_scaling=policy())
    interrupted.run(until_step=2)
    checkpoint = interrupted.save_checkpoint(tmp_path / 'planned')
    resumed = make_step(mass_scaling=policy())
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    assert resumed.history_records == complete.history_records
    np.testing.assert_array_equal(resumed.state.u.value.x.array, complete.state.u.value.x.array)
    # Even an estimate producing identical unscaled masses is part of identity.
    unchanged_mass = make_step(mass_scaling=policy(reference_modulus=1.))
    unchanged_mass.run(until_step=2)
    checkpoint = unchanged_mass.save_checkpoint(tmp_path / 'identity')
    changed = make_step(mass_scaling=policy(reference_modulus=2.))
    np.testing.assert_array_equal(unchanged_mass.numerical_inertia.factors,
                                  changed.numerical_inertia.factors)
    with pytest.raises(ValueError):
        changed.load_checkpoint(checkpoint)
    assert changed.completed_steps == 0


@pytest.mark.parametrize('cylinder', (False, True))
def test_actual_bond_tools_and_damping_are_present(cylinder):
    step = contact_step(bond=True, cylinder=cylinder, mass_scaling=policy(),
                        bulk_viscosity=elements.BulkViscosityPolicy(1., 1., 1.))
    preparation = step.summary()['mass_scaling_preparation']
    assert preparation['plan']['stiffness_contributions'] == [
        'body_hourglass', 'cohesive', 'rigid_contact:contact_0', 'rigid_contact:contact_1']
    assert preparation['plan']['damping_contributions'] == ['initial_bulk_viscosity']
    assert all(value > 0 for value in preparation['maximum_unscaled_stiffness_rows'].values())
    step.run()


def test_deformable_neighborhood_is_not_silently_omitted():
    step = deformable_step(mass_scaling=policy())
    report = step.summary()['mass_scaling_preparation']
    assert len(report['operator_details']) == 1
    detail = next(iter(report['operator_details'].values()))
    assert detail['points_with_initial_neighborhood'] == detail['candidate_points'] > 0
    assert detail['maximum_candidate_facets'] > 0
    assert not detail['moving_projection_geometric_tangent_included']
    step.run()


def test_candidate_diagonal_dominates_frozen_q1_stencil_without_double_counting():
    # Two adjacent patches share vertices. Every candidate is admitted;
    # the shared nodes must not receive repeated mass for the same point.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [2, 0, 0], [2, 1, 0]], dtype=float)
    quads = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    pair = SimpleNamespace(
        secondary=SimpleNamespace(vertices=np.array([[1, .5, .01]]), nodal_measures=np.array([.25])),
        primary=SimpleNamespace(vertices=vertices, quadrilaterals=quads),
        candidate_nodes=np.array([True]), secondary_dofs=np.array([6]),
        primary_dofs=np.arange(6), penalty=100., search_radius=.2)
    rows, report = _deformable_reference_rows(pair, 7)
    assert report['maximum_candidate_facets'] == 2
    np.testing.assert_array_equal(rows, 50.)
    rng = np.random.default_rng(164)
    for quad in quads:
        for _ in range(10):
            s, t = rng.uniform(size=2)
            weights = np.array([(1-s)*(1-t), s*(1-t), s*t, (1-s)*t])
            b = np.zeros(7)
            b[6] = 1
            b[quad] = -weights
            stiffness = .25*100*np.outer(b, b)
            assert np.linalg.eigvalsh(np.diag(rows[:, 0])-stiffness).min() > -1e-12


def test_reject_budget_and_increment_without_clipping():
    with pytest.raises(ValueError, match='added-mass budget'):
        make_step(mass_scaling=policy(maximum_added_mass_ratio=0.))
    with pytest.raises(ValueError, match='target increment'):
        make_step(dt=2e-4, mass_scaling=policy())


@pytest.mark.parametrize('changes', (
    {'target_increment': False}, {'target_increment': np.nan},
    {'reference_modulus': 0.}, {'reference_modulus': -1.},
    {'reference_modulus': np.inf}, {'reference_modulus': True},
    {'maximum_added_mass_ratio': -1.}, {'safety_factor': 0.}, {'safety_factor': 1.},
))
def test_invalid_controls(changes):
    with pytest.raises(ValueError):
        policy(**changes)

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded accepted normal-work ledger, not an exact continuum energy identity."""

import json

import numpy as np
import pytest

from agentfem import amplitudes
from test_finite_j2_normal_augmentation import _problem, _state


def _cycle(increments=16):
    path = amplitudes.tabular((0., .25, .5, .625, .75, 1.), (0., 1., 0., -.5, 0., 1.))
    return _problem(cells=1, penalty=5000., increments=increments,
                    transverse_boundary_fixed=True, load_amplitude=path,
                    penetration_tolerance=1e-9, complementarity_tolerance=1e-8,
                    maximum_augmentations=120)[1]


def test_normal_work_matches_independent_nodal_force_path_and_restarts(tmp_path):
    full = _cycle()
    previous_force = -full.contact_provider.evaluate().nodal_structural_residual.copy()
    previous_u = full.solution.x.array.reshape(-1, 3).copy()
    work = 0.
    for factor in np.linspace(1/16, 1., 16):
        full.solve(until=float(factor))
        force = -full.contact_provider.evaluate().nodal_structural_residual
        current_u = full.solution.x.array.reshape(-1, 3).copy()
        work += float(np.sum(.5*(previous_force + force)*(current_u-previous_u)))
        frame = full.energy_history[-1]
        assert frame.normal_contact_work == pytest.approx(work, abs=1e-13)
        assert abs(work) <= frame.normal_contact_work_bound + 1e-14
        assert frame.contact_potential == 0.
        previous_force, previous_u = force.copy(), current_u

    partial = _cycle()
    partial.solve(until=.625)  # Reverse plasticity has occurred.
    checkpoint = partial.save_checkpoint(tmp_path / 'reverse')
    resumed = _cycle()
    resumed.load_checkpoint(checkpoint)
    result = resumed.solve_result()
    for actual, expected in zip(resumed.energy_history, full.energy_history, strict=True):
        for key, value in expected.as_dict().items():
            assert actual.as_dict()[key] == pytest.approx(value, abs=1e-12)
    assert result.metadata['mechanical_energy_ledger']['requires_increment_refinement']
    assert 'normal_contact_corrected_energy_residual' in result.histories
    assert 'contact_potential' not in result.histories


@pytest.mark.parametrize('damage', ['bound', 'missing', 'coverage', 'nonfinite_load', 'work_jump'])
def test_corrupted_normal_work_ledger_rejects_without_restoring_state(tmp_path, damage):
    partial = _cycle()
    partial.solve(until=.25)
    checkpoint = partial.save_checkpoint(tmp_path / 'good')
    with np.load(checkpoint, allow_pickle=False) as data:
        archive = {key: data[key].copy() for key in data.files}
    records = json.loads(str(archive['energy_history']))
    if damage == 'bound':
        records[-1]['normal_contact_work_bound'] = -1.
    elif damage == 'missing':
        for name in ('normal_contact_work', 'normal_contact_work_bound', 'normal_contact_open_gap_work'):
            records[-1].pop(name)
    elif damage == 'coverage':
        records.pop()
    elif damage == 'nonfinite_load':
        records[-1]['load_factor'] = float('nan')
    else:
        records[-1]['normal_contact_work'] = records[-1]['normal_contact_work_bound']
        records[-1]['normal_contact_open_gap_work'] = records[-1]['normal_contact_work_bound']
    archive['energy_history'] = np.asarray(json.dumps(records))
    path = tmp_path / 'bad.npz'
    np.savez(path, **archive)
    target = _cycle()
    before_u, before_state = target.solution.x.array.copy(), _state(target)
    with pytest.raises(ValueError):
        target.load_checkpoint(path)
    np.testing.assert_array_equal(target.solution.x.array, before_u)
    np.testing.assert_array_equal(_state(target), before_state)
    assert target.energy_history == []


def test_legacy_missing_work_remains_unavailable_after_resume(tmp_path):
    partial = _cycle()
    partial.solve(until=.25)
    checkpoint = partial.save_checkpoint(tmp_path / 'good')
    with np.load(checkpoint, allow_pickle=False) as data:
        archive = {key: data[key].copy() for key in data.files}
    archive['energy_history'] = np.asarray('[]')
    archive['energy_history_complete'] = np.asarray(False)
    legacy = tmp_path / 'legacy.npz'
    np.savez(legacy, **archive)
    target = _cycle()
    target.load_checkpoint(legacy)
    result = target.solve_result()
    assert target.energy_history == []
    assert result.metadata['mechanical_energy_ledger']['status'].startswith('unavailable_legacy')
    assert 'normal_contact_work' not in result.histories


def test_post_recording_failure_rolls_back_energy_and_contact_together(monkeypatch):
    target = _cycle()
    target.solve(until=.25)
    energy = list(target.energy_history)
    u, state = target.solution.x.array.copy(), _state(target)
    contact = target.contact_provider.snapshot_accepted_state()
    def fail():
        raise RuntimeError('injected joint commit failure')
    monkeypatch.setattr(target.contact_provider, 'commit_accepted', fail)
    with pytest.raises(RuntimeError, match='injected joint commit failure'):
        target.solve(until=.5)
    assert target.energy_history == energy
    np.testing.assert_array_equal(target.solution.x.array, u)
    np.testing.assert_array_equal(_state(target), state)
    assert target.contact_provider.snapshot_accepted_state() == contact


def test_cyclic_energy_matches_scalar_reference_and_increment_refinement():
    from math import log
    from test_normal_augmentation_coaxial_reference import _equilibrium

    observations = []
    for increments in (16, 32, 64, 128):
        problem = _cycle(increments)
        path = amplitudes.tabular((0., .25, .5, .625, .75, 1.), (0., 1., 0., -.5, 0., 1.))
        a = p = traction_old = pressure_old = external = contact = bound = 0.
        stretch_old = 1.
        force_variation = 0.
        for factor in np.linspace(1/increments, 1., increments):
            traction = 10. * float(path(factor))
            stretch, a, p, pressure = _equilibrium(traction, a, p)
            delta = stretch - stretch_old
            external += .5 * (traction + traction_old) * delta
            contact -= .5 * (pressure + pressure_old) * delta
            bound += .5 * (pressure + pressure_old) * abs(delta)
            force_variation += abs(traction-traction_old)
            stretch_old, traction_old, pressure_old = stretch, traction, pressure
        # Hencky elastic energy + isotropic hardening energy; dissipation Y*p.
        stored = 500. * ((log(stretch)-a)**2 + .5*a*a) + 50.*p*p
        internal = stored + 2.*p
        problem.solve()
        frame = problem.energy_history[-1]
        assert frame.stored_energy == pytest.approx(stored, abs=2e-8)
        assert frame.plastic_dissipation == pytest.approx(2.*p, abs=2e-8)
        # Summation by parts propagates the independently checked 3e-9
        # stretch error into external work. Accumulated work cannot inherit
        # the tolerance of a single endpoint energy evaluation.
        work_tolerance = 3e-9 * (abs(traction) + force_variation)
        assert frame.external_work == pytest.approx(external, abs=work_tolerance)
        assert frame.normal_contact_work == pytest.approx(contact, abs=work_tolerance)
        assert frame.normal_contact_work_bound == pytest.approx(bound, abs=work_tolerance)
        corrected = frame.mechanical_energy_residual + frame.normal_contact_work
        assert corrected == pytest.approx(external + contact - internal, abs=2*work_tolerance+4e-8)
        observations.append((increments, frame.mechanical_energy_residual, corrected,
                             frame.normal_contact_work_bound))
    print('increment refinement (n, raw residual, contact-aware residual, work bound):', observations)
    assert abs(observations[-1][2]) < abs(observations[0][2]) / 4.
    assert observations[-1][3] < observations[0][3] / 4.

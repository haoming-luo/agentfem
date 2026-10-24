# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed tilted obstacle: a changing, spatially partial plastic contact patch.

Comparing acceleration policies tests execution equivalence, not independent
constitutive validation. The equality-constrained elastic partial-contact
oracle lives in test_finite_j2_normal_augmentation.py.
"""

from dataclasses import asdict
import json

import numpy as np
import pytest

from agentfem import amplitudes
from test_finite_j2_normal_augmentation import _problem, _state


def _partial_cycle(acceleration):
    return _problem(
        cells=2, increments=16, gap=.001, tilt=.02,
        penalty=5000., applied_traction=10., yield_stress=2.,
        transverse_boundary_fixed=True,
        load_amplitude=amplitudes.tabular((0., .5, .75, 1.), (0., 1., 0., 1.)),
        penetration_tolerance=1e-9, complementarity_tolerance=1e-8,
        maximum_augmentations=200, acceleration=acceleration,
    )[1]


def _capture(problem):
    sample = problem.contact_provider.accepted_point_samples()
    return (
        problem.solution.x.array.copy(), _state(problem),
        np.asarray(sample['normal_pressures']).copy(),
        np.asarray(sample['active']).copy(), asdict(problem.energy_history[-1]),
    )


def _compare(actual, expected):
    for a, b, tolerance in zip(actual[:3], expected[:3], (5e-8, 2e-7, 5e-5), strict=True):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=1e-6)
    np.testing.assert_array_equal(actual[3], expected[3])
    assert actual[4].keys() == expected[4].keys()
    for name, value in expected[4].items():
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            assert actual[4][name] == pytest.approx(value, abs=2e-7, rel=1e-6), name
        else:
            assert actual[4][name] == value, name


def test_partial_plastic_cycle_acceleration_and_restart_match_every_accepted_state(tmp_path):
    reference = _partial_cycle('none')
    accelerated = _partial_cycle('aitken')
    states = []
    for factor in np.linspace(1/16, 1., 16):
        reference.solve(until=float(factor))
        accelerated.solve(until=float(factor))
        states.append(_capture(accelerated))
        _compare(states[-1], _capture(reference))
        diagnostic = accelerated.contact_provider.current_point_diagnostics()['normal_augmentation']
        assert diagnostic['maximum_penetration'] <= 1e-9
        assert diagnostic['maximum_complementarity'] <= 1e-8
        if factor == .5:
            checkpoint = accelerated.save_checkpoint(tmp_path / 'partial-contact')

    activity = [state[3] for state in states]
    assert any(np.any(active) and np.any(~active) for active in activity)
    assert any(not np.array_equal(a, b) for a, b in zip(activity, activity[1:]))
    assert np.max(states[-1][1][:, 9]) > 0.
    assert any(info.normal_relaxation_factors for info in accelerated.accepted_increments)
    restored = _partial_cycle('aitken')
    restored.load_checkpoint(checkpoint)
    for factor, expected in zip(np.linspace(9/16, 1., 8), states[8:], strict=True):
        restored.solve(until=float(factor))
        _compare(_capture(restored), expected)


@pytest.mark.parametrize('acceleration', ['none', 'aitken'])
def test_failed_partial_contact_commit_restores_patch_and_can_retry(monkeypatch, acceleration):
    problem = _partial_cycle(acceleration)
    problem.solve(until=.5)
    before = _capture(problem)
    assert np.any(before[3]) and np.any(~before[3])
    provider = problem.contact_provider
    saved_contact = json.dumps(provider.snapshot_accepted_state(), sort_keys=True)
    history = list(problem.energy_history)
    accepted = list(problem.accepted_increments)
    commit = provider.commit_accepted

    def fail_after_contact_commit():
        commit()
        raise RuntimeError('injected partial-patch commit failure')

    with monkeypatch.context() as patch:
        patch.setattr(provider, 'commit_accepted', fail_after_contact_commit)
        with pytest.raises(RuntimeError, match='injected partial-patch commit failure'):
            problem.solve(until=9/16)

    assert problem.accepted_load_factor == .5
    assert problem.accepted_increments == accepted
    assert problem.energy_history == history
    assert json.dumps(provider.snapshot_accepted_state(), sort_keys=True) == saved_contact
    after = _capture(problem)
    for a, b in zip(after[:4], before[:4], strict=True):
        np.testing.assert_array_equal(a, b)
    assert after[4] == before[4]
    problem.solve(until=9/16)
    uninterrupted = _partial_cycle(acceleration)
    uninterrupted.solve(until=9/16)
    _compare(_capture(problem), _capture(uninterrupted))

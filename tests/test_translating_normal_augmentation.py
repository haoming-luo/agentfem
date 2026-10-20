# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Constant-normal prescribed plane: pressure follows the material-side trace."""

import json

import numpy as np
import pytest

from agentfem import amplitudes, results
from test_finite_j2_facet_contact import _patch
from test_finite_j2_normal_augmentation import _problem, _state
from test_normal_augmentation_coaxial_reference import _response


def _moving(*, increments=16, amplitude=None, translation=(-.02, 0., 0.), **kwargs):
    return _problem(
        penalty=5000., increments=increments, transverse_boundary_fixed=True,
        applied_traction=0., gap=.005, tool_translation=translation,
        load_amplitude=amplitude, penetration_tolerance=1e-9,
        complementarity_tolerance=1e-8, maximum_augmentations=120, **kwargs,
    )[1]


@pytest.mark.parametrize('cells', [1, 2])
def test_translating_stop_pressure_and_work_match_scalar_reference(cells):
    problem = _moving(cells=cells)
    a = p = work = old_force = 0.
    old_tool_x = 1.005
    previous_u = problem.solution.x.array.reshape(-1, 3).copy()
    previous_force = -problem.contact_provider.evaluate().nodal_structural_residual
    structural_work = 0.
    for factor in np.linspace(1/16, 1., 16):
        tool_x = 1.005 - .02*factor
        stretch = min(tool_x, 1.)
        nominal, a, p = _response(stretch, a, p)
        work += .5*(nominal+old_force)*(tool_x-old_tool_x)
        old_force, old_tool_x = nominal, tool_x
        problem.solve(until=float(factor))
        current_force = -problem.contact_provider.evaluate().nodal_structural_residual
        current_u = problem.solution.x.array.reshape(-1, 3).copy()
        structural_work += float(np.sum(.5*(previous_force+current_force)*(current_u-previous_u)))
        previous_u, previous_force = current_u.copy(), current_force.copy()
        frame = problem.energy_history[-1]
        fp = np.diag(np.exp([a, -a/2, -a/2])).ravel()
        np.testing.assert_allclose(
            _state(problem)[:, :9], np.broadcast_to(fp, _state(problem)[:, :9].shape),
            atol=3e-8, rtol=0.,
        )
        np.testing.assert_allclose(_state(problem)[:, 9], p, atol=3e-8, rtol=0.)
        np.testing.assert_allclose(_state(problem)[:, 10], 2*p, atol=6e-8, rtol=0.)
        np.testing.assert_allclose(problem.contact_provider.evaluate().contact_force_on_structure,
                                   [nominal, 0., 0.], atol=3e-6, rtol=0.)
        assert frame.tool_work == pytest.approx(work, abs=2e-7)
        assert frame.natural_load_work == 0.
        assert frame.tool_work + frame.normal_contact_work == pytest.approx(structural_work, abs=1e-12)
        assert problem.contact_provider.snapshot_accepted_state()['motion_factor'] == factor
    public_problem = _moving(cells=cells)
    result = public_problem.solve_result()
    np.testing.assert_allclose(public_problem.solution.x.array, problem.solution.x.array,
                               atol=1e-12)
    assert 'tool_work' in result.histories
    assert frame.tool_work > 0.
    assert 'contact_potential' not in result.histories


def test_tangential_tool_translation_changes_neither_pressure_nor_work():
    normal = _moving()
    oblique = _moving(translation=(-.02, .3, -.1))
    normal.solve()
    oblique.solve()
    np.testing.assert_allclose(oblique.solution.x.array, normal.solution.x.array, atol=1e-12)
    np.testing.assert_allclose(_state(oblique), _state(normal), atol=1e-12)
    for a, b in zip(oblique.energy_history, normal.energy_history, strict=True):
        assert a.tool_work == pytest.approx(b.tool_work, abs=1e-12)
        assert a.normal_contact_work == pytest.approx(b.normal_contact_work, abs=1e-12)
    # The material trace identity is independent of the prescribed rigid pose.
    assert oblique.contact_provider.normal_multiplier_state.accepted_snapshot()['trace_identity'] == (
        normal.contact_provider.normal_multiplier_state.accepted_snapshot()['trace_identity']
    )


def test_moving_equilibrium_and_checkpoint_refuse_a_changed_tool_pose():
    problem = _moving()
    problem.solve(until=.5)
    provider = problem.contact_provider
    provider.normal_response()
    provider.update_motion(.6)
    with pytest.raises(RuntimeError, match='another displacement or tool pose'):
        provider.finish_frozen_equilibrium(residual_norm=0., residual_tolerance=1e-9)
    with pytest.raises(ValueError, match='another tool pose'):
        provider.snapshot_accepted_state()
    with pytest.raises(ValueError, match='another tool pose'):
        provider.dual_evidence(problem)
    provider.update_motion(.5)
    provider.snapshot_accepted_state()


def test_provisional_tool_dual_cannot_escape_joint_work_recording(monkeypatch):
    problem = _moving()
    problem.solve(until=.5)  # A previous successful solve must not license trial output.
    original = problem._record_energy
    calls = []

    def record(*args, **kwargs):
        with pytest.raises(RuntimeError, match='reserved for the joint'):
            problem.contact_provider.dual_evidence(problem)
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(problem, '_record_energy', record)
    problem.solve()
    assert calls
    problem.contact_provider.dual_evidence(problem)  # Public only after the joint commit.


@pytest.mark.parametrize('augmented', [False, True])
def test_cyclic_point_export_uses_resolved_tool_pose_not_step_coordinate(
    tmp_path, monkeypatch, augmented,
):
    amplitude = amplitudes.tabular((0., .5, .75, 1.), (0., 1., 0., 1.))
    problem = (_moving(amplitude=amplitude) if augmented else _patch(
        triangulated=False, initial_penetration_limit=0., increments=16,
        applied_traction=0., tool_translation=(-.02, 0., 0.),
        load_amplitude=amplitude, transverse_boundary_fixed=True,
    )[3])
    for coordinate, pose in ((.5, 1.), (.75, 0.)):
        problem.solve(until=coordinate)
        paths = results.export_accepted_contact_points(problem, tmp_path / str(coordinate))
        metadata = json.loads(paths['contact_point_metadata'].read_text())
        assert metadata['coordinate_value'] == coordinate
        assert metadata['resolved_load_amplitude'] == pose
        assert metadata['pairs'][0]['tool_motion_factor'] == pose
        provider = problem.contact_provider
        sample = provider.accepted_point_samples()

        for index, stale in enumerate((coordinate, float('nan'))):
            def stale_pose(sample=sample, stale=stale):
                return {**sample, 'load_factor': stale}

            rejected = tmp_path / f'rejected-{coordinate}-{index}'
            with monkeypatch.context() as patch:
                patch.setattr(provider, 'accepted_point_samples', stale_pose)
                with pytest.raises(RuntimeError, match='AFM-CONTACT-EXPORT-POSE'):
                    results.export_accepted_contact_points(problem, rejected)
            assert not rejected.exists()


@pytest.mark.parametrize("acceleration", ["none", "aitken"])
def test_translating_cyclic_energy_matches_independent_reference_and_refines(acceleration):
    from math import log

    amplitude = amplitudes.tabular((0., .5, .75, 1.), (0., 1., 0., 1.))
    observations = []
    for increments in (16, 32, 64, 128):
        a = p = force_old = tool_work = relative_work = bound = 0.
        old_tool, old_gap = 1.005, .005
        for coordinate in np.linspace(1/increments, 1., increments):
            tool = 1.005 - .02*float(amplitude(coordinate))
            lo, hi = .9, 1.05
            for _ in range(55):
                mid = (lo+hi)/2
                if _response(mid, a, p)[0] < 0.:
                    lo = mid
                else:
                    hi = mid
            stretch = min(tool, (lo+hi)/2)  # Unilateral zero-traction equilibrium.
            force, a, p = _response(stretch, a, p)
            gap = tool-stretch
            tool_work += .5*(force+force_old)*(tool-old_tool)
            relative_work -= .5*(force+force_old)*(gap-old_gap)
            bound -= .5*(force+force_old)*abs(gap-old_gap)
            force_old, old_tool, old_gap = force, tool, gap
        stored = 500.*((log(stretch)-a)**2 + .5*a*a) + 50.*p*p
        problem = _moving(increments=increments, amplitude=amplitude, cells=1,
                          acceleration=acceleration)
        problem.solve()
        frame = problem.energy_history[-1]
        assert frame.stored_energy == pytest.approx(stored, abs=3e-8)
        assert frame.plastic_dissipation == pytest.approx(2.*p, abs=3e-8)
        assert frame.tool_work == pytest.approx(tool_work, abs=1e-7)
        assert frame.normal_contact_work == pytest.approx(relative_work, abs=1e-7)
        assert frame.normal_contact_work_bound == pytest.approx(bound, abs=1e-7)
        corrected = frame.mechanical_energy_residual + frame.normal_contact_work
        assert corrected == pytest.approx(tool_work+relative_work-stored-2*p, abs=3e-7)
        observations.append((increments, frame.tool_work, frame.mechanical_energy_residual,
                             corrected, frame.normal_contact_work_bound))
    print('moving-cycle refinement (n, tool work, raw residual, corrected residual, bound):',
          observations)
    assert abs(observations[-1][3]) < abs(observations[0][3])/4
    assert observations[-1][4] < observations[0][4]/4


@pytest.mark.parametrize("acceleration", ["none", "aitken"])
def test_translating_cycle_restart_and_atomic_failure(tmp_path, monkeypatch, acceleration):
    amplitude = amplitudes.tabular((0., .5, .75, 1.), (0., 1., 0., 1.))
    full = _moving(amplitude=amplitude, acceleration=acceleration)
    full.solve()
    partial = _moving(amplitude=amplitude, acceleration=acceleration)
    partial.solve(until=.5)
    path = partial.save_checkpoint(tmp_path / 'loaded')
    resumed = _moving(amplitude=amplitude, acceleration=acceleration)
    resumed.load_checkpoint(path)
    resumed.solve(until=.75)
    provider = resumed.contact_provider
    saved = provider.snapshot_accepted_state()
    energy = list(resumed.energy_history)
    u, state = resumed.solution.x.array.copy(), _state(resumed)
    original = provider.commit_accepted
    def fail():
        original()
        raise RuntimeError('failure after contact commit')
    monkeypatch.setattr(provider, 'commit_accepted', fail)
    with pytest.raises(RuntimeError, match='failure after contact commit'):
        resumed.solve()
    assert provider.snapshot_accepted_state() == saved
    assert provider.snapshot_motion() == .0  # amplitude is zero at the .75 station
    assert resumed.energy_history == energy
    np.testing.assert_array_equal(resumed.solution.x.array, u)
    np.testing.assert_array_equal(_state(resumed), state)
    monkeypatch.setattr(provider, 'commit_accepted', original)
    resumed.solve()
    np.testing.assert_allclose(resumed.solution.x.array, full.solution.x.array, atol=1e-12)
    np.testing.assert_allclose(_state(resumed), _state(full), atol=1e-12)
    for actual, expected in zip(resumed.energy_history, full.energy_history, strict=True):
        for key, value in expected.as_dict().items():
            assert actual.as_dict()[key] == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize('damage', ['factor', 'identity'])
def test_wrong_tool_checkpoint_is_atomically_rejected(tmp_path, damage):
    partial = _moving()
    partial.solve(until=.5)
    path = partial.save_checkpoint(tmp_path / 'loaded')
    with np.load(path, allow_pickle=False) as source:
        archive = {key: source[key].copy() for key in source.files}
    contact = json.loads(str(archive['contact_state']))
    if damage == 'factor':
        contact['motion_factor'] = .25
    else:
        contact['motion_identity'] = {}
    archive['contact_state'] = np.asarray(json.dumps(contact))
    broken = tmp_path / 'broken.npz'
    np.savez(broken, **archive)
    target = _moving()
    saved = target.contact_provider.snapshot_accepted_state()
    with pytest.raises(ValueError, match='motion identity or coordinate'):
        target.load_checkpoint(broken)
    assert target.contact_provider.snapshot_accepted_state() == saved
    assert target.contact_provider.snapshot_motion() == 0.


@pytest.mark.parametrize("saved_policy,target_policy", [("none", "aitken"), ("aitken", "none")])
def test_restart_cannot_silently_change_multiplier_acceleration(tmp_path, saved_policy, target_policy):
    source = _moving(acceleration=saved_policy)
    source.solve(until=.5)
    path = source.save_checkpoint(tmp_path / "saved")
    target = _moving(acceleration=target_policy)
    before = target.contact_provider.snapshot_accepted_state()
    u, state = target.solution.x.array.copy(), _state(target)
    with pytest.raises(ValueError):
        target.load_checkpoint(path)
    assert target.contact_provider.snapshot_accepted_state() == before
    np.testing.assert_array_equal(target.solution.x.array, u)
    np.testing.assert_array_equal(_state(target), state)
    assert target.accepted_load_factor == 0.

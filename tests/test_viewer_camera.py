# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Camera framing is resolution independent and contains the complete model."""
import itertools
import numpy as np
import pytest

from agentfem._viewer_camera import fit_camera, orbit_pose, text_size


def test_relative_typography():
    assert text_size(24, [3840, 2160]) == 2*text_size(24, [1920, 1080])
    assert text_size(18, [1280, 720]) == 12


@pytest.mark.parametrize('aspect', [.25, .5, 1920/1080])
@pytest.mark.parametrize('padding', [1.03, 1.12])
def test_all_orbit_frames_contain_bounds_despite_ui_zoom(aspect, padding):
    pv = pytest.importorskip('pyvista')
    camera = pv.Camera()
    camera.position = (100., 40., 15.)
    camera.focal_point = (99., 39., 14.)
    camera.parallel_scale = .0001
    bounds = [-50, 50, -2, 2, -1, 1]
    fit_camera(camera, bounds, aspect, padding=padding)
    base = camera.position, camera.focal_point, camera.up
    corners = np.array(list(itertools.product(*np.array(bounds).reshape(3, 2))))
    for style in ('showcase', 'turntable'):
        for phase in np.linspace(0, 1, 121):
            position, focal, up = orbit_pose(*base, phase, style)
            forward = (focal-position)/np.linalg.norm(focal-position)
            right = np.cross(forward, up)
            relative = corners-position
            depth = relative@forward
            vertical_limit = depth*np.tan(np.deg2rad(camera.view_angle)/2)
            assert np.all(depth > 0)
            assert np.all(abs(relative@up) < vertical_limit)
            assert np.all(abs(relative@right) < vertical_limit*aspect)
            assert np.max(abs((corners-focal)@up)) < camera.parallel_scale
        np.testing.assert_allclose(orbit_pose(*base, 0, style), orbit_pose(*base, 1, style), atol=1e-12)


def test_unknown_motion_is_rejected():
    with pytest.raises(ValueError):
        orbit_pose((1, 1, 1), (0, 0, 0), (0, 0, 1), .5, 'unknown')


def test_interactive_fit_is_tighter_without_changing_export_default():
    pv = pytest.importorskip('pyvista')
    camera = pv.Camera()
    camera.position = (100., 40., 15.)
    camera.focal_point = (0., 0., 0.)
    bounds = [-50, 50, -2, 2, -1, 1]
    _, export_distance = fit_camera(camera, bounds, 16/9)
    export_scale = camera.parallel_scale
    _, interactive_distance = fit_camera(camera, bounds, 16/9, padding=1.03)
    assert interactive_distance/export_distance == pytest.approx(1.03/1.12)
    assert camera.parallel_scale/export_scale == pytest.approx(1.03/1.12)


@pytest.mark.parametrize('padding', [.99, float('nan'), float('inf')])
def test_invalid_fit_padding(padding):
    with pytest.raises(ValueError, match='padding'):
        fit_camera(None, None, 1., padding=padding)


def test_showcase_has_steady_spatial_speed_without_breathing_zoom():
    poses = [orbit_pose((3, 3, 3), (0, 0, 0), (0, 0, 1), t) for t in np.linspace(0, 1, 241)]
    positions = np.array([p[0] for p in poses])
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert speed.max()/speed.min() < 1.02
    distance = np.linalg.norm(positions, axis=1)
    assert distance.max()/distance.min() == pytest.approx(1.)
    assert .1 < np.ptp(positions[:, 2]) < 2

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic display-only framing and restrained camera choreography."""

import numpy as np


def fit_camera(camera, bounds, aspect, *, padding=1.12):
    """Enclose the bounds' sphere at every orientation, independent of UI zoom."""
    if not np.isfinite(padding) or padding < 1.:
        raise ValueError("Camera padding must be finite and at least 1.")
    box = np.asarray(bounds, dtype=float).reshape(3, 2)
    center = box.mean(axis=1)
    radius = max(float(np.linalg.norm(box[:, 1]-box[:, 0]))/2, 1e-12)
    direction = np.asarray(camera.position)-camera.focal_point
    direction /= np.linalg.norm(direction)
    half_angle = np.deg2rad(camera.view_angle)/2
    half_angle = min(half_angle, np.arctan(np.tan(half_angle)*aspect))
    distance = radius*padding/np.sin(half_angle)
    camera.focal_point = center
    camera.position = center+direction*distance
    camera.parallel_scale = radius*padding/min(1., aspect)
    return center, distance


def orbit_pose(position, focal, up, phase, style="showcase"):
    """Full level orbit or a restrained 45-degree oblique showcase loop."""
    if style not in {"showcase", "turntable"}:
        raise ValueError("Select Showcase or Turntable camera motion.")
    t = float(np.clip(phase, 0., 1.))
    focal = np.asarray(focal, dtype=float)
    offset = np.asarray(position, dtype=float)-focal
    vertical = np.asarray(up, dtype=float)
    vertical /= np.linalg.norm(vertical)
    def route(u):
        phase_angle = 2*np.pi*u
        angle = np.deg2rad(22.5)*np.cos(phase_angle) if style == 'showcase' else phase_angle
        rotated = (offset*np.cos(angle)+np.cross(vertical, offset)*np.sin(angle)
                   +vertical*np.dot(vertical, offset)*(1-np.cos(angle)))
        if style == 'showcase':
            right = np.cross(vertical, rotated)
            right /= np.linalg.norm(right)
            lift = np.deg2rad(6)*np.sin(phase_angle)
            rotated = rotated*np.cos(lift)+np.cross(right, rotated)*np.sin(lift)
            # Constant distance: no breathing zoom, roll or underside orbit.
        return rotated
    u = t
    if style == 'showcase':
        stations = np.linspace(0., 1., 257)
        positions = np.array([route(s) for s in stations])
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
        u = float(np.interp(t*distance[-1], distance, stations))
    rotated = route(u)
    direction = rotated/np.linalg.norm(rotated)
    screen_up = vertical-direction*np.dot(vertical, direction)
    screen_up /= np.linalg.norm(screen_up)
    return focal+rotated, focal, screen_up


def text_size(points, size):
    """Constant relative typography at HD, Full HD and 4K."""
    return max(8, round(points*size[1]/1080))

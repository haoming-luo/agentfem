# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private batched polar-frame kinematics, without material or State ownership.

For F=R U, R.T D R = sym(Udot inv(U)). Integrate this rate along the
explicitly declared *linear right-stretch path*. Endpoint rotations alone do
not specify the actual deformation history. This is not an implementation of
Abaqus' undisclosed increment algorithm, nor log(U_new)-log(U_old) in general.
"""

from dataclasses import dataclass

import numpy as np


def _transpose(value):
    return np.swapaxes(value, -1, -2)


def _checked(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    array = np.asarray(value, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape (points, 3, 3).")
    return array


def _readonly(value):
    value = np.array(value, dtype=float, copy=True)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite corotational output; reject the increment.")
    value.setflags(write=False)
    return value


def _spectral(vectors, values):
    return (vectors * values[..., None, :]) @ _transpose(vectors)


def _polar(gradient, *, include_roots=False):
    identity = np.all(gradient == np.eye(3), axis=(1, 2))
    if np.any(identity):
        rotation = np.broadcast_to(np.eye(3), gradient.shape).copy()
        stretch = rotation.copy()
        outputs = [rotation, stretch]
        if include_roots:
            outputs.extend((rotation.copy(), rotation.copy()))
        if not np.all(identity):
            selected = _polar(gradient[~identity], include_roots=include_roots)
            for output, values in zip(outputs, selected, strict=True):
                output[~identity] = values
        return tuple(outputs)
    left, stretches, right = np.linalg.svd(gradient)
    rotation = left @ right
    if np.any(np.linalg.det(rotation) <= 0):
        raise ValueError("Deformation gradients must preserve orientation.")
    if np.any(stretches[:, -1] <= 1.0e-12 * stretches[:, 0]):
        raise ValueError("Deformation gradients are singular or ill-conditioned.")
    vectors = _transpose(right)
    stretch = _spectral(vectors, stretches)
    if include_roots:
        # Right singular vectors already diagonalize U. Do not diagonalize
        # the reconstructed stretch a second time just to obtain its roots.
        return (
            rotation,
            stretch,
            _spectral(vectors, np.sqrt(stretches)),
            _spectral(vectors, 1 / np.sqrt(stretches)),
        )
    return rotation, stretch


@dataclass(frozen=True)
class CorotationalIncrement:
    """Material-frame tensor strain increment and material-to-spatial frames."""

    strain_increment: np.ndarray
    frame_old: np.ndarray
    frame_new: np.ndarray
    method: str = "polar_linear_right_stretch_exact_integral"


def polar_strain_increment(
    gradient_old, gradient_new, *, reference_frame=None, integration_rule="exact"
):
    """Return one batch; proper reference material axes may vary by point.

    The integral is evaluated analytically using a symmetric generalized
    eigenproblem, not a Python loop over points or repeated quadrature. With
    B=U0^(-1/2) U1 U0^(-1/2), its unsymmetrized value is
    U0^(1/2) log(B) U0^(-1/2). This follows by integrating
    (U1-U0) inv(U0+s*(U1-U0)) over s in [0,1].

    The opt-in ``midpoint`` rule approximates that same path integral at s=1/2.
    It is second order under subdivision, not an exact finite logarithmic
    increment. Providers must include this choice in their numerical identity.
    """
    if integration_rule not in ("exact", "midpoint"):
        raise ValueError("Unknown polar right-stretch integration rule.")
    method = (
        "polar_linear_right_stretch_exact_integral"
        if integration_rule == "exact"
        else "polar_linear_right_stretch_midpoint"
    )
    old = _checked(gradient_old, "gradient_old")
    new = _checked(gradient_new, "gradient_new")
    if old.shape != new.shape:
        raise ValueError("Old and new batches must have equal shape.")
    count = len(old)
    frame = (
        np.broadcast_to(np.eye(3), old.shape)
        if reference_frame is None
        else reference_frame
    )
    frame = _checked(frame, "reference_frame")
    if frame.shape != old.shape:
        raise ValueError("Reference frames must match the point count.")
    if not np.allclose(
        _transpose(frame) @ frame,
        np.broadcast_to(np.eye(3), old.shape),
        rtol=0,
        atol=1.0e-12,
    ) or np.any(np.linalg.det(frame) <= 0):
        raise ValueError("Reference frames must be proper orthonormal axes.")
    if count == 0:
        return CorotationalIncrement(
            *(_readonly(np.empty((0, 3, 3))) for _ in range(3)), method=method
        )
    stationary = np.all(old == new, axis=(1, 2))
    if np.any(stationary):
        strain = np.zeros_like(old)
        r0 = np.empty_like(old)
        r1 = np.empty_like(old)
        stationary_rotation, _ = _polar(old[stationary])
        r0[stationary] = r1[stationary] = stationary_rotation @ frame[stationary]
        if not np.all(stationary):
            moving = polar_strain_increment(
                old[~stationary],
                new[~stationary],
                reference_frame=frame[~stationary],
                integration_rule=integration_rule,
            )
            strain[~stationary] = moving.strain_increment
            r0[~stationary], r1[~stationary] = moving.frame_old, moving.frame_new
        return CorotationalIncrement(
            _readonly(strain), _readonly(r0), _readonly(r1), method
        )
    if integration_rule == "midpoint":
        r0, u0 = _polar(old)
        r1, u1 = _polar(new)
        integral = _transpose(
            np.linalg.solve(_transpose((u0 + u1) / 2), _transpose(u1 - u0))
        )
    else:
        r0, u0, root, inverse_root = _polar(old, include_roots=True)
        r1, u1 = _polar(new)
        # Form B-I directly: log1p retains very small increments accurately.
        delta = inverse_root @ (u1 - u0) @ inverse_root
        values, vectors = np.linalg.eigh(0.5 * (delta + _transpose(delta)))
        if np.any(values <= -1):
            raise ValueError("Right-stretch path is numerically singular.")
        logarithm = _spectral(vectors, np.log1p(values))
        integral = root @ logarithm @ inverse_root
    strain = _transpose(frame) @ (0.5 * (integral + _transpose(integral))) @ frame
    return CorotationalIncrement(
        _readonly(strain), _readonly(r0 @ frame), _readonly(r1 @ frame), method
    )


def push_corotational_stress(stress, frame):
    """Rotate symmetric local Cauchy stress; no change of stress measure."""
    stress = _checked(stress, "stress")
    frame = _checked(frame, "frame")
    if stress.shape != frame.shape:
        raise ValueError("Stress and frame point counts differ.")
    if not np.allclose(
        _transpose(frame) @ frame,
        np.broadcast_to(np.eye(3), frame.shape),
        rtol=0,
        atol=1.0e-12,
    ) or np.any(np.linalg.det(frame) <= 0):
        raise ValueError("Frames must be proper orthonormal axes.")
    scale = np.maximum(np.max(np.abs(stress), axis=(1, 2)), np.finfo(float).tiny)
    if np.any(
        np.max(np.abs(stress - _transpose(stress)), axis=(1, 2)) > 1.0e-12 * scale
    ):
        raise ValueError("Cauchy stress must be symmetric.")
    return _readonly(frame @ stress @ _transpose(frame))

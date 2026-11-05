# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit-frame, two-endpoint rotation evidence for discrete local updates.

This is a verifier, not a material adapter or an objective-rate correction.
It never commits state and never infers history frames from array shapes.
"""

from dataclasses import dataclass, replace
from collections.abc import Mapping
from numbers import Real

import numpy as np

from .user_material import (
    MaterialPointInput,
    _VerificationMaterial,
    validated_material_update,
)


@dataclass(frozen=True)
class MaterialRotationCheck:
    """Measured covariance of one discrete update, not global objectivity proof."""

    errors: tuple[tuple[str, float], ...]
    tolerance: float
    old_rotation: tuple[tuple[float, ...], ...]
    new_rotation: tuple[tuple[float, ...], ...]
    state_frames: tuple[tuple[str, str], ...]
    stress_scale: float | None
    state_scales: tuple[tuple[str, float], ...]

    @property
    def accepted(self):
        return all(error <= self.tolerance for _, error in self.errors)

    def summary(self):
        return {
            "kind": "material_rotation_check",
            "scope": "declared_frame_two_endpoint_discrete_update",
            "not_general_objectivity_proof": True,
            "accepted": self.accepted,
            "tolerance": self.tolerance,
            "relative_errors": dict(self.errors),
            "old_rotation": [list(row) for row in self.old_rotation],
            "new_rotation": [list(row) for row in self.new_rotation],
            "state_frames": dict(self.state_frames),
            "stress_scale": self.stress_scale,
            "state_scales": dict(self.state_scales),
            "normalization": "max(observed_norms, declared_characteristic_scale)",
            "properties": "unchanged_scalar_or_reference_parameters",
            "external_fields": "not_admitted",
            "energy_assessed": False,
        }


def _rotation(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be a real proper orthogonal 3x3 matrix.")
    selected = np.asarray(value, dtype=float)
    if (
        selected.shape != (3, 3)
        or not np.isfinite(selected).all()
        or np.max(np.abs(selected)) > 1 + 1e-12
        or not np.allclose(selected.T @ selected, np.eye(3), rtol=0, atol=1e-12)
        or not np.isclose(np.linalg.det(selected), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError(f"{name} must be a real proper orthogonal 3x3 matrix.")
    return selected.copy()


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive.")
    selected = float(value)
    if not np.isfinite(selected) or selected <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return selected


def _relative_error(actual, expected, characteristic=None):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Rotation evidence requires finite transformed responses.")
    scale = max(
        float(np.max(np.abs(actual), initial=0)),
        float(np.max(np.abs(expected), initial=0)),
        0 if characteristic is None else characteristic,
    )
    if scale == 0:
        return 0.0
    a, b = actual / scale, expected / scale
    denominator = max(
        float(np.linalg.norm(a)),
        float(np.linalg.norm(b)),
        0 if characteristic is None else characteristic / scale,
    )
    return float(np.linalg.norm(a - b) / denominator)


def check_material_rotation(
    material,
    point: MaterialPointInput,
    rotation,
    *,
    state_frames: Mapping[str, str],
    old_rotation=None,
    tolerance: float = 1e-8,
    stress_scale: float | None = None,
    state_scales: Mapping[str, float] | None = None,
) -> MaterialRotationCheck:
    """Check stress, dP/dF and named history under declared observer rotations.

    New F is left-multiplied by ``rotation``; old F by ``old_rotation`` (default:
    the same rotation). Every history variable must explicitly be ``invariant``
    (scalar/reference/corotated), ``spatial_vector`` (shape 3), or
    ``spatial_tensor`` (shape 3x3). Old and returned spatial history use their
    respective endpoint rotations. Mixed tensors require a separate adapter.

    Properties remain scalar/reference parameters; scalar temperature is
    unchanged. Nonempty external-field arrays are refused because their frames
    are undeclared. Only reference dP/dF is admitted, including declared 9x9
    component permutations. The tangent test is covariance, not differentiation;
    run ``check_material_tangent`` separately.

    Errors use per-channel norms, with optional positive characteristic scales
    in that channel's physical units. Supply scales when testing near-zero
    responses: there is no hidden dimensional unit-size floor. State variables
    are checked separately to avoid mixing units or hiding a small history.
    No energy, general objective-rate, path-wide or experimental claim follows.
    """
    checked = _VerificationMaterial(material)
    convention = checked.tangent_convention
    standard = tuple(f"{i}{j}" for i in range(1, 4) for j in range(1, 4))
    if (
        convention.stress_measure != "first_piola"
        or convention.kinematic_measure != "deformation_gradient"
        or convention.configuration != "reference"
        or convention.storage != "matrix_9x9"
        or set(convention.component_order) != set(standard)
    ):
        raise ValueError("Rotation checking requires a declared reference 9x9 dP/dF.")
    Q = _rotation(rotation, "rotation")
    Qn = Q.copy() if old_rotation is None else _rotation(old_rotation, "old_rotation")
    tolerance = _positive(tolerance, "tolerance")
    if stress_scale is not None:
        stress_scale = _positive(stress_scale, "stress_scale")
    if point.field_variables is not None and point.field_variables.size:
        raise NotImplementedError(
            "Rotation checking requires declared external-field frames; nonempty field_variables are not admitted."
        )
    if not isinstance(state_frames, Mapping):
        raise TypeError(
            "state_frames must explicitly map every history variable to its frame."
        )
    frames = dict(state_frames)
    variables = checked.state_schema.variables
    names = {v.name for v in variables}
    if set(frames) != names:
        raise ValueError(
            "state_frames must declare every history variable exactly once."
        )
    for variable in variables:
        frame = frames[variable.name]
        if (
            not isinstance(frame, str)
            or frame not in {"invariant", "spatial_vector", "spatial_tensor"}
            or (frame == "spatial_vector" and variable.shape != (3,))
            or (frame == "spatial_tensor" and variable.shape != (3, 3))
        ):
            raise ValueError(
                f"Invalid rotation frame/shape for state {variable.name!r}."
            )
    if state_scales is not None and not isinstance(state_scales, Mapping):
        raise TypeError("state_scales must be a named mapping.")
    scales = dict(state_scales or {})
    if set(scales) - names:
        raise ValueError("state_scales contains an undeclared state variable.")
    scales = {
        name: _positive(value, f"state_scales[{name!r}]")
        for name, value in scales.items()
    }

    def transformed_history(state, frame_rotation):
        unpacked = checked.state_schema.unpack(state)
        values = []
        for variable in variables:
            value = np.asarray(unpacked[variable.name])
            if frames[variable.name] == "spatial_vector":
                value = frame_rotation @ value
            elif frames[variable.name] == "spatial_tensor":
                value = frame_rotation @ value @ frame_rotation.T
            values.append(value.reshape(-1))
        return np.concatenate(values) if values else np.empty(0)

    baseline = validated_material_update(checked, point)
    rotated_point = replace(
        point,
        deformation_gradient_old=Qn @ point.deformation_gradient_old,
        deformation_gradient_new=Q @ point.deformation_gradient_new,
        state_old=transformed_history(point.state_old, Qn),
    )
    rotated = validated_material_update(checked, rotated_point)
    index = [convention.component_order.index(label) for label in standard]
    tangent = baseline.consistent_tangent[np.ix_(index, index)].reshape(3, 3, 3, 3)
    rotated_tangent = rotated.consistent_tangent[np.ix_(index, index)]
    F, Fr = point.deformation_gradient_new, rotated_point.deformation_gradient_new
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        P = np.linalg.det(F) * baseline.cauchy_stress @ np.linalg.inv(F).T
        Pr = np.linalg.det(Fr) * rotated.cauchy_stress @ np.linalg.inv(Fr).T
        errors = [
            (
                "cauchy_stress",
                _relative_error(
                    rotated.cauchy_stress,
                    Q @ baseline.cauchy_stress @ Q.T,
                    stress_scale,
                ),
            ),
            ("first_piola_stress", _relative_error(Pr, Q @ P, stress_scale)),
            (
                "algorithmic_tangent",
                _relative_error(
                    rotated_tangent,
                    np.einsum("ia,ajbl,kb->ijkl", Q, tangent, Q).reshape(9, 9),
                    stress_scale,
                ),
            ),
        ]
    expected = checked.state_schema.unpack(transformed_history(baseline.state_new, Q))
    actual = checked.state_schema.unpack(rotated.state_new)
    for variable in variables:
        errors.append(
            (
                f"state:{variable.name}",
                _relative_error(
                    np.asarray(actual[variable.name]),
                    np.asarray(expected[variable.name]),
                    scales.get(variable.name),
                ),
            )
        )
    return MaterialRotationCheck(
        tuple(errors),
        tolerance,
        tuple(tuple(float(v) for v in row) for row in Qn),
        tuple(tuple(float(v) for v in row) for row in Q),
        tuple((v.name, frames[v.name]) for v in variables),
        stress_scale,
        tuple(sorted(scales.items())),
    )

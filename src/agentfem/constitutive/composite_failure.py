# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Material-axis ply failure assessments, separate from damage evolution."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, inf, isfinite, pi, sin
from typing import Mapping, Protocol, runtime_checkable

import numpy as np


def _positive(value: float, *, name: str) -> float:
    selected = float(value)
    if not isfinite(selected) or selected <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return selected


def _material_stress(value) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != (3,) or not np.all(np.isfinite(selected)):
        raise ValueError(
            "material_stress must contain finite (sigma_11, sigma_22, tau_12)."
        )
    return selected


@dataclass(frozen=True)
class CompositeStrengths2D:
    """Plane-stress unidirectional-ply strengths in material axes."""

    longitudinal_tension: float
    longitudinal_compression: float
    transverse_tension: float
    transverse_compression: float
    in_plane_shear: float
    name: str = "composite_strengths_2d"

    def __post_init__(self) -> None:
        for field_name in (
            "longitudinal_tension",
            "longitudinal_compression",
            "transverse_tension",
            "transverse_compression",
            "in_plane_shear",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive(getattr(self, field_name), name=field_name),
            )
        name = str(self.name).strip()
        if not name:
            raise ValueError("CompositeStrengths2D.name must be non-empty.")
        object.__setattr__(self, "name", name)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "composite_strengths_2d",
            "name": self.name,
            "longitudinal_tension": self.longitudinal_tension,
            "longitudinal_compression": self.longitudinal_compression,
            "transverse_tension": self.transverse_tension,
            "transverse_compression": self.transverse_compression,
            "in_plane_shear": self.in_plane_shear,
            "stress_order": ["sigma_11", "sigma_22", "tau_12"],
        }


@runtime_checkable
class PlyFailureCriterion(Protocol):
    """Extension contract for one material-axis ply failure surface."""

    name: str

    def evaluate(
        self,
        material_stress: np.ndarray,
        strengths: CompositeStrengths2D,
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class MaximumStress2D:
    """Sign-aware maximum-stress screening for a plane-stress ply."""

    name: str = "maximum_stress_2d"

    def evaluate(self, material_stress, strengths) -> Mapping[str, float]:
        sigma_1, sigma_2, tau_12 = _material_stress(material_stress)
        return {
            "fiber_tension": max(sigma_1, 0.0) / strengths.longitudinal_tension,
            "fiber_compression": max(-sigma_1, 0.0)
            / strengths.longitudinal_compression,
            "matrix_tension": max(sigma_2, 0.0) / strengths.transverse_tension,
            "matrix_compression": max(-sigma_2, 0.0)
            / strengths.transverse_compression,
            "in_plane_shear": abs(tau_12) / strengths.in_plane_shear,
        }


@dataclass(frozen=True)
class Hashin2D:
    """Plane-stress Hashin fibre/matrix initiation indices."""

    name: str = "hashin_2d"

    def evaluate(self, material_stress, strengths) -> Mapping[str, float]:
        sigma_1, sigma_2, tau_12 = _material_stress(material_stress)
        shear = (tau_12 / strengths.in_plane_shear) ** 2
        indices = {
            "fiber_tension": 0.0,
            "fiber_compression": 0.0,
            "matrix_tension": 0.0,
            "matrix_compression": 0.0,
        }
        if sigma_1 >= 0.0:
            indices["fiber_tension"] = (
                sigma_1 / strengths.longitudinal_tension
            ) ** 2 + shear
        else:
            indices["fiber_compression"] = (
                sigma_1 / strengths.longitudinal_compression
            ) ** 2
        if sigma_2 >= 0.0:
            indices["matrix_tension"] = (
                sigma_2 / strengths.transverse_tension
            ) ** 2 + shear
        else:
            ratio = strengths.transverse_compression / (
                2.0 * strengths.in_plane_shear
            )
            indices["matrix_compression"] = (
                sigma_2 / (2.0 * strengths.in_plane_shear)
            ) ** 2 + (ratio**2 - 1.0) * (
                sigma_2 / strengths.transverse_compression
            ) + shear
        return indices


@dataclass(frozen=True)
class TsaiWu2D:
    """Plane-stress Tsai--Wu surface with explicit normalized interaction."""

    interaction: float
    name: str = "tsai_wu_2d"

    def __post_init__(self) -> None:
        selected = float(self.interaction)
        if not isfinite(selected) or abs(selected) >= 1.0:
            raise ValueError(
                "TsaiWu2D.interaction must be finite and lie strictly between -1 and 1."
            )
        object.__setattr__(self, "interaction", selected)

    def evaluate(self, material_stress, strengths) -> Mapping[str, float]:
        sigma_1, sigma_2, tau_12 = _material_stress(material_stress)
        f1 = 1.0 / strengths.longitudinal_tension - 1.0 / strengths.longitudinal_compression
        f2 = 1.0 / strengths.transverse_tension - 1.0 / strengths.transverse_compression
        f11 = 1.0 / (
            strengths.longitudinal_tension * strengths.longitudinal_compression
        )
        f22 = 1.0 / (
            strengths.transverse_tension * strengths.transverse_compression
        )
        f66 = 1.0 / strengths.in_plane_shear**2
        f12 = self.interaction * np.sqrt(f11 * f22)
        return {
            "combined": float(
                f1 * sigma_1
                + f2 * sigma_2
                + f11 * sigma_1**2
                + f22 * sigma_2**2
                + 2.0 * f12 * sigma_1 * sigma_2
                + f66 * tau_12**2
            )
        }


@dataclass(frozen=True)
class PlyFailureAssessment:
    """Per-mode initiation indices and proportional first-failure factor."""

    criterion: str
    material_stress: np.ndarray
    indices: Mapping[str, float]
    governing_mode: str
    maximum_index: float
    load_factor_to_first_failure: float
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "ply_failure_assessment",
            "criterion": self.criterion,
            "material_stress": self.material_stress.tolist(),
            "stress_order": ["sigma_11", "sigma_22", "tau_12"],
            "indices": dict(self.indices),
            "governing_mode": self.governing_mode,
            "maximum_index": self.maximum_index,
            "load_factor_to_first_failure": self.load_factor_to_first_failure,
            "accepted": self.accepted,
            "interpretation": "initiation_screening_not_damage_evolution",
        }


@dataclass(frozen=True)
class PlyPointFailureAssessment:
    """One laminate section point assessed in its ply material frame."""

    point_id: str
    ply_name: str
    ply_index: int
    point_index: int
    z: float
    material_stress: np.ndarray
    assessment: PlyFailureAssessment

    def as_dict(self) -> dict[str, object]:
        return {
            "point_id": self.point_id,
            "ply_name": self.ply_name,
            "ply_index": self.ply_index,
            "point_index": self.point_index,
            "z": self.z,
            "material_stress": self.material_stress.tolist(),
            "assessment": self.assessment.as_dict(),
        }


@dataclass(frozen=True)
class LaminateFailureAssessment:
    """Stable section-point assessments and the governing laminate location."""

    points: tuple[PlyPointFailureAssessment, ...]
    governing_point_id: str
    governing_mode: str
    maximum_index: float
    load_factor_to_first_failure: float
    accepted: bool

    def by_id(self, point_id: str) -> PlyPointFailureAssessment:
        for point in self.points:
            if point.point_id == point_id:
                return point
        raise KeyError(f"Unknown laminate section point {point_id!r}.")

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "laminate_failure_assessment",
            "governing_point_id": self.governing_point_id,
            "governing_mode": self.governing_mode,
            "maximum_index": self.maximum_index,
            "load_factor_to_first_failure": self.load_factor_to_first_failure,
            "accepted": self.accepted,
            "points": [point.as_dict() for point in self.points],
            "interpretation": "first_ply_initiation_not_progressive_damage",
        }


_CRITERIA = {
    "maximum_stress": MaximumStress2D(),
    "maximum_stress_2d": MaximumStress2D(),
    "hashin": Hashin2D(),
    "hashin_2d": Hashin2D(),
}


def _criterion(value: str | PlyFailureCriterion) -> PlyFailureCriterion:
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _CRITERIA:
            raise ValueError(
                f"Unknown ply failure criterion {value!r}; choose maximum_stress_2d, "
                "hashin_2d, or supply a PlyFailureCriterion."
            )
        return _CRITERIA[key]
    if not isinstance(value, PlyFailureCriterion):
        raise TypeError("criterion must be a name or PlyFailureCriterion.")
    return value


def _maximum_index(
    criterion: PlyFailureCriterion,
    stress: np.ndarray,
    strengths: CompositeStrengths2D,
) -> float:
    indices = criterion.evaluate(stress, strengths)
    if not indices:
        raise ValueError("PlyFailureCriterion.evaluate must return at least one mode.")
    values = np.asarray(tuple(indices.values()), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Ply failure indices must be finite.")
    return float(max(0.0, np.max(values)))


def _first_failure_factor(
    criterion: PlyFailureCriterion,
    stress: np.ndarray,
    strengths: CompositeStrengths2D,
) -> float:
    if np.allclose(stress, 0.0):
        return inf
    lower = 0.0
    upper = 1.0
    while _maximum_index(criterion, upper * stress, strengths) < 1.0:
        upper *= 2.0
        if upper > 1.0e12:
            return inf
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if _maximum_index(criterion, midpoint * stress, strengths) < 1.0:
            lower = midpoint
        else:
            upper = midpoint
    return float(0.5 * (lower + upper))


def composite_strengths_2d(
    *,
    xt: float,
    xc: float,
    yt: float,
    yc: float,
    s12: float,
    name: str = "composite_strengths_2d",
) -> CompositeStrengths2D:
    """Create the common five-strength plane-stress ply contract."""

    return CompositeStrengths2D(
        longitudinal_tension=xt,
        longitudinal_compression=xc,
        transverse_tension=yt,
        transverse_compression=yc,
        in_plane_shear=s12,
        name=name,
    )


def assess_ply_failure(
    material_stress,
    strengths: CompositeStrengths2D,
    *,
    criterion: str | PlyFailureCriterion = "hashin_2d",
) -> PlyFailureAssessment:
    """Assess one plane-stress material-axis state without evolving damage."""

    if not isinstance(strengths, CompositeStrengths2D):
        raise TypeError("strengths must be CompositeStrengths2D.")
    stress = _material_stress(material_stress)
    selected = _criterion(criterion)
    raw = selected.evaluate(stress, strengths)
    indices = {name: float(value) for name, value in raw.items()}
    maximum = _maximum_index(selected, stress, strengths)
    governing = max(indices, key=indices.get)
    return PlyFailureAssessment(
        criterion=selected.name,
        material_stress=stress.copy(),
        indices=indices,
        governing_mode=governing,
        maximum_index=maximum,
        load_factor_to_first_failure=_first_failure_factor(
            selected,
            stress,
            strengths,
        ),
        accepted=maximum <= 1.0,
    )


def _stress_to_material_axes(stress, angle_degrees: float) -> np.ndarray:
    selected = _material_stress(stress)
    angle = float(angle_degrees) * pi / 180.0
    rotation = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    global_tensor = np.array(
        [[selected[0], selected[2]], [selected[2], selected[1]]]
    )
    local = rotation.T @ global_tensor @ rotation
    return np.array([local[0, 0], local[1, 1], local[0, 1]])


def assess_laminate_failure(
    section,
    response,
    strengths: CompositeStrengths2D | Mapping[str, CompositeStrengths2D],
    *,
    criterion: str | PlyFailureCriterion = "hashin_2d",
) -> LaminateFailureAssessment:
    """Assess all recovered section points in their named ply material axes."""

    plies = tuple(getattr(section, "plies", ()))
    recovered = tuple(getattr(response, "section_points", ()))
    if not plies or not recovered:
        raise TypeError(
            "section and response must provide plies and recovered section_points."
        )
    if isinstance(strengths, CompositeStrengths2D):
        by_name = {ply.name: strengths for ply in plies}
    else:
        by_name = dict(strengths)
        expected = {ply.name for ply in plies}
        missing = expected.difference(by_name)
        unknown = set(by_name).difference(expected)
        if missing or unknown:
            raise ValueError(
                "Laminate strength mapping must match ply names exactly; "
                f"missing={sorted(missing)!r}, unknown={sorted(unknown)!r}."
            )
    points = []
    for result in recovered:
        point = result.point
        if point.ply_index < 0 or point.ply_index >= len(plies):
            raise ValueError(f"Invalid recovered ply index at {point.id!r}.")
        ply = plies[point.ply_index]
        if point.ply_name != ply.name:
            raise ValueError(f"Recovered ply identity mismatch at {point.id!r}.")
        material_stress = _stress_to_material_axes(result.stress, ply.angle_degrees)
        assessment = assess_ply_failure(
            material_stress,
            by_name[ply.name],
            criterion=criterion,
        )
        points.append(
            PlyPointFailureAssessment(
                point_id=point.id,
                ply_name=point.ply_name,
                ply_index=point.ply_index,
                point_index=point.point_index,
                z=float(point.z),
                material_stress=material_stress,
                assessment=assessment,
            )
        )
    governing = max(points, key=lambda item: item.assessment.maximum_index)
    return LaminateFailureAssessment(
        points=tuple(points),
        governing_point_id=governing.point_id,
        governing_mode=governing.assessment.governing_mode,
        maximum_index=governing.assessment.maximum_index,
        load_factor_to_first_failure=min(
            point.assessment.load_factor_to_first_failure for point in points
        ),
        accepted=all(point.assessment.accepted for point in points),
    )


__all__ = [
    "CompositeStrengths2D",
    "Hashin2D",
    "LaminateFailureAssessment",
    "MaximumStress2D",
    "PlyFailureAssessment",
    "PlyFailureCriterion",
    "PlyPointFailureAssessment",
    "TsaiWu2D",
    "assess_laminate_failure",
    "assess_ply_failure",
    "composite_strengths_2d",
]

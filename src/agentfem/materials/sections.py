# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Composite ply and laminate-section mechanics.

Sections own through-thickness placement and recovery.  Constitutive material
objects own stress--strain response, and a Study owns the physical analysis;
keeping these concepts separate allows the same ply material to be reused in
solids, membranes, shells, and external providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Mapping, Sequence

import numpy as np


def _vector3(value, *, name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != (3,) or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be one finite 3-component engineering vector.")
    return selected


def transformed_reduced_stiffness(stiffness_voigt, angle_degrees: float) -> np.ndarray:
    """Rotate one planar engineering-Voigt stiffness into section axes.

    A tensor transformation is used rather than a special orthotropic formula,
    so a fully populated symmetric 3x3 reduced stiffness is supported.
    """

    stiffness = np.asarray(stiffness_voigt, dtype=float)
    if stiffness.shape != (3, 3):
        raise ValueError("A laminate ply requires a 3x3 reduced stiffness matrix.")
    angle = float(angle_degrees) * pi / 180.0
    rotation = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    result = np.zeros((3, 3), dtype=float)
    strain_basis = (
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 0.5], [0.5, 0.0]]),
    )
    for column, global_strain in enumerate(strain_basis):
        local_strain = rotation.T @ global_strain @ rotation
        local_voigt = np.array(
            [local_strain[0, 0], local_strain[1, 1], 2.0 * local_strain[0, 1]]
        )
        local_stress_voigt = stiffness @ local_voigt
        local_stress = np.array(
            [
                [local_stress_voigt[0], local_stress_voigt[2]],
                [local_stress_voigt[2], local_stress_voigt[1]],
            ]
        )
        global_stress = rotation @ local_stress @ rotation.T
        result[:, column] = (
            global_stress[0, 0],
            global_stress[1, 1],
            global_stress[0, 1],
        )
    return 0.5 * (result + result.T)


@dataclass(frozen=True)
class Ply:
    """One named lamina with a material, thickness, and section orientation."""

    name: str
    material: object
    thickness: float
    angle_degrees: float = 0.0
    integration_points: int = 3

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Ply.name must be non-empty.")
        if not np.isfinite(self.thickness) or self.thickness <= 0.0:
            raise ValueError("Ply.thickness must be positive and finite.")
        if not np.isfinite(self.angle_degrees):
            raise ValueError("Ply.angle_degrees must be finite.")
        if int(self.integration_points) < 1:
            raise ValueError("Ply.integration_points must be at least one.")
        stiffness = np.asarray(getattr(self.material, "stiffness_voigt", ()))
        if stiffness.shape != (3, 3):
            raise TypeError(
                "Ply.material must provide a 3x3 plane-stress stiffness_voigt."
            )
        object.__setattr__(self, "integration_points", int(self.integration_points))

    @property
    def reduced_stiffness(self) -> np.ndarray:
        return transformed_reduced_stiffness(
            self.material.stiffness_voigt, self.angle_degrees
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "composite_ply",
            "name": self.name,
            "material": getattr(self.material, "name", type(self.material).__name__),
            "thickness": float(self.thickness),
            "angle_degrees": float(self.angle_degrees),
            "integration_points": self.integration_points,
        }


@dataclass(frozen=True)
class SectionPoint:
    """Stable through-thickness integration-point identity."""

    ply_name: str
    ply_index: int
    point_index: int
    z: float
    weight: float

    @property
    def id(self) -> str:
        return f"{self.ply_name}:section-point-{self.point_index + 1}"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "ply_name": self.ply_name,
            "ply_index": self.ply_index,
            "point_index": self.point_index,
            "z": self.z,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class PlyPointResult:
    """Strain and stress at one stable laminate section point."""

    point: SectionPoint
    strain: np.ndarray
    stress: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            **self.point.as_dict(),
            "strain": self.strain.tolist(),
            "stress": self.stress.tolist(),
            "voigt_order": ["11", "22", "12"],
            "strain_shear": "engineering",
        }


@dataclass(frozen=True)
class LaminateResponse:
    """Classical-laminate generalized forces plus recoverable ply fields."""

    membrane_force: np.ndarray
    bending_moment: np.ndarray
    section_points: tuple[PlyPointResult, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "laminate_response",
            "membrane_force": self.membrane_force.tolist(),
            "bending_moment": self.bending_moment.tolist(),
            "generalized_order": ["11", "22", "12"],
            "section_points": [item.as_dict() for item in self.section_points],
        }


@dataclass(frozen=True)
class LaminateSection:
    """Ordered composite plies evaluated by classical laminate theory."""

    name: str
    plies: tuple[Ply, ...]
    reference_surface_offset: float = 0.0
    source: str = "user_defined"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        plies = tuple(self.plies)
        if not str(self.name).strip():
            raise ValueError("LaminateSection.name must be non-empty.")
        if not plies:
            raise ValueError("LaminateSection requires at least one ply.")
        names = [ply.name for ply in plies]
        if len(set(names)) != len(names):
            raise ValueError("Ply names must be unique within a laminate section.")
        if not np.isfinite(self.reference_surface_offset):
            raise ValueError("reference_surface_offset must be finite.")
        if not str(self.source).strip():
            raise ValueError("LaminateSection.source must be non-empty.")
        object.__setattr__(self, "plies", plies)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def thickness(self) -> float:
        return float(sum(ply.thickness for ply in self.plies))

    @property
    def interfaces(self) -> np.ndarray:
        bottom = -0.5 * self.thickness + float(self.reference_surface_offset)
        return bottom + np.concatenate(([0.0], np.cumsum([ply.thickness for ply in self.plies])))

    def abd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return membrane, membrane--bending, and bending matrices ``A,B,D``."""

        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        z = self.interfaces
        for index, ply in enumerate(self.plies):
            qbar = ply.reduced_stiffness
            lower, upper = float(z[index]), float(z[index + 1])
            A += qbar * (upper - lower)
            B += 0.5 * qbar * (upper**2 - lower**2)
            D += qbar * (upper**3 - lower**3) / 3.0
        return A, B, D

    def section_points(self) -> tuple[SectionPoint, ...]:
        """Return Gauss points with stable ply-local identities."""

        points: list[SectionPoint] = []
        interfaces = self.interfaces
        for ply_index, ply in enumerate(self.plies):
            abscissae, weights = np.polynomial.legendre.leggauss(ply.integration_points)
            lower = float(interfaces[ply_index])
            upper = float(interfaces[ply_index + 1])
            for point_index, (xi, weight) in enumerate(zip(abscissae, weights, strict=True)):
                points.append(
                    SectionPoint(
                        ply_name=ply.name,
                        ply_index=ply_index,
                        point_index=point_index,
                        z=0.5 * ((upper - lower) * float(xi) + upper + lower),
                        weight=0.5 * (upper - lower) * float(weight),
                    )
                )
        return tuple(points)

    def evaluate(self, membrane_strain, curvature=(0.0, 0.0, 0.0)) -> LaminateResponse:
        """Evaluate generalized resultants and all declared section points."""

        membrane = _vector3(membrane_strain, name="membrane_strain")
        bending = _vector3(curvature, name="curvature")
        A, B, D = self.abd()
        membrane_force = A @ membrane + B @ bending
        bending_moment = B @ membrane + D @ bending
        recovered = []
        for point in self.section_points():
            strain = membrane + point.z * bending
            stress = self.plies[point.ply_index].reduced_stiffness @ strain
            recovered.append(PlyPointResult(point=point, strain=strain, stress=stress))
        return LaminateResponse(
            membrane_force=membrane_force,
            bending_moment=bending_moment,
            section_points=tuple(recovered),
        )

    def as_dict(self) -> dict[str, object]:
        A, B, D = self.abd()
        return {
            "kind": "laminate_section",
            "name": self.name,
            "thickness": self.thickness,
            "reference_surface_offset": float(self.reference_surface_offset),
            "source": self.source,
            "metadata": dict(self.metadata),
            "plies": [ply.as_dict() for ply in self.plies],
            "A": A.tolist(),
            "B": B.tolist(),
            "D": D.tolist(),
            "section_points": [point.as_dict() for point in self.section_points()],
            "theory": "classical_laminate_theory",
        }


def ply(
    material,
    thickness: float,
    *,
    angle: float = 0.0,
    name: str = "ply",
    integration_points: int = 3,
) -> Ply:
    """Create one ply; ``angle`` follows the industry-standard degree convention."""

    return Ply(
        name=name,
        material=material,
        thickness=thickness,
        angle_degrees=angle,
        integration_points=integration_points,
    )


def laminate(
    plies: Sequence[Ply],
    *,
    name: str = "laminate",
    reference_surface_offset: float = 0.0,
    source: str = "user_defined",
    metadata: Mapping[str, object] | None = None,
) -> LaminateSection:
    """Create an ordered laminate section without coupling it to an element."""

    return LaminateSection(
        name=name,
        plies=tuple(plies),
        reference_surface_offset=reference_surface_offset,
        source=source,
        metadata={} if metadata is None else metadata,
    )


def laminate_from_abaqus_section(
    section,
    materials_by_name: Mapping[str, object],
    *,
    reviewed_by: str,
    name: str | None = None,
) -> LaminateSection:
    """Lower one reviewed Abaqus composite-section inventory.

    This consumes the source-preserving migration object, not a raw input file.
    Two documented row layouts are supported: composite solid/continuum rows
    ``thickness, material, angle`` and composite shell rows
    ``thickness, integration points, material, angle, ply name``. Any other
    layout fails closed for semantic review.
    """

    reviewer = str(reviewed_by).strip()
    if not reviewer:
        raise ValueError("Abaqus composite-section lowering requires reviewed_by.")
    flags = {str(item).strip().upper() for item in getattr(section, "flags", ())}
    if "COMPOSITE" not in flags:
        raise ValueError("The selected Abaqus section is not declared COMPOSITE.")
    if getattr(section, "status", None) != "review_required_composite":
        raise ValueError(
            "Abaqus composite section references must resolve before reviewed lowering."
        )
    section_type = str(getattr(section, "section_type", "")).upper()
    if "SHELL" not in section_type and "SOLID" not in section_type:
        raise NotImplementedError(
            f"Unsupported Abaqus composite section type {section_type!r}."
        )
    lookup = {str(key).upper(): value for key, value in materials_by_name.items()}
    lowered: list[Ply] = []
    for index, row in enumerate(getattr(section, "rows", ())):
        values = tuple(str(item).strip() for item in row)
        if "SHELL" in section_type:
            if len(values) < 3:
                raise ValueError(
                    f"Abaqus shell ply row {index + 1} needs thickness, integration points, and material."
                )
            try:
                integration_points = int(values[1])
            except ValueError as exc:
                raise ValueError(
                    f"Abaqus shell ply row {index + 1} has an invalid integration-point count."
                ) from exc
            material_name = values[2]
            angle = 0.0 if len(values) < 4 or not values[3] else float(values[3])
            ply_name = values[4] if len(values) >= 5 and values[4] else f"ply_{index + 1}"
        else:
            if len(values) != 3:
                raise ValueError(
                    f"Abaqus solid composite row {index + 1} must be thickness, material, angle."
                )
            integration_points = 3
            material_name = values[1]
            angle = float(values[2])
            ply_name = f"ply_{index + 1}"
        try:
            material = lookup[material_name.upper()]
        except KeyError as exc:
            raise KeyError(
                f"Abaqus ply {ply_name!r} references unknown material {material_name!r}."
            ) from exc
        lowered.append(
            ply(
                material,
                float(values[0]),
                angle=angle,
                name=ply_name,
                integration_points=integration_points,
            )
        )
    location = getattr(section, "location", None)
    source_record = location.summary() if hasattr(location, "summary") else None
    return laminate(
        lowered,
        name=name or f"abaqus_{getattr(section, 'region', 'composite')}_layup",
        source="reviewed_abaqus_composite_section",
        metadata={
            "reviewed_by": reviewer,
            "source_location": source_record,
            "source_section_type": section_type,
            "source_region": getattr(section, "region", None),
        },
    )


__all__ = [
    "LaminateResponse",
    "LaminateSection",
    "Ply",
    "PlyPointResult",
    "SectionPoint",
    "laminate",
    "laminate_from_abaqus_section",
    "ply",
    "transformed_reduced_stiffness",
]

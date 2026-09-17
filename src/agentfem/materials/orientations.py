# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Material directions independent of constitutive parameters and Studies.

The objects in this module describe how a material basis is embedded in the
reference configuration.  They are assignment assets: the same orthotropic
material may therefore be reused with many orientations without copying or
rewriting its constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


def _unit(vector, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.ndim != 1 or value.size not in {2, 3}:
        raise ValueError(f"{name} must be one 2D or 3D vector.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite values.")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-14:
        raise ValueError(f"{name} must be nonzero.")
    return value / norm


@dataclass(frozen=True)
class MaterialFrame:
    """Right-handed orthonormal material frame in the reference configuration."""

    name: str
    basis: np.ndarray
    evolution: str = "fixed"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        basis = np.asarray(self.basis, dtype=float)
        evolution = str(self.evolution).strip().lower().replace("-", "_")
        if not name:
            raise ValueError("MaterialFrame.name must be non-empty.")
        if basis.shape not in {(2, 2), (3, 3)}:
            raise ValueError("MaterialFrame.basis must be a 2x2 or 3x3 matrix.")
        if not np.all(np.isfinite(basis)):
            raise ValueError("MaterialFrame.basis must contain finite values.")
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[0]), atol=1.0e-10):
            raise ValueError("MaterialFrame.basis must be orthonormal.")
        if np.linalg.det(basis) <= 0.0:
            raise ValueError("MaterialFrame.basis must be right-handed.")
        if evolution not in {"fixed", "convected"}:
            raise ValueError("MaterialFrame.evolution must be 'fixed' or 'convected'.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "basis", basis.copy())
        object.__setattr__(self, "evolution", evolution)

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[0])

    @classmethod
    def from_angle(
        cls,
        angle: float,
        *,
        unit: str = "degree",
        name: str = "material_frame",
        evolution: str = "fixed",
    ) -> "MaterialFrame":
        """Create a planar frame rotated counter-clockwise from global x."""

        selected = str(unit).strip().lower()
        radians = float(angle) * pi / 180.0 if selected in {"degree", "degrees", "deg"} else float(angle)
        if selected not in {"degree", "degrees", "deg", "radian", "radians", "rad"}:
            raise ValueError("MaterialFrame angle unit must be degree or radian.")
        c = cos(radians)
        s = sin(radians)
        return cls(name=name, basis=np.array([[c, -s], [s, c]]), evolution=evolution)

    def current_basis(self, deformation_gradient=None) -> np.ndarray:
        """Return the fixed basis or its polar-rotation update.

        A material frame remains orthonormal.  Independent directions that
        shear into a non-orthogonal pair belong to :class:`FiberFrame`.
        """

        if self.evolution == "fixed" or deformation_gradient is None:
            return self.basis.copy()
        F = np.asarray(deformation_gradient, dtype=float)
        if F.shape != self.basis.shape:
            raise ValueError("deformation_gradient shape must match the material frame.")
        mapped = F @ self.basis
        left, _, right_transpose = np.linalg.svd(mapped)
        rotation = left @ right_transpose
        if np.linalg.det(rotation) <= 0.0:
            raise ValueError("deformation_gradient must preserve orientation.")
        return rotation

    def to_local(self, tensor) -> np.ndarray:
        value = np.asarray(tensor, dtype=float)
        return self.basis.T @ value @ self.basis

    def to_global(self, tensor) -> np.ndarray:
        value = np.asarray(tensor, dtype=float)
        return self.basis @ value @ self.basis.T

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "material_frame",
            "name": self.name,
            "dimension": self.dimension,
            "basis": self.basis.tolist(),
            "evolution": self.evolution,
            "orthonormal": True,
            "right_handed": True,
        }

    def summary(self) -> str:
        return f"{self.name}: {self.dimension}D {self.evolution} material frame"


@dataclass(frozen=True)
class FiberFrame:
    """Two independent structural directions for a woven reinforcement.

    Unlike :class:`MaterialFrame`, warp and weft need not remain orthogonal.
    This distinction is essential for trellising and in-plane shear.
    """

    name: str
    warp: np.ndarray
    weft: np.ndarray

    def __post_init__(self) -> None:
        warp = _unit(self.warp, name="FiberFrame.warp")
        weft = _unit(self.weft, name="FiberFrame.weft")
        if warp.shape != weft.shape:
            raise ValueError("FiberFrame warp and weft dimensions must match.")
        if abs(float(np.dot(warp, weft))) >= 1.0 - 1.0e-10:
            raise ValueError("FiberFrame warp and weft directions must be independent.")
        name = str(self.name).strip()
        if not name:
            raise ValueError("FiberFrame.name must be non-empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "warp", warp)
        object.__setattr__(self, "weft", weft)

    @property
    def dimension(self) -> int:
        return int(self.warp.size)

    @property
    def reference_angle(self) -> float:
        return float(np.arccos(np.clip(np.dot(self.warp, self.weft), -1.0, 1.0)))

    def convect(self, deformation_gradient) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Return current unit directions and their stretches."""

        F = np.asarray(deformation_gradient, dtype=float)
        if F.shape != (self.dimension, self.dimension):
            raise ValueError("deformation_gradient shape must match the fiber frame.")
        current_warp = F @ self.warp
        current_weft = F @ self.weft
        stretch_warp = float(np.linalg.norm(current_warp))
        stretch_weft = float(np.linalg.norm(current_weft))
        if min(stretch_warp, stretch_weft) <= 1.0e-14:
            raise ValueError("deformation_gradient collapsed a fiber direction.")
        return (
            current_warp / stretch_warp,
            current_weft / stretch_weft,
            stretch_warp,
            stretch_weft,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "fiber_frame",
            "name": self.name,
            "dimension": self.dimension,
            "warp": self.warp.tolist(),
            "weft": self.weft.tolist(),
            "reference_angle_radians": self.reference_angle,
            "orthogonal": bool(abs(np.dot(self.warp, self.weft)) <= 1.0e-10),
        }


@dataclass(frozen=True)
class OrientedMaterial:
    """One constitutive behavior combined with a separate material frame."""

    material: object
    orientation: MaterialFrame
    name: str | None = None

    def __post_init__(self) -> None:
        if not hasattr(self.material, "stiffness_voigt"):
            raise TypeError("OrientedMaterial currently requires an elastic stiffness_voigt.")
        size = int(np.asarray(self.material.stiffness_voigt).shape[0])
        expected = 3 if self.orientation.dimension == 2 else 6
        if size != expected:
            raise ValueError(
                f"Material stiffness has {size} Voigt components but the frame requires {expected}."
            )
        if self.name is None:
            object.__setattr__(self, "name", f"{getattr(self.material, 'name', 'material')} @ {self.orientation.name}")

    @property
    def stiffness_voigt(self):
        return self.material.stiffness_voigt

    @property
    def density(self) -> float:
        return float(self.material.density)

    @property
    def pressure_wave_speed(self) -> float:
        return float(self.material.pressure_wave_speed)

    @property
    def shear_wave_speed(self) -> float:
        return float(self.material.shear_wave_speed)

    @property
    def model(self) -> str:
        return f"oriented_{getattr(self.material, 'model', type(self.material).__name__)}"

    def as_dict(self) -> dict[str, object]:
        material = self.material.as_dict() if hasattr(self.material, "as_dict") else repr(self.material)
        return {
            "kind": "oriented_material",
            "name": self.name,
            "material": material,
            "orientation": self.orientation.as_dict(),
        }

    def summary(self) -> str:
        return f"{self.name}: {self.orientation.dimension}D oriented elastic material"


def material_frame(
    primary,
    secondary=None,
    *,
    normal=None,
    name: str = "material_frame",
    evolution: str = "fixed",
) -> MaterialFrame:
    """Build a checked right-handed orthonormal material frame from axes."""

    first = _unit(primary, name="primary")
    if first.size == 2:
        if normal is not None:
            raise ValueError("A 2D material frame does not accept normal=.")
        second = np.array([-first[1], first[0]])
        if secondary is not None:
            candidate = _unit(secondary, name="secondary")
            projected = candidate - float(np.dot(candidate, first)) * first
            second = _unit(projected, name="orthogonalized secondary")
            if np.linalg.det(np.column_stack((first, second))) <= 0.0:
                raise ValueError("primary and secondary define a left-handed 2D frame.")
        basis = np.column_stack((first, second))
    else:
        if secondary is None and normal is None:
            raise ValueError("A 3D material frame requires secondary= or normal=.")
        if secondary is None:
            candidate = _unit(normal, name="normal")
            third = _unit(
                candidate - float(np.dot(candidate, first)) * first,
                name="orthogonalized normal",
            )
            second = _unit(np.cross(third, first), name="inferred secondary")
        else:
            candidate = _unit(secondary, name="secondary")
            second = _unit(
                candidate - float(np.dot(candidate, first)) * first,
                name="orthogonalized secondary",
            )
        third = _unit(np.cross(first, second), name="inferred normal")
        if normal is not None and float(np.dot(third, _unit(normal, name="normal"))) < 1.0 - 1.0e-8:
            raise ValueError("normal is inconsistent with primary and secondary.")
        basis = np.column_stack((first, second, third))
    return MaterialFrame(name=name, basis=basis, evolution=evolution)


def fiber_frame(warp, weft, *, name: str = "fiber_frame") -> FiberFrame:
    """Build two independent reference yarn directions."""

    return FiberFrame(name=name, warp=warp, weft=weft)


def oriented(material, orientation: MaterialFrame, *, name: str | None = None) -> OrientedMaterial:
    """Assign an elastic material to a reusable material frame."""

    return OrientedMaterial(material=material, orientation=orientation, name=name)


__all__ = [
    "FiberFrame",
    "MaterialFrame",
    "OrientedMaterial",
    "fiber_frame",
    "material_frame",
    "oriented",
]

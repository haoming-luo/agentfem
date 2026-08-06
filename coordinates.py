"""Engineering coordinate systems and reference points.

The public objects in this module keep local component conventions explicit.
They deliberately do not own mesh topology or solver degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl


def _vector(value, dimension: int, *, label: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float).reshape(-1)
    if selected.size != dimension or not np.all(np.isfinite(selected)):
        raise ValueError(f"{label} must have {dimension} finite components.")
    return selected


@dataclass(frozen=True)
class CartesianSystem:
    """Right-handed orthonormal Cartesian coordinate system.

    ``axes`` stores local basis vectors by row in global components.  Thus a
    local vector ``v_l`` is mapped to global components by ``axes.T @ v_l``.
    """

    axes: object
    origin: object | None = None
    name: str = "local"
    tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        axes = np.asarray(self.axes, dtype=float)
        if axes.ndim != 2 or axes.shape[0] != axes.shape[1]:
            raise ValueError("CartesianSystem axes must be a square matrix.")
        dimension = int(axes.shape[0])
        if dimension not in {2, 3} or not np.all(np.isfinite(axes)):
            raise ValueError("CartesianSystem supports finite 2D or 3D axes.")
        tolerance = float(self.tolerance)
        if tolerance <= 0.0 or not np.isfinite(tolerance):
            raise ValueError("CartesianSystem tolerance must be positive.")
        if not np.allclose(axes @ axes.T, np.eye(dimension), atol=tolerance, rtol=0.0):
            raise ValueError("CartesianSystem axes must be orthonormal.")
        determinant = float(np.linalg.det(axes))
        if not np.isclose(determinant, 1.0, atol=tolerance, rtol=0.0):
            raise ValueError("CartesianSystem axes must form a right-handed basis.")
        origin = np.zeros(dimension) if self.origin is None else _vector(
            self.origin, dimension, label="origin"
        )
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def dimension(self) -> int:
        return int(self.axes.shape[0])

    def vector_to_global(self, value):
        """Map vector components from this local system to global components."""

        if isinstance(value, (tuple, list, np.ndarray)):
            return self.axes.T @ _vector(value, self.dimension, label="vector")
        return ufl.dot(ufl.as_matrix(self.axes.T), value)

    def vector_to_local(self, value):
        """Map global vector components into this local system."""

        if isinstance(value, (tuple, list, np.ndarray)):
            return self.axes @ _vector(value, self.dimension, label="vector")
        return ufl.dot(ufl.as_matrix(self.axes), value)

    def tensor_to_global(self, value):
        """Map a second-order tensor from local to global components."""

        if isinstance(value, np.ndarray):
            selected = np.asarray(value, dtype=float)
            if selected.shape != (self.dimension, self.dimension):
                raise ValueError("tensor has incompatible shape.")
            return self.axes.T @ selected @ self.axes
        rotation = ufl.as_matrix(self.axes)
        return ufl.dot(ufl.dot(rotation.T, value), rotation)

    def tensor_to_local(self, value):
        """Map a second-order tensor from global to local components."""

        if isinstance(value, np.ndarray):
            selected = np.asarray(value, dtype=float)
            if selected.shape != (self.dimension, self.dimension):
                raise ValueError("tensor has incompatible shape.")
            return self.axes @ selected @ self.axes.T
        rotation = ufl.as_matrix(self.axes)
        return ufl.dot(ufl.dot(rotation, value), rotation.T)

    def point_to_global(self, value) -> np.ndarray:
        return self.origin + self.vector_to_global(value)

    def point_to_local(self, value) -> np.ndarray:
        return self.vector_to_local(
            _vector(value, self.dimension, label="point") - self.origin
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "cartesian_coordinate_system",
            "dimension": self.dimension,
            "origin": self.origin.tolist(),
            "axes": self.axes.tolist(),
            "convention": "axes are local basis vectors in global components",
        }


@dataclass(frozen=True)
class ReferencePoint:
    """Named engineering point used for remote resultants and kinematics."""

    coordinates: object
    name: str = "reference_point"
    system: CartesianSystem | None = None

    def __post_init__(self) -> None:
        selected = np.asarray(self.coordinates, dtype=float).reshape(-1)
        if selected.size not in {2, 3} or not np.all(np.isfinite(selected)):
            raise ValueError("ReferencePoint coordinates must be a finite 2D or 3D vector.")
        if self.system is not None:
            if selected.size != self.system.dimension:
                raise ValueError("ReferencePoint and coordinate-system dimensions differ.")
            selected = self.system.point_to_global(selected)
        object.__setattr__(self, "coordinates", selected)
        object.__setattr__(self, "name", str(self.name))

    @property
    def dimension(self) -> int:
        return int(self.coordinates.size)

    def __array__(self, dtype=None):
        return np.asarray(self.coordinates, dtype=dtype)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "reference_point",
            "coordinates": self.coordinates.tolist(),
            "coordinate_system": None if self.system is None else self.system.name,
        }


def cartesian(*, origin=None, axes=None, x=None, y=None, z=None, name="local") -> CartesianSystem:
    """Create a Cartesian system from a matrix or named basis vectors."""

    if axes is not None and any(value is not None for value in (x, y, z)):
        raise ValueError("Specify axes= or x=/y=/z=, not both.")
    if axes is None:
        selected = tuple(value for value in (x, y, z) if value is not None)
        if len(selected) not in {2, 3}:
            raise ValueError("Provide two or three local basis vectors.")
        axes = np.vstack(selected)
    return CartesianSystem(axes=axes, origin=origin, name=name)


def reference_point(coordinates, *, name="reference_point", system=None) -> ReferencePoint:
    """Create a named engineering reference point."""

    return ReferencePoint(coordinates, name=name, system=system)


__all__ = [
    "CartesianSystem",
    "ReferencePoint",
    "cartesian",
    "reference_point",
]

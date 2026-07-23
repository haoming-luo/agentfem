"""Compatibility wrappers for mesh regions.

New application code should use ``agentfem.mesh.boundary(...)``. This module is
kept so older examples and notebooks can migrate gradually.
"""

from __future__ import annotations

from .mesh import (
    BoundaryRegion,
    boundary,
    boundary_region,
    region_marker as boundary_marker,
    region_measure as boundary_measure,
)

__all__ = [
    "BoundaryRegion",
    "boundary",
    "boundary_region",
    "boundary_marker",
    "boundary_measure",
]

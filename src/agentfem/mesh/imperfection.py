# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit, reversible mode-shaped perturbations of reference geometry."""

from dataclasses import dataclass
import numpy as np
from mpi4py import MPI
from . import quality


@dataclass
class GeometryImperfection:
    """Own the original coordinates; restore before reusing original forms/tags."""

    domain: object
    original_coordinates: np.ndarray
    applied_coordinates: np.ndarray
    amplitudes: tuple
    quality_report: object

    def restore(self):
        """Restore exactly; reject unrelated geometry edits after application."""
        changed = not np.array_equal(self.domain.geometry.x, self.applied_coordinates)
        if self.domain.comm.allreduce(changed, op=MPI.LOR):
            raise ValueError(
                "AFM-IMPERFECTION-005: geometry changed after imperfection."
            )
        self.domain.geometry.x[:] = self.original_coordinates


def apply_mode_imperfection(domain, modes, *, amplitudes, minimum_quality=1.0e-3):
    """Perturb geometry in place, retaining original coordinates in the receipt.

    Each mode is normalized by its global maximum vector norm at geometry nodes;
    each signed amplitude has the mesh's length unit. Combined modes need not
    attain the sum of amplitudes. Create the nonlinear model AFTER this operation. Existing topological facet
    tags remain valid; preserve those for boundaries that move with the mesh. Modes must live on this mesh. Initial scope is
    linear triangle/quad/tetra/hex geometry; FE mode degree may be higher.
    Failed quality checks restore coordinates, and no connectivity is changed.
    """
    modes, amplitudes = tuple(modes), tuple(float(a) for a in amplitudes)
    if (
        not modes
        or len(modes) != len(amplitudes)
        or not np.all(np.isfinite(amplitudes))
    ):
        raise ValueError(
            "AFM-IMPERFECTION-001: matching nonempty modes and finite amplitudes required."
        )
    if int(domain.geometry.cmaps[0].degree) != 1:
        raise NotImplementedError(
            "AFM-IMPERFECTION-002: only linear coordinate elements are supported."
        )
    if domain.topology.cell_type.name not in (
        "triangle",
        "quadrilateral",
        "tetrahedron",
        "hexahedron",
    ):
        raise NotImplementedError("AFM-IMPERFECTION-002: unsupported cell topology.")
    if domain.geometry.dim != domain.topology.dim:
        raise NotImplementedError(
            "Embedded meshes are not supported by this geometry transaction."
        )
    original = domain.geometry.x.copy()
    dim = domain.geometry.dim
    cells = np.full(len(original), -1, dtype=np.int32)
    for cell, nodes in enumerate(domain.geometry.dofmaps[0]):
        cells[nodes] = cell
    if domain.comm.allreduce(bool(np.any(cells < 0)), op=MPI.LOR):
        raise ValueError("AFM-IMPERFECTION-003: unreferenced geometry nodes.")
    delta = np.zeros_like(original)
    for mode, amplitude in zip(modes, amplitudes):
        field = getattr(mode, "value", mode)
        if field.function_space.mesh is not domain or field.ufl_shape != (dim,):
            raise ValueError(
                "AFM-IMPERFECTION-003: modes must be vector fields on this mesh."
            )
        field.x.scatter_forward()
        values = np.asarray(field.eval(original, cells)).reshape(-1, dim)
        if domain.comm.allreduce(not bool(np.all(np.isfinite(values))), op=MPI.LOR):
            raise ValueError("AFM-IMPERFECTION-003: nonfinite mode values.")
        norm = domain.comm.allreduce(
            float(np.max(np.linalg.norm(values, axis=1), initial=0)), op=MPI.MAX
        )
        if norm <= 0:
            raise ValueError(
                "AFM-IMPERFECTION-004: zero mode cannot define an imperfection."
            )
        delta[:, :dim] += amplitude * values / norm
    try:
        domain.geometry.x[:] = original + delta
        report = quality.audit(domain, threshold=minimum_quality, strict=True)
        # Quality is orientation-independent; also compare orientation to the
        # original map. The quality gate already rejects sign-changing maps.
        tdim = domain.topology.dim
        edges = (
            [1, 2, 4]
            if domain.topology.cell_type.name == "hexahedron"
            else list(range(1, tdim + 1))
        )
        bad = False
        for nodes in domain.geometry.dofmaps[0]:
            old, new = original[nodes, :dim], domain.geometry.x[nodes, :dim]
            bad |= (
                np.linalg.det((old[edges] - old[0]).T)
                * np.linalg.det((new[edges] - new[0]).T)
                <= 0
            )
        if domain.comm.allreduce(bool(bad), op=MPI.LOR):
            raise ValueError("AFM-IMPERFECTION-006: imperfection inverted an element.")
    except Exception:
        domain.geometry.x[:] = original
        raise
    return GeometryImperfection(
        domain, original, domain.geometry.x.copy(), amplitudes, report
    )

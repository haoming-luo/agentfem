# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Identity-preserving selection of owned nodal degrees of freedom."""

from dataclasses import dataclass
import numpy as np
from mpi4py import MPI
from . import fields


@dataclass(frozen=True)
class NodalSelection:
    """A small set of physical nodes, not a nearest-node approximation.

    Source labels require a source table whose order is the mesh input order.
    Spatial selection must match exactly one global node within tolerance.
    Coincident seam nodes must instead be selected by input identity.
    """

    labels: tuple[int, ...] = ()
    source: object = None
    points: tuple[tuple[float, ...], ...] = ()
    tolerance: float = 1e-10

    def __post_init__(self):
        if bool(self.labels) == bool(self.points):
            raise ValueError("AFM-NODE-SELECTION: choose source labels or points.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("AFM-NODE-SELECTION: tolerance must be positive.")
        if self.labels and (
            self.source is None or len(set(self.labels)) != len(self.labels)
        ):
            raise ValueError(
                "AFM-NODE-SELECTION: unique labels and source table required."
            )
        if self.points and (
            len(set(self.points)) != len(self.points)
            or any(
                len(p) not in (2, 3) or not np.all(np.isfinite(p)) for p in self.points
            )
        ):
            raise ValueError(
                "AFM-NODE-SELECTION: finite distinct coordinates required."
            )

    def summary(self):
        return {
            "kind": "nodal_selection",
            "labels": list(self.labels),
            "coordinates": [self.source.coordinate(i).tolist() for i in self.labels]
            if self.labels
            else [list(p) for p in self.points],
            "tolerance": self.tolerance,
            "matching": "unique_node_not_nearest",
        }

    def bind(self, target):
        V = fields.unwrap(target).function_space
        coords = V.tabulate_dof_coordinates()
        n = V.dofmap.index_map.size_local
        bs = V.dofmap.index_map_bs
        if V.dofmap.bs != bs or bs != V.mesh.geometry.dim:
            raise ValueError("AFM-NODE-SPACE: blocked nodal displacement required.")
        from .provenance import collective_call, collective_canonical_record

        summary = collective_call(
            self.summary, comm=V.mesh.comm, label="Nodal selection"
        )
        collective_canonical_record(summary, comm=V.mesh.comm, label="Nodal selection")

        def prepare_identity():
            source_ids = None
            if self.labels:
                source_ids = np.full(n, -1, dtype=np.int64)
                g = V.mesh.geometry
                nc = V.mesh.topology.index_map(V.mesh.topology.dim)
                for cell in range(nc.size_local + nc.num_ghosts):
                    gd = g.dofmaps[0][cell]
                    for d in V.dofmap.cell_dofs(cell):
                        if d >= n:
                            continue
                        match = np.flatnonzero(
                            np.linalg.norm(g.x[gd] - coords[d], axis=1)
                            <= self.tolerance
                        )
                        if len(match) == 0:
                            # Enriched field nodes need not carry a geometry label.
                            continue
                        if len(match) != 1:
                            raise ValueError(
                                "AFM-NODE-IDENTITY: field node lacks unique input geometry identity; use matching geometry order."
                            )
                        node = int(g.input_global_indices[gd[match[0]]])
                        if source_ids[d] not in (-1, node):
                            raise ValueError(
                                "AFM-NODE-IDENTITY: inconsistent source identity."
                            )
                        source_ids[d] = node
            return source_ids

        source_ids = collective_call(
            prepare_identity, comm=V.mesh.comm, label="Nodal identity"
        )
        selected = []
        for j, p in enumerate(summary["coordinates"]):
            point = np.pad(p, (0, 3 - len(p)))
            if source_ids is None:
                hits = np.flatnonzero(
                    np.linalg.norm(coords[:n] - point, axis=1) <= self.tolerance
                )
            else:
                hits = np.flatnonzero(source_ids == self.source.index(self.labels[j]))
                mismatch = bool(
                    len(hits)
                    and np.any(
                        np.linalg.norm(coords[hits] - point, axis=1) > self.tolerance
                    )
                )
                if V.mesh.comm.allreduce(mismatch, op=MPI.LOR):
                    raise ValueError(
                        "AFM-NODE-IDENTITY: source order/coordinates disagree with mesh."
                    )
            count = V.mesh.comm.allreduce(len(hits), op=MPI.SUM)
            if count != 1:
                raise ValueError(
                    f"AFM-NODE-SELECTION: expected one global owner, found {count}; use labels for coincident nodes."
                )
            selected.append(int(hits[0]) if len(hits) else -1)
        return V, tuple(selected)


def nodal_selection(*, labels=(), source=None, at=None, tolerance=1e-10):
    points = ()
    if at is not None:
        a = np.asarray(at, dtype=float)
        if a.ndim == 1:
            a = a[None, :]
        if a.ndim != 2:
            raise ValueError(
                "AFM-NODE-SELECTION: coordinates must be one point or a point array."
            )
        points = tuple(tuple(float(x) for x in p) for p in a)
    raw = np.atleast_1d(labels)
    if any(not np.isfinite(float(x)) or int(x) != float(x) for x in raw):
        raise ValueError("AFM-NODE-SELECTION: source labels must be integers.")
    return NodalSelection(tuple(int(x) for x in raw), source, points, float(tolerance))

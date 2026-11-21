# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Zero-thickness split of complete matching Tet4/Tet10 material partitions.

Topology operation only: no contact, cohesive law or material is introduced.
Connectivity is in Abaqus node order; coordinates/source labels are retained.
For a finite crack, shared crack-front nodes must be declared explicitly;
otherwise opening an unselected neighboring interface is rejected.
"""

from dataclasses import dataclass
from collections import defaultdict
import numpy as np

_FACES = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))
_EDGES = {(0, 1): 4, (1, 2): 5, (0, 2): 6, (0, 3): 7, (1, 3): 8, (2, 3): 9}


@dataclass(frozen=True)
class TetSeam:
    coordinates: np.ndarray
    cells: np.ndarray
    source_labels: np.ndarray
    duplicate_of: dict
    faces: tuple

    def summary(self):
        return {
            "kind": "zero_thickness_tetrahedral_seam",
            "formulation": "topology_only",
            "behavior": "two_free_surfaces",
            "node_count": len(self.coordinates),
            "duplicated_nodes": self.duplicate_of,
            "faces": list(self.faces),
        }


def split_tetrahedral_seam(
    coordinates, cells, *, positive_cells, faces, source_labels=None, front_nodes=()
):
    """Split a complete matching interface, including all Tet10 midside nodes.

    ``faces`` contains triples of original *array indices*, not source labels.
    ``positive_cells`` explicitly declares which material partition is moved to
    the copied node identities; the coordinates themselves never move.
    """
    x = np.array(coordinates, dtype=float, copy=True)
    c = np.asarray(cells)
    if c.dtype.kind not in "iu" or c.ndim != 2 or c.shape[1] not in (4, 10):
        raise ValueError("AFM-SEAM: integer Tet4 or Tet10 connectivity required.")
    c = c.astype(np.int64, copy=True)
    if (
        x.ndim != 2
        or x.shape[1] != 3
        or not np.all(np.isfinite(x))
        or np.any(c < 0)
        or np.any(c >= len(x))
    ):
        raise ValueError("AFM-SEAM: invalid coordinates/connectivity.")
    if any(len(set(row)) != len(row) for row in c):
        raise ValueError("AFM-SEAM: repeated node in cell.")
    labels = (
        np.arange(1, len(x) + 1) if source_labels is None else np.asarray(source_labels)
    )
    if (
        labels.shape != (len(x),)
        or labels.dtype.kind not in "iu"
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("AFM-SEAM: unique integer source labels required.")
    positive = set(positive_cells)
    if not positive or not positive <= set(range(len(c))):
        raise ValueError("AFM-SEAM: invalid positive cells.")
    requested = [tuple(sorted(f)) for f in faces]
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(len(set(f)) != 3 for f in requested)
    ):
        raise ValueError("AFM-SEAM: distinct triangular faces required.")
    adjacency = defaultdict(list)
    for i, row in enumerate(c):
        for face in _FACES:
            nodes = [int(row[j]) for j in face]
            if c.shape[1] == 10:
                nodes += [
                    int(row[_EDGES[tuple(sorted((face[j], face[(j + 1) % 3])))]])
                    for j in range(3)
                ]
            adjacency[tuple(sorted(nodes[:3]))].append((i, nodes))
    if any(len(v) > 2 for v in adjacency.values()):
        raise ValueError("AFM-SEAM: nonmanifold face.")
    split_nodes = set()
    records = []
    for face in requested:
        sides = adjacency.get(face, [])
        if len(sides) != 2 or sum(i in positive for i, _ in sides) != 1:
            raise ValueError(
                "AFM-SEAM: face must separate two cells and the declared partition; already split/exterior faces rejected."
            )
        if set(sides[0][1]) != set(sides[1][1]):
            raise ValueError("AFM-SEAM: nonmatching midside identities.")
        plus, minus = sorted(sides, key=lambda row: row[0] not in positive)
        p = x[list(face)]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("AFM-SEAM: degenerate face.")
        normal /= norm
        direction = x[c[plus[0], :4]].mean(0) - x[c[minus[0], :4]].mean(0)
        if normal @ direction < 0:
            normal = -normal
        split_nodes.update(plus[1])
        records.append(
            {
                "corners": list(face),
                "positive_cell": plus[0],
                "negative_cell": minus[0],
                "normal_negative_to_positive": normal.tolist(),
            }
        )
    front = set(front_nodes)
    edge_count = defaultdict(int)
    for face in requested:
        for j in range(3):
            edge_count[tuple(sorted((face[j], face[(j + 1) % 3])))] += 1
    front_candidates = set()
    for edge, count in edge_count.items():
        if count == 1:
            front_candidates.update(edge)
            if c.shape[1] == 10:
                for face in requested:
                    if set(edge) <= set(face):
                        cell = adjacency[face][0][0]
                        local = tuple(
                            sorted(
                                int(np.flatnonzero(c[cell, :4] == node)[0])
                                for node in edge
                            )
                        )
                        front_candidates.add(int(c[cell, _EDGES[local]]))
                        break
    if not front <= front_candidates:
        raise ValueError(
            "AFM-SEAM: shared crack-front nodes must lie on the selected surface boundary."
        )
    split_nodes -= front
    if not split_nodes:
        raise ValueError(
            "AFM-SEAM: crack-front declaration leaves no independent seam nodes."
        )
    for record in records:
        record["shared_front_source_labels"] = [int(labels[i]) for i in sorted(front)]
    # Reject a partial surface that would inadvertently split adjacent unselected
    # faces. The caller must provide the complete boundary between partitions.
    for face, sides in adjacency.items():
        if (
            len(sides) == 2
            and sum(i in positive for i, _ in sides) == 1
            and face not in requested
        ):
            if split_nodes.intersection(sides[0][1]):
                raise ValueError(
                    "AFM-SEAM: incomplete interface; splitting would open an unrequested face."
                )
    original = sorted(split_nodes)
    mapping = {old: len(x) + j for j, old in enumerate(original)}
    x = np.concatenate((x, x[original]), axis=0)
    for i in positive:
        c[i] = [mapping.get(int(j), int(j)) for j in c[i]]
    newlabels = np.arange(
        int(labels.max()) + 1, int(labels.max()) + 1 + len(original), dtype=np.int64
    )
    return TetSeam(
        x,
        c,
        np.concatenate((labels, newlabels)),
        {int(new): int(labels[old]) for old, new in zip(original, newlabels)},
        tuple(records),
    )

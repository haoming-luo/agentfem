"""Reusable exact-periodic 2 x 2 x 2 unit-cube fixture.

The source-node numbering follows ``1 + i + 3*j + 9*k``. Nodes on one or
more positive faces are related to their wrapped counterpart and the three
macroscopic reference nodes through Abaqus ``*EQUATION`` semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dolfinx import mesh as dolfinx_mesh

from agentfem import constraints
from agentfem.mesh import abaqus


@dataclass(frozen=True)
class PeriodicCubeFixture:
    """Structured unit cube plus exact periodic source semantics."""

    domain: object
    nodes: abaqus.AbaqusNodeTable
    equations: abaqus.AbaqusEquationSet
    deformation_gradient: np.ndarray
    anchor_node: int = 1
    reference_nodes: tuple[int, int, int] = (3, 7, 19)

    def constraint(self, displacement):
        return constraints.abaqus_periodic_cell(
            displacement,
            nodes=self.nodes,
            equations=self.equations,
            deformation_gradient=self.deformation_gradient,
            anchor_node=self.anchor_node,
            reference_nodes=self.reference_nodes,
        )


def periodic_unit_cube(comm, *, stretch: float = 1.02, cells: int = 2) -> PeriodicCubeFixture:
    """Return a structured periodic tetrahedral cell with isochoric macro tension."""

    if not np.isfinite(stretch) or stretch <= 0.0:
        raise ValueError("stretch must be finite and positive.")
    if type(cells) is not int or cells < 1:
        raise ValueError("cells must be a positive integer.")
    side = cells + 1
    domain = dolfinx_mesh.create_unit_cube(comm, cells, cells, cells)
    coordinates = []
    labels = []
    label_by_index = {}
    for k in range(side):
        for j in range(side):
            for i in range(side):
                label = 1 + i + side * j + side**2 * k
                labels.append(label)
                coordinates.append((i / cells, j / cells, k / cells))
                label_by_index[(i, j, k)] = label

    anchor = label_by_index[(0, 0, 0)]
    references = (
        label_by_index[(cells, 0, 0)],
        label_by_index[(0, cells, 0)],
        label_by_index[(0, 0, cells)],
    )
    control_nodes = {anchor, *references}
    equations = []
    for index, slave in label_by_index.items():
        active_axes = tuple(axis for axis, value in enumerate(index) if value == cells)
        if not active_axes or slave in control_nodes:
            continue
        base_index = list(index)
        for axis in active_axes:
            base_index[axis] = 0
        base = label_by_index[tuple(base_index)]
        for component in (1, 2, 3):
            terms = [
                abaqus.EquationTerm(slave, component, 1.0),
                abaqus.EquationTerm(base, component, -1.0),
            ]
            terms.extend(
                abaqus.EquationTerm(references[axis], component, -1.0)
                for axis in active_axes
            )
            terms.append(
                abaqus.EquationTerm(anchor, component, float(len(active_axes)))
            )
            equations.append(abaqus.LinearEquation(tuple(terms)))

    lateral = 1.0 / np.sqrt(float(stretch))
    return PeriodicCubeFixture(
        domain=domain,
        nodes=abaqus.AbaqusNodeTable(
            labels=np.asarray(labels, dtype=np.int64),
            coordinates=np.asarray(coordinates, dtype=float),
        ),
        equations=abaqus.AbaqusEquationSet(tuple(equations)),
        deformation_gradient=np.diag((float(stretch), lateral, lateral)),
        anchor_node=anchor,
        reference_nodes=references,
    )

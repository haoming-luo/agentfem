# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Published enriched degree-two mass-lumped tetrahedron (15 nodes).

Space P2 + four cubic face bubbles + one quartic interior bubble. Positive
nodal weights follow Geevers, Mulder & van der Vegt (2018), Table 1,
DOI:10.1137/18M1175549. This is NOT the Abaqus C3D10M formulation.
"""

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def scalar_element():
    """Construct the published polynomial space using Basix, not hand assembly."""
    import basix

    cell = basix.CellType.tetrahedron
    vertices = basix.geometry(cell)
    topology = basix.topology(cell)
    x = [
        [vertices[entity].mean(axis=0, keepdims=True) for entity in entities]
        for entities in topology
    ]
    M = [[np.ones((1, 1, 1, 1)) for entity in entities] for entities in topology]
    q, w = basix.make_quadrature(cell, 8)
    a, b, c = q.T
    bary = np.column_stack((1 - a - b - c, a, b, c))
    generators = np.array(
        [
            np.ones(len(q)),
            a,
            b,
            c,
            a * a,
            a * b,
            a * c,
            b * b,
            b * c,
            c * c,
            *[np.prod(bary[:, face], axis=1) for face in topology[2]],
            np.prod(bary, axis=1),
        ]
    )
    polynomials = basix.polynomials.tabulate_polynomials(
        basix.PolynomialType.legendre, cell, 4, q
    )
    coefficients = (generators * w) @ polynomials.T
    return basix.create_custom_element(
        cell,
        [],
        coefficients,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        2,
        4,
        basix.PolysetType.standard,
    )


def tet15_mass_lumped(*, shape=(3,)):
    """Return a UFL element for an explicitly selected enriched displacement."""
    import basix.ufl

    element = basix.ufl.wrap_element(scalar_element())
    return basix.ufl.blocked_element(element, shape=tuple(shape)) if shape else element


def nodal_weights():
    """Reference tetrahedron weights, in Basix entity order; sum = 1/6."""
    return np.array([17 / 5040] * 4 + [2 / 315] * 6 + [9 / 560] * 4 + [16 / 315])


def assemble_tet15_mass(V, density):
    from dolfinx import fem
    from petsc4py import PETSc
    from ..provenance import collective_call
    import basix

    if (
        basix.finite_element.FiniteElement(V.element.basix_element) != scalar_element()
        or V.mesh.geometry.dim != 3
    ):
        raise ValueError("AFM-TET15-MASS: use elements.tet15_mass_lumped on a 3D mesh.")
    rho = float(density)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("AFM-TET15-MASS: finite positive density required.")
    coords = V.tabulate_dof_coordinates()
    points = scalar_element().points
    weights = nodal_weights()

    def local():
        entries = []
        for cell in range(V.mesh.topology.index_map(3).size_local):
            ds = V.dofmap.cell_dofs(cell)
            x = coords[ds]
            J = (x[1:4] - x[0]).T
            h = np.max(np.linalg.norm(x[:, None] - x[None, :], axis=2))
            determinant = abs(np.linalg.det(J))
            if determinant <= 1e-12 * h**3:
                raise ValueError(
                    "AFM-TET15-MASS: degenerate or severely distorted cell."
                )
            if not np.allclose(
                x, x[0] + points @ J.T, atol=max(1e-14, 1e-10 * h), rtol=0
            ):
                raise NotImplementedError(
                    "AFM-TET15-MASS: affine straight-sided geometry required."
                )
            entries.append((ds.copy(), rho * determinant * weights))
        return entries

    entries = collective_call(local, comm=V.mesh.comm, label="Tet15 nodal mass")
    f = fem.Function(V)
    bs = V.dofmap.index_map_bs
    for ds, m in entries:
        np.add.at(f.x.array.reshape(-1, bs), ds, m[:, None])
    f.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
    )
    f.x.scatter_forward()
    return f.x.array.copy()


def is_tet15_space(V):
    """Recognize the exact published enriched basis, not merely its node count."""
    import basix

    try:
        element = V.element.basix_element
        return (
            element.family == basix.ElementFamily.custom
            and basix.finite_element.FiniteElement(element) == scalar_element()
        )
    except (RuntimeError, AttributeError):
        return False


def visualization_mesh(V):
    """Use all 15 field nodes in 24 linear display tetrahedra per source cell.

    This is display interpolation, not the solver's enriched polynomial. The
    caller must retain original cell DOFs and identify the display subdivision.
    """
    import basix

    if not is_tet15_space(V):
        raise ValueError("Expected the published Tet15 space.")
    topology = basix.topology(basix.CellType.tetrahedron)
    edges = {tuple(sorted(edge)): 4 + i for i, edge in enumerate(topology[1])}
    points = scalar_element().points
    subdivisions = []
    for f, face in enumerate(topology[2]):
        for j, a in enumerate(face):
            b = face[(j + 1) % 3]
            mid = edges[tuple(sorted((a, b)))]
            for tri in ((a, mid), (mid, b)):
                row = [14, 10 + f, *tri]
                p = points[row]
                if np.linalg.det((p[1:] - p[0]).T) < 0:
                    row[-2:] = row[-2:][::-1]
                subdivisions.append(row)
    subdivisions = np.array(subdivisions)
    cell_dofs = np.asarray(V.dofmap.list)
    cells = cell_dofs[:, subdivisions].reshape(-1, 4)
    topology_out = np.column_stack((np.full(len(cells), 4), cells)).ravel()
    return (
        topology_out,
        np.full(len(cells), 10, dtype=np.uint8),
        V.tabulate_dof_coordinates(),
    )

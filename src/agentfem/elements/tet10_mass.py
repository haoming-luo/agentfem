# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Element-local HRZ diagonal scaling for quadratic tetrahedral inertia.

This is ordinary quadratic Lagrange interpolation with HRZ inertia, NOT
Abaqus C3D10M. It has no auxiliary displacement or hourglass stabilization.
Local consistent-mass diagonal scaling conserves each element's total mass;
it does not reproduce all inertia moments or the consistent-mass spectrum.
"""

import numpy as np


def hrz_tet10_mass(coordinates, density):
    """Return scalar nodal masses in Basix P2 tetrahedron ordering.

    Coordinates are the ten interpolation points of a straight-sided affine
    tetrahedron. Curved coordinates require a separately verified integration
    rule and are rejected. This local kernel is independent of MPI/DOLFINx.
    """
    import basix

    x = np.asarray(coordinates, dtype=float)
    rho = float(density)
    if (
        x.shape != (10, 3)
        or not np.all(np.isfinite(x))
        or not np.isfinite(rho)
        or rho <= 0
    ):
        raise ValueError(
            "AFM-TET10-HRZ: finite (10,3) coordinates and positive density required."
        )
    el = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.tetrahedron,
        2,
        basix.LagrangeVariant.equispaced,
    )
    J = (x[1:4] - x[0]).T
    volume = abs(np.linalg.det(J)) / 6
    h = np.max(np.linalg.norm(x[:, None] - x[None, :], axis=2))
    if not np.isfinite(volume) or volume <= 1e-12 * h**3:
        raise ValueError("AFM-TET10-HRZ: degenerate or severely distorted tetrahedron.")
    if not np.allclose(x, x[0] + el.points @ J.T, rtol=0, atol=max(h * 1e-10, 1e-14)):
        raise NotImplementedError(
            "AFM-TET10-HRZ: curved/nonmidpoint geometry is not admitted."
        )
    q, w = basix.make_quadrature(basix.CellType.tetrahedron, 4)
    N = el.tabulate(0, q)[0, :, :, 0]
    diag = (N * N).T @ w
    return rho * volume * diag / diag.sum()


def assemble_tet10_hrz(V, density):
    """Assemble positive element-local inertia with owned-cell MPI reduction."""
    from dolfinx import fem
    from petsc4py import PETSc
    from ..provenance import collective_call

    if (
        V.mesh.topology.dim != 3
        or V.mesh.topology.cell_name() != "tetrahedron"
        or V.element.basix_element.degree != 2
    ):
        raise ValueError("AFM-TET10-HRZ: quadratic tetrahedral nodal field required.")
    if V.dofmap.bs != V.dofmap.index_map_bs:
        raise ValueError("AFM-TET10-HRZ: mixed spaces are unsupported.")
    coords = V.tabulate_dof_coordinates()
    n = V.mesh.topology.index_map(3).size_local

    def local():
        entries = []
        for cell in range(n):
            ds = V.dofmap.cell_dofs(cell)
            if len(ds) != 10:
                raise ValueError("AFM-TET10-HRZ: expected ten nodal blocks.")
            entries.append((ds.copy(), hrz_tet10_mass(coords[ds], density)))
        return entries

    entries = collective_call(local, comm=V.mesh.comm, label="Tet10 HRZ mass")
    f = fem.Function(V)
    bs = V.dofmap.index_map_bs
    for ds, mass in entries:
        np.add.at(f.x.array.reshape(-1, bs), ds, mass[:, None])
    f.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
    )
    f.x.scatter_forward()
    return f.x.array.copy()

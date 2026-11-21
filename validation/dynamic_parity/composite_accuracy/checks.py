# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Manufactured pure-bending convergence and production-kernel profiling.

Only synthetic meshes; static equilibrium is an independent scipy harness over
production element matrices, not a new public static Step provider.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
from agentfem import mesh, fields, elements
from agentfem.elements._composite_tet_dolfinx import CompositeTetResidual


def elastic(nu):
    mu = 1 / (2 * (1 + nu))
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    C = np.diag([2 * mu] * 3 + [mu] * 3)
    C[:3, :3] += lam
    return C


def make(n, nu, mean, distort=False):
    domain = mesh.cuboid(
        (0, 0, 0),
        (1, 1, 1),
        (n, n, n) if np.isscalar(n) else n,
        comm=MPI.COMM_SELF,
        cell_type="tetrahedron",
    )
    if distort:
        x = domain.geometry.x
        bump = np.prod(np.sin(np.pi * x), axis=1)
        x[:, :3] += 0.12 / n * bump[:, None] * np.array([1, -0.7, 0.4])
    u = fields.displacement(
        domain, element_policy=elements.composite_tet10_policy(mean_dilatation=mean)
    )
    count = domain.topology.index_map(3).size_local
    op = CompositeTetResidual(
        u,
        elastic(nu),
        np.ones(count),
        elements.composite_tet10_policy(mean_dilatation=mean),
    )
    return u, op


def bending(n, nu, mean, distort, reference=None):
    u, op = make(n, nu, mean, distort)
    if reference is not None:
        import basix
        from types import SimpleNamespace

        domain = u.function_space.mesh
        v = fields.displacement(domain, degree=reference)
        V = v.function_space
        count = domain.topology.index_map(3).size_local
        q, w = basix.make_quadrature(basix.CellType.tetrahedron, 4)
        tab = V.element.basix_element.tabulate(1, q)
        nodes = np.asarray([V.dofmap.cell_dofs(c) for c in range(count)])
        xyz = V.tabulate_dof_coordinates()
        matrices = []
        masses = np.zeros(len(xyz) * 3)
        for c, ns in enumerate(nodes):
            geom = domain.geometry.x[domain.geometry.dofmaps[0][c]][:4]
            J = (geom[1:] - geom[0]).T
            grad = tab[1:4, :, :, 0].transpose(1, 2, 0) @ np.linalg.inv(J)
            B = np.zeros((len(q), 6, len(ns) * 3))
            for i in range(3):
                B[:, i, i::3] = grad[:, :, i]
            for row, (i, j) in enumerate(((0, 1), (1, 2), (0, 2)), 3):
                B[:, row, i::3] = grad[:, :, j]
                B[:, row, j::3] = grad[:, :, i]
            matrices.append(
                abs(np.linalg.det(J))
                * np.einsum("q,qai,ab,qbj->ij", w, B, elastic(nu), B, optimize=True)
            )
            # Positive sampling norm, not a P2 row-sum mass model.
            np.add.at(
                masses,
                (3 * ns[:, None] + np.arange(3)).ravel(),
                abs(np.linalg.det(J)) / len(ns),
            )
        u = v
        op = SimpleNamespace(
            cell_nodes=nodes,
            stiffness=np.array(matrices),
            mass_diagonal=masses,
            stable_dt=lambda: None,
        )
    x, y, z = (u.function_space.tabulate_dof_coordinates() - 0.5).T
    exact = np.column_stack(
        (-x * y, (x * x + nu * (y * y - z * z)) / 2, nu * y * z)
    ).ravel()
    ids = (3 * op.cell_nodes[:, :, None] + np.arange(3)).reshape(len(op.cell_nodes), -1)
    rows = np.broadcast_to(ids[:, :, None], op.stiffness.shape).ravel()
    cols = np.broadcast_to(ids[:, None, :], op.stiffness.shape).ravel()
    K = coo_matrix(
        (op.stiffness.ravel(), (rows, cols)), shape=(len(exact),) * 2
    ).tocsr()
    boundary = np.repeat(
        np.any(np.isclose(np.abs(np.array([x, y, z]).T), 0.5), axis=1), 3
    )
    free = np.flatnonzero(~boundary)
    sol = np.where(boundary, exact, 0)
    sol[free] = spsolve(K[free][:, free], -(K @ sol)[free])
    err = np.sqrt(
        np.dot(op.mass_diagonal, (sol - exact) ** 2)
        / np.dot(op.mass_diagonal, exact**2)
    )
    energy = 0.5 * sol @ (K @ sol)
    return dict(
        n=n,
        reference=reference,
        poisson=nu,
        mean_dilatation=mean,
        distorted=distort,
        cells=len(ids),
        relative_nodal_error=float(err),
        energy=float(energy),
        exact_energy=1 / 24,
        energy_relative_error=float(abs(energy - 1 / 24) / (1 / 24)),
        dt=op.stable_dt(),
    )


def longitudinal_mode(n, mean=True):
    from scipy.sparse.linalg import eigsh

    _, op = make((n, 1, 1), 0.0, mean)
    xyz = op.displacement.function_space.tabulate_dof_coordinates()
    ids = (3 * op.cell_nodes[:, :, None] + np.arange(3)).reshape(-1, 30)
    shape = op.stiffness.shape
    K = coo_matrix(
        (
            op.stiffness.ravel(),
            (
                np.broadcast_to(ids[:, :, None], shape).ravel(),
                np.broadcast_to(ids[:, None, :], shape).ravel(),
            ),
        ),
        shape=(len(xyz) * 3,) * 2,
    ).tocsr()
    free = 3 * np.flatnonzero((xyz[:, 0] > 1e-10) & (xyz[:, 0] < 1 - 1e-10))
    from scipy.sparse import diags

    omega = float(
        np.sqrt(
            eigsh(
                K[free][:, free],
                M=diags(op.mass_diagonal[free]),
                k=1,
                sigma=0,
                return_eigenvectors=False,
            )[0]
        )
    )
    dt = op.stable_dt()
    return dict(
        n=n,
        omega=omega,
        exact_omega=float(np.pi),
        relative_error=float(abs(omega / np.pi - 1)),
        dt=dt,
        temporal_frequency_errors=[
            float(abs(2 * np.arcsin(omega * dt * f / 2) / (dt * f) / omega - 1))
            for f in (1.0, 0.5, 0.25)
        ],
    )


def profile(n=3):
    times = []
    for _ in range(3):
        t = time.perf_counter()
        u, op = make(n, 0.3, True)
        setup = time.perf_counter() - t
        rng = np.random.default_rng(10)
        u.value.x.array[:] = rng.normal(size=u.value.x.array.size) * 1e-4
        t = time.perf_counter()
        for _ in range(20):
            v = op.assemble_vector()
            v.destroy()
            op.energies()
        times.append(dict(setup_s=setup, force_energy_20_s=time.perf_counter() - t))
    return dict(cells=len(op.cell_nodes), samples=times)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--profile-only", action="store_true")
    a = p.parse_args()
    data = {"profile": profile()}
    if not a.profile_only:
        data["reference"] = [
            bending(4, nu, False, False, reference=d)
            for nu in (0.3, 0.4999)
            for d in (1, 2)
        ]
        data["wave"] = [longitudinal_mode(n) for n in (4, 8, 16)]
        data["bending"] = [
            bending(n, nu, mean, distort)
            for nu in (0.3, 0.4999)
            for mean in (False, True)
            for distort in (False, True)
            for n in (2, 3, 4)
        ]
    Path(a.output).write_text(json.dumps(data, indent=2) + "\n")

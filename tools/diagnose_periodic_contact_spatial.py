# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Separate elastic FE spatial error from AL and plastic return mapping.

Research diagnostic, NOT a new contact solver. On the supported periodic
fixture, impose the known fully active nodal patch by exact sparse elimination
and solve independent *small-strain linear elasticity* with DOLFINx/PETSc.
The material law differs from finite Hencky elasticity. No numerical agreement
with finite-J2 contact is presumed. All contact nodal reactions must be
compressive; otherwise the assumed active set is invalid and the study stops.
"""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import runpy
from time import perf_counter

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fp
from petsc4py import PETSc
from scipy.sparse import csr_matrix

from agentfem.checkpointing import atomic_write_text


def graded_coordinate_lines(cells):
    """Same physical patch, clustered near its edges and the loaded faces.

    This changes the mesh approximation, not the physical boundary conditions.
    It is a diagnostic family, not an automatic meshing policy.
    """
    s = np.linspace(0., 1., cells // 4 + 1)
    local = .5 * (1. - np.cos(np.pi * s))
    tangential = np.concatenate([(.25 * (k + local))[:-1] for k in range(4)] + [np.array([1.])])
    normal = .5 * (1. - np.cos(np.pi * np.linspace(0., 1., cells + 1)))
    return np.vstack([normal, tangential, tangential])


def solve_linear_reference(cells, *, amplitude=1., graded=False):
    if not np.isfinite(amplitude) or amplitude <= 0.:
        raise ValueError("The diagnostic requires positive finite amplitude.")
    make = runpy.run_path(str(
        Path(__file__).resolve().parents[1] / "examples/finite_j2_periodic_augmented_plane.py"
    ))["make_step"]
    step = make(cells=cells, increments=4, affine_end_face_support=True, yield_stress=1e6,
                coordinate_lines=graded_coordinate_lines(cells) if graded else None)
    reduction = step.constraint.reduction(.25)
    T = csr_matrix((reduction.coefficients, reduction.column_indices, reduction.row_offsets),
                   shape=(reduction.full_size, reduction.reduced_size))
    trace = step.contact_provider.adapter.trace
    nodes = trace._direct_nodes
    positions = step.contact_provider.adapter.reference_nodal_positions
    if nodes is None or len(set(nodes)) != len(nodes):
        raise ValueError("Diagnostic requires distinct direct nodal contact rows.")
    offset = amplitude * reduction.offset
    columns, factors = [], []
    qbar = np.zeros(reduction.reduced_size)
    for node in nodes:
        row = T.getrow(3 * int(node))
        if row.nnz != 1:
            raise ValueError("Diagnostic requires one Cartesian independent DOF per normal row.")
        column, coefficient = int(row.indices[0]), float(row.data[0])
        columns.append(column)
        factors.append(coefficient)
        qbar[column] = (amplitude * (1.005 - positions[node, 0]) - offset[3 * node]) / coefficient
    if len(set(columns)) != len(columns):
        raise ValueError("Contact rows are not disjoint.")
    free = np.setdiff1d(np.arange(reduction.reduced_size), columns)
    H = T[:, free].tocsr()
    imposed = offset + T @ qbar
    V = step.solution.function_space
    du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    mu, lame = 1000. / 2.6, 1000. * .3 / (1.3 * .4)
    strain = ufl.sym(ufl.grad(du))
    stress = 2 * mu * strain + lame * ufl.tr(strain) * ufl.Identity(3)
    form = fem.form(ufl.inner(stress, ufl.sym(ufl.grad(v))) * ufl.dx)
    started = perf_counter()
    with ExitStack() as owned:
        def keep(obj):
            owned.callback(obj.destroy)
            return obj
        K = keep(fp.assemble_matrix(form))
        K.assemble()
        transform = keep(PETSc.Mat().createAIJ(
            size=H.shape, csr=(H.indptr.astype(PETSc.IntType), H.indices.astype(PETSc.IntType), H.data),
            comm=PETSc.COMM_SELF,
        ))
        reduced = keep(K.PtAP(transform))
        prescribed = keep(K.createVecRight())
        prescribed.array[:] = imposed
        force = keep(K.createVecLeft())
        K.mult(prescribed, force)
        rhs = keep(transform.createVecRight())
        transform.multTranspose(force, rhs)
        rhs.scale(-1.)
        q = keep(rhs.duplicate())
        ksp = keep(PETSc.KSP().create(PETSc.COMM_SELF))
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setOperators(reduced)
        ksp.solve(rhs, q)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError(f"Independent linear reference failed: {ksp.getConvergedReason()}")
        u = keep(K.createVecRight())
        transform.mult(q, u)
        u.axpy(1., prescribed)
        K.mult(u, force)
        transform.multTranspose(force, rhs)
        residual = rhs.norm()
        # Positive magnitudes act against positive x motion. This formula
        # follows virtual work of the eliminated Cartesian q coordinates.
        reduced_force = np.asarray(T.T @ force.array_r)
        contact_force = -reduced_force[columns] / np.asarray(factors)
        if np.min(contact_force) < -1e-8 * amplitude:
            raise RuntimeError("Fully active reference is inadmissible: a nodal contact force is tensile.")
        points = positions[nodes]
        edge = np.any(np.isclose(points[:, 1:], .25) | np.isclose(points[:, 1:], .75), axis=1)
        total = float(np.sum(contact_force))
        numerical_gap = amplitude * .005 - u.array_r.reshape(-1, 3)[nodes, 0]
        return {
            "cells_per_edge": cells, "full_dofs": len(imposed), "contact_nodes": len(nodes),
            "amplitude": amplitude, "reaction_magnitude": total,
            "mesh_family": "edge_graded_cosine" if graded else "uniform",
            "amplitude_normalized_reaction": total / amplitude,
            "edge_nodal_force_fraction": float(np.sum(contact_force[edge]) / total),
            "minimum_nodal_force": float(np.min(contact_force)),
            "free_residual_norm": float(residual),
            "maximum_absolute_gap": float(np.max(abs(numerical_gap))),
            "elastic_energy": float(.5 * u.dot(force)),
            "assembly_and_solve_seconds": perf_counter() - started,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", type=int, default=[4, 8, 12, 16])
    parser.add_argument("--amplitude", type=float, default=1.)
    parser.add_argument("--graded", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.amplitude) or args.amplitude <= 0.:
        parser.error("Use a fresh output path and positive finite amplitude.")
    report = {
        "scope": "independent linear-elastic sparse exact-active-patch diagnostic",
        "exclusions": "Not finite-Hencky, plastic history, arbitrary active-set search or external-code validation",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for cells in args.cells:
        row = solve_linear_reference(cells, amplitude=args.amplitude, graded=args.graded)
        report["records"].append(row)
        atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Parametric re-entrant honeycomb geometry; independent of any solver."""

import math
import numpy as np
from mpi4py import MPI


def reentrant_honeycomb(
    *,
    angle=-30.0,
    ligament=1.0,
    vertical=1.5,
    thickness=0.12,
    repeats=(1, 1),
    mesh_size=0.06,
    comm=MPI.COMM_WORLD,
    connected_specimen=False,
):
    """Mesh a rectangular cut of a periodic rounded-joint honeycomb.

    ``angle`` (degrees) is the inclined ligament angle from horizontal and is
    negative for re-entrant cells. ``repeats`` counts rectangular periods.
    Walls are unions of width-thickness strips and radius-thickness/2 joints.
    With connected_specimen=True retain only the largest connected solid
    component after cutting, removing detached boundary fragments; this is
    a finite specimen, not the periodic cell. Otherwise opposite cut edges
    have matching Gmsh nodes. Serial geometry generation
    runs on rank zero; the imported solid mesh is distributed. Requires Gmsh.
    """
    from . import import_gmsh_model, require_gmsh

    if not all(
        np.isfinite(x) for x in (angle, ligament, vertical, thickness, mesh_size)
    ):
        raise ValueError("Honeycomb dimensions must be finite.")
    if not -80.0 < angle < 0.0 or min(ligament, vertical, thickness, mesh_size) <= 0:
        raise ValueError(
            "Re-entrant angle must lie in (-80,0) degrees; dimensions must be positive."
        )
    if len(repeats) != 2 or any(
        isinstance(n, bool) or int(n) != n or n < 1 for n in repeats
    ):
        raise ValueError("repeats must be two positive integers.")
    w = ligament * math.cos(math.radians(angle))
    s = ligament * math.sin(math.radians(angle))
    if vertical <= -2 * s + thickness or thickness >= min(ligament, vertical) / 3:
        raise ValueError("Honeycomb walls overlap or close the re-entrant opening.")
    width, height = 2 * w * repeats[0], 2 * (vertical + s) * repeats[1]
    gmsh = require_gmsh()
    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize()
    previous_model = gmsh.model.getCurrent()
    saved_sizes = [
        gmsh.option.getNumber(name) for name in ("Mesh.MeshSizeMin", "Mesh.MeshSizeMax")
    ]
    error = None
    try:
        if comm.rank == 0:
            try:
                gmsh.model.add("agentfem_reentrant_honeycomb")
                occ = gmsh.model.occ
                vertices = np.array(
                    [
                        [0.0, 0.0],
                        [w, s],
                        [w, s + vertical],
                        [0.0, 2 * s + vertical],
                        [-w, s + vertical],
                        [-w, s],
                    ]
                )
                edges, points = {}, {}
                for j in range(-2, 2 * int(repeats[1]) + 3):
                    for i in range(-int(repeats[1]) - 3, int(repeats[0]) + 3):
                        vv = vertices + [2 * w * i + w * j, (vertical + s) * j]
                        for a, b in zip(vv, np.roll(vv, -1, axis=0)):
                            if (
                                max(a[0], b[0]) < -thickness
                                or min(a[0], b[0]) > width + thickness
                                or max(a[1], b[1]) < -thickness
                                or min(a[1], b[1]) > height + thickness
                            ):
                                continue
                            ka, kb = tuple(np.round(a, 12)), tuple(np.round(b, 12))
                            edges[tuple(sorted((ka, kb)))] = (a, b)
                            points[ka] = a
                            points[kb] = b
                pieces = []
                for a, b in edges.values():
                    d = b - a
                    n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * thickness / 2
                    vs = [a + n, b + n, b - n, a - n]
                    ids = [occ.addPoint(float(p[0]), float(p[1]), 0) for p in vs]
                    lines = [occ.addLine(ids[k], ids[(k + 1) % 4]) for k in range(4)]
                    pieces.append((2, occ.addPlaneSurface([occ.addCurveLoop(lines)])))
                for p in points.values():
                    pieces.append(
                        (
                            2,
                            occ.addDisk(
                                float(p[0]),
                                float(p[1]),
                                0,
                                thickness / 2,
                                thickness / 2,
                            ),
                        )
                    )
                fused, _ = occ.fuse(pieces[:1], pieces[1:])
                clipped, _ = occ.intersect(
                    fused, [(2, occ.addRectangle(0, 0, 0, width, height))]
                )
                if connected_specimen:
                    largest = max(clipped, key=lambda item: occ.getMass(*item))
                    occ.remove(
                        [item for item in clipped if item != largest], recursive=True
                    )
                    clipped = [largest]
                occ.synchronize()
                gmsh.model.addPhysicalGroup(2, [t for d, t in clipped if d == 2], 1)
                gmsh.model.setPhysicalName(2, 1, "solid")
                curves = gmsh.model.getBoundary(clipped, oriented=False, combined=True)
                for axis, span in (
                    () if connected_specimen else ((0, width), (1, height))
                ):
                    low = []
                    high = []
                    for d, t in curves:
                        bounds = gmsh.model.getBoundingBox(d, t)
                        if (
                            abs(bounds[axis]) < 1.0e-6
                            and abs(bounds[axis + 3]) < 1.0e-6
                        ):
                            low.append(t)
                        if (
                            abs(bounds[axis] - span) < 1.0e-6
                            and abs(bounds[axis + 3] - span) < 1.0e-6
                        ):
                            high.append(t)
                    used = set()
                    for slave in high:
                        center = np.array(occ.getCenterOfMass(1, slave))
                        center[axis] -= span
                        matches = [
                            master
                            for master in low
                            if np.linalg.norm(
                                np.array(occ.getCenterOfMass(1, master)) - center
                            )
                            < 1.0e-7
                        ]
                        if len(matches) != 1 or matches[0] in used:
                            raise ValueError(
                                "Honeycomb opposite-edge matching is ambiguous."
                            )
                        used.add(matches[0])
                        transform = np.eye(4)
                        transform[axis, 3] = span
                        gmsh.model.mesh.setPeriodic(
                            1, [slave], matches, transform.reshape(-1).tolist()
                        )
                    if len(used) != len(low):
                        raise ValueError("Honeycomb cut is not periodic.")
                gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
                gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
                gmsh.model.mesh.generate(2)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = comm.bcast(error, root=0)
        if error:
            raise RuntimeError(error)
        return import_gmsh_model(gmsh.model, comm, gdim=2)
    finally:
        if comm.rank == 0:
            gmsh.model.remove()
        for option, value in zip(("Mesh.MeshSizeMin", "Mesh.MeshSizeMax"), saved_sizes):
            gmsh.option.setNumber(option, value)
        if previous_model and previous_model in gmsh.model.list():
            gmsh.model.setCurrent(previous_model)
        if owned:
            gmsh.finalize()

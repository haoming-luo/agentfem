# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the guide figure from numerical solutions; run at repository root."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np, runpy
from agentfem import io

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
step = runpy.run_path("tests/test_stability_periodic.py")["column"](nx=48, length=20.0)
r = step.solve_result()
u = r.field("Buckling_mode_1")


def mesh_lines(function, scale):
    p = io.interpolate_for_xdmf(function, degree=1)
    V = p.function_space
    xyz = V.tabulate_dof_coordinates()[:, :2]
    vals = p.x.array.reshape(-1, 2)
    conn = [
        V.dofmap.cell_dofs(i)
        for i in range(V.mesh.topology.index_map(V.mesh.topology.dim).size_local)
    ]
    lines = []
    for c in conn:
        order = [0, 1, 3, 2, 0] if len(c) == 4 else [0, 1, 2, 0]
        pts = (xyz + scale * vals)[c[order]]
        lines.extend([[pts[k], pts[k + 1]] for k in range(len(pts) - 1)])
    return lines


ax[0].add_collection(LineCollection(mesh_lines(u, 0), color="#b7bbc1", lw=0.3))
ax[0].add_collection(LineCollection(mesh_lines(u, 2), color="#315b77", lw=0.35))
ax[0].autoscale()
ax[0].set_aspect("equal")
ax[0].set_title("(a) First buckling mode", loc="left", weight="bold")
ax[0].set_xlabel("Axial coordinate")
ax[0].set_ylabel("Transverse coordinate")
ax[0].text(
    0.02,
    0.95,
    "Relative shape; display amplitude = 2\nCritical stress = %.6f"
    % r.quantity("load_factors")[0],
    transform=ax[0].transAxes,
    va="top",
    fontsize=9,
)
result = runpy.run_path("examples/reentrant_honeycomb/case.py")["main"]()
u = result.field("Displacement")
ax[1].add_collection(LineCollection(mesh_lines(u, 0), color="#b7bbc1", lw=0.2))
ax[1].add_collection(LineCollection(mesh_lines(u, 50), color="#267f78", lw=0.25))
ax[1].autoscale()
ax[1].set_aspect("equal")
ax[1].set_title("(b) Connected re-entrant specimen", loc="left", weight="bold")
ax[1].set_xlabel("Axial coordinate")
ax[1].set_ylabel("Transverse coordinate")
ax[1].text(
    0.02,
    0.98,
    "Displacement magnified 50 times\nApparent Poisson ratio = %.3f"
    % result.quantity("apparent_poisson_ratio"),
    transform=ax[1].transAxes,
    va="top",
    fontsize=9,
    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
)
fig.savefig("docs/assets/stability_periodic.png", dpi=190)

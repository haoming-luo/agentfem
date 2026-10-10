# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce analytical and continuum arch continuation evidence."""

import json
import runpy
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

functions = runpy.run_path("tests/test_arc_length.py")
path = functions["arch_path"]()
path.run(100, stop=lambda r: r["displacement"][0] > 0.42)
q = np.array([r["displacement"][0] for r in path.history])
force = np.array([r["load_factor"] * 0.03 for r in path.history])
exact = np.array([functions["arch_force"](np.array([x]))[0][0] for x in q])
solve = runpy.run_path("examples/shallow_arch_arc_length/case.py")["solve_arch"]
records = []
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
axes[0].plot(q, exact, color="#345f82", label="Analytical two-bar equilibrium")
axes[0].plot(
    q[::3],
    force[::3],
    "o",
    ms=3,
    mfc="white",
    color="#aa6844",
    label="Arc-length continuation",
)
for nx, size, n, style, color in [
    (32, 0.04, 80, "-", "#345f82"),
    (64, 0.02, 160, "--", "#2b8c7f"),
]:
    result, disp, load = solve(nx=nx, arc_size=size, increments=n)
    axes[1].plot(-disp, load, style, color=color, label=f"nx={nx}, arc={size}")
    records.append(
        dict(
            nx=nx,
            arc_size=size,
            displacement=(-disp).tolist(),
            load_factor=load.tolist(),
            maximum_residual=max(result.quantity("equilibrium_residuals")),
            maximum_arc_error=max(result.quantity("arc_constraint_errors")),
        )
    )
selected = np.linspace(0.02, 0.32, 30)
coarse = np.interp(selected, records[0]["displacement"], records[0]["load_factor"])
fine = np.interp(selected, records[1]["displacement"], records[1]["load_factor"])
report = dict(
    two_bar_maximum_force_error=float(max(abs(force - exact))),
    continuum_curves=records,
    normalized_curve_difference=float(max(abs(coarse - fine)) / max(abs(fine))),
)
for ax, title, ylabel in zip(
    axes,
    ["(a) Analytical shallow two-bar arch", "(b) Clamped 2D solid arch"],
    ["Vertical force", "Reference traction load factor"],
):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("Downward crown displacement")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=7)
fig.savefig("docs/assets/arc_length_arches.png", dpi=180)
Path("evidence/stability_periodic/arc_length.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(report["two_bar_maximum_force_error"], report["normalized_curve_difference"])

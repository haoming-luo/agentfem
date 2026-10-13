# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce imperfect-column path data and a guide figure from repository root."""

import json
import runpy
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

solve = runpy.run_path("examples/imperfect_column/case.py")["solve_column"]
records = []
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
for amplitude, nx, increment, color, style in [
    (0.02, 24, 0.1, "#305f88", "-"),
    (0.10, 24, 0.1, "#a56035", "-"),
    (0.02, 40, 0.05, "#32917d", "--"),
]:
    result, curve = solve(imperfection=amplitude, nx=nx, increment=increment)
    label = f"a={amplitude}, nx={nx}, step={increment}"
    axes[0].plot(
        -curve["displacement"], -curve["reaction"], style, color=color, label=label
    )
    axes[1].plot(
        -curve["displacement"],
        curve["monitor_displacement"],
        style,
        color=color,
        label=label,
    )
    records.append(
        dict(
            amplitude=amplitude,
            nx=nx,
            maximum_increment=increment,
            ideal_critical_stress=result.quantity("ideal_critical_stress"),
            curve={k: v.tolist() for k, v in curve.items()},
        )
    )
for ax in axes:
    ax.set_xlabel("End shortening")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
axes[0].set_ylabel("Compressive reaction")
axes[1].set_ylabel("Additional transverse tip displacement")
axes[0].legend(fontsize=7)
fig.savefig("docs/assets/imperfect_column_paths.png", dpi=180)
Path("evidence/stability_periodic/imperfect_column.json").write_text(
    json.dumps(records, indent=2) + "\n"
)
print(
    [
        (
            r["amplitude"],
            r["nx"],
            r["curve"]["reaction"][-1],
            r["curve"]["monitor_displacement"][-1],
        )
        for r in records
    ]
)

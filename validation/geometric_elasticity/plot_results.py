"""Plot measured curves and actual displacement fields without warp exaggeration."""

import json
from pathlib import Path
import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

root = Path(__file__).resolve().parent
out = root / "results"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)
fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
colors = ["#245e83", "#be6b38"]
for a, color in zip((0.1, 0.2), colors):
    for nx, style in [(40, ":"), (80, "--"), (120, "-")]:
        c = np.genfromtxt(
            out / f"column_a{a}_n{nx}_dt0.05.csv", delimiter=",", names=True
        )
        axes[0, 0].plot(
            c["added_tip_displacement_mm"],
            c["load_N"] / 1000,
            style,
            color=color,
            label=f"a = {a} mm; {nx} × {nx // 20}",
        )
axes[0, 0].set(
    xlabel="Additional lateral tip displacement (mm)",
    ylabel="Axial load (kN)",
    title="a  Imperfect elastic column",
)
axes[0, 0].legend(fontsize=7, frameon=False)


def draw_shape(ax, file, color, label):
    with h5py.File(file, "r") as f:
        x = f["Frames/0000/Geometry"][:, :2]
        u = f["Frames/0000/Point/U"][:, :2]
        t = f["Mesh/Topology"][:, :4]
    ax.add_collection(
        PolyCollection(x[t], facecolor="none", edgecolor="#c4c9cd", linewidth=0.12)
    )
    ax.add_collection(
        PolyCollection(
            (x + u)[t],
            facecolor=color,
            alpha=0.45,
            edgecolor=color,
            linewidth=0.12,
            label=label,
        )
    )
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")


draw_shape(axes[0, 1], out / "column_a0.2_n120_dt0.025.h5", colors[1], "Loaded shape")
axes[0, 1].set(
    xlabel="x (mm)", ylabel="y (mm)", title="b  10.7 kN; a = 0.2 mm; deformation ×1"
)
rows = json.loads((out / "elastica.json").read_text())
ref = [r for r in rows if r["nx"] == 80]
axes[1, 0].plot(
    [0] + [r["alpha"] for r in ref],
    [0] + [r["reference"]["y"] for r in ref],
    "-",
    color="#333333",
    label="Independent elastica BVP",
)
axes[1, 0].plot(
    [r["alpha"] for r in ref],
    [r["y"] for r in ref],
    "o",
    mfc="white",
    mec=colors[0],
    label="StVK solid FEM",
)
axes[1, 0].set(
    xlabel=r"Load parameter $PL^2/(EI)$",
    ylabel=r"Tip deflection $u_y/L$",
    title="c  Large-rotation cantilever",
)
axes[1, 0].legend(frameon=False, fontsize=8)
draw_shape(axes[1, 1], out / "elastica_a4.0_n80.h5", colors[0], "Loaded shape")
axes[1, 1].set(
    xlabel="x (mm)", ylabel="y (mm)", title="d  Load parameter 4; deformation ×1"
)
for ax in (axes[0, 0], axes[1, 0]):
    ax.grid(alpha=0.15)
fig.savefig(root / "geometric_elasticity_validation.png", dpi=220)
fig.savefig(root / "geometric_elasticity_validation.svg")

# Matplotlib emits trailing blanks in SVG path data; keep review diffs clean.
svg = root / "geometric_elasticity_validation.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")

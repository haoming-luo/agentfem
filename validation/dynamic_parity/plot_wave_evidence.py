# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Plot all retained raw pulse-wave runs without filtering or normalization."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parent / "evidence" / "half_sine_waves"
records = json.loads((root / "convergence.json").read_text())
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for i, mode in enumerate(("longitudinal", "shear")):
    selected = sorted(
        (r for r in records if r["mode"] == mode and r["dt_fraction"] == 0.4),
        key=lambda r: r["nx"],
    )
    for r in selected:
        data = np.genfromtxt(
            root / f"{mode}_{r['nx']}_0.4.csv", delimiter=",", names=True
        )
        axes[i, 0].plot(
            data["time"] * 1e6, data["velocity"], label=f"nx={r['nx']}", lw=1.2
        )
    axes[i, 0].plot(
        data["time"] * 1e6,
        data["analytic_velocity"],
        color="black",
        ls="--",
        lw=1.5,
        label="Analytical",
    )
    axes[i, 0].set(
        xlabel="Time (microseconds)",
        ylabel="Velocity (m/s)",
        title=mode.capitalize() + " pulse: raw histories",
    )
    axes[i, 0].legend(frameon=False, fontsize=8)
    for fraction, style in ((0.8, "o-"), (0.4, "s--")):
        rows = sorted(
            (r for r in records if r["mode"] == mode and r["dt_fraction"] == fraction),
            key=lambda r: r["nx"],
        )
        axes[i, 1].loglog(
            [r["nx"] for r in rows],
            [r["relative_waveform_L2"] for r in rows],
            style,
            label=f"dt / safe dt = {fraction}",
        )
    axes[i, 1].set(
        xlabel="Longitudinal mesh subdivisions",
        ylabel="Relative waveform L2 error",
        title="Space/time refinement",
    )
    axes[i, 1].legend(frameon=False, fontsize=8)
fig.savefig(root / "raw_wave_convergence.png", dpi=220)
fig.savefig(root / "raw_wave_convergence.pdf")

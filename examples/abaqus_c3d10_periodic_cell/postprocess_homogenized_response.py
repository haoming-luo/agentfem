"""Plot the exact periodic-cell homogenized history written by AgentFEM.

XDMF remains the field-visualization product.  This script reads the compact
NPZ scientific history because its stresses were integrated on the original
quadrature forms and normalized by the complete RVE volume; re-averaging P0
centroid visualization fields from XDMF would be less accurate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def run(history=None, output=None):
    history = Path(history or HERE / "output" / "homogenized_history.npz")
    output = Path(output or HERE / "output" / "homogenized_response.png")
    return plot_response(history, output)


def plot_response(history_path, output_path):
    """Case-level plot; change freely without extending AgentFEM core."""

    import matplotlib.pyplot as plt

    history = np.load(history_path)
    F = history["deformation_gradient"]
    P = history["first_piola_stress"]
    sigma = history["cauchy_stress"]
    engineering_strain = F[:, 0, 0] - 1.0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(engineering_strain, P[:, 0, 0], "o-", label=r"$\bar P_{11}$")
    axes[0].plot(engineering_strain, sigma[:, 0, 0], "s--", label=r"$\bar\sigma_{11}$")
    axes[0].plot(engineering_strain, P[:, 1, 1], ".-", label=r"$\bar P_{22}$")
    axes[0].plot(engineering_strain, P[:, 2, 2], ".-", label=r"$\bar P_{33}$")
    axes[0].set(xlabel=r"macroscopic engineering strain $F_{11}-1$", ylabel="homogenized stress")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(engineering_strain, history["deformation_jacobian"], "o-", label=r"$\det(\bar F)$")
    axes[1].plot(engineering_strain, history["solid_current_fraction"], "s-", label="current solid fraction")
    energy = history["strain_energy_density"]
    if np.max(np.abs(energy)) > 0.0:
        axes[1].plot(engineering_strain, energy / np.max(np.abs(energy)), "^-", label="normalized energy")
    axes[1].set(xlabel=r"macroscopic engineering strain $F_{11}-1$", ylabel="volume / normalized energy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    print(run(arguments.history, arguments.output))

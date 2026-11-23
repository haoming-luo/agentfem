"""Aligned diagnostic panels for a frozen reference, without response smoothing."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.p92_bam_source import read_source


def plot(frozen, source_dir, output):
    if output.exists():
        raise ValueError("Choose a new diagnostic figure")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for col, protocol in enumerate(("RF", "SLR", "SLC")):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        prediction = frozen["protocols"][protocol]["prediction"]
        ax = axes[0, col]
        ids = np.asarray(source["cycle"]) == 3
        ax.plot(np.asarray(source["strain"])[ids]*100,
                np.asarray(source["stress_Pa"])[ids]/1e6, color="#333333", label="Measured")
        segments = [r for r in prediction["segments"] if r["cycle"] == 3]
        strain = np.concatenate([r["sampled_strain"] for r in segments])*100
        stress = np.concatenate([r["sampled_stress_MPa"] for r in segments])
        ax.plot(strain, stress, color="#0072B2", linestyle="--", label="Frozen prediction")
        ax.set(title=f"{protocol}: cycle 3", xlabel="Total strain (%)", ylabel="Stress (MPa)")
        ax.legend(fontsize=8)
        hold = next(h for h in prediction["hold_diagnostics"] if h["cycle"] == 1 and
                    (protocol == "RF" or h["duration_s"] > 1000))
        segment = prediction["segments"][hold["segment"]]
        ids = segment["source_indices"]
        times = np.asarray(hold["sample_times_s"])
        ax = axes[1, col]
        if protocol == "SLC":
            observed = np.asarray(source["strain"])[ids]*100
            predicted = np.asarray(segment["sampled_strain"])*100
            ylabel = "Total strain (%)"
        else:
            observed = np.asarray(source["stress_Pa"])[ids]/1e6
            predicted = segment["sampled_stress_MPa"]
            ylabel = "Stress (MPa)"
        ax.plot(times, observed, "o-", markersize=3, color="#333333", label="Measured samples")
        ax.plot(times, predicted, "s--", markersize=3, color="#0072B2", label="Predicted samples")
        ax.set(title="First-cycle hold", xlabel="Time from hold entry (s)", ylabel=ylabel)
        ax.set_xscale("symlog", linthresh=1.)
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("P92 development diagnostic — not engineering qualification", fontsize=13)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.frozen.read_bytes()), args.source_dir, args.output)

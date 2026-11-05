"""Compare existing published-equation cycles to sparse independent figure reads.

The experimental life normalizes the plot axis; it is an INPUT, not a predicted
life. No refitting and no digitized raw-data precision are implied.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .p92_dwell_figure_screen import axis_map
    from .p92_published_recovery_reference import SOURCE_SHA256
except ImportError:
    from p92_dwell_figure_screen import axis_map
    from p92_published_recovery_reference import SOURCE_SHA256

SOURCE = Path(__file__).resolve().parents[1] / "evidence/creep/p92-cyclic-figure-points.json"


def screen(campaign, observations=None):
    data = json.loads(SOURCE.read_text()) if observations is None else observations
    if data["source_pdf_sha256"] != SOURCE_SHA256:
        raise ValueError("Figure source differs from the equation reference.")
    bound = float(data["coordinate_reading_bound_px"])
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("Positive reading sensitivity required.")
    comparisons = []
    for panel in data["panels"]:
        runs = [r for r in campaign["runs"]
                if r["temperature_C"] == panel["temperature_C"]
                and r["strain_range"] == panel["strain_range"]
                and r["tensile_dwell_s"] == panel["dwell_s"]]
        if len(runs) != 1:
            raise ValueError("Require exactly one matching cyclic run per panel.")
        run = runs[0]
        if (run["source_pdf_sha256"] != SOURCE_SHA256
                or run["reported_life_used_to_select_run_length"] != panel["reported_life_cycles"]):
            raise ValueError("Run source or normalization differs from the figure.")
        cycles = np.array([r["cycle"] for r in run["cycles"]], dtype=float)
        peaks = np.array([r["peak_stress_MPa"] for r in run["cycles"]])
        if (len(cycles) < 2 or not np.isfinite(cycles).all()
                or not np.isfinite(peaks).all() or np.any(np.diff(cycles) <= 0)):
            raise ValueError("Require finite ordered cycle/peak records.")
        sx, ox, rx = axis_map(panel["x_ticks_pixel_normalized_cycle"])
        sy, oy, ry = axis_map(panel["y_ticks_pixel_stress_MPa"])
        if sx <= 0 or sy >= 0 or max(rx, ry) > bound:
            raise ValueError("Inconsistent figure axes.")
        pixels = np.asarray(panel["experimental_peak_centres_px"], dtype=float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
            raise ValueError("Require finite pixel pairs.")
        sample_cycles = (sx * pixels[:, 0] + ox) * panel["reported_life_cycles"] / 2
        delta_cycle = sx * bound * panel["reported_life_cycles"] / 2
        observed = sy * pixels[:, 1] + oy
        if (np.any(sample_cycles-delta_cycle < cycles[0])
                or np.any(sample_cycles+delta_cycle > cycles[-1])
                or np.any(observed <= 0)):
            raise ValueError("Run does not cover full reading intervals; no endpoint clamping.")
        predicted = np.interp(sample_cycles, cycles, peaks)
        sensitivity = []
        for center in sample_cycles:
            left, right = center-delta_cycle, center+delta_cycle
            knots = np.concatenate(([left, right], cycles[(cycles > left) & (cycles < right)]))
            values = np.interp(knots, cycles, peaks)
            sensitivity.append((values.min(), values.max()))
        intervals = np.asarray(sensitivity) - observed[:, None]
        intervals += np.array([-abs(sy)*bound, abs(sy)*bound])
        comparisons.append(dict(
            panel=panel["id"], count=len(observed), method=run["method"],
            mean_absolute_relative_error=float(np.mean(abs(predicted-observed)/observed)),
            maximum_absolute_error_MPa=float(np.max(abs(predicted-observed))),
            samples=[dict(cycle=float(n), observed_peak_MPa=float(o),
                          predicted_peak_MPa=float(p), reading_error_interval_MPa=interval.tolist())
                     for n, o, p, interval in zip(sample_cycles, observed, predicted, intervals, strict=True)],
        ))
    return dict(schema="agentfem.research.p92-cyclic-screen.v1", source=data["source"],
                engineering_qualified=False, reported_lives_are_inputs_not_predictions=True,
                scope="Sparse long-cycle response screen; no life fit or full experimental curve",
                comparisons=comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="Existing --published-campaign JSON evidence")
    args = parser.parse_args()
    print(json.dumps(screen(json.loads(args.campaign.read_text())), indent=2, allow_nan=False))

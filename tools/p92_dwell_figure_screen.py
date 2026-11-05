# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Sparse source-figure screen, not a fitted material or FEM validation.

Run as: python tools/p92_dwell_figure_screen.py --output <new-json-path>
The observation file preserves source pixels and uncertainty, not invented raw
experimental precision. NumPy/SciPy suffice; no PDF/image dependency at runtime.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

try:
    from .p92_published_recovery_reference import SOURCE_SHA256, integrate_segment, published_parameters
except ImportError:  # direct command-line execution from tools/
    from p92_published_recovery_reference import SOURCE_SHA256, integrate_segment, published_parameters


OBSERVATIONS = Path(__file__).resolve().parents[1] / "evidence/creep/p92-published-figure-points.json"


def axis_map(ticks):
    """Affine pixel-to-physical map; preserve the tick-fit residual in pixels."""
    ticks = np.asarray(ticks, dtype=float)
    if (ticks.ndim != 2 or ticks.shape[1] != 2 or len(ticks) < 2
            or not np.isfinite(ticks).all() or np.ptp(ticks[:, 0]) == 0
            or np.ptp(ticks[:, 1]) == 0):
        raise ValueError("Axis calibration requires distinct finite pixel and physical coordinates.")
    slope, intercept = np.polyfit(ticks[:, 0], ticks[:, 1], 1)
    residual = np.max(np.abs((slope*ticks[:, 0]+intercept-ticks[:, 1])/slope))
    return float(slope), float(intercept), float(residual)


def load_observations(path=OBSERVATIONS):
    data = json.loads(Path(path).read_text())
    if (data.get("schema") != "agentfem.p92-figure-observations.v1"
            or data.get("source_pdf_sha256") != SOURCE_SHA256):
        raise ValueError("Observation source differs from the pinned published-equation reference.")
    bound = data["coordinate_reading_bound_px"]
    if isinstance(bound, bool) or not np.isfinite(bound) or bound <= 0:
        raise ValueError("A positive pixel-reading sensitivity bound is required.")
    ids = [panel["id"] for panel in data["panels"]]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Observation panels must have distinct identities.")
    for panel in data["panels"]:
        published_parameters(panel["temperature_C"])
        if panel["cycle"] != 1 or panel["strain_range"] != .008 or panel["dwell_s"] not in (60., 600.):
            raise ValueError("This screen admits only declared first-cycle held-out dwell conditions.")
        sx, ox, rx = axis_map(panel["x_ticks_pixel_time_s"])
        sy, oy, ry = axis_map(panel["y_ticks_pixel_stress_MPa"])
        if sx <= 0 or sy >= 0 or max(rx, ry) > bound:
            raise ValueError("Invalid axis orientation or inconsistent tick calibration.")
        for key in ("experimental_square_centres_px", "model_dash_centres_px"):
            points = np.asarray(panel[key], dtype=float).reshape((-1, 2))
            if (not np.isfinite(points).all() or np.any(points < 0)
                    or np.any(points >= np.asarray(panel["image_size"]))):
                raise ValueError("Observation coordinates lie outside the source image.")
            times = sx*points[:, 0]+ox
            if np.any(times < 0) or np.any(times > panel["dwell_s"]) or np.any(np.diff(times) <= 0):
                raise ValueError("Selected samples must be ordered inside the declared dwell; do not clamp them.")
    return data


def compare_samples(panel, pixels, hold, bound):
    sx, ox, rx = axis_map(panel["x_ticks_pixel_time_s"])
    sy, oy, ry = axis_map(panel["y_ticks_pixel_stress_MPa"])
    if not pixels:
        return {"count": 0, "samples": []}
    pixels = np.asarray(pixels, dtype=float)
    times = sx*pixels[:, 0]+ox
    observed = sy*pixels[:, 1]+oy
    predicted = hold.sol(times)[0]
    dt, ds = abs(sx)*bound, abs(sy)*bound
    # These first-cycle tensile holds are monotone. Evaluate the two ends of
    # the declared time-reading interval rather than assuming a flat curve.
    early = hold.sol(np.maximum(0., times-dt))[0]
    late = hold.sol(np.minimum(panel["dwell_s"], times+dt))[0]
    lower = np.minimum(early, late) - (observed+ds)
    upper = np.maximum(early, late) - (observed-ds)
    errors = predicted-observed
    return {
        "count": len(times), "tick_fit_max_residual_px": {"x": rx, "y": ry},
        "reading_sensitivity": {"time_s": dt, "stress_MPa": ds},
        "mean_signed_error_MPa": float(np.mean(errors)),
        "maximum_absolute_error_MPa": float(np.max(np.abs(errors))),
        "selected_marker_mean_absolute_relative_error": float(np.mean(np.abs(errors)/np.abs(observed))),
        "all_error_intervals_include_zero": bool(np.all((lower <= 0) & (upper >= 0))),
        "samples": [{"time_s": float(t), "observed_MPa": float(o), "predicted_MPa": float(p),
                     "signed_error_MPa": float(e), "reading_error_interval_MPa": [float(lo), float(hi)]}
                    for t, o, p, e, lo, hi in zip(times, observed, predicted, errors, lower, upper, strict=True)],
    }


def run_screen(path=OBSERVATIONS, *, method="Radau"):
    data = load_observations(path)
    rows = []
    for panel in data["panels"]:
        parameters = published_parameters(panel["temperature_C"])
        ramp = integrate_segment(np.zeros(5), panel["strain_range"]/(2*.001), .001,
                                 parameters, method=method)
        hold = integrate_segment(ramp.y[:, -1], panel["dwell_s"], 0., parameters, method=method)
        if np.any(np.diff(hold.sol(np.linspace(0., panel["dwell_s"], 1001))[0]) > 1.e-8):
            raise RuntimeError("Nonmonotone hold requires interior uncertainty extrema, not endpoint bounds.")
        rows.append({
            "panel": panel["id"], "temperature_C": panel["temperature_C"],
            "dwell_s": panel["dwell_s"], "cycle": 1,
            "experimental_markers": compare_samples(panel, panel["experimental_square_centres_px"],
                                                     hold, data["coordinate_reading_bound_px"]),
            "published_model_dashes": compare_samples(panel, panel["model_dash_centres_px"],
                                                       hold, data["coordinate_reading_bound_px"]),
        })
    return {
        "scope": "sparse first-cycle published-figure screen; no engineering material admission",
        "source_doi": data["source_doi"], "source_pdf_sha256": SOURCE_SHA256,
        "observation_sha256": sha256(Path(path).read_bytes()).hexdigest(),
        "method": method, "relative_tolerance": 1.e-8,
        "parameters": "published Table 4, unchanged; no refit or temperature interpolation",
        "startup": "zero state; ramp from zero strain to positive half-range at 0.001/s",
        "experimental_uncertainty_known": False, "material_admitted": False,
        "metric_is_article_one_second_grid_MRE": False,
        "reading_bound_px": data["coordinate_reading_bound_px"], "panels": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("Radau", "BDF"), default="Radau")
    parser.add_argument("--output", type=Path, help="New JSON evidence file; never overwrite.")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output exists; choose a new evidence path.")
    report = run_screen(method=args.method)
    serialized = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output is None:
        print(serialized, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        for row in report["panels"]:
            metric = row["experimental_markers"]["selected_marker_mean_absolute_relative_error"]
            print(f"{row['panel']}: selected experimental-marker mean absolute relative error {metric:.2%}")
        print(f"Evidence: {args.output}; this does not admit a validated material.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

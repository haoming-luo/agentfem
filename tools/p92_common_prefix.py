"""Compare recorded first holds before the service-like protocols diverge.

This is an observational diagnostic, not a population scatter estimate or a
correction to validation residuals. No material state is fitted or reset.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new


def aligned_difference(left_time, left, right_time, right):
    if any(np.iscomplexobj(a) for a in (left_time, left, right_time, right)):
        raise ValueError("Require real records")
    arrays = [np.asarray(a, dtype=float) for a in (left_time, left, right_time, right)]
    t, a, u, b = arrays
    if (any(x.ndim != 1 or not np.isfinite(x).all() for x in arrays)
            or len(t) < 2 or len(u) < 2 or a.shape != t.shape or b.shape != u.shape
            or np.any(np.diff(t) <= 0) or np.any(np.diff(u) <= 0)):
        raise ValueError("Require finite ordered scalar records")
    lo, hi = max(t[0], u[0]), min(t[-1], u[-1])
    if hi <= lo:
        raise ValueError("No shared time interval")
    grid = np.unique(np.r_[lo, t[(t > lo) & (t < hi)], u[(u > lo) & (u < hi)], hi])
    av, bv = np.interp(grid, t, a), np.interp(grid, u, b)
    delta = bv-av
    # Exact integration of the piecewise-linear difference and its square.
    dt = np.diff(grid)
    mean = np.sum(dt*(delta[:-1]+delta[1:])/2)/(hi-lo)
    rms = np.sqrt(np.sum(dt*(delta[:-1]**2+delta[:-1]*delta[1:]+delta[1:]**2)/3)/(hi-lo))
    return dict(interval_s=[float(lo), float(hi)], knots=len(grid),
        right_minus_left_first=float(delta[0]), right_minus_left_last=float(delta[-1]),
        right_minus_left_time_mean=float(mean), difference_time_rms=float(rms),
        difference_max_abs=float(abs(delta).max()),
        left_change=float(av[-1]-av[0]), right_change=float(bv[-1]-bv[0]),
        interpolation="piecewise linear on shared support; no extrapolation")


def evaluate(directory, frozen=None):
    records = {}
    for protocol in ("SLR", "SLC"):
        source = read_source(directory/f"Data_{protocol}.lis", protocol=protocol)
        path = measured_path(source, cycles=1)
        first, end = path["switches"][:2]
        start, stop = first["time_s"], end["time_s"]
        ids = np.flatnonzero((source["time_s"] > start) & (source["time_s"] < stop))
        segment = path["segments"][1]
        if not segment["strain_hold"] or segment["target"] != -.002:
            raise ValueError("Unexpected shared first-hold protocol")
        records[protocol] = dict(source=source, path=path, ids=ids, time=source["time_s"][ids]-start,
            switch=first, duration=stop-start)
    left, right = records["SLR"], records["SLC"]
    fields = {}
    for name, key, scale in (("stress_MPa", "stress_Pa", 1e-6),
                             ("strain", "strain", 1.), ("temperature_K", "temperature_K", 1.)):
        fields[name] = aligned_difference(left["time"], left["source"][key][left["ids"]]*scale,
            right["time"], right["source"][key][right["ids"]]*scale)
    prediction = None
    if frozen is not None:
        from tools.p92_recoverable_frozen import decode_material
        from tools.p92_two_rate_controls import replay
        from tools.p92_waveform_controls import attach_waveforms

        kernel, material = decode_material(frozen["parameters"], frozen["model_id"])
        prediction = {}
        for mode in ("linear", "recorded_ramps"):
            runs = {}
            for protocol, record in records.items():
                if record["source"]["sha256"] != frozen["source_sha256"][protocol]:
                    raise ValueError("Frozen source identity changed")
                path = record["path"]
                if mode == "recorded_ramps":
                    path = attach_waveforms(path, record["source"])
                path = dict(path, segments=path["segments"][:3])
                _, result = replay(path, material, kernel=kernel)
                _, independent = replay(path, material, kernel=kernel, method="BDF", rtol=1e-10)
                samples = np.asarray(result["segments"][1]["sampled_stress_MPa"])
                alternate = np.asarray(independent["segments"][1]["sampled_stress_MPa"])
                runs[protocol] = dict(times=path["segments"][1]["sample_times"].tolist(),
                    samples=samples.tolist(),
                    independent_stress_max_difference_MPa=float(abs(samples-alternate).max()),
                    hold=result["hold_diagnostics"][0],
                    final_reload_stress_MPa=result["segments"][2]["end_state"][0])
            prediction[mode] = dict(protocols=runs, stress_difference=aligned_difference(
                runs["SLR"]["times"], runs["SLR"]["samples"],
                runs["SLC"]["times"], runs["SLC"]["samples"]))
    return dict(schema="agentfem.research.common-prefix.v1", fields=fields, prediction=prediction,
        sources={p: dict(sha256=r["source"]["sha256"], first_hold_switch=r["switch"],
                        duration_s=r["duration"], samples=len(r["ids"])) for p, r in records.items()},
        fitted=False, engineering_qualified=False, life_qualified=False,
        scope="Two specimens before protocol divergence; not a scatter distribution, bias correction, or changed gate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    names = ("p92_common_prefix.py", "p92_bam_source.py", "p92_measured_controls.py",
             "p92_recoverable_frozen.py", "p92_two_rate_controls.py", "p92_waveform_controls.py",
             "p92_recoverable_spectrum.py", "p92_two_rate_reference.py")
    before = {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    raw = None if args.frozen is None else args.frozen.read_bytes()
    result = evaluate(args.source_dir, None if raw is None else json.loads(raw))
    if (any(sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() != h for n, h in before.items())
            or (raw is not None and raw != args.frozen.read_bytes())):
        raise RuntimeError("Common-prefix inputs changed")
    result.update(driver_sha256=before, frozen_sha256=None if raw is None else sha256(raw).hexdigest())
    write_new(args.output, result)
    print(json.dumps(result["fields"], indent=2))

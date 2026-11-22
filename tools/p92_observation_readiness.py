"""Check the existing secant observation gate before any material solve.

Resolution is a recording property, not a noise model or accuracy waiver.
This diagnostic does not widen windows or alter the response policy.
"""

import argparse
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_response_screen import POLICY


def secant_support(times, values, resolution):
    times, values = np.asarray(times), np.asarray(values)
    quantum = np.asarray(resolution)
    if (times.ndim != 1 or values.shape != times.shape
            or quantum.ndim != 0
            or any(a.dtype.kind not in "fiu" for a in (times, values, quantum))
            or not np.isfinite(times).all() or not np.isfinite(values).all()
            or np.any(np.diff(times) <= 0)
            or not np.isfinite(quantum) or quantum <= 0):
        raise ValueError("Require aligned ordered real observations and positive resolution")
    resolution = float(quantum)
    if len(times) < 2:
        return dict(status="unresolved", reason="insufficient_samples", samples=len(times))
    duration = float(times[-1]-times[0])
    change = float(values[-1]-values[0])
    units = abs(change)/resolution
    return dict(status="resolved" if units >= POLICY["minimum_resolution_units"] else "unresolved",
        reason="recorded_signal" if units >= POLICY["minimum_resolution_units"] else "recorded_resolution",
        samples=len(times), interval_s=[float(times[0]), float(times[-1])],
        observed_change=change, observed_secant_per_s=change/duration,
        resolution_units=units, resolution=resolution,
        required_signal=POLICY["minimum_resolution_units"]*resolution)


def audit(source, *, cycles):
    path = measured_path(source, cycles=cycles)
    temperatures = np.asarray(source["temperature_K"])
    if (temperatures.shape != np.asarray(source["time_s"]).shape
            or temperatures.dtype.kind not in "fiu" or not np.isfinite(temperatures).all()
            or np.any(temperatures <= 0)):
        raise ValueError("Require aligned finite positive absolute temperatures")
    records = []
    for i, segment in enumerate(path["segments"]):
        if not (segment["stress_control"] or segment["strain_hold"]):
            continue
        channel = "strain" if segment["stress_control"] else "stress"
        resolution = POLICY["strain_resolution" if channel == "strain" else "stress_resolution_MPa"]
        times = np.asarray(segment["sample_times"])
        early = times <= min(10., .1*segment["duration"])
        windows = dict(early=(times[early], np.asarray(segment["measured_"+channel])[early],
                              np.asarray(segment["source_indices"])[early]),
                       late=(segment["tail"]["times"], segment["tail"][channel],
                             segment["tail"]["source_indices"]))
        for window, (t, values, indices) in windows.items():
            thermal = None
            if len(indices) >= 2:
                a, b = int(indices[0]), int(indices[-1])
                all_samples = temperatures[a:b+1]
                thermal = dict(start_K=float(all_samples[0]), end_K=float(all_samples[-1]),
                    minimum_K=float(all_samples.min()), maximum_K=float(all_samples.max()),
                    change_K=float(all_samples[-1]-all_samples[0]),
                    source_indices=[a, b], samples=len(all_samples),
                    scope="All recorded temperatures between window endpoints; no thermal correction")
            records.append(dict(cycle=segment["cycle"], segment=i, window=window,
                channel="strain" if channel == "strain" else "stress_MPa",
                temperature_context=thermal,
                **secant_support(t, values, resolution)))
    selected_temperature = temperatures[np.asarray(source["cycle"]) <= cycles]
    return dict(schema="agentfem.research.observation-readiness.v1", protocol=source["protocol"],
        source_sha256=source["sha256"], cycles=cycles, response_policy=dict(POLICY),
        recorded_temperature_range_K=[float(selected_temperature.min()), float(selected_temperature.max())],
        records=records, all_secants_resolved=all(r["status"] == "resolved" for r in records),
        material_solved=False, fitted=False, engineering_qualified=False, life_qualified=False,
        scope="Existing early/late observation windows only; not experimental uncertainty or response acceptance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()
    result = {p: audit(read_source(args.source_dir/f"Data_{p}.lis", protocol=p), cycles=args.cycles)
              for p in ("RF", "SLR", "SLC")}
    write_new(args.output, result)

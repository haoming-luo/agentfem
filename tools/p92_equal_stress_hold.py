"""Compare positive strain holds over identical stress intervals, without fitting.

Recorded strain drift is retained in the observed nonelastic increment. The
resolution envelope is a deterministic one-quantum perturbation, not a
confidence interval or an experimental uncertainty estimate.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material
from tools import p92_recoverable_spectrum as spectrum


def observed_interval(times, strain, stress, modulus):
    t, e, s = (np.asarray(v, dtype=float) for v in (times, strain, stress))
    if (t.ndim != 1 or len(t) < 3 or e.shape != t.shape or s.shape != t.shape
            or not all(np.isfinite(v).all() for v in (t, e, s))
            or np.any(np.diff(t) <= 0) or not np.isfinite(modulus) or modulus <= 0
            or s[0] <= s[-1]):
        raise ValueError("Require finite aligned relaxing observations")
    dt = t[-1]-t[0]
    rate = ((e[-1]-e[0])-(s[-1]-s[0])/modulus)/dt
    return dict(duration_s=float(dt), stress_endpoints_MPa=[float(s[0]), float(s[-1])],
        average_nonelastic_rate_per_s=float(rate),
        one_quantum_endpoint_sensitivity_per_s=float((2e-5+2*.01/modulus)/dt),
        samples=len(t))


def predicted_interval(solution, stress_endpoints, modulus):
    roots = []
    for level in stress_endpoints:
        values = solution.y[0]-level
        intervals = np.flatnonzero((values[:-1] > 0) & (values[1:] <= 0))
        if len(intervals) != 1:
            raise ValueError("Require a unique descending predicted stress crossing")
        i = intervals[0]
        roots.append(brentq(lambda t, level=level: solution.sol(t)[0]-level,
                            solution.t[i], solution.t[i+1], xtol=1e-11))
    if roots[1] <= roots[0]:
        raise ValueError("Unordered predicted stress crossings")
    y = solution.sol(roots)
    duration = roots[1]-roots[0]
    return dict(times_s=roots, duration_s=duration,
        average_nonelastic_rate_per_s=float((y[7, 1]-y[7, 0]-(y[0, 1]-y[0, 0])/modulus)/duration))


def audit(frozen, source_dir):
    module, mat = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not spectrum:
        raise ValueError("Require the frozen spectrum reference")
    records = []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        item = frozen["protocols"][protocol]
        if source["sha256"] != frozen["source_sha256"][protocol] or not item["numerical"]["passed"]:
            raise ValueError("Require verified source and numerical trajectory")
        path = measured_path(source, cycles=3)
        t, e, s = (np.asarray(source[k]) for k in ("time_s", "strain", "stress_Pa"))
        s = s/1e6
        for i, seg in enumerate(path["segments"]):
            if not seg["strain_hold"] or seg["target"] <= 0:
                continue
            old = np.asarray(item["prediction"]["segments"][i-1]["end_state"])
            solution = module.integrate(old, seg["duration"], 0., mat)
            expected = np.asarray(item["prediction"]["segments"][i]["end_state"])
            if not np.allclose(solution.y[:, -1], expected, rtol=1e-9, atol=1e-9):
                raise ValueError("Hold replay differs from frozen state")
            lo, hi = path["switches"][i-1]["bracket_s"][1], path["switches"][i]["bracket_s"][0]
            for center in (190., 210., 230.):
                ids = np.flatnonzero((t > lo) & (t < hi) & (abs(s-center) < 5.))
                row = dict(protocol=protocol, cycle=seg["cycle"], segment=i, band_center_MPa=center,
                           source_indices=ids.tolist())
                if len(ids) < 3 or np.any(np.diff(ids) != 1):
                    records.append(dict(**row, status="unresolved_sampling"))
                    continue
                observed = observed_interval(t[ids], e[ids], s[ids], mat.backbone.modulus)
                try:
                    predicted = predicted_interval(solution, observed["stress_endpoints_MPa"], mat.backbone.modulus)
                    row.update(status="compared", observed=observed, predicted=predicted)
                except ValueError as exc:
                    row.update(status="unresolved_prediction", observed=observed, reason=str(exc))
                records.append(row)
    return dict(schema="agentfem.research.equal-stress-hold.v1", records=records,
        source_sha256=frozen["source_sha256"], modulus_MPa=mat.backbone.modulus,
        scope="Early positive holds at aligned stress endpoints; inferred nonelasticity, not measured mechanism shares",
        fitted=False, engineering_qualified=False, acceptance_thresholds_changed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    result = audit(json.loads(raw), args.source_dir)
    result["frozen_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)

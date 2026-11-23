"""Conditional constant-stress Kelvin capacity bound; no fitting or new law."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools import p92_recoverable_spectrum as spectrum
from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material


def capacity(amplitudes, start, end):
    """Supremum of signed late secant over independent positive time constants.

    With fixed A_i=sigma/H_i-a_i(0), each mode contributes
    A_i*(exp(-start/tau_i)-exp(-end/tau_i))/(end-start).
    Negative terms have supremum zero, approached at zero/infinite tau.
    This is not a bound when the preceding path, moduli or amplitudes change.
    """
    a = np.asarray(amplitudes, dtype=float)
    if (a.ndim != 1 or not a.size or not np.isfinite(a).all()
            or not np.isfinite([start, end]).all() or not 0 < start < end):
        raise ValueError("Require finite amplitudes and positive ordered late times")
    duration = end-start
    tau = duration/np.log1p(duration/start)
    kernel = np.exp(-start/tau)*(-np.expm1(-duration/tau))/duration
    return dict(amplitudes=a.tolist(), time_constant_at_positive_maximum_s=float(tau),
        maximum_per_unit_amplitude_per_s=float(kernel),
        maximum_average_rate_per_s=float(np.maximum(a, 0.).sum()*kernel),
        finite_maximum_attained=bool(np.all(a >= 0.)),
        scope="Fixed hold-entry state and moduli; not a complete-path or material-family bound")


def audit(frozen, source_dir):
    module, mat = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not spectrum or mat.recoverable_powers != (1., 1.):
        raise ValueError("Require linear Kelvin spectrum identity")
    item = frozen["protocols"]["SLC"]
    if not item["numerical"]["passed"] or "prediction" not in item:
        raise ValueError("Require verified full frozen trajectory")
    source = read_source(source_dir/"Data_SLC.lis", protocol="SLC")
    if source["sha256"] != frozen["source_sha256"]["SLC"]:
        raise ValueError("Source identity changed")
    path = measured_path(source, cycles=3)
    records = []
    for i, segment in enumerate(path["segments"]):
        if not segment["stress_control"]:
            continue
        old = np.asarray(item["prediction"]["segments"][i-1]["end_state"])
        if old.shape != (12,) or not np.isfinite(old).all():
            raise ValueError("Invalid spectrum hold-entry state")
        tail = segment["tail"]
        lo, hi = tail["times"][0], tail["times"][-1]
        amplitudes = old[0]/np.asarray(mat.moduli)-old[10:12]
        bound = capacity(amplitudes, lo, hi)
        actual = np.sum(amplitudes*np.exp(-lo/np.asarray(mat.times)) *
                        (-np.expm1(-(hi-lo)/np.asarray(mat.times))))/(hi-lo)
        observed = (tail["strain"][-1]-tail["strain"][0])/(hi-lo)
        elastic = (tail["stress"][-1]-tail["stress"][0])/mat.backbone.modulus/(hi-lo)
        sensitivity = (2e-5+2*.01/mat.backbone.modulus)/(hi-lo)
        records.append(dict(cycle=segment["cycle"], segment=i, start_state=old.tolist(),
            tail_times_s=[lo, hi], source_indices=tail["source_indices"],
            conditional_bound=bound, original_recoverable_rate_per_s=float(actual),
            observed_nonelastic_rate_per_s=float(observed-elastic),
            one_quantum_endpoint_sensitivity_per_s=float(sensitivity),
            observed_strain_excursion=float(tail["strain"][-1]-tail["strain"][0]),
            resolved_at_five_quanta=bool(abs(tail["strain"][-1]-tail["strain"][0]) >= 5e-5)))
    return dict(schema="agentfem.research.recovery-capacity.v1", records=records,
        parameters=frozen["parameters"], source_sha256=frozen["source_sha256"],
        fitted=False, engineering_qualified=False, acceptance_thresholds_changed=False,
        scope="Conditional analytic capacity, not an observed mechanism partition; resolution is not statistical uncertainty")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    result = audit(json.loads(raw), args.source_dir)
    result["frozen_sha256"] = sha256(raw).hexdigest()
    result["driver_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    write_new(args.output, result)

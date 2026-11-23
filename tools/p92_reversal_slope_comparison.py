"""Compare local reversal slopes without equating them to elastic modulus."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools import p92_recoverable_reference as kernel
from tools.p92_recoverable_frozen import decode_material
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms, observed_reversal_slopes


def compare(source, prediction, observations):
    if source["sha256"] != prediction["source_sha256"] or source["sha256"] != observations["source_sha256"]:
        raise ValueError("Slope comparison requires matching source identity")
    records = []
    for row in observations["records"]:
        seg = prediction["segments"][row["segment"]]
        ids = seg["source_indices"]
        if (len(ids) != len(set(ids)) or len(ids) != len(seg["sampled_strain"])
                or len(ids) != len(seg["sampled_stress_MPa"])):
            raise ValueError("Duplicate or misaligned predicted source indices")
        positions = {v: i for i, v in enumerate(ids)}
        if any(i not in positions for i in row["source_indices"]):
            raise ValueError("Missing predicted slope observations; do not interpolate")
        chosen = [positions[i] for i in row["source_indices"]]
        e = np.asarray(seg["sampled_strain"])[chosen]
        s = np.asarray(seg["sampled_stress_MPa"])[chosen]
        if not np.isfinite([e, s]).all() or len(e) < 3 or np.ptp(e) <= 0:
            raise ValueError("Unresolved predicted slope")
        slope, intercept = np.polyfit(e, s, 1)
        observed_e = np.asarray(source["strain"])[row["source_indices"]]
        fixed_input_slope = np.polyfit(observed_e, s, 1)[0]
        records.append(dict(**row, predicted_slope_MPa=float(slope),
            predicted_on_recorded_strain_slope_MPa=float(fixed_input_slope),
            predicted_stress_fit_rms_MPa=float(np.sqrt(np.mean((s-slope*e-intercept)**2))),
            predicted_strain_span=float(np.ptp(e)), recorded_strain_span=float(np.ptp(observed_e))))
    return dict(source_sha256=source["sha256"], records=records,
        scope="Aligned local hysteresis slopes, not an elastic modulus calibration or acceptance tolerance")


def audit(frozen, source_dir):
    card = frozen["parameters"]
    module, mat = decode_material(card, frozen.get("model_id", kernel.MODEL_ID))
    records = {}
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = attach_waveforms(measured_path(source, cycles=3), source)
        observed = observed_reversal_slopes(path, source)
        expanded = deepcopy(path)
        for row in observed["records"]:
            i = row["segment"]
            seg = expanded["segments"][i]
            ids = sorted(set(seg["source_indices"]) | set(row["source_indices"]))
            start = path["switches"][i-1]["time_s"]
            seg.update(source_indices=ids, sample_times=np.asarray(source["time_s"])[ids]-start,
                measured_strain=np.asarray(source["strain"])[ids],
                measured_stress=np.asarray(source["stress_Pa"])[ids]/1e6)
        # Only observations are expanded; the control path and state history
        # remain fixed. The resulting residual vector is NOT used for fitting.
        _, prediction = replay(expanded, mat, kernel=module)
        records[protocol] = compare(source, prediction, observed)
    return dict(schema="agentfem.research.reversal-slope-comparison.v1", protocols=records,
        parameters=card, refitted=False, engineering_qualified=False)


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

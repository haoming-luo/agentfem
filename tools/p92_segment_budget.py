"""Localize frozen response error at aligned observations, without fitting.

The inferred nonelastic strain includes every departure from Hooke elasticity;
it does not identify plastic, creep or recoverable strain separately.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_fit import write_new
from tools.p92_measured_controls import measured_path


def segment_budget(source, prediction, modulus, *, path=None):
    if not np.isfinite(modulus) or modulus <= 0:
        raise ValueError("Require positive finite modulus in MPa")
    if source["sha256"] != prediction["source_sha256"]:
        raise ValueError("Source identity mismatch")
    records = []
    for segment in prediction["segments"]:
        ids = np.asarray(segment["source_indices"])
        if (ids.ndim != 1 or len(ids) < 2 or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(ids >= len(source["time_s"]))
                or np.any(np.diff(ids) <= 0)):
            raise ValueError("Require ordered distinct observation identities")
        time = np.asarray(source["time_s"])[ids]
        if np.any(np.diff(time) <= 0):
            raise ValueError("Require increasing observation times")
        if np.any(np.asarray(source["cycle"])[ids] != segment["cycle"]):
            raise ValueError("Observation cycle mismatch")
        observed_s = np.asarray(source["stress_Pa"])[ids]/1e6
        observed_e = np.asarray(source["strain"])[ids]
        model_s = np.asarray(segment["sampled_stress_MPa"])
        model_e = np.asarray(segment["sampled_strain"])
        if (any(v.shape != ids.shape for v in (model_s, model_e))
                or not all(np.isfinite(v).all() for v in
                           (time, observed_s, observed_e, model_s, model_e))):
            raise ValueError("Require finite aligned observations and predictions")
        excluded = []
        if path is not None:
            if path["source_sha256"] != source["sha256"]:
                raise ValueError("Control path source identity mismatch")
            i = segment["segment"]
            left = source["time_s"][0] if i == 0 else path["switches"][i-1]["bracket_s"][1]
            right = path["switches"][i]["bracket_s"][0]
            safe = (time > left) & (time <= right)
            excluded = ids[~safe].tolist()
            if np.count_nonzero(safe) < 2:
                raise ValueError("Insufficient observations outside switch ambiguity")
            ids, time, observed_s, observed_e, model_s, model_e = (
                v[safe] for v in (ids, time, observed_s, observed_e, model_s, model_e))
        stress_bias = model_s-observed_s
        strain_bias = model_e-observed_e
        # Difference of increments over the SAME recorded interval, no switch
        # extrapolation, interpolation, hidden-state reset or endpoint substitution.
        nonelastic_error = np.diff(strain_bias)-np.diff(stress_bias)/modulus
        input_term = modulus*np.diff(strain_bias)
        constitutive_term = -modulus*nonelastic_error
        records.append(dict(segment=segment["segment"], cycle=segment["cycle"],
            stress_control=segment["stress_control"],
            source_indices=ids.tolist(), observation_times_s=time.tolist(),
            excluded_switch_indices=excluded,
            stress_bias_MPa=stress_bias.tolist(),
            increment_nonelastic_error=nonelastic_error.tolist(),
            increment_input_term_MPa=input_term.tolist(),
            increment_nonelastic_term_MPa=constitutive_term.tolist(),
            interval_stress_bias_change_MPa=float(stress_bias[-1]-stress_bias[0]),
            interval_input_term_MPa=float(input_term.sum()),
            interval_nonelastic_term_MPa=float(constitutive_term.sum()),
            algebraic_closure_MPa=float(np.max(abs(
                np.diff(stress_bias)-input_term-constitutive_term)))))
    return dict(source_sha256=source["sha256"], modulus_MPa=modulus, records=records,
        scope="Aligned recorded intervals, not full control segments or causal mechanism attribution",
        engineering_qualified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = args.frozen.read_bytes()
    frozen = json.loads(payload)
    card = frozen["parameters"]
    modulus = card.get("backbone", card)["modulus"]
    result = dict(schema="agentfem.research.segment-error-budget.v1",
                  frozen_sha256=sha256(payload).hexdigest(), protocols={})
    for protocol, entry in frozen["protocols"].items():
        source = read_source(args.source_dir/f"Data_{protocol}.lis", protocol=protocol)
        result["protocols"][protocol] = segment_budget(
            source, entry["prediction"], modulus,
            path=measured_path(source, cycles=len(entry["prediction"]["cycles"])))
    write_new(args.output, result)

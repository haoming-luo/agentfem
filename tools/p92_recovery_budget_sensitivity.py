"""Finite, unfitted sensitivity of a conditional observed-path decomposition."""

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_fit import write_new
from tools.p92_observed_recovery_budget import evaluate as original_budget, linear_memory, window_budget
from tools.p92_recoverable_frozen import decode_material


def cards(material):
    """A declared lattice inside the existing log10 radius, not a fit/search."""
    for modulus in (151000., material.backbone.modulus, 162000.):
        for z in product((-.15, 0., .15), repeat=4):
            full = np.r_[np.log10(modulus/material.backbone.modulus), z]
            if np.linalg.norm(full) > .3+1e-14:
                continue
            yield dict(modulus_MPa=modulus, log10_coordinates=full.tolist(),
                       moduli=(np.asarray(material.moduli)*10.**np.array(z[:2])).tolist(),
                       times=(np.asarray(material.times)*10.**np.array(z[2:])).tolist())


def contrast(a, b):
    """SLR minus SLC; endpoint-resolution boxes are not sampling distributions."""
    left = np.array(a["remainder_interval"])/a["duration_s"]
    right = np.array(b["remainder_interval"])/b["duration_s"]
    return [float(left[0]-right[1]), float(left[1]-right[0])]


def evaluate(frozen, directory):
    _, material = decode_material(frozen["parameters"], frozen["model_id"])
    original = original_budget(frozen, directory)
    inputs = {}
    for protocol in ("SLR", "SLC"):
        source = read_source(directory/f"Data_{protocol}.lis", protocol=protocol)
        bands = [r for r in original["records"][protocol]["equal_stress_bands"]
                 if r["cycle"] == 1 and r["band_center_MPa"] == 210.]
        if len(bands) != 1 or bands[0]["status"] != "resolved":
            raise ValueError("Require the original resolved first-hold equal-stress band")
        selected = source["cycle"] <= 3
        inputs[protocol] = (source["time_s"][selected], source["stress_Pa"][selected]/1e6,
                            source["strain"][selected], bands[0]["source_indices"])
    records = []
    for card in cards(material):
        rows = {}
        k, tau = np.array(card["moduli"]), np.array(card["times"])
        for protocol, (t, s, e, ids) in inputs.items():
            memory = linear_memory(t, s, k, tau, np.zeros(2))
            rows[protocol] = window_budget(t, s, e, memory, *ids,
                modulus=card["modulus_MPa"], moduli=k, times=tau, initial_bounds=max(abs(s))/k)
        interval = contrast(rows["SLR"], rows["SLC"])
        records.append(dict(**card, protocols=rows, contrast_interval_per_s=interval,
                            strictly_positive=interval[0] > 0))
    return dict(schema="agentfem.research.observed-recovery-sensitivity.v1", records=records,
        source_sha256=frozen["source_sha256"], model_id=frozen["model_id"],
        maximum_log10_coordinate_norm=.3, fitted=False, continuous_prediction=False,
        engineering_qualified=False, life_qualified=False,
        all_sampled_contrasts_positive=all(r["strictly_positive"] for r in records),
        minimum_sampled_lower_contrast=min(r["contrast_interval_per_s"][0] for r in records),
        scope="Finite lattice sensitivity, not a certified full-neighborhood bound or confidence interval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_recovery_budget_sensitivity.py", "p92_observed_recovery_budget.py",
             "p92_bam_source.py", "p92_recoverable_frozen.py", "p92_measured_controls.py",
             "p92_response_screen.py")
    before = {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    result = evaluate(json.loads(raw), args.source_dir)
    if raw != args.frozen.read_bytes() or any(
            sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() != h for n, h in before.items()):
        raise RuntimeError("Sensitivity input changed during evaluation")
    result.update(driver_sha256=before, frozen_sha256=sha256(raw).hexdigest())
    write_new(args.output, result)

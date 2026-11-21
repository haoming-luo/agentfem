"""Unfitted shared-drag architecture on early BAM paths, not parameter transfer."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools import p92_shared_drag_reference as kernel
from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_response_screen import screen
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms


def bam_start(source_card):
    """Declared positive dimensional start, NOT literature or fitted constants."""
    return kernel.Material(source_card["backbone"]["modulus"], 200., (1e-5, 1e-7), (5., 12.),
        (80000., 20000., 10000.), (1000., 100., 50.), (1200., 150., 75.),
        (1000., 100., 100.), (.01, 1e-5, 1e-6), 10000., 50., 60., 100., 1e-5,
        tuple(1/np.asarray(source_card["moduli"])), tuple(source_card["times"]))


def probe(seed, source_dir, *, dimensional_start=False):
    source_card = seed["parameters"]
    material = replace(kernel.published_table2(), modulus=source_card["backbone"]["modulus"],
        compliances=tuple(1/np.asarray(source_card["moduli"])), times=tuple(source_card["times"]))
    if dimensional_start:
        material = bam_start(source_card)
    protocols, residuals = {}, []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != seed["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = attach_waveforms(measured_path(source, cycles=3), source)
        try:
            residual, prediction = replay(path, material, kernel=kernel)
        except (ValueError, RuntimeError) as exc:
            protocols[protocol] = dict(failure=str(exc), accepted=False)
            print(json.dumps(dict(protocol=protocol, failure=str(exc))), flush=True)
            continue
        comparison = compare_candidate(source, prediction)
        protocols[protocol] = dict(prediction=prediction, comparison=comparison,
            response_screen=screen(prediction, comparison, requested_cycles=3))
        residuals.extend(residual)
        print(json.dumps(dict(protocol=protocol, cost=float(np.dot(residual, residual)/2))), flush=True)
    return dict(schema="agentfem.research.shared-drag-probe.v1", model_id=kernel.MODEL_ID,
        parameters=asdict(material), source_sha256=seed["source_sha256"], training_cycles=[1, 2, 3],
        protocols=protocols, cost=(float(np.dot(residuals, residuals)/2)
            if all("failure" not in v for v in protocols.values()) else None), fitted=False,
        dimensional_start=dimensional_start,
        engineering_qualified=False, scope="Architecture starting point, NOT transfer of 600 C/10%Cr qualification to 620 C/P92")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bam-start", action="store_true")
    args = parser.parse_args()
    raw = args.seed.read_bytes()
    result = probe(json.loads(raw), args.source_dir, dimensional_start=args.bam_start)
    result["seed_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)

"""Matched early-only signed/accumulated recovery-memory discriminator."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material
from tools import p92_recoverable_spectrum as spectrum
from tools import p92_reversal_memory_reference as kernel
from tools.p92_response_screen import screen
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms


def probe(frozen, source_dir):
    module, base = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not spectrum:
        raise ValueError("Require a frozen spectrum card")
    paths = []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths.append((source, attach_waveforms(measured_path(source, cycles=3), source)))
    records = []
    grid = [("signed", (1., 1.))]+[(c, r) for c in ("signed", "accumulated")
        for r in ((2., 1.), (1., 2.), (.5, 1.), (1., .5))]
    for coordinate, ratios in grid:
        mat = kernel.Material(base, ratios, 1000., coordinate)
        residuals, protocols = [], {}
        for source, path in paths:
            residual, prediction = replay(path, mat, kernel=kernel)
            residuals.extend(residual)
            comparison = compare_candidate(source, prediction)
            protocols[source["protocol"]] = dict(comparison=comparison,
                response_screen=screen(prediction, comparison, requested_cycles=3),
                hold_diagnostics=prediction["hold_diagnostics"], cycles=prediction["cycles"])
        cost = float(np.dot(residuals, residuals)/2)
        records.append(dict(parameters=asdict(mat), cost=cost, protocols=protocols))
        print(json.dumps(dict(coordinate=coordinate, ratios=ratios, cost=cost)), flush=True)
    return dict(schema="agentfem.research.reversal-memory-discriminator.v1",
        training_cycles=[1, 2, 3], source_sha256=frozen["source_sha256"], records=records,
        model_id=kernel.MODEL_ID, fitted=False, engineering_qualified=False,
        scope="Shared frozen coefficients; signed versus accumulated history; no event reset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    result = probe(json.loads(raw), args.source_dir)
    result["frozen_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)

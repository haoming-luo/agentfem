"""Separate branch-resistance history on a fixed recoverable-spectrum card."""

import argparse
from dataclasses import replace
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
from tools.p92_response_screen import screen
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms


def probe(frozen, source_dir, *, mechanism="resistance"):
    module, base = decode_material(frozen["parameters"], frozen["model_id"])
    if (module is not spectrum or base.strength_ratios != (1., 1.)
            or base.hardening_ratios != (1., 1.) or mechanism not in {"resistance", "hardening"}):
        raise ValueError("Require frozen spectrum without existing history evolution and known mechanism")
    paths = []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths.append((source, attach_waveforms(measured_path(source, cycles=3), source)))
    records = []
    for ratios in ((1., 1.), (.8, 1.), (1., .8), (.8, .8)):
        updates = (dict(strength_ratios=ratios, strength_rates=(100., 100.))
                   if mechanism == "resistance" else
                   dict(hardening_ratios=ratios, hardening_rates=(100., 100.)))
        material = replace(base, **updates)
        residuals, protocols = [], {}
        for source, path in paths:
            residual, prediction = replay(path, material, kernel=module)
            residuals.extend(residual)
            comparison = compare_candidate(source, prediction)
            protocols[source["protocol"]] = dict(comparison=comparison,
                response_screen=screen(prediction, comparison, requested_cycles=3),
                hold_diagnostics=prediction["hold_diagnostics"], cycles=prediction["cycles"])
        cost = float(np.dot(residuals, residuals)/2)
        records.append(dict(ratios=ratios, cost=cost, protocols=protocols))
        print(json.dumps(dict(mechanism=mechanism, ratios=ratios, cost=cost)), flush=True)
    return dict(schema="agentfem.research.spectrum-history-discriminator.v2", mechanism=mechanism,
        training_cycles=[1, 2, 3], source_sha256=frozen["source_sha256"], records=records,
        evolution_rates=[100., 100.], fitted=False, engineering_qualified=False,
        scope="Fixed shared backbone and spectrum; branch histories, not protocol-specific corrections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mechanism", choices=("resistance", "hardening"), default="resistance")
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    result = probe(json.loads(raw), args.source_dir, mechanism=args.mechanism)
    result["frozen_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)

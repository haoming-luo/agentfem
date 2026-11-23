"""Predeclared slow recoverability discriminator with a fixed fast backbone."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools import p92_recoverable_reference as single
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_response_screen import screen
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_reference import Branch, Material
from tools.p92_waveform_controls import attach_waveforms


def probe(frozen, source_dir):
    card = frozen["parameters"]
    b = card["backbone"]
    backbone = Material(b["modulus"], tuple(Branch(**x) for x in b["branches"]))
    if card.get("strength_ratio", 1.) != 1.:
        raise ValueError("This discriminator does not silently discard strength history")
    paths = []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths.append((source, attach_waveforms(measured_path(source, cycles=3), source)))
    records = []
    for ratio, time in [(None, None), *((r, t) for r in (20., 100.) for t in (20., 200., 2000.))]:
        module = single if ratio is None else spectrum
        material = (single.Material(backbone, card["recoverable_modulus"], card["relaxation_time"])
                    if ratio is None else spectrum.Material(backbone,
                        (card["recoverable_modulus"], ratio*backbone.modulus),
                        (card["relaxation_time"], time)))
        residuals, predictions = [], {}
        for source, path in paths:
            residual, prediction = replay(path, material, kernel=module)
            residuals.extend(residual)
            predictions[source["protocol"]] = dict(prediction=prediction,
                response_screen=screen(prediction, compare_candidate(source, prediction), requested_cycles=3))
        record = dict(slow_modulus_over_E=ratio, slow_time_s=time,
            cost=float(np.dot(residuals, residuals)/2), protocols=predictions)
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "protocols"}), flush=True)
    return dict(schema="agentfem.research.recoverable-spectrum-probe.v1", parameters=card,
        source_sha256=frozen["source_sha256"], training_cycles=[1, 2, 3], records=records,
        fitted=False, engineering_qualified=False,
        scope="Fixed fast component and backbone; six declared slow-spectrum scales; no protocol correction")


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

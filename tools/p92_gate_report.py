"""Report early constraints of an optimizer candidate, including failed fits.

Unlike frozen verification this diagnostic intentionally accepts an unfinished
optimizer result, labels it, and never treats it as a qualified material.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_gate_objective import residuals
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools import p92_recoverable_reference as kernel
from tools.p92_recoverable_frozen import decode_material
from tools.p92_response_screen import screen
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_reference import Branch, Material
from tools.p92_waveform_controls import attach_waveforms


def evaluate(fit, source_dir):
    if fit["training_cycles"] != [1, 2, 3]:
        raise ValueError("Early diagnostic requires declared first three cycles")
    card = fit["fit"]["parameters"]
    plain = fit.get("without_recoverable", False)
    backbone = card if plain else card["backbone"]
    mat = Material(backbone["modulus"], tuple(Branch(**b) for b in backbone["branches"]))
    if not plain:
        module, mat = decode_material(card, fit.get("model_id", kernel.MODEL_ID))
    protocols = {}
    for protocol in ("RF", "SLR", "SLC"):
        data = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if data["sha256"] != fit["source_sha256"][protocol]:
            raise ValueError("Source changed")
        path = attach_waveforms(measured_path(data, cycles=3), data)
        _, prediction = replay(path, mat) if plain else replay(path, mat, kernel=module)
        comparison = compare_candidate(data, prediction)
        report = screen(prediction, comparison, requested_cycles=3)
        _, summary = residuals(report)
        protocols[protocol] = dict(prediction=prediction, response_screen=report, summary=summary)
        print(json.dumps(dict(protocol=protocol, failed_count=summary["failed_count"],
            maximum_normalized_error=summary["maximum_normalized_error"],
            unresolved_count=len(summary["unresolved"]))), flush=True)
    return dict(schema="agentfem.research.early-gate-diagnostic.v1", protocols=protocols,
        optimizer_converged=fit["fit"]["success"], engineering_qualified=False,
        independent_validation=False, scope="Early calibration diagnostics, not frozen validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.fit.read_bytes()
    result = evaluate(json.loads(raw), args.source_dir)
    result["fit_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)

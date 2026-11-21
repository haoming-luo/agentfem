"""Prospective fixed-card coupling discriminator; early protocols only."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools import p92_coupled_recovery_reference as coupled
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material
from tools.p92_response_screen import screen
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms


def probe(frozen, source_dir, *, candidates=None, kernel=coupled, coordinate="couplings"):
    module, base = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not spectrum:
        raise ValueError("Require fixed spectrum baseline")
    paths = []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths.append((source, attach_waveforms(measured_path(source, cycles=3), source)))
    records = []
    variants = ([(c, coupled.Material(base, c)) for c in ((0., 0.), (.1, .1), (.25, .25), (.5, .5))]
                if candidates is None else candidates(base))
    for coefficients, material in variants:
        residuals, protocols, completed = [], {}, True
        for source, path in paths:
            try:
                residual, prediction = replay(path, material, kernel=kernel)
            except (ValueError, RuntimeError) as exc:
                completed = False
                protocols[source["protocol"]] = dict(completed=False,
                    failure=dict(type=type(exc).__name__, message=str(exc)))
                continue
            residuals.extend(residual)
            comparison = compare_candidate(source, prediction)
            for row in comparison["records"]:
                row["role"] = "early_mechanism_discrimination"
            comparison.pop("groups")
            protocols[source["protocol"]] = dict(completed=True, comparison=comparison,
                response_screen=screen(prediction, comparison, requested_cycles=3),
                hold_diagnostics=prediction["hold_diagnostics"], cycles=prediction["cycles"])
        cost = float(np.dot(residuals, residuals)/2) if completed else None
        records.append(dict(**{coordinate: coefficients}, cost=cost, completed=completed, protocols=protocols))
        print(json.dumps(dict(**{coordinate: coefficients}, cost=cost, completed=completed)), flush=True)
    return dict(schema="agentfem.research.energetic-coupling-discriminator.v1",
        model_id=kernel.MODEL_ID, training_cycles=[1, 2, 3],
        source_sha256=frozen["source_sha256"], records=records,
        fitted=False, engineering_qualified=False,
        scope="Fixed shared backbone and spectrum; declared interaction, no protocol-specific correction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    guarded = [Path(__file__), Path(coupled.__file__), Path(spectrum.__file__)]
    identities = {str(p): sha256(p.read_bytes()).hexdigest() for p in guarded}
    result = probe(json.loads(raw), args.source_dir)
    if identities != {str(p): sha256(p.read_bytes()).hexdigest() for p in guarded}:
        raise RuntimeError("Research implementation changed during probe")
    if sha256(args.frozen.read_bytes()).hexdigest() != sha256(raw).hexdigest():
        raise RuntimeError("Frozen input changed during probe")
    result.update(frozen_sha256=sha256(raw).hexdigest(), implementation_sha256=identities)
    write_new(args.output, result)

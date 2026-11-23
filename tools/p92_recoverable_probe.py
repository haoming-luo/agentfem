"""Predetermined recoverable compliance/time discriminator; no optimization."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools import p92_recoverable_reference as kernel
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_reference import Branch, Material
from tools.p92_waveform_controls import attach_waveforms


def probe(frozen, source_dir, *, history=False, strength=False):
    if history and strength:
        raise ValueError("Choose one mechanism discriminator")
    card = frozen["parameters"]["backbone"] if history or strength else frozen["parameters"]
    backbone = Material(card["modulus"], tuple(Branch(**b) for b in card["branches"]))
    paths = {}
    for protocol in ("RF", "SLR", "SLC"):
        data = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if data["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source changed")
        paths[protocol] = attach_waveforms(measured_path(data, cycles=3), data)
    records = []
    # Compliance is 5–20% of elastic compliance. Times bracket ramps and holds.
    # These are hypothesis scales, NOT measured parameters or fitted bounds.
    grid = ([(1., 0.), *((r, t) for r in ((.7, .3) if strength else (.8, .5))
                        for t in (100., 1000.))] if history or strength else
            [(None, None), *((r, t) for r in (5., 10., 20.) for t in (.2, 2., 20., 200.))])
    for ratio, tau in grid:
        residuals, protocols = [], {}
        if strength:
            mat = kernel.Material(backbone, frozen["parameters"]["recoverable_modulus"],
                frozen["parameters"]["relaxation_time"], strength_ratio=ratio, strength_softening_rate=tau)
        elif history:
            a, b = backbone.branches
            evolved = replace(backbone, branches=(replace(a, saturation_ratio=ratio, softening_rate=tau), b))
            mat = kernel.Material(evolved, frozen["parameters"]["recoverable_modulus"],
                                  frozen["parameters"]["relaxation_time"])
        else:
            mat = backbone if ratio is None else kernel.Material(backbone, ratio*backbone.modulus, tau)
        try:
            for protocol, path in paths.items():
                residual, prediction = (replay(path, mat) if ratio is None else
                                        replay(path, mat, kernel=kernel))
                residuals.extend(residual)
                protocols[protocol] = prediction
            cost = float(np.dot(residuals, residuals)/2)
            failure = None
        except (ValueError, RuntimeError) as exc:
            cost, failure = None, str(exc)
        coordinates = (dict(strength_ratio=ratio, strength_softening_rate=tau) if strength else
                       dict(saturation_ratio=ratio, softening_rate=tau) if history else
                       dict(modulus_ratio=ratio, relaxation_time_s=tau))
        records.append(dict(**coordinates, cost=cost, failure=failure, protocols=protocols))
        print(json.dumps({k: v for k, v in records[-1].items() if k != "protocols"}), flush=True)
    return dict(schema="agentfem.research.recoverable-probe.v1", backbone=asdict(backbone),
                source_sha256=frozen["source_sha256"], training_cycles=[1, 2, 3],
                records=records, fitted=False, engineering_qualified=False,
                history_probe=history, strength_probe=strength,
                scope="Fixed-backbone mechanism discriminator, not material identification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--strength", action="store_true")
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    result = probe(json.loads(raw), args.source_dir, history=args.history, strength=args.strength)
    result["frozen_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)

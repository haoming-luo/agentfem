"""Prespecified early-only positive saturation-memory discriminator."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_probe import bridge_seed
from tools.p92_two_rate_reference import Material, Branch
from tools.p92_waveform_controls import attach_waveforms


def probe(seed, source_dir, *, ratcheting=False, creep=False):
    if ratcheting and creep:
        raise ValueError("Keep mechanism experiments separate")
    if ratcheting or creep:
        if not seed["conditional_fit"]["success"] or seed["training_cycles"] != [1, 2, 3]:
            raise ValueError("Require completed early conditional fit")
        card = seed["conditional_fit"]["parameters"]
        base = Material(card["modulus"], tuple(Branch(**b) for b in card["branches"]))
        seed_identity = sha256(json.dumps(card, sort_keys=True).encode()).hexdigest()
    else:
        base = bridge_seed(seed)
        seed_identity = seed["parameter_sha256"]
    paths = []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != seed["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths.append(attach_waveforms(measured_path(source, cycles=3), source))
    records = []
    # Fixed diagnostic grid, not an expanding search or fitted parameter card.
    branches = ([replace(base.branches[0], ratcheting=(chi, chi)) for chi in (0., 1., 3., 6.)]
        if ratcheting else [replace(base.branches[0], saturation_ratio=ratio, softening_rate=rate)
            for ratio, rate in ((1., 0.), (.8, 100.), (.8, 1000.), (.5, 100.), (.5, 1000.))])
    cards = ([replace(base, branches=(base.branches[0], replace(base.branches[1],
              strength=0., reference_flow=q))) for q in (1e-9, 1e-8, 1e-7, 1e-6)] if creep else
             [replace(base, branches=(branch, base.branches[1])) for branch in branches])
    for card in cards:
        protocols = {}
        failure = None
        for path in paths:
            try:
                residual, result = replay(path, card)
            except (ValueError, RuntimeError) as error:
                failure = dict(protocol=path["protocol"], reason=str(error))
                break
            protocols[path["protocol"]] = dict(cost=float(.5*residual@residual), prediction=result)
        records.append(dict(parameters=asdict(card), protocols=protocols,
            cost=None if failure else sum(p["cost"] for p in protocols.values()), failure=failure))
    return dict(schema="agentfem.research.two-rate-history-probe.v1", training_cycles=[1, 2, 3],
        source_sha256=seed["source_sha256"], seed_parameter_sha256=seed_identity,
        candidates=records, scope=("Fixed zero-threshold branch-B creep grid" if creep else
            "Fixed fast-branch ratcheting grid" if ratcheting else
            "Fixed fast-branch saturation-memory grid")+"; no later-cycle selection",
        fitted=False, engineering_qualified=False, life_qualified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ratcheting", action="store_true")
    parser.add_argument("--creep", action="store_true")
    args = parser.parse_args()
    result = probe(json.loads(args.seed.read_text()), args.source_dir, ratcheting=args.ratcheting, creep=args.creep)
    write_new(args.output, result)
    print(json.dumps([dict(ratio=r["parameters"]["branches"][0]["saturation_ratio"],
        rate=r["parameters"]["branches"][0]["softening_rate"],
        ratcheting=r["parameters"]["branches"][0]["ratcheting"], cost=r["cost"])
        for r in result["candidates"]]))

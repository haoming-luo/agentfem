"""Unfitted bridge seed for independent two-rate control-path discrimination."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_reference import Branch, Material
from tools.p92_waveform_controls import attach_waveforms


def bridge_seed(seed):
    """Match initial flow/tangent scales, not complete constitutive behavior.

    Linear recovery matches the old nonlinear recovery at nominal saturation.
    Isotropic softening is deliberately absent; this is NOT parameter transfer
    validation, and must never be described as a published parameter set.
    """
    cards = seed["parameters"]
    if sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest() != seed["parameter_sha256"]:
        raise ValueError("Seed identity changed")
    if len(cards) != 2 or not all(c["enabled"] for c in cards):
        raise ValueError("Require two active seed branches")
    branches = []
    for card in cards:
        p = card["parameters"]
        a = (p["a1"], p["a2"])
        branches.append(Branch(p["k"], (120./p["Z"])**p["n"], p["n"],
            (p["C1"]*a[0], p["C2"]*a[1]), a,
            tuple((p["gamma1"]+p["gamma2"])*x**(p["m"]-1.) for x in a),
            (card["exponent"],)*2))
    if cards[0]["parameters"]["E"] != cards[1]["parameters"]["E"]:
        raise ValueError("Branches must share elastic modulus")
    return Material(cards[0]["parameters"]["E"], tuple(branches))


def probe(seed, source_dir):
    material = bridge_seed(seed)
    results = {}
    for protocol in ("RF", "SLR", "SLC"):
        data = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if data["sha256"] != seed["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = attach_waveforms(measured_path(data, cycles=3), data)
        r, prediction = replay(path, material)
        results[protocol] = dict(cost=float(.5*r@r), prediction=prediction)
    return dict(schema="agentfem.research.two-rate-probe.v1", parameters=asdict(material),
        seed_parameter_sha256=seed["parameter_sha256"], protocols=results,
        scope="Unfitted bridge scales, initial strength fixed; no isotropic softening or life prediction",
        fitted=False, engineering_qualified=False, life_qualified=False,
        cost=float(sum(x["cost"] for x in results.values())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = probe(json.loads(args.seed.read_text()), args.source_dir)
    write_new(args.output, result)
    print(json.dumps(dict(cost=result["cost"], fitted=False)))

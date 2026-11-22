"""Fixed-card test of recoverable/backstress interaction; no parameter fit."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_coupling_probe as common
from tools import p92_recoverable_backstress_reference as kernel
from tools.p92_measured_fit import write_new


def candidates(base):
    a, b = base.backbone.branches
    reduced = replace(base, backbone=replace(base.backbone,
                      branches=(replace(a, static_recovery=(0., 0.)), b)))
    return [("original", kernel.Material(base)), ("deletion-only", kernel.Material(reduced)),
            *[(str(k), kernel.Material(reduced, (k, k))) for k in (.1, .25, -.1)]]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frozen", type=Path)
    p.add_argument("source_dir", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    names = ("p92_recoverable_backstress_probe.py", "p92_recoverable_backstress_reference.py",
             "p92_recoverable_spectrum.py", "p92_coupling_probe.py", "p92_recoverable_frozen.py",
             "p92_two_rate_reference.py", "p92_two_rate_controls.py", "p92_bounded_softening.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_bam_source.py",
             "p92_response_screen.py", "p92_hold_observations.py", "p92_measured_fit.py")
    def identities():
        return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    raw, before = args.frozen.read_bytes(), identities()
    result = common.probe(json.loads(raw), args.source_dir, candidates=candidates,
                          kernel=kernel, coordinate="variant")
    if identities() != before or args.frozen.read_bytes() != raw:
        raise RuntimeError("Research input changed during experiment")
    result.update(schema="agentfem.research.recoverable-backstress-discriminator.v1",
                  driver_sha256=before, seed_sha256=sha256(raw).hexdigest(),
                  removed_coefficients="A.static_recovery[0:2]", added_coefficients="couplings[0:2]")
    write_new(args.output, result)

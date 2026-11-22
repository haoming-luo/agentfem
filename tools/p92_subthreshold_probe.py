"""Fixed rate/slope-matched removal of branch B's threshold; research only."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools import p92_coupling_probe as common
from tools import p92_recoverable_spectrum as kernel
from tools.p92_measured_fit import write_new


def matched_branch(branch, anchor):
    if not np.isfinite(anchor) or anchor <= branch.strength:
        raise ValueError("Matching stress must exceed the positive threshold")
    exponent = branch.exponent*anchor/(anchor-branch.strength)
    log_rate = (np.log(branch.reference_flow)
                + branch.exponent*np.log((anchor-branch.strength)/120.)
                - exponent*np.log(anchor/120.))
    return replace(branch, strength=0., exponent=exponent, reference_flow=float(np.exp(log_rate)))


def candidates(base):
    a, b = base.backbone.branches
    return [("control", base), *[(f"anchor-{s:g}", replace(base,
        backbone=replace(base.backbone, branches=(a, matched_branch(b, s)))))
        for s in (200., 250., 300.)]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = ("p92_subthreshold_probe.py", "p92_recoverable_spectrum.py", "p92_coupling_probe.py",
             "p92_recoverable_frozen.py", "p92_two_rate_reference.py", "p92_two_rate_controls.py",
             "p92_two_rate_frozen.py", "p92_measured_controls.py", "p92_waveform_controls.py",
             "p92_response_screen.py", "p92_hold_observations.py", "p92_bam_source.py",
             "p92_bounded_softening.py", "p92_waveform_experiment.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    raw, before = args.frozen.read_bytes(), identities()
    result = common.probe(json.loads(raw), args.source_dir, candidates=candidates,
                          kernel=kernel, coordinate="variant")
    if identities() != before or args.frozen.read_bytes() != raw:
        raise RuntimeError("Input or implementation changed during experiment")
    result.update(schema="agentfem.research.subthreshold-discriminator.v1",
                  driver_sha256=before, seed_sha256=sha256(raw).hexdigest(), fitted=False)
    write_new(args.output, result)

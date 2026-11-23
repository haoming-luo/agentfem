"""Predeclared original and matched B activation, with no fitting."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_coupling_probe as common
from tools import p92_activated_flow_reference as kernel
from tools.p92_measured_fit import write_new


def candidates(base):
    return [(enabled, kernel.Material(base, enabled)) for enabled in (False, True)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = ("p92_activated_flow_probe.py", "p92_activated_flow_reference.py",
             "p92_recoverable_spectrum.py", "p92_coupling_probe.py", "p92_recoverable_frozen.py",
             "p92_two_rate_reference.py", "p92_two_rate_controls.py", "p92_bounded_softening.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_bam_source.py",
             "p92_response_screen.py", "p92_hold_observations.py", "p92_measured_fit.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    raw, before = args.frozen.read_bytes(), identities()
    result = common.probe(json.loads(raw), args.source_dir, candidates=candidates,
                          kernel=kernel, coordinate="activated")
    if identities() != before or args.frozen.read_bytes() != raw:
        raise RuntimeError("Research input changed during experiment")
    result.update(schema="agentfem.research.activated-flow-discriminator.v1",
                  driver_sha256=before, seed_sha256=sha256(raw).hexdigest(),
                  added_coefficients=[], added_state=[])
    write_new(args.output, result)

"""Predeclared nonlinear recoverable kinetics arms; no fitted parameters."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_coupling_probe as common
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_measured_fit import write_new


def candidates(base):
    if base.recoverable_powers != (1., 1.):
        raise ValueError("Require original linear recoverable kinetics")
    return [(powers, replace(base, recoverable_powers=powers))
            for powers in ((1., 1.), (2., 1.), (1., 2.), (2., 2.))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = ("p92_recoverable_kinetics_probe.py", "p92_recoverable_spectrum.py", "p92_coupling_probe.py",
             "p92_recoverable_frozen.py", "p92_two_rate_reference.py", "p92_two_rate_controls.py",
             "p92_two_rate_frozen.py", "p92_measured_controls.py", "p92_waveform_controls.py",
             "p92_response_screen.py", "p92_hold_observations.py", "p92_bam_source.py",
             "p92_bounded_softening.py", "p92_waveform_experiment.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    raw, before = args.frozen.read_bytes(), identities()
    result = common.probe(json.loads(raw), args.source_dir, candidates=candidates,
                          kernel=spectrum, coordinate="recoverable_powers")
    if identities() != before or args.frozen.read_bytes() != raw:
        raise RuntimeError("Input or implementation changed during experiment")
    result.update(schema="agentfem.research.recoverable-kinetics-discriminator.v1",
                  driver_sha256=before, seed_sha256=sha256(raw).hexdigest(),
                  force_normalization_MPa=120., added_fitted_parameters=0)
    write_new(args.output, result)

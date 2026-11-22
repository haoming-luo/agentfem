"""Independent static-softening / strain-rehardening fixed-card discriminator."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_coupling_probe as common
from tools import p92_resistance_recovery_reference as kernel
from tools.p92_measured_fit import write_new


def candidates(base):
    return [(name, kernel.Material(base, "strength", minimum, dynamic, 1/tau,
                                   evolution="static_softening", history="shared"))
            for name, minimum, dynamic, tau in (
                ("control", 1., 250., 1800.),
                ("b250-t1800", .9, 250., 1800.),
                ("b1000-t1800", .9, 1000., 1800.),
                ("b250-t180", .9, 250., 180.))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = ("p92_static_strength_probe.py", "p92_resistance_recovery_reference.py",
             "p92_recoverable_spectrum.py", "p92_coupling_probe.py",
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
    result.update(schema="agentfem.research.static-strength-discriminator.v1",
                  driver_sha256=before, seed_sha256=sha256(raw).hexdigest(),
                  evolution="static_softening", history="shared", fitted=False)
    write_new(args.output, result)

"""Fixed-card critical-state recovery discriminator, without calibration."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_coupling_probe as common
from tools import p92_critical_recovery_reference as kernel
from tools.p92_measured_fit import write_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = ("p92_critical_recovery_probe.py", "p92_critical_recovery_reference.py",
             "p92_coupling_probe.py", "p92_recoverable_frozen.py", "p92_recoverable_spectrum.py",
             "p92_two_rate_reference.py", "p92_two_rate_controls.py", "p92_bounded_softening.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_bam_source.py",
             "p92_response_screen.py", "p92_hold_observations.py", "p92_measured_fit.py")

    def identities():
        return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}

    raw, before = args.frozen.read_bytes(), identities()
    result = common.probe(json.loads(raw), args.source_dir,
                          candidates=lambda base: [("critical_state", kernel.Material(base))],
                          kernel=kernel, coordinate="replacement")
    if identities() != before or args.frozen.read_bytes() != raw:
        raise RuntimeError("Research input changed during experiment")
    result.update(schema="agentfem.research.critical-recovery-discriminator.v1",
                  driver_sha256=before, seed_sha256=sha256(raw).hexdigest(),
                  added_coefficients=[], added_state=[],
                  removed_coefficients=["backstress_recall_exponents"],
                  scope="Fixed coefficients, constrained recovery replacing smooth recall; no fit")
    write_new(args.output, result)

"""Predeclared static recovery eigenmode assignments; no parameter fit."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_coupling_probe as common
from tools import p92_static_modes_reference as kernel
from tools.p92_measured_fit import write_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recoverable", action="store_true", help="Separate Kelvin/backstress mobility comparison")
    parser.add_argument("--exchange", action="store_true", help="Exchange existing slow static rates, without coupling")
    parser.add_argument("--exchange-recoverable", action="store_true", help="Exchange existing slow Kelvin/B time scales")
    args = parser.parse_args()
    if sum((args.recoverable, args.exchange, args.exchange_recoverable)) > 1:
        parser.error("Run one isolated comparison at a time")
    names = ("p92_static_modes_probe.py", "p92_static_modes_reference.py",
             "p92_coupling_probe.py", "p92_recoverable_frozen.py", "p92_recoverable_spectrum.py",
             "p92_two_rate_reference.py", "p92_two_rate_controls.py", "p92_bounded_softening.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_bam_source.py",
             "p92_response_screen.py", "p92_hold_observations.py", "p92_measured_fit.py")

    def identities():
        return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}

    raw, before = args.frozen.read_bytes(), identities()
    modes = (("diagonal", "recoverable_exchanged") if args.exchange_recoverable else
             ("diagonal", "exchanged_slow") if args.exchange else
             ("diagonal", "recoverable_shared", "recoverable_difference") if args.recoverable
             else ("diagonal", "shared_slow", "difference_slow"))
    result = common.probe(json.loads(raw), args.source_dir,
        candidates=lambda base: [(name, kernel.Material(base, name))
                                for name in modes],
        kernel=kernel, coordinate="mode")
    if identities() != before or args.frozen.read_bytes() != raw:
        raise RuntimeError("Research input changed during experiment")
    result.update(schema="agentfem.research.static-recovery-modes.v1",
        driver_sha256=before, seed_sha256=sha256(raw).hexdigest(), added_coefficients=[], added_state=[],
        recovery_eigenvalues_unchanged=True, scope="Discrete static mobility assignment, fixed coefficients")
    write_new(args.output, result)

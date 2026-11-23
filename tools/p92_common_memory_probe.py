"""Isolate common kinematic memory without drag or parameter changes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_common_memory_reference as kernel
from tools.p92_coupling_probe import probe
from tools.p92_measured_fit import write_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--slow-only", action="store_true")
    args = parser.parse_args()
    names = ("p92_common_memory_probe.py", "p92_common_memory_reference.py", "p92_coupling_probe.py",
             "p92_recoverable_frozen.py", "p92_recoverable_spectrum.py", "p92_two_rate_reference.py",
             "p92_two_rate_controls.py", "p92_bounded_softening.py", "p92_measured_controls.py",
             "p92_waveform_controls.py", "p92_bam_source.py", "p92_response_screen.py",
             "p92_hold_observations.py", "p92_measured_fit.py")
    raw = args.frozen.read_bytes()
    before = {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    arms = (False, "slow") if args.slow_only else (False, True)
    result = probe(json.loads(raw), args.source_dir, kernel=kernel, coordinate="shared",
                   candidates=lambda s: [(v, kernel.Material(s, v)) for v in arms])
    if raw != args.frozen.read_bytes() or any(
            sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() != h for n, h in before.items()):
        raise RuntimeError("Research input changed during comparison")
    result.update(schema="agentfem.research.common-memory-probe.v1", driver_sha256=before,
                  frozen_sha256=sha256(raw).hexdigest(), added_coefficients=[], added_state=[],
                  scope="Fixed-card common effective stress versus independent memory; no new drag")
    write_new(args.output, result)

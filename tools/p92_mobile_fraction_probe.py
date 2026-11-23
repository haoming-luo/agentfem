"""Early-only population-mobility replacement with a deletion control."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_coupling_probe as common
from tools import p92_mobile_fraction_reference as kernel
from tools.p92_measured_fit import write_new


def candidates(base):
    a, b = base.backbone.branches
    deleted = replace(base, backbone=replace(base.backbone,
        branches=(replace(a, static_recovery=(0., 0.)), b)))
    return [("original", kernel.Material(base, 0., 250.)),
            ("deletion-only", kernel.Material(deleted, 0., 250.)),
            ("mobile-population", kernel.Material(deleted, 1./1800., 250.))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seed", "source_dir", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    names = ("p92_mobile_fraction_probe.py", "p92_mobile_fraction_reference.py",
        "p92_recoverable_spectrum.py", "p92_two_rate_reference.py", "p92_coupling_probe.py",
        "p92_recoverable_frozen.py", "p92_two_rate_controls.py", "p92_measured_controls.py",
        "p92_waveform_controls.py", "p92_bam_source.py", "p92_hold_observations.py",
        "p92_response_screen.py", "p92_bounded_softening.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    raw, before = args.seed.read_bytes(), identities()
    result = common.probe(json.loads(raw), args.source_dir, candidates=candidates,
                          kernel=kernel, coordinate="variant")
    if before != identities() or raw != args.seed.read_bytes():
        raise RuntimeError("Probe inputs changed")
    result.update(schema="agentfem.research.mobile-fraction-probe.v1",
                  driver_sha256=before, seed_sha256=sha256(raw).hexdigest(),
                  added_independent_coefficients=0, substituted_coordinates=["A.static1", "A.static2"],
                  life_qualified=False)
    write_new(args.output, result)

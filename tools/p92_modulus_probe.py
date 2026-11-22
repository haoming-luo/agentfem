"""Fixed literature-modulus sensitivity; not an elastic calibration."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_recoverable_spectrum as kernel
from tools.p92_coupling_probe import probe
from tools.p92_measured_fit import write_new


def variants(material, *, source_modulus=False):
    """No inherited evolved states and no other coefficient changes."""
    if type(source_modulus) is not bool:
        raise ValueError("Require explicit source-modulus comparison flag")
    return [(e, replace(material, backbone=replace(material.backbone, modulus=e)))
            for e in (material.backbone.modulus, 151000. if source_modulus else 162000.)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-modulus", action="store_true")
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_modulus_probe.py", "p92_coupling_probe.py", "p92_recoverable_spectrum.py",
             "p92_two_rate_reference.py", "p92_two_rate_controls.py", "p92_recoverable_frozen.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_bam_source.py",
             "p92_response_screen.py", "p92_hold_observations.py", "p92_bounded_softening.py")
    before = {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    result = probe(json.loads(raw), args.source_dir,
                   candidates=lambda m: variants(m, source_modulus=args.source_modulus), kernel=kernel,
                   coordinate="modulus_MPa")
    if raw != args.frozen.read_bytes() or any(
            sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() != h for n, h in before.items()):
        raise RuntimeError("Research input changed during comparison")
    result.update(schema="agentfem.research.modulus-sensitivity.v1", driver_sha256=before,
        frozen_sha256=sha256(raw).hexdigest(), life_qualified=False,
        reference=("https://doi.org/10.1016/j.ijfatigue.2022.107381" if args.source_modulus
                   else "https://doi.org/10.3390/met9010099"),
        reference_locator="page 5 section 3.1" if args.source_modulus else "Table 2",
        modulus_is_not_specimen_measurement=True, added_coefficients=[],
        scope="Fixed literature sensitivity point; not a new fit or transferable material card")
    write_new(args.output, result)

"""One declared secant-mapped replacement, with no parameter fitting."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_coupling_probe as common
from tools import p92_recovery_replacement_reference as kernel
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", type=Path)
    p.add_argument("source_dir", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    names = ("p92_recovery_replacement_probe.py", "p92_recovery_replacement_reference.py",
             "p92_coupling_probe.py", "p92_recoverable_frozen.py", "p92_recoverable_spectrum.py",
             "p92_two_rate_reference.py", "p92_two_rate_controls.py", "p92_bam_source.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_hold_observations.py",
             "p92_bounded_softening.py", "p92_response_screen.py")

    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}

    raw, before = args.seed.read_bytes(), identities()
    seed = json.loads(raw)
    _, base = decode_material(seed["parameters"], seed["model_id"])
    material = kernel.replace_low_stress_flow(base)
    result = common.probe(seed, args.source_dir, kernel=kernel, coordinate="replacement",
        candidates=lambda original: [("A_to_Kelvin", kernel.replace_low_stress_flow(original))])
    if identities() != before or args.seed.read_bytes() != raw:
        raise RuntimeError("Research inputs changed during replacement screen")
    result.update(schema="agentfem.research.recovery-replacement-discriminator.v1",
        mapped_parameters=asdict(material), driver_sha256=before, seed_sha256=sha256(raw).hexdigest(),
        scope="One irreversible flow replaced, not augmented; no fitted parameters; not a steady-creep life model")
    write_new(args.output, result)

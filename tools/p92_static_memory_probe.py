"""Declared own/shared static-recovery history experiment; no fitting."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_coupling_probe as common
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_recoverable_frozen import decode_material, evaluate
from tools.p92_measured_fit import write_new


def candidates(base):
    if base.static_ratios != (1., 1.):
        raise ValueError("Require unchanged static-history baseline")
    return [("control", base)]+[(f"{history}-{ratio}", replace(base,
        static_ratios=(ratio, ratio), static_history_rates=(250., 250.), static_history=history))
        for history in ("own", "shared") for ratio in (.5, .1)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--freeze", choices=("own", "shared"))
    args = parser.parse_args()
    names = ("p92_static_memory_probe.py", "p92_recoverable_spectrum.py", "p92_coupling_probe.py",
             "p92_recoverable_frozen.py", "p92_two_rate_reference.py", "p92_two_rate_controls.py",
             "p92_two_rate_frozen.py", "p92_measured_controls.py", "p92_waveform_controls.py",
             "p92_response_screen.py", "p92_hold_observations.py", "p92_bam_source.py",
             "p92_bounded_softening.py", "p92_waveform_experiment.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    raw, before = args.frozen.read_bytes(), identities()
    frozen = json.loads(raw)
    if args.freeze:
        module, base = decode_material(frozen["parameters"], frozen["model_id"])
        if module is not spectrum:
            raise ValueError("Require spectrum baseline")
        variant = dict(candidates(base))[f"{args.freeze}-0.5"]
        frozen["parameters"] = asdict(variant)
        result = evaluate(frozen, args.source_dir, fixed_card=True)
        result["fixed_variant"] = args.freeze
    else:
        result = common.probe(frozen, args.source_dir, candidates=candidates,
                              kernel=spectrum, coordinate="variant")
        result.update(schema="agentfem.research.static-memory-discriminator.v1",
                      static_history_rate_per_strain=250., static_ratios=[.5, .1])
    if identities() != before or args.frozen.read_bytes() != raw:
        raise RuntimeError("Input or implementation changed during experiment")
    result.update(driver_sha256=before, seed_sha256=sha256(raw).hexdigest())
    write_new(args.output, result)

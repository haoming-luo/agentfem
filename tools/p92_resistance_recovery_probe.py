"""Predeclared mobility versus strength recovery, shared early control paths."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_resistance_recovery_reference as kernel
from tools import p92_coupling_probe as common
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_measured_fit import write_new


def candidates(base):
    variants = [("control", kernel.Material(base, "strength", 1., 250., 1/180.))]
    variants.extend((f"{kind}-{int(tau)}s", kernel.Material(base, kind, .9, 250., 1/tau))
                    for kind in ("strength", "mobility") for tau in (180., 1800.))
    return variants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    guarded = [Path(__file__), Path(kernel.__file__), Path(common.__file__), Path(spectrum.__file__)]
    identities = {str(p): sha256(p.read_bytes()).hexdigest() for p in guarded}
    result = common.probe(json.loads(raw), args.source_dir, candidates=candidates,
                          kernel=kernel, coordinate="variant")
    if identities != {str(p): sha256(p.read_bytes()).hexdigest() for p in guarded}:
        raise RuntimeError("Research implementation changed during probe")
    if args.frozen.read_bytes() != raw:
        raise RuntimeError("Frozen input changed during probe")
    result.update(schema="agentfem.research.resistance-recovery-discriminator.v1",
        minimum=.9, dynamic_per_strain=250., recovery_times_s=[180., 1800.],
        anchor_overstress_MPa=120., control_minimum=1.,
        frozen_sha256=sha256(raw).hexdigest(), implementation_sha256=identities)
    write_new(args.output, result)

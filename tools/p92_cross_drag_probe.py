"""Fixed early cross-drag discriminator; identical controls and response gates."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_cross_drag_reference as cross
from tools import p92_coupling_probe as common
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_measured_fit import write_new


def candidates(base):
    return [(maximum, cross.Material(base, maximum, 1000., 1000., 1/1800.))
            for maximum in (0., .5, 1., 2.)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    guarded = [Path(__file__), Path(cross.__file__), Path(common.__file__), Path(spectrum.__file__)]
    identities = {str(p): sha256(p.read_bytes()).hexdigest() for p in guarded}
    result = common.probe(json.loads(raw), args.source_dir, candidates=candidates,
                          kernel=cross, coordinate="maximum_kinetic_history")
    if identities != {str(p): sha256(p.read_bytes()).hexdigest() for p in guarded}:
        raise RuntimeError("Research implementation changed during probe")
    if args.frozen.read_bytes() != raw:
        raise RuntimeError("Frozen input changed during probe")
    result.update(schema="agentfem.research.cross-drag-discriminator.v1",
        build=1000., erase=1000., recovery_per_s=1/1800.,
        frozen_sha256=sha256(raw).hexdigest(), implementation_sha256=identities)
    write_new(args.output, result)

"""Fixed-card replacement of one signed memory, with original control."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_cross_resistance_reference as replacement
from tools import p92_recoverable_spectrum as original
from tools.p92_coupling_probe import probe
from tools.p92_measured_fit import write_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_cross_resistance_probe.py", "p92_cross_resistance_reference.py",
        "p92_coupling_probe.py", "p92_recoverable_spectrum.py", "p92_two_rate_reference.py",
        "p92_two_rate_controls.py", "p92_measured_controls.py", "p92_waveform_controls.py",
        "p92_recoverable_frozen.py", "p92_bam_source.py", "p92_response_screen.py",
        "p92_hold_observations.py", "p92_bounded_softening.py")
    before = {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    records = []
    for kernel, factory in ((original, lambda m: m), (replacement, replacement.Material)):
        report = probe(json.loads(raw), args.source_dir, kernel=kernel, coordinate="model_id",
                       candidates=lambda m, k=kernel, f=factory: [(k.MODEL_ID, f(m))])
        records.extend(report["records"])
    if raw != args.frozen.read_bytes() or any(
            sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() != h for n, h in before.items()):
        raise RuntimeError("Research input changed during paired probe")
    result = dict(schema="agentfem.research.cross-resistance-replacement.v1", records=records,
        source_sha256=report["source_sha256"], training_cycles=[1, 2, 3],
        frozen_sha256=sha256(raw).hexdigest(), driver_sha256=before,
        added_coefficients=[], added_state=[], inactive_original_coefficient="B.ratcheting[1]",
        replaced_state="B slow signed backstress -> nonnegative shared resistance",
        fitted=False, engineering_qualified=False, life_qualified=False,
        scope="One parameter-neutral fixed-card topology comparison; no family rejection proof")
    write_new(args.output, result)

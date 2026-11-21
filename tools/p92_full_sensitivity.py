"""Audit all positive backbone/recoverable directions before further fitting.

Dimensionless log10 coordinates; singular values are numerical sensitivities,
not statistical identifiability or parameter uncertainty.
"""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools import p92_recoverable_reference as kernel
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_reference import Branch, Material
from tools.p92_waveform_controls import attach_waveforms

FIELDS = (("strength", None), ("reference_flow", None), ("exponent", None),
          ("hardening", 0), ("hardening", 1), ("saturation", 0), ("saturation", 1),
          ("static_recovery", 0), ("static_recovery", 1))
NAMES = tuple(f"{b}.{field}{'' if i is None else i+1}" for b in ("A", "B") for field, i in FIELDS)+(
    "recoverable.modulus", "recoverable.time")


def decode(card, z):
    z = np.asarray(z, dtype=float)
    if z.shape != (20,) or not np.isfinite(z).all():
        raise ValueError("Require twenty finite logarithmic coordinates")
    candidate = deepcopy(card)
    for branch, dz in zip(candidate["backbone"]["branches"], (z[:9], z[9:18]), strict=True):
        for (field, i), value in zip(FIELDS, dz, strict=True):
            if i is None:
                shift = 1. if field == "exponent" else 0.
                branch[field] = shift+(branch[field]-shift)*10**value
            else:
                branch[field] = list(branch[field])
                branch[field][i] *= 10**value
        for field in ("hardening", "saturation", "static_recovery", "ratcheting"):
            branch[field] = tuple(branch[field])
    candidate["recoverable_modulus"] *= 10**z[18]
    candidate["relaxation_time"] *= 10**z[19]
    b = candidate.pop("backbone")
    return kernel.Material(Material(b["modulus"], tuple(Branch(**x) for x in b["branches"])), **candidate)


def audit(frozen, source_dir):
    card = frozen["parameters"]
    paths = []
    for protocol in ("RF", "SLR", "SLC"):
        data = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if data["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths.append(attach_waveforms(measured_path(data, cycles=3), data))

    def response(z):
        return np.concatenate([replay(path, decode(card, z), kernel=kernel)[0] for path in paths])

    # Two central-difference scales distinguish weak directions from subtraction
    # artifacts. No parameter optimization or post-cycle data are used here.
    matrices, summaries = [], []
    for step in (.001, .0005):
        columns = []
        for i, name in enumerate(NAMES):
            delta = np.eye(20)[i]*step
            column = (response(delta)-response(-delta))/(2*step)
            columns.append(column)
            print(json.dumps(dict(step=step, coordinate=name, norm=float(np.linalg.norm(column)))), flush=True)
        jac = np.column_stack(columns)
        _, singular, vectors = np.linalg.svd(jac, full_matrices=False)
        matrices.append(jac)
        summaries.append(dict(step=step, singular_values=singular.tolist(),
            column_norms=np.linalg.norm(jac, axis=0).tolist(), right_vectors=vectors.tolist(),
            numerical_rank_1e_minus_3_relative=int(np.count_nonzero(singular > singular[0]*1e-3))))
    return dict(schema="agentfem.research.full-sensitivity.v1", coordinates=NAMES,
        parameters=card, source_sha256=frozen["source_sha256"], training_cycles=[1, 2, 3],
        audits=summaries, relative_jacobian_difference=float(np.linalg.norm(matrices[1]-matrices[0])/
                                                            np.linalg.norm(matrices[1])),
        fitted=False, engineering_qualified=False,
        scope="Local dimensionless sensitivity; neither uniqueness nor uncertainty certification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    result = audit(json.loads(raw), args.source_dir)
    result["frozen_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)

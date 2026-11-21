"""Matched early-only identification of linear/nonlinear static recovery.

One card per hypothesis, shared by all loading protocols. Never a life card.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tools import p92_recoverable_spectrum as kernel
from tools.p92_full_sensitivity import decode as decode_backbone, NAMES as BASE_NAMES
from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms

NAMES = (*BASE_NAMES, "recoverable.slow_modulus", "recoverable.slow_time")


def decode(card, z, powers):
    z = np.asarray(z, dtype=float)
    if z.shape != (22,) or not np.isfinite(z).all():
        raise ValueError("Require 22 finite logarithmic coordinates")
    if any(card.get(name, default) != default and tuple(card.get(name, default)) != default
           for name, default in (("strength_ratios", (1., 1.)), ("hardening_ratios", (1., 1.)),
                                 ("static_powers", (0., 0.)))):
        raise ValueError("Require an unevolved baseline spectrum")
    base = decode_backbone(dict(backbone=card["backbone"],
        recoverable_modulus=card["moduli"][0], relaxation_time=card["times"][0]), z[:20])
    return kernel.Material(base.backbone,
        (base.recoverable_modulus, card["moduli"][1]*10**z[20]),
        (base.relaxation_time, card["times"][1]*10**z[21]), static_powers=tuple(powers))


def identify(seed, source_dir, powers):
    if seed["model_id"] != kernel.MODEL_ID or tuple(powers) not in {(0., 0.), (0., 1.)}:
        raise ValueError("Require predeclared matched recovery hypotheses")
    paths = []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != seed["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths.append(attach_waveforms(measured_path(source, cycles=3), source))
    calls, rejected = 0, []
    def response(z):
        nonlocal calls
        calls += 1
        mat = decode(seed["parameters"], z, powers)
        result = np.concatenate([replay(path, mat, kernel=kernel)[0] for path in paths])
        if calls % 10 == 1:
            print(json.dumps(dict(powers=powers, evaluation=calls, cost=float(result@result/2))), flush=True)
        return result
    baseline = response(np.zeros(22))
    matrices, audits = [], []
    for step in (.001, .0005):
        matrix = np.column_stack([(response(np.eye(22)[i]*step)-response(-np.eye(22)[i]*step))/(2*step)
                                  for i in range(22)])
        _, values, vectors = np.linalg.svd(matrix, full_matrices=False)
        matrices.append(matrix)
        audits.append(dict(step=step, singular_values=values.tolist(), right_vectors=vectors.tolist()))
    relative = float(np.linalg.norm(matrices[1]-matrices[0])/np.linalg.norm(matrices[1]))
    va, vb = (np.asarray(a["right_vectors"])[:10] for a in audits)
    distance = float(np.linalg.norm(va.T@va-vb.T@vb, ord=2))
    leading = audits[1]["singular_values"][:10]
    if (not np.isfinite([relative, distance, *leading]).all() or relative > .002 or distance > .02
            or leading[-1] < 1e-5 or leading[0]/leading[-1] > 1e4):
        raise ValueError(f"Unresolved recovery derivative audit: {relative=}, {distance=}")
    basis, bound = vb.T, .3/np.sqrt(10)
    def objective(z):
        try:
            return response(basis@z)
        except (ValueError, RuntimeError) as exc:
            rejected.append(dict(coordinates=z.tolist(), reason=str(exc)))
            return np.full_like(baseline, 1000.)
    def jacobian(z):
        columns = []
        for i in range(10):
            lo, hi = z.copy(), z.copy()
            lo[i], hi[i] = max(-bound, z[i]-.0005), min(bound, z[i]+.0005)
            columns.append((objective(hi)-objective(lo))/(hi[i]-lo[i]))
        return np.column_stack(columns)
    fit = least_squares(objective, np.zeros(10), jac=jacobian, bounds=(-bound, bound),
        max_nfev=40, x_scale="jac", ftol=1e-5, xtol=1e-5, gtol=1e-5)
    final = response(basis@fit.x)
    return dict(schema="agentfem.research.spectrum-recovery-joint.v1", model_id=kernel.MODEL_ID,
        source_sha256=seed["source_sha256"], training_cycles=[1, 2, 3], static_powers=powers,
        coordinates=NAMES, derivative_audits=audits, relative_jacobian_difference=relative,
        subspace_distance=distance, maximum_log_coordinate_norm=.3, basis=basis.tolist(),
        rejected_trials=rejected, fit=dict(success=bool(fit.success), message=fit.message,
            cost=float(final@final/2), parameters=asdict(decode(seed["parameters"], basis@fit.x, powers)),
            coordinates=fit.x.tolist(), log_multipliers=(basis@fit.x).tolist(),
            active_bounds=fit.active_mask.tolist(), nfev=int(fit.nfev), residual_evaluations=calls),
        engineering_qualified=False, life_qualified=False,
        scope="Matched early-only research; identical budget and response gates; no late data objective")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nonlinear", action="store_true")
    args = parser.parse_args()
    raw = args.seed.read_bytes()
    names = ("p92_spectrum_recovery_joint.py", "p92_recoverable_spectrum.py", "p92_two_rate_reference.py",
             "p92_full_sensitivity.py", "p92_two_rate_controls.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    before = identities()
    result = identify(json.loads(raw), args.source_dir, (0., 1.) if args.nonlinear else (0., 0.))
    if identities() != before or args.seed.read_bytes() != raw:
        raise RuntimeError("Recovery comparison inputs changed during identification")
    result["driver_sha256"], result["seed_sha256"] = before, sha256(raw).hexdigest()
    write_new(args.output, result)

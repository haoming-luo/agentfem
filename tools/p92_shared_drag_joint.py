"""Audited early-only shared-drag identification; no conditional protocol cards."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tools import p92_shared_drag_reference as kernel
from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms

NAMES = ("flow_1", "flow_2", "hardening_1", "hardening_2", "hardening_3",
    "static_1", "static_2", "static_3", "drag_hardening", "drag_static",
    "dynamic_pair_1", "dynamic_pair_2", "dynamic_pair_3", "drag_dynamic_pair")


def decode(card, z):
    z = np.asarray(z, dtype=float)
    if z.shape != (14,) or not np.isfinite(z).all():
        raise ValueError("Require fourteen finite shared-drag coordinates")
    m = kernel.Material(**card)
    factors = 10**z
    return replace(m, flow_coefficients=tuple(np.asarray(m.flow_coefficients)*factors[:2]),
        hardening=tuple(np.asarray(m.hardening)*factors[2:5]),
        static_recovery=tuple(np.asarray(m.static_recovery)*factors[5:8]),
        drag_hardening=m.drag_hardening*factors[8], drag_static_recovery=m.drag_static_recovery*factors[9],
        dynamic_initial=tuple(np.asarray(m.dynamic_initial)*factors[10:13]),
        dynamic_final=tuple(np.asarray(m.dynamic_final)*factors[10:13]),
        drag_dynamic_initial=m.drag_dynamic_initial*factors[13],
        drag_dynamic_final=m.drag_dynamic_final*factors[13])


def identify(seed, source_dir):
    if (seed["training_cycles"] != [1, 2, 3] or seed["fitted"]
            or seed["model_id"] != kernel.MODEL_ID or seed["cost"] is None):
        raise ValueError("Require a feasible early-only shared-drag starting record")
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
        mat = decode(seed["parameters"], z)
        result = np.concatenate([replay(path, mat, kernel=kernel)[0] for path in paths])
        if calls % 10 == 1:
            print(json.dumps(dict(evaluation=calls, cost=float(result@result/2))), flush=True)
        return result

    baseline = response(np.zeros(14))
    jacobians, audits = [], []
    for step in (.001, .0005):
        columns = []
        for i in range(14):
            delta = np.eye(14)[i]*step
            columns.append((response(delta)-response(-delta))/(2*step))
        jac = np.column_stack(columns)
        _, values, vectors = np.linalg.svd(jac, full_matrices=False)
        jacobians.append(jac)
        audits.append(dict(step=step, singular_values=values.tolist(), right_vectors=vectors.tolist()))
    relative = float(np.linalg.norm(jacobians[1]-jacobians[0])/np.linalg.norm(jacobians[1]))
    va, vb = (np.asarray(a["right_vectors"])[:8] for a in audits)
    distance = float(np.linalg.norm(va.T@va-vb.T@vb, ord=2))
    leading = audits[1]["singular_values"][:8]
    if (not np.isfinite([relative, distance, *leading]).all() or relative > .002 or distance > .02
            or leading[-1] < 1e-5 or leading[0]/leading[-1] > 1e4):
        raise ValueError(f"Unresolved shared-drag sensitivity: relative={relative}, subspace={distance}")
    basis = vb.T
    bound = 1.2/np.sqrt(8)

    def objective(z):
        try:
            return response(basis@z)
        except (ValueError, RuntimeError) as exc:
            rejected.append(dict(coordinates=z.tolist(), reason=str(exc)))
            # Numerical-domain barrier only. Never report it as physical data.
            return np.full_like(baseline, 1000.)

    def jacobian(z):
        columns = []
        for i in range(8):
            lo, hi = z.copy(), z.copy()
            lo[i], hi[i] = max(-bound, z[i]-.0005), min(bound, z[i]+.0005)
            columns.append((objective(hi)-objective(lo))/(hi[i]-lo[i]))
        return np.column_stack(columns)

    fit = least_squares(objective, np.zeros(8), jac=jacobian, bounds=(-bound, bound),
        max_nfev=40, x_scale="jac", ftol=1e-5, xtol=1e-5, gtol=1e-5)
    final = response(basis@fit.x)  # Fresh strict replay, never the barrier.
    return dict(schema="agentfem.research.shared-drag-joint.v1", model_id=kernel.MODEL_ID,
        source_sha256=seed["source_sha256"], training_cycles=[1, 2, 3], coordinates=NAMES,
        derivative_audits=audits, relative_jacobian_difference=relative, subspace_distance=distance,
        maximum_log_coordinate_norm=1.2, basis=basis.tolist(), rejected_trials=rejected,
        fit=dict(success=bool(fit.success), message=fit.message, cost=float(final@final/2),
            parameters=asdict(decode(seed["parameters"], basis@fit.x)), coordinates=fit.x.tolist(),
            log_multipliers=(basis@fit.x).tolist(), active_bounds=fit.active_mask.tolist(),
            nfev=int(fit.nfev), residual_evaluations=calls), engineering_qualified=False,
        scope="Early joint shared-drag research; later cycles excluded; unchanged response gates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.seed.read_bytes()
    names = ("p92_shared_drag_joint.py", "p92_shared_drag_reference.py", "p92_two_rate_controls.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    before = identities()
    result = identify(json.loads(raw), args.source_dir)
    if identities() != before or args.seed.read_bytes() != raw:
        raise RuntimeError("Identification code or seed changed during execution")
    result["driver_sha256"], result["seed_sha256"] = before, sha256(raw).hexdigest()
    write_new(args.output, result)

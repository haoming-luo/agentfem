"""Bounded joint early-cycle reference identification after local rank screening."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_probe import bridge_seed
from tools.p92_waveform_controls import attach_waveforms


def decode(base, z):
    z = np.asarray(z, dtype=float)
    if z.shape != (6,) or not np.isfinite(z).all():
        raise ValueError("Require six finite joint coordinates")
    a, b = base.branches
    a = replace(a, strength=a.strength+50*z[0], reference_flow=a.reference_flow*10**z[1],
                exponent=a.exponent+z[2], saturation=(a.saturation[0], a.saturation[1]*10**z[3]))
    b = replace(b, static_recovery=tuple(r*10**d for r, d in zip(b.static_recovery, z[4:], strict=True)))
    return replace(base, branches=(a, b))


def identify(seed, source_dir):
    base = bridge_seed(seed)
    paths = []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != seed["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths.append(attach_waveforms(measured_path(source, cycles=3), source))
    calls = 0

    def objective(z):
        nonlocal calls
        calls += 1
        card = decode(base, z)
        residual = np.concatenate([replay(path, card)[0] for path in paths])
        if calls % 10 == 1:
            print(json.dumps(dict(evaluation=calls, cost=float(.5*residual@residual))), flush=True)
        return residual

    def jacobian(z):
        return np.column_stack([(objective(z+np.eye(6)[i]*.001)-
                                 objective(z-np.eye(6)[i]*.001))/.002 for i in range(6)])

    initial_jacobian = jacobian(np.zeros(6))
    singular = np.linalg.svd(initial_jacobian, compute_uv=False)
    result = dict(schema="agentfem.research.two-rate-joint.v1", training_cycles=[1, 2, 3],
        source_sha256=seed["source_sha256"], seed_parameter_sha256=seed["parameter_sha256"],
        initial_singular_values=singular.tolist(),
        coordinate_names=["threshold_A_increment_per_50MPa", "log10_reference_flow_A",
            "exponent_A_increment", "log10_saturation_A2_at_fixed_H2", "log10_recovery_B1", "log10_recovery_B2"],
        scope="Joint early response identification; no memory term or later-cycle objective",
        engineering_qualified=False, life_qualified=False)
    # Screening threshold prespecified for these dimensionless coordinates.
    # This is a numerical rank gate, not statistical identifiability proof.
    if singular[-1] < 1e-5 or singular[0]/singular[-1] > 1e4:
        result["conditional_fit"] = dict(success=False, message="Initial coordinate rank screen failed")
        return result
    bounds = ([(30-base.branches[0].strength)/50, -.5, -3., -.3, -1., -1.],
              [(160-base.branches[0].strength)/50, .5, 2., .3, 1., 1.])
    fit = least_squares(objective, np.zeros(6), jac=jacobian, bounds=bounds,
                        max_nfev=15, x_scale="jac", ftol=1e-5, xtol=1e-5, gtol=1e-5)
    card = asdict(decode(base, fit.x))
    result["conditional_fit"] = dict(success=bool(fit.success), cost=float(fit.cost),
        coordinates=fit.x.tolist(), active_bounds=fit.active_mask.tolist(), nfev=int(fit.nfev),
        residual_evaluations=calls, message=fit.message, parameters=card, bounds=bounds,
        singular_values=np.linalg.svd(fit.jac, compute_uv=False).tolist(),
        parameter_sha256=sha256(json.dumps(card, sort_keys=True).encode()).hexdigest())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = identify(json.loads(args.seed.read_text()), args.source_dir)
    write_new(args.output, result)
    print(json.dumps(result))

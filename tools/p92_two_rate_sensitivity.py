"""Early-only component recovery discriminator for the verified scalar reference."""

import argparse
from dataclasses import asdict, replace
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


def perturb(material, component, log_scale):
    if component not in range(4) or not np.isfinite(log_scale):
        raise ValueError("Require one of four finite recovery directions")
    branches = list(material.branches)
    branch_index, index = divmod(component, 2)
    values = list(branches[branch_index].static_recovery)
    values[index] *= 10.**log_scale
    branches[branch_index] = replace(branches[branch_index], static_recovery=tuple(values))
    return replace(material, branches=tuple(branches))


def discriminate(seed, source_dir, *, fit_recovery=False, fit_flow=False):
    if fit_recovery and fit_flow:
        raise ValueError("Separate conditional experiments; do not combine parameter searches")
    material = bridge_seed(seed)
    paths = []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != seed["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths.append(attach_waveforms(measured_path(source, cycles=3), source))

    def objective(card):
        return np.concatenate([replay(path, card)[0] for path in paths])

    base = objective(material)
    records, directions = [], []
    for index in range(4):
        minus = objective(perturb(material, index, -1.))
        plus = objective(perturb(material, index, 1.))
        direction = (plus-minus)/2
        directions.append(direction)
        records.append(dict(component=index, minus_cost=float(.5*minus@minus),
            plus_cost=float(.5*plus@plus), response_norm=float(np.linalg.norm(direction)),
            curvature_norm=float(np.linalg.norm(plus+minus-2*base))))
    result = dict(schema="agentfem.research.two-rate-recovery-discriminator.v1",
        training_cycles=[1, 2, 3], parameters=asdict(material),
        source_sha256=seed["source_sha256"], seed_parameter_sha256=seed["parameter_sha256"],
        base_cost=float(.5*base@base), directions=records,
        secant_singular_values=np.linalg.svd(np.column_stack(directions), compute_uv=False).tolist(),
        scope="One-decade secants are exploratory, not a local covariance or uncertainty estimate",
        fitted=False, engineering_qualified=False, life_qualified=False)
    if fit_recovery or fit_flow:
        # Only the two observable branch-B directions; branch A stays fixed.
        def decode(z):
            if fit_flow:
                return replace(material, branches=(replace(material.branches[0],
                    strength=material.branches[0].strength+z[0]*50.,
                    reference_flow=material.branches[0].reference_flow*10.**z[1]), material.branches[1]))
            return perturb(perturb(material, 2, z[0]), 3, z[1])

        def residual(z):
            return objective(decode(z))

        def jacobian(z):
            return np.column_stack([(residual(z+np.eye(2)[i]*.001)-
                                     residual(z-np.eye(2)[i]*.001))/.002 for i in range(2)])

        bounds = ([(30.-material.branches[0].strength)/50., -1.],
                  [(160.-material.branches[0].strength)/50., 1.]) if fit_flow else ([-1., -1.], [1., 1.])
        fit = least_squares(residual, np.zeros(2), jac=jacobian, bounds=bounds,
                            max_nfev=20 if fit_flow else 15, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        result["conditional_fit"] = dict(success=bool(fit.success), cost=float(fit.cost),
            coordinates=fit.x.tolist(), active_bounds=fit.active_mask.tolist(), nfev=int(fit.nfev),
            message=fit.message, parameters=asdict(decode(fit.x)),
            scope=("Only branch-A threshold and reference flow" if fit_flow else "Only branch-B recovery")+
                  "; early three protocols; not admitted material", bounds=bounds)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fit-recovery", action="store_true")
    parser.add_argument("--fit-flow", action="store_true")
    args = parser.parse_args()
    result = discriminate(json.loads(args.seed.read_text()), args.source_dir,
                          fit_recovery=args.fit_recovery, fit_flow=args.fit_flow)
    write_new(args.output, result)
    print(json.dumps(result))

"""Joint early-protocol identification of the recoverable-strain hypothesis."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_gate_objective import residuals as gate_residuals
from tools.p92_response_screen import screen
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools import p92_recoverable_reference as kernel
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_joint import decode as decode_backbone
from tools.p92_two_rate_reference import Branch, Material
from tools.p92_waveform_controls import attach_waveforms


def decode(base, z):
    z = np.asarray(z, dtype=float)
    if z.shape != (8,) or not np.isfinite(z).all():
        raise ValueError("Require eight finite joint coordinates")
    return kernel.Material(decode_backbone(base, z[:6]),
                           base.modulus*20*10**z[6], .2*10**z[7])


def identify(frozen, source_dir, *, gates=False, without_recoverable=False):
    if without_recoverable and not gates:
        raise ValueError("Backbone comparison requires the matched gate objective")
    card = frozen["parameters"]
    base = Material(card["modulus"], tuple(Branch(**b) for b in card["branches"]))
    paths, sources = [], []
    for protocol in ("RF", "SLR", "SLC"):
        data = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if data["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths.append(attach_waveforms(measured_path(data, cycles=3), data))
        sources.append(data)
    calls = 0
    dimensions = 6 if without_recoverable else 8

    def material(z):
        return decode_backbone(base, z) if without_recoverable else decode(base, z)

    def objective(z):
        nonlocal calls
        calls += 1
        blocks = []
        for path, source in zip(paths, sources, strict=True):
            raw, prediction = (replay(path, material(z)) if without_recoverable else
                               replay(path, material(z), kernel=kernel))
            if gates:
                comparison = compare_candidate(source, prediction)
                block, _ = gate_residuals(screen(prediction, comparison, requested_cycles=3))
                blocks.append(block)
            else:
                blocks.append(raw)
        residual = np.concatenate(blocks)
        if calls % 10 == 1:
            print(json.dumps(dict(evaluation=calls, cost=float(residual@residual/2))), flush=True)
        return residual

    def jacobian(z):
        return np.column_stack([(objective(z+np.eye(dimensions)[i]*.001)-
                                 objective(z-np.eye(dimensions)[i]*.001))/.002 for i in range(dimensions)])

    singular = np.linalg.svd(jacobian(np.zeros(dimensions)), compute_uv=False)
    result = dict(schema="agentfem.research.recoverable-joint.v1", training_cycles=[1, 2, 3],
        source_sha256=frozen["source_sha256"], initial_singular_values=singular.tolist(),
        coordinate_names=["threshold_A_per_50MPa", "log10_flow_A", "exponent_A",
            "log10_saturation_A2", "log10_recovery_B1", "log10_recovery_B2",
            "log10_recoverable_modulus", "log10_relaxation_time"][:dimensions],
        gate_objective=gates, without_recoverable=without_recoverable,
        scope="All three early protocols; no protocol-specific parameters, later data or reset",
        engineering_qualified=False, life_qualified=False)
    if singular[-1] < 1e-5 or singular[0]/singular[-1] > 1e4:
        result["fit"] = dict(success=False, message="Initial coordinate rank screen failed")
        return result
    a = base.branches[0]
    bounds = ([(1-a.strength)/50, -.5, 2-a.exponent, -.3, -1., -1., -.6, -1.],
              [(160-a.strength)/50, .5, 10-a.exponent, .3, 1., 1., .6, 1.])
    bounds = tuple(b[:dimensions] for b in bounds)
    fit = least_squares(objective, np.zeros(dimensions), jac=jacobian, bounds=bounds,
                        max_nfev=20, x_scale="jac", ftol=1e-5, xtol=1e-5, gtol=1e-5)
    result["fit"] = dict(success=bool(fit.success), message=fit.message, cost=float(fit.cost),
        parameters=asdict(material(fit.x)), coordinates=fit.x.tolist(),
        active_bounds=fit.active_mask.tolist(), bounds=bounds, nfev=int(fit.nfev),
        residual_evaluations=calls, singular_values=np.linalg.svd(fit.jac, compute_uv=False).tolist())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gate-objective", action="store_true")
    parser.add_argument("--without-recoverable", action="store_true")
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    result = identify(json.loads(raw), args.source_dir, gates=args.gate_objective,
                      without_recoverable=args.without_recoverable)
    result["frozen_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)

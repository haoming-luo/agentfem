"""Bounded joint identification in audited response-sensitive directions.

This is a local research discriminator, not a material qualification or a
statistical identifiability certificate. Weak singular directions stay fixed.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tools.p92_bam_source import read_source
from tools.p92_full_sensitivity import decode
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools import p92_recoverable_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms


def sensitive_basis(audit, dimensions=8):
    if audit["training_cycles"] != [1, 2, 3] or audit["fitted"]:
        raise ValueError("Require early-only unfitted sensitivity audit")
    difference = audit["relative_jacobian_difference"]
    if (type(dimensions) is not int or not 1 <= dimensions <= 8
            or not np.isfinite(difference) or not 0 <= difference <= .002):
        raise ValueError("Unresolved derivative audit or unsupported dimension")
    a, b = audit["audits"]
    va = np.asarray(a["right_vectors"], dtype=float)[:dimensions]
    vb = np.asarray(b["right_vectors"], dtype=float)[:dimensions]
    if va.shape != (dimensions, 20) or vb.shape != va.shape:
        raise ValueError("Unexpected sensitivity coordinates")
    if any(not np.isfinite(v).all() or not np.allclose(v@v.T, np.eye(dimensions), atol=1e-8)
           for v in (va, vb)):
        raise ValueError("Sensitivity basis must be finite and orthonormal")
    distance = float(np.linalg.norm(va.T@va-vb.T@vb, ord=2))
    if distance > .02:
        raise ValueError("Sensitive subspace is not stable under difference refinement")
    return vb.T, distance


def identify(audit, source_dir, *, spectrum_card=None):
    basis, distance = sensitive_basis(audit)
    card = audit["parameters"] if spectrum_card is None else spectrum_card["parameters"]
    if spectrum_card is not None and spectrum_card["source_sha256"] != audit["source_sha256"]:
        raise ValueError("Spectrum seed and sensitivity audit use different sources")
    module = kernel if spectrum_card is None else spectrum
    dimensions = 8 if spectrum_card is None else 10
    paths = []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != audit["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths.append(attach_waveforms(measured_path(source, cycles=3), source))
    calls = 0

    def material(z):
        mat = decode(card, basis@z[:8])
        if spectrum_card is None:
            return mat
        if mat.strength_ratio != 1.:
            raise ValueError("Spectrum seed cannot discard strength evolution")
        return spectrum.Material(mat.backbone,
            (mat.recoverable_modulus, 20*mat.backbone.modulus*10**z[8]),
            (mat.relaxation_time, 200*10**z[9]))

    def objective(z):
        nonlocal calls
        calls += 1
        mat = material(z)
        residual = np.concatenate([replay(path, mat, kernel=module)[0] for path in paths])
        if calls % 10 == 1:
            print(json.dumps(dict(evaluation=calls, cost=float(residual@residual/2))), flush=True)
        return residual

    def jacobian(z):
        return np.column_stack([(objective(z+np.eye(dimensions)[i]*.0005)-
                                 objective(z-np.eye(dimensions)[i]*.0005))/.001 for i in range(dimensions)])

    # Orthonormal coordinates and this box imply ||log10 multiplier||_2 <= .3:
    # no individual positive coefficient can move by more than a factor two.
    bound = .3/np.sqrt(8)
    bounds = ([-bound]*8+([-.5, -.5] if dimensions == 10 else []),
              [bound]*8+([.5, .5] if dimensions == 10 else []))
    initial = np.zeros(dimensions)
    singular = np.linalg.svd(jacobian(initial), compute_uv=False)
    if singular[-1] < 1e-5 or singular[0]/singular[-1] > 1e4:
        raise ValueError("Joint sensitive-spectrum derivative rank screen failed")
    fit = least_squares(objective, initial, jac=jacobian, bounds=bounds,
                        max_nfev=20, x_scale="jac", ftol=1e-5, xtol=1e-5, gtol=1e-5)
    return dict(schema="agentfem.research.sensitive-joint.v1", training_cycles=[1, 2, 3],
        source_sha256=audit["source_sha256"], basis=basis.tolist(),
        model_id=module.MODEL_ID, initial_singular_values=singular.tolist(), bounds=bounds,
        subspace_distance=distance, maximum_log_coordinate_norm=.3,
        fit=dict(success=bool(fit.success), message=fit.message, cost=float(fit.cost),
            parameters=asdict(material(fit.x)),
            coordinates=fit.x.tolist(), log_multipliers=(basis@fit.x[:8]).tolist(),
            active_bounds=fit.active_mask.tolist(), nfev=int(fit.nfev), residual_evaluations=calls),
        engineering_qualified=False, life_qualified=False,
        scope="Fixed early-only sensitive basis; shared parameters; no new mechanism or cycle reset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spectrum-seed", type=Path)
    args = parser.parse_args()
    raw = args.audit.read_bytes()
    seed = args.spectrum_seed.read_bytes() if args.spectrum_seed else None
    result = identify(json.loads(raw), args.source_dir,
                      spectrum_card=json.loads(seed) if seed else None)
    if seed:
        result["spectrum_seed_sha256"] = sha256(seed).hexdigest()
    result["audit_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)

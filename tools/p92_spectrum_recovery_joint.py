"""Matched early-only identification of linear/nonlinear static recovery.

One card per hypothesis, shared by all loading protocols. Never a life card.
"""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tools import p92_recoverable_spectrum as kernel
from tools import p92_primary_creep_reference as primary_kernel
from tools import p92_transient_strength_reference as transient_kernel
from tools import p92_recoverable_mobility_reference as mobility_kernel
from tools import p92_critical_recovery_reference as critical_kernel
from tools import p92_rate_recovery_reference as rate_kernel
from tools.p92_full_sensitivity import decode as decode_backbone, NAMES as BASE_NAMES
from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms
from tools.p92_fit_subspace_audit import replacement_basis_from_svd

NAMES = (*BASE_NAMES, "recoverable.slow_modulus", "recoverable.slow_time")


def decode_history(card, z):
    z = np.asarray(z, dtype=float)
    if z.shape != (22,) or not np.isfinite(z).all():
        raise ValueError("Require 22 finite substituted coordinates")
    backbone_z = z.copy()
    backbone_z[7:9] = 0.
    base = decode(card, backbone_z, (0., 0.))
    ratio = 1/(1+10**(-z[7]))
    rate = 250.*10**z[8]
    return replace(base, static_ratios=(ratio, ratio),
                   static_history_rates=(rate, rate), static_history="shared")


def history_identifiability(matrix, basis):
    other = np.delete(matrix, (7, 8), axis=1)
    u, _, _ = np.linalg.svd(other, full_matrices=False)
    columns = matrix[:, [7, 8]]
    residual = columns-u[:, :8]@(u[:, :8].T@columns)
    values = np.linalg.svd(residual, compute_uv=False)
    gram = basis[[7, 8]]@basis[[7, 8]].T
    eigenvalues = np.linalg.eigvalsh(gram)
    ratio = float(values[-1]/values[0]) if values[0] > 0 else 0.
    return dict(residualized_singular_values=values.tolist(), residualized_singular_ratio=ratio,
        retained_coordinate_gram_eigenvalues=eigenvalues.tolist(),
        column_norms=np.linalg.norm(columns, axis=0).tolist(),
        passed=bool(np.isfinite([*values, *eigenvalues, ratio]).all()
                    and eigenvalues[0] >= 1e-4 and ratio >= 1e-3),
        scope="Local scaled numerical observability, not statistical identifiability")


def decode_recovery_shape(card, z):
    z = np.asarray(z, dtype=float)
    if z.shape != (22,) or not np.isfinite(z).all():
        raise ValueError("Require 22 finite substituted coordinates")
    backbone_z = z.copy()
    backbone_z[7:9] = 0.
    material = decode(card, backbone_z, (0., 0.))
    a, b = material.backbone.branches
    b = replace(b, ratcheting=tuple(np.asarray(b.ratcheting)*10.**z[7:9]))
    return replace(material, backbone=replace(material.backbone, branches=(a, b)))


def decode_kinetics(card, z):
    z = np.asarray(z, dtype=float)
    if z.shape != (22,) or not np.isfinite(z).all():
        raise ValueError("Require 22 finite substituted coordinates")
    backbone_z = z.copy()
    backbone_z[7:9] = 0.
    return replace(decode(card, backbone_z, (0., 0.)),
                   recoverable_powers=tuple(1.+10.**z[7:9]))


def decode_transient(card, z):
    z = np.asarray(z, dtype=float)
    if z.shape != (22,) or not np.isfinite(z).all():
        raise ValueError("Require 22 finite substituted coordinates")
    backbone_z = z.copy()
    backbone_z[7:9] = 0.
    return transient_kernel.Material(decode(card, backbone_z, (0., 0.)),
        maximum=.1, build=250.*10.**z[7], recovery=10.**z[8]/1800.)


def decode(card, z, powers):
    z = np.asarray(z, dtype=float)
    if z.shape != (22,) or not np.isfinite(z).all():
        raise ValueError("Require 22 finite logarithmic coordinates")
    if card.get("directional_recovery", False) is not False:
        raise ValueError("Require magnitude-recovery baseline seed")
    if any(card.get(name, default) != default and tuple(card.get(name, default)) != default
           for name, default in (("strength_ratios", (1., 1.)), ("hardening_ratios", (1., 1.)),
                                 ("static_powers", (0., 0.)), ("static_ratios", (1., 1.)),
                                 ("recoverable_powers", (1., 1.)))):
        raise ValueError("Require an unevolved baseline spectrum")
    base = decode_backbone(dict(backbone=card["backbone"],
        recoverable_modulus=card["moduli"][0], relaxation_time=card["times"][0]), z[:20])
    return kernel.Material(base.backbone,
        (base.recoverable_modulus, card["moduli"][1]*10**z[20]),
        (base.relaxation_time, card["times"][1]*10**z[21]), static_powers=tuple(powers))


def identify(seed, source_dir, powers, *, shared_history=False, directional=False, primary=False,
             recoverable_kinetics=False, transient_strength=False, replacement_space=False,
             recoverable_mobility=False, critical_recovery=False, recovery_shape=False,
             rate_recovery=False):
    if seed["model_id"] != kernel.MODEL_ID or tuple(powers) not in {(0., 0.), (0., 1.)}:
        raise ValueError("Require predeclared matched recovery hypotheses")
    if type(rate_recovery) is not bool or (rate_recovery and (
            shared_history or directional or primary or recoverable_kinetics or transient_strength
            or replacement_space or recoverable_mobility or critical_recovery or recovery_shape
            or tuple(powers) != (0., 0.))):
        raise ValueError("Rate recovery must isolate a parameter-neutral mechanism")
    if (type(recovery_shape) is not bool or (recovery_shape and (
            shared_history or directional or primary or recoverable_kinetics or transient_strength
            or replacement_space or recoverable_mobility or critical_recovery or tuple(powers) != (0., 0.)))):
        raise ValueError("Recovery shape must isolate existing exponent substitution")
    if type(replacement_space) is not bool or (replacement_space and not shared_history):
        raise ValueError("Explicit replacement geometry is limited to shared-history comparison")
    if type(shared_history) is not bool or (shared_history and tuple(powers) != (0., 0.)):
        raise ValueError("History comparison requires linear static powers and explicit boolean")
    decoder = decode_history if shared_history else lambda card, z: decode(card, z, powers)
    if type(directional) is not bool or (directional and (shared_history or tuple(powers) != (0., 0.))):
        raise ValueError("Directional comparison must isolate one mechanism")
    if directional:
        decoder = lambda card, z: replace(decode(card, z, powers), directional_recovery=True)
    if type(primary) is not bool or (primary and (directional or shared_history or tuple(powers) != (0., 0.))):
        raise ValueError("Primary-creep comparison must isolate one mechanism")
    if (type(recoverable_kinetics) is not bool or
            (recoverable_kinetics and (primary or directional or shared_history or tuple(powers) != (0., 0.)))):
        raise ValueError("Recoverable-kinetics comparison must isolate one mechanism")
    if recoverable_kinetics:
        decoder = decode_kinetics
    execution_kernel = primary_kernel if primary else kernel
    if (type(transient_strength) is not bool or
            (transient_strength and (primary or directional or shared_history or recoverable_kinetics
                                     or tuple(powers) != (0., 0.)))):
        raise ValueError("Transient-strength comparison must isolate one mechanism")
    if transient_strength:
        decoder, execution_kernel = decode_transient, transient_kernel
    if (type(recoverable_mobility) is not bool or
            (recoverable_mobility and (primary or directional or shared_history or recoverable_kinetics
                                       or transient_strength or tuple(powers) != (0., 0.)))):
        raise ValueError("Recoverable mobility must isolate one mechanism")
    if recoverable_mobility:
        decoder = lambda card, z: mobility_kernel.Material(decode(card, z, powers), "static_recovery")
        execution_kernel = mobility_kernel
    if (type(critical_recovery) is not bool or
            (critical_recovery and (primary or directional or shared_history or recoverable_kinetics
                                   or transient_strength or recoverable_mobility or tuple(powers) != (0., 0.)))):
        raise ValueError("Critical recovery must isolate one mechanism")
    if critical_recovery:
        decoder = lambda card, z: critical_kernel.Material(decode(card, z, powers))
        execution_kernel = critical_kernel
    if recovery_shape:
        decoder = decode_recovery_shape
    if rate_recovery:
        decoder = lambda card, z: rate_kernel.Material(decode(card, z, powers))
        execution_kernel = rate_kernel
    coordinates = list(NAMES)
    if primary:
        coordinates[20:] = ["primary.hardening_scale", "primary.time_scale"]
    if shared_history:
        coordinates[7:9] = ["shared.static_fraction_log10_odds", "shared.static_history_rate"]
    if recoverable_kinetics:
        coordinates[7:9] = ["recoverable.fast_power_minus_one", "recoverable.slow_power_minus_one"]
    if transient_strength:
        coordinates[7:9] = ["transient.strength_build", "transient.strength_recovery"]
    if recovery_shape:
        coordinates[7:9] = ["B.ratcheting1", "B.ratcheting2"]
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
        mat = decoder(seed["parameters"], z)
        result = np.concatenate([replay(path, mat, kernel=execution_kernel)[0] for path in paths])
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
    basis, restricted = vb.T, None
    if replacement_space:
        pairs = [replacement_basis_from_svd(a["singular_values"], a["right_vectors"])
                 for a in audits]
        basis = pairs[1][0]
        restricted = [pair[1] for pair in pairs]
        distance = float(np.linalg.norm(pairs[0][0]@pairs[0][0].T-basis@basis.T, ord=2))
        leading = restricted[1]["singular_values"]
    if (not np.isfinite([relative, distance, *leading]).all() or relative > .002 or distance > .02
            or leading[-1] < 1e-5 or leading[0]/leading[-1] > 1e4
            or (restricted is not None and not all(a["passed"] for a in restricted))):
        return dict(schema="agentfem.research.spectrum-recovery-joint.v1", model_id=execution_kernel.MODEL_ID,
            replacement_space=replacement_space, restricted_derivative_audits=restricted,
            source_sha256=seed["source_sha256"], training_cycles=[1, 2, 3],
            coordinates=coordinates, derivative_audits=audits,
            relative_jacobian_difference=relative, subspace_distance=distance,
            fit=dict(success=False, message="Unresolved derivative audit; no optimization",
                     residual_evaluations=calls), engineering_qualified=False, life_qualified=False)
    bound = .3/np.sqrt(10)
    history_audit = (history_identifiability(matrices[1], basis)
                     if shared_history or recoverable_kinetics or transient_strength or recovery_shape else None)
    if history_audit is not None and not history_audit["passed"]:
        return dict(schema="agentfem.research.spectrum-recovery-joint.v1", model_id=execution_kernel.MODEL_ID,
            replacement_space=replacement_space, restricted_derivative_audits=restricted,
            source_sha256=seed["source_sha256"], training_cycles=[1, 2, 3],
            coordinates=coordinates, derivative_audits=audits, history_audit=history_audit,
            relative_jacobian_difference=relative, subspace_distance=distance,
            fit=dict(success=False, message="Unresolved history directions; no optimization",
                     residual_evaluations=calls), engineering_qualified=False, life_qualified=False)
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
    return dict(schema="agentfem.research.spectrum-recovery-joint.v1", model_id=execution_kernel.MODEL_ID,
        replacement_space=replacement_space, restricted_derivative_audits=restricted,
        source_sha256=seed["source_sha256"], training_cycles=[1, 2, 3], static_powers=powers,
        coordinates=coordinates, derivative_audits=audits, relative_jacobian_difference=relative,
        shared_history=shared_history, history_audit=history_audit,
        subspace_distance=distance, maximum_log_coordinate_norm=.3, basis=basis.tolist(),
        rejected_trials=rejected, fit=dict(success=bool(fit.success), message=fit.message,
            cost=float(final@final/2), parameters=asdict(decoder(seed["parameters"], basis@fit.x)),
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
    parser.add_argument("--shared-history", action="store_true")
    parser.add_argument("--directional", action="store_true")
    parser.add_argument("--primary", action="store_true")
    parser.add_argument("--recoverable-kinetics", action="store_true")
    parser.add_argument("--transient-strength", action="store_true")
    parser.add_argument("--replacement-space", action="store_true")
    parser.add_argument("--recoverable-mobility", action="store_true")
    parser.add_argument("--critical-recovery", action="store_true")
    parser.add_argument("--recovery-shape", action="store_true")
    parser.add_argument("--rate-recovery", action="store_true")
    args = parser.parse_args()
    raw = args.seed.read_bytes()
    names = ("p92_spectrum_recovery_joint.py", "p92_recoverable_spectrum.py", "p92_two_rate_reference.py",
             "p92_full_sensitivity.py", "p92_two_rate_controls.py", "p92_bam_source.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_measured_fit.py",
             "p92_primary_creep_reference.py", "p92_transient_strength_reference.py")
    names += ("p92_fit_subspace_audit.py", "p92_recoverable_reference.py", "p92_hold_observations.py")
    names += ("p92_recoverable_mobility_reference.py", "p92_critical_recovery_reference.py")
    names += ("p92_rate_recovery_reference.py",)
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    before = identities()
    result = identify(json.loads(raw), args.source_dir, (0., 1.) if args.nonlinear else (0., 0.),
                      shared_history=args.shared_history, directional=args.directional, primary=args.primary,
                      recoverable_kinetics=args.recoverable_kinetics,
                      transient_strength=args.transient_strength, replacement_space=args.replacement_space,
                      recoverable_mobility=args.recoverable_mobility, critical_recovery=args.critical_recovery,
                      recovery_shape=args.recovery_shape, rate_recovery=args.rate_recovery)
    if identities() != before or args.seed.read_bytes() != raw:
        raise RuntimeError("Recovery comparison inputs changed during identification")
    result["driver_sha256"], result["seed_sha256"] = before, sha256(raw).hexdigest()
    result["directional_recovery"] = args.directional
    result["primary_creep"] = args.primary
    result["recoverable_kinetics"] = args.recoverable_kinetics
    result["transient_strength"] = args.transient_strength
    result["recoverable_mobility"] = args.recoverable_mobility
    result["critical_recovery"] = args.critical_recovery
    result["recovery_shape"] = args.recovery_shape
    result["rate_recovery"] = args.rate_recovery
    write_new(args.output, result)

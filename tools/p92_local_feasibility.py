"""One bounded linearized gate tradeoff, followed by real path checks."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from tools import p92_recoverable_spectrum as spectrum
from tools import p92_recoverable_mobility_reference as mobility
from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_fit_subspace_audit import audit as audit_basis
from tools.p92_gate_objective import residuals as gate_residuals
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_response_screen import screen
from tools.p92_recoverable_frozen import decode_material
from tools.p92_spectrum_recovery_joint import decode
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms


def same_card(a, b):
    """Permit matrix-product roundoff only, not a different material card."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_card(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(
            same_card(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, (float, int)) and not isinstance(a, bool):
        return isinstance(b, (float, int)) and bool(np.isclose(a, b, rtol=64*np.finfo(float).eps, atol=0.))
    return a == b


def linear_step(values, jacobian, coordinates, bound):
    g, j, u = (np.asarray(a, dtype=float) for a in (values, jacobian, coordinates))
    if (g.ndim != 1 or not len(g) or u.ndim != 1 or not len(u)
            or j.shape != (len(g), len(u)) or not all(np.isfinite(a).all() for a in (g, j, u))
            or np.any(g < 0) or not np.isfinite(bound) or bound <= 0 or np.any(abs(u) > bound+1e-12)):
        raise ValueError("Require finite normalized errors, Jacobian and bounded coordinates")
    result = linprog(np.r_[np.zeros(len(u)), 1.],
        A_ub=np.column_stack([j, -np.ones(len(g))]), b_ub=-g,
        bounds=[(-bound-x, bound-x) for x in u]+[(0., None)], method="highs")
    if not result.success:
        raise RuntimeError(f"Linearized diagnostic failed: {result.message}")
    return result.x[:-1], float(result.x[-1])


def signed_step(values, jacobian, coordinates, bound):
    """Linearize signed errors; impose both sides of the same absolute gate."""
    r, j = np.asarray(values, dtype=float), np.asarray(jacobian, dtype=float)
    u = np.asarray(coordinates, dtype=float)
    if (r.ndim != 1 or not len(r) or j.shape != (len(r), len(u))
            or not all(np.isfinite(a).all() for a in (r, j, u))
            or not np.isfinite(bound) or bound <= 0 or np.any(abs(u) > bound+1e-12)):
        raise ValueError("Require finite signed errors and bounded coordinates")
    both = np.vstack((j, -j))
    result = linprog(np.r_[np.zeros(len(u)), 1.],
        A_ub=np.column_stack((both, -np.ones(2*len(r)))), b_ub=-np.r_[r, -r],
        bounds=[(-bound-x, bound-x) for x in u]+[(0., None)], method="highs")
    if not result.success:
        raise RuntimeError(f"Signed linearized diagnostic failed: {result.message}")
    return result.x[:-1], float(result.x[-1])


def signed_errors(prediction, comparison, keys, magnitudes):
    """Restore signs before differentiation, without changing any magnitude."""
    cycles = {row["cycle"]: row for row in comparison["records"]}
    holds = {(h["cycle"], h["segment"]): h for h in prediction["hold_diagnostics"]}
    values = []
    for (name, cycle, segment), magnitude in zip(keys, magnitudes, strict=True):
        if name in {"peak_stress", "valley_stress"}:
            raw = cycles[cycle][name.split("_")[0]+"_error_MPa"]
        elif name.endswith("_rms"):
            raw = magnitude  # Norm remains one-sided; its negative side is redundant.
        elif name.endswith("_last"):
            field = holds[cycle, segment]["fields"][name[len("hold_"):-len("_last")]]
            raw = field["last_predicted"]-field["last_observed"]
        elif name.endswith("_secant"):
            window, field_name = name.split("_", 1)
            field = holds[cycle, segment]["fields"][field_name[:-len("_secant")]]
            raw = field[f"{window}_secant_predicted_per_s"]-field[f"{window}_secant_observed_per_s"]
        else:
            raise ValueError(f"Unknown signed response metric: {name}")
        if not np.isfinite(raw) or (raw == 0 and magnitude != 0):
            raise ValueError("Signed metric and gate magnitude disagree")
        values.append(float(np.sign(raw)*magnitude))
    return np.asarray(values)


def admissible(old, new, protocols):
    old, new = np.asarray(old), np.asarray(new)
    if old.shape != new.shape or old.ndim != 1 or len(protocols) != len(old):
        raise ValueError("Require aligned diagnostic vectors")
    return bool(np.max(new) < np.max(old) and np.all(new[old <= 1.] <= 1.) and all(
        np.max(new[np.asarray(protocols) == p]) < np.max(old[np.asarray(protocols) == p])
        for p in set(protocols)))


def evaluate(fit, seed, source_dir, *, signed=False):
    audit_basis(fit)
    if (not fit["fit"]["success"] or fit["training_cycles"] != [1, 2, 3]
            or fit["model_id"] not in {spectrum.MODEL_ID, mobility.MODEL_ID}
            or fit.get("shared_history", False) or tuple(fit["static_powers"]) != (0., 0.)):
        raise ValueError("Require original or mobility matched early-only fit")
    basis, center = np.asarray(fit["basis"]), np.asarray(fit["fit"]["coordinates"])
    if basis.shape != (22, 10) or fit["maximum_log_coordinate_norm"] != .3:
        raise ValueError("Keep the original matched geometry")
    bound = .3/np.sqrt(10)
    module = mobility if fit["model_id"] == mobility.MODEL_ID else spectrum
    def material(u):
        base = decode(seed["parameters"], basis@u, (0., 0.))
        return mobility.Material(base, "static_recovery") if module is mobility else base
    _, frozen_material = decode_material(fit["fit"]["parameters"], fit["model_id"])
    if not same_card(asdict(material(center)), asdict(frozen_material)):
        raise ValueError("Seed/coordinates do not reproduce frozen parameters")
    sources, paths = [], []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != fit["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        sources.append(source)
        paths.append(attach_waveforms(measured_path(source, cycles=3), source))
    calls = 0
    def response(u):
        nonlocal calls
        calls += 1
        values, keys, unresolved = [], [], []
        for source, path in zip(sources, paths, strict=True):
            _, prediction = replay(path, material(u), kernel=module)
            comparison = compare_candidate(source, prediction)
            report = screen(prediction, comparison, requested_cycles=3)
            squared, meta = gate_residuals(report)
            magnitudes = np.sqrt(squared)
            values.extend(signed_errors(prediction, comparison, meta["keys"], magnitudes)
                          if signed else magnitudes)
            keys.extend((source["protocol"], *k) for k in meta["keys"])
            unresolved.extend((source["protocol"], *k) for k in meta["unresolved"])
            for seg, observed in zip(prediction["segments"], path["segments"], strict=True):
                if observed["strain_hold"]:
                    value = observed["measured_stress"][0]
                    error = (seg["sampled_stress_MPa"][0]-value)/max(5., .05*abs(value))
                    values.append(error if signed else abs(error))
                    keys.append((source["protocol"], "entry_diagnostic", seg["cycle"], seg["segment"]))
        print(json.dumps(dict(evaluation=calls, maximum=float(max(abs(np.asarray(values)))))), flush=True)
        return np.asarray(values), keys, unresolved
    initial, keys, unresolved = response(center)
    matrices = []
    perturbations = []
    for step in (.001, .0005):
        columns = []
        for i in range(10):
            low, high = center.copy(), center.copy()
            low[i], high[i] = max(-bound, center[i]-step), min(bound, center[i]+step)
            lo, lk, lu = response(low)
            hi, hk, hu = response(high)
            if lk != keys or hk != keys or lu != unresolved or hu != unresolved:
                raise ValueError("Diagnostic identity changed under perturbation")
            columns.append((hi-lo)/(high[i]-low[i]))
            perturbations.append(dict(step=step, coordinate=i, lower=low[i], upper=high[i],
                                      lower_values=lo.tolist(), upper_values=hi.tolist()))
        matrices.append(np.column_stack(columns))
    discrepancy = float(np.linalg.norm(matrices[0]-matrices[1])/max(np.linalg.norm(matrices[1]), 1e-30))
    result = dict(schema="agentfem.research.local-feasibility.v1", model_id=fit["model_id"],
        source_sha256=fit["source_sha256"], keys=keys, unresolved=unresolved,
        initial_values=abs(initial).tolist(), center=center.tolist(), bound=bound,
        signed_constraint_linearization=signed, initial_signed_values=initial.tolist() if signed else None,
        derivative_relative_discrepancy=discrepancy, candidates=[], fitted=False,
        derivative_matrices=[m.tolist() for m in matrices], perturbations=perturbations,
        engineering_qualified=False, acceptance_thresholds_changed=False,
        scope="Local normalized-error linearization; not a nonlinear feasibility bound")
    if discrepancy <= .002:
        step, predicted = (signed_step if signed else linear_step)(initial, matrices[1], center, bound)
        result.update(linearized_maximum=predicted, proposed_step=step.tolist())
        for fraction in (1., .5, .25):
            u = center+fraction*step
            try:
                values, ck, cu = response(u)
                if ck != keys or cu != unresolved:
                    raise ValueError("Candidate diagnostic identity changed")
                row = dict(completed=True, fraction=fraction, values=values.tolist(),
                    maximum=float(max(abs(values))), coordinates=u.tolist(), parameters=asdict(material(u)),
                    merits_frozen_continuation=admissible(abs(initial), abs(values), [k[0] for k in keys]))
            except (ValueError, RuntimeError) as exc:
                row = dict(completed=False, fraction=fraction, failure=str(exc),
                           merits_frozen_continuation=False)
            result["candidates"].append(row)
    result["response_evaluations"] = calls
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fit", type=Path)
    p.add_argument("seed", type=Path)
    p.add_argument("source_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--signed", action="store_true", help="Equivalent two-sided constraints, avoiding absolute-error kinks")
    args = p.parse_args()
    fit_raw, seed_raw = args.fit.read_bytes(), args.seed.read_bytes()
    fit = json.loads(fit_raw)
    if sha256(seed_raw).hexdigest() != fit["seed_sha256"]:
        raise ValueError("Seed identity changed")
    names = ("p92_local_feasibility.py", "p92_spectrum_recovery_joint.py",
             "p92_recoverable_spectrum.py", "p92_recoverable_mobility_reference.py",
             "p92_full_sensitivity.py", "p92_two_rate_reference.py", "p92_two_rate_controls.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_bam_source.py",
             "p92_gate_objective.py", "p92_response_screen.py", "p92_hold_observations.py",
             "p92_bounded_softening.py", "p92_fit_subspace_audit.py")
    def identities():
        return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    before = identities()
    result = evaluate(fit, json.loads(seed_raw), args.source_dir, signed=args.signed)
    if identities() != before or args.fit.read_bytes() != fit_raw or args.seed.read_bytes() != seed_raw:
        raise RuntimeError("Diagnostic inputs changed")
    result.update(driver_sha256=before, fit_sha256=sha256(fit_raw).hexdigest(), seed_sha256=sha256(seed_raw).hexdigest())
    write_new(args.output, result)

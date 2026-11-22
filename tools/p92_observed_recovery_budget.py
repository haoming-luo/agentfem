"""Conditional observed-path Kelvin budget, not a material prediction."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material
from tools import p92_recoverable_spectrum as kernel
from tools.p92_response_screen import POLICY


def linear_memory(times, stress, moduli, relaxation_times, initial):
    """Exact convolution for a continuous piecewise-linear stress history."""
    if any(np.iscomplexobj(v) for v in (times, stress, moduli, relaxation_times, initial)):
        raise ValueError("Require real histories and coefficients")
    t, s, k, tau, a = [np.asarray(v, dtype=float) for v in
                       (times, stress, moduli, relaxation_times, initial)]
    if (t.ndim != 1 or len(t) < 2 or s.shape != t.shape or np.any(np.diff(t) <= 0)
            or k.ndim != 1 or not len(k) or tau.shape != k.shape or a.shape != k.shape
            or any(not np.isfinite(v).all() for v in (t, s, k, tau, a))
            or np.any(k <= 0) or np.any(tau <= 0)):
        raise ValueError("Require finite ordered history and positive Kelvin coefficients")
    out = np.empty((len(t), len(k)))
    out[0] = a
    for i, h in enumerate(np.diff(t)):
        x = h/tau
        one_minus_decay = -np.expm1(-x)
        # h - tau*(1-exp(-h/tau)); avoid cancellation at small h/tau.
        remainder = x + np.expm1(-x)
        small = x < 1e-4
        remainder[small] = x[small]**2*(.5-x[small]/6+x[small]**2/24-x[small]**3/120)
        out[i+1] = (np.exp(-x)*out[i] + s[i]/k*one_minus_decay
                    + (s[i+1]-s[i])/k*(remainder/x))
    return out


def window_budget(t, stress, strain, memory, i, j, *, modulus, moduli, times, initial_bounds):
    """Finite increment and deterministic bounds; no local differentiation."""
    if not 0 <= i < j < len(t):
        raise ValueError("Require two distinct ordered observations")
    dt = t[j]-t[i]
    recovered = memory[j]-memory[i]
    nonelastic = strain[j]-strain[i]-(stress[j]-stress[i])/modulus
    remainder = nonelastic-sum(recovered)
    decay_difference = abs(np.exp(-(t[j]-t[0])/times)-np.exp(-(t[i]-t[0])/times))
    initial_bound = float(np.dot(initial_bounds, decay_difference))
    # Stable convolution is positive and has L1 gain <=1/K. Endpoint
    # differences have gain <=2/K; conservative even for correlated noise.
    q_s, q_e = POLICY["stress_resolution_MPa"], POLICY["strain_resolution"]
    resolution = float(2*q_e+2*q_s/modulus+2*q_s*sum(1/moduli))
    return dict(source_indices=[int(i), int(j)], time_s=[float(t[i]), float(t[j])],
                duration_s=float(dt), observed_nonelastic_increment=float(nonelastic),
                recoverable_increments=recovered.tolist(),
                required_remainder_increment=float(remainder),
                required_remainder_average_per_s=float(remainder/dt),
                initial_memory_sensitivity=float(initial_bound),
                resolution_sensitivity=float(resolution),
                remainder_interval=[float(remainder-initial_bound-resolution),
                                    float(remainder+initial_bound+resolution)],
                scope="Conditional kinematic remainder; not measured plastic strain or dissipation")


def evaluate(frozen, directory):
    module, material = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not kernel or tuple(material.recoverable_powers) != (1., 1.):
        raise ValueError("Observed convolution requires the original linear Kelvin spectrum")
    records = {}
    for protocol in ("RF", "SLR", "SLC"):
        data = read_source(directory/f"Data_{protocol}.lis", protocol=protocol)
        if data["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = measured_path(data, cycles=3)
        selected = data["cycle"] <= 3
        t, s, e = data["time_s"][selected], data["stress_Pa"][selected]/1e6, data["strain"][selected]
        k, tau = np.array(material.moduli), np.array(material.times)
        memory = linear_memory(t, s, k, tau, np.zeros(2))
        bounds = max(abs(s))/k
        rows, bands = [], []
        for index, segment in enumerate(path["segments"]):
            if not (segment["strain_hold"] or segment["stress_control"]):
                continue
            early = np.asarray(segment["source_indices"])[
                np.asarray(segment["sample_times"]) <= min(10., .1*segment["duration"])]
            late = np.asarray(segment["tail"]["source_indices"])
            for name, ids in (("early", early), ("late", late)):
                if len(ids) < 2:
                    rows.append(dict(cycle=segment["cycle"], segment=index, window=name, status="unresolved"))
                    continue
                row = window_budget(t, s, e, memory, int(ids[0]), int(ids[-1]),
                    modulus=material.backbone.modulus, moduli=k, times=tau, initial_bounds=bounds)
                rows.append(dict(cycle=segment["cycle"], segment=index, window=name,
                                 stress_control=segment["stress_control"], **row))
            if segment["strain_hold"] and segment["target"] > 0:
                lo = path["switches"][index-1]["bracket_s"][1]
                hi = path["switches"][index]["bracket_s"][0]
                for center in (190., 210., 230.):
                    ids = np.flatnonzero((t > lo) & (t < hi) & (abs(s-center) < 5.))
                    label = dict(cycle=segment["cycle"], segment=index, band_center_MPa=center)
                    if len(ids) < 3 or np.any(np.diff(ids) != 1) or s[ids[0]] <= s[ids[-1]]:
                        bands.append(dict(**label, status="unresolved_sampling"))
                        continue
                    row = window_budget(t, s, e, memory, int(ids[0]), int(ids[-1]),
                        modulus=material.backbone.modulus, moduli=k, times=tau, initial_bounds=bounds)
                    bands.append(dict(**label, status="resolved", **row))
        records[protocol] = dict(initial_memory_box=bounds.tolist(), records=rows, equal_stress_bands=bands)
    return dict(schema="agentfem.research.observed-recovery-budget.v1", records=records,
                parameters=frozen["parameters"], model_id=frozen["model_id"],
                source_sha256=frozen["source_sha256"], fitted=False, continuous_prediction=False,
                engineering_qualified=False, life_qualified=False,
                scope="Observed stress and strain diagnostic under a fixed linear recovery hypothesis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_observed_recovery_budget.py", "p92_bam_source.py", "p92_measured_controls.py",
             "p92_recoverable_frozen.py", "p92_recoverable_spectrum.py", "p92_response_screen.py")
    before = {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    result = evaluate(json.loads(raw), args.source_dir)
    if raw != args.frozen.read_bytes() or any(
            sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest
            for name, digest in before.items()):
        raise RuntimeError("Diagnostic input changed")
    result.update(driver_sha256=before, frozen_sha256=sha256(raw).hexdigest())
    write_new(args.output, result)

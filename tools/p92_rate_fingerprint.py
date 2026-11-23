"""Instantaneous frozen-state flow inversion, not a stress-dip experiment."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, bisect

from tools import p92_aging_resistance_reference as aging
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_recoverable_frozen import decode_material
from tools.p92_measured_fit import write_new

RATES = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


def components(state, stress, module, material):
    y = np.array(state, dtype=float, copy=True)
    if module not in (spectrum, aging) or y.shape != (len(module.STATE_LAYOUT),):
        raise ValueError("Require a supported complete scalar state")
    if not np.isfinite(y).all() or not np.isfinite(stress):
        raise ValueError("Require finite state and stress")
    y[0] = stress
    dy = module.rates(y, 0., material, stress_control=True)
    branches = [float(dy[p]*np.sign(stress-sum(y[x:x+2]))) for p, x in ((3, 1), (6, 4))]
    recoverable = float(sum(dy[10:12]))
    total = float(dy[7])
    if not np.isfinite([*branches, recoverable, total]).all():
        raise ValueError("Nonfinite rate fingerprint")
    if abs(sum(branches)+recoverable-total) > 1e-12+1e-10*abs(total):
        raise ValueError("Nonelastic rate decomposition failed")
    return dict(irreversible=sum(branches), recoverable=recoverable, total=total,
                branch_rates=branches)


def inverse(state, module, material, target, channel):
    if channel not in {"irreversible", "total"} or not np.isfinite(target) or target == 0:
        raise ValueError("Use a finite nonzero rate and a declared channel")
    def residual(stress):
        return components(state, stress, module, material)[channel]-target
    a, b = residual(-2000.), residual(2000.)
    if not a < 0 < b:
        raise ValueError("Rate outside predeclared diagnostic stress bracket")
    root = brentq(residual, -2000., 2000., xtol=1e-9, rtol=1e-14)
    independent = bisect(residual, -2000., 2000., xtol=1e-9, rtol=1e-14)
    error = abs(residual(root))
    if abs(root-independent) > 1e-6 or error > 1e-12+1e-7*abs(target):
        raise RuntimeError("Independent root verification failed")
    return dict(stress_MPa=root, residual_per_s=error,
                independent_difference_MPa=abs(root-independent))


def fingerprint(state, module, material):
    y = np.asarray(state, dtype=float)
    # Validate before indexing; queries never advance or modify this state.
    components(y, 0., module, material)
    base = material if module is spectrum else material.spectrum
    if module is spectrum:
        evolving, _ = spectrum.evolving_backbone(y, base)
        branches = evolving.branches
    else:
        branches = base.backbone.branches
    factor = 1. if module is spectrum else 1.+y[12]
    if factor < 0:
        raise ValueError("Negative resistance factor")
    centers = [float(sum(y[i:i+2])) for i in (1, 4)]
    lo = max(x-b.strength*factor for x, b in zip(centers, branches, strict=True))
    hi = min(x+b.strength*factor for x, b in zip(centers, branches, strict=True))
    interval = [lo, hi] if lo <= hi else None
    records = []
    for magnitude in RATES:
        channels = {}
        for channel in ("irreversible", "total"):
            negative = inverse(y, module, material, -magnitude, channel)
            positive = inverse(y, module, material, magnitude, channel)
            a, b = negative["stress_MPa"], positive["stress_MPa"]
            channels[channel] = dict(negative=negative, positive=positive,
                center_MPa=(a+b)/2, half_width_MPa=(b-a)/2)
        records.append(dict(rate_per_s=magnitude, **channels))
    return dict(inactive_interval_MPa=interval, branch_centers_MPa=centers,
        at_inherited_stress=components(y, y[0], module, material), records=records,
        common_stress_scan=[dict(stress_MPa=s, **components(y, s, module, material))
                            for s in (-300., -250., -200., -150., -100., 100., 150., 200., 250., 300.)])


def evaluate(frozen):
    module, material = decode_material(frozen["parameters"], frozen["model_id"])
    records = []
    if set(frozen["protocols"]) != {"RF", "SLR", "SLC"}:
        raise ValueError("Require all three protocols")
    for protocol, item in frozen["protocols"].items():
        prediction = item["prediction"]
        if (not item["numerical"]["passed"] or prediction["protocol"] != protocol
                or prediction["model_id"] != frozen["model_id"]
                or prediction["source_sha256"] != frozen["source_sha256"][protocol]):
            raise ValueError("Require numerically checked, aligned frozen evidence")
        for hold in prediction["hold_diagnostics"]:
            if hold["cycle"] not in (1, 2, 3) or hold["stress_control"]:
                continue
            i = hold["segment"]
            segment = prediction["segments"][i]
            if segment["end_state"][0] <= 0:
                continue
            if i <= 0 or segment["cycle"] != hold["cycle"] or segment["segment"] != i:
                raise ValueError("Invalid causal hold identity")
            for label, state in (("entry", prediction["segments"][i-1]["end_state"]),
                                 ("exit", segment["end_state"])):
                records.append(dict(protocol=protocol, cycle=hold["cycle"], segment=i,
                    station=label, hold_duration_s=hold["duration_s"],
                    hold_strain=hold["fields"]["strain"]["first_predicted"],
                    state=state, **fingerprint(state, module, material)))
    counts = {p: sum(r["protocol"] == p for r in records) for p in frozen["protocols"]}
    if counts != {"RF": 6, "SLR": 12, "SLC": 6}:
        raise ValueError("Require peak holds and the additional SLR intermediate holds")
    return dict(schema="agentfem.research.frozen-rate-fingerprint.v1",
        model_id=frozen["model_id"], parameters=frozen["parameters"],
        source_sha256=frozen["source_sha256"], records=records, cycles=[1, 2, 3],
        fitted=False, experimental_validation=False, engineering_qualified=False, life_qualified=False,
        scope="Instantaneous fixed-history queries, not physical jumps or experimental stress-dip measurements")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_rate_fingerprint.py", "p92_recoverable_frozen.py", "p92_recoverable_spectrum.py",
             "p92_two_rate_reference.py", "p92_aging_resistance_reference.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    before = identities()
    result = evaluate(json.loads(raw))
    if raw != args.frozen.read_bytes() or before != identities():
        raise RuntimeError("Fingerprint inputs changed")
    result.update(frozen_sha256=sha256(raw).hexdigest(), driver_sha256=before)
    write_new(args.output, result)

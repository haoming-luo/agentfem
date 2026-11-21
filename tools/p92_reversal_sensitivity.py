"""Prospective early-cycle second-backstress discriminator; no refit or admission."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path, replay
from tools.p92_measured_fit import write_new
from tools.p92_published_recovery_reference import Parameters
from tools.p92_two_mechanism import Mechanism
from tools.p92_waveform_controls import attach_waveforms


def perturb(mechanisms, branch, direction, log_scale):
    """Separate second-backstress saturation from its initial signed tangent."""
    if branch not in (0, 1) or direction not in {"shape", "tangent"} or not np.isfinite(log_scale):
        raise ValueError("Invalid conditional coordinate")
    out = list(mechanisms)
    m, factor = out[branch], 10.**log_scale
    p = m.parameters
    p = (replace(p, a2=p.a2*factor, C2=p.C2/factor) if direction == "shape"
         else replace(p, C2=p.C2*factor))
    out[branch] = replace(m, parameters=p)
    return tuple(out)


def diagnose(seed, sources, *, radius=.1):
    if not np.isfinite(radius) or not 0 < radius <= .2:
        raise ValueError("Use a bounded prospective perturbation")
    cards = seed["parameters"]
    digest = sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest()
    if digest != seed["parameter_sha256"] or not seed["fit"]["success"]:
        raise ValueError("Require an intact frozen card")
    if len(sources) != 3 or {s["protocol"] for s in sources} != {"RF", "SLR", "SLC"}:
        raise ValueError("Require all three protocols")
    if any(seed["source_sha256"][s["protocol"]] != s["sha256"] for s in sources):
        raise ValueError("Source identity changed")
    mechanisms = tuple(Mechanism(Parameters(**c["parameters"]), c["exponent"], c["enabled"])
                       for c in cards)
    paths = [attach_waveforms(measured_path(s, cycles=3), s) for s in sources]

    def evaluate(m):
        residuals, records = [], []
        for path in paths:
            residual, prediction = replay(path, m)
            if prediction["reversed_post_hold_ramps"] or len(prediction["cycles"]) != 3:
                raise ValueError("Perturbation leaves the declared continuous control path")
            residuals.append(residual)
            records.append(dict(protocol=path["protocol"], cycles=prediction["cycles"],
                                holds=prediction["hold_diagnostics"]))
        r = np.concatenate(residuals)
        return r, dict(cost=float(.5*r@r), protocols=records)

    baseline, original = evaluate(mechanisms)
    columns, experiments = [], []
    for branch in (0, 1):
        for direction in ("shape", "tangent"):
            pair, rows = [], []
            for sign in (-1, 1):
                residual, record = evaluate(perturb(mechanisms, branch, direction, sign*radius))
                pair.append(residual)
                rows.append(dict(log10_scale=sign*radius, **record))
            derivative = (pair[1]-pair[0])/(2*radius)
            columns.append(derivative)
            experiments.append(dict(branch=branch, direction=direction, records=rows,
                residual_derivative_norm=float(np.linalg.norm(derivative)),
                objective_directional_derivative=float(baseline@derivative),
                centered_nonlinearity_norm=float(np.linalg.norm(pair[1]+pair[0]-2*baseline))))
            print(json.dumps(dict(branch=branch, direction=direction,
                                  costs=[r["cost"] for r in rows])), flush=True)
    return dict(schema="agentfem.research.p92-reversal-sensitivity.v1",
        parameter_sha256=digest, source_sha256=seed["source_sha256"], radius_log10=radius,
        observed_cycles=[1, 2, 3], baseline=original, experiments=experiments,
        residual_jacobian_singular_values=np.linalg.svd(np.column_stack(columns), compute_uv=False).tolist(),
        scope="Conditional second-backstress early-cycle sensitivity; no parameter selection or later-cycle access",
        engineering_qualified=False, life_qualified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    data = [read_source(args.source_dir/f"Data_{p}.lis", protocol=p) for p in ("RF", "SLR", "SLC")]
    write_new(args.output, diagnose(json.loads(args.seed.read_text()), data))

"""Two-coordinate early-cycle fit after frozen waveform discrimination."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path, replay
from tools.p92_measured_fit import write_new
from tools.p92_published_recovery_reference import Parameters
from tools.p92_reversal_sensitivity import perturb
from tools.p92_two_mechanism import Mechanism
from tools.p92_waveform_controls import attach_waveforms


def decode(z, mechanisms):
    z = np.asarray(z, dtype=float)
    if z.shape != (2,) or not np.isfinite(z).all():
        raise ValueError("Require two finite conditional coordinates")
    return perturb(perturb(mechanisms, 0, "shape", z[0]), 0, "tangent", z[1])


def identify(seed, sources, *, max_nfev=25):
    if type(max_nfev) is not int or not 1 <= max_nfev <= 25:
        raise ValueError("Require a bounded evaluation budget")
    cards = seed["parameters"]
    if sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest() != seed["parameter_sha256"]:
        raise ValueError("Frozen card identity changed")
    if not seed["fit"]["success"] or seed["training_cycles"] != [1, 2, 3]:
        raise ValueError("Require converged early-cycle seed")
    if len(sources) != 3 or {s["protocol"] for s in sources} != {"RF", "SLR", "SLC"}:
        raise ValueError("Require three distinct protocols")
    if any(seed["source_sha256"][s["protocol"]] != s["sha256"] for s in sources):
        raise ValueError("Source identity changed")
    mechanisms = tuple(Mechanism(Parameters(**c["parameters"]), c["exponent"], c["enabled"])
                       for c in cards)
    paths = [attach_waveforms(measured_path(s, cycles=3), s) for s in sources]
    calls = 0

    def objective(z, rtol=1e-8):
        nonlocal calls
        calls += 1
        candidate = decode(z, mechanisms)
        values = []
        for path in paths:
            r, prediction = replay(path, candidate, rtol=rtol)
            if len(prediction["cycles"]) != 3 or prediction["reversed_post_hold_ramps"]:
                raise ValueError("Candidate leaves the declared continuous control path")
            values.append(r)
        residual = np.concatenate(values)
        if calls % 10 == 1:
            print(json.dumps(dict(evaluation=calls, cost=float(.5*residual@residual))), flush=True)
        return residual

    def jacobian(z):
        return np.column_stack([(objective(z+np.eye(2)[i]*.001)-
                                 objective(z-np.eye(2)[i]*.001))/.002 for i in range(2)])

    result = least_squares(objective, np.zeros(2), jac=jacobian, bounds=(-.3, .3),
        x_scale="jac", max_nfev=max_nfev, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    identity = [asdict(m) for m in decode(result.x, mechanisms)]
    return dict(schema="agentfem.research.p92-reversal-fit.v1", training_cycles=[1, 2, 3],
        coordinate_names=["log10_fast_a2_at_fixed_C2a2", "log10_fast_C2_at_fixed_a2"],
        coordinates=result.x.tolist(), bounds=[[-.3, -.3], [.3, .3]],
        parameters=identity, parameter_sha256=sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        source_sha256=seed["source_sha256"], seed_parameter_sha256=seed["parameter_sha256"],
        fit=dict(success=bool(result.success), cost=float(result.cost), message=result.message,
            nfev=int(result.nfev), residual_evaluations=calls, active_bounds=result.active_mask.tolist(),
            singular_values=np.linalg.svd(result.jac, compute_uv=False).tolist()),
        training_indices={p["protocol"]: [s["source_indices"] for s in p["segments"]] for p in paths},
        scope="Conditional fast second-backstress fit; measured ramp shape; no later-cycle fitting",
        engineering_qualified=False, life_qualified=False, independent_validation=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    data = [read_source(args.source_dir/f"Data_{p}.lis", protocol=p) for p in ("RF", "SLR", "SLC")]
    write_new(args.output, identify(json.loads(args.seed.read_text()), data))

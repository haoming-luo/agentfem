"""Compact, auditable summary of completed bounded-recovery experiments."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate


def summarize(coarse, fine, data):
    for key in ("parameters", "limit", "rate", "protocol", "calibration_sha256",
                "segment_end_times_s"):
        if key not in coarse or key not in fine:
            raise ValueError(f"Missing refinement identity {key}; regenerate the incomplete artifact")
        if coarse[key] != fine[key]:
            raise ValueError("Refinement requires unchanged model, initial-state identity and loading path")
    a, b = coarse["cycles"], fine["cycles"]
    if not a or [r["cycle"] for r in a] != [r["cycle"] for r in b]:
        raise ValueError("Refinement requires identical accepted cycle identities")
    if not 0 < fine["rtol"] < coarse["rtol"]:
        raise ValueError("Fine integration must tighten relative tolerance")
    delta = np.asarray([r["end_state"] for r in a])-np.asarray([r["end_state"] for r in b])
    if delta.shape != (len(a), 5) or not np.isfinite(delta).all():
        raise ValueError("Refinement states must be finite")
    comparison = compare_candidate(data, fine)
    extrema = np.asarray([[x[key]-y[key] for key in
                           ("sampled_peak_MPa", "sampled_valley_MPa")]
                          for x, y in zip(a, b, strict=True)])
    if not np.isfinite(extrema).all():
        raise ValueError("Refinement extrema must be finite")
    peak_change = float(np.max(np.abs(extrema[:, 0])))
    valley_change = float(np.max(np.abs(extrema[:, 1])))
    # Numerical screening target declared independently of experimental errors.
    # It does not imply the material model predicts the experiments accurately.
    numerical = (max(peak_change, valley_change) < 1.e-4
                 and np.max(np.abs(delta[:, :4])) < 1.e-4
                 and np.max(np.abs(delta[:, 4])) < 1.e-8)
    return dict(schema="agentfem.research.p92-bounded-recovery-screen.v1",
                protocol=fine["protocol"], parameters=fine["parameters"],
                limit=fine["limit"], rate=fine["rate"],
                calibration_sha256=fine["calibration_sha256"],
                accepted_cycles=len(b), coarse_rtol=coarse["rtol"], fine_rtol=fine["rtol"],
                peak_refinement_max_MPa=peak_change,
                valley_refinement_max_MPa=valley_change,
                state_refinement_max_abs=np.max(np.abs(delta), axis=0).tolist(),
                numerical_refinement_passed=bool(numerical),
                numerical_stress_tolerance_MPa=1.e-4,
                numerical_accumulated_strain_tolerance=1.e-8,
                comparison=comparison, engineering_qualified=False, life_qualified=False,
                selected_cycles=[r for r in b if r["cycle"] in {1, 3, 10, 100, 200, 400, 500, 1000}],
                scope="Mechanism-selection experiment on inspected data; no fresh holdout or failure prediction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    fine = json.loads(args.fine.read_text())
    report = summarize(json.loads(args.coarse.read_text()), fine,
                       read_source(args.source, protocol=fine["protocol"]))
    report["input_artifact_sha256"] = {key: sha256(path.read_bytes()).hexdigest()
                                        for key, path in (("coarse", args.coarse), ("fine", args.fine))}
    print(json.dumps(report, indent=2, allow_nan=False))

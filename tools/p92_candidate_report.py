"""Compact, auditable summary of completed bounded-recovery experiments."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate


def summarize(coarse, fine, data):
    for key in ("parameters", "limit", "rate"):
        if coarse[key] != fine[key]:
            raise ValueError("Refinement requires unchanged model parameters")
    a, b = coarse["cycles"], fine["cycles"]
    if not a or [r["cycle"] for r in a] != [r["cycle"] for r in b]:
        raise ValueError("Refinement requires identical accepted cycle identities")
    if not 0 < fine["rtol"] < coarse["rtol"]:
        raise ValueError("Fine integration must tighten relative tolerance")
    delta = np.asarray([r["end_state"] for r in a])-np.asarray([r["end_state"] for r in b])
    if not np.isfinite(delta).all():
        raise ValueError("Refinement states must be finite")
    comparison = compare_candidate(data, fine)
    peak_change = max(abs(x["sampled_peak_MPa"]-y["sampled_peak_MPa"])
                      for x, y in zip(a, b, strict=True))
    # Numerical screening target declared independently of experimental errors.
    # It does not imply the material model predicts the experiments accurately.
    numerical = peak_change < 1.e-4 and np.max(np.abs(delta[:, :4])) < 1.e-4
    return dict(schema="agentfem.research.p92-bounded-recovery-screen.v1",
                protocol=fine["protocol"], parameters=fine["parameters"],
                limit=fine["limit"], rate=fine["rate"],
                calibration_sha256=fine["calibration_sha256"],
                accepted_cycles=len(b), coarse_rtol=coarse["rtol"], fine_rtol=fine["rtol"],
                peak_refinement_max_MPa=peak_change,
                state_refinement_max_abs=np.max(np.abs(delta), axis=0).tolist(),
                numerical_refinement_passed=bool(numerical),
                numerical_stress_tolerance_MPa=1.e-4,
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

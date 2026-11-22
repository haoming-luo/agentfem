"""Identity-checked research comparison; never promotes a failed material."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from tools.p92_measured_fit import write_new
from tools.p92_response_screen import POLICY, screen


def fit_identity(raw):
    """Read a run or its lossless archive without trusting a provenance label."""
    fit = json.loads(raw)
    claimed = fit.get("archived_source_sha256")
    if claimed is None:
        return fit, sha256(raw).hexdigest()
    original = {k: v for k, v in fit.items() if k != "archived_source_sha256"}
    encoded = (json.dumps(original, indent=2, allow_nan=False)+"\n").encode()
    digest = sha256(encoded).hexdigest()
    if digest != claimed:
        raise ValueError("Identification archive is not a lossless original run")
    return fit, digest


def check_pair(control, candidate):
    for name in ("source_sha256", "seed_sha256", "training_cycles",
                 "maximum_log_coordinate_norm", "driver_sha256"):
        if name not in control or control[name] != candidate.get(name):
            raise ValueError(f"Unmatched identification {name}")
    if control["training_cycles"] != [1, 2, 3]:
        raise ValueError("Require the unchanged early-only partition")
    for fit in (control, candidate):
        actual = fit["fit"]["coordinates"]
        if (not fit["fit"]["success"] or len(fit["coordinates"]) != 22
                or len(actual) != 10 or not 1 <= fit["fit"]["nfev"] <= 40
                or any(not math.isfinite(x) or abs(x) > .3/math.sqrt(10)+1e-12 for x in actual)
                or fit["fit"]["residual_evaluations"] < fit["fit"]["nfev"]
                or fit["maximum_log_coordinate_norm"] != .3
                or not math.isfinite(fit["fit"]["cost"]) or fit["fit"]["cost"] <= 0):
            raise ValueError("Require converged matched-budget identification")


def response_summary(fit, frozen, fit_digest):
    if (frozen["fit_file_sha256"] != fit_digest
            or frozen["parameters"] != fit["fit"]["parameters"]
            or frozen["model_id"] != fit["model_id"]
            or frozen["source_sha256"] != fit["source_sha256"]):
        raise ValueError("Frozen response does not belong to identification")
    if set(frozen["protocols"]) != {"RF", "SLR", "SLC"}:
        raise ValueError("Require all three protocols, including failures")
    protocols = {}
    for name, result in frozen["protocols"].items():
        if "failure" in result:
            protocols[name] = dict(status="failed", failure=result["failure"], numerical_passed=False)
            continue
        if result["response_screen"]["policy"] != POLICY:
            raise ValueError("Changed response policy")
        prediction = result["prediction"]
        if (prediction["protocol"] != name or prediction["model_id"] != frozen["model_id"]
                or prediction["source_sha256"] != frozen["source_sha256"][name]
                or result["comparison"]["source_sha256"] != frozen["source_sha256"][name]):
            raise ValueError("Frozen protocol identity changed")
        if [r["cycle"] for r in prediction["cycles"]] != list(range(1, 11)):
            raise ValueError("Incomplete frozen development path")
        if screen(prediction, result["comparison"], requested_cycles=10) != result["response_screen"]:
            raise ValueError("Frozen checks do not reproduce from retained observations")
        unresolved_observations = [dict(name=c["name"], cycle=c["cycle"],
            segment=c["segment"], resolution_units=c["resolution_units"])
            for c in result["response_screen"]["checks"]
            if "resolution_units" in c and c["resolution_units"] < POLICY["minimum_resolution_units"]]
        holds = []
        for h in prediction["hold_diagnostics"]:
            fields = {}
            for field in ("strain", "stress_MPa") if h["stress_control"] else ("stress_MPa",):
                f = h["fields"][field]
                fields[field] = {key: f.get(key) for key in (
                    "rms", "first_observed", "first_predicted", "first_residual", "last_observed",
                    "last_predicted", "anchored_shape_rms", "early_secant_observed_per_s",
                    "early_secant_predicted_per_s", "late_secant_observed_per_s", "late_secant_predicted_per_s")}
            holds.append(dict(cycle=h["cycle"], segment=h["segment"],
                              stress_control=h["stress_control"], fields=fields))
        protocols[name] = dict(status=result["response_screen"]["status"],
            numerical_passed=result["numerical"]["passed"],
            observation_resolution_ready=not unresolved_observations,
            unresolved_observations=unresolved_observations,
            checks=result["response_screen"]["checks"],
            peaks=result["comparison"]["records"], holds=holds)
    return dict(cost=fit["fit"]["cost"], optimizer_evaluations=fit["fit"]["nfev"],
        residual_evaluations=fit["fit"]["residual_evaluations"], protocols=protocols)


def compare(control, candidate, control_frozen, candidate_frozen, control_digest, candidate_digest):
    check_pair(control, candidate)
    original = response_summary(control, control_frozen, control_digest)
    replacement = response_summary(candidate, candidate_frozen, candidate_digest)
    return dict(schema="agentfem.research.matched-response-report.v1",
        source_sha256=control["source_sha256"], seed_sha256=control["seed_sha256"],
        control=original, candidate=replacement, response_policy=POLICY,
        relative_cost_change=(replacement["cost"]-original["cost"])/original["cost"],
        engineering_qualified=False, life_qualified=False, independent_validation=False,
        scope="All original checks retained; later cycles previously inspected; no automatic model selection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "candidate", "control_frozen", "candidate_frozen", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    paths = [getattr(args, n) for n in ("control", "candidate", "control_frozen", "candidate_frozen")]
    raw = [p.read_bytes() for p in paths]
    fits = [fit_identity(r) for r in raw[:2]]
    result = compare(*(f[0] for f in fits), *(json.loads(r) for r in raw[2:]),
                     *(f[1] for f in fits))
    if any(p.read_bytes() != r for p, r in zip(paths, raw, strict=True)):
        raise RuntimeError("Comparison inputs changed")
    result["input_sha256"] = [sha256(r).hexdigest() for r in raw]
    write_new(args.output, result)

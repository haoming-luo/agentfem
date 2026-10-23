# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded serial cyclic J2/contact/MPC increment-controller comparison.

Different accepted paths are compared at common physical stations against a
finite fixed-path reference, not against an exact continuum solution. Compact
scalar evidence only; no private geometry, material or full field archives.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter

import numpy as np

import agentfem
from agentfem import steps
from agentfem.checkpointing import atomic_write_text
from diagnose_periodic_contact_order import make_step, observe_penalty


def difference(actual, reference):
    delta = float(np.linalg.norm(np.asarray(actual) - np.asarray(reference)))
    scale = float(np.linalg.norm(reference))
    return {"absolute_l2": delta, "relative_l2": delta / scale if scale > 1e-12 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--limit", type=float, default=.00075)
    parser.add_argument("--reference-increments", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Select a fresh output path; retain prior evidence.")
    root = Path(__file__).resolve().parents[1]
    source_files = sorted((root / "src" / "agentfem").rglob("*.py"))
    source_files += [Path(__file__), Path(__file__).with_name("diagnose_periodic_contact_order.py"),
                     Path(__file__).with_name("verify_affine_augmented_path.py")]
    def source_hash():
        digest = sha256()
        for path in source_files:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()
    frozen = source_hash()
    report = {
        "schema": "agentfem.inelastic-increment-prediction-screen.v1",
        "scope": "serial cyclic displacement J2, P2 tetrahedra, Gaussian plane penalty and periodic MPC",
        "not_claimed": ["explicit composite acceleration", "MPI support", "continuum convergence",
                        "general mechanical energy closure", "universal error estimator"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": frozen,
        "tool_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": sha256(Path(__file__).with_name("diagnose_periodic_contact_order.py").read_bytes()).hexdigest(),
        "agentfem_version": agentfem.__version__, "platform": platform.platform(),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "options": {name: value for name, value in vars(args).items() if name != "output"},
        "timing_boundary": "setup_solve_observe; reference first primes compilation before ABBA; individual observations retained",
        "reference_is_exact": False, "records": [],
    }
    def save():
        atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    def require_frozen_source():
        if source_hash() != frozen:
            report.update(status="failed_source_identity", error="Runtime or diagnostic source changed during the experiment.")
            save()
            raise RuntimeError(report["error"])
    references = []
    repeat_reference = {}
    names = ("fixed_reference", "ordinary", "prediction", "prediction", "ordinary")
    for name in names:
        require_frozen_source()
        control = (steps.fixed(args.reference_increments) if name == "fixed_reference" else
                   steps.automatic(initial=1/32, minimum=1/4096, maximum=1/16,
                                   max_increments=1024, max_cutbacks=12, cutback_factor=.5,
                                   maximum_inelastic_increment=args.limit,
                                   inelastic_prediction_safety=.8 if name == "prediction" else None))
        row = {"policy": name, "control": control.summary(), "status": "running", "stations": []}
        report["records"].append(row)
        save()
        started = perf_counter()
        try:
            step = make_step(cells=args.cells, degree=2, bulk_degree=6, contact_degree=11,
                             incrementation=control)
            current = []
            for index, factor in enumerate((.25, .5, .75, 1.)):
                step.solve(until=factor)
                observed = observe_penalty(step, factor)
                arrays = (step.solution.x.array.copy(), step.response.state.committed_state_vectors().copy(),
                          step.response.first_piola_stress.values.copy(), step.response.tangent.values.copy())
                assert all(np.isfinite(value).all() for value in arrays)
                if name == "fixed_reference":
                    references.append((observed, arrays[:2]))
                else:
                    previous, previous_arrays = references[index]
                    observed["differences_to_fixed_reference"] = {
                        key: difference(observed[key], previous[key]) for key in (
                            "force_on_structure", "pressure_area_mean", "maximum_pressure", "active_reference_area",
                            "peeq_reference_volume_mean", "maximum_peeq", "displacement_l2")}
                    observed["differences_to_fixed_reference"].update({
                        key: difference(value, previous_value)
                        for key, value, previous_value in zip(("displacement", "state"), arrays[:2], previous_arrays, strict=True)})
                    if name in repeat_reference:
                        for actual, expected in zip(arrays, repeat_reference[name][index], strict=True):
                            np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-10)
                    current.append(arrays)
                assert observed["mpc_equation_mismatch"] < 1e-10
                row["stations"].append(observed)
                save()
                print(json.dumps({"policy": name, "factor": factor}), flush=True)
            if name != "fixed_reference":
                repeat_reference[name] = current
                assert all(item.checks["maximum_plastic_increment"] <= args.limit for item in step.accepted_increments)
            assert all(any(abs(item.load_factor - knot) < 1e-12 for item in step.accepted_increments)
                       for knot in (.25, .5, .75, 1.))
            row.update(status="completed", accepted=len(step.accepted_increments),
                       rejected=sum(not item.converged for item in step.attempted_increments),
                       newton_iterations=sum(item.iterations for item in step.attempted_increments),
                       maximum_accepted_plastic_increment=max(item.checks["maximum_plastic_increment"] for item in step.accepted_increments),
                       accepted_factors=[item.load_factor for item in step.accepted_increments])
        except BaseException as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            save()
            raise
        row["setup_solve_observe_seconds"] = perf_counter() - started
        save()
        print(json.dumps({key: value for key, value in row.items()
                          if key not in ("stations", "accepted_factors", "control")}), flush=True)
    require_frozen_source()
    means = {name: statistics.mean(row["setup_solve_observe_seconds"] for row in report["records"] if row["policy"] == name)
             for name in ("ordinary", "prediction")}
    report["paired_timing"] = {"order": "ABBA", "mean_seconds": means,
                               "elapsed_reduction": 1 - means["prediction"] / means["ordinary"]}
    report["source_unchanged"] = True
    report["status"] = "completed_bounded_comparison"
    save()
    print(json.dumps(report["paired_timing"]), flush=True)


if __name__ == "__main__":
    main()

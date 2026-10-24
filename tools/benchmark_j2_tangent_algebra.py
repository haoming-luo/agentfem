# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Same-law spectral tangent comparison against a fixed local Git reference.

Run alone with numerical threads limited. The kernel comparison is not a
solver speedup. Optional cyclic P2/J2/contact/MPC runs compare accepted fields
and histories at four stations. No model parameters or solver policy change.
"""

import argparse
import ast
from hashlib import sha256
from functools import partial
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from unittest.mock import patch

import numpy as np
from mpi4py import MPI

import agentfem
from agentfem.constitutive.finite_strain_plasticity import FiniteStrainJ2Logarithmic
from agentfem.checkpointing import atomic_write_text
from benchmark_finite_hex_kernels import paired
from diagnose_periodic_contact_order import make_step, observe_penalty


REFERENCE = "197ba2c89b2e624c931258fa3416550af846ac24"
MATERIAL = "constitutive/finite_strain_plasticity.py"


def reference_tangent():
    source = subprocess.check_output(
        ["git", "show", f"{REFERENCE}:src/agentfem/{MATERIAL}"],
        cwd=Path(__file__).resolve().parents[1], text=True,
    )
    owner = next(n for n in ast.parse(source).body
                 if isinstance(n, ast.ClassDef) and n.name == "FiniteStrainJ2Logarithmic")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_analytic_algorithmic_tangent_batch")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]),
                 "trusted-committed-tangent", "exec"), namespace)
    return namespace[method.name], sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-path", action="store_true")
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1 or args.output.exists():
        parser.error("Use one process and a fresh evidence path.")
    reference, reference_hash = reference_tangent()
    candidate = FiniteStrainJ2Logarithmic._analytic_algorithmic_tangent_batch
    root = Path(agentfem.__file__).parent
    repository = Path(__file__).resolve().parents[1]
    source_paths = sorted(root.rglob("*.py"))
    fixture_paths = [Path(__file__).resolve(), *(repository / "tools" / name for name in (
        "diagnose_periodic_contact_order.py", "verify_affine_augmented_path.py",
        "benchmark_finite_hex_kernels.py"))]
    identified_paths = [("src/agentfem/" + str(path.relative_to(root)), path) for path in source_paths]
    identified_paths += [("tools/" + path.name, path) for path in fixture_paths]
    def fingerprint():
        digest = sha256()
        for name, path in identified_paths:
            digest.update(name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()
    frozen = fingerprint()
    report = {"schema": "agentfem.j2-tangent-algebra.v1", "status": "running",
              "scope": "serial spectral finite-J2; not explicit composite or universal speedup",
              "reference_revision": REFERENCE, "reference_source_sha256": reference_hash,
              "runtime_source_sha256": frozen,
              "candidate_material_sha256": sha256((root / MATERIAL).read_bytes()).hexdigest(),
              "tool_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "platform": platform.platform(), "numpy": np.__version__,
              "thread_environment": {key: os.environ.get(key) for key in (
                  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "FI_PROVIDER")},
              "kernels": [], "paths": []}
    def save():
        atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save()
    try:
        rng = np.random.default_rng(641024)
        material = FiniteStrainJ2Logarithmic(young=1000., poisson=.3,
                                            yield_stress=2., hardening_modulus=100.)
        for count in (1, 64, 1025, 4096):
            states = np.tile(material.state_schema.initial_state(), (count, 1))
            preload = np.eye(3) + rng.uniform(-.04, .04, (count, 3, 3))
            states = material._integrate_batch(preload, states)["state"]
            gradients = preload + rng.uniform(-.003, .003, (count, 3, 3))
            gradients[::3] = np.eye(3)
            baseline = material._integrate_batch(gradients, states)
            result = paired(
                partial(reference, material, gradients, states, baseline=baseline),
                partial(candidate, material, gradients, states, baseline=baseline),
                repetitions=21,
                equivalent=lambda a, b: np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-11),
            )
            report["kernels"].append({"points": count, **result})
            save()
        if args.global_path:
            # Discard one complete small warmup per route before ABBA timing.
            expected = None
            for name in ("reference_warmup", "candidate_warmup", "reference",
                         "candidate", "candidate", "reference"):
                method = reference if name.startswith("reference") else candidate
                step = make_step(cells=4, degree=2, increments=32,
                                 contact_degree=11, bulk_degree=6)
                snapshots, observations = [], []
                with patch.object(FiniteStrainJ2Logarithmic, "_analytic_algorithmic_tangent_batch", method):
                    started = perf_counter()
                    for factor in (.25, .5, .75, 1.):
                        step.solve(until=factor)
                        observations.append(observe_penalty(step, factor))
                        snapshots.append((step.solution.x.array.copy(),
                            step.response.state.committed_state_vectors().copy(),
                            step.response.first_piola_stress.values.copy(),
                            step.response.tangent.values.copy()))
                    elapsed = perf_counter() - started
                path = [(item.load_factor, item.iterations, item.accepted_step_lengths,
                         item.linear_converged_reasons) for item in step.accepted_increments]
                if expected is None:
                    expected = snapshots, path, observations
                if path != expected[1]:
                    raise AssertionError("Accepted increment/Newton paths changed.")
                maximum_difference = 0.
                for actual, previous in zip(snapshots, expected[0], strict=True):
                    for a, b in zip(actual, previous, strict=True):
                        np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-10)
                        maximum_difference = max(maximum_difference, float(np.max(abs(a-b))))
                for actual, previous in zip(observations, expected[2], strict=True):
                    for key in ("force_on_structure", "pressure_area_mean", "maximum_pressure",
                                "active_reference_area", "peeq_reference_volume_mean",
                                "maximum_peeq", "displacement_l2"):
                        np.testing.assert_allclose(actual[key], previous[key], rtol=2e-8, atol=2e-10)
                    if actual["mpc_equation_mismatch"] >= 1e-10:
                        raise AssertionError("Periodic equations failed the original gate.")
                row = {"route": name, "solve_observe_seconds": elapsed,
                       "maximum_array_difference": maximum_difference,
                       "accepted": len(path), "newton_iterations": sum(i[1] for i in path),
                       "stations": observations}
                report["paths"].append(row)
                save()
                print(json.dumps({k: v for k, v in row.items() if k != "stations"}), flush=True)
            means = {name: float(np.mean([row["solve_observe_seconds"] for row in report["paths"]
                                         if row["route"] == name])) for name in ("reference", "candidate")}
            report["global_abba"] = {"mean_seconds": means,
                                     "elapsed_reduction": 1-means["candidate"]/means["reference"]}
        if fingerprint() != frozen:
            raise RuntimeError("Source changed during comparison.")
        report.update(status="completed", source_unchanged=True)
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    save()
    print(json.dumps(report.get("global_abba", {"status": report["status"]})))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Same-path ABBA test of optional residual-only contact observation.

Run without another numerical job for meaningful timing. Both modes use the
same current Newton tangent, load path, material and accepted-state ownership.
Only dual/norm observations change: baseline evaluates and discards a tangent
and assembles both path sides separately; candidate uses the residual hook and
shares equal path directions. This is not an explicit-composite benchmark.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from unittest.mock import patch

import dolfinx
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

import agentfem
from agentfem.boundary_models.dolfinx_implicit_facet_contact import DolfinxImplicitRigidContact
from agentfem.checkpointing import atomic_write_text
from agentfem.constraints.affine import AbaqusPeriodicConstraint
from benchmark_j2_tangent_algebra import compare_tangent_tensors
from diagnose_periodic_contact_order import make_step, observe_penalty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--increments", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1 or args.output.exists():
        parser.error("Use one process and a fresh evidence path.")
    if any(value < 4 or value % 4 for value in (args.cells, args.increments)):
        parser.error("Use cell/increment counts divisible by four.")
    root = Path(__file__).resolve().parents[1]
    sources = [*Path(agentfem.__file__).parent.rglob("*.py"), Path(__file__),
               *(root / "tools" / name for name in (
                   "diagnose_periodic_contact_order.py", "verify_affine_augmented_path.py",
                   "benchmark_j2_tangent_algebra.py"))]

    def fingerprint():
        digest = sha256()
        for source in sorted(sources):
            digest.update(str(source.relative_to(root)).encode())
            digest.update(source.read_bytes())
        return digest.hexdigest()

    frozen = fingerprint()
    report = dict(
        schema="agentfem.contact-residual-observation.v1", status="running",
        scope="serial_cyclic_P2_finite_J2_plane_penalty_MPC_not_explicit_composite",
        cells=args.cells, increments=args.increments,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        runtime_source_sha256=frozen, platform=platform.platform(),
        versions={"numpy": np.__version__, "dolfinx": dolfinx.__version__,
                  "petsc": PETSc.Sys.getVersion(), "mpi": MPI.Get_library_version().splitlines()[0]},
        thread_environment={key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "FI_PROVIDER")},
        timing_boundary="solve and four observations, excludes setup; two whole-path warmups then ABBA",
        gates={"field_state_stress_history_rtol": 2e-8, "atol": 2e-10,
               "per_point_tangent_frobenius_rtol": 2e-10,
               "accepted_factors_newton_linesearch_linear_reasons": "identical"}, records=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")

    full_evaluate = DolfinxImplicitRigidContact.evaluate
    current_residual = DolfinxImplicitRigidContact.add_current_residual
    paired_duals = AbaqusPeriodicConstraint._accepted_dual_evidence_pair
    expected = None
    save()
    try:
        for mode in ("full_warmup", "residual_warmup", "full", "residual", "residual", "full"):
            counts = dict(full_evaluations=0, residual_observations=0)

            def evaluated(self, _counts=counts):
                _counts["full_evaluations"] += 1
                return full_evaluate(self)

            def observed(self, vector, _counts=counts, _mode=mode):
                _counts["residual_observations"] += 1
                if _mode.startswith("full"):
                    self.add_residual(vector, self.evaluate())
                else:
                    current_residual(self, vector)

            def dual_pair(self, problem, *, load_factor, _mode=mode):
                if _mode.startswith("full"):
                    return (self.dual_evidence(problem, load_factor=load_factor),
                            self.dual_evidence(problem, load_factor=load_factor, path_side="right"))
                return paired_duals(self, problem, load_factor=load_factor)

            step = make_step(cells=args.cells, degree=2, increments=args.increments,
                             contact_degree=11, bulk_degree=6)
            row = dict(mode=mode, status="running", stations=[])
            report["records"].append(row)
            save()
            snapshots = []
            with patch.object(DolfinxImplicitRigidContact, "evaluate", evaluated), patch.object(
                DolfinxImplicitRigidContact, "add_current_residual", observed,
            ), patch.object(
                AbaqusPeriodicConstraint, "_accepted_dual_evidence_pair", dual_pair,
            ):
                started = perf_counter()
                for factor in (.25, .5, .75, 1.):
                    step.solve(until=factor)
                    row["stations"].append(observe_penalty(step, factor))
                    snapshots.append((step.solution.x.array.copy(),
                                      step.response.state.committed_state_vectors().copy(),
                                      step.response.first_piola_stress.values.copy(),
                                      step.response.tangent.values.copy()))
                row["solve_observe_seconds"] = perf_counter() - started
            if counts["residual_observations"] == 0:
                raise AssertionError("No residual-only observations exercised.")
            row.update(counts)
            path = [(i.load_factor, i.iterations, i.accepted_step_lengths, i.linear_converged_reasons)
                    for i in step.accepted_increments]
            frames = step.accepted_history_recorders["material_energy"].frames
            energy = np.array([(f.factor, *f.values) for f in frames])
            duals = np.array([(d["load_factor"], *np.ravel(d["force"]), *np.ravel(d["outgoing_force"]))
                             for d in step.constraint_dual_history.records])
            if expected is None:
                expected = snapshots, path, row["stations"], energy, duals
            if path != expected[1]:
                raise AssertionError("Accepted/Newton/line-search path changed.")
            delta, tangent_delta = 0., 0.
            for actual, previous in zip(snapshots, expected[0], strict=True):
                for a, b in zip(actual[:3], previous[:3], strict=True):
                    np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-10)
                    delta = max(delta, float(np.max(np.abs(a-b), initial=0.)))
                tangent_delta = max(tangent_delta, compare_tangent_tensors(actual[3], previous[3]))
            np.testing.assert_allclose(energy, expected[3], rtol=2e-8, atol=2e-10)
            np.testing.assert_allclose(duals, expected[4], rtol=2e-8, atol=2e-10)
            for actual, previous in zip(row["stations"], expected[2], strict=True):
                for key in ("force_on_structure", "maximum_pressure", "active_reference_area",
                            "maximum_penetration", "peeq_reference_volume_mean", "maximum_peeq"):
                    np.testing.assert_allclose(actual[key], previous[key], rtol=2e-8, atol=2e-10)
                if actual["mpc_equation_mismatch"] >= 1e-10:
                    raise AssertionError("Periodic equations failed.")
            row.update(status="completed", accepted=len(path), newton_iterations=sum(i[1] for i in path),
                       maximum_field_state_stress_difference=delta,
                       maximum_relative_tangent_tensor_error=tangent_delta,
                       maximum_energy_history_difference=float(np.max(np.abs(energy-expected[3]), initial=0.)),
                       maximum_dual_history_difference=float(np.max(np.abs(duals-expected[4]), initial=0.)))
            save()
            print(json.dumps({key: value for key, value in row.items() if key != "stations"}), flush=True)
        if fingerprint() != frozen:
            raise RuntimeError("Source changed during comparison.")
        means = {mode: float(np.mean([r["solve_observe_seconds"] for r in report["records"]
                                     if r["mode"] == mode])) for mode in ("full", "residual")}
        report.update(status="completed", source_unchanged=True, mean_seconds=means,
                      elapsed_reduction_fraction=1. - means["residual"] / means["full"])
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if report["records"]:
            report["records"][-1]["status"] = "failed"
        save()
        raise
    save()


if __name__ == "__main__":
    main()

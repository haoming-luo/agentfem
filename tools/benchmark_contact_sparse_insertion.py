# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Isolated whole-path ABBA comparison of contact stencil integration/insertion.

Use one process and requested numerical thread limits. The pinned row-wise
reference and batched candidate assemble identical current Operators. This
compares four accepted stations, material state and Newton histories; it is
not independent continuum validation or an explicit-composite speed claim.
"""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from unittest.mock import patch

import numpy as np
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc

import agentfem
from agentfem.boundary_models import dolfinx_implicit_facet_contact as contact_backend
from agentfem.boundary_models import implicit_contact as contact_algebra
from agentfem.boundary_models._petsc_contact_rows import add_contact_coo_tangent
from agentfem.checkpointing import atomic_write_text
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
        parser.error("Use positive cell/increment counts divisible by four.")
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
    reference_commit = "bf0033c4"
    reference_path = "src/agentfem/boundary_models/_petsc_contact_rows.py"
    reference_source = subprocess.check_output(
        ["git", "show", f"{reference_commit}:{reference_path}"], cwd=root, text=True)
    tree = ast.parse(reference_source)
    definition = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "add_contact_coo_tangent")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), reference_path, "exec"), namespace)
    legacy = namespace["add_contact_coo_tangent"]
    frozen = fingerprint()
    report = dict(
        schema="agentfem.contact-coo-batching.v1", status="running",
        scope="serial_cyclic_P2_finite_J2_plane_penalty_MPC_not_explicit_composite",
        cells=args.cells, increments=args.increments,
        reference_commit=reference_commit, reference_source_sha256=sha256(reference_source.encode()).hexdigest(),
        reference_strategy="pinned row insertion; disable only adjacent-stencil quadrature coalescing",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        runtime_source_sha256=frozen, platform=platform.platform(),
        versions={"numpy": np.__version__, "dolfinx": dolfinx.__version__,
                  "petsc": PETSc.Sys.getVersion(), "mpi": MPI.Get_library_version().splitlines()[0]},
        thread_environment={key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "FI_PROVIDER")},
        timing_boundary="solve and observation, excludes setup; two complete warmups then ABBA",
        gates={"field_state_stress_rtol": 2e-8, "field_state_stress_atol": 2e-10,
               "per_point_tangent_frobenius_rtol": 2e-10,
               "accepted_factors_newton_linesearch_linear_reasons": "identical"},
        records=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    expected = None
    save()
    try:
        for mode in ("rowwise_warmup", "batched_warmup", "rowwise", "batched", "batched", "rowwise"):
            implementation = add_contact_coo_tangent if mode.startswith("batched") else legacy
            coalesce = contact_algebra._coalesce_trace_tangent if mode.startswith("batched") else lambda n, k: (n, k)
            insertion_calls = 0
            insertion_seconds = 0.
            inserted_entries = 0
            def selected(*a, _implementation=implementation, **kw):
                nonlocal insertion_calls, insertion_seconds, inserted_entries
                began = perf_counter()
                try:
                    return _implementation(*a, **kw)
                finally:
                    insertion_calls += 1
                    inserted_entries += len(a[3])
                    insertion_seconds += perf_counter() - began
            step = make_step(cells=args.cells, degree=2, increments=args.increments,
                             contact_degree=11, bulk_degree=6)
            row = dict(mode=mode, status="running", stations=[])
            report["records"].append(row)
            save()
            snapshots = []
            with patch.object(contact_backend, "add_contact_coo_tangent", selected), patch.object(
                contact_algebra, "_coalesce_trace_tangent", coalesce,
            ):
                started = perf_counter()
                for factor in (.25, .5, .75, 1.):
                    step.solve(until=factor)
                    row["stations"].append(observe_penalty(step, factor))
                    row["stations"][-1]["contact_potential_energy"] = float(
                        step.contact_provider.evaluate().potential_energy)
                    snapshots.append((step.solution.x.array.copy(),
                        step.response.state.committed_state_vectors().copy(),
                        step.response.first_piola_stress.values.copy(),
                        step.response.tangent.values.copy()))
                row["solve_observe_seconds"] = perf_counter() - started
            if insertion_calls == 0:
                raise AssertionError("No contact insertions were exercised.")
            row.update(insertion_calls=insertion_calls, insertion_seconds=insertion_seconds,
                       inserted_entries=inserted_entries)
            path = [(i.load_factor, i.iterations, i.accepted_step_lengths, i.linear_converged_reasons)
                    for i in step.accepted_increments]
            if expected is None:
                expected = snapshots, path, row["stations"]
            if path != expected[1]:
                raise AssertionError("Accepted increment/Newton path changed.")
            delta, tangent_delta = 0., 0.
            for actual, previous in zip(snapshots, expected[0], strict=True):
                for a, b in zip(actual[:3], previous[:3], strict=True):
                    np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-10)
                    delta = max(delta, float(np.max(np.abs(a-b), initial=0.)))
                tangent_delta = max(tangent_delta, compare_tangent_tensors(actual[3], previous[3]))
            for actual, previous in zip(row["stations"], expected[2], strict=True):
                for key in ("force_on_structure", "maximum_pressure", "active_reference_area",
                            "maximum_penetration", "peeq_reference_volume_mean", "maximum_peeq",
                            "contact_potential_energy"):
                    np.testing.assert_allclose(actual[key], previous[key], rtol=2e-8, atol=2e-10)
                if actual["mpc_equation_mismatch"] >= 1e-10:
                    raise AssertionError("Periodic equations failed.")
            row.update(status="completed", accepted=len(path), newton_iterations=sum(i[1] for i in path),
                       maximum_field_state_stress_difference=delta,
                       maximum_relative_tangent_tensor_error=tangent_delta)
            save()
            print(json.dumps({key: value for key, value in row.items() if key != "stations"}), flush=True)
        if fingerprint() != frozen:
            raise RuntimeError("Source changed during comparison.")
        means = {mode: float(np.mean([r["solve_observe_seconds"] for r in report["records"]
                                     if r["mode"] == mode])) for mode in ("rowwise", "batched")}
        report.update(status="completed", source_unchanged=True, mean_seconds=means,
                      elapsed_reduction_fraction=1. - means["batched"] / means["rowwise"])
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if report["records"]:
            report["records"][-1]["status"] = "failed"
        save()
        raise
    save()


if __name__ == "__main__":
    main()

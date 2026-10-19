# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded whole-Step AL penalty-cost screen, not an automatic tuner.

Run from the source checkout with PYTHONPATH=src. Geometry, plastic material,
load path, mesh and acceptance tolerances are fixed within each comparison.
Only the explicit AL numerical penalty changes. The script records failed
candidates and does not promote a setting, silently retune or relax tolerances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
import subprocess
from time import perf_counter

import numpy as np

from agentfem import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--penalties', type=float, nargs='+', default=[500., 1500., 5000.])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--minimum-step-length', type=float, default=1/128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or any(c < 1 for c in args.cells):
        parser.error('At least two repeats and positive cell counts are required.')
    if any(not np.isfinite(p) or p <= 0 for p in args.penalties):
        parser.error('Penalties must be positive and finite.')
    if len(set(args.penalties)) != len(args.penalties):
        parser.error('Penalty candidates must be distinct.')
    if not 0 < args.minimum_step_length <= 1:
        parser.error('The Newton line-search floor must be in (0, 1].')
    if args.output.exists():
        parser.error('Use a fresh output path.')
    root = Path(__file__).resolve().parents[1]
    example = root / 'examples' / 'finite_j2_augmented_plane.py'
    build = runpy.run_path(str(example))['make_step']
    records, comparisons = [], []
    rng = np.random.default_rng(20261019)
    for cells in args.cells:
        # Warm each candidate once; record but exclude that run from timing comparisons.
        jobs = [(p, -1) for p in args.penalties]
        for repeat in range(args.repeats):
            jobs.extend((float(p), repeat) for p in rng.permutation(args.penalties))
        reference = None
        reference_penalty = None
        for penalty, repeat in jobs:
            row = dict(cells_per_edge=cells, penalty=penalty, repeat=repeat,
                       warmup=repeat == -1, status='not_started')
            started = perf_counter()
            try:
                _, step = build(cells=cells, penalty=penalty,
                                minimum_step_length=args.minimum_step_length)
                row['setup_seconds'] = perf_counter() - started
                started = perf_counter()
                result = step.solve_result()
                row['solve_and_result_seconds'] = perf_counter() - started
                if result.status != 'completed':
                    raise RuntimeError(f'Unexpected solve status {result.status}')
                response = step.contact_provider.normal_response()
                u = step.solution.x.array.copy()
                state = step.response.state.committed_state_vectors().copy()
                force = step.contact_provider.evaluate().contact_force_on_structure.copy()
                snapshots = [(
                    item.load_factor, item.solution.x.array.copy(),
                    {name: getattr(value, 'function', value).x.array.copy()
                     for name, value in item.fields.items()},
                ) for item in step.snapshots]
                if reference is None:
                    reference = (u, state, force, snapshots)
                    reference_penalty = penalty
                errors = [float(np.max(np.abs(a - b), initial=0.))
                          for a, b in zip((u, state, force), reference[:3], strict=True)]
                path_displacement_error = path_field_error = 0.
                for actual, expected in zip(snapshots, reference[3], strict=True):
                    if actual[0] != expected[0] or actual[2].keys() != expected[2].keys():
                        raise ValueError('Accepted snapshot identities differ.')
                    path_displacement_error = max(path_displacement_error, float(np.max(np.abs(actual[1] - expected[1]))))
                    for name in actual[2]:
                        scale = max(1., float(np.max(np.abs(expected[2][name]))))
                        path_field_error = max(path_field_error, float(np.max(np.abs(
                            actual[2][name] - expected[2][name]
                        ))) / scale)
                row.update(
                    status='completed', maximum_displacement_difference=errors[0],
                    reference_penalty=reference_penalty,
                    maximum_material_state_difference=errors[1],
                    maximum_force_difference=errors[2],
                    maximum_path_displacement_difference=path_displacement_error,
                    maximum_scaled_path_field_difference=path_field_error,
                    within_declared_response_screen=(errors[0] <= 3e-7 and errors[1] <= 1e-6
                                                    and errors[2] <= 5e-4
                                                    and path_displacement_error <= 3e-7
                                                    and path_field_error <= 5e-4),
                    augmentations=sum(i.normal_augmentations for i in step.accepted_increments),
                    newton_iterations=sum(i.iterations for i in step.accepted_increments),
                    line_search_evaluations=sum(i.line_search_evaluations for i in step.accepted_increments),
                    maximum_inner_newton_iterations=max(i.adaptation_iterations for i in step.accepted_increments),
                    diagnostics=step.contact_provider.normal_multiplier_state.acceptance_diagnostics(
                        response.evaluation, response.record,
                    ),
                )
            except (RuntimeError, ValueError) as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                           failure_elapsed_seconds=perf_counter() - started)
            records.append(row)
        for penalty in args.penalties:
            measured = [r for r in records if r['cells_per_edge'] == cells
                        and r['penalty'] == penalty and not r['warmup']]
            passed = all(r['status'] == 'completed' and r['within_declared_response_screen']
                         for r in measured)
            times = [r['solve_and_result_seconds'] for r in measured if r['status'] == 'completed']
            comparisons.append(dict(cells_per_edge=cells, penalty=penalty,
                                    all_measured_runs_passed=passed,
                                    median_seconds=float(np.median(times)) if times else None))
    report = dict(
        schema='agentfem.normal-augmentation-parameter-screen.v1',
        agentfem_version=__version__, platform=platform.platform(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        tracked_changes=subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=root, text=True).splitlines(),
        example_sha256=hashlib.sha256(example.read_bytes()).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='serial_fixed_plane_plastic_example_not_general_performance_or_validation',
        acceptance_units='m_Pa', penalty_units='Pa/m', physical_energy_audit_available=False,
        minimum_step_length=args.minimum_step_length,
        reference='first_successful_same_mesh_candidate_not_independent_scientific_reference',
        records=records, comparisons=comparisons,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(comparisons, indent=2))
    print(f'Evidence: {args.output.resolve()}')


if __name__ == '__main__':
    main()

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Same-model comparison of fresh versus scoped KSP ownership.

Research driver: substitutes only the private object-lifecycle comparison
switch. Every Newton iteration still assembles and solves its current tangent.
No public numerical setting, constitutive update or contact tolerance changes.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import runpy
import statistics
from time import perf_counter
from unittest.mock import patch

import numpy as np
from mpi4py import MPI

from agentfem import solvers
from agentfem._affine_equilibrium import solve_serial_affine_equilibrium
from agentfem.checkpointing import atomic_write_text
from agentfem.mechanics import _affine_normal_augmentation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, default=8)
    parser.add_argument('--increments', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        parser.error('This experiment requires one process.')
    if args.repeats < 2 or args.output.exists():
        parser.error('Use at least two repetitions and a fresh output path.')
    if any(n < 4 or n % 4 for n in (args.cells, args.increments)):
        parser.error('Use positive cell/increment counts divisible by four.')
    root = Path(__file__).resolve().parents[1]
    example = root / 'examples/finite_j2_periodic_augmented_plane.py'
    build = runpy.run_path(str(example))['make_step']
    report = dict(
        schema='agentfem.affine-linear-reuse-screen.v1',
        scope='serial_supported_periodic_J2_plane_not_explicit_composite_or_external_validation',
        platform=platform.platform(), cells=args.cells, increments=args.increments,
        penalty=40000., affine_end_face_support=True,
        timing_scope='solve_only_excludes_model_construction_and_array_comparison',
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                       for name in ('src/agentfem/_affine_equilibrium.py',
                                    'src/agentfem/solvers.py', str(example.relative_to(root)),
                                    str(Path(__file__).resolve().relative_to(root)))},
        records=[], complete=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False)+'\n')
    expected = None
    create = solvers.create_ksp
    for repeat in range(-1, args.repeats):
        order = ('fresh', 'scoped') if repeat % 2 == 0 else ('scoped', 'fresh')
        for mode in order:
            row = dict(mode=mode, repeat=repeat, warmup=repeat < 0, status='running')
            report['records'].append(row)
            save()
            counters = {'created_ksp': 0}
            def create_counted(*a, _counters=counters, **kw):
                _counters['created_ksp'] += 1
                return create(*a, **kw)
            def selected_equilibrium(*a, _reuse=mode == 'scoped', **kw):
                return solve_serial_affine_equilibrium(*a, **kw,
                                                       reuse_linear_solver=_reuse)
            try:
                with ExitStack() as stack:
                    for owner in (solvers, _affine_normal_augmentation):
                        stack.enter_context(patch.object(owner, 'solve_serial_affine_equilibrium',
                                                         selected_equilibrium))
                    stack.enter_context(patch.object(solvers, 'create_ksp', create_counted))
                    step = build(cells=args.cells, increments=args.increments,
                                 penalty=40000., affine_end_face_support=True)
                    step.output_every = None
                    started = perf_counter()
                    step.solve()
                    row['solve_seconds'] = perf_counter()-started
                arrays = [step.solution.x.array.copy(),
                          step.response.state.committed_state_vectors().copy(),
                          step.contact_provider.evaluate().nodal_structural_residual.copy()]
                work = step.accepted_history_recorders['normal_contact_work:interior_plane'].records
                arrays.append(np.array([item['work'] for item in work]))
                path = [(i.load_factor, i.iterations, i.accepted_step_lengths,
                         i.linear_converged_reasons) for i in step.accepted_increments]
                if expected is None:
                    expected = arrays, path
                if path != expected[1]:
                    raise AssertionError('Accepted Newton path changed.')
                delta = []
                for value, reference in zip(arrays, expected[0], strict=True):
                    np.testing.assert_allclose(value, reference, rtol=1e-10, atol=1e-10)
                    delta.append(float(np.max(abs(value-reference), initial=0.)))
                row.update(status='completed', **counters,
                           newton_iterations=sum(i.iterations for i in step.accepted_increments),
                           maximum_array_difference=max(delta))
            except BaseException as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                save()
                raise
            save()
            print(json.dumps(row), flush=True)
    medians = {mode: statistics.median(r['solve_seconds'] for r in report['records']
                                      if r['mode'] == mode and not r['warmup'])
               for mode in ('fresh', 'scoped')}
    report.update(complete=True, median_seconds=medians,
                  elapsed_reduction=1-medians['scoped']/medians['fresh'])
    save()
    print(json.dumps({'median_seconds': medians, 'elapsed_reduction': report['elapsed_reduction']}))


if __name__ == '__main__':
    main()

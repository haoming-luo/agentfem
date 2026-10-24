# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded same-model linear-policy screen; never changes the solver default.

Scientific observables at four common stations are compared with the declared
direct baseline. Numerical tolerances and load path remain fixed. Backend
changes need not reproduce the same Newton iteration count or floating bits.
This is an exploratory serial implicit benchmark, not explicit-composite timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
from time import perf_counter
from unittest.mock import patch

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from agentfem import solvers
from agentfem.checkpointing import atomic_write_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, default=8)
    parser.add_argument('--increments', type=int, default=16)
    parser.add_argument('--policies', nargs='+', choices=('default', 'mumps', 'gmres_ilu'),
                        default=['default', 'mumps', 'gmres_ilu'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1 or args.output.exists():
        parser.error('Use one process and a fresh evidence path.')
    if args.policies[0] != 'default':
        parser.error('The same-input default reference must be run first.')
    root = Path(__file__).resolve().parents[1]
    example = root / 'examples/finite_j2_periodic_augmented_plane.py'
    make_step = runpy.run_path(str(example))['make_step']
    choices = {
        'default': solvers.direct_solver(),
        'mumps': solvers.direct_solver(package='mumps'),
        'gmres_ilu': solvers.LinearSolverOptions(ksp_type='gmres', pc_type='ilu',
                                                rtol=1e-12, atol=1e-14, max_it=1000),
    }
    report = dict(schema='agentfem.affine-linear-policy-screen.v1',
                  platform=platform.platform(), cells=args.cells, increments=args.increments,
                  scope='serial_supported_periodic_finite_J2_not_explicit_composite',
                  timing_scope='solve_only_excludes_setup_and_station_comparison',
                  source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                 for name in ('src/agentfem/solvers.py',
                                              'src/agentfem/_affine_equilibrium.py',
                                              str(example.relative_to(root)),
                                              str(Path(__file__).resolve().relative_to(root)))},
                  mumps_available=PETSc.Sys.hasExternalPackage('mumps'), records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False)+'\n')
    references = []
    original_solve = solvers.solve_matrix_system
    for name in args.policies:
        policy = choices[name]
        row = dict(policy=name, options=policy.summary(), status='running', stations=[],
                   solve_seconds=0., actual_factor_backends=[], linear_iterations=0)
        report['records'].append(row)
        save()
        def measured_solve(*a, _row=row, **kw):
            info = original_solve(*a, **kw)
            _row['linear_iterations'] += info.iterations
            ksp = kw.get('ksp')
            if ksp is not None and ksp.getPC().getType() in ('lu', 'ilu'):
                backend = ksp.getPC().getFactorSolverType()
                if backend not in _row['actual_factor_backends']:
                    _row['actual_factor_backends'].append(backend)
            return info
        try:
            step = make_step(cells=args.cells, increments=args.increments,
                             affine_end_face_support=True, penalty=40000., linear_solver=policy)
            step.output_every = None
            for i, factor in enumerate((.25, .5, .75, 1.)):
                with patch.object(solvers, 'solve_matrix_system', measured_solve):
                    started = perf_counter()
                    step.solve(until=factor)
                    row['solve_seconds'] += perf_counter()-started
                arrays = [step.solution.x.array.copy(),
                          step.response.state.committed_state_vectors().copy(),
                          step.contact_provider.evaluate().nodal_structural_residual.copy()]
                work = step.accepted_history_recorders['normal_contact_work:interior_plane'].records[-1]['work']
                arrays.append(np.asarray(work))
                if name == 'default':
                    references.append(arrays)
                deltas = []
                for a, b in zip(arrays, references[i], strict=True):
                    # Substantially below the already observed mesh/path error;
                    # compare forces absolutely near contact opening, not by a
                    # relative metric divided by an almost-zero reaction.
                    np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7)
                    deltas.append(float(np.max(abs(a-b), initial=0.)))
                row['stations'].append(dict(factor=factor, maximum_array_differences=deltas))
                save()
            row.update(status='completed', newton_iterations=sum(i.iterations for i in step.accepted_increments),
                       maximum_penetration=max(i.checks['normal_acceptance']['maximum_penetration']
                                               for i in step.accepted_increments),
                       maximum_complementarity=max(i.checks['normal_acceptance']['maximum_complementarity']
                                                   for i in step.accepted_increments))
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        print(json.dumps({k:v for k,v in row.items() if k != 'stations'}), flush=True)
        if name == 'default' and row['status'] != 'completed':
            break


if __name__ == '__main__':
    main()

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Operational observations; never a solver control or scientific verdict."""
import json
import os
from pathlib import Path
import sys
import time

PHASES = {
    'step_started': 'initializing', 'increment_started': 'increment_started',
    'assembly_started': 'assembling', 'linear_solve_started': 'linear_solving',
    'material_update_started': 'material_updating', 'iteration': 'iterating',
    'increment_cutback': 'retrying', 'increment_converged': 'increment_accepted',
    'time_increment': 'increment_accepted', 'step_failed': 'solve_failed',
    'step_completed': 'solve_completed', 'result_saved': 'result_saved',
    'case_returned': 'case_returned', 'case_failed': 'case_failed',
}
_warned = set()


def record_event(record, *, rank=0, directory=None):
    directory = directory or os.environ.get('AGENTFEM_OUTPUT_DIR')
    if rank != 0 or not directory:
        return
    path = Path(directory) / 'runtime-events.jsonl'
    row = {**record, 'schema': 'agentfem.runtime-event.v1',
           'phase': PHASES.get(record.get('kind'), 'provider_event'),
           'observed_at': time.time(), 'process_exit_observed': False}
    try:
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    except (OSError, ValueError, TypeError) as error:
        key = str(path)
        if key not in _warned:
            print(f'AgentFEM runtime journal unavailable: {type(error).__name__}', file=sys.stderr)
            _warned.add(key)


def phase(kind, *, rank=0, directory=None, **details):
    record_event({'kind': kind, **details}, rank=rank, directory=directory)

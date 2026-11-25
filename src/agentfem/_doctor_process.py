# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Publish CLI health only after the numerical interpreter actually exits."""
import json
import os
import subprocess
import sys


def observe(command, *, timeout=30):
    from .mpi_runtime import _terminate_process_group
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=os.name == 'posix')
    try:
        out, err = process.communicate(timeout=timeout)
        return process.returncode, out, err, False
    except subprocess.TimeoutExpired:
        _terminate_process_group(process, grace=1)
        out, err = process.communicate(timeout=5)
        return 124, out, err, True
    except BaseException:
        _terminate_process_group(process, grace=1)
        process.communicate(timeout=5)
        raise


def main(argv):
    if any(key in os.environ for key in ('PMI_SIZE', 'PMIX_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE')):
        message = 'Run doctor once outside mpiexec; use mpi-run for parallel validation.'
        code, timed_out, out, err = 2, False, '', ''
    else:
        worker = 'import sys; from agentfem.cli import main; raise SystemExit(main(sys.argv[1:]))'
        code, out, err, timed_out = observe([sys.executable, '-c', worker, *argv])
        message = ('Numerical health check did not exit within 30 seconds. Its process group was stopped. '
                   'Check MPI/PETSc shutdown and the active communication provider; no runtime settings were changed.')
    if timed_out or not out:
        record = {'schema': 'agentfem.cli-error', 'schema_version': '0.1.0',
                  'status': 'failed', 'solver_ready': False,
                  'process_exit_observed': not timed_out,
                  'error': {'code': 'AFM-RUNTIME-EXIT-TIMEOUT' if timed_out else 'AFM-RUNTIME-PROBE',
                            'message': message if timed_out or not err else err[-1000:]}}
        print(json.dumps(record) if '--json' in argv else record['error']['message'], flush=True)
        return code or 2
    if '--json' in argv:
        try:
            record = json.loads(out)
            record['process_exit_observed'] = True
            record['process_return_code'] = code
            if code:
                record['solver_ready'] = False
            out = json.dumps(record) + '\n'
        except (ValueError, TypeError):
            pass  # Preserve the original diagnostic on an unexpected failure.
    print(out, end='', flush=True)
    print(err, end='', file=sys.stderr, flush=True)
    return code

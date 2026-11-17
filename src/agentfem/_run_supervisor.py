# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Observe child exit outside MPI and outside the application's finalizers."""
import json
import os
import subprocess
import sys
import time
import uuid


def supervise(args, argv):
    from .cli import _project
    from .project import RunContext
    from ._atomic_text import atomic_write_text
    from .mpi_runtime import _terminate_process_group

    if args.inside_mpi or any(key in os.environ for key in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'PMIX_SIZE')):
        raise ValueError('AFM-SUPERVISE-NESTED: start supervision outside an MPI job.')
    config = _project(args.project)
    run_id = args.run_id or 'supervised-' + uuid.uuid4().hex[:16]
    context = RunContext.create(config, run_id=run_id, run_name=args.name,
                                output_directory=args.output).prepare()
    shell, separator, case = list(argv), [], []
    if '--' in shell:
        index = shell.index('--')
        shell, separator, case = shell[:index], ['--'], shell[index+1:]
    cleaned = []
    skip = False
    for token in shell:
        if skip:
            skip = False
            continue
        if token == '--supervise':
            continue
        if token == '--timeout':
            skip = True
            continue
        if token.startswith('--timeout='):
            continue
        cleaned.append(token)
    if not args.run_id:
        cleaned.extend(('--run-id', run_id))
    command = [sys.executable, '-m', 'agentfem.cli', *cleaned, *separator, *case]
    if args.mpi > 1:
        from ._cli_entry import _probe_vendor
        from .mpi_runtime import audit_mpi_runtime, mpi_command
        audit = audit_mpi_runtime(vendor=_probe_vendor(), rank_count=1)
        index = command.index('--') if '--' in command else len(command)
        command.insert(index, '--inside-mpi')
        command = mpi_command(args.mpi, command, audit=audit)
    path = context.output_directory / 'process.json'
    started = time.time()
    record = {'schema': 'agentfem.process-observation.v1', 'run_id': run_id,
              'status': 'starting', 'return_code': None, 'started_at': started,
              'scientific_validity': 'not_evaluated', 'result_manifest': str(context.manifest_path)}

    def write():
        atomic_write_text(path, json.dumps(record, indent=2, allow_nan=False) + '\n')

    write()
    process = None
    try:
        process = subprocess.Popen(command, start_new_session=(os.name == 'posix'))
        record.update(status='running', pid=process.pid)
        write()
        code = process.wait(timeout=args.timeout)
        record.update(status='exited' if code == 0 else 'failed', return_code=code)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process, grace=2.0)
        record.update(status='timed_out', return_code=process.returncode)
        code = 124
    except BaseException as error:
        if process is not None and process.poll() is None:
            _terminate_process_group(process, grace=2.0)
        record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__)
        raise
    finally:
        record.update(finished_at=time.time(), manifest_exists=context.manifest_path.is_file(),
                      process_exit_observed=process is not None and process.poll() is not None)
        write()
    print(f'AgentFEM process {record["status"]}; record: {path}', file=sys.stderr, flush=True)
    return code

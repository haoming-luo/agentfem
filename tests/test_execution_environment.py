# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from agentfem import _execution_environment as env


def test_longest_mount_and_escaped_path():
    mounts = ('1 0 0:1 / / rw - ext4 root rw\n'
              '2 1 0:2 / /data\\040drive rw - 9p D: rw,aname=drvfs\n'
              '3 2 0:3 / /data\\040drive/native rw - ext4 root rw\n')
    assert env._storage('/data drive/project', mounts)['windows_mount']
    assert not env._storage('/data drive/native/project', mounts)['windows_mount']
    assert not env._storage('/home/user/project', mounts)['windows_mount']


def test_wsl_advisories_do_not_modify_environment(monkeypatch):
    monkeypatch.setattr(env.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(env.platform, 'release', lambda: 'microsoft-standard-WSL2')
    monkeypatch.setenv('OMP_NUM_THREADS', '8')
    monkeypatch.setattr(env.os, 'sched_getaffinity', lambda _: {0, 1}, raising=False)
    monkeypatch.setattr(env, '_read', lambda path: (
        'MemTotal: 1000 kB\nMemAvailable: 50 kB\nSwapTotal: 100 kB\nSwapFree: 10 kB\n'
        if path.endswith('meminfo') else '1 0 0:1 / /mnt/e rw - 9p E: rw,aname=drvfs'))
    record = env.execution_environment(project='/mnt/e/case', output='/mnt/e/case/outputs')
    assert {item['code'] for item in record['findings']} == {
        'AFM-PERF-WSL-STORAGE', 'AFM-PERF-MEMORY-PRESSURE', 'AFM-PERF-THREAD-BUDGET'}
    assert record['memory_bytes']['MemTotal'] == 1024000
    assert record['visible_cpus'] == 2
    assert env.os.environ['OMP_NUM_THREADS'] == '8'
    assert not record['settings_modified']


def test_non_wsl_no_false_windows_path_warning(monkeypatch):
    monkeypatch.setattr(env.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(env, '_read', lambda _: '')
    record = env.execution_environment(project='/mnt/e', output='/mnt/e/out')
    assert not record['under_wsl']
    assert not record['storage']
    assert not any(item['code'] == 'AFM-PERF-WSL-STORAGE' for item in record['findings'])


def test_path_fallback_and_missing_proc(monkeypatch):
    monkeypatch.setattr(env.platform, 'system', lambda: 'Linux')
    monkeypatch.setenv('WSL_DISTRO_NAME', 'Ubuntu')
    monkeypatch.setattr(env, '_read', lambda _: '')
    record = env.execution_environment(project='/mnt/e/case', output='/home/user/out')
    assert record['storage']['project']['detection'] == 'path_fallback'
    assert record['storage']['project']['windows_mount']
    assert not record['storage']['output']['windows_mount']
    assert not record['memory_bytes']


def test_operational_environment_is_allowlisted_and_never_changed(monkeypatch):
    monkeypatch.setattr(env.platform, 'system', lambda: 'Darwin')
    for name in ('FI_PROVIDER', 'MPICH_ASYNC_PROGRESS', 'MPIR_CVAR_ASYNC_PROGRESS',
                 'I_MPI_ASYNC_PROGRESS'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('FI_PROVIDER', 'tcp')
    monkeypatch.setenv('MPICH_ASYNC_PROGRESS', '0')
    monkeypatch.setenv('VECLIB_MAXIMUM_THREADS', '1')
    monkeypatch.setenv('NUMEXPR_NUM_THREADS', '2')
    monkeypatch.setenv('PRIVATE_SERVICE_TOKEN', 'must-not-be-recorded')
    before = dict(env.os.environ)
    record = env.execution_environment(project='/tmp/project', output='/tmp/output')
    assert record['mpi_environment'] == {'FI_PROVIDER': 'tcp', 'MPICH_ASYNC_PROGRESS': '0'}
    assert record['thread_environment']['VECLIB_MAXIMUM_THREADS'] == '1'
    assert record['thread_environment']['NUMEXPR_NUM_THREADS'] == '2'
    assert 'must-not-be-recorded' not in repr(record)
    assert dict(env.os.environ) == before
    assert not record['settings_modified']
    assert record['scope'] == 'startup_snapshot_not_execution_route_or_speed_prediction'


def test_unset_mpi_environment_is_unknown_not_an_inferred_default(monkeypatch):
    for name in ('FI_PROVIDER', 'MPICH_ASYNC_PROGRESS', 'MPIR_CVAR_ASYNC_PROGRESS',
                 'I_MPI_ASYNC_PROGRESS'):
        monkeypatch.delenv(name, raising=False)
    record = env.execution_environment(project='/tmp/project', output='/tmp/output')
    assert record['mpi_environment'] == {}

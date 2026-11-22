# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import json
import os
import numpy as np
import pytest
from mpi4py import MPI
from agentfem import diagnostics, fields, mesh, models, studies
from agentfem import _resource_budget as resources


def model():
    d = mesh.rectangle((0., 0.), (2., 1.), (4, 2), comm=MPI.COMM_WORLD)
    m = models.create(mesh=d, study=studies.dynamic_solid(dimension=2, assumption='plane_stress'))
    u = m.field(fields.displacement(d))
    return m,u


def test_payload_budget_is_not_file_or_solver_memory_prediction(tmp_path):
    m,u = model()
    before = u.value.x.array.copy(), dict(os.environ)
    path = tmp_path/'not created'/'outputs'
    report = diagnostics.resource_budget(m, target=u, fields=(u, 'S'),
        step_options={'steps': 6, 'save_every': 3}, output_directory=path)
    assert report['discretization']['registered_scalar_dofs'] == 30
    out = report['output']
    assert out['planned_frames'] == 3
    assert out['coefficient_payload_bytes'] == 3 * 30 * u.value.x.array.dtype.itemsize
    assert out['payload_complete'] is False
    assert out['fields'][1]['bytes_per_frame'] is None
    assert out['file_size_bytes'] is None and report['solver']['peak_memory_bytes'] is None
    assert len(report['rank_resources']) == MPI.COMM_WORLD.size
    assert not path.exists() and not m.steps
    assert dict(os.environ) == before[1]
    np.testing.assert_array_equal(u.value.x.array, before[0])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('frames', [0, -1, True, 1.5])
def test_invalid_frames_fail_without_solve(frames):
    m,u = model()
    with pytest.raises((ValueError, RuntimeError), match='positive integer'):
        diagnostics.resource_budget(m, fields=(u,), frames=frames)


def test_unknown_output_payload_and_explicit_frames():
    m,_ = model()
    report = diagnostics.resource_budget(m, fields=('S',), frames=np.int64(4))
    assert report['output']['planned_frames'] == 4
    assert report['output']['coefficient_payload_bytes'] is None
    assert report['solver']['selection'] == 'provider_default_not_built'


def test_nested_cgroup_limit_includes_parent_when_child_unlimited(monkeypatch):
    files = {'/proc/self/cgroup': '0::/jobs/run\n',
        '/proc/self/mountinfo': '1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n',
        '/sys/fs/cgroup/jobs/run/memory.max': 'max',
        '/sys/fs/cgroup/jobs/memory.max': '1000', '/sys/fs/cgroup/jobs/memory.current': '800',
        '/sys/fs/cgroup/memory.max': '2000', '/sys/fs/cgroup/memory.current': '1500'}
    monkeypatch.setattr(resources.environment, '_read', lambda p: files.get(str(p), ''))
    records = resources.memory_limits()['cgroups']
    assert len(records) == 2
    assert min(r['remaining_bytes'] for r in records) == 200
    assert min(r['limit_bytes'] for r in records) == 1000


def test_missing_and_malformed_limits_do_not_claim_infinite_memory(monkeypatch):
    monkeypatch.setattr(resources.environment, '_read', lambda _: '')
    assert resources.memory_limits()['cgroups'] == []


def test_frame_declarations_must_agree_across_ranks():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip('requires multiple ranks')
    m,u = model()
    with pytest.raises(RuntimeError, match='differs across MPI ranks'):
        diagnostics.resource_budget(m, fields=(u,), frames=1+MPI.COMM_WORLD.rank)

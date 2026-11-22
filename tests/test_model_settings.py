# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import json
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from agentfem import constitutive, fields, mesh, models, provenance, studies


def build(*, young=1000., force=2., size=(4, 2), end=2., component=0):
    d = mesh.rectangle((0., 0.), (end, 1.), size, comm=MPI.COMM_WORLD)
    m = models.create(mesh=d, study=studies.static_solid(dimension=2, assumption='plane_stress'))
    u = m.field(fields.displacement(d))
    m.material(constitutive.isotropic_elastic(young=young, poisson=.3, density=1.))
    left = mesh.face(d, axis='x', value=0., tag=1, name='left')
    right = mesh.face(d, axis='x', value=end, tag=2, name='right')
    m.fix(u, on=left, component=component)
    m.traction((force, 0.), on=right)
    return m, u


def freeze(**kw):
    m, u = build(**kw)
    return provenance.model_settings(m, target=u)


@pytest.mark.parametrize('change,word', [({'young': 2000.}, 'materials'), ({'force': 3.}, 'loads'),
    ({'size': (6, 2)}, 'mesh'), ({'end': 3.}, 'mesh'), ({'component': 1}, 'constraint_components')])
def test_effective_setting_changes(change, word):
    a, b = freeze(), freeze(**change)
    report = provenance.compare_model_settings(a, b)
    assert report['status'] == 'changed'
    assert any(word in item['path'] for item in report['scientific_changes'])
    assert not report['operational_changes']
    json.dumps(report, allow_nan=False)


def test_snapshot_detached_coefficients_and_operational_difference():
    m,u = build()
    first = provenance.model_settings(m, target=u, step_options={'output': 'a.xdmf'})
    original = json.dumps(first, sort_keys=True)
    second = provenance.model_settings(m, target=u, step_options={'output': 'b.xdmf'})
    report = provenance.compare_model_settings(first, second)
    assert report['recorded_settings_equal'] and report['operational_changes']
    m.loads[0].value.value[:] = [7., 0.]
    after = provenance.model_settings(m, target=u)
    assert not provenance.compare_model_settings(first, after)['recorded_settings_equal']
    assert json.dumps(first, sort_keys=True) == original


def test_opaque_expression_is_inconclusive_and_no_state_changes():
    m,u = build()
    right = m.loads[0].location
    m.loads.clear()
    m.traction((ufl.SpatialCoordinate(m.domain)[0], 0.), on=right)
    before = u.value.x.array.copy()
    a = provenance.model_settings(m, target=u)
    report = provenance.compare_model_settings(a, a)
    assert report['status'] == 'inconclusive' and not report['coverage_complete']
    assert a['coverage']['gaps']
    assert not m.steps
    np.testing.assert_array_equal(u.value.x.array, before)


def test_source_change_and_tampered_snapshot(tmp_path):
    # Each rank may own a distinct file path; source bytes are a separate audit.
    p = tmp_path/'case.py'; p.write_text('x=1\n')
    m,u = build()
    a = provenance.model_settings(m, target=u, source_files=(p,))
    p.write_text('x=2\n')
    b = provenance.model_settings(m, target=u, source_files=(p,))
    report = provenance.compare_model_settings(a, b)
    assert report['recorded_settings_equal'] and report['source_changes']
    b['scientific_inputs']['record']['target'] = 'tampered'
    with pytest.raises(ValueError, match='fingerprint'):
        provenance.compare_model_settings(a, b)


def test_rank_mismatched_options_fail_before_region_collectives():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip('requires two ranks')
    m,u = build()
    with pytest.raises(RuntimeError, match='differs across MPI ranks'):
        provenance.model_settings(m, target=u, step_options={'increments': MPI.COMM_WORLD.rank+1})


def test_changed_tag_membership_with_same_name_is_visible():
    a = freeze()
    m,u = build()
    m.loads.clear()
    top = mesh.face(m.domain, axis='y', value=1., tag=2, name='right')
    m.traction((2., 0.), on=top)
    b = provenance.model_settings(m, target=u)
    report = provenance.compare_model_settings(a, b)
    assert any('region_membership' in item['path'] for item in report['scientific_changes'])


def test_old_result_has_no_invented_snapshot(tmp_path, capsys):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip('serial CLI rendering; collective model settings tested separately')
    from agentfem import cli
    p = tmp_path/'old.json'
    p.write_text(json.dumps({'metadata': None}))
    assert cli.main(['compare', str(p), str(p), '--settings', '--json']) == 2
    error = json.loads(capsys.readouterr().out)
    assert 'model_settings snapshot' in error['error']['message']

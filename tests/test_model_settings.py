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


def test_coincident_seam_nodes_are_not_unchanged_geometry():
    comm = MPI.COMM_SELF
    points = np.array([[0.,0.], [1.,0.], [2.,0.], [0.,1.], [1.,1.], [2.,1.]])
    snapshots = []
    for seam in (False, True):
        coordinates = np.vstack((points, points[[1,4]])) if seam else points
        cells = np.array([[0,1,3,4], [6,2,7,5] if seam else [1,2,4,5]], dtype=np.int64)
        d = mesh.from_arrays(cells=cells, coordinates=coordinates, cell_type='quadrilateral', comm=comm)
        m = models.create(mesh=d, study=studies.static_solid(dimension=2, assumption='plane_stress'))
        u = m.field(fields.displacement(d))
        m.material(constitutive.isotropic_elastic(young=1000., poisson=.3, density=1.))
        snapshots.append(provenance.model_settings(m, target=u))
    report = provenance.compare_model_settings(*snapshots)
    assert not report['recorded_settings_equal']
    assert any('mesh' in item['path'] for item in report['scientific_changes'])


def test_tagged_sides_of_coincident_seam_remain_distinct():
    from dolfinx import mesh as dm
    points = np.array([[0.,0.], [1.,0.], [2.,0.], [0.,1.], [1.,1.], [2.,1.], [1.,0.], [1.,1.]])
    d = mesh.from_arrays(cells=np.array([[0,1,3,4], [6,2,7,5]], dtype=np.int64),
        coordinates=points, cell_type='quadrilateral', comm=MPI.COMM_SELF)
    facets = dm.locate_entities_boundary(d, 1, lambda x: np.isclose(x[0], 1.))
    assert len(facets) == 2
    snapshots = []
    for facet in facets:
        m = models.create(mesh=d, study=studies.static_solid(dimension=2, assumption='plane_stress'))
        u = m.field(fields.displacement(d))
        m.material(constitutive.isotropic_elastic(young=1000., poisson=.3, density=1.))
        tags = dm.meshtags(d, 1, np.array([facet], dtype=np.int32), np.array([1], dtype=np.int32))
        region = mesh.BoundaryRegion('seam', d, None, 1, tags, ufl.Measure('ds', domain=d, subdomain_data=tags))
        m.traction((2.,0.), on=region)
        snapshots.append(provenance.model_settings(m, target=u))
    report = provenance.compare_model_settings(*snapshots)
    assert any('region_membership' in item['path'] for item in report['scientific_changes'])


def test_source_records_keep_v1_hashes_and_refresh_after_geometry_change():
    from agentfem._model_settings import _SourceNodes
    m,u = build()
    d = m.domain
    records = _SourceNodes(d)
    for nodes in d.geometry.dofmaps[0][:3]:
        reference = sorted((int(d.geometry.input_global_indices[i]),
            *d.geometry.x[i, :d.geometry.dim].tolist()) for i in nodes)
        assert provenance.content_fingerprint(records(nodes)) == provenance.content_fingerprint(reference)
    first = provenance.model_settings(m, target=u)
    d.geometry.x[:, 1] *= 2.
    second = provenance.model_settings(m, target=u)
    assert not provenance.compare_model_settings(first, second)['recorded_settings_equal']


def test_region_evaluation_failure_reaches_every_rank(monkeypatch):
    from dolfinx import mesh as dm
    m,u = build()
    if MPI.COMM_WORLD.rank == 0:
        def fail(*args, **kwargs):
            raise RuntimeError('injected region evaluation failure')
        monkeypatch.setattr(dm, 'entities_to_geometry', fail)
    with pytest.raises(RuntimeError, match='injected region evaluation failure'):
        provenance.model_settings(m, target=u)


def test_rank_local_foreign_region_is_rejected_by_shared_plan():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip('requires two ranks')
    from dataclasses import replace
    m,u = build()
    other = mesh.rectangle((0.,0.), (2.,1.), (4,2), comm=MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank == 0:
        # Keep names/IR summaries identical: only domain ownership differs.
        m.regions[0] = replace(m.regions[0], domain=other)
    with pytest.raises(RuntimeError, match='region identity plan.*differs across MPI ranks'):
        provenance.model_settings(m, target=u)


def test_compare_does_not_trust_stale_complete_flags():
    from copy import deepcopy
    m,u = build()
    right = m.loads[0].location
    m.loads.clear()
    m.traction((ufl.SpatialCoordinate(m.domain)[0], 0.), on=right)
    snapshot = provenance.model_settings(m, target=u)
    stale = deepcopy(snapshot)
    stale['coverage'].update(complete=True, gaps=[])
    before = json.dumps(stale, sort_keys=True)
    report = provenance.compare_model_settings(stale, stale)
    assert report['status'] == 'inconclusive' and not report['coverage_complete']
    assert json.dumps(stale, sort_keys=True) == before
    assert report['reference_coverage']['gaps']


def test_output_policy_is_retained_separately_from_physics():
    from agentfem import results
    m,u = build()
    snapshots = [provenance.model_settings(m, target=u, step_options={
        'output': results.OutputPlan(directory='outputs', field=results.field_output('U', every=n))})
        for n in (1, 3)]
    report = provenance.compare_model_settings(*snapshots)
    assert report['recorded_settings_equal']
    assert any(item['path'].endswith('every') for item in report['operational_changes'])

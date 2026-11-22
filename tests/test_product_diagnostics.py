# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Local actionable evidence remains complete without changing telemetry."""
import json
import pytest
from mpi4py import MPI
from agentfem import cli, constitutive, feedback, fields, mesh, models, results, studies
from agentfem.validation import ModelValidationError


def model():
    domain = mesh.rectangle((0.,0.),(1.,1.),(2,2),comm=MPI.COMM_SELF)
    m = models.create(mesh=domain,study=studies.static_solid(dimension=2,assumption='plane_stress'))
    u = m.field(fields.displacement(domain))
    m.material(constitutive.isotropic_elastic(young=100.,poisson=.3,density=1.))
    return m,u,domain


@pytest.mark.parametrize('kind', ['option','capability','empty','duplicate','schedule','outside'])
def test_owner_error_evidence_survives_run_and_diagnose(tmp_path, monkeypatch, capsys, kind):
    monkeypatch.setenv('AGENTFEM_FEEDBACK_HOME',str(tmp_path/'feedback'))
    m,u,domain = model()
    try:
        if kind == 'option':
            m.step(target=u, unknown_option=True)
        elif kind == 'capability':
            m.materials.clear()
            m.step(target=u)
        elif kind == 'empty':
            empty = mesh.boundary_region(domain,lambda x:x[0]>2.,name='empty')
            results.region_average(u.value[0],on=empty)
        elif kind == 'duplicate':
            boundary = mesh.face(domain,axis='x',value=0.,tag=1)
            m.traction((1.,0.),on=boundary,name='same')
            m.traction((1.,0.),on=boundary,name='same')
            report=m.validate()
            assert any(i.code=='AFM-NAME-001' and i.severity=='warning' for i in report.issues)
            # Exercise report rendering without changing the owner's warning severity.
            raise ModelValidationError(report)
        elif kind == 'schedule':
            results.field_output('U',every=1,intervals=2)
        else:
            results.sample_points(u,[(2.,2.)])
    except Exception as exc:
        error=cli._rank_error(exc,rank=0)
    else:
        pytest.fail('expected a minimal owner failure')
    record={'status':'failed','stage':error['stage'],'error':error}
    path=tmp_path/'execution.json'; path.write_text(json.dumps(record))
    before=path.read_bytes()
    assert cli.main(['diagnose',str(path),'--json'])==0
    diagnosis=json.loads(capsys.readouterr().out)
    assert diagnosis['status']=='failed' and diagnosis['network_used'] is False
    assert diagnosis['failure']['code']==error['code']
    assert diagnosis['failure']['message']==error['message']
    assert diagnosis['failure']['rank']==0
    assert diagnosis['failure']['details']==json.loads(json.dumps(error.get('details')))
    assert diagnosis['failure']['validation']==json.loads(json.dumps(error.get('validation')))
    assert path.read_bytes()==before
    human=feedback.format_diagnosis(diagnosis)
    assert error['message'] in human
    if kind in {'empty','outside','schedule','duplicate'}:
        assert 'next:' in human
    event=feedback.build_event('run','failed',error=error)
    assert 'message' not in event['failure'] and 'validation' not in event['failure']


def test_model_diagnostic_does_not_promise_cli_syntax_check_executes_model():
    hints=feedback._support_suggestions({'stage':'model_preflight'})
    assert 'model.validate' in hints[0] and 'not the constructed model' in hints[0]


@pytest.mark.parametrize('details', ['legacy detail text', {'issues':None},
    {'issues':['legacy issue', {'message':'check this', 'suggestions':'dt'}],
     'accepted_options':'dt, steps'}])
def test_legacy_diagnostic_details_remain_readable(tmp_path, capsys, details):
    path = tmp_path/'execution.json'
    record = {'status':'failed', 'error':{'type':'ValueError','message':'original failure',
        'details':details, 'validation':'legacy validation text'}}
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    assert cli.main(['diagnose',str(path)]) == 0
    text = capsys.readouterr().out
    assert 'original failure' in text
    assert path.read_bytes() == before
    diagnosis = feedback.diagnose(path)
    assert diagnosis['failure']['details'] == details

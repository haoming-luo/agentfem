# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Public installed-project compositions and independent result observations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from mpi4py import MPI
from agentfem import cli, constitutive, fields, mesh, models, results, studies
from agentfem._task_discovery import task_record


@pytest.mark.parametrize('template', ['static-results','dynamic-results','nonlinear-results'])
def test_result_recipe_from_project(tmp_path, capsys, template):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip('installed serial project recipe; distributed output tested separately')
    project = tmp_path/'中文 项目'/template
    assert cli.main(['init', str(project), '--template', template, '--json']) == 0
    capsys.readouterr()
    assert cli.main(['run','--project',str(project),'--json']) == 0
    run = json.loads(capsys.readouterr().out)
    manifest = next(project.glob('outputs/**/result.json'))
    record = json.loads(manifest.read_text())
    assert record['status'] == 'completed', run
    assert cli.main(['show',str(manifest),'--json']) == 0
    menu = json.loads(capsys.readouterr().out)['result_menu']
    assert menu['recommended_visualization'] and not menu['missing_files']
    histories = {h['name']:h for h in record['history_records']}
    if template == 'static-results':
        quantities = {q['name']:q for q in record['quantity_records']}
        assert quantities['tip_U2']['value'] < 0
        np.testing.assert_allclose(quantities['fixed_reaction']['value'], [0., 40000., 0.], atol=1e-5)
        pv = pytest.importorskip('pyvista')
        source = record['artifacts']['fields_xdmf']
        frames = results.read_unified_xdmf_series(manifest.parent/source)
        original = frames[0].points.copy()
        warped = frames[0].warp_by_vector('U', factor=1.)
        np.testing.assert_allclose(warped.points-original, frames[0].point_data['U'], atol=1e-12)
        assert 'S' in frames[0].cell_data and pv is not None
    elif template == 'dynamic-results':
        assert histories['tip_U2']['sample_count'] == 7
        assert histories['tip_U2']['abscissa_unit'] == 's'
        assert menu['frame_count'] == 3
        assert len(menu['frame_times']) == 3
    else:
        curve = histories['loaded_end_reaction']
        assert curve['abscissa_name'] == 'mean_loaded_end_displacement'
        assert curve['unit'] == 'N' and curve['abscissa_unit'] == 'm'
        assert np.all(np.diff(curve['abscissa']) > 0)
        assert np.asarray(curve['values'])[-1] > 0
        assert menu['frame_count'] >= 4


@pytest.mark.parametrize('task', ['mesh-input','solver-options','modal'])
def test_other_task_snippets_execute(task):
    namespace = {}
    if task == 'modal':
        domain = mesh.cuboid((0.,0.,0.),(1.,.2,.2),(2,1,1),comm=MPI.COMM_SELF)
        model = models.create(mesh=domain, study=studies.modal_solid(dimension=3))
        u = model.field(fields.displacement(domain))
        model.material(constitutive.isotropic_elastic(young=210e9,poisson=.3,density=7800))
        model.clamp(u,on=mesh.face(domain,axis='x',value=0.,tag=1))
        namespace.update(model=model,u=u)
    exec(compile(task_record('task:'+task)['matches'][0]['snippet'],'<installed-task>','exec'),namespace)
    if task == 'modal':
        assert np.all(namespace['frequencies_hz'] > 0)


def test_fresh_installed_style_discover_repair_run_show(tmp_path):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip('fresh serial CLI process')
    root = tmp_path/'安装态 中文 项目'
    package = Path(cli.__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(package), 'AGENTFEM_TELEMETRY': 'off',
        'AGENTFEM_STATE_HOME': str(tmp_path/'state')}
    def command(*args, expected=0):
        process = subprocess.run([sys.executable,'-m','agentfem.cli',*args,'--json'],
            cwd=tmp_path, env=env, capture_output=True,text=True,timeout=60)
        assert process.returncode == expected, process.stdout+process.stderr
        return json.loads(process.stdout)
    task = command('capabilities','--task','端点位移随时间')['matches'][0]
    command('init',str(root),'--template',task['template'])
    case = root/'case.py'; original = case.read_text()
    broken = original.replace('dt=1.0e-4','dtt=1.0e-4')
    assert broken != original
    case.write_text(broken)
    failed = command('run','--project',str(root),expected=1)
    diagnosis = command('diagnose',failed['execution_record'])
    assert diagnosis['failure']['code'].startswith('AFM-STEP-OPTION')
    assert diagnosis['failure']['details']['issues'][0]['option']=='dtt'
    assert case.read_text()==broken
    case.write_text(original)
    command('run','--project',str(root))
    shown = command('show','latest','--project',str(root))
    assert shown['status']=='completed'
    assert shown['result_menu']['frame_count']==3
    assert any(h['name']=='tip_U2' and h['sample_count']==7 for h in shown['result_menu']['histories'])


def test_result_menu_keeps_missing_and_legacy_metadata_unknown(tmp_path):
    from copy import deepcopy
    from agentfem._result_menu import result_menu
    record = {'schema':'agentfem.simulation-result','status':'partial',
        'artifacts':{'fields_hdf5':'not-delivered.h5'},
        'history_records':[{'name':'old_curve','sample_count':3}]}
    before = deepcopy(record)
    menu = result_menu(record,tmp_path/'result.json')
    assert menu['frame_count'] is None and menu['geometry'] is None
    assert menu['histories'][0]['abscissa_range'] is None
    assert menu['missing_files']==['fields_hdf5'] and record==before
    result = results.SimulationResult('missing-history')
    result.verify(required_histories=('required_curve',))
    failed = result_menu(result.manifest(),tmp_path/'missing.json')['failed_requirements']
    assert any(item['name']=='required_outputs_present' for item in failed)

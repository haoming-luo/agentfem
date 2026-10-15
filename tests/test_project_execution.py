# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import json
import numpy as np
import pytest
from agentfem import project


def heat_step():
    from agentfem import mesh, models, studies, fields, constitutive
    domain = mesh.rectangle((0., 0.), (1., .2), (4, 2))
    model = models.create(study=studies.transient_heat_transfer(dimension=2), mesh=domain)
    temperature = model.field(fields.temperature(domain, value=300.))
    model.material(constitutive.thermoelastic(young=200e9, poisson=.3, density=1.,
        thermal_expansion=1e-5, conductivity=1., specific_heat=1., reference_temperature=300.))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.), name="hot", tag=1)
    model.prescribed_temperature(temperature, 350., on=left)
    return model.step(target=temperature, dt=.01, steps=4, progress=False)


def context(root):
    root.mkdir()
    return project.current_run(project_root=root, project_name=root.name)


def test_real_partial_restart_equals_continuous(tmp_path):
    step = heat_step()
    first = context(tmp_path / "partial")
    policy = project.RunPolicy(until_step=2, journal_every=1, progress_every=1)
    report = project.execute_step(step, context=first, policy=policy)
    assert report["status"] == "paused"
    assert not first.manifest_path.exists()
    restored = heat_step()
    resumed = project.execute_step(restored, context=context(tmp_path / "resumed"),
        policy=project.RunPolicy(restart=report["artifacts"]["checkpoint"], journal_every=1))
    full = heat_step()
    final_context = context(tmp_path / "full")
    completed = project.execute_step(full, context=final_context,
        policy=project.RunPolicy(journal_every=1))
    assert completed["complete"] and resumed["complete"]
    np.testing.assert_array_equal(full.current.x.array, restored.current.x.array)
    assert json.loads(final_context.manifest_path.read_text())["status"] == "completed"


def test_observation_failure_preserves_original_and_accepted_checkpoint(tmp_path):
    step = heat_step()
    run = context(tmp_path / "failed")
    def bad(_):
        raise RuntimeError("sentinel observer failure")
    with pytest.raises(RuntimeError, match="sentinel"):
        project.execute_step(step, context=run, observations={"bad": bad},
            policy=project.RunPolicy(observation_every=1, journal_every=1))
    report = json.loads((run.output_directory / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["accepted_steps"] == 1
    assert report["failure_checkpoint"]
    assert not run.manifest_path.exists()


@pytest.mark.parametrize("kwargs", [{"journal_every":0}, {"wall_limit":float("nan")},
    {"until_step":True}, {"prepare_only":True,"restart":"x"}])
def test_bad_policy(kwargs):
    with pytest.raises(ValueError, match="AFM-RUN-POLICY"):
        project.RunPolicy(**kwargs)


def test_response_export_and_generic_run(tmp_path):
    from agentfem import results
    from test_response_material_step import make_step
    step = make_step(material_storage="owned_arrays")
    run = context(tmp_path / "response")
    report = project.execute_step(step, context=run,
        policy=project.RunPolicy(journal_every=1),
        observations={"contacts": lambda s: s.accepted_contact_evidence()},
        export=lambda s, directory: results.export_accepted_response(s, directory,
            state_fields={"clock": ("clock", None)}, units={"time": "s"}))
    assert report["complete"]
    archive = np.load(report["artifacts"]["accepted_material"])
    np.testing.assert_array_equal(archive["state"], step.accepted_material_fields()["state"])
    from xml.etree import ElementTree as ET
    tree = ET.parse(report["artifacts"]["fields"])
    grids = tree.findall(".//Grid[@GridType='Uniform']")
    assert len(grids) == 1
    assert grids[0].find("Topology").attrib["TopologyType"] == "Hexahedron"
    assert "clock" in {a.attrib["Name"] for a in grids[0].findall("Attribute")}
    with pytest.raises(FileExistsError):
        results.export_accepted_response(step, run.output_directory)
    with pytest.raises(ValueError, match="COMPONENT"):
        results.export_accepted_response(step, tmp_path / "invalid", state_fields={"bad": ("clock", 0)})
    assert not (tmp_path / "invalid").exists()


def test_project_required_api_preflight(tmp_path):
    (tmp_path/'case.py').write_text("raise RuntimeError('must not execute')")
    path = tmp_path/'agentfem.toml'
    path.write_text('[project]\nname="api"\nrequires_api=["results.AcceptedResponseSeries", "constitutive.StrainHistoryRuntime"]\n')
    assert not project.ProjectConfig.load(path).check()
    path.write_text('[project]\nname="api"\nrequires_api=["results.DoesNotExist"]\n')
    assert 'AFM-PROJECT-API' in project.ProjectConfig.load(path).check()[0]
    path.write_text('[project]\nname="api"\nrequires_api=["_private.secret"]\n')
    with pytest.raises(ValueError, match='requires_api'):
        project.ProjectConfig.load(path)


def test_project_runtime_requirement(tmp_path):
    (tmp_path / "case.py").write_text("raise RuntimeError('not executed')")
    path = tmp_path / "agentfem.toml"
    for requirement, rejected in ((">=0.4.3.dev0", False), (">=99", True), ("garbage", True)):
        path.write_text('[project]\nname="test"\nrequires_agentfem="'+requirement+'"\n')
        errors = project.ProjectConfig.load(path).check()
        assert bool(errors) is rejected
        if errors:
            assert "AFM-PROJECT-VERSION" in errors[0]


def test_project_required_accelerators_are_opt_in_preflight(tmp_path, monkeypatch):
    from agentfem import platforms
    calls = []
    def probe():
        calls.append(True)
        return {'hex_arithmetic': {'available': True, 'status': 'available'},
                'q1_geometry': {'available': False, 'status': 'load_failed',
                    'error': {'message': 'GMP library unavailable'}}}
    monkeypatch.setattr(platforms, '_native_accelerators', probe)
    (tmp_path/'case.py').write_text("raise RuntimeError('must not run')")
    path = tmp_path/'agentfem.toml'
    path.write_text('[project]\nname="native"\n')
    assert not project.ProjectConfig.load(path).check()
    assert not calls
    path.write_text('[project]\nname="native"\nrequires_accelerators=["hex_arithmetic"]\n')
    assert not project.ProjectConfig.load(path).check()
    path.write_text('[project]\nname="native"\nrequires_accelerators=["q1_geometry", "typo"]\n')
    errors = project.ProjectConfig.load(path).check()
    assert len(errors) == 2 and all('AFM-PROJECT-ACCELERATOR' in e for e in errors)
    assert 'GMP library unavailable' in errors[0]
    assert 'unknown accelerator' in errors[1]
    path.write_text('[project]\nname="native"\nrequires_accelerators="hex_arithmetic"\n')
    with pytest.raises(ValueError, match='requires_accelerators'):
        project.ProjectConfig.load(path)


def test_partial_presentation_selects_declared_visualization(tmp_path):
    from agentfem import presentation
    record = {'schema': 'agentfem.step-execution', 'status': 'paused', 'accepted_steps': 600,
        'visualization': {'primary_artifact': 'process_fields'},
        'artifacts': {'process_fields': 'fields.xdmf'}}
    text = presentation.format_record(record, tmp_path/'report.json')
    assert 'accepted steps: 600' in text
    assert 'open in ParaView: fields.xdmf' in text
    assert 'not a completed scientific result' in text


def test_streaming_frames_and_completion_retention(tmp_path):
    from agentfem import results, checkpointing
    from test_response_material_step import make_step
    import h5py
    from xml.etree import ElementTree as ET
    run = context(tmp_path/'frames')
    step = make_step(steps=7, checkpoint=checkpointing.every(2,
        directory=run.output_directory/'checkpoints', keep_last=2))
    report = project.execute_step(step, context=run,
        policy=project.RunPolicy(field_intervals=3, completed_checkpoint_keep=1),
        frames=results.AcceptedResponseSeries(run.artifact('process.xdmf'),
            state_fields={'clock': ('clock', None)}, velocity=True, acceleration=True))
    assert report['complete']
    assert report['visualization']['primary'] == str(run.artifact('process.xdmf'))
    timing = report['timings_seconds']
    assert all(value >= 0 for value in timing.values())
    assert timing['advance_inclusive'] >= timing['accepted_callbacks']
    assert len(list(run.output_directory.rglob('*.checkpoint.json'))) == 1
    assert not (run.output_directory/'accepted-final.checkpoint.json').exists()
    final = report['artifacts']['checkpoint']
    restored = make_step(steps=7)
    restored.load_checkpoint(final)
    np.testing.assert_array_equal(restored.state.u.value.x.array, step.state.u.value.x.array)
    assert [s['increment'] for s in json.loads(run.artifact('process.json').read_text())['stations']] == [0,3,5,7]
    with h5py.File(run.artifact('process.h5')) as f:
        assert len(f['Frames']) == 4
        assert f['Frames/0000/Geometry'].id == f['Frames/0003/Geometry'].id
    grids = ET.parse(run.artifact('process.xdmf')).findall(".//Grid[@GridType='Uniform']")
    assert len(grids) == 4
    assert all(g.find('Topology').attrib['TopologyType']=='Hexahedron' for g in grids)
    plain = make_step(steps=7)
    plain.run()
    np.testing.assert_array_equal(plain.state.u.value.x.array, step.state.u.value.x.array)


def test_series_with_raw_endpoint_has_one_visualization(tmp_path):
    from agentfem import results
    from test_response_material_step import make_step
    run = context(tmp_path/'single-view')
    step = make_step(steps=3)
    report = project.execute_step(step, context=run,
        policy=project.RunPolicy(field_intervals=3),
        frames=results.AcceptedResponseSeries(run.artifact('fields.xdmf')),
        export=lambda s, d: results.export_accepted_response(s, d, visualization=False))
    assert len(list(run.output_directory.glob('*.xdmf'))) == 1
    assert len(list(run.output_directory.glob('*.h5'))) == 1
    from agentfem import presentation
    manifest = json.loads(run.manifest_path.read_text())
    assert manifest['metadata']['visualization']['primary_artifact'] == 'process_fields'
    assert 'open in ParaView: fields.xdmf' in presentation.format_result(manifest, run.manifest_path)
    with np.load(report['artifacts']['accepted_material']) as archive:
        np.testing.assert_array_equal(archive['state'], step.accepted_material_fields()['state'])


def test_cell_transport_reuses_mapping_not_field_storage():
    from agentfem.results.cell_data import _CellFieldTransport
    from test_response_material_step import make_step
    step = make_step()
    domain = step.state.u.value.function_space.mesh
    keys = np.asarray(domain.topology.original_cell_index)
    transport = _CellFieldTransport(domain, keys)
    values = np.arange(len(keys), dtype=float)
    first = transport.field(values, name='first')
    frozen = first.x.array.copy()
    second = transport.field(values+10, name='second')
    assert len(transport.spaces) == 1
    assert first.function_space is second.function_space
    np.testing.assert_array_equal(first.x.array, frozen)
    assert not np.shares_memory(first.x.array, second.x.array)
    assert transport.matches(domain, keys)
    assert not transport.matches(domain, keys+1)


@pytest.mark.parametrize('field', ['state', 'cauchy_stress', 'deformation_gradient'])
def test_raw_endpoint_rejects_invalid_arrays_before_creating_files(tmp_path, field):
    from agentfem import results
    from test_response_material_step import make_step
    from types import SimpleNamespace
    step = make_step()
    record = dict(step.accepted_material_fields())
    record[field] = np.full_like(record[field], np.nan)
    fake = SimpleNamespace(state=step.state, steps=step.steps,
        completed_steps=step.completed_steps, accepted_material_fields=lambda: record)
    output = tmp_path/'bad'
    with pytest.raises(ValueError, match='AFM-EXPORT-ARRAY'):
        results.export_accepted_response(fake, output, visualization=False)
    assert not output.exists()


def test_frames_partial_restart_keeps_sources(tmp_path):
    from agentfem import results, checkpointing
    from test_response_material_step import make_step
    run = context(tmp_path/'partial-frames')
    step = make_step(steps=7, checkpoint=checkpointing.every(2,
        directory=run.output_directory/'checkpoints', keep_last=2))
    report = project.execute_step(step, context=run,
        policy=project.RunPolicy(until_step=4, field_intervals=3, completed_checkpoint_keep=1),
        frames=results.AcceptedResponseSeries(run.artifact('process.xdmf')))
    assert not report['complete']
    assert len(list(run.output_directory.rglob('*.checkpoint.json'))) == 2
    assert [s['increment'] for s in json.loads(run.artifact('process.json').read_text())['stations']] == [0,3,4]
    source = report['artifacts']['checkpoint']
    continued = context(tmp_path/'continued')
    step = make_step(steps=7)
    project.execute_step(step, context=continued,
        policy=project.RunPolicy(restart=source, field_intervals=3, completed_checkpoint_keep=1),
        frames=results.AcceptedResponseSeries(continued.artifact('process.xdmf')))
    from pathlib import Path
    assert Path(source).exists()
    assert [s['increment'] for s in json.loads(continued.artifact('process.json').read_text())['stations']] == [4,5,7]


def test_failed_run_keeps_frames_and_checkpoints(tmp_path):
    from agentfem import results, checkpointing
    from test_response_material_step import make_step
    run = context(tmp_path/'failed-frames')
    step = make_step(checkpoint=checkpointing.every(1, directory=run.output_directory/'checkpoints', keep_last=2))
    def fail(s):
        if s.completed_steps == 3:
            raise RuntimeError('output sentinel')
        return {}
    with pytest.raises(RuntimeError, match='output sentinel'):
        project.execute_step(step, context=run, observations={'test': fail},
            policy=project.RunPolicy(field_intervals=4, observation_every=1, completed_checkpoint_keep=1),
            frames=results.AcceptedResponseSeries(run.artifact('process.xdmf')))
    metadata=json.loads(run.artifact('process.json').read_text())
    assert not metadata['complete']
    assert metadata['stations'][-1]['increment'] == 2
    assert len(list(run.output_directory.rglob('*.checkpoint.json'))) >= 2


@pytest.mark.parametrize('options', [{'field_intervals':0}, {'field_intervals':True},
    {'completed_checkpoint_keep':0}, {'completed_checkpoint_keep':1.5}])
def test_output_policy_validation(options):
    with pytest.raises(ValueError, match='AFM-RUN-POLICY'):
        project.RunPolicy(**options)


def test_final_integrity_failure_preserves_previous(tmp_path):
    from agentfem import checkpointing
    from agentfem._project_checkpoints import retain_completed
    from test_response_material_step import make_step
    step=make_step(checkpoint=checkpointing.every(2,directory=tmp_path,keep_last=2))
    step.run()
    paths=[r.path for r in step.checkpoints]
    manifest=json.loads(paths[-1].read_text())
    (paths[-1].parent/manifest['shards'][0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        retain_completed(step,tmp_path,paths[-1],1,step.state.u.value.function_space.mesh.comm)
    assert all(p.exists() for p in paths)


def test_export_failure_does_not_prune(tmp_path):
    from agentfem import checkpointing
    from test_response_material_step import make_step
    run = context(tmp_path/'failed-final-export')
    step = make_step(checkpoint=checkpointing.every(2,directory=run.output_directory/'checkpoints',keep_last=2))
    def fail(*args):
        raise OSError('disk failure')
    with pytest.raises(OSError, match='disk failure'):
        project.execute_step(step, context=run,
            policy=project.RunPolicy(completed_checkpoint_keep=1), export=fail)
    assert len(list((run.output_directory/'checkpoints').glob('*.checkpoint.json'))) == 2


def test_series_refuses_existing_output(tmp_path):
    from agentfem import results
    from test_response_material_step import make_step
    path = tmp_path/'process.xdmf'
    path.write_text('precious')
    series = results.AcceptedResponseSeries(path)
    with pytest.raises(FileExistsError):
        series.write(make_step())
    assert series.close() == {}
    assert path.read_text() == 'precious'

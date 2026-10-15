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


def test_project_runtime_requirement(tmp_path):
    (tmp_path / "case.py").write_text("raise RuntimeError('not executed')")
    path = tmp_path / "agentfem.toml"
    for requirement, rejected in ((">=0.4.3.dev0", False), (">=99", True), ("garbage", True)):
        path.write_text('[project]\nname="test"\nrequires_agentfem="'+requirement+'"\n')
        errors = project.ProjectConfig.load(path).check()
        assert bool(errors) is rejected
        if errors:
            assert "AFM-PROJECT-VERSION" in errors[0]

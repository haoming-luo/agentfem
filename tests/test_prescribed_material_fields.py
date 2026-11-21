# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Reference-input mapping, accepted observations and physical thickness."""
import json
import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI
from agentfem import constitutive as c, fields, mesh, models, results, steps, studies
from agentfem.constitutive.quadrature import QuadratureField
from test_history_material_temperature import ThermalHistory


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("degree", [1, 3])
def test_reference_mapping_matches_exact_values_and_fe(dimension, degree):
    domain = (mesh.rectangle((0., 0.), (1., 1.), (2, 2), comm=MPI.COMM_WORLD) if dimension == 2 else
        mesh.cuboid((0., 0., 0.), (1., 1., 1.), (2, 2, 2), comm=MPI.COMM_WORLD))
    q = QuadratureField.create(domain, name="Q", degree=degree)
    X = q.owned_physical_points.reshape(-1, dimension)
    source = fields.prescribed(lambda X, t: 300 + 5*X[:, 0] + 10*t,
        name="TEMP", unit="K", times=[0., 1., 2.], time_unit="s")
    frozen = source.bind(q)
    np.testing.assert_allclose(frozen.at(.3)[:len(X)], 303+5*X[:, 0])
    function = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
    function.interpolate(lambda x: 300+5*x[0])
    mapped = fields.prescribed(function, name="TEMP", unit="K", times=[0., 2.]).bind(q)
    np.testing.assert_allclose(mapped.at(1.)[:len(X)], 300+5*X[:, 0])
    function.x.array[:] = 1.
    np.testing.assert_allclose(mapped.at(1.)[:len(X)], 300+5*X[:, 0])
    with pytest.raises(ValueError):
        mapped.values.setflags(write=True)
    with pytest.raises(ValueError, match="TIME"):
        mapped.at(3.)
    vector = fields.prescribed(lambda X, t: X*(1+t), name="ORIENTATION", unit="1",
        times=[0., 1.], value_shape=(dimension,)).bind(q)
    np.testing.assert_allclose(vector.at(.5)[:len(X)], 1.5*X)
    json.dumps(frozen.summary(), allow_nan=False)


def test_bad_layout_fails_on_every_rank():
    domain = mesh.rectangle((0., 0.), (1., 1.), (2, 2), comm=MPI.COMM_WORLD)
    q = QuadratureField.create(domain, name="Q", degree=2)
    def bad(X, time):
        return np.zeros((len(X), 1)) if domain.comm.rank == 0 else np.zeros(len(X))
    with pytest.raises(ValueError, match="AFM-PRESET-MAPPING"):
        fields.prescribed(bad, name="TEMP", unit="K", times=[0., 1.]).bind(q)


def test_declared_field_factory_and_constructor_are_discoverable():
    from agentfem._api_discovery import callable_record
    factory = callable_record('fields.prescribed')
    constructor = callable_record('fields.PrescribedField')
    assert factory['callable_kind'] == 'function'
    assert constructor['callable_kind'] == 'constructor'
    assert constructor['case_executed'] is False
    assert {'source', 'name', 'unit', 'times'} <= {p['name'] for p in factory['parameters'] if p['required']}


@pytest.mark.parametrize("name", ["DPDF", "F33", "E_GL", "P33", "THICKNESS", "TEMP", "MATERIAL_DPDF"])
def test_prescribed_inputs_cannot_overwrite_physical_response_channels(name):
    source = fields.prescribed(1., name=name, unit="1", times=[0., 2.], time_unit="s")
    with pytest.raises(ValueError, match="reserved"):
        build(temperature=None, field_variables={name: source})


@pytest.mark.parametrize("name", ["F33", "E_GL", "THICKNESS", "MATERIAL_DPDF"])
def test_state_channels_cannot_shadow_accepted_observations(name):
    material = ThermalHistory()
    material.state_schema = c.MaterialStateSchema("test.observation.collision", (
        c.MaterialStateVariable("exposure", output_name=name),))
    with pytest.raises(ValueError, match="reserved"):
        build(material=material)


def build(*, thickness=1., temperature=350., increments=4, field_variables=None, material=None, cells=(2, 2), units=None):
    domain = mesh.rectangle((0., 0.), (1., 1.), cells, cell_type="quadrilateral", comm=MPI.COMM_SELF)
    model = models.create(mesh=domain, study=studies.static_solid(dimension=2,
        assumption="plane_stress", nonlinear=True), units=units)
    u = model.field(fields.displacement(domain))
    model.material(ThermalHistory() if material is None else material)
    for component, axis in enumerate(("x", "y")):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component+1), component=component)
    right = mesh.face(domain, axis="x", value=1., tag=3)
    model.fix(u, on=right, component=0, value=.05)
    step = model.step(target=u, duration=2., time_unit="s", temperature=temperature,
        reference_thickness=thickness, plane_stress=c.PlaneStressReduction(symmetry="reflection_symmetric"),
        incrementation=steps.fixed(increments), progress=False, field_variables=field_variables)
    return step, right


def test_thickness_scales_residual_reactions_and_energy_once():
    from test_plane_stress_array import ArrayNeoHookean
    reference, edge = build(thickness=.5, material=ArrayNeoHookean())
    doubled, edge2 = build(thickness=1., material=ArrayNeoHookean())
    reference.solve()
    doubled.solve()
    np.testing.assert_allclose(reference.solution.x.array, doubled.solution.x.array, atol=2e-10)
    np.testing.assert_allclose(results.reaction_resultant(doubled, on=edge2),
        2*results.reaction_resultant(reference, on=edge), rtol=1e-9, atol=1e-10)
    a, b = reference.accepted_fields, doubled.accepted_fields
    np.testing.assert_allclose(a["F"], b["F"], atol=2e-10)
    assert b.statistics("F33").total_weight == pytest.approx(2*a.statistics("F33").total_weight)
    np.testing.assert_allclose(b["THICKNESS"], 2*a["THICKNESS"])
    assert np.dot(b['SENER'], b.weights) == pytest.approx(2*np.dot(a['SENER'], a.weights))


@pytest.mark.parametrize("kind", ["traction", "surface_force"])
def test_thickness_weights_load_controlled_patch_with_planar_resultant_convention(kind):
    from test_plane_stress_array import ArrayNeoHookean

    gradients = []
    for thickness in (.3, .8):
        domain = mesh.rectangle((0., 0.), (1., 1.), (2, 2), cell_type="quadrilateral", comm=MPI.COMM_SELF)
        model = models.create(mesh=domain, study=studies.static_solid(dimension=2,
            assumption="plane_stress", nonlinear=True))
        u = model.field(fields.displacement(domain))
        model.material(ArrayNeoHookean())
        for component, axis in enumerate(("x", "y")):
            model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component+1), component=component)
        right = mesh.face(domain, axis="x", value=1., tag=3)
        # Existing planar surface_force is a per-reference-thickness resultant.
        getattr(model, kind)((5., 0.), on=right)
        step = model.step(target=u, duration=1., reference_thickness=thickness,
            plane_stress=c.PlaneStressReduction(symmetry="reflection_symmetric"),
            incrementation=steps.fixed(2), progress=False)
        step.solve()
        reaction = results.reaction_force_moment(step, about=(0., 0.))
        np.testing.assert_allclose(reaction.force, (-5*thickness, 0.), atol=1e-8)
        np.testing.assert_allclose(step.accepted_fields["P"][:, 0, 0], 5., atol=1e-8)
        gradients.append(step.accepted_fields["F"])
    np.testing.assert_allclose(*gradients, atol=1e-9)


def test_planar_reference_thickness_matches_3d_thin_uniform_patch():
    from test_plane_stress_array import ArrayNeoHookean
    planar, edge = build(thickness=.03, material=ArrayNeoHookean())
    domain = mesh.cuboid((0., 0., 0.), (1., 1., .03), (2, 2, 1),
        cell_type='hexahedron', comm=MPI.COMM_SELF)
    model = models.create(mesh=domain, study=studies.static_solid(dimension=3, nonlinear=True))
    u = model.field(fields.displacement(domain))
    model.material(ArrayNeoHookean())
    for component, axis in enumerate(('x', 'y', 'z')):
        model.fix(u, on=mesh.face(domain, axis=axis, value=0., tag=component+1), component=component)
    right = mesh.face(domain, axis='x', value=1., tag=4)
    model.fix(u, on=right, component=0, value=.05)
    solid = model.step(target=u, duration=2., time_unit='s', incrementation=steps.fixed(4), progress=False)
    planar.solve()
    solid.solve()
    np.testing.assert_allclose(results.reaction_resultant(planar, on=edge),
        results.reaction_resultant(solid, on=right)[:2], atol=1e-9)
    np.testing.assert_allclose(planar.accepted_fields['F'][0], solid.accepted_fields['F'][0], atol=1e-9)


def test_unit_thickness_default_and_explicit_thickness_retain_distinct_units():
    from agentfem import units
    implicit, _ = build(thickness=None, units=units.si())
    explicit, _ = build(thickness=1., units=units.si())
    a, b = implicit.solve_result(), explicit.solve_result()
    np.testing.assert_allclose(implicit.solution.x.array, explicit.solution.x.array)
    assert a.fields['RF'].unit == f'({units.si().force})/m'
    assert b.fields['RF'].unit == units.si().force
    assert implicit.accepted_fields.summary()['weight'] == 'reference_area_per_unit_thickness'
    assert explicit.accepted_fields.summary()['weight'] == 'reference_volume'
    assert 'THICKNESS' not in implicit.accepted_fields.fields


def test_spatial_temperature_accepted_history_restart_and_output(tmp_path):
    temperature = fields.prescribed(lambda X, t: 300+20*X[:, 0]+15*t,
        name="TEMP", unit="K", times=[0., 1., 2.], time_unit="s")
    whole, _ = build(temperature=temperature)
    with pytest.raises(RuntimeError, match="AFM-OBSERVATION-STATE"):
        _ = whole.accepted_fields
    saved = []
    whole.solve(on_accept=lambda step: saved.append(step.accepted_fields))
    snapshot = saved[-1]
    X = snapshot.coordinates
    np.testing.assert_allclose(snapshot["TEMP"], 330+20*X[:, 0])
    # The synthetic law integrates temperature exposure by the trapezoidal rule.
    np.testing.assert_allclose(snapshot["EXPOSURE"], 2*(315+20*X[:, 0]), atol=1e-8)
    first_values = saved[0]["TEMP"].copy()
    with pytest.raises(ValueError):
        saved[0]["TEMP"].setflags(write=True)
    np.testing.assert_array_equal(saved[0]["TEMP"], first_values)
    assert snapshot.statistics("TEMP").mean == pytest.approx(340.)
    assert snapshot.statistics("TEMP", within=lambda X: X[:, 0] < .5).mean < 340
    first, _ = build(temperature=temperature)
    first.solve(until=.5)
    checkpoint = first.save_checkpoint(tmp_path/"half")
    resumed, _ = build(temperature=temperature)
    resumed.load_checkpoint(checkpoint)
    resumed.solve_result(output=tmp_path/"fields.xdmf")
    np.testing.assert_allclose(resumed.accepted_fields["F"], snapshot["F"], atol=2e-10)
    np.testing.assert_allclose(resumed.accepted_fields["EXPOSURE"], snapshot["EXPOSURE"])
    changed, _ = build(temperature=fields.prescribed(lambda X, t: 301+20*X[:, 0]+15*t,
        name="TEMP", unit="K", times=[0., 1., 2.], time_unit="s"))
    with pytest.raises(ValueError, match="identity"):
        changed.load_checkpoint(checkpoint)
    assert (tmp_path/"fields.xdmf").is_file()
    # Independent XML/HDF5 read: verify association and physical spatial values.
    import xml.etree.ElementTree as ET
    import h5py
    frame = ET.parse(tmp_path/'fields.xdmf').findall(".//Grid[@GridType='Uniform']")[-1]
    def read(data):
        file, dataset = data.text.strip().split(':', 1)
        with h5py.File(tmp_path/file, 'r') as archive:
            return np.asarray(archive[dataset])
    topology = read(frame.find('Topology/DataItem'))
    coordinates = read(frame.find('Geometry/DataItem'))
    attribute = frame.find("Attribute[@Name='TEMP_CELL']")
    assert attribute.get('Center') == 'Cell'
    values = read(attribute.find('DataItem')).reshape(-1)
    centroids = coordinates[topology].mean(axis=1)
    np.testing.assert_allclose(values, 330+20*centroids[:, 0])


def test_general_prescribed_columns_have_named_units_and_atomic_restart(tmp_path):
    class EnvironmentMaterial(ThermalHistory):
        def update(self, point):
            assert point.field_variables.shape == (2,)
            response = super().update(point)
            # A synthetic environmental history, independent of temperature.
            from dataclasses import replace
            return replace(response, state_new=point.state_old + point.time_increment*point.field_variables)
    humidity = fields.prescribed(lambda X, t: .2+.1*X[:, 0]+.05*t,
        name="HUMIDITY", unit="1", times=[0., .7, 2.], time_unit="s")
    concentration = fields.prescribed(4., name="CONCENTRATION", unit="mol/m3", times=[0., 2.], time_unit="s")
    inputs = {"HUMIDITY": humidity, "CONCENTRATION": concentration}
    step, _ = build(field_variables=inputs, material=EnvironmentMaterial())
    step.solve(until=.5)
    snapshot = step.accepted_fields
    np.testing.assert_allclose(snapshot["HUMIDITY"], .25+.1*snapshot.coordinates[:, 0])
    assert any(record["time"] == pytest.approx(.7) for record in step.accepted_increments)
    path = step.save_checkpoint(tmp_path/"environment")
    reversed_columns, _ = build(field_variables=dict(reversed(list(inputs.items()))), material=EnvironmentMaterial())
    with pytest.raises(ValueError, match="identity"):
        reversed_columns.load_checkpoint(path)
    continued, _ = build(field_variables=inputs, material=EnvironmentMaterial())
    continued.load_checkpoint(path)
    continued.solve()
    step.solve()
    np.testing.assert_allclose(continued.accepted_fields["EXPOSURE"], step.accepted_fields["EXPOSURE"])
    np.testing.assert_allclose(continued.accepted_fields["DELTA_T"], 8.)


def test_failed_spatial_trial_restores_all_accepted_observations(monkeypatch):
    field = fields.prescribed(lambda X, t: .2+.1*t+0*X[:, 0], name='HUMIDITY',
        unit='1', times=[0., 2.], time_unit='s')
    temperature = fields.prescribed(lambda X, t: 300+10*X[:, 0]+15*t,
        name='TEMP', unit='K', times=[0., 2.], time_unit='s')
    step, _ = build(field_variables={'HUMIDITY': field}, temperature=temperature)
    step.solve(until=.5)
    before = step.accepted_fields
    def fail(point):
        raise RuntimeError('injected local history failure')
    monkeypatch.setattr(step.material, 'update', fail)
    with pytest.raises(RuntimeError, match='injected local history failure'):
        step.solve()
    after = step.accepted_fields
    assert after.time == before.time
    for key in before.fields:
        np.testing.assert_array_equal(after[key], before[key])


@pytest.mark.parametrize("rotation", [0., .7])
def test_gauge_affine_stretch_is_objective(rotation):
    domain = mesh.rectangle((0., 0.), (1., 1.), (2, 2), comm=MPI.COMM_WORLD)
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (2,))))
    R = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    F = R @ np.diag([1.12, .95])
    u.interpolate(lambda X: (F-np.eye(2)) @ X[:2] + np.array([[.2], [-.1]]))
    measurement = results.gauge(u, start=(0., .5), end=(1., .5))
    assert measurement.engineering_strain == pytest.approx(.12)
    assert measurement.axial_projected_strain == pytest.approx(1.12*np.cos(rotation)-1)
    json.dumps(measurement.summary(), allow_nan=False)


def test_model_card_reports_thickness_weighted_load_without_solving():
    domain = mesh.rectangle((0., 0.), (1., 1.), (2, 2), comm=MPI.COMM_SELF)
    model = models.create(mesh=domain, study=studies.static_solid(dimension=2,
        assumption="plane_stress", nonlinear=True))
    u = model.field(fields.displacement(domain))
    model.material(ThermalHistory())
    edge = mesh.face(domain, axis="x", value=1.)
    model.traction((2., 0.), on=edge)
    options = dict(duration=1., reference_thickness=.3,
        plane_stress=c.PlaneStressReduction(symmetry="reflection_symmetric"))
    card = model.review(target=u, step_options=options)
    assert card['solver_executed'] is False
    assert card['physics']['section']['reference_thickness'] == .3
    np.testing.assert_allclose(card['loads'][0]['resultant'], [.6, 0.], atol=1e-14)
    assert not models.step_capability(model, target=u, options={**options, "reference_thickness": -1.})['supported']

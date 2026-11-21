# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact fields, independent equilibrium and saved-state selection contracts."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI
from agentfem import constitutive, fields, mesh, models, results, studies


def scalar_field(domain, offset=0., shape=()):
    function = fem.Function(fem.functionspace(domain, ("Lagrange", 1, shape)), name="T")
    size = int(np.prod(shape)) if shape else 1
    function.interpolate(lambda x: np.tile(x[0] + 2*x[1] + offset, (size, 1)) if shape else x[0] + 2*x[1] + offset)
    return function


def test_polyline_distance_and_values():
    domain = mesh.rectangle((0., 0.), (2., 1.), (4, 4), comm=MPI.COMM_WORLD)
    function = scalar_field(domain)
    sampled = results.sample_path(function, vertices=((0., 0.), (2., 0.), (2., 1.)), count=7)
    np.testing.assert_allclose(sampled.distance, np.linspace(0., 3., 7))
    np.testing.assert_allclose(sampled.values, [0., .5, 1., 1.5, 2., 3., 4.], atol=1e-13)
    with pytest.raises(ValueError, match="distinct"):
        results.sample_path(function, vertices=((0., 0.), (0., 0.)))
    with pytest.raises(ValueError, match="AFM-PATH-INPUT"):
        results.sample_path(function, start=(0., 0.), end=(1., 0.),
                            count=1 if domain.comm.rank == 0 else 5)


@pytest.mark.parametrize("shape", [(), (3,), (3, 3)])
def test_section_scalar_vector_tensor(shape):
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (2, 2, 2), comm=MPI.COMM_WORLD)
    function = fem.Function(fem.functionspace(domain, ("Lagrange", 1, shape)))
    size = int(np.prod(shape)) if shape else 1
    function.interpolate(lambda x: np.tile(x[0] + 2*x[1] + 3*x[2], (size, 1)) if shape else x[0] + 2*x[1] + 3*x[2])
    sampled = results.sample_section(function, origin=(0., 0., .5), axis_u=(1., 0., 0.),
                                     axis_v=(0., 1., 0.), u=[0., .5, 1.], v=[0., 1.])
    expected = np.array([[1.5, 2., 2.5], [3.5, 4., 4.5]])
    np.testing.assert_allclose(sampled.values.reshape(2, 3, size), np.repeat(expected[..., None], size, axis=2), atol=1e-13)
    assert sampled.located.all() and sampled.finite.all()
    json.dumps(sampled.summary(), allow_nan=False)


def test_section_missing_and_bad_axes():
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (2, 2, 2), comm=MPI.COMM_WORLD)
    function = scalar_field(domain)
    options = dict(origin=(0., 0., .5), axis_u=(1., 0., 0.), axis_v=(0., 1., 0.), u=[0., 2.], v=[0., 1.])
    sample = results.sample_section(function, **options, missing="nan")
    assert sample.located.sum() == 2
    assert np.isnan(sample.values[:, 1]).all()
    with pytest.raises(ValueError, match="AFM-SECTION-INPUT"):
        results.sample_section(function, **{**options, "axis_v": (1., 0., 0.)})


@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
def test_compare_fields_across_meshes_and_coverage(shape):
    a = scalar_field(mesh.rectangle((0., 0.), (1., 1.), (2, 2), comm=MPI.COMM_WORLD), shape=shape)
    b = scalar_field(mesh.rectangle((0., 0.), (1., 1.), (4, 4), comm=MPI.COMM_WORLD), offset=.5, shape=shape)
    comparison = results.compare_fields(a, b, at=((.25, .25), (.75, .75)))
    np.testing.assert_allclose(comparison.difference, .5, atol=1e-13)
    assert comparison.maximum_absolute_error == pytest.approx(.5)
    assert comparison.rms_error == pytest.approx(.5)
    assert comparison.relative_l2_error == pytest.approx(np.sqrt(.5)/np.linalg.norm([.75, 2.25]))
    partial = results.compare_fields(a, b, at=((.5, .5), (2., 2.)), missing="nan")
    assert partial.valid.tolist() == [True, False]
    zero = scalar_field(a.function_space.mesh, shape=shape)
    zero.x.array[:] = 0.
    assert results.compare_fields(zero, zero, at=((.5, .5),)).relative_l2_error is None
    json.dumps(comparison.summary(), allow_nan=False)


@pytest.mark.parametrize("degree", [1, 2])
def test_reaction_force_moment_from_solved_cantilever(degree):
    domain = mesh.rectangle((0., 0.), (2., 1.), (4, 4), comm=MPI.COMM_WORLD)
    model = models.create(study=studies.static_solid(dimension=2, assumption="plane_stress"), mesh=domain)
    u = model.field(fields.displacement(domain, degree=degree))
    model.material(constitutive.isotropic_elastic(young=1000., poisson=.3, density=1.))
    left = mesh.face(domain, axis="x", value=0., name="left", tag=1)
    right = mesh.face(domain, axis="x", value=2., name="right", tag=2)
    model.clamp(u, on=left)
    model.traction((0., -3.), on=right)
    step = model.step(target=u)
    step.solve_result()
    before = u.value.x.array.copy()
    support = results.reaction_force_moment(step.problem, on=left, about=(0., 0.))
    np.testing.assert_allclose(support.force, [0., 3.], atol=1e-10)
    assert support.moment == pytest.approx(6., abs=1e-10)
    shifted = results.reaction_force_moment(step.problem, on=left, about=(1., 0.))
    assert shifted.moment == pytest.approx(3., abs=1e-10)
    np.testing.assert_allclose(support.force, results.reaction_resultant(step.problem, on=left))
    np.testing.assert_array_equal(u.value.x.array, before)


def test_reaction_3d_nodal_force_and_moment():
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (2, 2, 2), comm=MPI.COMM_WORLD)
    force = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    coordinates = force.function_space.tabulate_dof_coordinates()
    # Symmetric unit cube nodes: forces (0, 0, 1) give centroid cross total force.
    force.interpolate(lambda x: np.vstack((0*x[0], 0*x[0], 1+0*x[0])))
    problem = SimpleNamespace(reaction_field=lambda **kw: force)
    result = results.reaction_force_moment(problem, about=(0., 0., 0.))
    n = domain.comm.allreduce(force.function_space.dofmap.index_map.size_local, op=MPI.SUM)
    np.testing.assert_allclose(result.force, [0., 0., n])
    np.testing.assert_allclose(result.moment, [.5*n, -.5*n, 0.])
    assert len(coordinates) >= 1


def test_retained_snapshot_selection_and_ambiguity():
    domain = mesh.rectangle((0., 0.), (1., 1.), (2, 2), comm=MPI.COMM_WORLD)
    frames = tuple(SimpleNamespace(index=i, load_factor=f, solution=scalar_field(domain, offset=f))
                   for i, f in enumerate([0., .5, 1., .5]))
    step = SimpleNamespace(snapshots=frames)
    selected = results.accepted_snapshot(step, index=2)
    assert results.probe(selected.solution, at=(.5, .5)) == pytest.approx(2.5)
    assert results.accepted_snapshot(step, at=.99, tolerance=.02) is frames[2]
    with pytest.raises(ValueError, match="ambiguous"):
        results.accepted_snapshot(step, at=.5)
    with pytest.raises(ValueError, match="SELECTION"):
        results.accepted_snapshot(step, at=.7)
    with pytest.raises(ValueError, match="MISSING"):
        results.accepted_snapshot(SimpleNamespace(snapshots=()), index=0)
    times = SimpleNamespace(snapshots=tuple(SimpleNamespace(index=i, time=t) for i, t in enumerate([0., .2, .4])))
    assert results.accepted_snapshot(times, at=.2).index == 1


def test_selection_from_actual_nonlinear_accepted_path():
    from agentfem import steps
    domain = mesh.rectangle((0., 0.), (1., .2), (4, 2), comm=MPI.COMM_SELF)
    model = models.create(mesh=domain, study=studies.static_solid(
        dimension=2, assumption="plane_stress", nonlinear=True))
    u = model.field(fields.displacement(domain, degree=2))
    model.material(constitutive.saint_venant_kirchhoff(young=1000., poisson=0., plane_stress=True))
    model.clamp(u, on=mesh.face(domain, axis="x", value=0., name="left", tag=1))
    model.traction((10., 0.), on=mesh.face(domain, axis="x", value=1., name="right", tag=2))
    step = model.step(target=u, incrementation=steps.fixed(4), progress=False)
    step.solve()
    before = u.value.x.array.copy()
    frame = results.accepted_snapshot(step, at=.5)
    tip = results.probe(frame.solution, at=(1., .1))[0]
    assert 500*((1+tip)**3 - (1+tip)) == pytest.approx(5., rel=1e-7)
    np.testing.assert_array_equal(u.value.x.array, before)


def test_section_invalid_rank_fails_collectively():
    domain = mesh.cuboid((0., 0., 0.), (1., 1., 1.), (2, 2, 2), comm=MPI.COMM_WORLD)
    function = scalar_field(domain)
    with pytest.raises(ValueError, match="AFM-SECTION-INPUT"):
        results.sample_section(function, origin=(0., 0., .5), axis_u=(1., 0., 0.),
                               axis_v=(1., 0., 0.) if domain.comm.rank == 0 else (0., 1., 0.),
                               u=[0., 1.], v=[0., 1.])


@pytest.mark.parametrize("option", ["missing", "padding", "shape"])
def test_point_sampling_rejects_rank_inconsistent_contract_before_evaluation(option):
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires two MPI ranks")
    domain = mesh.rectangle((0., 0.), (1., 1.), (2, 2), comm=comm)
    shape = (2,) if option == "shape" and comm.rank == 0 else ()
    function = fem.Function(fem.functionspace(domain, ("Lagrange", 1, shape)))
    missing = "raise" if option == "missing" and comm.rank == 0 else "nan"
    padding = 1e-5 if option == "padding" and comm.rank == 0 else 1e-10
    with pytest.raises(ValueError, match="AFM-POINT-CONTRACT"):
        results.sample_points(function, ((2., 2.),), missing=missing, padding=padding)


def test_field_comparison_invalid_on_one_rank_fails_collectively():
    comm = MPI.COMM_WORLD
    domain = mesh.rectangle((0., 0.), (1., 1.), (2, 2), comm=comm)
    reference = scalar_field(domain)
    candidate = (fem.Function(fem.functionspace(domain, ("Lagrange", 1, (2,))))
                 if comm.rank == 0 else scalar_field(domain))
    with pytest.raises(ValueError, match="AFM-FIELD-SHAPE"):
        results.compare_fields(reference, candidate, at=((.5, .5),))


def test_point_evaluation_failure_is_delivered_to_all_ranks(monkeypatch):
    from dolfinx import mesh as dm

    comm = MPI.COMM_WORLD
    domain = mesh.rectangle((0., 0.), (1., 1.), (2, 2), comm=comm)
    function = scalar_field(domain)
    point = (dm.compute_midpoints(domain, 2, np.array([0], dtype=np.int32))[0]
             if comm.rank == 0 else None)
    point = comm.bcast(point, root=0)
    if comm.rank == 0:
        def fail(*args):
            raise RuntimeError("injected field evaluation failure")
        monkeypatch.setattr(function, "eval", fail)
    with pytest.raises(RuntimeError, match="injected field evaluation failure"):
        results.sample_points(function, (point,))

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import json
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from agentfem import constitutive, fields, loads, mesh, models, results, studies


def model():
    d = mesh.rectangle((0., 0.), (2., 1.), (4, 2), comm=MPI.COMM_WORLD)
    m = models.create(study=studies.static_solid(dimension=2, assumption='plane_stress'), mesh=d)
    u = m.field(fields.displacement(d, degree=1))
    m.material(constitutive.isotropic_elastic(young=1000., poisson=.3, density=1.))
    left = mesh.boundary(d, lambda x: np.isclose(x[0], 0), name='left', tag=1)
    right = mesh.boundary(d, lambda x: np.isclose(x[0], 2), name='right', tag=2)
    m.clamp(u, on=left)
    return d,m,u,right


def test_expression_tuple_matches_explicit_vector_and_resultant():
    d,m,u,right = model()
    x = ufl.SpatialCoordinate(d)
    a = loads.traction((1 + .8*ufl.cos(2*np.pi*x[1]), 0.), on=right)
    b = loads.traction(ufl.as_vector((1 + .8*ufl.cos(2*np.pi*x[1]), 0.)), on=right)
    np.testing.assert_allclose(results.boundary_resultant(a.value, on=right), results.boundary_resultant(b.value, on=right))
    m.load(a)
    before = u.value.x.array.copy()
    card = m.review(target=u)
    json.dumps(card, allow_nan=False)
    assert card['capability']['supported']
    assert card['solver_executed'] is False
    assert card['loads'][0]['reference_measure'] == pytest.approx(1.)
    assert card['constraints'][0]['owned_scalar_dof_count'] > 0
    assert card['rigid_modes']['remaining_modes'] == 0
    np.testing.assert_array_equal(before, u.value.x.array)


def test_load_errors_are_addressable():
    d,_,_,right = model()
    # Keep enough cells for every rank when this contract is checked under MPI.
    other = mesh.rectangle((0., 0.), (1., 1.), (2, 2), comm=MPI.COMM_WORLD)
    with pytest.raises(ValueError, match='AFM-LOAD-DOMAIN'):
        loads.traction((ufl.SpatialCoordinate(other)[0], 0), on=right)
    with pytest.raises(ValueError, match='AFM-LOAD-SHAPE'):
        loads.traction((1, 2, 3), on=right)
    with pytest.raises(ValueError, match='AFM-LOAD-FINITE'):
        loads.traction((float('nan'), 0), on=right)


def test_pressure_semantics_are_shared_with_discovery():
    from agentfem._api_discovery import callable_record
    d,_,u,right = model()
    for mode in ('reference', 'current'):
        pressure = loads.pressure(1, on=right, configuration=mode, displacement=u)
        assert pressure.summary()['physical_semantics'] == callable_record('loads.pressure')['physical_semantics'][mode]
    assert callable_record('models.Model.traction')['physical_semantics']['direction'] == 'supplied vector components'


def test_review_reports_unconstrained_model_and_empty_load():
    d,m,u,_ = model()
    m.constraints.clear()
    empty = mesh.boundary(d, lambda x: x[0] > 10, name='empty', tag=3)
    m.traction((1, 0), on=empty)
    card = m.review(target=u)
    assert card['rigid_modes']['remaining_modes'] == 3
    assert card['findings'][0]['code'] == 'AFM-REVIEW-EMPTY-LOAD'

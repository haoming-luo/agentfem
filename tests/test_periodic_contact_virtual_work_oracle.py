# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent oracle controls before trusting its mesh-discrepancy diagnosis."""

from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ORACLE = runpy.run_path(str(ROOT / "tools/audit_periodic_contact_virtual_work.py"))
MAKE = runpy.run_path(str(ROOT / "tools/diagnose_periodic_contact_order.py"))["make_step"]


def test_right_metric_hencky_stress_is_energy_derivative_and_objective():
    law = ORACLE["hencky_energy_stress"]
    F = np.array([[1.03, .02, -.01], [.01, .98, .03], [0., -.01, 1.02]])
    direction = np.random.default_rng(293).normal(size=(3, 3))
    energy, stress = law(F)
    for h in (1e-5, 1e-6):
        derivative = (law(F + h * direction)[0] - law(F - h * direction)[0]) / (2. * h)
        assert derivative == pytest.approx(np.sum(stress * direction), rel=2e-7)
    Q, _ = np.linalg.qr(np.random.default_rng(83).normal(size=(3, 3)))
    rotated_energy, rotated_stress = law(Q @ F)
    assert rotated_energy == pytest.approx(energy, abs=1e-12)
    np.testing.assert_allclose(rotated_stress, Q @ stress, atol=1e-11)
    with pytest.raises(ValueError, match="orientation-preserving"):
        law(np.diag([-1., 1., 1.]))


@pytest.mark.parametrize("degree", [1, 2])
def test_oracle_total_energy_virtual_work_and_periodic_fluctuations(degree):
    step = MAKE(degree=degree, increments=4, yield_stress=1e6, bulk_degree=6, contact_degree=11)
    oracle = ORACLE["ElasticCubeOracle"](step.solution)
    rng = np.random.default_rng(12)
    direction = oracle.lift(rng.normal(size=int(oracle.groups.max()) + 1))
    # Check the null space independently against runtime equations; reference
    # assembly itself never consumes this transformation.
    reduction = step.constraint.reduction(.25)
    assert reduction.reduced_size == int(oracle.groups.max()) + 1
    from scipy.sparse import csr_matrix
    T = csr_matrix((reduction.coefficients, reduction.column_indices, reduction.row_offsets),
                   shape=(reduction.full_size, reduction.reduced_size))
    reconstructed = T @ direction.ravel()[reduction.independent_full_dofs]
    np.testing.assert_allclose(reconstructed, direction.ravel(), atol=1e-14)
    u = oracle.coordinates @ np.diag([.02, -.009, -.009]) + 1e-4 * direction
    response = oracle.evaluate(u)
    work = np.sum(response["full_residual"] * direction)
    h = 1e-7
    a, b = oracle.evaluate(u + h * direction), oracle.evaluate(u - h * direction)
    derivative = sum(a[k] - b[k] for k in ("bulk_energy", "contact_energy")) / (2. * h)
    assert derivative == pytest.approx(work, rel=1e-7, abs=1e-7)
    assert oracle.area == pytest.approx(.25)


@pytest.mark.parametrize("family,cells", [("uniform", 4), ("edge_sine", 8),
                                         ("normal_cosine", 8), ("patch_edge_sine", 8)])
def test_independent_elastic_station_and_read_only_state(family, cells):
    step = MAKE(cells=cells, mesh_family=family, degree=2, increments=4,
                yield_stress=1e6, bulk_degree=6, contact_degree=11)
    step.solve(until=.25)
    state = step.response.state.committed_state_vectors().copy()
    displacement = step.solution.x.array.copy()
    report = ORACLE["audit_elastic_station"](step)
    assert report["passed"], report
    expected = -step.contact_provider.evaluate().contact_force_on_structure[0]
    assert report["contact_force"] == pytest.approx(expected, rel=1e-11)
    np.testing.assert_array_equal(step.response.state.committed_state_vectors(), state)
    np.testing.assert_array_equal(step.solution.x.array, displacement)
    # The audit must detect a solution which is no longer in equilibrium.
    oracle = ORACLE["ElasticCubeOracle"](step.solution)
    q = np.random.default_rng(65).normal(size=int(oracle.groups.max()) + 1)
    step.solution.x.array[:] += 1e-5 * oracle.lift(q).ravel()
    assert not ORACLE["audit_elastic_station"](step)["passed"]


def test_elastic_oracle_rejects_plastic_history_and_different_penalty():
    step = MAKE(degree=2, increments=4)
    step.solve(until=.25)
    with pytest.raises(ValueError, match="plastic-history"):
        ORACLE["audit_elastic_station"](step)
    elastic = MAKE(degree=2, increments=4, yield_stress=1e6, penalty=20000.)
    with pytest.raises(ValueError, match="fixture"):
        ORACLE["audit_elastic_station"](elastic)


def test_oracle_rejects_duplicate_or_outside_coordinates():
    with pytest.raises(ValueError, match="Duplicate"):
        ORACLE["periodic_groups"](np.zeros((2, 3)))
    with pytest.raises(ValueError, match="unit-cube"):
        ORACLE["periodic_groups"](np.array([[2., 0., 0.]]))

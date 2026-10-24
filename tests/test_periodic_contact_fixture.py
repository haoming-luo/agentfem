# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""A physical support change is explicit, not a contact-solver repair."""

from pathlib import Path
import runpy

import numpy as np
import pytest
from mpi4py import MPI


pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size != 1,
    reason="The bounded augmented-contact fixture is serial only.",
)


def _step(supported, **material):
    make = runpy.run_path(str(
        Path(__file__).resolve().parents[1]
        / "examples/finite_j2_periodic_augmented_plane.py"
    ))["make_step"]
    return make(cells=4, increments=8, penalty=40000.,
                affine_end_face_support=supported, **material)


@pytest.mark.parametrize("yield_stress", [2., 1e6])
def test_explicit_yield_stress_separates_plastic_and_elastic_diagnostic(yield_stress):
    step = _step(True, yield_stress=yield_stress)
    step.solve(until=.25)
    peeq = step.response.state.committed_state_vectors()[:, 9]
    if yield_stress == 2.:
        assert np.max(peeq) > 0.
    else:
        assert np.max(peeq) == 0.
    assert np.linalg.norm(step.contact_provider.evaluate().contact_force_on_structure) > .1


@pytest.mark.parametrize("yield_stress", [0., -1., np.nan, np.inf])
def test_invalid_diagnostic_yield_stress_rejects_before_model_construction(yield_stress):
    with pytest.raises(ValueError, match="physical yield stress"):
        _step(True, yield_stress=yield_stress)


@pytest.mark.parametrize("supported", [False, True])
def test_periodic_contact_fixture_identifies_where_the_net_force_is_carried(supported):
    step = _step(supported)
    step.solve()
    coordinates = step.solution.function_space.tabulate_dof_coordinates()
    corners = np.all(np.isclose(coordinates, 0.) | np.isclose(coordinates, 1.), axis=1)
    residual = step.assemble_full_residual()
    try:
        values = residual.array_r.reshape(-1, 3)
        resultant = values.sum(axis=0)
        corner_resultant = values[corners].sum(axis=0)
        tool_force = step.contact_provider.evaluate().contact_force_on_structure
        np.testing.assert_allclose(resultant, -tool_force, atol=1e-9)
        assert abs(resultant[0]) > .1  # An actual loaded contact path.
        if not supported:
            # Periodic classes outside the pinned corner class balance.
            np.testing.assert_allclose(corner_resultant, resultant, atol=1e-8)
        else:
            assert abs(resultant[0] - corner_resultant[0]) > .1 * abs(resultant[0])
            ends = np.isclose(coordinates[:, 2], 0.) | np.isclose(coordinates[:, 2], 1.)
            macro = step.constraint.deformation_gradient_at(1.) - np.eye(3)
            expected = coordinates[ends] @ macro[0]
            np.testing.assert_allclose(step.solution.x.array.reshape(-1, 3)[ends, 0],
                                       expected, atol=1e-12)
        assert step.constraint.mismatch(1.) < 1e-12
    finally:
        residual.destroy()


@pytest.mark.parametrize("change", ["support", "yield_stress"])
def test_changing_the_physical_fixture_cannot_silently_reuse_a_checkpoint(tmp_path, change):
    partial = _step(True)
    partial.solve(until=.5)
    checkpoint = partial.save_checkpoint(tmp_path / "supported")
    wrong_fixture = _step(False) if change == "support" else _step(True, yield_stress=1e6)
    before = wrong_fixture.solution.x.array.copy()
    with pytest.raises(ValueError, match="identity"):
        wrong_fixture.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(wrong_fixture.solution.x.array, before)
    resumed = _step(True)
    resumed.load_checkpoint(checkpoint)
    resumed.solve()
    continuous = _step(True)
    continuous.solve()
    np.testing.assert_allclose(resumed.solution.x.array, continuous.solution.x.array, atol=1e-12)
    np.testing.assert_allclose(resumed.response.state.committed_state_vectors(),
                               continuous.response.state.committed_state_vectors(), atol=1e-11)

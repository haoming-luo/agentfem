"""Automatic affine J2 restart at an accepted adaptive-path boundary."""

from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import constitutive, fields, models, solvers, steps, studies

from periodic_cube_fixture import periodic_unit_cube


def _automatic_periodic_j2_step():
    fixture = periodic_unit_cube(MPI.COMM_SELF)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="finite_strain_j2_periodic_automatic_restart",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2_000.0,
        )
    )
    periodicity = model.constraint(fixture.constraint(displacement))
    return model.step(
        target=displacement,
        material=material,
        constraints=periodicity,
        incrementation=steps.automatic(
            initial=0.25,
            minimum=1.0e-4,
            maximum=0.5,
            max_increments=100,
            max_cutbacks=12,
            cutback_factor=0.5,
            growth_factor=1.5,
            maximum_inelastic_increment=0.0045,
        ),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
    )


def _assert_same_increment_path(actual, expected):
    assert [
        (
            item.increment,
            item.attempt,
            item.converged,
            item.iterations,
            item.message,
        )
        for item in actual
    ] == [
        (
            item.increment,
            item.attempt,
            item.converged,
            item.iterations,
            item.message,
        )
        for item in expected
    ]
    np.testing.assert_allclose(
        [
            (
                item.start_load_factor,
                item.load_factor,
                item.checks.get("maximum_plastic_increment", np.nan),
            )
            for item in actual
        ],
        [
            (
                item.start_load_factor,
                item.load_factor,
                item.checks.get("maximum_plastic_increment", np.nan),
            )
            for item in expected
        ],
        rtol=2.0e-10,
        atol=2.0e-12,
    )


def test_affine_j2_automatic_checkpoint_restores_next_increment_and_cutback(
    tmp_path,
):
    reference = _automatic_periodic_j2_step()
    reference.solve()

    first_boundary = reference.accepted_increments[0].load_factor
    assert first_boundary == pytest.approx(0.25)
    rejected = [
        item for item in reference.attempted_increments if not item.converged
    ]
    assert rejected
    assert any(
        "maximum equivalent plastic-strain increment" in item.message
        for item in rejected
    )

    partial = _automatic_periodic_j2_step()
    partial.solve(until=first_boundary)
    assert partial.accepted_load_factor == pytest.approx(first_boundary)
    assert partial.last_solve_info.converged
    assert not partial.last_solve_info.completed_step
    expected_next_increment = partial.next_increment_size
    assert expected_next_increment is not None
    checkpoint = partial.save_checkpoint(
        tmp_path / "finite_strain_j2_periodic_automatic_restart"
    )

    restarted = _automatic_periodic_j2_step()
    restarted.load_checkpoint(checkpoint)
    assert restarted.accepted_load_factor == pytest.approx(first_boundary)
    assert restarted.next_increment_size == pytest.approx(expected_next_increment)
    _assert_same_increment_path(
        restarted.accepted_increments,
        partial.accepted_increments,
    )
    restarted.solve()

    np.testing.assert_allclose(
        restarted.solution.x.array,
        reference.solution.x.array,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        restarted.response.state.committed_state_vectors(),
        reference.response.state.committed_state_vectors(),
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        restarted.response.first_piola_stress.values,
        reference.response.first_piola_stress.values,
        rtol=2.0e-8,
        atol=2.0e-8,
    )
    _assert_same_increment_path(
        restarted.accepted_increments,
        reference.accepted_increments,
    )
    _assert_same_increment_path(
        restarted.attempted_increments,
        reference.attempted_increments,
    )
    assert restarted.last_solve_info.completed_step
    assert restarted.state_transaction.accepted_factor == pytest.approx(1.0)

"""Equation normalization only: no multiaxial experimental validation."""

import numpy as np
import pytest

from tools.p92_published_recovery_reference import published_parameters, rates
from tools.p92_tensor_normalization import (
    deviator,
    equivalent,
    reduced_uniaxial_rates,
    tensor_rates,
    uniaxial_tensors,
)


@pytest.mark.parametrize("temperature", [600, 650])
def test_packed_jacobian_in_noncoaxial_state(temperature):
    from tools.p92_tensor_normalization import (
        pack_uniaxial,
        packed_rates_and_jacobian,
        unpack_state,
        tensor,
        mandel,
        _DEVIATORIC_BASIS,
    )

    p = published_parameters(temperature)
    old = pack_uniaxial([350.0, 25.0, 50.0, -10.0, 0.2])
    old[3:6] = [20.0, -30.0, 15.0]
    old[8:11] = [5.0, 3.0, -2.0]
    old[13:16] = [-2.0, 4.0, 1.0]
    loading = np.array([0.001, -0.0003, 0.0002, 0.0001, -0.0002, 0.0003])
    actual, jac = packed_rates_and_jacobian(old, loading, p, poisson=0.3)
    stress, x1, x2, R, accumulated = unpack_state(old)
    reference = tensor_rates(
        stress, x1, x2, R, accumulated, tensor(loading), p, poisson=0.3
    )
    independent = np.concatenate(
        (
            mandel(reference["stress"]),
            _DEVIATORIC_BASIS.T @ mandel(reference["backstress1"]),
            _DEVIATORIC_BASIS.T @ mandel(reference["backstress2"]),
            [reference["isotropic"], reference["accumulated"]],
        )
    )
    np.testing.assert_allclose(actual, independent, rtol=1.0e-12, atol=1.0e-10)
    columns = []
    for i in range(18):
        perturbation = np.zeros(18)
        perturbation[i] = 1.0e-4 if i == 17 else 1.0e-5
        plus = packed_rates_and_jacobian(old + perturbation, loading, p, poisson=0.3)[0]
        minus = packed_rates_and_jacobian(old - perturbation, loading, p, poisson=0.3)[
            0
        ]
        columns.append((plus - minus) / (2 * perturbation[i]))
    np.testing.assert_allclose(jac, np.column_stack(columns), rtol=2.0e-5, atol=2.0e-6)


@pytest.mark.parametrize("temperature", [600, 650])
def test_full_tensor_algorithmic_tangent_after_noncoaxial_history(temperature):
    from tools.p92_tensor_normalization import (
        implicit_tensor_update,
        pack_uniaxial,
        tensor,
    )

    p = published_parameters(temperature)
    old = pack_uniaxial([300.0, 25.0, 50.0, -10.0, 0.2])
    old[3:6] = [20.0, -30.0, 15.0]
    old[8:11] = [5.0, 3.0, -2.0]
    before = old.copy()
    increment = np.array([0.001, -0.0002, 0.0001, 0.0003, -0.0002, 0.0004])
    state, tangent = implicit_tensor_update(old, tensor(increment), 0.2, p, poisson=0.3)
    columns = []
    for i in range(6):
        perturbation = np.zeros(6)
        perturbation[i] = 1.0e-8
        plus = implicit_tensor_update(
            old, tensor(increment + perturbation), 0.2, p, poisson=0.3
        )[0]
        minus = implicit_tensor_update(
            old, tensor(increment - perturbation), 0.2, p, poisson=0.3
        )[0]
        columns.append((plus[:6] - minus[:6]) / (2.0e-8))
    np.testing.assert_allclose(
        tangent, np.column_stack(columns), rtol=2.0e-5, atol=0.01
    )
    np.testing.assert_array_equal(old, before)
    assert state[17] >= old[17]


@pytest.mark.parametrize("temperature", [600, 650])
@pytest.mark.parametrize("increment", [-0.001, 0.0, 0.001])
def test_lateral_condensation_recovers_scalar_implicit_update(temperature, increment):
    from scipy.optimize import root
    from tools.p92_published_recovery_reference import (
        implicit_update,
        integrate_segment,
    )
    from tools.p92_tensor_normalization import (
        implicit_tensor_update,
        pack_uniaxial,
        unpack_state,
    )

    p = published_parameters(temperature)
    old_scalar = integrate_segment(np.zeros(5), 4.0, 0.001, p).y[:, -1]
    old = pack_uniaxial(old_scalar)
    dt = 0.2

    def evaluate(lateral):
        return implicit_tensor_update(
            old, np.diag([increment, *lateral]), dt, p, poisson=0.3
        )

    solved = root(
        lambda v: evaluate(v)[0][1:3],
        [-0.3 * increment] * 2,
        jac=lambda v: evaluate(v)[1][1:3, 1:3],
        tol=1.0e-10,
    )
    assert solved.success or np.linalg.norm(solved.fun) < 1.0e-9
    state, tangent = evaluate(solved.x)
    stress, x1, x2, R, accumulated = unpack_state(state)
    reduced = np.array([stress[0, 0], 1.5 * x1[0, 0], 1.5 * x2[0, 0], R, accumulated])
    expected, expected_tangent = implicit_update(old_scalar, increment, dt, p)
    np.testing.assert_allclose(reduced, expected, rtol=1.0e-9, atol=1.0e-8)
    condensed = tangent[0, 0] - tangent[0, 1:3] @ np.linalg.solve(
        tangent[1:3, 1:3], tangent[1:3, 0]
    )
    assert condensed == pytest.approx(expected_tangent, rel=1.0e-8)


def test_integrated_tensor_response_and_tangent_rotate_covariantly():
    from tools.p92_tensor_normalization import (
        implicit_tensor_update,
        pack_uniaxial,
        unpack_state,
        mandel,
        tensor,
        _DEVIATORIC_BASIS,
    )

    p = published_parameters(600)
    old = pack_uniaxial([300.0, 25.0, 50.0, -10.0, 0.2])
    old[3:6] = [20.0, -30.0, 15.0]
    old[8:11] = [5.0, 3.0, -2.0]
    stress, x1, x2, R, accumulated = unpack_state(old)
    q, _ = np.linalg.qr(np.array([[1.0, 2.0, 3.0], [-2.0, 3.0, 1.0], [3.0, 1.0, -2.0]]))
    rotated_old = np.concatenate(
        (
            mandel(q @ stress @ q.T),
            _DEVIATORIC_BASIS.T @ mandel(q @ x1 @ q.T),
            _DEVIATORIC_BASIS.T @ mandel(q @ x2 @ q.T),
            [R, accumulated],
        )
    )
    increment = tensor([0.001, -0.0002, 0.0001, 0.0003, -0.0002, 0.0004])
    state, tangent = implicit_tensor_update(old, increment, 0.2, p, poisson=0.3)
    rotated, rotated_tangent = implicit_tensor_update(
        rotated_old, q @ increment @ q.T, 0.2, p, poisson=0.3
    )
    for base, changed in zip(unpack_state(state), unpack_state(rotated)):
        expected = q @ base @ q.T if np.ndim(base) == 2 else base
        np.testing.assert_allclose(changed, expected, rtol=1.0e-9, atol=1.0e-8)
    perturbation = tensor([0.1, 0.2, -0.3, 0.4, -0.2, 0.05])
    expected = q @ tensor(tangent @ mandel(perturbation)) @ q.T
    actual = tensor(rotated_tangent @ mandel(q @ perturbation @ q.T))
    np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-6)


def test_hydrostatic_loading_has_no_inelastic_flow():
    from tools.p92_tensor_normalization import (
        implicit_tensor_update,
        pack_uniaxial,
        tensor,
    )

    p = published_parameters(600)
    old = pack_uniaxial(np.zeros(5))
    state, _ = implicit_tensor_update(old, 0.001 * np.eye(3), 0.2, p, poisson=0.3)
    np.testing.assert_allclose(
        tensor(state[:6]), p.E / 0.4 * 0.001 * np.eye(3), atol=1.0e-10
    )
    np.testing.assert_allclose(state[6:16], 0.0, atol=1.0e-13)
    assert state[17] == pytest.approx(0.0, abs=1.0e-13)
    assert state[16] < 0


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, True])
def test_invalid_increment_preserves_old_state(dt):
    from tools.p92_tensor_normalization import implicit_tensor_update, pack_uniaxial

    old = pack_uniaxial([300.0, 25.0, 50.0, -10.0, 0.2])
    before = old.copy()
    with pytest.raises(ValueError, match="physical dt"):
        implicit_tensor_update(
            old, 0.001 * np.eye(3), dt, published_parameters(600), poisson=0.3
        )
    np.testing.assert_array_equal(old, before)


@pytest.mark.parametrize("scale", [1.0e-3, 1.0e6])
def test_tensor_update_is_covariant_under_stress_unit_conversion(scale):
    from dataclasses import replace
    from tools.p92_tensor_normalization import (
        implicit_tensor_update,
        pack_uniaxial,
        tensor,
    )

    p = published_parameters(600)
    converted = replace(
        p,
        E=p.E * scale,
        k=p.k * scale,
        Q=p.Q * scale,
        H=p.H * scale,
        Qr=p.Qr * scale,
        a1=p.a1 * scale,
        a2=p.a2 * scale,
        Z=p.Z * scale,
        gamma1=p.gamma1 * scale ** (1 - p.m),
        gamma2=p.gamma2 * scale ** (1 - p.m),
        gamma_r1=p.gamma_r1 * scale ** (1 - p.mr),
        gamma_r2=p.gamma_r2 * scale ** (1 - p.mr),
    )
    old = pack_uniaxial([300.0, 25.0, 50.0, -10.0, 0.2])
    old[3:6] = [20.0, -30.0, 15.0]
    changed = old.copy()
    changed[:17] *= scale
    increment = tensor([0.001, -0.0002, 0.0001, 0.0003, -0.0002, 0.0004])
    state, tangent = implicit_tensor_update(old, increment, 0.2, p, poisson=0.3)
    actual, actual_tangent = implicit_tensor_update(
        changed, increment, 0.2, converted, poisson=0.3
    )
    actual[:17] /= scale
    np.testing.assert_allclose(actual, state, rtol=1.0e-8, atol=1.0e-7)
    np.testing.assert_allclose(
        actual_tangent / scale, tangent, rtol=1.0e-8, atol=1.0e-5
    )


def test_noncoaxial_backward_euler_refines_toward_continuous_reference():
    from scipy.integrate import solve_ivp
    from tools.p92_tensor_normalization import (
        implicit_tensor_update,
        pack_uniaxial,
        packed_rates_and_jacobian,
        tensor,
    )

    p = published_parameters(600)
    old = pack_uniaxial([300.0, 25.0, 50.0, -10.0, 0.2])
    old[3:6] = [20.0, -30.0, 15.0]
    old[8:11] = [5.0, 3.0, -2.0]
    duration = 2.0
    increment = np.array([0.001, -0.0002, 0.0001, 0.0003, -0.0002, 0.0004])
    reference = solve_ivp(
        lambda t, y: packed_rates_and_jacobian(y, increment / duration, p, poisson=0.3)[
            0
        ],
        (0.0, duration),
        old,
        method="Radau",
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert reference.success
    errors = []
    for count in (4, 8, 16, 32):
        state = old.copy()
        for _ in range(count):
            state, _ = implicit_tensor_update(
                state, tensor(increment / count), duration / count, p, poisson=0.3
            )
        errors.append(np.max(np.abs(state[:17] - reference.y[:17, -1])))
    assert all(fine < 0.65 * coarse for coarse, fine in zip(errors, errors[1:]))


@pytest.mark.parametrize("temperature", [600, 650])
@pytest.mark.parametrize("poisson", [0.0, 0.3, 0.49])
@pytest.mark.parametrize(
    "state,rate",
    [
        ([350.0, 25.0, 50.0, -10.0, 0.2], 0.001),
        ([-350.0, -25.0, -50.0, -10.0, 0.2], -0.001),
        ([20.0, 10.0, 5.0, -10.0, 0.2], 0.0),
        ([0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
    ],
)
def test_exact_reduction_to_published_scalar_rates(temperature, poisson, state, rate):
    p = published_parameters(temperature)
    actual, lateral = reduced_uniaxial_rates(state, rate, p, poisson=poisson)
    np.testing.assert_allclose(
        actual, rates(state, rate, p), rtol=1.0e-12, atol=1.0e-10
    )
    np.testing.assert_allclose(lateral, 0.0, atol=1.0e-10)


@pytest.mark.parametrize("temperature", [600, 650])
def test_tensor_rotation_covariance_and_hydrostatic_independence(temperature):
    p = published_parameters(temperature)
    stress = np.array([[300.0, 60.0, 20.0], [60.0, -50.0, 10.0], [20.0, 10.0, 80.0]])
    x1 = deviator(np.array([[25.0, 3.0, 0.0], [3.0, -10.0, 4.0], [0.0, 4.0, -2.0]]))
    x2 = deviator(np.array([[50.0, -2.0, 4.0], [-2.0, 20.0, 0.0], [4.0, 0.0, 10.0]]))
    strain_rate = np.array(
        [[0.001, -0.0002, 0.0001], [-0.0002, 0.0, 0.0003], [0.0001, 0.0003, 0.0]]
    )
    copies = [item.copy() for item in (stress, x1, x2, strain_rate)]
    original = tensor_rates(stress, x1, x2, -10.0, 0.2, strain_rate, p, poisson=0.3)
    q, _ = np.linalg.qr(np.array([[1.0, 2.0, 3.0], [-2.0, 3.0, 1.0], [3.0, 1.0, -2.0]]))
    rotated = tensor_rates(
        q @ stress @ q.T,
        q @ x1 @ q.T,
        q @ x2 @ q.T,
        -10.0,
        0.2,
        q @ strain_rate @ q.T,
        p,
        poisson=0.3,
    )
    hydrostatic = tensor_rates(
        stress + 100 * np.eye(3), x1, x2, -10.0, 0.2, strain_rate, p, poisson=0.3
    )
    for name, value in original.items():
        expected = q @ value @ q.T if np.ndim(value) == 2 else value
        np.testing.assert_allclose(rotated[name], expected, rtol=1.0e-12, atol=1.0e-10)
        np.testing.assert_allclose(hydrostatic[name], value, rtol=1.0e-12, atol=1.0e-10)
    for old, actual in zip(copies, (stress, x1, x2, strain_rate)):
        np.testing.assert_array_equal(actual, old)
    for name in ("inelastic", "backstress1", "backstress2"):
        assert abs(np.trace(original[name])) < 1.0e-10
    effective = deviator(stress) - x1 - x2
    assert np.sum(effective * original["inelastic"]) == pytest.approx(
        equivalent(effective) * original["accumulated"]
    )


def test_scalar_backstress_norm_is_not_raw_frobenius_norm():
    _, x1, _ = uniaxial_tensors([300.0, 27.0, 50.0, -10.0, 0.2])
    assert equivalent(x1) == pytest.approx(27.0)
    assert np.linalg.norm(x1) == pytest.approx(np.sqrt(2 / 3) * 27.0)


@pytest.mark.parametrize("poisson", [-1.0, 0.5, np.nan, True])
def test_invalid_elastic_extension_rejected(poisson):
    with pytest.raises(ValueError, match="Poisson"):
        tensor_rates(
            np.eye(3),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            0.0,
            0.0,
            np.zeros((3, 3)),
            published_parameters(600),
            poisson=poisson,
        )


def test_local_evaluation_reuses_only_identical_owned_candidates(monkeypatch):
    from tools import p92_tensor_normalization as reference

    original = reference.packed_rates_and_jacobian
    evaluated = []

    def counted(candidate, *args, **kwargs):
        evaluated.append(candidate.copy())
        return original(candidate, *args, **kwargs)

    def probe_root(residual, guess, *, jac, **kwargs):
        candidate = guess.copy()
        first = residual(candidate)
        np.testing.assert_array_equal(residual(candidate.copy()), first)
        jac(candidate)
        assert len(evaluated) == 1
        # A reused MINPACK buffer changes value: object identity is not a key.
        candidate[0] += 1.
        residual(candidate)
        jac(candidate)
        assert len(evaluated) == 2
        np.testing.assert_array_equal(evaluated[0], guess)
        candidate[0] -= 1.
        residual(candidate)
        assert len(evaluated) == 3
        raise RuntimeError("probe complete")

    monkeypatch.setattr(reference, "packed_rates_and_jacobian", counted)
    monkeypatch.setattr(reference, "root", probe_root)
    for _ in range(2):
        evaluated.clear()
        with pytest.raises(RuntimeError, match="probe complete"):
            reference.implicit_tensor_update(np.zeros(18), np.zeros((3, 3)), .1,
                                             published_parameters(650), poisson=.3)
        assert len(evaluated) == 3  # No cache crosses a new local update.

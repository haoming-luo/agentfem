# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Candidate J2 tensor normalization of a published uniaxial recovery law.

An equation-level research discriminator, not a material provider or solver.
The paper does not identify Poisson ratio or validate multiaxial response.
"""

import numpy as np
from scipy.optimize import root


_MANDEL = np.zeros((6, 3, 3))
for _index in range(3):
    _MANDEL[_index, _index, _index] = 1.0
for _index, (_i, _j) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
    _MANDEL[_index, _i, _j] = _MANDEL[_index, _j, _i] = 1 / np.sqrt(2)
_TRACE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_DEV = np.eye(6) - np.outer(_TRACE, _TRACE) / 3
_DEVIATORIC_BASIS = np.zeros((6, 5))
_DEVIATORIC_BASIS[:3, 0] = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
_DEVIATORIC_BASIS[:3, 1] = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
_DEVIATORIC_BASIS[3:, 2:] = np.eye(3)


def mandel(tensor):
    return np.einsum("aij,ij->a", _MANDEL, _symmetric(tensor, "tensor"))


def tensor(vector):
    if np.iscomplexobj(vector):
        raise ValueError("Require real Mandel components.")
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError("Require six finite Mandel components.")
    return np.einsum("a,aij->ij", vector, _MANDEL)


def pack_uniaxial(state):
    stress, x1, x2 = uniaxial_tensors(state)
    return np.concatenate(
        (
            mandel(stress),
            _DEVIATORIC_BASIS.T @ mandel(x1),
            _DEVIATORIC_BASIS.T @ mandel(x2),
            np.asarray(state)[3:],
        )
    )


def unpack_state(state):
    state = np.asarray(state, dtype=float)
    if state.shape != (18,) or not np.isfinite(state).all():
        raise ValueError("Require 18 finite tensor-state components.")
    return (
        tensor(state[:6]),
        tensor(_DEVIATORIC_BASIS @ state[6:11]),
        tensor(_DEVIATORIC_BASIS @ state[11:16]),
        float(state[16]),
        float(state[17]),
    )


def _elastic_matrix(parameters, poisson):
    if (
        isinstance(poisson, (bool, np.bool_))
        or np.iscomplexobj(poisson)
        or not np.isfinite(poisson)
        or not -1 < poisson < 0.5
    ):
        raise ValueError("Require a real Poisson ratio in (-1, .5).")
    return (
        parameters.E / (3 * (1 - 2 * poisson)) * np.outer(_TRACE, _TRACE)
        + parameters.E / (1 + poisson) * _DEV
    )


def packed_rates_and_jacobian(state, strain_rate_vector, parameters, *, poisson):
    """Analytical state Jacobian for the 18-variable candidate rate system.

    Internal Newton candidates may have negative accumulated strain; only a
    converged update is admitted. This function is not a material-state validator.
    """
    y = np.asarray(state, dtype=float)
    p = parameters
    elastic = _elastic_matrix(p, poisson)
    d = _DEVIATORIC_BASIS
    eta = _DEV @ y[:6] - d @ (y[6:11] + y[11:16])
    q = np.sqrt(1.5) * np.linalg.norm(eta)
    overstress = max(q - y[16] - p.k, 0.0)
    speed = (overstress / p.Z) ** p.n
    eta_gradient = np.zeros((6, 18))
    eta_gradient[:, :6] = _DEV
    eta_gradient[:, 6:11] = eta_gradient[:, 11:16] = -d
    speed_gradient = np.zeros(18)
    flow = np.zeros(6)
    flow_gradient = np.zeros((6, 18))
    if overstress > 0 and q > 0:
        normal = 1.5 * eta / q
        speed_slope = p.n / p.Z * (overstress / p.Z) ** (p.n - 1)
        speed_gradient = speed_slope * (normal @ eta_gradient)
        speed_gradient[16] -= speed_slope
        normal_gradient = (1.5 * _DEV - np.outer(normal, normal)) / q
        flow = speed * normal
        flow_gradient = (
            np.outer(normal, speed_gradient) + speed * normal_gradient @ eta_gradient
        )
    rate = np.zeros(18)
    jacobian = np.zeros((18, 18))
    rate[:6] = elastic @ (strain_rate_vector - flow)
    jacobian[:6] = -elastic @ flow_gradient
    gamma_tail = p.gamma2 * np.exp(-p.q * y[17])
    gamma = p.gamma1 + gamma_tail
    for start, coefficient, saturation in ((6, p.C1, p.a1), (11, p.C2, p.a2)):
        x = y[start : start + 5]
        norm = np.sqrt(1.5) * np.linalg.norm(x)
        static = norm ** (p.m - 1) * x
        static_gradient = (
            np.zeros((5, 5))
            if norm == 0
            else (
                norm ** (p.m - 1) * np.eye(5)
                + 1.5 * (p.m - 1) * norm ** (p.m - 3) * np.outer(x, x)
            )
        )
        rate[start : start + 5] = (
            coefficient * (2 / 3 * saturation * d.T @ flow - x * speed) - gamma * static
        )
        jacobian[start : start + 5] = coefficient * (
            2 / 3 * saturation * d.T @ flow_gradient - np.outer(x, speed_gradient)
        )
        jacobian[start : start + 5, start : start + 5] -= (
            coefficient * speed * np.eye(5) + gamma * static_gradient
        )
        jacobian[start : start + 5, 17] += p.q * gamma_tail * static
    gamma_r_tail = p.gamma_r2 * np.exp(-p.r * y[17])
    gamma_r = p.gamma_r1 + gamma_r_tail
    distance = p.Qr - y[16]
    recovery = abs(distance) ** (p.mr - 1) * distance
    rate[16] = (p.b * (p.Q - y[16]) + p.H) * speed + gamma_r * recovery
    jacobian[16] = (p.b * (p.Q - y[16]) + p.H) * speed_gradient
    jacobian[16, 16] -= p.b * speed + gamma_r * p.mr * abs(distance) ** (p.mr - 1)
    jacobian[16, 17] -= p.r * gamma_r_tail * recovery
    rate[17] = speed
    jacobian[17] = speed_gradient
    return rate, jacobian


def implicit_tensor_update(old, strain_increment, dt, parameters, *, poisson):
    """Coupled backward Euler and full Mandel d(stress)/d(strain); research only."""
    if (
        np.iscomplexobj(old)
        or np.asarray(old).shape != (18,)
        or not np.isfinite(old).all()
        or old[17] < 0
        or old[16] + parameters.k <= 0
        or isinstance(dt, (bool, np.bool_))
        or np.iscomplexobj(dt)
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError(
            "Require finite admissible old state and positive physical dt."
        )
    old = np.array(old, dtype=float, copy=True)
    increment = mandel(strain_increment)
    elastic = _elastic_matrix(parameters, poisson)
    scale = np.concatenate((np.full(17, parameters.k), [1.0]))
    strain_rate = increment / dt
    identity = np.eye(18)
    cached_candidate = None
    cached_evaluation = None

    def evaluate(candidate):
        nonlocal cached_candidate, cached_evaluation
        # Residual, Jacobian and final tangent often request the exact same
        # candidate. Cache only inside this one fixed-old-state local update;
        # MINPACK can reuse its input buffer, so retain an owned key copy.
        if cached_candidate is None or not np.array_equal(candidate, cached_candidate):
            value = packed_rates_and_jacobian(
                candidate, strain_rate, parameters, poisson=poisson
            )
            cached_candidate = candidate.copy()
            cached_evaluation = value
        return cached_evaluation

    def residual(candidate):
        rate, _ = evaluate(candidate)
        return (candidate - old - dt * rate) / scale

    def jacobian(candidate):
        _, derivative = evaluate(candidate)
        return (identity - dt * derivative) / scale[:, None]

    guess = old.copy()
    guess[:6] += elastic @ increment
    solved = root(
        residual, guess, jac=jacobian, method="hybr", options={"xtol": 1.0e-10}
    )
    state = solved.x
    if (
        not solved.success
        or not np.isfinite(state).all()
        or np.max(np.abs(residual(state))) > 1.0e-8
        or state[17] < old[17] - 1.0e-12
        or state[16] + parameters.k <= 0
    ):
        raise RuntimeError(f"Candidate tensor update failed: {solved.message}")
    _, derivative = evaluate(state)
    forcing = np.zeros((18, 6))
    forcing[:6] = elastic
    sensitivity = np.linalg.solve(identity - dt * derivative, forcing)
    return state.copy(), sensitivity[:6].copy()


def _symmetric(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3x3 tensor.")
    if not np.allclose(array, array.T, rtol=1.0e-12, atol=1.0e-13):
        raise ValueError(f"{name} must be symmetric.")
    return array


def deviator(value):
    return value - np.trace(value) * np.eye(3) / 3


def equivalent(value):
    """J2 stress norm, not the unscaled Frobenius norm."""
    return np.sqrt(1.5 * np.sum(value * value))


def tensor_rates(
    stress,
    backstress1,
    backstress2,
    isotropic,
    accumulated,
    strain_rate,
    parameters,
    *,
    poisson,
):
    """Candidate small-strain, isotropic, isothermal J2 rates.

    All inputs remain owned by the caller. Positive backstress follows the
    source paper, not NEML's opposite-sign hardening convention.
    """
    stress = _symmetric(stress, "stress")
    x1 = _symmetric(backstress1, "backstress1")
    x2 = _symmetric(backstress2, "backstress2")
    strain_rate = _symmetric(strain_rate, "strain_rate")
    if any(abs(np.trace(x)) > 1.0e-12 * max(1.0, np.linalg.norm(x)) for x in (x1, x2)):
        raise ValueError("Backstress tensors must be deviatoric.")
    if (
        any(
            np.iscomplexobj(x) or isinstance(x, (bool, np.bool_))
            for x in (isotropic, accumulated, poisson)
        )
        or not np.isfinite([isotropic, accumulated, poisson]).all()
        or accumulated < 0
        or not -1 < poisson < 0.5
        or parameters.k + isotropic <= 0
    ):
        raise ValueError(
            "Require valid Poisson ratio, nonnegative history and positive strength."
        )
    p = parameters
    effective = deviator(stress) - x1 - x2
    q = equivalent(effective)
    speed = (max(q - isotropic - p.k, 0.0) / p.Z) ** p.n
    flow = np.zeros((3, 3)) if q == 0 else 1.5 * effective * (speed / q)
    gamma = p.gamma1 + p.gamma2 * np.exp(-p.q * accumulated)
    gamma_r = p.gamma_r1 + p.gamma_r2 * np.exp(-p.r * accumulated)

    def backstress_rate(x, coefficient, saturation):
        return (
            coefficient * (2 / 3 * saturation * flow - x * speed)
            - gamma * equivalent(x) ** (p.m - 1) * x
        )

    elastic_rate = strain_rate - flow
    bulk = p.E / (3 * (1 - 2 * poisson))
    shear = p.E / (2 * (1 + poisson))
    response = {
        "stress": bulk * np.trace(elastic_rate) * np.eye(3)
        + 2 * shear * deviator(elastic_rate),
        "backstress1": backstress_rate(x1, p.C1, p.a1),
        "backstress2": backstress_rate(x2, p.C2, p.a2),
        "isotropic": (
            p.b * (p.Q - isotropic) * speed
            + gamma_r * abs(p.Qr - isotropic) ** p.mr * np.sign(p.Qr - isotropic)
            + p.H * speed
        ),
        "accumulated": speed,
        "inelastic": flow,
    }
    if any(not np.isfinite(value).all() for value in response.values()):
        raise ValueError("Candidate tensor rate left its finite numerical domain.")
    return response


def uniaxial_tensors(state):
    """Map positive scalar chi to diag(2 chi/3, -chi/3, -chi/3)."""
    if np.iscomplexobj(state):
        raise ValueError("Require real uniaxial state.")
    state = np.asarray(state, dtype=float)
    if state.shape != (5,) or not np.isfinite(state).all():
        raise ValueError("Require five finite uniaxial state components.")
    direction = np.diag([2 / 3, -1 / 3, -1 / 3])
    return np.diag([state[0], 0.0, 0.0]), direction * state[1], direction * state[2]


def reduced_uniaxial_rates(state, axial_strain_rate, parameters, *, poisson):
    """Stress-free lateral reduction; no plane-strain substitution."""
    stress, x1, x2 = uniaxial_tensors(state)
    trial = tensor_rates(
        stress,
        x1,
        x2,
        state[3],
        state[4],
        np.zeros((3, 3)),
        parameters,
        poisson=poisson,
    )
    flow = trial["inelastic"]
    axial_stress_rate = parameters.E * (axial_strain_rate - flow[0, 0])
    strain_rate = flow + np.diag(
        [
            axial_stress_rate / parameters.E,
            -poisson * axial_stress_rate / parameters.E,
            -poisson * axial_stress_rate / parameters.E,
        ]
    )
    result = tensor_rates(
        stress, x1, x2, state[3], state[4], strain_rate, parameters, poisson=poisson
    )
    return np.array(
        [
            result["stress"][0, 0],
            1.5 * result["backstress1"][0, 0],
            1.5 * result["backstress2"][0, 0],
            result["isotropic"],
            result["accumulated"],
        ]
    ), result["stress"][1:, 1:]

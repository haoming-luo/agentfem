"""Independent convergence tests; no PDEAgent-Bench reference data."""

import numpy as np
import pytest
from agentfem.integrations.pdeagent_bench.adapter import BenchmarkPolicy, solve_case


def polynomial_case(dt, *, explicit=False):
    q = "x*(1-x)*y*(1-y)*z*(1-z)"
    diffusion = "2*(y*(1-y)*z*(1-z)+x*(1-x)*z*(1-z)+x*(1-x)*y*(1-y))"
    advection = "0.8*(1-2*x)*y*(1-y)*z*(1-z)+0.3*x*(1-x)*(1-2*y)*z*(1-z)"
    time = {"t0": 0.0, "t_end": 0.2, "dt": dt}
    if explicit:
        time["scheme"] = "backward_euler"
    return {
        "pde": {
            "type": "convection_diffusion",
            "pde_params": {
                "epsilon": 0.2,
                "beta": [0.8, 0.3, 0.0],
                "stabilization": "supg",
            },
            "initial_condition": q,
            "source_term": f"3*(1+t)**2*({q})+(1+t)**3*(0.2*({diffusion})+({advection}))",
            "time": time,
        },
        "domain": {"type": "unit_cube"},
        "bc": {"dirichlet": {"on": "all", "value": 0.0}},
        "output": {
            "field": "u",
            "grid": {"bbox": [0, 1, 0, 1, 0, 1], "nx": 9, "ny": 9, "nz": 9},
        },
    }


@pytest.mark.parametrize("theta,minimum_ratio", [(0.5, 3.4), (1.0, 1.7)])
def test_transport_supg_time_order(theta, minimum_ratio):
    z, y, x = np.meshgrid(*([np.linspace(0, 1, 9)] * 3), indexing="ij")
    ref = 1.2**3 * x * (1 - x) * y * (1 - y) * z * (1 - z)
    errors = []
    for dt in [0.05, 0.025]:
        value = solve_case(
            polynomial_case(dt),
            BenchmarkPolicy(
                spatial_transport_resolution=2,
                spatial_transport_degree=3,
                spatial_transport_theta=theta,
            ),
        )
        errors.append(np.linalg.norm(value["u"] - ref) / np.linalg.norm(ref))
    assert errors[0] / errors[1] > minimum_ratio, errors


def test_explicit_backward_euler_is_preserved():
    result = solve_case(
        polynomial_case(0.1, explicit=True),
        BenchmarkPolicy(spatial_transport_resolution=2),
    )
    assert result["solver_info"]["time_scheme"] == "backward_euler"
    assert result["solver_info"]["n_steps"] == 2


def test_three_dimensional_transport_time_varying_boundary():
    # Spatially linear exact field: validates old/new forcing and endpoint BCs.
    case = polynomial_case(0.05)
    case["pde"].update(
        initial_condition="x+2*y+3*z+1",
        source_term="2*(1+t)*(x+2*y+3*z+1)+1.4*(1+t)**2",
    )
    case["bc"]["dirichlet"]["value"] = "(1+t)**2*(x+2*y+3*z+1)"
    z, y, x = np.meshgrid(*([np.linspace(0, 1, 9)] * 3), indexing="ij")
    reference = 1.2**2 * (x + 2 * y + 3 * z + 1)
    errors = []
    for dt in [0.05, 0.025]:
        case["pde"]["time"]["dt"] = dt
        result = solve_case(case, BenchmarkPolicy(spatial_transport_resolution=2))
        errors.append(
            np.linalg.norm(result["u"] - reference) / np.linalg.norm(reference)
        )
    # Trapezoidal time integration is exact for this quadratic history.
    assert max(errors) < 1e-8

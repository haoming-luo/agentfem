import numpy as np
from agentfem.integrations.pdeagent_bench.adapter import BenchmarkPolicy, solve_case


def test_three_dimensional_heat_second_order_time_refinement():
    # A tensor-quadratic spatial field is represented exactly. The cubic
    # time factor isolates the temporal method without a benchmark oracle.
    q = "x*(1-x)*y*(1-y)*z*(1-z)"
    lap = "y*(1-y)*z*(1-z)+x*(1-x)*z*(1-z)+x*(1-x)*y*(1-y)"
    case = {
        "pde": {
            "type": "heat",
            "initial_condition": q,
            "source_term": f"3*(1+t)**2*({q})+2*(1+t)**3*({lap})",
            "coefficients": {"kappa": {"type": "constant", "value": 1.0}},
            "time": {
                "t0": 0.0,
                "t_end": 0.2,
                "dt": 0.1,
                "scheme": "crank_nicolson",
            },
        },
        "domain": {"type": "unit_cube"},
        "bc": {"dirichlet": {"on": "all", "value": 0.0}},
        "output": {
            "field": "u",
            "grid": {"bbox": [0, 1, 0, 1, 0, 1], "nx": 9, "ny": 9, "nz": 9},
        },
    }
    axis = np.linspace(0, 1, 9)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    ref = 1.2**3 * x * (1 - x) * y * (1 - y) * z * (1 - z)
    errors = []
    for refinement in [1, 2]:
        result = solve_case(
            case, policy=BenchmarkPolicy(spatial_time_refinement=refinement)
        )
        errors.append(np.linalg.norm(result["u"] - ref) / np.linalg.norm(ref))
        assert result["solver_info"]["time_scheme"] == "crank_nicolson"
    assert errors[0] / errors[1] > 3.2
    assert errors[1] < 1e-3


def test_three_dimensional_poisson_resolves_directional_bandwidth():
    case = {
        "pde": {
            "type": "poisson",
            "source_term": "32.2*pi**2*sin(2*pi*x)*sin(pi*y)*sin(3*pi*z)",
            "coefficients": {"kappa": {"type": "constant", "value": 2.3}},
        },
        "domain": {"type": "unit_cube"},
        "bc": {"dirichlet": {"on": "all", "value": 0.0}},
        "output": {
            "field": "u",
            "grid": {"bbox": [0, 1, 0, 1, 0, 1], "nx": 13, "ny": 13, "nz": 13},
        },
    }
    axis = np.linspace(0, 1, 13)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    ref = np.sin(2 * np.pi * x) * np.sin(np.pi * y) * np.sin(3 * np.pi * z)
    result = solve_case(case)
    assert np.linalg.norm(result["u"] - ref) / np.linalg.norm(ref) < 5e-4


def test_three_dimensional_transport_uses_explicit_theta_and_supg_policy():
    q = "x*(1-x)*y*(1-y)*z*(1-z)"
    grad = (
        "(1-2*x)*y*(1-y)*z*(1-z)"
        "+0.5*x*(1-x)*(1-2*y)*z*(1-z)"
        "-0.25*x*(1-x)*y*(1-y)*(1-2*z)"
    )
    curvature = (
        "y*(1-y)*z*(1-z)+x*(1-x)*z*(1-z)+x*(1-x)*y*(1-y)"
    )
    case = {
        "pde": {
            "type": "convection_diffusion",
            "pde_params": {
                "epsilon": 0.2,
                "beta": [1.0, 0.5, -0.25],
            },
            "initial_condition": q,
            "source_term": (
                f"3*(1+t)**2*({q})+(1+t)**3*(({grad})+0.4*({curvature}))"
            ),
            "time": {
                "t0": 0.0,
                "t_end": 0.2,
                "dt": 0.05,
                "scheme": "crank_nicolson",
            },
        },
        "domain": {"type": "unit_cube"},
        "bc": {"dirichlet": {"on": "all", "value": 0.0}},
        "output": {
            "field": "u",
            "grid": {"bbox": [0, 1, 0, 1, 0, 1], "nx": 9, "ny": 9, "nz": 9},
        },
    }
    axis = np.linspace(0, 1, 9)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    ref = 1.2**3 * x * (1 - x) * y * (1 - y) * z * (1 - z)

    result = solve_case(case)

    assert np.linalg.norm(result["u"] - ref) / np.linalg.norm(ref) < 5e-4
    assert result["solver_info"]["time_scheme"] == "crank_nicolson"
    assert result["solver_info"]["stabilization"] == "supg"
    assert result["solver_info"]["matrix_reused"] is True

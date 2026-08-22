from __future__ import annotations

import json

import numpy as np
import pytest

from agentfem.integrations.pdeagent_bench import (
    BENCHMARK_COMMIT,
    SUPPORTED_FAMILIES,
    BenchmarkContractError,
    BenchmarkPolicy,
    combine_official_summaries,
    read_official_summary,
    solve_case,
    validate_case_spec,
)


def _case(pde, *, field="scalar"):
    return {
        "id": "public_contract_smoke",
        "pde_classification": {"equation_type": pde["type"]},
        "pde": pde,
        "domain": {"type": "unit_square"},
        "bc": {"dirichlet": {"on": "all", "value": "0.0"}},
        "output": {
            "format": "npz",
            "field": field,
            "grid": {"bbox": [0.0, 1.0, 0.0, 1.0], "nx": 9, "ny": 7},
        },
    }


def test_benchmark_schema_rejects_unknown_pde_without_guessing():
    case = _case({"type": "unknown_equation"})

    with pytest.raises(BenchmarkContractError, match="AFM-PDEB-008"):
        validate_case_spec(case)


def test_benchmark_family_capability_is_machine_readable():
    assert SUPPORTED_FAMILIES == {
        "burgers",
        "biharmonic",
        "poisson",
        "heat",
        "linear_elasticity",
        "navier_stokes",
        "helmholtz",
        "convection_diffusion",
        "reaction_diffusion",
        "stokes",
        "wave",
    }


def test_poisson_adapter_returns_strict_grid_and_solver_evidence():
    case = _case({"type": "poisson", "source_term": "2*pi**2*sin(pi*x)*sin(pi*y)"})
    case["bc"]["dirichlet"]["value"] = "sin(pi*x)*sin(pi*y)"

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=8))

    assert result["u"].shape == (7, 9)
    assert np.all(np.isfinite(result["u"]))
    assert result["solver_info"]["benchmark_commit"] == BENCHMARK_COMMIT
    assert result["solver_info"]["converged"] is True
    assert result["solver_info"]["coverage"] == 1.0


def test_heat_adapter_returns_initial_and_final_grids():
    case = _case(
        {
            "type": "heat",
            "coefficients": {"kappa": {"type": "constant", "value": 1.0}},
            "time": {"t0": 0.0, "t_end": 0.02, "dt": 0.01, "scheme": "backward_euler"},
            "source_term": "0.0",
            "initial_condition": "sin(pi*x)*sin(pi*y)",
        }
    )

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=8))

    assert result["u"].shape == (7, 9)
    assert result["u_initial"].shape == (7, 9)
    assert result["solver_info"]["num_timesteps"] == 2
    assert result["solver_info"]["matrix_reused"] is True


def test_burgers_periodic_adapter_closes_matching_faces():
    pytest.importorskip("dolfinx_mpc")
    case = _case(
        {
            "type": "burgers",
            "pde_params": {"nu": 0.01},
            "source_term": (
                "((-1 + 0.04*pi**2)*exp(t) + 2*pi*cos(2*pi*x))*exp(-2*t)*sin(2*pi*x)"
            ),
            "initial_condition": "sin(2*pi*x)",
            "t_final": 0.02,
            "dt": 0.01,
        }
    )
    case["domain"] = {
        "type": "periodic_square",
        "geometry_params": {"bounds": [0.0, 1.0, 0.0, 1.0]},
    }
    case["bc"] = {"periodic": {"on": ["x", "y"]}}

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=8))

    assert result["solver_info"]["boundary_method"] == "periodic_mpc"
    assert result["solver_info"]["ksp_type"] == "preonly"
    assert np.allclose(result["u"][:, 0], result["u"][:, -1], atol=1.0e-10)
    assert np.allclose(result["u"][0, :], result["u"][-1, :], atol=1.0e-10)


@pytest.mark.parametrize(
    ("pde", "boundary", "exact", "tolerance"),
    [
        (
            {
                "type": "helmholtz",
                "pde_params": {"k": 2.0},
                "source_term": "(-4 + 2*pi**2)*sin(pi*x)*sin(pi*y)",
            },
            "sin(pi*x)*sin(pi*y)",
            lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y),
            2.0e-3,
        ),
        (
            {
                "type": "convection_diffusion",
                "pde_params": {
                    "epsilon": 0.1,
                    "beta": [1.0, 0.5],
                    "stabilization": "supg",
                },
                "source_term": (
                    "0.2*pi**2*sin(pi*x)*sin(pi*y) + "
                    "pi*cos(pi*x)*sin(pi*y) + "
                    "0.5*pi*sin(pi*x)*cos(pi*y)"
                ),
            },
            "sin(pi*x)*sin(pi*y)",
            lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y),
            4.0e-3,
        ),
    ],
)
def test_steady_scalar_families_follow_public_equations(
    pde, boundary, exact, tolerance
):
    case = _case(pde)
    case["bc"]["dirichlet"]["value"] = boundary

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=12))
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.0, 7)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    reference = exact(xx, yy)
    error = np.linalg.norm(result["u"] - reference) / np.linalg.norm(reference)

    assert error < tolerance


def test_transient_supg_uses_the_complete_backward_euler_residual():
    end_time = 0.02
    mode = "sin(pi*x)*sin(pi*y)"
    case = _case(
        {
            "type": "convection_diffusion",
            "pde_params": {
                "epsilon": 0.1,
                "beta": [1.0, 0.5],
                "stabilization": "supg",
            },
            "time": {"t0": 0.0, "t_end": end_time, "dt": 0.002},
            "source_term": (
                f"(-1 + 0.2*pi**2)*{mode}*exp(-t) + "
                "pi*cos(pi*x)*sin(pi*y)*exp(-t) + "
                "0.5*pi*sin(pi*x)*cos(pi*y)*exp(-t)"
            ),
            "initial_condition": mode,
        }
    )
    case["bc"]["dirichlet"]["value"] = f"{mode}*exp(-t)"

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=12))
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.0, 7)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    reference = np.sin(np.pi * xx) * np.sin(np.pi * yy) * np.exp(-end_time)
    error = np.linalg.norm(result["u"] - reference) / np.linalg.norm(reference)

    assert error < 4.0e-3
    assert result["solver_info"]["n_steps"] == 10
    assert result["solver_info"]["time_scheme"] == "backward_euler"


def test_nonlinear_reaction_diffusion_advances_with_transactional_state():
    case = _case(
        {
            "type": "reaction_diffusion",
            "pde_params": {
                "epsilon": 0.05,
                "reaction": {"type": "allen_cahn", "lambda": 1.0},
            },
            "time": {
                "t0": 0.0,
                "t_end": 0.02,
                "dt": 0.01,
                "scheme": "backward_euler",
            },
            "source_term": "0.0",
            "initial_condition": "0.1*sin(pi*x)*sin(pi*y)",
        }
    )

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=8))

    assert result["solver_info"]["reaction_type"] == "allen_cahn"
    assert result["solver_info"]["num_timesteps"] == 2
    assert result["solver_info"]["nonlinear_iterations_total"] > 0
    assert np.all(np.isfinite(result["u"]))


def test_logistic_reaction_uses_positive_residual_convention():
    end_time = 0.02
    mode = "sin(pi*x)*sin(pi*y)"
    solution = f"0.15 + 0.05*{mode}*exp(-t)"
    case = _case(
        {
            "type": "reaction_diffusion",
            "pde_params": {
                "epsilon": 0.1,
                "reaction": {"type": "logistic", "rho": 2.0},
            },
            "time": {
                "t0": 0.0,
                "t_end": end_time,
                "dt": 0.002,
                "scheme": "backward_euler",
            },
            "source_term": (
                f"(-0.05 + 0.01*pi**2)*{mode}*exp(-t) + "
                f"2.0*({solution})*(1.0 - ({solution}))"
            ),
            "initial_condition": f"0.15 + 0.05*{mode}",
        }
    )
    case["bc"]["dirichlet"]["value"] = "0.15"

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=12))
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.0, 7)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    reference = 0.15 + 0.05 * np.sin(np.pi * xx) * np.sin(np.pi * yy) * np.exp(
        -end_time
    )
    error = np.linalg.norm(result["u"] - reference) / np.linalg.norm(reference)

    assert error < 2.0e-3
    assert len(result["solver_info"]["nonlinear_iterations"]) == 10
    assert result["solver_info"]["n_steps"] == 10


def test_wave_newmark_path_preserves_a_public_manufactured_mode():
    end_time = 0.02
    case = _case(
        {
            "type": "wave",
            "pde_params": {"c": 1.0},
            "time": {"t0": 0.0, "t_end": end_time, "dt": 0.002},
            "source_term": "(-1 + 2*pi**2)*sin(pi*x)*sin(pi*y)*cos(t)",
            "initial_condition": "sin(pi*x)*sin(pi*y)",
            "initial_velocity": "0.0",
        }
    )
    case["bc"]["dirichlet"]["value"] = "sin(pi*x)*sin(pi*y)*cos(t)"

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=10))
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.0, 7)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    reference = np.sin(np.pi * xx) * np.sin(np.pi * yy) * np.cos(end_time)
    error = np.linalg.norm(result["u"] - reference) / np.linalg.norm(reference)

    assert error < 4.0e-3
    assert result["solver_info"]["time_integrator"] == "newmark_average_acceleration"


def test_burgers_adapter_advances_a_public_manufactured_mode():
    end_time = 0.02
    mode = "sin(pi*x)*sin(pi*y)"
    case = _case(
        {
            "type": "burgers",
            "pde_params": {"nu": 0.01},
            "t_final": end_time,
            "dt": 0.002,
            "source_term": (
                f"((-1 + 0.02*pi**2)*exp(t) + pi*sin(pi*(x + y)))*exp(-2*t)*{mode}"
            ),
            "initial_condition": mode,
        }
    )
    case["bc"]["dirichlet"]["value"] = f"exp(-t)*{mode}"

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=12))
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.0, 7)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    reference = np.exp(-end_time) * np.sin(np.pi * xx) * np.sin(np.pi * yy)
    error = np.linalg.norm(result["u"] - reference) / np.linalg.norm(reference)

    assert error < 4.0e-3
    assert result["solver_info"]["formulation"] == "semi_implicit_scalar_burgers"
    assert result["solver_info"]["n_steps"] == 10


def test_stokes_adapter_recovers_a_public_divergence_free_velocity():
    u0 = "pi*sin(pi*x)*cos(pi*y)"
    u1 = "-pi*cos(pi*x)*sin(pi*y)"
    case = _case(
        {
            "type": "stokes",
            "pde_params": {"nu": 0.5},
            "source_term": [f"pi**2*({u0})", f"pi**2*({u1})"],
        },
        field="velocity_magnitude",
    )
    case["bc"]["dirichlet"]["value"] = [u0, u1]

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=10))
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.0, 7)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    exact_u0 = np.pi * np.sin(np.pi * xx) * np.cos(np.pi * yy)
    exact_u1 = -np.pi * np.cos(np.pi * xx) * np.sin(np.pi * yy)
    reference = np.sqrt(exact_u0**2 + exact_u1**2)
    error = np.linalg.norm(result["u"] - reference) / np.linalg.norm(reference)

    assert error < 5.0e-3
    assert result["solver_info"]["formulation"] == "Taylor-Hood"
    assert result["solver_info"]["mixed_num_dofs"] > result["solver_info"]["num_dofs"]


def test_three_dimensional_stokes_uses_block_taylor_hood_and_pressure_nullspace():
    case = {
        "id": "public_3d_stokes_contract_smoke",
        "pde_classification": {"equation_type": "stokes", "dim": 3},
        "pde": {
            "type": "stokes",
            "pde_params": {"nu": 1.0},
            "source_term": [
                "pi*(6*pi*sin(pi*x)*cos(pi*y)*cos(pi*z) + sin(pi*y)*sin(pi*z)*cos(pi*x))",
                "pi*(sin(pi*x)*sin(pi*z)*cos(pi*y) - 3*pi*sin(pi*y)*cos(pi*x)*cos(pi*z))",
                "pi*(sin(pi*x)*sin(pi*y)*cos(pi*z) - 3*pi*sin(pi*z)*cos(pi*x)*cos(pi*y))",
            ],
        },
        "domain": {"type": "unit_cube"},
        "bc": {
            "dirichlet": {
                "on": "all",
                "value": [
                    "2*sin(pi*x)*cos(pi*y)*cos(pi*z)",
                    "-sin(pi*y)*cos(pi*x)*cos(pi*z)",
                    "-sin(pi*z)*cos(pi*x)*cos(pi*y)",
                ],
            }
        },
        "output": {
            "format": "npz",
            "field": "velocity_magnitude",
            "grid": {
                "bbox": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "nx": 7,
                "ny": 7,
                "nz": 7,
            },
        },
    }

    result = solve_case(case)
    axis = np.linspace(0.0, 1.0, 7)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    exact = np.sqrt(
        (2.0 * np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)) ** 2
        + (np.sin(np.pi * y) * np.cos(np.pi * x) * np.cos(np.pi * z)) ** 2
        + (np.sin(np.pi * z) * np.cos(np.pi * x) * np.cos(np.pi * y)) ** 2
    )
    relative_error = np.linalg.norm(result["u"] - exact) / np.linalg.norm(exact)

    assert relative_error < 3.0e-4
    assert result["solver_info"]["formulation"] == "block_taylor_hood"
    assert result["solver_info"]["pressure_reference"] == "constant_nullspace"
    assert result["solver_info"]["velocity_degree"] == 3
    assert result["solver_info"]["pressure_degree"] == 2
    assert result["solver_info"]["converged"] is True


def test_navier_stokes_adapter_uses_stokes_predictor_and_newton():
    case = _case(
        {
            "type": "navier_stokes",
            "pde_params": {"nu": 0.1},
            "source_term": ["0.0", "0.0"],
        },
        field="velocity_magnitude",
    )
    case["bc"]["dirichlet"]["value"] = ["y", "0.0"]

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=8))
    y = np.linspace(0.0, 1.0, 7)
    reference = np.repeat(y[:, None], 9, axis=1)
    error = np.linalg.norm(result["u"] - reference) / np.linalg.norm(reference)

    assert error < 1.0e-8
    assert result["solver_info"]["formulation"] == "steady_newton_taylor_hood"
    assert result["solver_info"]["initialization"] == "stokes_predictor"


def test_biharmonic_adapter_uses_public_two_poisson_closure():
    case = _case(
        {
            "type": "biharmonic",
            "source_term": "4*pi**4*sin(pi*x)*sin(pi*y)",
        }
    )
    case["bc"]["dirichlet"]["value"] = "sin(pi*x)*sin(pi*y)"

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=10))
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.0, 7)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    reference = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    error = np.linalg.norm(result["u"] - reference) / np.linalg.norm(reference)

    assert error < 2.0e-3
    assert result["solver_info"]["formulation"] == "mixed_two_poisson"
    assert (
        result["solver_info"]["auxiliary_boundary"]
        == "negative_laplacian_of_public_boundary_expression"
    )


def test_biharmonic_constant_boundary_has_zero_auxiliary_laplacian():
    case = _case({"type": "biharmonic", "source_term": "1.0"})

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=6))

    assert np.all(np.isfinite(result["u"]))
    assert result["solver_info"]["converged"] is True


def test_biharmonic_boundary_derivative_stays_on_imported_mesh():
    pytest.importorskip("gmsh")
    case = _case({"type": "biharmonic", "source_term": "0.0"})
    case["domain"] = {
        "type": "star",
        "char_length": 0.25,
        "geometry_params": {
            "center": [0.0, 0.0],
            "points": 5,
            "inner_r": 0.35,
            "outer_r": 0.75,
        },
    }
    case["bc"]["dirichlet"]["value"] = "x**2 + y**2"
    case["output"]["grid"] = {
        "bbox": [-0.8, 0.8, -0.8, 0.8],
        "nx": 15,
        "ny": 15,
    }

    result = solve_case(case, policy=BenchmarkPolicy(planar_resolution=6))

    assert np.any(np.isfinite(result["u"]))
    assert (
        result["solver_info"]["auxiliary_boundary_representation"]
        == "same_mesh_finite_element"
    )


def test_three_dimensional_poisson_uses_the_public_equation_not_an_oracle():
    case = {
        "id": "public_3d_contract_smoke",
        "pde_classification": {"equation_type": "poisson"},
        "pde": {
            "type": "poisson",
            "source_term": "3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)",
        },
        "domain": {"type": "unit_cube"},
        "bc": {
            "dirichlet": {
                "on": "all",
                "value": "sin(pi*x)*sin(pi*y)*sin(pi*z)",
            }
        },
        "output": {
            "format": "npz",
            "field": "scalar",
            "grid": {
                "bbox": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "nx": 9,
                "ny": 9,
                "nz": 9,
            },
        },
    }

    result = solve_case(
        case,
        policy=BenchmarkPolicy(spatial_resolution=8, spatial_degree=2),
    )
    axis = np.linspace(0.0, 1.0, 9)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    exact = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    relative_error = np.linalg.norm(result["u"] - exact) / np.linalg.norm(exact)

    assert relative_error < 5.0e-3
    assert result["solver_info"]["dimension"] == 3


def test_resolution_bandwidth_uses_trigonometric_phase_not_source_amplitude():
    policy = BenchmarkPolicy()
    high_amplitude = {"source_term": "(-1 + 20*pi**2)*sin(pi*x)*sin(pi*y)*sin(pi*z)"}
    high_frequency = {"source_term": "sin(4*pi*x)*sin(pi*y)*sin(pi*z)"}

    assert policy.resolution(3, {"type": "unit_cube"}, high_amplitude) == 14
    assert policy.resolution(3, {"type": "unit_cube"}, high_frequency) == 16


def test_flow_resolution_uses_dimension_geometry_and_public_sampling_density():
    policy = BenchmarkPolicy()
    boundary = {"dirichlet": {"on": "all", "value": [0.0, 0.0]}}
    output = {"grid": {"nx": 144, "ny": 144}}

    assert (
        policy.flow_resolution(
            2,
            {"type": "periodic_square"},
            {"type": "stokes", "source_term": ["sin(2*pi*x)", "0.0"]},
            boundary,
            output,
        )
        == 72
    )
    assert (
        policy.flow_resolution(
            3,
            {"type": "unit_cube"},
            {"type": "stokes", "source_term": ["0.0", "0.0", "0.0"]},
            boundary,
            {"grid": {"nx": 16, "ny": 16, "nz": 16}},
        )
        == 4
    )


def test_official_summary_is_normalized_to_failure_taxonomy(tmp_path):
    source = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "case_id": "ok",
                        "equation_type": "poisson",
                        "status": "PASS",
                        "error": 1e-5,
                        "time": 0.2,
                        "gate_breakdown": {"final_pass": True},
                    },
                    {
                        "case_id": "slow",
                        "equation_type": "poisson",
                        "status": "FAIL",
                        "error": 1e-5,
                        "time": 2.0,
                        "gate_breakdown": {
                            "final_pass": False,
                            "failure_stage": "time",
                        },
                    },
                ]
            }
        )
    )

    catalog = tmp_path / "benchmark.jsonl"
    catalog.write_text(
        json.dumps({"id": "ok", "pde_classification": {"dim": 2}})
        + "\n"
        + json.dumps({"id": "slow", "pde_classification": {"dim": 3}})
        + "\n"
    )
    report = read_official_summary(source, case_catalog=catalog)

    assert report.total == 2
    assert report.passed == 1
    assert report.failures == {"TIME_FAIL": 1}
    assert report.by_family["poisson"]["pass_rate"] == 0.5
    assert report.by_dimension["2"]["pass_rate"] == 1.0
    assert report.by_dimension["3"]["pass_rate"] == 0.0
    assert "Failure taxonomy" in report.markdown()


def test_disjoint_official_family_runs_combine_without_double_counting(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "case_id": "poisson_ok",
                        "equation_type": "poisson",
                        "status": "PASS",
                        "gate_breakdown": {"final_pass": True},
                    }
                ]
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "case_id": "stokes_accuracy",
                        "equation_type": "stokes",
                        "status": "FAIL",
                        "gate_breakdown": {
                            "final_pass": False,
                            "failure_stage": "accuracy",
                        },
                    }
                ]
            }
        )
    )

    report = combine_official_summaries((first, second))

    assert report.total == 2
    assert report.passed == 1
    assert set(report.by_family) == {"poisson", "stokes"}
    assert report.as_dict()["source"] == [str(first), str(second)]
    with pytest.raises(ValueError, match="duplicate benchmark case"):
        combine_official_summaries((first, first))

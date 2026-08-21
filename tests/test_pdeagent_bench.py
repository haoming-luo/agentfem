from __future__ import annotations

import json

import numpy as np
import pytest

from agentfem.integrations.pdeagent_bench import (
    BENCHMARK_COMMIT,
    BenchmarkContractError,
    BenchmarkPolicy,
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
    high_amplitude = {
        "source_term": "(-1 + 20*pi**2)*sin(pi*x)*sin(pi*y)*sin(pi*z)"
    }
    high_frequency = {"source_term": "sin(4*pi*x)*sin(pi*y)*sin(pi*z)"}

    assert policy.resolution(3, {"type": "unit_cube"}, high_amplitude) == 14
    assert policy.resolution(3, {"type": "unit_cube"}, high_frequency) == 16


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

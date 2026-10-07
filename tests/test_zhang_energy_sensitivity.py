"""Energy-attribution diagnostics must not create a promotion shortcut."""

import numpy as np
import pytest

from zhang_2021_energy_sensitivity import assess
from zhang_2021_plane_strain_promotion import assess_convergence


def _runs():
    return [
        {
            "path": f"level-{index}.json",
            "sha256": str(index) * 64,
            "candidate": {
                "mesh_size": size,
                "geometry_source": "figure-10a",
                "formulation": "Q9/DPC1",
                "quadrature_degree": 4,
                "requested_fixed_increments": 40,
                "mpi_ranks": 1,
            },
            "payload": {
                "runtime": {
                    "manifest": {
                        "identity": {
                            "execution": {
                                "source": {
                                    "scientific_runtime_sha256": "a" * 64,
                                }
                            }
                        }
                    }
                },
                "benchmark_implementation": {
                    "schema": "agentfem.benchmark-implementation-identity.v1",
                    "files": [
                        {"name": "driver.py", "sha256": "b" * 64},
                        {"name": "fixture.py", "sha256": "c" * 64},
                    ],
                },
                "first_piola": [1.0, 2.0, 3.0, 4.0],
                "homogenized_algorithmic_tangent": {"values": np.eye(4).tolist()},
                "mixed_elastic_energy_diagnostics": {
                    "primal_elastic_energy_density": 2.0 + defect,
                    "condensed_elastic_energy_density": 2.0,
                    "pressure_orthogonality_density": 0.0,
                    "pressure_constraint_defect_energy_density": defect,
                },
            },
        }
        for index, (size, defect) in enumerate(((0.2, 0.4), (0.1, 0.2), (0.05, 0.1)))
    ]


def test_pressure_defect_attribution_does_not_promote_unconverged_energy():
    report = assess(_runs())
    assert report["successive_changes"][-1][
        "signed_defect_fraction_of_primal_change"
    ] == pytest.approx(1.0)
    assert not report["benchmark_promotion_authorized"]
    assert not report["mesh_audit"]["checks"]["elastic_energy_density"]["passed"]


@pytest.mark.parametrize(
    "corruption", ["missing", "nan", "identity", "geometry", "loading", "source"]
)
def test_reject_invalid_energy_or_uncontrolled_problem(corruption):
    runs = _runs()
    energy = runs[-1]["payload"]["mixed_elastic_energy_diagnostics"]
    if corruption == "missing":
        del energy["condensed_elastic_energy_density"]
    elif corruption == "nan":
        energy["pressure_constraint_defect_energy_density"] = float("nan")
    elif corruption == "identity":
        energy["primal_elastic_energy_density"] += 1.0
    elif corruption == "geometry":
        runs[-1]["candidate"]["geometry_source"] = "section-3.2.1-text"
    elif corruption == "source":
        runs[-1]["payload"]["runtime"]["manifest"]["identity"]["execution"]["source"][
            "scientific_runtime_sha256"
        ] = "d" * 64
    else:
        runs[-1]["candidate"]["macroscopic_deformation_gradient"] = [
            [1.0, 0.2],
            [0.0, 1.0],
        ]
    with pytest.raises(ValueError):
        assess(runs)


def test_zero_energy_change_has_no_invented_fraction():
    runs = _runs()
    for run in runs[1:]:
        run["payload"]["mixed_elastic_energy_diagnostics"] = dict(
            runs[0]["payload"]["mixed_elastic_energy_diagnostics"]
        )
    report = assess(runs)
    assert all(
        item["signed_defect_fraction_of_primal_change"] is None
        for item in report["successive_changes"]
    )


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("macroscopic_deformation_gradient", [[1.0, 0.2], [0.0, 1.0]]),
        ("cell_repetitions", [2, 1]),
        ("reference_cell_area", 2.0),
        ("deformation_gradient_path", {"name": "different_path"}),
    ],
)
def test_promotion_axis_rejects_changed_physical_problem(parameter, value):
    runs = _runs()
    runs[-1]["candidate"][parameter] = value
    report = assess_convergence(mesh_runs=runs)
    assert not report["axis_audits"]["mesh_converged"]["setup_consistent"]

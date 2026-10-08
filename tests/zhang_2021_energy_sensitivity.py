"""Explain mixed-energy mesh changes without changing benchmark acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zhang_2021_plane_strain_promotion import (
    _axis_audit,
    _benchmark_identity,
    _scientific_identity,
    load_candidate,
)


CHANNELS = (
    "primal_elastic_energy_density",
    "condensed_elastic_energy_density",
    "pressure_orthogonality_density",
    "pressure_constraint_defect_energy_density",
)


def assess(runs) -> dict[str, object]:
    selected = sorted(runs, key=lambda item: -float(item["candidate"]["mesh_size"]))
    if (
        len({_scientific_identity(run["payload"]) for run in selected}) != 1
        or len({_benchmark_identity(run["payload"]) for run in selected}) != 1
    ):
        raise ValueError(
            "Energy sensitivity requires identical runtime and benchmark sources."
        )
    control = _axis_audit(
        selected,
        name="mesh_size",
        coordinate=lambda item: float(item["mesh_size"]),
        fine_order=lambda item: -float(item["mesh_size"]),
        invariant_parameters=(
            "formulation",
            "quadrature_degree",
            "requested_fixed_increments",
            "mpi_ranks",
            "macroscopic_deformation_gradient",
            "cell_repetitions",
            "reference_cell_area",
            "deformation_gradient_path",
        ),
    )
    if not control["setup_consistent"]:
        raise ValueError(
            "Energy sensitivity requires at least three controlled mesh levels."
        )
    rows = []
    for run in selected:
        energy = run["payload"].get("mixed_elastic_energy_diagnostics", {})
        try:
            values = np.asarray([energy[key] for key in CHANNELS], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Candidate lacks explicit mixed-energy channels.") from exc
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("Energy channels must be finite scalars.")
        scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
        tolerance = 1024.0 * np.finfo(float).eps * scale
        residual = float(values[0] - sum(values[1:]))
        if abs(residual) > tolerance or values[3] < -tolerance:
            raise ValueError("Candidate violates the mixed-energy decomposition.")
        rows.append(
            {
                "mesh_size": run["candidate"]["mesh_size"],
                "channels": dict(zip(CHANNELS, values.tolist())),
                "identity_residual": residual,
            }
        )
    changes = []
    for coarse, fine in zip(rows, rows[1:]):
        delta = {
            key: fine["channels"][key] - coarse["channels"][key] for key in CHANNELS
        }
        scale = max(
            *(abs(row["channels"][CHANNELS[0]]) for row in (coarse, fine)),
            np.finfo(float).tiny,
        )
        resolved = abs(delta[CHANNELS[0]]) > 1024.0 * np.finfo(float).eps * scale
        changes.append(
            {
                "coarse_mesh_size": coarse["mesh_size"],
                "fine_mesh_size": fine["mesh_size"],
                "signed_channel_changes": delta,
                "signed_defect_fraction_of_primal_change": (
                    delta[CHANNELS[3]] / delta[CHANNELS[0]] if resolved else None
                ),
                "primal_change_resolved": bool(resolved),
            }
        )
    return {
        "schema": "agentfem.zhang-2021-energy-sensitivity.v1",
        "benchmark_promotion_authorized": False,
        "interpretation": "Algebraic change attribution, not an error estimate or convergence certificate.",
        "rows": rows,
        "successive_changes": changes,
        "mesh_audit": control,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = assess([load_candidate(path) for path in arguments.mesh_run])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(arguments.output)
    print(f"Diagnostic only; no benchmark promotion: {arguments.output}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Record bounded contact sensitivity without claiming external validation.

The same reviewed synthetic tool, geometry, material and terminal motion are
used for each run. This deliberately varies one numerical choice at a time:
mesh density, load increments, or penalty stiffness. It is a small acceptance
fixture, not a proof of asymptotic convergence or hard-contact accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from finite_j2_closed_planar_branch import build_case


CASES = (
    ("baseline", 1, 6, 5000.0),
    ("mesh_2", 2, 6, 5000.0),
    ("mesh_4", 4, 6, 5000.0),
    ("mesh_8", 8, 6, 5000.0),
    ("mesh_16", 16, 6, 5000.0),
    ("increments_3", 1, 3, 5000.0),
    ("increments_12", 1, 12, 5000.0),
    ("increments_24", 1, 24, 5000.0),
    ("penalty_2500", 1, 6, 2500.0),
    ("penalty_10000", 1, 6, 10000.0),
    ("penalty_20000", 1, 6, 20000.0),
)


def run_ladder() -> list[dict[str, object]]:
    """Run the frozen one-axis-at-a-time matrix and retain failed cases."""

    records = []
    for label, cells, increments, penalty in CASES:
        record: dict[str, object] = {
            "case": label,
            "cells_per_axis": cells,
            "increments": increments,
            "penalty": penalty,
            "status": "not_started",
        }
        start = perf_counter()
        try:
            step, displacement = build_case(
                cells=cells, increments=increments, penalty=penalty,
            )
            result = step.solve_result()
            record["status"] = result.status
            if result.status == "completed":
                energy = step.energy_history[-1]
                contact = step.contact_provider.current_point_diagnostics()
                record.update(
                    maximum_equivalent_plastic_strain=float(result.quantity(
                        "maximum_equivalent_plastic_strain",
                    )),
                    maximum_displacement=float(np.max(np.linalg.norm(
                        displacement.value.x.array.reshape((-1, 3)), axis=1,
                    ))),
                    tool_work=float(energy.tool_work),
                    contact_potential=float(energy.contact_potential),
                    maximum_sampled_penetration=float(contact[
                        "maximum_penetration"
                    ]),
                    active_reference_area=float(contact["active_reference_area"]),
                    plastic_dissipation=float(energy.plastic_dissipation),
                    mechanical_energy_residual=float(energy.mechanical_energy_residual),
                    contact_integration_certified=bool(
                        contact["feature_transition_integral_certified"]
                    ),
                )
        except (RuntimeError, ValueError, NotImplementedError) as error:
            record["status"] = "failed"
            record["failure_type"] = type(error).__name__
            record["failure_reason"] = str(error)
        record["wall_seconds"] = perf_counter() - start
        records.append(record)
        print(f"{label}: {record['status']} ({record['wall_seconds']:.2f} s)")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/closed_planar_branch_ladder.json"),
    )
    args = parser.parse_args()
    records = run_ladder()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": "agentfem.closed-planar-branch-sensitivity.v1",
        "scope": "serial synthetic fixture; one-axis-at-a-time; not external validation",
        "cases": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Evidence: {args.output.resolve()}")
    if any(item["status"] != "completed" or not item.get(
        "contact_integration_certified", False,
    ) for item in records):
        raise SystemExit("At least one contact sensitivity case is not certified and completed.")


if __name__ == "__main__":
    main()

"""Run the scheduled SIMULIA 316-steel ratcheting convergence certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from agentfem import __version__, benchmarks, platforms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--mesh-sizes", type=float, nargs="+", default=(3.5, 2.5))
    parser.add_argument("--refinements", type=int, nargs="+")
    parser.add_argument(
        "--maximum-inelastic-increments",
        type=float,
        nargs="+",
        default=(4.0e-3, 2.0e-3, 1.0e-3),
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--require-acceptable", action="store_true")
    options = parser.parse_args()

    started = perf_counter()
    if options.refinements is None:
        path_control = "maximum_inelastic_increment"
        certificate, cases = (
            benchmarks.certify_simulia_316_shouldered_ratcheting_accuracy(
                cycle_count=options.cycles,
                mesh_sizes=tuple(options.mesh_sizes),
                maximum_inelastic_increments=tuple(
                    options.maximum_inelastic_increments
                ),
                relative_tolerance=options.relative_tolerance,
                progress=options.progress,
            )
        )
    else:
        path_control = "fixed_refinement"
        certificate, cases = (
            benchmarks.certify_simulia_316_shouldered_ratcheting_convergence(
                cycle_count=options.cycles,
                mesh_sizes=tuple(options.mesh_sizes),
                refinements=tuple(options.refinements),
                relative_tolerance=options.relative_tolerance,
                progress=options.progress,
            )
        )
    case_records = []
    for (mesh_size, control_value), (assessment, result) in sorted(cases.items()):
        case_records.append(
            {
                "mesh_size": mesh_size,
                "path_control": path_control,
                "path_control_value": control_value,
                "assessment": assessment.as_dict(),
                "path_control_evidence": result.metadata["external_benchmark"][
                    "path_control"
                ],
                "cycle": result.histories[
                    "maximum_center_axial_strain"
                ].abscissa.tolist(),
                "maximum_center_axial_strain": result.histories[
                    "maximum_center_axial_strain"
                ].values.tolist(),
            }
        )
    record = {
        "schema": "agentfem.external-structural-benchmark",
        "schema_version": "0.1.0",
        "benchmark": "SIMULIA 316-steel shouldered ratcheting specimen",
        "status": "passed" if certificate.accepted else "not_promoted",
        "agentfem_version": __version__,
        "configuration": {
            "cycles": options.cycles,
            "mesh_sizes": options.mesh_sizes,
            "refinements": options.refinements,
            "maximum_inelastic_increments": (options.maximum_inelastic_increments),
            "path_control": path_control,
            "relative_tolerance": options.relative_tolerance,
        },
        "certificate": certificate.as_dict(),
        "cases": case_records,
        "runtime_seconds": perf_counter() - started,
        "runtime": platforms.runtime_report().summary(),
    }
    options.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = options.report.with_suffix(options.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(options.report)
    print(json.dumps(record, indent=2, sort_keys=True))
    if options.require_acceptable and not certificate.accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

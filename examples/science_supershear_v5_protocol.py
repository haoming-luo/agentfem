"""Audit and inspect the public Science 2023 supershear dataset.

Download the files manually from the pinned Dryad landing page, then run:

    python examples/science_supershear_v5_protocol.py path/to/dryad/files

This example does not fit parameters or declare validation.  It creates the
first reproducible evidence boundary for an independent V5 research study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentfem import datasets


def inspect_public_data(directory: str | Path) -> dict[str, object]:
    manifest = datasets.science_supershear_dryad_manifest()
    audit = manifest.audit(directory).require()
    root = Path(directory)
    workbooks = {}
    for role in (
        "crack_speed",
        "mach_cone",
        "wave_speed",
        "material_response",
        "sed_ked_field",
    ):
        workbooks[role] = [
            datasets.read_xlsx_workbook(root / item.path).summary()
            for item in manifest.files_for_roles((role,))
        ]
    return {
        "schema": "agentfem.science-supershear-v5-data-inspection.v1",
        "manifest": manifest.summary(),
        "audit": audit.summary(),
        "workbooks": workbooks,
        "interpretation": (
            "The public files are authentic and structurally readable; "
            "calibration, unit interpretation, and prediction tests remain a "
            "separate research task."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("science_supershear_v5_data_report.json"),
    )
    args = parser.parse_args(argv)
    report = inspect_public_data(args.data_directory)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["audit"], indent=2))
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

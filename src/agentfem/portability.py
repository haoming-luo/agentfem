"""Cross-runtime numerical equivalence contracts for AgentFEM results."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


COMPARISON_SCHEMA = "agentfem.runtime-equivalence"
COMPARISON_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class RuntimeComparison:
    """Comparison of declared scientific quantities across execution routes."""

    manifests: tuple[Path, ...]
    quantities: tuple[dict[str, object], ...]
    issues: tuple[str, ...]
    relative_tolerance: float
    absolute_tolerance: float
    tolerance_overrides: Mapping[str, tuple[float, float]]
    integrity: tuple[dict[str, object], ...]

    @property
    def accepted(self) -> bool:
        return not self.issues and all(
            item.get("status") == "accepted" for item in self.quantities
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": COMPARISON_SCHEMA,
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "status": "accepted" if self.accepted else "rejected",
            "accepted": self.accepted,
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "tolerance_overrides": {
                name: {"relative": values[0], "absolute": values[1]}
                for name, values in self.tolerance_overrides.items()
            },
            "manifests": tuple(str(path) for path in self.manifests),
            "integrity": self.integrity,
            "quantities": self.quantities,
            "issues": self.issues,
        }

    def format(self) -> str:
        lines = [
            f"Runtime equivalence: {'accepted' if self.accepted else 'rejected'}"
        ]
        for item in self.quantities:
            lines.append(
                f"  {item['name']}: {item['status']} "
                f"(max normalized error={item['maximum_normalized_error']:.6g})"
            )
        lines.extend(f"  issue: {item}" for item in self.issues)
        return "\n".join(lines)


def compare_results(
    manifests: Iterable[str | Path],
    *,
    quantities: Iterable[str] | None = None,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-10,
    tolerances: Mapping[str, tuple[float, float]] | None = None,
) -> RuntimeComparison:
    """Compare common numerical result quantities without demanding bit identity."""

    paths = tuple(Path(item).expanduser().resolve() for item in manifests)
    if len(paths) < 2:
        raise ValueError("Runtime equivalence requires at least two result manifests.")
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("Runtime equivalence tolerances must be non-negative.")
    selected_tolerances = dict(tolerances or {})
    for name, values in selected_tolerances.items():
        if len(values) != 2 or values[0] < 0.0 or values[1] < 0.0:
            raise ValueError(
                f"Runtime equivalence tolerance for {name!r} must be "
                "(non-negative rtol, non-negative atol)."
            )
    records = tuple(_load_manifest(path) for path in paths)
    issues: list[str] = []
    integrity: list[dict[str, object]] = []
    from .provenance import verify_manifest

    for path in paths:
        report = verify_manifest(path).summary()
        integrity.append(report)
        if not report.get("verified"):
            issues.append(f"{path} failed provenance integrity verification.")
    for path, record in zip(paths, records, strict=True):
        if record.get("status") != "completed":
            issues.append(f"{path} is not a completed simulation result.")
    inventories = tuple(_numerical_quantities(record) for record in records)
    requested = None if quantities is None else tuple(dict.fromkeys(quantities))
    if requested is None:
        names = tuple(sorted(set.intersection(*(set(item) for item in inventories))))
    else:
        names = requested
    if not names:
        issues.append("No common numerical quantities are available for comparison.")

    comparisons: list[dict[str, object]] = []
    for name in names:
        missing = tuple(
            str(path) for path, inventory in zip(paths, inventories, strict=True)
            if name not in inventory
        )
        if missing:
            issues.append(f"Quantity {name!r} is missing from: {', '.join(missing)}.")
            continue
        selected = tuple(inventory[name] for inventory in inventories)
        units_match = all(item["unit"] == selected[0]["unit"] for item in selected)
        shapes = {tuple(item["shape"]) for item in selected}
        if not units_match or len(shapes) != 1:
            issues.append(f"Quantity {name!r} has incompatible unit or shape metadata.")
            continue
        reference = np.asarray(selected[0]["value"], dtype=float)
        quantity_rtol, quantity_atol = selected_tolerances.get(
            name, (relative_tolerance, absolute_tolerance)
        )
        errors: list[float] = []
        absolute_errors: list[float] = []
        accepted = True
        for item in selected[1:]:
            candidate = np.asarray(item["value"], dtype=float)
            difference = np.abs(candidate - reference)
            scale = quantity_atol + quantity_rtol * np.abs(reference)
            normalized = np.divide(
                difference,
                scale,
                out=np.where(difference == 0.0, 0.0, np.inf),
                where=scale > 0.0,
            )
            if difference.size == 0:
                maximum_absolute = 0.0
                maximum_normalized = 0.0
            else:
                maximum_absolute = float(np.max(difference))
                maximum_normalized = float(np.max(normalized))
            absolute_errors.append(maximum_absolute)
            errors.append(maximum_normalized)
            accepted = accepted and bool(
                np.allclose(
                    candidate,
                    reference,
                    rtol=quantity_rtol,
                    atol=quantity_atol,
                    equal_nan=False,
                )
            )
        comparisons.append(
            {
                "name": name,
                "unit": selected[0]["unit"],
                "shape": selected[0]["shape"],
                "kind": selected[0]["kind"],
                "status": "accepted" if accepted else "rejected",
                "relative_tolerance": quantity_rtol,
                "absolute_tolerance": quantity_atol,
                "maximum_absolute_error": max(absolute_errors, default=0.0),
                "maximum_normalized_error": max(errors, default=0.0),
                "values": tuple(item["value"] for item in selected),
            }
        )
    return RuntimeComparison(
        manifests=paths,
        quantities=tuple(comparisons),
        issues=tuple(issues),
        relative_tolerance=float(relative_tolerance),
        absolute_tolerance=float(absolute_tolerance),
        tolerance_overrides=selected_tolerances,
        integrity=tuple(integrity),
    )


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"AgentFEM result manifest not found: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != "agentfem.simulation-result":
        raise ValueError(f"Not an AgentFEM simulation result manifest: {path}")
    return record


def _numerical_quantities(record: dict[str, object]) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for item in record.get("quantity_records", ()):
        if not isinstance(item, dict) or not item.get("name"):
            continue
        try:
            value = np.asarray(item.get("value"), dtype=float)
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(value)):
            continue
        inventory[str(item["name"])] = {
            "name": str(item["name"]),
            "kind": str(item.get("kind", "diagnostic")),
            "unit": item.get("unit"),
            "shape": tuple(item.get("shape", value.shape)),
            "value": item.get("value"),
        }
    return inventory


__all__ = [
    "COMPARISON_SCHEMA",
    "COMPARISON_SCHEMA_VERSION",
    "RuntimeComparison",
    "compare_results",
]

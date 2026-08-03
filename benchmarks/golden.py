"""Versioned numerical acceptance contracts for release benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class GoldenQuantity:
    """One expected physical observable with explicit numerical tolerances."""

    name: str
    expected: object
    relative_tolerance: float
    absolute_tolerance: float
    unit: str | None = None
    description: str = ""

    def accepts(self, actual) -> bool:
        return bool(
            np.allclose(
                np.asarray(actual, dtype=float),
                np.asarray(self.expected, dtype=float),
                rtol=self.relative_tolerance,
                atol=self.absolute_tolerance,
            )
        )

    def assert_accepts(self, actual) -> None:
        np.testing.assert_allclose(
            actual,
            self.expected,
            rtol=self.relative_tolerance,
            atol=self.absolute_tolerance,
            err_msg=f"Golden quantity {self.name!r} is outside its contract.",
        )


@dataclass(frozen=True)
class GoldenBenchmark:
    """A named collection of numerical observables from a benchmark card."""

    identifier: str
    reference_version: str
    quantities: tuple[GoldenQuantity, ...]

    def quantity(self, name: str) -> GoldenQuantity:
        for item in self.quantities:
            if item.name == name:
                return item
        raise KeyError(
            f"Unknown golden quantity {name!r}; "
            f"available={tuple(item.name for item in self.quantities)}."
        )

    def verify(self, actual: Mapping[str, object]) -> dict[str, bool]:
        return {
            item.name: item.accepts(actual[item.name])
            for item in self.quantities
        }


def golden_benchmark(identifier: str) -> GoldenBenchmark:
    """Load a numerical contract by stable benchmark-card identifier."""

    directory = Path(__file__).resolve().parents[1] / "knowledge" / "benchmarks"
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("id") != identifier:
            continue
        golden = record.get("golden")
        if not isinstance(golden, dict):
            raise ValueError(f"Benchmark {identifier!r} has no golden contract.")
        return GoldenBenchmark(
            identifier=identifier,
            reference_version=str(golden["reference_version"]),
            quantities=tuple(
                GoldenQuantity(
                    name=str(item["name"]),
                    expected=item["expected"],
                    relative_tolerance=float(item["relative_tolerance"]),
                    absolute_tolerance=float(item["absolute_tolerance"]),
                    unit=item.get("unit"),
                    description=str(item.get("description", "")),
                )
                for item in golden["quantities"]
            ),
        )
    raise KeyError(f"Unknown benchmark card {identifier!r}.")


"""Reusable verification obligations and benchmark metadata."""

from .registry import BenchmarkSpec, benchmark, list_benchmarks
from .golden import GoldenBenchmark, GoldenQuantity, golden_benchmark
from .dynamic_fracture import (
    CohesiveEnergyBenchmark,
    ClassicalCrackBenchmark,
    WaveArrivalBenchmark,
    cohesive_energy_balance,
    classical_cohesive_crack,
    finite_strain_wave_arrival,
)

__all__ = [
    "BenchmarkSpec",
    "GoldenBenchmark",
    "GoldenQuantity",
    "benchmark",
    "golden_benchmark",
    "WaveArrivalBenchmark",
    "CohesiveEnergyBenchmark",
    "ClassicalCrackBenchmark",
    "classical_cohesive_crack",
    "cohesive_energy_balance",
    "finite_strain_wave_arrival",
    "list_benchmarks",
]

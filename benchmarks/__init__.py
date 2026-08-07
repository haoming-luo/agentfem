"""Reusable verification obligations and benchmark metadata."""

from .registry import BenchmarkSpec, benchmark, list_benchmarks
from .golden import GoldenBenchmark, GoldenQuantity, golden_benchmark
from .dynamic_fracture import (
    CohesiveEnergyBenchmark,
    ClassicalCrackBenchmark,
    WaveArrivalBenchmark,
    WeakInterfaceTransitionBenchmark,
    WeakInterfaceTransitionSuite,
    cohesive_energy_balance,
    classical_cohesive_crack,
    finite_strain_wave_arrival,
    jmps_weak_interface_transition_v4,
    prestressed_weak_interface_separation,
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
    "WeakInterfaceTransitionBenchmark",
    "WeakInterfaceTransitionSuite",
    "classical_cohesive_crack",
    "cohesive_energy_balance",
    "finite_strain_wave_arrival",
    "jmps_weak_interface_transition_v4",
    "prestressed_weak_interface_separation",
    "list_benchmarks",
]

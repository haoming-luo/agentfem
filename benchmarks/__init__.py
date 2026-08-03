"""Reusable verification obligations and benchmark metadata."""

from .registry import BenchmarkSpec, benchmark, list_benchmarks
from .golden import GoldenBenchmark, GoldenQuantity, golden_benchmark

__all__ = [
    "BenchmarkSpec",
    "GoldenBenchmark",
    "GoldenQuantity",
    "benchmark",
    "golden_benchmark",
    "list_benchmarks",
]

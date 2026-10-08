# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


def test_public_coupled_decay_has_separate_space_and_time_convergence():
    path = Path(__file__).resolve().parents[1] / "tools" / "benchmark_coupled_decay.py"
    spec = importlib.util.spec_from_file_location("coupled_decay_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    report = benchmark.study()
    assert report["accepted"], report["observed_orders"]
    for axis in ("spatial", "temporal"):
        for record in report[axis]:
            assert record["matrix_assemblies"] == {"thermal": 1, "mechanical": 1}
            assert record["max_quadratic_residual"] < 1e-9

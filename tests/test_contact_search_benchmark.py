# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


def test_search_measurement_checks_projection_and_reproducible_inputs():
    path = Path(__file__).resolve().parents[1] / "tools/benchmark_contact_search.py"
    spec = importlib.util.spec_from_file_location("search_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = module.measure(cells=4, queries=3, repeats=1)
    second = module.measure(cells=4, queries=3, repeats=1)
    assert first["projection_equivalence"] == "passed"
    assert first["query_sha256"] == second["query_sha256"]
    assert first["geometry_fingerprint"] == second["geometry_fingerprint"]
    assert first["exact_facet_evaluations"] == second["exact_facet_evaluations"]
    assert 0 < first["exact_work_fraction"] <= 1
    assert first["tree_build_seconds"] >= 0
    # Timing is evidence, never a hardware-dependent CI pass threshold.
    assert len(first["query_seconds"]["bvh"]) == 1
